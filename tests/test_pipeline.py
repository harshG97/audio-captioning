"""End-to-end checks for the RECAP training and generation path on tiny models.

Builds a tiny GPT-2 (with a small byte-level BPE tokenizer) and a tiny CLAP
checkpoint locally, so nothing is downloaded. Covers:
  - RecapDataset: prompt masking, the decoder_input_ids/labels shift, EOS,
    prompt dropout, and RecapCollator padding;
  - RECAP.forward on a collated batch;
  - one training step: only cross-attention gets gradients and the loss drops;
  - save/load through load_recap_checkpoint;
  - evaluate.generate_caption, checked against a cache-free argmax loop.

Run: python -m pytest tests/test_pipeline.py -q  (CPU, a few seconds)
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (
    ClapAudioConfig,
    ClapConfig,
    ClapModel,
    ClapTextConfig,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.recap_dataset import RecapCollator, RecapDataset
from evaluate import generate_caption
from model.build_recap import XATTN_PARAM_KEYS, build_recap, load_recap_checkpoint
from retrieval.prompt_builder import BASELINE_PROMPT, build_prompt

HIDDEN = 64          # tiny GPT-2 width == tiny CLAP audio width
ENC_LEN = 8          # length of the fake cached encoder outputs

ROWS = [
    ("abc", 0, "a dog barks loudly"),
    ("def", 0, "rain falling on a roof while thunder rumbles in the distance"),
]
RETRIEVED = {
    "abc_0": ["a dog is barking", "a small dog yelps"],
    "def_0": ["water dripping", "a storm with heavy rain"],
}


@pytest.fixture(scope="module")
def model_dirs(tmp_path_factory):
    """Save a tiny GPT-2 + tokenizer and a tiny CLAP in the hub formats."""
    torch.manual_seed(0)
    corpus = [c for *_, c in ROWS] + [c for v in RETRIEVED.values() for c in v]
    corpus += [build_prompt(v) for v in RETRIEVED.values()]
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(corpus * 20, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = GPT2TokenizerFast(tokenizer_object=bpe, bos_token="<|endoftext|>",
                            eos_token="<|endoftext|>", unk_token="<|endoftext|>")

    gpt2_dir = tmp_path_factory.mktemp("tiny_gpt2")
    # bos/eos default to 50256 in GPT2Config; point them at the tiny vocab's
    # <|endoftext|>, which is what the real GPT-2 uses.
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=256, n_embd=HIDDEN,
                               n_layer=2, n_head=4, bos_token_id=tok.eos_token_id,
                               eos_token_id=tok.eos_token_id)).save_pretrained(gpt2_dir)
    tok.save_pretrained(gpt2_dir)

    clap_dir = tmp_path_factory.mktemp("tiny_clap")
    audio = ClapAudioConfig(depths=[1, 1, 1, 1], num_attention_heads=[1, 1, 1, 1],
                            patch_embeds_hidden_size=8, hidden_size=HIDDEN,
                            enable_fusion=True, projection_hidden_size=32)
    text = ClapTextConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=64,
                          max_position_embeddings=40, pad_token_id=1)
    ClapModel(ClapConfig(text_config=text.to_dict(), audio_config=audio.to_dict(),
                         projection_dim=16)).save_pretrained(clap_dir)
    return str(clap_dir), str(gpt2_dir)


@pytest.fixture
def recap(model_dirs):
    """Fresh RECAP per test (training mutates it), cross-attention-only training."""
    clap_dir, gpt2_dir = model_dirs
    torch.manual_seed(0)
    return build_recap(encoder_name=clap_dir, decoder_name=gpt2_dir,
                       cross_attention_reduce_factor=4)


@pytest.fixture
def data_files(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame([{"youtube_id": y, "start_time": s, "caption": c} for y, s, c in ROWS]
                 ).to_csv(csv, index=False)
    h5 = tmp_path / "feats.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(h5, "w") as f:
        for key in RETRIEVED:
            f.create_dataset(key, data=rng.standard_normal((ENC_LEN, HIDDEN)).astype(np.float32))
    cache = tmp_path / "retrieved.json"
    cache.write_text(json.dumps(RETRIEVED))
    return str(csv), str(h5), str(cache)


def _dataset(data_files, tokenizer, **kw):
    csv, h5, cache = data_files
    return RecapDataset(csv_path=csv, hdf5_path=h5, retrieval_cache_path=cache,
                        tokenizer=tokenizer, dataset="audiocaps", max_length=128, **kw)


def _batch(data_files, tokenizer):
    ds = _dataset(data_files, tokenizer)
    return RecapCollator(pad_token_id=tokenizer.pad_token_id)([ds[0], ds[1]])


def _forward(model, batch):
    return model(encoder_outputs=batch["encoder_outputs"],
                 decoder_input_ids=batch["decoder_input_ids"],
                 decoder_attention_mask=batch["decoder_attention_mask"],
                 labels=batch["labels"], return_dict=True)


def test_special_tokens_are_wired(recap):
    model, tok = recap
    assert tok.pad_token == "!" and tok.eos_token == "."
    assert model.config.pad_token_id == model.decoder.config.pad_token_id == tok.pad_token_id
    assert model.config.eos_token_id == model.decoder.config.eos_token_id == tok.eos_token_id


def test_dataset_item_layout(recap, data_files):
    _, tok = recap
    ds = _dataset(data_files, tok)
    item = ds[0]
    dec_in, labels = item["decoder_input_ids"].tolist(), item["labels"].tolist()
    prompt_ids = tok(build_prompt(RETRIEVED["abc_0"]), add_special_tokens=False)["input_ids"]

    assert len(dec_in) == len(labels)
    assert dec_in[0] == ds.dec_start
    # The retrieval prompt is input only; the caption plus EOS is the target.
    assert labels[:len(prompt_ids)] == [-100] * len(prompt_ids)
    target = labels[len(prompt_ids):]
    assert -100 not in target and target[-1] == tok.eos_token_id
    assert tok.decode(target[:-1]).strip() == ROWS[0][2]
    # Teacher forcing: the decoder input is [start] + (prompt + target) minus
    # its last token, so position t predicts the token fed at t + 1.
    assert dec_in[1:] == (prompt_ids + target)[:-1]
    assert tuple(item["encoder_hidden_states"].shape) == (ENC_LEN, HIDDEN)


def test_prompt_dropout_uses_baseline_prompt(recap, data_files):
    _, tok = recap
    item = _dataset(data_files, tok, prompt_dropout=1.0)[0]
    baseline = tok(BASELINE_PROMPT, add_special_tokens=False)["input_ids"]
    assert item["decoder_input_ids"].tolist()[1:len(baseline) + 1] == baseline
    assert item["labels"].tolist()[:len(baseline)] == [-100] * len(baseline)


def test_collator_pads_right(recap, data_files):
    _, tok = recap
    ds = _dataset(data_files, tok)
    items = [ds[0], ds[1]]
    batch = RecapCollator(pad_token_id=tok.pad_token_id)(items)
    lengths = [len(i["decoder_input_ids"]) for i in items]
    assert lengths[0] != lengths[1], "rows should differ in length to exercise padding"
    assert tuple(batch["encoder_outputs"].shape) == (2, ENC_LEN, HIDDEN)
    for row, n in enumerate(lengths):
        assert batch["decoder_attention_mask"][row].tolist() == [1] * n + [0] * (max(lengths) - n)
        assert (batch["labels"][row, n:] == -100).all()
        assert (batch["decoder_input_ids"][row, n:] == tok.pad_token_id).all()


def test_forward_on_collated_batch(recap, data_files):
    model, tok = recap
    batch = _batch(data_files, tok)
    model.eval()
    with torch.no_grad():
        out = _forward(model, batch)
    assert torch.isfinite(out.loss)
    assert out.logits.shape == (*batch["decoder_input_ids"].shape, model.decoder.config.vocab_size)


def test_training_step_updates_only_cross_attention(recap, data_files):
    model, tok = recap
    batch = _batch(data_files, tok)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    model.eval()
    with torch.no_grad():
        loss_before = _forward(model, batch).loss.item()

    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        loss = _forward(model, batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()

    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed, "no parameter was updated"
    assert all(any(k in n for k in XATTN_PARAM_KEYS) for n in changed), sorted(changed)[:5]

    model.eval()
    with torch.no_grad():
        assert _forward(model, batch).loss.item() < loss_before


def test_save_load_roundtrip(recap, data_files, tmp_path):
    model, tok = recap
    batch = _batch(data_files, tok)
    model.eval()
    with torch.no_grad():
        ref = _forward(model, batch).logits

    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    loaded, loaded_tok = load_recap_checkpoint(str(tmp_path))
    loaded.eval()
    assert (loaded_tok.pad_token_id, loaded_tok.eos_token_id) == (tok.pad_token_id, tok.eos_token_id)
    assert loaded.decoder.config.cross_attention_reduce_factor == 4
    with torch.no_grad():
        assert torch.allclose(_forward(loaded, batch).logits, ref, atol=1e-5)


def _generate(model, tok, data_files, num_beams, max_new_tokens=8):
    ds = _dataset(data_files, tok)
    enc = ds[0]["encoder_hidden_states"].unsqueeze(0)
    prompt_ids = tok(build_prompt(RETRIEVED["abc_0"]), add_special_tokens=False)["input_ids"]
    out = generate_caption(
        model=model, tokenizer=tok, encoder_hidden=enc, decoder_start_id=ds.dec_start,
        prompt_ids=prompt_ids, eos_id=tok.eos_token_id, max_new_tokens=max_new_tokens,
        num_beams=num_beams, device=torch.device("cpu"), length_penalty=0.0,
        return_stop_reason=True,
    )
    return out, enc, [ds.dec_start, *prompt_ids]


def test_greedy_generation_matches_manual_decoding(recap, data_files):
    """generate_caption (HF generate, KV cache) must equal a plain argmax loop
    that re-runs the full sequence without a cache at every step."""
    model, tok = recap
    model.eval()
    (text, reason), enc, ids = _generate(model, tok, data_files, num_beams=1)

    gen, manual_reason = [], "max_new_tokens"
    with torch.no_grad():
        for _ in range(8):
            logits = model(encoder_outputs=enc, decoder_input_ids=torch.tensor([ids + gen]),
                           use_cache=False, return_dict=True).logits
            nxt = int(logits[0, -1].argmax())
            if nxt == tok.eos_token_id:
                manual_reason = "eos"
                break
            gen.append(nxt)
    assert reason == manual_reason
    assert text == tok.decode(gen, skip_special_tokens=True).strip()


def test_beam_generation_runs(recap, data_files):
    model, tok = recap
    model.eval()
    (text, reason), _, _ = _generate(model, tok, data_files, num_beams=3)
    assert isinstance(text, str)
    assert reason in {"eos", "max_new_tokens"}
