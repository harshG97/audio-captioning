"""Tests for the prefix model (model/prefix_lm.py): a decoder-only LLM that
reads the audio features as prefix tokens and is trained with LoRA.

Uses a tiny random Qwen2 model with a Qwen-style tokenizer (no BOS token;
<|endoftext|> as end and padding token), saved locally, so nothing is
downloaded.

Run: python -m pytest tests/test_prefix_lm.py -q  (CPU, a few seconds)
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2TokenizerFast, Qwen2Config, Qwen2ForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.recap_dataset import RecapCollator, RecapDataset
from evaluate import caption_end_token_ids, generate_caption
from model.prefix_lm import AudioPrefixLM, build_prefix_lm, load_captioner
from retrieval.prompt_builder import BASELINE_PROMPT, build_prompt

FEAT = 48            # feature width (differs from the LLM's 64)
ENC_LEN = 8          # feature vectors per clip -> 2 audio tokens at pool 4
ROWS = [("abc", 0, "a dog barks loudly."), ("def", 0, "rain falls on a roof while thunder rumbles.")]
RETRIEVED = {"abc_0": ["a dog is barking.", "a small dog yelps."],
             "def_0": ["water dripping.", "a storm with heavy rain."]}


@pytest.fixture(scope="module")
def llm_dir(tmp_path_factory):
    corpus = [c for *_, c in ROWS] + [c for v in RETRIEVED.values() for c in v]
    corpus += [build_prompt(v) for v in RETRIEVED.values()] + [BASELINE_PROMPT]
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(corpus * 20, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = GPT2TokenizerFast(tokenizer_object=bpe, bos_token=None, eos_token="<|endoftext|>",
                            pad_token="<|endoftext|>", unk_token="<|endoftext|>")
    d = tmp_path_factory.mktemp("tiny_qwen2")
    torch.manual_seed(0)
    Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=True, bos_token_id=None, eos_token_id=tok.eos_token_id)).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture
def prefix(llm_dir):
    torch.manual_seed(0)
    return build_prefix_lm(llm_dir, feature_width=FEAT)


@pytest.fixture
def data_files(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame([{"youtube_id": y, "start_time": s, "caption": c} for y, s, c in ROWS]).to_csv(csv, index=False)
    h5 = tmp_path / "feats.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(h5, "w") as f:
        for key in RETRIEVED:
            f.create_dataset(key, data=rng.standard_normal((ENC_LEN, FEAT)).astype(np.float32))
    cache = tmp_path / "retrieved.json"
    cache.write_text(json.dumps(RETRIEVED))
    return str(csv), str(h5), str(cache)


def _batch(model, tok, data_files, **kw):
    csv, h5, cache = data_files
    ds = RecapDataset(csv_path=csv, hdf5_path=h5, retrieval_cache_path=cache, tokenizer=tok,
                      max_length=128, decoder_start_token_id=model.config.decoder_start_token_id, **kw)
    return RecapCollator(pad_token_id=tok.pad_token_id)([ds[0], ds[1]]), ds


def _forward(model, batch, feats=None):
    return model(encoder_outputs=(batch["encoder_outputs"] if feats is None else feats,),
                 decoder_input_ids=batch["decoder_input_ids"],
                 decoder_attention_mask=batch["decoder_attention_mask"], labels=batch["labels"])


def test_only_projector_and_lora_train(prefix):
    model, tok = prefix
    trained = model._trained_names()
    assert trained and all(n.startswith("projector.") or ".lora_" in n for n in trained)
    assert any(".lora_" in n for n in trained) and any(n.startswith("projector.") for n in trained)
    for target in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert any(f"{target}.lora_A" in n for n in trained), target
    assert tok.bos_token_id is None and tok.eos_token == tok.pad_token == "<|endoftext|>"
    assert model.config.decoder_start_token_id == model.config.eos_token_id == tok.eos_token_id


def test_audio_becomes_pooled_prefix_tokens(prefix):
    model, _ = prefix
    feats = torch.randn(2, ENC_LEN, FEAT, generator=torch.Generator().manual_seed(0))
    ids = torch.tensor([[0, 5, 6], [0, 7, 8]])
    embeds, mask, n_audio = model.build_inputs(feats, ids)
    assert n_audio == ENC_LEN // 4 and embeds.shape == (2, n_audio + 3, 64)
    assert mask.tolist() == [[1] * (n_audio + 3)] * 2
    expected = model.projector.mlp(model.projector.norm(feats.view(2, 2, 4, FEAT).mean(2)))
    assert torch.allclose(embeds[:, :n_audio], expected)
    assert torch.equal(embeds[:, n_audio:], model.llm.get_input_embeddings()(ids))
    with pytest.raises(ValueError, match="groups of 4"):
        model.build_inputs(torch.zeros(1, 7, FEAT), ids[:1])


def test_loss_covers_only_caption_and_end_token(prefix, data_files):
    model, tok = prefix
    model.eval()
    batch, ds = _batch(model, tok, data_files)
    out = _forward(model, batch)
    assert out.logits.shape[:2] == batch["decoder_input_ids"].shape     # one per text position
    labels = batch["labels"]
    manual = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    assert torch.allclose(out.loss, manual)
    prompt_len = len(tok(build_prompt(RETRIEVED["abc_0"]), add_special_tokens=False)["input_ids"])
    target = labels[0][labels[0] != -100].tolist()
    assert (labels[0, :prompt_len] == -100).all() and target[-1] == tok.eos_token_id
    assert tok.decode(target[:-1]).strip() == ROWS[0][2]
    assert batch["decoder_input_ids"][0, 0].item() == tok.eos_token_id       # start token after the audio


def test_audio_matters_and_padding_does_not(prefix, data_files):
    model, tok = prefix
    model.eval()
    batch, ds = _batch(model, tok, data_files)
    with torch.no_grad():
        intact = _forward(model, batch).loss
        zeroed = _forward(model, batch, torch.zeros_like(batch["encoder_outputs"])).loss
        alone = _forward(model, RecapCollator(tok.pad_token_id)([ds[0]])).logits
        padded = _forward(model, batch).logits[:1, :alone.shape[1]]
    assert not torch.allclose(intact, zeroed)
    assert torch.allclose(alone, padded, atol=1e-5)      # row 0 is unaffected by row 1's length


def test_text_positions_only_see_the_past(prefix, data_files):
    model, tok = prefix
    model.eval()
    batch, _ = _batch(model, tok, data_files)
    ids = batch["decoder_input_ids"][:1].clone()
    feats = batch["encoder_outputs"][:1]
    with torch.no_grad():
        before = model(encoder_outputs=(feats,), decoder_input_ids=ids).logits
        ids[0, 5] = (ids[0, 5] + 1) % len(tok)
        after = model(encoder_outputs=(feats,), decoder_input_ids=ids).logits
    assert torch.allclose(before[0, :5], after[0, :5]) and not torch.allclose(before[0, 5], after[0, 5])


def test_training_moves_only_trained_weights_and_lowers_loss(prefix, data_files):
    model, tok = prefix
    batch, _ = _batch(model, tok, data_files)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    trained = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    losses = []
    for _ in range(15):
        loss = _forward(model, batch).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5
    params = dict(model.named_parameters())
    assert all(torch.equal(params[n], v) for n, v in frozen.items())
    assert any(not torch.equal(params[n], v) for n, v in trained.items() if ".lora_B" in n)
    assert any(not torch.equal(params[n], v) for n, v in trained.items() if n.startswith("projector."))


def test_save_and_load_keep_only_trained_weights(prefix, data_files, tmp_path):
    model, tok = prefix
    batch, _ = _batch(model, tok, data_files)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn_like(p))       # LoRA B starts at zero
    model.eval()
    model.save_pretrained(tmp_path / "ck")
    tok.save_pretrained(tmp_path / "ck")
    from safetensors.torch import load_file
    saved = load_file(str(tmp_path / "ck" / "model.safetensors"))
    assert set(saved) == set(model._trained_names())
    loaded = load_captioner(str(tmp_path / "ck"))
    assert isinstance(loaded, AudioPrefixLM) and not loaded.training
    with torch.no_grad():
        assert torch.equal(_forward(model, batch).logits, _forward(loaded, batch).logits)
    saved.pop(next(iter(saved)))
    with pytest.raises(RuntimeError, match="does not match"):
        loaded.load_state_dict(saved)


def _greedy_by_hand(model, feats, ids, n):
    out = []
    for _ in range(n):
        with torch.no_grad():
            nxt = model(encoder_outputs=(feats,), decoder_input_ids=ids).logits[0, -1].argmax().item()
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return out


@pytest.mark.parametrize("num_beams", [1, 3])
def test_generate_caption_matches_the_training_forward(prefix, data_files, num_beams):
    model, tok = prefix
    model.eval()
    csv, h5, cache = data_files
    with h5py.File(h5, "r") as f:
        feats = torch.from_numpy(f["abc_0"][()]).unsqueeze(0)
    prompt_ids = tok(build_prompt(RETRIEVED["abc_0"]), add_special_tokens=False)["input_ids"]
    seen = []
    original = model.llm.generate
    model.llm.generate = lambda *a, **k: seen.append(o := original(*a, **k)) or o
    try:
        text, reason = generate_caption(
            model=model, tokenizer=tok, encoder_hidden=feats, decoder_start_id=tok.eos_token_id,
            prompt_ids=prompt_ids, eos_id=tok.eos_token_id, max_new_tokens=6, num_beams=num_beams,
            device=torch.device("cpu"), length_penalty=0.0, return_stop_reason=True)
    finally:
        model.llm.generate = original
    new = seen[0][0].tolist()
    assert len(new) <= 6                                  # new tokens only, no prompt echo
    body = new[:-1] if new and new[-1] == tok.eos_token_id else new
    assert text == tok.decode(body, skip_special_tokens=True).strip()
    if num_beams == 1:
        start = torch.tensor([[tok.eos_token_id, *prompt_ids]])
        assert new == _greedy_by_hand(model, feats, start, len(new))
    assert reason in ("eos", "max_new_tokens")


@pytest.mark.parametrize("num_beams", [1, 3])
def test_min_new_tokens_works_for_the_prefix_model(prefix, data_files, num_beams):
    model, tok = prefix
    model.eval()
    csv, h5, cache = data_files
    with h5py.File(h5, "r") as f:
        feats = torch.from_numpy(f["abc_0"][()]).unsqueeze(0)
    eos = tok.eos_token_id
    head = model.llm.get_output_embeddings()
    hook = head.register_forward_hook(lambda m, i, o: o + 100 * F.one_hot(torch.tensor(eos), o.shape[-1]))
    seen = []
    original = model.llm.generate
    model.llm.generate = lambda *a, **k: seen.append(o := original(*a, **k)) or o
    try:
        for n in (0, 3):
            generate_caption(model=model, tokenizer=tok, encoder_hidden=feats, decoder_start_id=eos,
                             prompt_ids=[5, 6], eos_id=eos, max_new_tokens=8, num_beams=num_beams,
                             device=torch.device("cpu"), length_penalty=0.0, min_new_tokens=n)
    finally:
        hook.remove()
        model.llm.generate = original
    free, held = seen[0][0].tolist(), seen[1][0].tolist()
    assert free[0] == eos
    ends = set(caption_end_token_ids(tok, eos))
    assert not ends & set(held[:3]) and held[3] == eos


def test_load_captioner_still_loads_recap(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "recap"}))
    import model.recap as recap_module
    called = []
    original = recap_module.RECAP.from_pretrained
    recap_module.RECAP.from_pretrained = classmethod(lambda cls, p: called.append(p) or "recap")
    try:
        assert load_captioner(str(tmp_path)) == "recap" and called == [str(tmp_path)]
    finally:
        recap_module.RECAP.from_pretrained = original


def _train(monkeypatch, llm_dir, data_files, out, epochs, *extra):
    import train
    csv, h5, cache = data_files
    argv = ["train.py", "--arch", "prefix", "--decoder_name", llm_dir,
            "--train_csv", csv, "--train_hdf5", h5, "--train_retrieval_cache", cache,
            "--val_csv", csv, "--val_hdf5", h5, "--val_retrieval_cache", cache,
            "--output_dir", str(out), "--rag", "--retrieval_k", "2", "--retrieval_strategy", "topk",
            "--num_train_epochs", str(epochs), "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "2", "--learning_rate", "1e-2", "--eval_save_strategy", "epoch",
            "--save_total_limit", "2", "--num_workers", "0", "--logging_steps", "1", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()


def _trained(path):
    from safetensors.torch import load_file
    return load_file(str(Path(path) / "model.safetensors"))


def test_train_py_keeps_the_best_epoch_and_saves_small_checkpoints(monkeypatch, llm_dir, data_files, tmp_path):
    out = tmp_path / "run"
    _train(monkeypatch, llm_dir, data_files, out, 3, "--load_best_model_at_end")
    state = json.loads(max(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
                       .joinpath("trainer_state.json").read_text())
    best = state["best_model_checkpoint"]
    final = _trained(out)
    assert all(torch.equal(final[k], v) for k, v in _trained(best).items())
    model = load_captioner(str(out))
    assert isinstance(model, AudioPrefixLM) and model.config.retrieval_k == 2 and model.config.rag
    assert set(final) == set(model._trained_names())
    assert json.loads((out / "config.json").read_text())["model_type"] == "audio_prefix_lm"


def test_train_py_resumes_exactly(monkeypatch, llm_dir, data_files, tmp_path, capsys):
    """A run cut short after epoch 2 and resumed ends with exactly the weights
    of the uninterrupted run (same schedule, RNG and data order)."""
    import shutil
    straight, cut = tmp_path / "straight", tmp_path / "cut"
    _train(monkeypatch, llm_dir, data_files, straight, 3)
    shutil.copytree(straight, cut)
    last = max(cut.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    shutil.rmtree(last)                                  # as if killed during epoch 3
    (cut / "model.safetensors").unlink()
    capsys.readouterr()
    _train(monkeypatch, llm_dir, data_files, cut, 3, "--resume")
    assert f"[resume] continuing from {cut / 'checkpoint-4'}" in capsys.readouterr().out
    a, b = _trained(straight), _trained(cut)
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_train_py_rejects_cross_attention_options(monkeypatch, llm_dir, data_files, tmp_path):
    for bad in (["--attention_size", "28"], ["--train_decoder"]):
        with pytest.raises(SystemExit):
            _train(monkeypatch, llm_dir, data_files, tmp_path / "x", 1, *bad)


def test_decoding_ignores_the_llms_own_sampling_settings(prefix, data_files):
    """Llama 3.2's generation_config samples; decoding must stay greedy/beam."""
    model, tok = prefix
    model.eval()
    csv, h5, cache = data_files
    with h5py.File(h5, "r") as f:
        feats = torch.from_numpy(f["abc_0"][()]).unsqueeze(0)
    start = torch.tensor([[tok.eos_token_id, 5, 6]])
    gc = model.llm.get_base_model().generation_config
    gc.do_sample, gc.temperature, gc.top_k, gc.repetition_penalty = True, 5.0, 0, 3.0
    runs = [model.generate(feats, start, max_new_tokens=6, num_beams=1, eos_token_id=tok.eos_token_id,
                           pad_token_id=tok.pad_token_id)[0].tolist() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2] == _greedy_by_hand(model, feats, start, len(runs[0]))


def test_checkpoints_pin_the_llm_revision(prefix, tmp_path, monkeypatch):
    model, tok = prefix
    model.config.llm_revision = "abc123"
    model.save_pretrained(tmp_path / "ck")
    import model.prefix_lm as pl
    seen = {}
    original = pl.AutoModelForCausalLM.from_pretrained
    def spy(name, **kw):
        seen.update(kw)
        kw.pop("revision")
        return original(name, **kw)
    monkeypatch.setattr(pl.AutoModelForCausalLM, "from_pretrained", spy)
    AudioPrefixLM.from_pretrained(str(tmp_path / "ck"))
    assert seen["revision"] == "abc123"
