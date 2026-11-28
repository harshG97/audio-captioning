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
from model.prefix_lm import AudioPrefixConfig, AudioPrefixLM, AudioProjector, build_prefix_lm, load_captioner
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


def _argv(llm_dir, data_files, out, epochs, *extra, arch="prefix"):
    csv, h5, cache = data_files
    return ["train.py", "--arch", arch, "--decoder_name", llm_dir,
            "--train_csv", csv, "--train_hdf5", h5, "--train_retrieval_cache", cache,
            "--val_csv", csv, "--val_hdf5", h5, "--val_retrieval_cache", cache,
            "--output_dir", str(out), "--rag", "--retrieval_k", "2", "--retrieval_strategy", "topk",
            "--num_train_epochs", str(epochs), "--per_device_train_batch_size", "1",
            "--per_device_eval_batch_size", "2", "--learning_rate", "1e-2", "--eval_save_strategy", "epoch",
            "--save_total_limit", "2", "--num_workers", "0", "--logging_steps", "1", *extra]


def _train(monkeypatch, llm_dir, data_files, out, epochs, *extra):
    import train
    monkeypatch.setattr(sys, "argv", _argv(llm_dir, data_files, out, epochs, *extra))
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


# ---- audio tokens: pooling and stacking ----

def test_projector_concat_keeps_what_mean_loses():
    torch.manual_seed(0)
    feats = torch.randn(2, ENC_LEN, FEAT)
    concat = AudioProjector(FEAT, 64, pool=2, merge="concat")
    out = concat(feats)
    assert out.shape == (2, ENC_LEN // 2, 64)
    stacked = torch.cat([feats[:, 0::2], feats[:, 1::2]], dim=-1)      # [v0|v1], [v2|v3], ...
    assert torch.allclose(out, concat.mlp(concat.norm(stacked)))
    # Swapping the two vectors of a group keeps their mean but not their order.
    swapped = feats.clone()
    swapped[:, 0], swapped[:, 1] = feats[:, 1], feats[:, 0]
    mean = AudioProjector(FEAT, 64, pool=2, merge="mean")
    assert torch.allclose(mean(feats), mean(swapped), atol=1e-6)
    assert not torch.allclose(out[:, 0], concat(swapped)[:, 0])
    assert torch.allclose(out[:, 1:], concat(swapped)[:, 1:])


def test_projector_pool_one_gives_a_token_per_vector():
    torch.manual_seed(0)
    feats = torch.randn(1, ENC_LEN, FEAT)
    mean, concat = (AudioProjector(FEAT, 64, pool=1, merge=m) for m in ("mean", "concat"))
    concat.load_state_dict(mean.state_dict())                # same shapes at pool 1
    assert mean(feats).shape == (1, ENC_LEN, 64) and torch.equal(mean(feats), concat(feats))
    for bad in (dict(pool=0), dict(pool=2, merge="max")):
        with pytest.raises(ValueError):
            AudioProjector(FEAT, 64, **{"pool": 2, **bad})


def test_audio_merge_is_saved_and_old_checkpoints_default_to_mean(llm_dir, data_files, tmp_path):
    torch.manual_seed(0)
    model, tok = build_prefix_lm(llm_dir, feature_width=FEAT, audio_pool=2, audio_merge="concat")
    assert model.projector.norm.normalized_shape == (2 * FEAT,)
    batch, _ = _batch(model, tok, data_files)
    model.eval()
    model.save_pretrained(tmp_path / "ck")
    loaded = load_captioner(str(tmp_path / "ck"))
    assert loaded.config.audio_merge == "concat" and loaded.projector.merge == "concat"
    with torch.no_grad():
        assert torch.equal(_forward(model, batch).logits, _forward(loaded, batch).logits)
        assert _forward(model, batch).logits.shape[:2] == batch["decoder_input_ids"].shape
    # A config written before the option existed has no audio_merge entry.
    cfg = json.loads((tmp_path / "ck" / "config.json").read_text())
    cfg.pop("audio_merge")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "config.json").write_text(json.dumps(cfg))
    assert AudioPrefixConfig.from_pretrained(str(tmp_path / "old")).audio_merge == "mean"
    with pytest.raises(ValueError):
        AudioPrefixConfig(audio_merge="max")


# ---- LoRA options, separate learning rates, projector warm-up ----

def test_lora_targets_attn_leaves_the_mlp_layers_alone(llm_dir):
    torch.manual_seed(0)
    model, _ = build_prefix_lm(llm_dir, feature_width=FEAT, lora_targets="attn", lora_r=4, lora_dropout=0.2)
    lora = [n for n in model._trained_names() if ".lora_" in n]
    assert lora and not any(t in n for n in lora for t in ("gate_proj", "up_proj", "down_proj"))
    for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert any(f"{target}.lora_A" in n for n in lora)
    a = next(p for n, p in model.named_parameters() if ".lora_A" in n)
    assert a.shape[0] == 4 and model.config.lora_targets == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert model.config.lora_dropout == 0.2


def test_linear_schedule_matches_the_trainers_default_and_can_be_delayed():
    from transformers.optimization import _get_linear_schedule_with_warmup_lr_lambda as default
    from train import linear_schedule
    for step in range(0, 101):
        assert linear_schedule(step, start=0, ramp=5, total=100) == pytest.approx(
            default(step, num_warmup_steps=5, num_training_steps=100))
    delayed = [linear_schedule(s, start=40, ramp=5, total=100) for s in range(101)]
    assert all(v == 0 for v in delayed[:41]) and delayed[45] == 1.0 and delayed[100] == 0.0
    assert delayed[41:45] == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert all(a >= b for a, b in zip(delayed[45:], delayed[46:]))        # decays from there


def _trainer(model, tmp_path, **kw):
    from transformers import TrainingArguments
    from train import RecapTrainer
    args = TrainingArguments(output_dir=str(tmp_path), learning_rate=1e-3, num_train_epochs=4,
                             warmup_ratio=0.05, weight_decay=0.1, report_to=["none"])
    return RecapTrainer(model=model, args=args, **kw)


def test_projector_and_lora_get_their_own_learning_rates(prefix, tmp_path):
    model, _ = prefix
    trainer = _trainer(model, tmp_path, projector_lr=1e-2)
    opt = trainer.create_optimizer()
    by_param = {id(p): g for g in opt.param_groups for p in g["params"]}
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert set(by_param) == {id(p) for p in trainable.values()}          # every weight, once
    assert sum(len(g["params"]) for g in opt.param_groups) == len(trainable)
    for n, p in trainable.items():
        g = by_param[id(p)]
        assert g["lr"] == (1e-2 if n.startswith("projector.") else 1e-3), n
        assert g["part"] == ("projector" if n.startswith("projector.") else "lora")
    assert {g["weight_decay"] for g in opt.param_groups} == {0.0, 0.1}   # as the Trainer splits them
    # Without the options the Trainer's own optimizer is used: one rate for all.
    plain = _trainer(model, tmp_path).create_optimizer()
    assert {g["lr"] for g in plain.param_groups} == {1e-3} and "part" not in plain.param_groups[0]


def test_lora_warmup_holds_loras_rate_at_zero(prefix, tmp_path):
    model, _ = prefix
    trainer = _trainer(model, tmp_path, lora_warmup_epochs=1)
    opt = trainer.create_optimizer()
    sched = trainer.create_scheduler(num_training_steps=40, optimizer=opt)    # 10 steps per epoch
    rates = []
    for _ in range(40):
        rates.append({g["part"]: g["lr"] for g in opt.param_groups})
        opt.step()
        sched.step()
    assert all(r["lora"] == 0 for r in rates[:11]) and rates[11]["lora"] > 0
    assert rates[0]["projector"] == 0 and rates[2]["projector"] == pytest.approx(1e-3)   # 2 ramp steps
    assert rates[12]["lora"] == pytest.approx(1e-3)
    assert rates[-1]["lora"] > 0 and rates[-1]["projector"] > 0


def test_train_py_warmup_freezes_lora_then_trains_it(monkeypatch, llm_dir, data_files, tmp_path):
    out = tmp_path / "run"
    _train(monkeypatch, llm_dir, data_files, out, 4, "--lora_warmup_epochs", "2", "--projector_lr", "2e-2",
           "--save_total_limit", "9")
    def lora_b(step):
        return [v for k, v in _trained(out / f"checkpoint-{step}").items() if ".lora_B" in k]
    assert all(not b.any() for b in lora_b(2)) and all(not b.any() for b in lora_b(4))   # 2 steps per epoch
    assert any(b.any() for b in lora_b(8))
    proj = lambda step: _trained(out / f"checkpoint-{step}")["projector.mlp.0.weight"]
    assert not torch.equal(proj(2), proj(4))                    # the projector trains meanwhile
    # Adam's state for LoRA starts afresh when the hold ends (step 4); the
    # projector's runs on. The count is per weight, in optimizer.pt.
    def adam_steps(step):
        opt = torch.load(out / f"checkpoint-{step}" / "optimizer.pt", weights_only=True)
        return {g["part"]: {int(opt["state"][i]["step"]) for i in g["params"]} for g in opt["param_groups"]}
    assert adam_steps(4) == {"projector": {4}, "lora": {4}}
    assert adam_steps(6) == {"projector": {6}, "lora": {2}}
    assert adam_steps(8) == {"projector": {8}, "lora": {4}}


def test_train_py_resumes_exactly_across_the_warmup(monkeypatch, llm_dir, data_files, tmp_path, capsys):
    import shutil
    extra = ("--lora_warmup_epochs", "2", "--projector_lr", "2e-2", "--save_total_limit", "9")
    straight, cut = tmp_path / "straight", tmp_path / "cut"
    _train(monkeypatch, llm_dir, data_files, straight, 4, *extra)
    shutil.copytree(straight, cut)
    for step in (6, 8):
        shutil.rmtree(cut / f"checkpoint-{step}")               # as if killed right after the warm-up
    (cut / "model.safetensors").unlink()
    capsys.readouterr()
    _train(monkeypatch, llm_dir, data_files, cut, 4, *extra, "--resume")
    assert f"[resume] continuing from {cut / 'checkpoint-4'}" in capsys.readouterr().out
    a, b = _trained(straight), _trained(cut)
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert any(v.any() for k, v in a.items() if ".lora_B" in k)


def test_prefix_options_are_refused_for_recap_and_checked(monkeypatch, llm_dir, data_files, tmp_path):
    import train
    for bad in (["--projector_lr", "1e-3"], ["--lora_warmup_epochs", "1"], ["--audio_merge", "concat"],
                ["--audio_pool", "2"], ["--lora_targets", "attn"], ["--lora_r", "8"]):
        monkeypatch.setattr(sys, "argv", _argv("gpt2", data_files, tmp_path, 3, *bad, arch="xattn"))
        with pytest.raises(SystemExit):
            train.parse_args()
    for bad in (["--lora_warmup_epochs", "3"], ["--lora_warmup_epochs", "-1"], ["--projector_lr", "0"]):
        monkeypatch.setattr(sys, "argv", _argv(llm_dir, data_files, tmp_path, 3, *bad))
        with pytest.raises(SystemExit):
            train.parse_args()
    monkeypatch.setattr(sys, "argv", _argv(llm_dir, data_files, tmp_path, 3, "--lora_r", "8"))
    assert train.parse_args().lora_alpha == 16                  # alpha follows the rank


def test_train_py_passes_the_model_options_on(monkeypatch, llm_dir, data_files, tmp_path):
    out = tmp_path / "run"
    _train(monkeypatch, llm_dir, data_files, out, 1, "--audio_pool", "2", "--audio_merge", "concat",
           "--lora_r", "4", "--lora_dropout", "0.2", "--lora_targets", "attn")
    cfg = json.loads((out / "config.json").read_text())
    assert (cfg["audio_pool"], cfg["audio_merge"], cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"]) \
        == (2, "concat", 4, 8, 0.2)
    assert cfg["lora_targets"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    model = load_captioner(str(out))
    assert model.projector.merge == "concat" and model.projector.pool == 2
    out2 = tmp_path / "default"
    _train(monkeypatch, llm_dir, data_files, out2, 1)
    cfg = json.loads((out2 / "config.json").read_text())
    assert (cfg["audio_pool"], cfg["audio_merge"], cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"]) \
        == (4, "mean", 16, 32, 0.05) and len(cfg["lora_targets"]) == 7
