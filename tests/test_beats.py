"""Tests for the BEATs feature extractor (data/extract_beats_features.py).

Uses a tiny randomly initialised BEATs with the real model's options
(relative positions, gated relative positions, deep norm, an AudioSet
classifier head), saved in the official checkpoint format, and a few short
wav files, so nothing is downloaded.

Run: python -m pytest tests/test_beats.py -q  (CPU, a few seconds)
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data.extract_beats_features as xb
from model.beats import BEATs, BEATsConfig

WIDTH = 48
CFG = dict(input_patch_size=16, embed_dim=32, encoder_layers=2, encoder_embed_dim=WIDTH,
           encoder_ffn_embed_dim=96, encoder_attention_heads=4, relative_position_embedding=True,
           num_buckets=32, max_distance=128, gru_rel_pos=True, deep_norm=True,
           layer_wise_gradient_decay_ratio=0.6, conv_pos=16, conv_pos_groups=4,
           finetuned_model=True, predictor_class=3)
LABELS = {0: "/m/09x0r", 1: "/m/0bt9lr", 2: "/m/06mb1"}     # speech, dog, rain
CLIPS = [("abc", 0, "a man speaks", 12.0), ("def", 30, "a dog barks", 10.0),
         ("ghi", 5, "rain falls", 3.0)]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = BEATs(BEATsConfig(CFG))
    path = tmp_path_factory.mktemp("beats") / "tiny_beats.pt"
    torch.save({"cfg": CFG, "model": model.state_dict(), "label_dict": LABELS}, path)
    return path


@pytest.fixture(scope="module")
def audio(tmp_path_factory):
    """Clips longer than, equal to and shorter than 10 s, at 44.1 kHz."""
    root = tmp_path_factory.mktemp("audio")
    rng = np.random.default_rng(0)
    for yid, start, _, seconds in CLIPS:
        sr = 44100
        sf.write(root / f"{yid}_{start}.wav", 0.1 * rng.standard_normal(int(sr * seconds)), sr)
    csv = root / "clips.csv"
    pd.DataFrame([{"youtube_id": y, "start_time": s, "caption": c} for y, s, c, _ in CLIPS]
                 ).to_csv(csv, index=False)
    return root, csv


def _extract(audio, out, checkpoint, *extra):
    root, csv = audio
    xb.main(["--checkpoint", str(checkpoint), "--audio_dir", str(root), "--captions_csv", str(csv),
             "--output_dir", str(out), "--dataset", "audiocaps", "--split", "val", *extra])
    with h5py.File(out / "audiocaps_val.hdf5", "r") as f:
        return {k: f[k][()] for k in f}


def test_fit_to_clip_cuts_and_repeats():
    long = np.arange(20, dtype=np.float32)
    assert np.array_equal(xb.fit_to_clip(long, 8), long[:8])
    short = np.arange(3, dtype=np.float32)
    assert np.array_equal(xb.fit_to_clip(short, 8), [0, 1, 2, 0, 1, 2, 0, 1])
    assert np.array_equal(xb.fit_to_clip(short, 3), short)
    with pytest.raises(ValueError):
        xb.fit_to_clip(np.zeros(0, np.float32), 8)


def test_load_clip_is_ten_seconds_at_16k(audio):
    root, _ = audio
    for yid, start, _, seconds in CLIPS:
        wave = xb.load_clip(str(root / f"{yid}_{start}.wav"))
        assert wave.shape == (xb.CLIP_SAMPLES,) and wave.dtype == np.float32
    short = xb.load_clip(str(root / "ghi_5.wav"))
    n = 3 * xb.SAMPLE_RATE
    assert np.allclose(short[n:2 * n], short[:n])          # repeated, not silent


def test_patches_are_time_major(checkpoint):
    """BEATs flattens its (time, frequency) patch grid row by row, so each run
    of 8 consecutive vectors is one time step: what pool_bands assumes."""
    model, _ = xb.load_beats(str(checkpoint))
    seen = {}
    model.patch_embedding.register_forward_hook(lambda m, i, o: seen.update(shape=o.shape))
    enc = xb.BEATsEncoder(model, torch.device("cpu"), pool=1)
    feats, _ = enc([np.zeros(xb.CLIP_SAMPLES, np.float32)])
    assert tuple(seen["shape"][2:]) == (62, 8)              # (time, frequency)
    assert enc.n_bands == 8 and feats.shape == (1, 496, WIDTH)


def test_pool_bands_averages_neighbouring_bands_in_each_time_step():
    t, f, d = 3, 8, 2
    hidden = torch.arange(t * f * d, dtype=torch.float32).view(1, t * f, d)
    grid = hidden.view(t, f, d)
    pooled = xb.pool_bands(hidden, f, 4)
    assert pooled.shape == (1, t * 2, d)
    for step in range(t):
        assert torch.equal(pooled[0, 2 * step], grid[step, :4].mean(0))
        assert torch.equal(pooled[0, 2 * step + 1], grid[step, 4:].mean(0))
    assert torch.equal(xb.pool_bands(hidden, f, 1), hidden)
    with pytest.raises(ValueError):
        xb.pool_bands(hidden, f, 3)


def test_features_are_the_last_layer_output(checkpoint):
    """The hook must return what BEATs' transformer encoder produces."""
    model, _ = xb.load_beats(str(checkpoint))
    wave = np.random.default_rng(1).standard_normal(xb.CLIP_SAMPLES).astype(np.float32) * 0.1
    enc = xb.BEATsEncoder(model, torch.device("cpu"), pool=1)
    feats, probs = enc([wave])
    with torch.no_grad():
        fbank = model.preprocess(torch.from_numpy(wave)[None]).unsqueeze(1)
        x = model.patch_embedding(fbank)
        x = model.layer_norm(x.reshape(1, x.shape[1], -1).transpose(1, 2))
        x = model.post_extract_proj(x) if model.post_extract_proj is not None else x
        expected, _ = model.encoder(x)
        expected_probs, _ = model.extract_features(torch.from_numpy(wave)[None])
    assert np.allclose(feats[0], expected[0].numpy(), atol=1e-6)
    assert torch.allclose(probs, expected_probs)


def test_extraction_writes_one_entry_per_clip(audio, checkpoint, tmp_path):
    feats = _extract(audio, tmp_path / "p4", checkpoint)
    assert sorted(feats) == ["abc_0", "def_30", "ghi_5"]
    for v in feats.values():
        assert v.shape == (124, WIDTH) and v.dtype == np.float32 and np.isfinite(v).all()
    full = _extract(audio, tmp_path / "p1", checkpoint, "--pool", "1")
    assert all(v.shape == (496, WIDTH) for v in full.values())
    for k in feats:     # pooled = mean of the full features over 4 neighbouring bands
        expect = full[k].reshape(62, 2, 4, WIDTH).mean(2).reshape(124, WIDTH)
        assert np.allclose(feats[k], expect, atol=1e-5)


def test_batching_and_workers_do_not_change_features(audio, checkpoint, tmp_path):
    one = _extract(audio, tmp_path / "b1", checkpoint, "--batch_size", "1")
    three = _extract(audio, tmp_path / "b3", checkpoint, "--batch_size", "3", "--num_workers", "2")
    again = _extract(audio, tmp_path / "b3b", checkpoint, "--batch_size", "3")
    for k in one:
        assert np.allclose(one[k], three[k], atol=1e-5)
        assert np.array_equal(three[k], again[k])


def test_checkpoint_must_match_the_model(checkpoint, tmp_path):
    ckpt = torch.load(checkpoint, weights_only=True)
    ckpt["model"].pop(next(iter(ckpt["model"])))
    torch.save(ckpt, tmp_path / "missing.pt")
    with pytest.raises(RuntimeError):
        xb.load_beats(str(tmp_path / "missing.pt"))
    torch.save({"state_dict": {}}, tmp_path / "other.pt")
    with pytest.raises(SystemExit):
        xb.load_beats(str(tmp_path / "other.pt"))


def test_check_prints_labels_and_writes_nothing(audio, checkpoint, tmp_path, capsys):
    root, csv = audio
    xb.main(["--checkpoint", str(checkpoint), "--audio_dir", str(root), "--captions_csv", str(csv),
             "--output_dir", str(tmp_path / "out"), "--dataset", "audiocaps", "--check", "2"])
    out = capsys.readouterr().out
    assert "caption: a man speaks" in out and "caption: a dog barks" in out
    assert "Speech" in out and "/m/" not in out          # label ids shown by name
    assert not (tmp_path / "out").exists()


def test_check_skips_a_checkpoint_without_a_classifier(audio, tmp_path, capsys):
    torch.manual_seed(0)
    cfg = dict(CFG, finetuned_model=False)
    torch.save({"cfg": cfg, "model": BEATs(BEATsConfig(cfg)).state_dict()}, tmp_path / "pre.pt")
    root, csv = audio
    xb.main(["--checkpoint", str(tmp_path / "pre.pt"), "--audio_dir", str(root),
             "--captions_csv", str(csv), "--output_dir", str(tmp_path), "--dataset", "audiocaps",
             "--check", "1"])
    assert "--check skipped" in capsys.readouterr().out


def test_match_hdf5_stops_when_audio_is_missing(audio, checkpoint, tmp_path):
    root, csv = audio
    ref = tmp_path / "clap_val.hdf5"
    with h5py.File(ref, "w") as f:                       # CLAP has all three clips
        for yid, start, *_ in CLIPS:
            f.create_dataset(f"{yid}_{start}", data=np.zeros((2, 2), np.float32))
    feats = _extract(audio, tmp_path / "ok", checkpoint, "--match_hdf5", str(ref))
    assert len(feats) == 3
    partial = tmp_path / "partial"
    partial.mkdir()
    for wav in list(root.glob("*.wav"))[:2]:
        (partial / wav.name).symlink_to(wav)
    with pytest.raises(SystemExit, match="1 clips"):
        _extract((partial, csv), tmp_path / "bad", checkpoint, "--match_hdf5", str(ref))
    assert not (tmp_path / "bad").exists()


# ---- several layers side by side ----

def test_layers_are_stored_side_by_side(audio, checkpoint, tmp_path):
    last = _extract(audio, tmp_path / "last", checkpoint)
    both = _extract(audio, tmp_path / "both", checkpoint, "--layers", "1,2")
    assert sorted(both) == sorted(last)
    model, _ = xb.load_beats(str(checkpoint))
    root, _ = audio
    for key, feats in both.items():
        assert feats.shape == (124, 2 * WIDTH) and feats.dtype == np.float32
        assert np.array_equal(feats[:, WIDTH:], last[key])          # the last layer, exactly as before
        # Layer 1, computed straight from the model.
        wave = torch.from_numpy(xb.load_clip(str(root / f"{key}.wav")))[None]
        with torch.no_grad():
            x = model.patch_embedding(model.preprocess(wave).unsqueeze(1))
            x = model.layer_norm(x.reshape(1, x.shape[1], -1).transpose(1, 2))
            x = model.post_extract_proj(x) if model.post_extract_proj is not None else x
            layer1, _ = model.encoder.extract_features(x, tgt_layer=0)
        expected = xb.pool_bands(layer1, 8, 4)[0].numpy()
        assert np.allclose(feats[:, :WIDTH], expected, atol=1e-5)
        assert not np.allclose(feats[:, :WIDTH], feats[:, WIDTH:], atol=1e-3)
    one_by_one = _extract(audio, tmp_path / "b1", checkpoint, "--layers", "1,2", "--batch_size", "1")
    assert all(np.allclose(one_by_one[k], both[k], atol=1e-5) for k in both)   # each batch gets its own layers
    only_first = _extract(audio, tmp_path / "first", checkpoint, "--layers", "1")
    assert all(np.allclose(only_first[k], both[k][:, :WIDTH], atol=1e-6) for k in both)


def test_layers_option_is_checked():
    assert xb.parse_layers(None, 12) is None and xb.parse_layers("4,8,12", 12) == [4, 8, 12]
    for bad in ("0,4", "4,13", "8,4", "4,4", "a,b", "4;8"):
        with pytest.raises(SystemExit):
            xb.parse_layers(bad, 12)


# ---- sound labels from the stored features ----

def _features_dir(audio, checkpoint, tmp_path, *extra):
    out = tmp_path / "feats"
    _extract(audio, out, checkpoint, *extra)
    encoder = f"beats {Path(checkpoint).name} pool=4" + (f" layers={extra[1]}" if extra else "")
    (out / "encoder.txt").write_text(encoder + "\n")
    return out


def test_sound_labels_match_a_full_beats_pass(audio, checkpoint, tmp_path):
    import data.build_sound_labels as bl
    out = _features_dir(audio, checkpoint, tmp_path)
    bl.main(["--checkpoint", str(checkpoint), "--feat_dir", str(out), "--splits", "val", "--top", "3"])
    labels = json.loads((out / "labels_val.json").read_text())
    model, label_dict = xb.load_beats(str(checkpoint))
    enc = xb.BEATsEncoder(model, torch.device("cpu"), pool=4)
    root, _ = audio
    names = {"/m/09x0r": "speech", "/m/0bt9lr": "dog", "/m/06mb1": "rain"}
    assert sorted(labels) == ["abc_0", "def_30", "ghi_5"]
    for key, entries in labels.items():
        _, probs = enc([xb.load_clip(str(root / f"{key}.wav"))])        # the model's own answer
        expected = {names[label_dict[i]]: float(probs[0, i]) for i in range(3)}
        assert [n for n, _ in entries] == sorted(expected, key=expected.get, reverse=True)
        for name, prob in entries:
            assert prob == pytest.approx(expected[name], abs=2e-4)
        assert [p for _, p in entries] == sorted((p for _, p in entries), reverse=True)


def test_sound_labels_refuse_the_wrong_inputs(audio, checkpoint, tmp_path):
    import data.build_sound_labels as bl
    out = _features_dir(audio, checkpoint, tmp_path)
    args = ["--feat_dir", str(out), "--splits", "val"]
    other = tmp_path / "other_name.pt"
    other.write_bytes(Path(checkpoint).read_bytes())
    with pytest.raises(SystemExit, match="was built with"):               # features from another checkpoint
        bl.main(["--checkpoint", str(other), *args])
    (out / "encoder.txt").write_text(f"beats {Path(checkpoint).name} pool=4 layers=1,2\n")
    with pytest.raises(SystemExit, match="several layers"):
        bl.main(["--checkpoint", str(checkpoint), *args])
    (out / "encoder.txt").write_text(f"beats pre.pt pool=4\n")
    torch.manual_seed(0)
    cfg = dict(CFG, finetuned_model=False)
    torch.save({"cfg": cfg, "model": BEATs(BEATsConfig(cfg)).state_dict()}, tmp_path / "pre.pt")
    with pytest.raises(SystemExit, match="no AudioSet classifier"):
        bl.main(["--checkpoint", str(tmp_path / "pre.pt"), *args])
    (out / "encoder.txt").unlink()
    with pytest.raises(SystemExit, match="no encoder.txt"):
        bl.main(["--checkpoint", str(checkpoint), *args])
