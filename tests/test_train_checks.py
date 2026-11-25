"""Tests for train.py's feature-width check, snapshot callback and argument checks.

Run: python -m pytest tests/test_train_checks.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train  # noqa: E402


def _features(path, width):
    with h5py.File(path, "w") as f:
        f["clip_a"] = np.zeros((64, width), dtype=np.float32)
        f["clip_b"] = np.zeros((64, width), dtype=np.float32)
    return str(path)


def test_feature_width_reads_last_dimension(tmp_path):
    assert train.feature_width(_features(tmp_path / "f.hdf5", 768)) == 768


def test_matching_width_passes(tmp_path):
    train.check_feature_width(_features(tmp_path / "f.hdf5", 768), 768)


def test_mismatched_width_stops_with_a_clear_message(tmp_path):
    path = _features(tmp_path / "f.hdf5", 1024)
    with pytest.raises(SystemExit, match="1024-wide features but the decoder is 768 wide"):
        train.check_feature_width(path, 768)


def _checkpoint(run_dir, step):
    ckpt = Path(run_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    for name in ["model.safetensors", "config.json", "tokenizer.json",
                 "trainer_state.json", "optimizer.pt", "scheduler.pt",
                 "rng_state.pth", "scaler.pt"]:
        (ckpt / name).write_text(name)
    return ckpt


def _save(callback, run_dir, epoch, step):
    callback.on_save(SimpleNamespace(output_dir=str(run_dir)),
                     SimpleNamespace(epoch=epoch, global_step=step, is_world_process_zero=True),
                     None)


def test_snapshot_keeps_every_nth_epoch_without_optimizer_state(tmp_path):
    cb = train.SnapshotCallback(every=10)
    for epoch in range(1, 21):
        _checkpoint(tmp_path, epoch * 7)
        _save(cb, tmp_path, float(epoch), epoch * 7)
    snaps = sorted(p.name for p in (tmp_path / "snapshots").iterdir())
    assert snaps == ["epoch010", "epoch020"]
    kept = sorted(p.name for p in (tmp_path / "snapshots" / "epoch010").iterdir())
    assert kept == ["config.json", "model.safetensors", "tokenizer.json", "trainer_state.json"]


def test_snapshot_is_not_overwritten_on_resume(tmp_path):
    cb = train.SnapshotCallback(every=2)
    _checkpoint(tmp_path, 4)
    _save(cb, tmp_path, 2.0, 4)
    marker = tmp_path / "snapshots" / "epoch002" / "config.json"
    marker.write_text("original")
    _save(cb, tmp_path, 2.0, 4)          # the same save again, e.g. after a resume
    assert marker.read_text() == "original"


def test_snapshot_skips_other_processes(tmp_path):
    cb = train.SnapshotCallback(every=1)
    _checkpoint(tmp_path, 3)
    cb.on_save(SimpleNamespace(output_dir=str(tmp_path)),
               SimpleNamespace(epoch=1.0, global_step=3, is_world_process_zero=False), None)
    assert not (tmp_path / "snapshots").exists()


REQUIRED = ["--train_csv", "t.csv", "--train_hdf5", "t.h5", "--train_retrieval_cache", "c.json",
            "--output_dir", "out", "--retrieval_k", "4", "--retrieval_strategy", "topk"]


def test_snapshot_every_needs_epoch_saves(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, "--snapshot_every", "10",
                                      "--eval_save_strategy", "steps"])
    with pytest.raises(SystemExit):
        train.parse_args()
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, "--snapshot_every", "10",
                                      "--eval_save_strategy", "epoch"])
    assert train.parse_args().snapshot_every == 10
