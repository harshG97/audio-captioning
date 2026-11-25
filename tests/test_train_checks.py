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
                 "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"]:
        (ckpt / name).write_text(name)
    (ckpt / "trainer_state.json").write_text(f'{{"global_step": {step}}}')
    return ckpt


def _save(callback, run_dir, epoch, step):
    callback.on_save(SimpleNamespace(output_dir=str(run_dir)),
                     SimpleNamespace(epoch=epoch, global_step=step, is_world_process_zero=True),
                     None)


def _run_epochs(cb, run_dir, n):
    for epoch in range(1, n + 1):
        _checkpoint(run_dir, epoch * 7)
        _save(cb, run_dir, float(epoch), epoch * 7)
    return sorted(p.name for p in (run_dir / "snapshots").iterdir())


def test_full_checkpoints_only(tmp_path):
    """The KEEP=last default: every 20th epoch, as a resumable checkpoint."""
    assert _run_epochs(train.SnapshotCallback(every=0, full_every=20), tmp_path, 45) == \
        ["epoch020", "epoch040"]
    kept = sorted(p.name for p in (tmp_path / "snapshots" / "epoch020").iterdir())
    assert kept == sorted(p.name for p in (tmp_path / "checkpoint-140").iterdir())
    assert train.is_full_checkpoint(str(tmp_path / "snapshots" / "epoch020"))


def test_model_snapshots_between_full_checkpoints(tmp_path):
    cb = train.SnapshotCallback(every=10, full_every=20)
    assert _run_epochs(cb, tmp_path, 40) == ["epoch010", "epoch020", "epoch030", "epoch040"]
    snaps = tmp_path / "snapshots"
    model_only = sorted(p.name for p in (snaps / "epoch010").iterdir())
    assert model_only == ["config.json", "model.safetensors", "tokenizer.json", "trainer_state.json"]
    assert not train.is_full_checkpoint(str(snaps / "epoch030"))
    assert train.is_full_checkpoint(str(snaps / "epoch020"))
    assert train.is_full_checkpoint(str(snaps / "epoch040"))


def test_snapshot_is_not_overwritten_on_resume(tmp_path):
    cb = train.SnapshotCallback(full_every=2)
    _checkpoint(tmp_path, 4)
    _save(cb, tmp_path, 2.0, 4)
    marker = tmp_path / "snapshots" / "epoch002" / "config.json"
    marker.write_text("original")
    _save(cb, tmp_path, 2.0, 4)          # the same save again, e.g. after a resume
    assert marker.read_text() == "original"


def test_snapshot_skips_other_processes(tmp_path):
    cb = train.SnapshotCallback(every=1, full_every=1)
    _checkpoint(tmp_path, 3)
    cb.on_save(SimpleNamespace(output_dir=str(tmp_path)),
               SimpleNamespace(epoch=1.0, global_step=3, is_world_process_zero=False), None)
    assert not (tmp_path / "snapshots").exists()


REQUIRED = ["--train_csv", "t.csv", "--train_hdf5", "t.h5", "--train_retrieval_cache", "c.json",
            "--output_dir", "out", "--retrieval_k", "4", "--retrieval_strategy", "topk"]


@pytest.mark.parametrize("flag", ["--snapshot_every", "--checkpoint_every"])
def test_snapshot_flags_need_epoch_saves(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, flag, "10",
                                      "--eval_save_strategy", "steps"])
    with pytest.raises(SystemExit):
        train.parse_args()
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, flag, "-1",
                                      "--eval_save_strategy", "epoch"])
    with pytest.raises(SystemExit):
        train.parse_args()
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, flag, "10",
                                      "--eval_save_strategy", "epoch"])
    assert getattr(train.parse_args(), flag.lstrip("-")) == 10


def test_keep_best_checkpoint_copies_and_replaces(tmp_path):
    first = _checkpoint(tmp_path, 10)
    assert train.keep_best_checkpoint(str(first), str(tmp_path)) == str(tmp_path / "snapshots" / "best")
    best = tmp_path / "snapshots" / "best"
    assert (best / "optimizer.pt").read_text() == "optimizer.pt"
    (best / "stale.txt").write_text("from an earlier best")
    second = _checkpoint(tmp_path, 20)
    (second / "config.json").write_text("second")
    train.keep_best_checkpoint(str(second), str(tmp_path))
    assert (best / "config.json").read_text() == "second"
    assert not (best / "stale.txt").exists()


def test_keep_best_checkpoint_without_a_best(tmp_path):
    assert train.keep_best_checkpoint(None, str(tmp_path)) is None
    assert train.keep_best_checkpoint(str(tmp_path / "gone"), str(tmp_path)) is None
    assert not (tmp_path / "snapshots").exists()


def test_resume_from_must_be_a_full_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, "--resume_from", str(tmp_path)])
    with pytest.raises(SystemExit):
        train.parse_args()
    model_only = tmp_path / "model_only"
    model_only.mkdir()
    for name in ["trainer_state.json", "model.safetensors", "config.json"]:
        (model_only / name).write_text(name)
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, "--resume_from", str(model_only)])
    with pytest.raises(SystemExit):
        train.parse_args()
    ckpt = _checkpoint(tmp_path, 5)
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, "--resume_from", str(ckpt)])
    assert train.parse_args().resume_from == str(ckpt)


def test_track_best_needs_a_validation_set(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, "--track_best"])
    with pytest.raises(SystemExit):
        train.parse_args()
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, "--track_best", "--val_csv", "v.csv"])
    assert train.parse_args().track_best


def test_seed_is_set_before_the_model_is_built(monkeypatch):
    """The cross-attention layers are randomly initialised in build_recap, so
    the seed must be set first or runs with the same seed differ."""
    calls = []
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, "--seed", "7"])
    monkeypatch.setattr(train, "set_seed", lambda seed: calls.append(("seed", seed)))

    def fake_build(**kwargs):
        calls.append(("build", None))
        raise RuntimeError("stop after building")

    monkeypatch.setattr(train, "build_recap", fake_build)
    with pytest.raises(RuntimeError, match="stop after building"):
        train.main()
    assert calls == [("seed", 7), ("build", None)]
