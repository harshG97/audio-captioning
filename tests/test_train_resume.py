"""Tests for train.py's resume support and the options the Slurm runs rely on.

transformers 4.45 pickles numpy RNG state into checkpoint rng_state.pth, and
torch >= 2.6 refuses to load that with its default weights_only=True, so a
resumed run used to crash. checkpoint_safe_globals() allowlists exactly those
numpy types.

Run: python -m pytest tests/test_train_resume.py -q
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train  # noqa: E402


def _save_rng_state_like_transformers(path):
    # Same structure as Trainer._save_rng_state in transformers 4.45.
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state()}, path)


def test_rng_state_loads_with_allowlist(tmp_path):
    path = tmp_path / "rng_state.pth"
    _save_rng_state_like_transformers(path)
    with torch.serialization.safe_globals(train.checkpoint_safe_globals()):
        state = torch.load(path)
    assert set(state) == {"python", "numpy", "cpu"}
    assert np.array_equal(state["numpy"][1], np.random.get_state()[1])


@pytest.mark.skipif(not hasattr(torch.serialization, "safe_globals"),
                    reason="torch without weights_only allowlists")
def test_rng_state_needs_the_allowlist(tmp_path):
    """Guards the reason the allowlist exists: if a future torch or numpy
    loads this without help, the allowlist is no longer needed."""
    path = tmp_path / "rng_state.pth"
    _save_rng_state_like_transformers(path)
    with pytest.raises(Exception, match="Weights only load failed"):
        torch.load(path)


def test_resume_checkpoint_picks_latest(tmp_path):
    assert train.resume_checkpoint(str(tmp_path / "missing")) is None
    assert train.resume_checkpoint(str(tmp_path)) is None
    for step in (5, 40, 12):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert train.resume_checkpoint(str(tmp_path)).endswith("checkpoint-40")


REQUIRED = ["train.py", "--train_csv", "t.csv", "--train_hdf5", "t.h5",
            "--train_retrieval_cache", "c.json", "--output_dir", "out", "--no-rag"]


def test_early_stopping_needs_best_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", REQUIRED + ["--early_stopping_patience", "3"])
    with pytest.raises(SystemExit):
        train.parse_args()
    monkeypatch.setattr(sys, "argv", REQUIRED + ["--early_stopping_patience", "3",
                                                 "--load_best_model_at_end"])
    assert train.parse_args().early_stopping_patience == 3


def test_new_options_default_to_old_behaviour(monkeypatch):
    monkeypatch.setattr(sys, "argv", REQUIRED)
    args = train.parse_args()
    assert (args.eval_save_strategy, args.early_stopping_patience, args.resume) == ("steps", 0, False)


def test_write_history(tmp_path):
    log = [{"loss": 3.2, "learning_rate": 5e-5, "grad_norm": 1.1, "epoch": 0.5, "step": 50},
           {"eval_loss": 2.9, "eval_runtime": 4.2, "epoch": 1.0, "step": 100},
           {"train_runtime": 99.0, "train_loss": 3.0, "epoch": 1.0, "step": 100}]
    path = tmp_path / "history.csv"
    assert train.write_history(log, str(path)) == 2
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(train.HISTORY_FIELDS)
    assert lines[1] == "50,0.5,3.2,,5e-05,1.1"
    assert lines[2] == "100,1.0,,2.9,,"
