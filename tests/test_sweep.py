"""Tests for the --skip_existing logic of scripts/sweep.py's score stage.

Run: python -m pytest tests/test_sweep.py -q
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("sweep", REPO_ROOT / "scripts" / "sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

FULL = {"CIDEr": 0.5, "METEOR": 0.2, "SPICE": 0.1, "SPIDEr": 0.3,
        "train_copy_rate": 0.4, "tokenizer": "ptb"}


def _write(tmp_path, metrics):
    path = tmp_path / "metrics" / sweep.metrics_name(3, 0, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics))
    return path


@pytest.mark.parametrize("drop, settings, expect", [
    ([], {}, []),
    (["METEOR"], {}, ["METEOR"]),
    (["METEOR"], {"skip_meteor": True}, []),
    (["SPICE", "SPIDEr"], {}, ["SPICE", "SPIDEr"]),
    (["SPICE", "SPIDEr"], {"skip_spice": True}, []),
    (["train_copy_rate"], {}, ["train_copy_rate"]),
    (["train_copy_rate"], {"train_csv": None}, []),
])
def test_missing_metrics(tmp_path, drop, settings, expect):
    path = _write(tmp_path, {k: v for k, v in FULL.items() if k not in drop})
    kw = {"skip_meteor": False, "skip_spice": False, "train_csv": "train.csv", **settings}
    assert sweep.missing_metrics(path, **kw) == expect


def test_untokenized_file_is_rescored(tmp_path):
    path = _write(tmp_path, {**FULL, "tokenizer": "none"})
    assert sweep.missing_metrics(path, skip_meteor=False, skip_spice=False,
                                 train_csv="train.csv") == ["PTB tokenization"]


def test_stage_score_skips_complete_file(tmp_path, capsys):
    _write(tmp_path, FULL)
    sweep.stage_score(run_dir=tmp_path, pred_path=tmp_path / "p.json", nb=3, nrn=0, lp=0.0,
                      skip_existing=True, dry_run=True, train_csv="train.csv")
    out = capsys.readouterr().out
    assert "[skip score]" in out and "score_predictions.py" not in out


def test_stage_score_rescores_incomplete_file(tmp_path, capsys):
    _write(tmp_path, {k: v for k, v in FULL.items() if k != "METEOR"})
    sweep.stage_score(run_dir=tmp_path, pred_path=tmp_path / "p.json", nb=3, nrn=0, lp=0.0,
                      skip_existing=True, dry_run=True, train_csv="train.csv")
    out = capsys.readouterr().out
    assert "[rescore]" in out and "METEOR" in out and "score_predictions.py" in out
