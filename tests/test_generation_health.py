"""Tests for score_predictions.generation_health.

Run: python -m pytest tests/test_generation_health.py -q
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score_predictions import generation_health


def _pred(text, stop="eos", cut=False):
    return {"references": ["x"], "prediction": text, "raw_prediction": text,
            "stop_reason": stop, "cut_by_postprocessing": cut}


def test_rates():
    preds = {
        "a": _pred("a dog barks"),
        "b": _pred("a dog barks", stop="max_new_tokens", cut=True),
        "c": _pred("rain falls on a tin roof."),
        "d": _pred("", stop="max_new_tokens", cut=True),
    }
    h = generation_health(preds, train_captions=["A dog barks", "Rain falls on a tin roof"])
    assert h["unique_rate"] == pytest.approx(3 / 4)
    assert h["empty_rate"] == pytest.approx(1 / 4)
    assert h["eos_stop_rate"] == pytest.approx(2 / 4)
    assert h["max_tokens_rate"] == pytest.approx(2 / 4)
    assert h["postprocess_cut_rate"] == pytest.approx(2 / 4)
    # Matching ignores case, spacing and trailing punctuation.
    assert h["train_copy_rate"] == pytest.approx(3 / 4)
    assert h["mean_words"] == pytest.approx((3 + 3 + 6 + 0) / 4)


def test_old_prediction_files_without_diagnostics():
    preds = {"a": {"references": ["x"], "prediction": "a dog barks"}}
    h = generation_health(preds)
    assert set(h) == {"unique_rate", "empty_rate", "mean_words"}
