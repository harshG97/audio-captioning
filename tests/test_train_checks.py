"""Tests for train.py's check that feature width matches the decoder width.

Run: python -m pytest tests/test_train_checks.py -q
"""
import sys
from pathlib import Path

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
