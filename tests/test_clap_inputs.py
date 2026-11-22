"""Tests for data/clap_inputs.clap_inputs_per_clip.

Run: python -m pytest tests/test_clap_inputs.py -q  (CPU, no downloads)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import ClapFeatureExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.clap_inputs import clap_inputs_per_clip

SR = 48000


def _clips():
    rng = np.random.default_rng(0)
    return [(rng.standard_normal(int(SR * s)) * 0.1).astype(np.float32) for s in (4, 10, 7)]


def test_batched_extractor_marks_one_random_clip():
    """Documents the pitfall: a batched fusion call sets is_longer on one clip only."""
    out = ClapFeatureExtractor()(_clips(), sampling_rate=SR, return_tensors="pt")
    assert int(out["is_longer"].sum()) == 1


def test_per_clip_inputs_are_deterministic_and_batch_independent():
    fe = ClapFeatureExtractor()
    clips = _clips()
    np.random.seed(1)
    a = clap_inputs_per_clip(fe, clips, SR)
    np.random.seed(2)
    b = clap_inputs_per_clip(fe, clips[::-1], SR)
    assert a["is_longer"].flatten().tolist() == [True, True, True]
    for key in a:
        assert torch.equal(a[key], b[key].flip(0))

    single = fe([clips[0]], sampling_rate=SR, return_tensors="pt")
    for key in single:
        assert torch.equal(a[key][:1], single[key])
