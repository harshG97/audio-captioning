"""Tests for data/caption_filter.py.

Run: python -m pytest tests/test_caption_filter.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.caption_filter import drop_placeholder_captions, is_placeholder_caption

# Every placeholder string found in data/train.csv.
PLACEHOLDERS = [
    "video unvailable", "video uanvailable", "video unavailabel", "video unavailabel.",
    "not available", "no sounds available", "video is unplayable", "video will not play",
    "video does not play", "video not available", "no audio/video available",
    "no audio no video", "no audio broken video", "no audio during video playback",
    "", "   ", None, np.nan, "nan", "Video Unavailable",
]

# Real descriptions that must be kept, including ones that mention a video.
REAL = [
    "silence occurs through the entire video",
    "silence occurs throughout the entire video",
    "silence, almost no sound",
    "gunshots occur in a video game",
    "someone is playing a video game",
    "video game punching swooshing and thuds",
    "women speaking and narrating a video",
    "no sound followed by two clicks and then a bird chirping",
    "a man speaks while a video plays in the background and the audio is not available later",
    "toilet flushing",
    "a woman talks nearby as water pours.",
]


@pytest.mark.parametrize("caption", PLACEHOLDERS)
def test_placeholders_detected(caption):
    assert is_placeholder_caption(caption)


@pytest.mark.parametrize("caption", REAL)
def test_real_captions_kept(caption):
    assert not is_placeholder_caption(caption)


def test_shipped_csvs():
    """train.csv has 23 placeholder captions + 3 empty rows; val/test have none."""
    train = pd.read_csv(REPO_ROOT / "data" / "train.csv")
    assert len(train) - len(drop_placeholder_captions(train)) == 26
    for split in ("val", "test"):
        df = pd.read_csv(REPO_ROOT / "data" / f"{split}.csv")
        assert len(drop_placeholder_captions(df)) == len(df)


def test_drop_resets_index():
    df = pd.DataFrame({"caption": ["a dog barks", "video unvailable", "rain falls"]})
    out = drop_placeholder_captions(df)
    assert out["caption"].tolist() == ["a dog barks", "rain falls"]
    assert out.index.tolist() == [0, 1]
