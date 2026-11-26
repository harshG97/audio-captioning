"""Tests for data/build_expanded_split.py (paper train + extra AudioCaps 2.0 clips)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data.build_expanded_split as be
from retrieval.build_retrieval_cache import caption_key

PAPER_TRAIN = [("p0", 0, "a dog barks."), ("p1", 10, "a dog barks."), ("p2", 20, "a dog barks."),
               ("p3", 30, "rain falls."), ("p4", 40, "a man speaks.")]
VAL = [("v0", 0, "a cat meows.")]
TEST = [("t0", 0, "a car passes."), ("t1", 5, "birds chirp.")]
# Extra 2.0 rows: a paper clip, a val clip and a test clip (must be dropped),
# a placeholder, a clip without audio, repeated captions, odd formatting.
EXTRA = ([("p0", 0, "a dog barks"), ("v0", 0, "a cat meows"), ("t1", 5, "birds chirp"),
          ("x_ph", 0, "n/a"), ("x_noaudio", 0, "a bell rings")]
         + [(f"xd{i}", i, "A dog  barks") for i in range(3)]           # paper already has 3
         + [(f"xr{i}", i, "rain falls") for i in range(5)]             # paper has 1 -> 2 kept at cap 3
         + [(f"xs{i}", i, "a man speaks") for i in range(2)]           # paper has 1 -> both kept
         + [(f"xu{i}", i, f"unique sound number {i}!") for i in range(10)])


def _csv(path, rows, with_id=False):
    df = pd.DataFrame(rows, columns=["youtube_id", "start_time", "caption"])
    if with_id:
        df.insert(0, "audiocap_id", range(len(df)))
    df.to_csv(path, index=False)


@pytest.fixture
def setup(tmp_path):
    base = tmp_path / "paper_split"
    base.mkdir()
    _csv(base / "train.csv", PAPER_TRAIN)
    _csv(base / "val.csv", VAL)
    _csv(base / "test.csv", TEST)
    extra = tmp_path / "train_2.0.csv"
    _csv(extra, EXTRA, with_id=True)
    emb = tmp_path / "emb"
    emb.mkdir()
    # Text rows follow the extra CSV after the placeholder filter, like
    # precompute_text_embeddings.py; audio exists for all but x_noaudio.
    rows = [r for r in EXTRA if r[2] != "n/a"]
    rng = np.random.default_rng(0)
    text = rng.standard_normal((len(rows), 8)).astype(np.float32)
    ids = [f"{y}_{s}" for y, s, _ in rows]
    # Audio = text + a little noise, except xu0, whose audio fits its caption worst.
    noise = {"xu0_0": 10.0}
    audio, audio_ids = [], []
    for i, a in enumerate(ids):
        if a == "x_noaudio_0":
            continue
        audio.append(text[i] + noise.get(a, 0.1) * rng.standard_normal(8))
        audio_ids.append(a)
    np.save(emb / "text_train.npy", text)
    np.save(emb / "audio_train.npy", np.array(audio, dtype=np.float32))
    pd.DataFrame({"access_id": audio_ids}).to_csv(emb / "audio_train_ids.csv", index=False)
    return base, extra, emb, tmp_path / "out"


def _build(setup, *extra_args):
    base, extra, emb, out = setup
    be.main(["--base_dir", str(base), "--extra_csv", str(extra), "--emb_dir", str(emb),
             "--out_dir", str(out), "--cap", "3", "--drop_frac", "0.1", *extra_args])
    read = lambda d: pd.read_csv(out / d / "train.csv", dtype={"youtube_id": str})
    return read("cap3"), read("cap3_agree10"), out


def test_val_and_test_are_untouched_and_never_trained_on(setup):
    a, b, out = _build(setup)
    base = setup[0]
    for d in ("cap3", "cap3_agree10"):
        for split in ("val", "test"):
            assert (out / d / f"{split}.csv").read_bytes() == (base / f"{split}.csv").read_bytes()
    for train in (a, b):
        ids = set(train.youtube_id + "_" + train.start_time.astype(str))
        assert not ids & {"v0_0", "t0_0", "t1_5"}


def test_paper_rows_come_first_unchanged(setup):
    a, b, _ = _build(setup)
    paper = pd.read_csv(setup[0] / "train.csv", dtype={"youtube_id": str})
    for train in (a, b):
        pd.testing.assert_frame_equal(train.iloc[:len(paper)].reset_index(drop=True), paper)
        extra_ids = set(train.youtube_id.iloc[len(paper):])
        assert "p0" not in extra_ids                                # paper clip not added twice


def test_extra_rows_are_filtered_capped_and_formatted(setup):
    a, _, out = _build(setup)
    extra = a.iloc[len(PAPER_TRAIN):]
    assert "x_ph" not in set(extra.youtube_id) and "x_noaudio" not in set(extra.youtube_id)
    keys = extra.caption.map(caption_key)
    assert (keys == caption_key("a dog barks")).sum() == 0       # paper already holds 3
    assert (keys == caption_key("rain falls")).sum() == 2         # 1 in paper + 2 = cap
    assert (keys == caption_key("a man speaks")).sum() == 2
    assert all(c == c.lower() and c.endswith(".") and not c.endswith("..") and "  " not in c
               for c in extra.caption)
    assert "unique sound number 3." in set(extra.caption)          # "!" replaced by "."
    info = json.load(open(out / "cap3" / "info.json"))
    assert info["extra_rows"] == len(extra) == 14 and info["extra_with_audio"] == 20


def test_agreement_filter_drops_the_worst_fitting_extra_clips(setup):
    a, b, out = _build(setup)
    n_extra = len(a) - len(PAPER_TRAIN)
    dropped = set(a.youtube_id) - set(b.youtube_id)
    assert len(dropped) == round(0.1 * n_extra) == 1
    assert dropped == {"xu0"}                                      # noisiest audio
    assert set(b.youtube_id) <= set(a.youtube_id)
    assert json.load(open(out / "cap3_agree10" / "info.json"))["drop_frac"] == 0.1


def test_seed_makes_it_repeatable(setup):
    a1, _, _ = _build(setup)
    a2, _, _ = _build(setup)
    pd.testing.assert_frame_equal(a1, a2)


def test_cap_counts_paper_copies_first():
    extra = pd.DataFrame({"caption": ["x.", "x.", "x.", "y."]})
    kept = be.cap_captions(extra, ["X", "x", float("nan")], cap=3, seed=0)
    assert kept.caption.tolist().count("x.") == 1 and "y." in kept.caption.tolist()
    assert list(kept.index) == sorted(kept.index)                 # input order kept


def test_embeddings_must_match_the_csv(setup):
    base, extra, emb, out = setup
    np.save(emb / "text_train.npy", np.zeros((3, 8), np.float32))
    with pytest.raises(SystemExit, match="not built from this CSV"):
        _build(setup)
