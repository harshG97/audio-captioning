"""
build_expanded_split.py — Paper-split training data plus the rest of AudioCaps 2.0.

The paper split trains on 42,797 AudioCaps 2.0 train clips. This adds the
other 2.0 train clips, filtered, and writes two splits:

  <out_dir>/cap<C>/            paper train + extra clips, each caption kept at
                               most C times (counting the paper train copies)
  <out_dir>/cap<C>_agree<P>/   the same, minus the P% of extra clips whose
                               caption fits their audio worst (CLAP cosine
                               similarity between the clip and its caption)

val.csv and test.csv are copied unchanged from the paper split, so model
selection and test scores stay comparable. The paper train rows are kept as
they are; only extra clips are filtered. Extra clips are restricted to those
with CLAP audio embeddings (i.e. audio that preprocessing could read), since
training skips clips without features anyway.

Extra captions are written in the paper split's style: lowercase, single
spaces, ending with "." (the paper split's captions are the 2.0 captions in
that form).

The CLAP embeddings come from preprocessing on the full 2.0 splits
(scripts/submit.sh preprocess): <emb_dir>/text_train.npy (rows of
data/train.csv after the placeholder filter), audio_train.npy and
audio_train_ids.csv.

Usage:
    python data/build_expanded_split.py --emb_dir ~/scratch/recap/features \\
        --out_dir data/expanded --cap 3 --drop_frac 0.10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.access_id import build_access_id_column
from data.caption_filter import drop_placeholder_captions
from retrieval.build_retrieval_cache import caption_key

COLUMNS = ["youtube_id", "start_time", "caption"]


def read_captions(path: str) -> pd.DataFrame:
    """CSV with access ids, as the retrieval scripts read it (same row order)."""
    df = pd.read_csv(path, dtype={"youtube_id": str})
    return build_access_id_column(df, "audiocaps")


def paper_style(caption: str) -> str:
    """Lowercase, single spaces, ending with exactly one '.'."""
    text = re.sub(r"\s+", " ", str(caption)).strip().lower().rstrip(" .!?,;:")
    return text + "."


def agreement_scores(extra_csv: str, emb_dir: str) -> pd.Series:
    """Cosine similarity between each clip's CLAP audio embedding and the CLAP
    text embedding of its caption, indexed by access_id. Clips without an
    audio embedding are absent."""
    text_df = drop_placeholder_captions(read_captions(extra_csv), source=extra_csv)
    text = np.load(os.path.join(emb_dir, "text_train.npy")).astype(np.float32)
    if len(text) != len(text_df):
        raise SystemExit(f"{emb_dir}/text_train.npy has {len(text)} rows but {extra_csv} has "
                         f"{len(text_df)} captions: the embeddings were not built from this CSV")
    audio = np.load(os.path.join(emb_dir, "audio_train.npy")).astype(np.float32)
    audio_ids = pd.read_csv(os.path.join(emb_dir, "audio_train_ids.csv"))["access_id"].astype(str)
    if len(audio) != len(audio_ids):
        raise SystemExit(f"{emb_dir}: audio_train.npy and audio_train_ids.csv do not match")
    text /= np.linalg.norm(text, axis=1, keepdims=True).clip(1e-12)
    audio /= np.linalg.norm(audio, axis=1, keepdims=True).clip(1e-12)
    audio_row = {a: i for i, a in enumerate(audio_ids)}
    scores = {}
    for i, access_id in enumerate(text_df["access_id"].astype(str)):
        j = audio_row.get(access_id)
        if j is not None and access_id not in scores:
            scores[access_id] = float(text[i] @ audio[j])
    return pd.Series(scores, dtype=float)


def cap_captions(extra: pd.DataFrame, base_captions, cap: int, seed: int) -> pd.DataFrame:
    """Keep extra rows while their caption has fewer than `cap` copies so far,
    counting the base (paper) captions first. Extra rows are visited in a
    seeded random order, so which copies survive does not depend on file
    order; the result keeps the input order."""
    counts: dict[str, int] = {}
    for c in base_captions:
        if isinstance(c, str):                     # empty captions are NaN in pandas
            counts[caption_key(c)] = counts.get(caption_key(c), 0) + 1
    order = np.random.default_rng(seed).permutation(len(extra))
    keep = np.zeros(len(extra), dtype=bool)
    keys = extra["caption"].map(caption_key).to_numpy()
    for i in order:
        if counts.get(keys[i], 0) < cap:
            counts[keys[i]] = counts.get(keys[i], 0) + 1
            keep[i] = True
    return extra[keep]


def drop_worst(extra: pd.DataFrame, scores: pd.Series, frac: float) -> tuple[pd.DataFrame, float]:
    """Remove the `frac` of extra rows with the lowest scores (ties by id)."""
    n_drop = int(round(frac * len(extra)))
    ranked = sorted(extra["access_id"], key=lambda a: (scores[a], a))
    dropped = set(ranked[:n_drop])
    threshold = scores[ranked[n_drop - 1]] if n_drop else float("-inf")
    return extra[~extra["access_id"].isin(dropped)], threshold


def write_split(out: str, base_dir: str, paper: pd.DataFrame, extra: pd.DataFrame, info: dict) -> None:
    os.makedirs(out, exist_ok=True)
    train = pd.concat([paper[COLUMNS], extra[COLUMNS]], ignore_index=True)
    train.to_csv(os.path.join(out, "train.csv"), index=False)
    for split in ("val", "test"):
        shutil.copyfile(os.path.join(base_dir, f"{split}.csv"), os.path.join(out, f"{split}.csv"))
    info = dict(info, train_rows=len(train), paper_rows=len(paper), extra_rows=len(extra))
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"{out}: {len(paper)} paper + {len(extra)} extra = {len(train)} training clips")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base_dir", default="data/paper_split")
    p.add_argument("--extra_csv", default="data/train.csv")
    p.add_argument("--emb_dir", required=True, help="CLAP embeddings of --extra_csv (preprocess output)")
    p.add_argument("--out_dir", default="data/expanded")
    p.add_argument("--cap", type=int, default=3)
    p.add_argument("--drop_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cap < 1 or not 0 < args.drop_frac < 1:
        p.error("--cap must be >= 1 and --drop_frac between 0 and 1")

    paper = read_captions(os.path.join(args.base_dir, "train.csv"))
    held_out = set()
    for split in ("val", "test"):
        held_out |= set(read_captions(os.path.join(args.base_dir, f"{split}.csv"))["access_id"])

    extra = drop_placeholder_captions(read_captions(args.extra_csv), source=args.extra_csv)
    n_all = len(extra)
    extra = extra[~extra["access_id"].isin(set(paper["access_id"]) | held_out)]
    extra = extra.drop_duplicates(subset=["access_id"])
    n_new = len(extra)
    scores = agreement_scores(args.extra_csv, args.emb_dir)
    extra = extra[extra["access_id"].isin(scores.index)].copy()
    n_audio = len(extra)
    extra["start_time"] = extra["start_time"].astype(int)
    extra["caption"] = extra["caption"].map(paper_style)

    capped = cap_captions(extra, paper["caption"], args.cap, args.seed)
    filtered, threshold = drop_worst(capped, scores, args.drop_frac)
    assert not (set(capped["access_id"]) & held_out)

    info = dict(base_dir=args.base_dir, extra_csv=args.extra_csv, emb_dir=os.path.abspath(args.emb_dir),
                cap=args.cap, seed=args.seed, extra_candidates=n_all, extra_not_in_paper=n_new,
                extra_with_audio=n_audio, extra_after_cap=len(capped))
    write_split(os.path.join(args.out_dir, f"cap{args.cap}"), args.base_dir, paper, capped, info)
    pct = int(round(args.drop_frac * 100))
    write_split(os.path.join(args.out_dir, f"cap{args.cap}_agree{pct}"), args.base_dir, paper, filtered,
                dict(info, drop_frac=args.drop_frac, agreement_threshold=threshold))


if __name__ == "__main__":
    main()
