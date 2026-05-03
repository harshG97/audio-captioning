"""
check_retrieval_leakage.py — Quantify how often a clip's GT caption (or a
near-duplicate) appears among its top-k retrieved captions.

Usage:
    python scripts/check_retrieval_leakage.py \
        --csv_path data/train.csv \
        --retrieval_cache $FEAT_DIR/retrieved_topk_train.json \
        --dataset audiocaps \
        --num_examples 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Make repo modules importable when invoked as `python scripts/check_retrieval_leakage.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from data.access_id import DATASETS, build_access_id_column


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_normalize(a).split()), set(_normalize(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--retrieval_cache", type=str, required=True)
    p.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    p.add_argument("--num_examples", type=int, default=10,
                   help="Print this many random clips with GT + retrieved.")
    p.add_argument("--jaccard_threshold", type=float, default=0.7,
                   help="A retrieved caption counts as a near-duplicate if its "
                        "word-Jaccard with the GT is at or above this value.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    df = build_access_id_column(df, args.dataset)
    if "caption" not in df.columns:
        raise ValueError("CSV must contain a 'caption' column.")

    cache: dict[str, list[str]] = json.loads(Path(args.retrieval_cache).read_text())

    # Group GTs by access_id (train usually 1 caption per id; val/test up to 5).
    gt_by_id: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        gt_by_id.setdefault(str(row["access_id"]), []).append(str(row["caption"]))

    eval_ids = [aid for aid in gt_by_id if aid in cache]
    print(f"clips evaluated         : {len(eval_ids)} "
          f"(of {len(gt_by_id)} unique access_ids in CSV)")

    exact_hits = 0          # any GT appears verbatim among top-k
    norm_hits = 0           # any GT appears among top-k after normalization
    near_hits = 0           # any retrieved caption has Jaccard >= threshold to any GT
    rank_buckets = Counter()  # rank (1..k) at which the first exact match occurs
    k_seen = Counter()      # distribution of k across the cache

    for aid in eval_ids:
        gts = gt_by_id[aid]
        retrieved = cache[aid]
        k_seen[len(retrieved)] += 1

        gt_set = set(gts)
        norm_gt_set = {_normalize(g) for g in gts}

        first_exact_rank = None
        for rank, r in enumerate(retrieved, 1):
            if r in gt_set:
                first_exact_rank = rank
                break
        if first_exact_rank is not None:
            exact_hits += 1
            rank_buckets[first_exact_rank] += 1

        if any(_normalize(r) in norm_gt_set for r in retrieved):
            norm_hits += 1

        if any(_jaccard(r, g) >= args.jaccard_threshold for r in retrieved for g in gts):
            near_hits += 1

    n = len(eval_ids)
    print()
    print(f"k distribution           : {dict(k_seen)}")
    print(f"exact-match (any rank)   : {exact_hits} / {n}  ({100 * exact_hits / n:.1f}%)")
    print(f"normalized-match         : {norm_hits} / {n}  ({100 * norm_hits / n:.1f}%)")
    print(f"near-dup (Jaccard >= {args.jaccard_threshold:.2f}): {near_hits} / {n}  ({100 * near_hits / n:.1f}%)")
    print()
    if rank_buckets:
        print("first-exact-match rank distribution:")
        for r in sorted(rank_buckets):
            print(f"  rank {r}: {rank_buckets[r]}")

    # Print random examples
    print()
    print(f"=== {args.num_examples} random examples ===")
    sample = pd.Series(eval_ids).sample(
        n=min(args.num_examples, len(eval_ids)),
        random_state=args.seed,
    ).tolist()
    for aid in sample:
        gts = gt_by_id[aid]
        retrieved = cache[aid]
        print(f"\n--- {aid} ---")
        for g in gts:
            print(f"  GT       : {g}")
        for i, r in enumerate(retrieved, 1):
            tag = ""
            if r in set(gts):
                tag = "  [EXACT GT]"
            elif _normalize(r) in {_normalize(g) for g in gts}:
                tag = "  [NORM GT]"
            else:
                jmax = max((_jaccard(r, g) for g in gts), default=0.0)
                if jmax >= args.jaccard_threshold:
                    tag = f"  [NEAR GT j={jmax:.2f}]"
            print(f"  top-{i}    : {r}{tag}")


if __name__ == "__main__":
    main()
