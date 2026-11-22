"""
copy_retrieval_baseline.py — "Copy the retrieved caption" baseline.

For every clip, predict the rank-r caption from the retrieval cache (rank 1 =
nearest caption in CLAP space) and write a predictions JSON in the same
format as evaluate.py, so it can be scored with score_predictions.py.

A RECAP model that does not beat this baseline is not adding anything over
retrieval itself.

Usage:
    python scripts/copy_retrieval_baseline.py \
        --csv_path data/test.csv \
        --retrieval_cache $FEAT_DIR/retrieved_top4_test.json \
        --hdf5_path $FEAT_DIR/audiocaps_test.hdf5 \
        --output_path $RUNS_DIR/copy-top1/preds_test.json

    python score_predictions.py \
        --predictions $RUNS_DIR/copy-top1/preds_test.json \
        --output $RUNS_DIR/copy-top1/metrics_test.json \
        --train_csv data/train.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Make repo modules importable when invoked as `python scripts/copy_retrieval_baseline.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.access_id import DATASETS, build_access_id_column


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a copy-the-retrieved-caption baseline.")
    p.add_argument("--csv_path", type=str, required=True, help="Eval split CSV (references).")
    p.add_argument("--retrieval_cache", type=str, required=True)
    p.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--rank", type=int, default=1,
                   help="Which retrieved caption to copy (1 = most similar).")
    p.add_argument("--hdf5_path", type=str, default=None,
                   help="Optional: keep only clips that have cached audio features, "
                        "so the clip set matches evaluate.py exactly.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.rank < 1:
        raise SystemExit("--rank must be >= 1")

    df = build_access_id_column(pd.read_csv(args.csv_path), args.dataset)
    cache = json.loads(Path(args.retrieval_cache).read_text())

    references_by_id: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        references_by_id.setdefault(str(row["access_id"]), []).append(str(row["caption"]))

    available = None
    if args.hdf5_path:
        import h5py
        with h5py.File(args.hdf5_path, "r") as f:
            available = set(f.keys())

    predictions: dict[str, dict] = {}
    too_short = 0
    for access_id, refs in references_by_id.items():
        if access_id not in cache or (available is not None and access_id not in available):
            continue
        retrieved = cache[access_id]
        if len(retrieved) < args.rank:
            too_short += 1
            continue
        caption = str(retrieved[args.rank - 1]).strip().lower()
        predictions[access_id] = {
            "references": refs,
            "prediction": caption,
            "raw_prediction": caption,
        }

    if not predictions:
        raise SystemExit("No clips had a retrieved caption at the requested rank.")
    if too_short:
        print(f"[copy_baseline] Skipped {too_short} clips with fewer than {args.rank} retrieved captions.")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"Wrote {len(predictions)} copy-rank-{args.rank} predictions to {out}")


if __name__ == "__main__":
    main()
