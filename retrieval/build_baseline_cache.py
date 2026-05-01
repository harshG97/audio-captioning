"""
build_baseline_cache.py

Write a "no-retrieval" cache: {access_id: []} for every row in a CSV. Use
this in place of a real retrieval cache to train and evaluate a baseline
without retrieval. The dataset class will see empty caption lists and
build the BASELINE_PROMPT ("This audio sounds like:") instead of the
RECAP prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.access_id import DATASETS, build_access_id_column


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit empty retrieval cache for the no-retrieval baseline.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = build_access_id_column(df, args.dataset)
    access_ids = df["access_id"].astype(str).drop_duplicates().tolist()

    cache: dict[str, list[str]] = {aid: [] for aid in access_ids}
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"Wrote {len(cache)} empty retrieval entries to {out}")


if __name__ == "__main__":
    main()
