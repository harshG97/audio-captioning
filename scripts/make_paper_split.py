"""
make_paper_split.py — Build the AudioCaps splits used by the RECAP paper.

The RECAP reference code (github.com/Sreyan88/RECAP, data/*_audiocaps.csv)
trains and evaluates on a subset of AudioCaps: 42,797 train clips, 345 val
clips and 748 test clips. All of them are in this repo's data/{train,val,test}.csv,
which are larger (91k train clips, 975 test clips). Its captions are the
cleaned versions (typos and spacing fixed), so this script keeps the
reference caption text and only maps each clip onto this repo's
youtube_id/start_time keys, giving CSVs that work with every other script.

Usage:
    git clone https://github.com/Sreyan88/RECAP /tmp/RECAP
    python scripts/make_paper_split.py --reference_data_dir /tmp/RECAP/data

Writes data/paper_split/{train,val,test}.csv with columns
youtube_id, start_time, caption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS = {"train": "train_audiocaps.csv", "val": "dev_audiocaps.csv", "test": "test_audiocaps.csv"}


def build_split(ours: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    ours = ours.dropna(subset=["youtube_id", "start_time"])
    starts = ours.groupby("youtube_id")["start_time"].unique()
    if (starts.map(len) > 1).any():
        raise SystemExit("A youtube_id maps to several start times; cannot map reference clips.")
    start_of = starts.map(lambda s: int(s[0]))

    ref = reference.copy()
    ref["youtube_id"] = ref["file_name"].astype(str).str.replace(r"\.wav$", "", regex=True)
    missing = sorted(set(ref["youtube_id"]) - set(start_of.index))
    if missing:
        raise SystemExit(f"{len(missing)} reference clips are not in this repo's CSV, e.g. {missing[:5]}")
    ref["start_time"] = ref["youtube_id"].map(start_of).astype(int)
    return ref[["youtube_id", "start_time", "caption"]].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the RECAP paper's AudioCaps splits.")
    p.add_argument("--reference_data_dir", type=str, required=True,
                   help="data/ folder of a clone of github.com/Sreyan88/RECAP")
    p.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data"))
    p.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "data" / "paper_split"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ref_name in SPLITS.items():
        ours = pd.read_csv(Path(args.data_dir) / f"{split}.csv")
        reference = pd.read_csv(Path(args.reference_data_dir) / ref_name)
        df = build_split(ours, reference)
        df.to_csv(out_dir / f"{split}.csv", index=False)
        print(f"{split}: {df['youtube_id'].nunique()} clips, {len(df)} captions -> {out_dir / f'{split}.csv'}")


if __name__ == "__main__":
    main()
