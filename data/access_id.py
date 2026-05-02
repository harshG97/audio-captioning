"""
access_id.py

Dataset-agnostic helpers for the canonical `access_id` used to key audio
features (HDF5), captions, and retrieval-cache entries.

AudioCaps:  access_id = f"{youtube_id}_{start_time}"
Clotho:     access_id = filename without extension
"""

from __future__ import annotations

import os
import pandas as pd

DATASETS = ("audiocaps", "clotho")


def build_access_id_column(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return a copy of `df` with an `access_id` column added when missing."""
    if "access_id" in df.columns:
        return df

    df = df.copy()
    if dataset == "audiocaps":
        required = {"youtube_id", "start_time"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"AudioCaps CSV missing columns {required}. Found: {set(df.columns)}"
            )
        before = len(df)
        df = df.dropna(subset=["youtube_id", "start_time"])
        if len(df) < before:
            print(f"[access_id] Dropped {before - len(df)} rows with missing youtube_id/start_time")
        df["access_id"] = (
            df["youtube_id"].astype(str)
            + "_"
            + df["start_time"].astype(int).astype(str)
        )
    elif dataset == "clotho":
        required = {"file_name"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Clotho CSV missing columns {required}. Found: {set(df.columns)}"
            )
        df["access_id"] = df["file_name"].astype(str).map(
            lambda fn: os.path.splitext(fn)[0]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use one of {DATASETS}.")
    return df
