from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .clap_embeddings import get_text_embeddings


def _build_access_id_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    if "access_id" in df.columns:
        return df

    required_cols = {"youtube_id", "start_time"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV missing required columns to build access_id: {required_cols}. "
            f"Found: {set(df.columns)}"
        )
    df = df.copy()
    df["access_id"] = df["youtube_id"].astype(str) + "_" + df["start_time"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLAP text embeddings from train CSV.")
    parser.add_argument("--csv_path", type=str, default="data/train.csv")
    parser.add_argument("--output_path", type=str, default="text_train.npy")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _build_access_id_if_missing(df)

    if "caption" not in df.columns:
        raise ValueError(f"CSV must contain 'caption' column. Found: {set(df.columns)}")

    captions = df["caption"].astype(str).tolist()
    embeddings = get_text_embeddings(captions)
    embeddings = embeddings.cpu().numpy().astype(np.float32)
    if embeddings.shape[0] != len(captions):
        raise ValueError(
            f"Embedding row count mismatch: {embeddings.shape[0]} != {len(captions)}"
        )

    output_path = Path(args.output_path)
    np.save(output_path, embeddings.astype(np.float32))
    print(f"Saved text embeddings: {output_path} with shape {embeddings.shape}")


if __name__ == "__main__":
    main()
