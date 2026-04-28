from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .clap_embeddings import get_audio_embedding


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


def _cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ query) / (matrix_norm * query_norm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLAP retrieval with top-k captions.")
    parser.add_argument("--wav_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default="data/train.csv")
    parser.add_argument("--embeddings_path", type=str, default="text_train.npy")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--strategy", type=str, default="topk", choices=["topk"])
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be >= 1")

    csv_path = Path(args.csv_path)
    emb_path = Path(args.embeddings_path)
    wav_path = Path(args.wav_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    df = pd.read_csv(csv_path)
    df = _build_access_id_if_missing(df)
    if "caption" not in df.columns:
        raise ValueError(f"CSV must contain 'caption' column. Found: {set(df.columns)}")

    text_embeddings = np.load(emb_path).astype(np.float32)
    captions = df["caption"].astype(str).tolist()
    if text_embeddings.shape[0] != len(captions):
        raise ValueError(
            f"Embedding rows ({text_embeddings.shape[0]}) must match CSV rows ({len(captions)})"
        )

    query_embedding = get_audio_embedding(str(wav_path)).astype(np.float32)
    if text_embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            "Embedding dimensions mismatch: "
            f"text d={text_embeddings.shape[1]} vs query d={query_embedding.shape[0]}"
        )

    similarities = _cosine_similarity_matrix(query_embedding, text_embeddings)
    top_k = min(args.k, len(captions))
    top_indices = np.argsort(-similarities)[:top_k]

    print(f"Top-{top_k} captions (strategy={args.strategy}):")
    for rank, idx in enumerate(top_indices, start=1):
        print(f"{rank}. score={similarities[idx]:.6f} | caption={captions[idx]}")


if __name__ == "__main__":
    main()
