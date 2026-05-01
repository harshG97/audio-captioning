"""
build_retrieval_cache.py

For each query clip, find top-k captions in a datastore by cosine similarity
in CLAP space. Writes {access_id: [caption, ...]} JSON.

Inputs (paired):
  Query    : audio embeddings .npy  + access-id CSV (output of
             precompute_audio_embeddings.py).
  Datastore: text embeddings .npy   + caption CSV with access_id + caption
             (output of precompute_text_embeddings.py + the train CSV).

Self-exclusion: if a query access_id appears in the datastore (in-domain
training case), its row is masked before top-k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.access_id import DATASETS, build_access_id_column


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def _mmr_select(
    relevance: np.ndarray,
    datastore_norm: np.ndarray,
    k: int,
    lambda_: float,
) -> list[int]:
    selected: list[int] = []
    candidates = np.arange(datastore_norm.shape[0])
    candidate_mask = np.ones_like(candidates, dtype=bool)
    candidate_mask &= np.isfinite(relevance)

    for _ in range(min(k, candidate_mask.sum())):
        if not selected:
            best = int(np.argmax(np.where(candidate_mask, relevance, -np.inf)))
        else:
            sel_mat = datastore_norm[selected]                       # (s, d)
            redundancy = datastore_norm @ sel_mat.T                  # (N, s)
            max_red = redundancy.max(axis=1)
            mmr = lambda_ * relevance - (1.0 - lambda_) * max_red
            mmr = np.where(candidate_mask, mmr, -np.inf)
            best = int(np.argmax(mmr))
        selected.append(best)
        candidate_mask[best] = False
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval cache (top-k captions per query).")
    parser.add_argument("--query_emb_path", type=str, required=True)
    parser.add_argument("--query_ids_path", type=str, required=True,
                        help="CSV with an access_id column aligned to query_emb_path rows.")
    parser.add_argument("--datastore_emb_path", type=str, required=True)
    parser.add_argument("--datastore_csv_path", type=str, required=True,
                        help="CSV with caption + youtube_id/start_time or file_name "
                             "(access_id will be derived).")
    parser.add_argument("--datastore_dataset", type=str, default="audiocaps", choices=DATASETS)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--strategy", type=str, default="topk", choices=["topk", "mmr"])
    parser.add_argument("--mmr_lambda", type=float, default=0.7)
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1")

    query_emb = np.load(args.query_emb_path).astype(np.float32)
    query_ids = pd.read_csv(args.query_ids_path)["access_id"].astype(str).tolist()
    if query_emb.shape[0] != len(query_ids):
        raise ValueError(
            f"query embeddings {query_emb.shape[0]} != access_ids {len(query_ids)}"
        )

    ds_emb = np.load(args.datastore_emb_path).astype(np.float32)
    ds_df = pd.read_csv(args.datastore_csv_path)
    ds_df = build_access_id_column(ds_df, args.datastore_dataset)
    if "caption" not in ds_df.columns:
        raise ValueError("Datastore CSV must contain a 'caption' column.")
    if ds_emb.shape[0] != len(ds_df):
        raise ValueError(
            f"Datastore embeddings {ds_emb.shape[0]} != caption rows {len(ds_df)}"
        )
    if ds_emb.shape[1] != query_emb.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: query {query_emb.shape[1]} vs ds {ds_emb.shape[1]}"
        )

    ds_access_ids = ds_df["access_id"].astype(str).to_numpy()
    ds_captions = ds_df["caption"].astype(str).tolist()

    query_norm = _l2_normalize(query_emb)
    ds_norm = _l2_normalize(ds_emb)

    cache: dict[str, list[str]] = {}
    for i, q_id in enumerate(query_ids):
        sims = ds_norm @ query_norm[i]                              # (N,)
        sims = np.where(ds_access_ids == q_id, -np.inf, sims)

        if args.strategy == "topk":
            top = np.argpartition(-sims, kth=min(args.k, len(sims) - 1))[: args.k]
            top = top[np.argsort(-sims[top])]
            picks = top.tolist()
        else:
            picks = _mmr_select(sims, ds_norm, args.k, args.mmr_lambda)

        cache[q_id] = [ds_captions[j] for j in picks]

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"Wrote {len(cache)} retrieval entries to {out}")


if __name__ == "__main__":
    main()
