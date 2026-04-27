"""
End-to-end retrieval: caption CSV + precomputed text .npy + query audio → top-k captions.

Text and query audio use the same CLAP checkpoint and ``get_*_features`` APIs, so
dimensions match for cosine retrieval.

Usage:
    python run_retrieval.py \\
        --csv ../data/audiocaps/train.clean.csv \\
        --text_npy text_train.npy \\
        --query_wav /path/to/query.wav \\
        --k 4 \\
        --strategy topk
"""

from __future__ import annotations

import argparse
import os

try:
    from .clap_embeddings import DEFAULT_CLAP_MODEL, encode_audio_file, load_clap_model
    from .datastore import Datastore
    from .retriever import MMRRetriever, TopKRetriever
except ImportError:
    from clap_embeddings import DEFAULT_CLAP_MODEL, encode_audio_file, load_clap_model
    from datastore import Datastore
    from retriever import MMRRetriever, TopKRetriever


def main() -> None:
    p = argparse.ArgumentParser(description="Audio-to-text-caption retrieval with CLAP")
    p.add_argument("--csv", type=str, required=True, help="Caption CSV (access_id, caption)")
    p.add_argument("--text_npy", type=str, required=True, help="Precomputed text embeddings .npy")
    p.add_argument("--query_wav", type=str, required=True, help="Query audio .wav path")
    p.add_argument("--model", type=str, default=DEFAULT_CLAP_MODEL, help="Same model as text .npy")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--strategy", type=str, choices=["topk", "mmr"], default="topk")
    p.add_argument("--lambda_", type=float, default=0.7, help="MMR λ (only for --strategy mmr)")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    ds = Datastore()
    ds.load_from_csv(args.csv)
    ds.load_embeddings_from_file(args.text_npy)

    model, processor, device = load_clap_model(args.model, device=args.device)
    q = encode_audio_file(args.query_wav, model, processor, device)

    if q.shape[0] != ds.embeddings_matrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query {q.shape} vs datastore {ds.embeddings_matrix.shape}. "
            "Regenerate text_npy with the same --model."
        )

    if args.strategy == "topk":
        retriever = TopKRetriever(ds, k=args.k)
    else:
        retriever = MMRRetriever(ds, k=args.k, lambda_=args.lambda_)

    results = retriever.retrieve(q)
    for i, cap in enumerate(results, 1):
        print(f"{i}. {cap}")


if __name__ == "__main__":
    main()
