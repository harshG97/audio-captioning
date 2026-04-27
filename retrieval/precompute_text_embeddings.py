"""
Precompute CLAP text embeddings for a caption CSV and save as .npy (N, d).

Aligns row order with Datastore.load_from_csv so the matrix can be loaded with
Datastore.load_embeddings_from_file.

Usage:
    python precompute_text_embeddings.py \\
        --csv ../data/audiocaps/train.clean.csv \\
        --output text_train.npy \\
        --model laion/clap-htsat-fused \\
        --batch_size 32
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from .clap_embeddings import (
        DEFAULT_CLAP_MODEL,
        encode_texts,
        get_embedding_dim,
        load_clap_model,
    )
    from .datastore import Datastore
except ImportError:
    from clap_embeddings import (
        DEFAULT_CLAP_MODEL,
        encode_texts,
        get_embedding_dim,
        load_clap_model,
    )
    from datastore import Datastore


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLAP text embeddings to .npy")
    parser.add_argument("--csv", type=str, required=True, help="Caption CSV with access_id, caption")
    parser.add_argument("--output", type=str, required=True, help="Output .npy path (N, d)")
    parser.add_argument("--model", type=str, default=DEFAULT_CLAP_MODEL, help="HF CLAP checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cuda:0, cpu")
    args = parser.parse_args()

    ds = Datastore()
    ds.load_from_csv(args.csv)
    captions = ds.captions

    model, processor, device = load_clap_model(args.model, device=args.device)
    d = get_embedding_dim(model)
    print(f"[precompute_text_embeddings] projection_dim={d}")

    embeddings = encode_texts(
        captions,
        model,
        processor,
        device,
        batch_size=args.batch_size,
    )
    assert embeddings.shape == (len(captions), d), (
        f"Expected ({len(captions)}, {d}), got {embeddings.shape}"
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.save(args.output, embeddings)
    print(f"[precompute_text_embeddings] Saved {embeddings.shape} to {args.output}")


if __name__ == "__main__":
    main()
