from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, ClapModel

from data.access_id import DATASETS, build_access_id_column
from model.device import best_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLAP text embeddings from train CSV.")
    parser.add_argument("--csv_path", type=str, default="data/train.csv")
    parser.add_argument("--output_path", type=str, default="text_train.npy")
    parser.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = build_access_id_column(df, args.dataset)

    if "caption" not in df.columns:
        raise ValueError(f"CSV must contain 'caption' column. Found: {set(df.columns)}")

    device = best_device()
    print(f"Using device: {device}")
    model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

    captions = df["caption"].astype(str).tolist()

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(captions), args.batch_size):
            batch = captions[start : start + args.batch_size]
            inputs = processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeds = model.get_text_features(**inputs)
            chunks.append(embeds.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(chunks, axis=0)
    if embeddings.shape[0] != len(captions):
        raise ValueError(
            f"Embedding row count mismatch: {embeddings.shape[0]} != {len(captions)}"
        )

    output_path = Path(args.output_path)
    np.save(output_path, embeddings)
    print(f"Saved text embeddings: {output_path} with shape {embeddings.shape}")


if __name__ == "__main__":
    main()
