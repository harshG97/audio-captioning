"""
precompute_audio_embeddings.py

Compute one CLAP retrieval-projection audio embedding per clip in a query CSV.
Outputs a .npy (shape: N x projection_dim) aligned to the input CSV row order
after deduplication on access_id, plus a sibling .csv listing access_ids in
the same order so downstream scripts can map row -> access_id.

These are the *projected* embeddings from CLAP's audio tower
(`get_audio_features`), used for retrieval. They are NOT the encoder hidden
states cached by `data/audio_preprocess.py` for cross-attention.

Audio files are loaded in a worker pool so I/O and resampling overlap with
GPU encoding.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, ClapModel

from data.access_id import DATASETS, build_access_id_column
from model.device import best_device

TARGET_SR = 48000


def _audio_path_for(row: pd.Series, dataset: str, audio_dir: Path) -> Path:
    if dataset == "audiocaps":
        return audio_dir / f"{row['youtube_id']}_{int(row['start_time'])}.wav"
    if dataset == "clotho":
        return audio_dir / str(row["file_name"])
    raise ValueError(f"Unknown dataset {dataset!r}")


def _load_audio(path: str) -> np.ndarray:
    """Worker function: load + resample + downmix one wav. Module-level so it pickles."""
    wave, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return wave.astype(np.float32)


def _encode_batch(model, processor, waves, device) -> np.ndarray:
    inputs = processor(
        audios=waves, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeds = model.get_audio_features(**inputs)
    return embeds.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLAP audio retrieval embeddings.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to write the audio embeddings .npy")
    parser.add_argument("--ids_path", type=str, required=True,
                        help="Path to write the aligned access_id CSV")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int,
                        default=min(8, (os.cpu_count() or 2)),
                        help="Worker processes for parallel audio loading.")
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-fused")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    audio_dir = Path(args.audio_dir)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not audio_dir.is_dir():
        raise FileNotFoundError(audio_dir)

    df = pd.read_csv(csv_path)
    if args.dataset == "audiocaps":
        df = df.dropna(subset=["youtube_id", "start_time"])
        df["start_time"] = df["start_time"].astype(int)
    df = build_access_id_column(df, args.dataset)
    df = df.drop_duplicates(subset=["access_id"]).reset_index(drop=True)

    paths = [_audio_path_for(r, args.dataset, audio_dir) for _, r in df.iterrows()]
    keep = [p.exists() for p in paths]
    if not all(keep):
        missing = sum(1 for k in keep if not k)
        print(f"WARNING: {missing}/{len(keep)} audio files missing, skipping.")
    df = df.loc[keep].reset_index(drop=True)
    paths_str = [str(p) for p, k in zip(paths, keep) if k]

    device = best_device()
    print(f"Using device: {device}")
    print(f"Loading audio with {args.num_workers} workers")
    model = ClapModel.from_pretrained(args.clap_model).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.clap_model)

    chunks: list[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as pool, torch.no_grad():
        for start in tqdm(range(0, len(paths_str), args.batch_size), desc="CLAP audio"):
            batch_paths = paths_str[start : start + args.batch_size]
            waves = list(pool.map(_load_audio, batch_paths))
            chunks.append(_encode_batch(model, processor, waves, device))

    embeddings = np.concatenate(chunks, axis=0)
    if embeddings.shape[0] != len(df):
        raise RuntimeError(
            f"Embedding row count {embeddings.shape[0]} != access_id count {len(df)}"
        )

    out_emb = Path(args.output_path)
    out_ids = Path(args.ids_path)
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, embeddings)
    df[["access_id"]].to_csv(out_ids, index=False)
    print(f"Saved {embeddings.shape} embeddings to {out_emb}")
    print(f"Saved {len(df)} access_ids to {out_ids}")


if __name__ == "__main__":
    main()
