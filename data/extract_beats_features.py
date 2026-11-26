"""
extract_beats_features.py — Extract BEATs audio features and cache them to HDF5.

The BEATs counterpart of data/audio_preprocess.py (CLAP). It writes the same
layout, {output_dir}/{dataset}_{split}.hdf5 with one (frames, 768) float32
array per clip, keyed by the same clip ids, so training and evaluation read
it unchanged. Retrieval still uses CLAP: only the features the decoder
attends to change.

Audio is loaded at 16 kHz mono and made exactly 10 s long: longer clips are
cut, shorter ones are repeated (as CLAP's "repeatpad" does), so no clip has a
stretch of silence. BEATs turns 10 s into 62 time steps x 8 frequency bands
of patches, 496 vectors of 768. --pool N averages groups of N neighbouring
frequency bands within each time step: 4 (default) gives 62 x 2 = 124
vectors per clip, 1 keeps all 496.

The features are the last transformer layer's output. With a checkpoint
fine-tuned on AudioSet, --check N prints the top AudioSet labels for N clips
next to their captions and writes nothing: a quick way to confirm the
weights loaded correctly.

Usage:
    python data/extract_beats_features.py \
        --checkpoint ~/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
        --audio_dir ~/data/audiocaps_raw_audio --captions_csv data/paper_split/train.csv \
        --output_dir ~/scratch/recap/features_beats --dataset audiocaps --split train
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.access_id import build_access_id_column
from data.audio_preprocess import build_file_list
from model.beats import BEATs, BEATsConfig
from model.device import best_device

SAMPLE_RATE = 16000
CLIP_SECONDS = 10
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
LABELS_CSV = os.path.join(_REPO_ROOT, "data", "audioset_labels.csv")


def load_beats(checkpoint_path: str) -> tuple[BEATs, dict]:
    """Load a BEATs checkpoint ({"cfg", "model", optional "label_dict"}).

    weights_only=True: the file is unpickled without running any code it
    contains. Every weight must match the model, so a wrong or truncated
    file fails here instead of giving partly random features."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("cfg", "model"):
        if key not in ckpt:
            raise SystemExit(f"{checkpoint_path} is not a BEATs checkpoint (no '{key}')")
    model = BEATs(BEATsConfig(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt.get("label_dict") or {}


def fit_to_clip(waveform: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    """Cut to `length` samples, or repeat a shorter clip until it fills them."""
    if waveform.size == 0:
        raise ValueError("empty audio")
    if waveform.size >= length:
        return waveform[:length]
    reps = -(-length // waveform.size)
    return np.tile(waveform, reps)[:length]


def load_clip(file_path: str) -> np.ndarray:
    """16 kHz mono, exactly 10 s, float32 in [-1, 1]."""
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return fit_to_clip(waveform.astype(np.float32))


def pool_bands(hidden: torch.Tensor, n_bands: int, pool: int) -> torch.Tensor:
    """(B, T*F, D) patches in time-major order -> (B, T*F/pool, D), averaging
    each group of `pool` neighbouring frequency bands within a time step."""
    b, n, d = hidden.shape
    if n % n_bands:
        raise ValueError(f"{n} patches do not split into {n_bands} frequency bands")
    if n_bands % pool:
        raise ValueError(f"--pool must divide the {n_bands} frequency bands (got {pool})")
    t = n // n_bands
    return hidden.view(b, t, n_bands // pool, pool, d).mean(dim=3).reshape(b, t * (n_bands // pool), d)


class BEATsEncoder:
    """Runs BEATs and returns the last layer's output, optionally pooled.

    The output is taken from the transformer encoder with a forward hook, so
    the vendored BEATs code runs unchanged; for fine-tuned checkpoints the
    AudioSet label probabilities come back too."""

    def __init__(self, model: BEATs, device, pool: int = 4):
        self.model = model.to(device).eval()
        self.device = device
        self.pool = pool
        # 128 mel bins cut into square patches: 8 frequency bands for 16x16.
        self.n_bands = 128 // model.cfg.input_patch_size
        self._hidden = None
        model.encoder.register_forward_hook(self._keep_hidden)

    def _keep_hidden(self, module, inputs, output):
        self._hidden = output[0]

    @torch.no_grad()
    def __call__(self, waveforms: list) -> tuple[np.ndarray, torch.Tensor | None]:
        source = torch.from_numpy(np.stack(waveforms)).to(self.device)
        out, _ = self.model.extract_features(source)
        hidden = pool_bands(self._hidden, self.n_bands, self.pool)
        probs = out if self.model.predictor is not None else None
        return hidden.float().cpu().numpy(), probs


def encode_to_hdf5(file_list, encoder: BEATsEncoder, output_path: str,
                   batch_size: int = 32, num_workers: int = 0) -> int:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    seen = set()
    pool = ProcessPoolExecutor(max_workers=num_workers) if num_workers > 0 else None
    try:
        with h5py.File(output_path, "w") as h5f:
            for idx in tqdm(range(0, len(file_list), batch_size), desc="Extracting BEATs features"):
                batch = [item for item in file_list[idx:idx + batch_size] if item[0] not in seen]
                if not batch:
                    continue
                paths = [p for _, p in batch]
                waves = list(pool.map(load_clip, paths)) if pool else [load_clip(p) for p in paths]
                feats, _ = encoder(waves)
                for (audio_id, _), feat in zip(batch, feats):
                    h5f.create_dataset(str(audio_id), data=feat)
                    seen.add(audio_id)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    with h5py.File(output_path, "r") as h5f:
        first = next(iter(h5f.keys()))
        print(f"Saved {len(seen)} clips to {output_path}; '{first}' has shape {h5f[first].shape}")
    return len(seen)


def missing_against(file_list, captions_csv: str, dataset: str, reference_hdf5: str) -> list:
    """Clips of the CSV that the reference features (e.g. CLAP's) cover but
    whose audio was not found here: a comparison between the two would use
    different clips, since training skips clips without features."""
    ids = set(build_access_id_column(pd.read_csv(captions_csv), dataset)["access_id"].astype(str))
    with h5py.File(reference_hdf5, "r") as f:
        covered = ids & set(f.keys())
    return sorted(covered - {audio_id for audio_id, _ in file_list})


def label_names(path: str = LABELS_CSV) -> dict:
    """AudioSet label id (e.g. /m/09x0r) -> display name."""
    with open(path, newline="") as f:
        return {row["mid"]: row["display_name"] for row in csv.DictReader(f)}


def check(file_list, encoder: BEATsEncoder, label_dict: dict, captions_csv: str,
          dataset: str, n: int) -> None:
    """Print the top AudioSet labels for n clips beside one of their captions."""
    if encoder.model.predictor is None or not label_dict:
        print("--check skipped: the checkpoint has no AudioSet classifier (not fine-tuned)")
        return
    names = label_names()
    df = build_access_id_column(pd.read_csv(captions_csv), dataset)
    caption = dict(zip(df["access_id"], df["caption"]))
    batch = file_list[:n]
    _, probs = encoder([load_clip(p) for _, p in batch])
    for (audio_id, _), p in zip(batch, probs):
        top = torch.topk(p, 3)
        labels = ", ".join(f"{names.get(label_dict[int(i)], label_dict[int(i)])} {v:.2f}"
                           for v, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"{audio_id}\n  caption: {caption.get(audio_id, '?')}\n  BEATs:   {labels}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract BEATs features to HDF5")
    p.add_argument("--checkpoint", required=True, help="BEATs .pt checkpoint")
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--captions_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["clotho", "audiocaps"])
    p.add_argument("--split", choices=["train", "val", "test", "tmp"])
    p.add_argument("--pool", type=int, default=4, choices=[1, 2, 4, 8],
                   help="Average this many neighbouring frequency bands per time step: "
                        "4 gives 124 vectors per 10 s clip, 1 keeps all 496 (default 4).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0, help="Processes for audio loading")
    p.add_argument("--match_hdf5", default=None,
                   help="Feature file to match (e.g. the CLAP features of this split): stop "
                        "if any clip it has for this CSV has no audio here.")
    p.add_argument("--check", type=int, default=0, metavar="N",
                   help="Print top AudioSet labels for N clips and exit without writing.")
    args = p.parse_args(argv)
    if not args.check and not args.split:
        p.error("--split is required unless --check is given")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    device = best_device()
    print(f"Using device: {device}")
    print(f"Loading BEATs: {args.checkpoint}")
    model, label_dict = load_beats(args.checkpoint)
    encoder = BEATsEncoder(model, device, pool=args.pool)
    file_list = build_file_list(args.audio_dir, args.captions_csv, args.dataset)
    if not file_list:
        raise SystemExit("no audio files found")
    if args.match_hdf5:
        missing = missing_against(file_list, args.captions_csv, args.dataset, args.match_hdf5)
        if missing:
            raise SystemExit(f"{len(missing)} clips in {args.match_hdf5} have no audio in "
                             f"{args.audio_dir} (e.g. {missing[:3]}); is the audio fully synced?")
    if args.check:
        check(file_list, encoder, label_dict, args.captions_csv, args.dataset, args.check)
        return
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.split}.hdf5")
    encode_to_hdf5(file_list, encoder, output_path,
                   batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
