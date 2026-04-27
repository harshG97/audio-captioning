"""
audio_preprocess.py — Extract CLAP audio features and cache to HDF5.

Two formats (``--feature_format``):

- **joint** (default): ``ClapModel.get_audio_features`` → one vector per clip
  shape **(projection_dim,)**, typically **512**, aligned with CLAP **text**
  embeddings for retrieval. Output file name: ``{dataset}_{split}_joint.hdf5``.

- **sequence**: ``ClapAudioModel`` last hidden state reshaped to **(64, 768)** for
  downstream sequence models — **not** comparable to text embeddings.

Run ONCE before training / retrieval; cached outputs are deterministic.

Usage:
    # Retrieval-aligned joint embeddings (same space as CLAP text features):
    python audio_preprocess.py \\
        --audio_dir ... --captions_csv ... --output_dir ... \\
        --dataset audiocaps --split train --feature_format joint

    # Legacy sequence features for models that consume (64, 768):
    python audio_preprocess.py ... --feature_format sequence
"""

import argparse
import os
import h5py
import torch
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, ClapAudioModel, ClapModel
 
 
def load_audio(file_path, target_sr=48000):
    """
    Load an audio file and resample to the target sample rate.
    CLAP (htsat-fused) expects 48kHz audio.
 
    Returns:
        waveform: np.ndarray of shape (num_samples,)
        sr: int, the sample rate (will be target_sr)
    """
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return waveform, sr
 
 
def build_file_list(audio_dir, captions_csv, dataset_type):
    """
    Read the captions CSV and build a list of (audio_id, file_path) tuples.
    Also verifies that each referenced audio file actually exists on disk.
 
    Returns:
        list of (audio_id: str, file_path: str) tuples
    """
    file_list = []
    missing = []
 
    if dataset_type == "clotho":
        df = pd.read_csv(captions_csv)
        for _, row in df.iterrows():
            file_name = row["file_name"]
            # Use filename without extension as audio_id
            audio_id = os.path.splitext(file_name)[0]
            file_path = os.path.join(audio_dir, file_name)
 
            if os.path.exists(file_path):
                file_list.append((audio_id, file_path))
            else:
                missing.append(file_name)
 
    elif dataset_type == "audiocaps":
        # AudioCaps naming pattern: youtube_id+start_time.wav
        df = pd.read_csv(captions_csv)
        df = df.dropna(subset=["youtube_id", "start_time"])
        df["start_time"] = df["start_time"].astype(int)
        unique_ids = df.drop_duplicates(subset=["youtube_id", "start_time"])
        for _, row in unique_ids.iterrows():
            youtube_id = row["youtube_id"]
            start_time = row["start_time"]
            audio_id = f"{youtube_id}_{start_time}"
            file_path = os.path.join(audio_dir, f"{audio_id}.wav")
 
            if os.path.exists(file_path):
                file_list.append((audio_id, file_path))
            else:
                missing.append(audio_id)
 
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
 
    if missing:
        print(f"WARNING: {len(missing)} audio files not found on disk.")
        print(f"  First few missing: {missing[:5]}")
 
    print(f"Found {len(file_list)} audio files to process.")
    return file_list
 
 
def encode_sequence_to_hdf5(
    file_list, model: ClapAudioModel, processor, device, output_path, batch_size=1
):
    """
    Extract sequence features (64, 768) per clip — not aligned with CLAP text space.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    seen_ids = set()

    with h5py.File(output_path, "w") as h5f:
        for idx in tqdm(range(0, len(file_list), batch_size), desc="Extracting sequence features"):
            batch = file_list[idx:idx + batch_size]

            audio_ids = [item[0] for item in batch]
            file_paths = [item[1] for item in batch]

            waveforms = [load_audio(fp)[0] for fp in file_paths]

            inputs = processor(
                audios=waveforms,
                sampling_rate=48000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            encodings = torch.flatten(outputs.last_hidden_state, 2)
            encodings = encodings.permute(0, 2, 1).detach().cpu().numpy()

            for audio_id, encoding in zip(audio_ids, encodings):
                if audio_id not in seen_ids:
                    h5f.create_dataset(str(audio_id), data=encoding)
                    seen_ids.add(audio_id)

    print(f"Saved {len(seen_ids)} entries to {output_path}")

    with h5py.File(output_path, "r") as h5f:
        sample_id = list(h5f.keys())[0]
        sample_shape = h5f[sample_id][()].shape
        print(f"Sample entry '{sample_id}': shape = {sample_shape}")
        print(f"  Expected (sequence mode): (64, 768)")


def encode_joint_to_hdf5(
    file_list, model: ClapModel, processor, device, output_path, batch_size=8
):
    """
    Extract joint embedding (projection_dim,) per clip via ``get_audio_features`` —
    same space as ``ClapModel.get_text_features`` for retrieval.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    seen_ids = set()
    proj_dim = model.config.projection_dim

    with h5py.File(output_path, "w") as h5f:
        for idx in tqdm(range(0, len(file_list), batch_size), desc="Extracting joint embeddings"):
            batch = file_list[idx:idx + batch_size]

            audio_ids = [item[0] for item in batch]
            file_paths = [item[1] for item in batch]

            waveforms = [load_audio(fp)[0] for fp in file_paths]

            inputs = processor(
                audios=waveforms,
                sampling_rate=48000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                feats = model.get_audio_features(**inputs)

            feats_np = feats.detach().cpu().numpy().astype(np.float32)

            for audio_id, row in zip(audio_ids, feats_np):
                if audio_id not in seen_ids:
                    h5f.create_dataset(str(audio_id), data=row)
                    seen_ids.add(audio_id)

    print(f"Saved {len(seen_ids)} entries to {output_path}")

    with h5py.File(output_path, "r") as h5f:
        sample_id = list(h5f.keys())[0]
        sample_shape = h5f[sample_id][()].shape
        print(f"Sample entry '{sample_id}': shape = {sample_shape}")
        print(f"  Expected (joint mode): ({proj_dim},)")
 
 
def main():
    parser = argparse.ArgumentParser(description="Extract CLAP features to HDF5")
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing audio .wav files")
    parser.add_argument("--captions_csv", type=str, required=True,
                        help="Path to captions CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output HDF5 file")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["clotho", "audiocaps"],
                        help="Which dataset format to expect")
    parser.add_argument("--encoder_name", type=str,
                        default="laion/clap-htsat-fused",
                        help="HuggingFace CLAP model name")
    parser.add_argument("--split", type=str, required=True,
                    choices=["train", "val", "test", "tmp"],
                    help="Which split to process")
    parser.add_argument("--batch_size", type=int, default=1,
                    help="Number of audio files to process at once")
    parser.add_argument(
        "--feature_format",
        type=str,
        choices=["joint", "sequence"],
        default="joint",
        help="joint: (projection_dim,) aligned with CLAP text; sequence: (64,768) encoder states",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading CLAP model: {args.encoder_name}")
    processor = AutoProcessor.from_pretrained(args.encoder_name)

    file_list = build_file_list(args.audio_dir, args.captions_csv, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.feature_format == "joint":
        model = ClapModel.from_pretrained(args.encoder_name).to(device)
        suffix = "_joint"
        bs = max(args.batch_size, 1)
        encode_joint_to_hdf5(
            file_list, model, processor, device,
            os.path.join(args.output_dir, f"{args.dataset}_{args.split}{suffix}.hdf5"),
            batch_size=bs,
        )
    else:
        model = ClapAudioModel.from_pretrained(args.encoder_name).to(device)
        encode_sequence_to_hdf5(
            file_list, model, processor, device,
            os.path.join(args.output_dir, f"{args.dataset}_{args.split}.hdf5"),
            batch_size=args.batch_size,
        )
 
    print("Done.")
 
 
if __name__ == "__main__":
    main()
 