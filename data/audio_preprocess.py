"""
audio_preprocess.py — Extract CLAP audio encoder features and cache to HDF5.

Run this ONCE before training. The CLAP encoder is frozen, so outputs
never change. Caching avoids recomputing them every epoch.

Usage:
    python audio_preprocess.py \
        --audio_dir /scratch/team/recap/data/clotho/development \
        --captions_csv /scratch/team/recap/data/clotho/clotho_captions_development.csv \
        --output_path /scratch/team/recap/features/train.hdf5 \
        --dataset clotho

    python audio_preprocess.py \
        --audio_dir /scratch/team/recap/data/clotho/evaluation \
        --captions_csv /scratch/team/recap/data/clotho/clotho_captions_evaluation.csv \
        --output_path /scratch/team/recap/features/test.hdf5 \
        --dataset clotho
"""

import argparse
import os
import h5py
import torch
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import ClapAudioModel, AutoProcessor

from model.device import best_device
 
 
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
 
 
def encode_to_hdf5(file_list, model, processor, device, output_path, batch_size=1):
    """
    Extract features and write immediately to an HDF5 file.
    Each audio_id becomes a dataset key.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
 
    seen_ids = set()
 
    with h5py.File(output_path, "w") as h5f:
        for idx in tqdm(range(0, len(file_list), batch_size), desc="Extracting features"):
            batch = file_list[idx:idx + batch_size]
 
            audio_ids = [item[0] for item in batch]
            file_paths = [item[1] for item in batch]
 
            # Load and resample all audio in this batch
            waveforms = [load_audio(fp)[0] for fp in file_paths]
 
            # Process batch through the CLAP feature extractor
            inputs = processor(
                audios=waveforms,
                sampling_rate=48000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
 
            with torch.no_grad():
                outputs = model(**inputs)
 
            # last_hidden_state shape: (batch, seq_len, hidden_dim)
            # Apply flatten + permute to get shape (batch, 64, 768)
            encodings = torch.flatten(outputs.last_hidden_state, 2)
            encodings = encodings.permute(0, 2, 1).detach().cpu().numpy()
 
            # Write each encoding to HDF5, skipping duplicates
            for audio_id, encoding in zip(audio_ids, encodings):
                if audio_id not in seen_ids:
                    h5f.create_dataset(str(audio_id), data=encoding)
                    seen_ids.add(audio_id)
 
    print(f"Saved {len(seen_ids)} entries to {output_path}")
 
    # Verify by reading back a sample entry
    with h5py.File(output_path, "r") as h5f:
        sample_id = list(h5f.keys())[0]
        sample_shape = h5f[sample_id][()].shape
        print(f"Sample entry '{sample_id}': shape = {sample_shape}")
        print(f"  Expected: (64, 768)")
 
 
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
    args = parser.parse_args()
 
    # Set device (cuda > mps > cpu)
    device = best_device()
    print(f"Using device: {device}")
 
    # Load model and processor
    print(f"Loading CLAP model: {args.encoder_name}")
    model = ClapAudioModel.from_pretrained(args.encoder_name).to(device)
    processor = AutoProcessor.from_pretrained(args.encoder_name)
 
    # Build file list and verify files exist
    file_list = build_file_list(args.audio_dir, args.captions_csv, args.dataset)
 
    # Extract features and save to HDF5
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.split}.hdf5")
    encode_to_hdf5(file_list, model, processor, device, output_path, batch_size=args.batch_size)
 
    print("Done.")
 
 
if __name__ == "__main__":
    main()
 