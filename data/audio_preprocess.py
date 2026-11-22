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
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, ClapAudioModel, ClapModel, PretrainedConfig

# Make `model.device` importable when this file is invoked as a script
# (`python data/audio_preprocess.py ...`) instead of `python -m data.audio_preprocess`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.clap_inputs import clap_inputs_per_clip
from model.device import best_device
 
 
def load_clap_audio_encoder(name_or_path):
    """Load CLAP's audio tower with its pretrained weights.

    Hub checkpoints such as laion/clap-htsat-fused are full ClapModel
    checkpoints, whose audio weights are stored under "audio_model.".
    ClapAudioModel.from_pretrained() on them matches none of those keys and
    silently returns a randomly initialised encoder, so load the full model
    and take its audio tower instead. Raises if any audio weight is missing.
    """
    # Read model_type from the raw config: AutoConfig does not know "clap_audio_model".
    model_type = PretrainedConfig.get_config_dict(name_or_path)[0].get("model_type")
    if model_type == "clap":
        clap, info = ClapModel.from_pretrained(name_or_path, output_loading_info=True)
        missing = [k for k in info["missing_keys"] if k.startswith("audio_model.")]
        encoder = clap.audio_model
    else:  # an audio-only (ClapAudioModel) checkpoint
        encoder, info = ClapAudioModel.from_pretrained(name_or_path, output_loading_info=True)
        missing = list(info["missing_keys"])
    if missing:
        raise RuntimeError(
            f"{len(missing)} CLAP audio-encoder weights were not found in "
            f"{name_or_path} (e.g. {missing[:3]}); refusing to extract features "
            f"from a partly random encoder."
        )
    return encoder


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
 
 
def _load_audio_path(file_path):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    return load_audio(file_path)[0]


def encode_to_hdf5(file_list, model, processor, device, output_path, batch_size=1, num_workers=0):
    """
    Extract features and write immediately to an HDF5 file.
    Each audio_id becomes a dataset key.

    Args:
        num_workers: number of parallel processes for audio loading.
                     0 = sequential (original behavior).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    seen_ids = set()

    # Reuse a single pool across all batches; closure-bound nested functions
    # don't pickle, so the loader is module-level (`_load_audio_path`).
    pool = ProcessPoolExecutor(max_workers=num_workers) if num_workers > 0 else None

    try:
        with h5py.File(output_path, "w") as h5f:
            for idx in tqdm(range(0, len(file_list), batch_size), desc="Extracting features"):
                batch = file_list[idx:idx + batch_size]

                audio_ids = [item[0] for item in batch]
                file_paths = [item[1] for item in batch]

                # Load and resample audio (parallel or sequential)
                if pool is not None:
                    waveforms = list(pool.map(_load_audio_path, file_paths))
                else:
                    waveforms = [load_audio(fp)[0] for fp in file_paths]

                # CLAP feature extraction, one clip at a time (see data/clap_inputs.py:
                # batched calls set `is_longer` on a random clip). `padding` is left
                # unset so the extractor uses its configured "repeatpad".
                inputs = clap_inputs_per_clip(processor, waveforms, sampling_rate=48000)
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
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    print(f"Saved {len(seen_ids)} entries to {output_path}")

    # Verify by reading back a sample entry
    with h5py.File(output_path, "r") as h5f:
        sample_id = list(h5f.keys())[0]
        sample_shape = h5f[sample_id][()].shape
        print(f"Sample entry '{sample_id}': shape = {sample_shape}")
        print("  Expected: (64, 768)")
 
 
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
    parser.add_argument("--num_workers", type=int, default=0,
                    help="Parallel processes for audio loading (0 = sequential)")
    args = parser.parse_args()
 
    # Set device (cuda > mps > cpu)
    device = best_device()
    print(f"Using device: {device}")
 
    # Load model and processor
    print(f"Loading CLAP model: {args.encoder_name}")
    model = load_clap_audio_encoder(args.encoder_name).to(device)
    processor = AutoProcessor.from_pretrained(args.encoder_name)
 
    # Build file list and verify files exist
    file_list = build_file_list(args.audio_dir, args.captions_csv, args.dataset)
 
    # Extract features and save to HDF5
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.split}.hdf5")
    encode_to_hdf5(file_list, model, processor, device, output_path,
                   batch_size=args.batch_size, num_workers=args.num_workers)
 
    print("Done.")
 
 
if __name__ == "__main__":
    main()
 