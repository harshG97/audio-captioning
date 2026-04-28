from __future__ import annotations

import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoProcessor, ClapModel

MODEL_NAME = "laion/clap-htsat-fused"
TARGET_SAMPLE_RATE = 48000

_MODEL: ClapModel | None = None
_PROCESSOR: AutoProcessor | None = None
_DEVICE: torch.device | None = None


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _load_clap() -> Tuple[ClapModel, AutoProcessor]:
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        _MODEL = ClapModel.from_pretrained(MODEL_NAME)
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME)
        _MODEL.eval()
        _MODEL.to(_get_device())
    return _MODEL, _PROCESSOR


def _read_wav_mono_float32(wav_path: str) -> Tuple[np.ndarray, int]:
    wav_file = Path(wav_path)
    if not wav_file.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    with wave.open(str(wav_file), "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()
        audio_bytes = wf.readframes(num_frames)

    if sample_width == 1:
        audio = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    return audio.astype(np.float32), sample_rate


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    if audio.size == 0:
        return audio.astype(np.float32)

    duration = audio.shape[0] / float(orig_sr)
    target_length = max(1, int(round(duration * target_sr)))
    orig_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, orig_positions, audio)
    return resampled.astype(np.float32)


def get_text_embeddings(text_list: List[str]) -> np.ndarray:
    if len(text_list) == 0:
        model, _ = _load_clap()
        return np.empty((0, model.config.projection_dim), dtype=np.float32)

    model, processor = _load_clap()
    inputs = processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(_get_device()) for k, v in inputs.items()}

    with torch.no_grad():
        # Use the dedicated CLAP text encoder API for projected embeddings.
        text_embeds_tensor = model.get_text_features(**inputs)

    if not isinstance(text_embeds_tensor, torch.Tensor):
        raise TypeError("CLAP text embeddings must be returned as a torch.Tensor.")

    text_embeds = text_embeds_tensor.detach().cpu().numpy().astype(np.float32)
    expected_dim = model.config.projection_dim
    if text_embeds.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected text embedding dimension: {text_embeds.shape[1]} != {expected_dim}"
        )
    return text_embeds


def get_audio_embedding(wav_path: str) -> np.ndarray:
    model, processor = _load_clap()
    audio, sample_rate = _read_wav_mono_float32(wav_path)
    audio = _resample_audio(audio, sample_rate, TARGET_SAMPLE_RATE)

    batch = processor(
        audios=[audio],
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    batch = {k: v.to(_get_device()) for k, v in batch.items()}

    with torch.no_grad():
        audio_embeds = model.get_audio_features(**batch)

    audio_embeds = audio_embeds.detach().cpu().numpy().astype(np.float32)
    expected_dim = model.config.projection_dim
    if audio_embeds.shape != (1, expected_dim):
        raise ValueError(
            f"Unexpected audio embedding shape: {audio_embeds.shape} != (1, {expected_dim})"
        )
    return audio_embeds[0]
