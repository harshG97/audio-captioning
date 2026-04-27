"""
CLAP joint embedding utilities (laion/clap-htsat-fused and compatible checkpoints).

Uses HuggingFace ``ClapModel.get_text_features`` / ``get_audio_features`` so text
and audio live in the same projection space (same dimensionality as
``model.config.projection_dim``, typically 512).

This matches the datastore + retriever contract: (N, d) caption matrix and (d,)
query vectors for cosine similarity.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import ClapModel, ClapProcessor

DEFAULT_CLAP_MODEL = "laion/clap-htsat-fused"


def load_clap_model(
    model_name: str = DEFAULT_CLAP_MODEL,
    device: Union[str, "torch.device", None] = None,
):
    """Load CLAP model + processor on the given device."""
    import torch
    from transformers import AutoProcessor, ClapModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = ClapModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor, device


def get_embedding_dim(model: "ClapModel") -> int:
    """Projection / joint embedding size (same for audio and text)."""
    return int(model.config.projection_dim)


def encode_texts(
    texts: Sequence[str],
    model: "ClapModel",
    processor: "ClapProcessor",
    device: "torch.device",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode captions to float32 array of shape (N, projection_dim).

    Batched for memory; order matches input ``texts``.
    """
    import torch

    if not texts:
        return np.zeros((0, get_embedding_dim(model)), dtype=np.float32)

    out: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        out.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def encode_audio_waveforms(
    waveforms: Sequence[np.ndarray],
    model: "ClapModel",
    processor: "ClapProcessor",
    device: "torch.device",
    sampling_rate: int = 48000,
) -> np.ndarray:
    """
    Encode raw waveforms (list of 1-D float arrays) to (B, projection_dim).

    All waveforms are passed in a single processor batch (padded internally).
    For very large batches, call with smaller waveform groups.
    """
    import torch

    if not waveforms:
        return np.zeros((0, get_embedding_dim(model)), dtype=np.float32)

    inputs = processor(
        audios=list(waveforms),
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_audio_features(**inputs)
    return feats.cpu().numpy().astype(np.float32)


def encode_audio_file(
    file_path: str,
    model: "ClapModel",
    processor: "ClapProcessor",
    device: "torch.device",
    sampling_rate: int = 48000,
) -> np.ndarray:
    """
    Load one audio file and return shape (projection_dim,) float32 vector.
    """
    import librosa

    wav, _ = librosa.load(file_path, sr=sampling_rate, mono=True)
    mat = encode_audio_waveforms([wav], model, processor, device, sampling_rate)
    return mat[0]


def build_text_embed_fn(
    model_name: str = DEFAULT_CLAP_MODEL,
    batch_size: int = 32,
    device: Union[str, "torch.device", None] = None,
):
    """
    Returns a callable ``(list[str]) -> (N, d)`` suitable for
    ``Datastore.build_embeddings(embed_fn)``.
    """
    model, processor, dev = load_clap_model(model_name, device)

    def embed_fn(captions: List[str]) -> np.ndarray:
        return encode_texts(captions, model, processor, dev, batch_size=batch_size)

    return embed_fn
