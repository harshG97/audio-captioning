"""
clap_inputs.py

Build CLAP model inputs for a batch of waveforms, one clip at a time.

With the fused CLAP checkpoints (truncation="fusion"), the feature extractor
marks one RANDOM clip of the batch as `is_longer` whenever no clip in the
batch exceeds 10 s, which is always the case for AudioCaps. `is_longer`
switches the encoder's fusion path, so a clip's features depended on which
other clips shared its batch and on the random draw. Running the extractor
on each clip separately makes every clip of 10 s or less `is_longer=True`,
which is deterministic and matches the RECAP reference extraction (one clip
per call). The per-clip outputs have fixed shapes, so they are stacked into
one batch for the (GPU) encoder.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def clap_inputs_per_clip(processor, waveforms: Sequence[np.ndarray], sampling_rate: int) -> dict:
    """Return a batched {name: tensor} dict, built by processing each clip on its own.

    `processor` may be a ClapProcessor or a bare ClapFeatureExtractor."""
    extractor = getattr(processor, "feature_extractor", processor)
    parts = [extractor([w], sampling_rate=sampling_rate, return_tensors="pt") for w in waveforms]
    return {key: torch.cat([p[key] for p in parts], dim=0) for key in parts[0].keys()}
