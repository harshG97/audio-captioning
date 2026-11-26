"""BEATs audio encoder (Chen et al., 2022, https://arxiv.org/abs/2212.09058).

BEATs.py, backbone.py and modules.py are vendored unchanged from
github.com/microsoft/unilm/tree/master/beats (commit 31c5b90, MIT License,
see LICENSE.beats) apart from their imports, made relative here. kaldi.py is
torchaudio's Kaldi-compatible filterbank (BSD 2-Clause, see
LICENSE.torchaudio), vendored so BEATs needs no torchaudio install.

The BEATs code is not in Hugging Face transformers, and its checkpoints are
plain torch files ({"cfg", "model", "label_dict"}): see
data/extract_beats_features.py.
"""

from .BEATs import BEATs, BEATsConfig

__all__ = ["BEATs", "BEATsConfig"]
