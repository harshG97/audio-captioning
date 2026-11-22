"""Tests for data/audio_preprocess.load_clap_audio_encoder.

Builds a tiny random CLAP checkpoint locally (no downloads) in the same
format as the hub checkpoints (a full ClapModel).

Run: python -m pytest tests/test_clap_encoder_loading.py -q
"""
import sys
from pathlib import Path

import pytest
import torch
from transformers import ClapAudioConfig, ClapAudioModel, ClapConfig, ClapModel, ClapTextConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.audio_preprocess import load_clap_audio_encoder


def _tiny_audio_config():
    return ClapAudioConfig(depths=[1, 1, 1, 1], num_attention_heads=[1, 1, 1, 1],
                           patch_embeds_hidden_size=8, hidden_size=64, enable_fusion=True,
                           projection_hidden_size=32)


@pytest.fixture(scope="module")
def clap_dir(tmp_path_factory):
    torch.manual_seed(0)
    text = ClapTextConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                          intermediate_size=64, max_position_embeddings=40, pad_token_id=1)
    cfg = ClapConfig(text_config=text.to_dict(), audio_config=_tiny_audio_config().to_dict(), projection_dim=16)
    d = tmp_path_factory.mktemp("tiny_clap")
    ClapModel(cfg).save_pretrained(d)
    return d


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_loads_pretrained_audio_tower(clap_dir):
    reference = ClapModel.from_pretrained(clap_dir).audio_model
    assert _same_weights(load_clap_audio_encoder(str(clap_dir)), reference)


def test_plain_clap_audio_model_from_full_checkpoint_is_random(clap_dir):
    """Documents the pitfall the loader works around."""
    reference = ClapModel.from_pretrained(clap_dir).audio_model
    assert not _same_weights(ClapAudioModel.from_pretrained(clap_dir), reference)


def test_audio_only_checkpoint(tmp_path):
    torch.manual_seed(1)
    audio = ClapAudioModel(_tiny_audio_config())
    audio.save_pretrained(tmp_path)
    assert _same_weights(load_clap_audio_encoder(str(tmp_path)), audio)
