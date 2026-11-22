"""Tests for cross-attention initialisation (model/build_recap.zero_init_cross_attention).

Run: python -m pytest tests/test_xattn_init.py -q  (CPU, no downloads)
"""
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.build_recap import zero_init_cross_attention
from model.gpt2_xattn import ThisGPT2Config, ThisGPT2LMHeadModel

KW = dict(n_layer=2, n_embd=64, n_head=4, vocab_size=100, n_positions=64)


def _load_decoder(tmp_path):
    """Mirror RECAP.from_encoder_decoder_pretrained: a plain GPT-2 checkpoint
    loaded into ThisGPT2LMHeadModel with fresh cross-attention layers."""
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(**KW)).save_pretrained(tmp_path)
    cfg = ThisGPT2Config.from_pretrained(tmp_path)
    cfg.add_cross_attention = True
    cfg.is_decoder = True
    cfg.cross_attention_reduce_factor = 1
    return ThisGPT2LMHeadModel.from_pretrained(tmp_path, config=cfg)


def test_default_init_is_random(tmp_path):
    dec = _load_decoder(tmp_path)
    assert all(blk.crossattention.c_proj.weight.abs().sum() > 0 for blk in dec.transformer.h)


def test_zero_init_after_load(tmp_path):
    dec = _load_decoder(tmp_path)
    assert zero_init_cross_attention(dec) == KW["n_layer"]
    for blk in dec.transformer.h:
        assert blk.crossattention.c_proj.weight.abs().sum() == 0
        assert blk.crossattention.c_proj.bias.abs().sum() == 0

    # With zeroed projections the decoder must match plain GPT-2 exactly.
    ids = torch.randint(1, KW["vocab_size"], (2, 10))
    enc = torch.randn(2, 8, KW["n_embd"])
    plain = GPT2LMHeadModel.from_pretrained(tmp_path).eval()
    dec.eval()
    with torch.no_grad():
        assert torch.allclose(dec(input_ids=ids, encoder_hidden_states=enc).logits,
                              plain(input_ids=ids).logits, atol=1e-5)

    # ...and the zeroed projection must still be trainable.
    out = dec(input_ids=ids, encoder_hidden_states=enc, labels=ids)
    out.loss.backward()
    assert dec.transformer.h[0].crossattention.c_proj.weight.grad.abs().sum() > 0
