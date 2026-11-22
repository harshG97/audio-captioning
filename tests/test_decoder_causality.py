"""Regression tests for the RECAP GPT-2 decoder's attention masking.

Run: python -m pytest tests/test_decoder_causality.py -q  (CPU, a few seconds)

Guards against the bug where a padding-only mask switched off causal masking
under GPT2SdpaAttention (transformers >= 4.42), letting training see future tokens.
"""
import inspect
import sys
from pathlib import Path

import pytest
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.gpt2_xattn import ThisGPT2Block, ThisGPT2Config, ThisGPT2LMHeadModel

V, H, T, S = 100, 64, 12, 8


def make(impl: str, seed: int = 0) -> ThisGPT2LMHeadModel:
    torch.manual_seed(seed)
    cfg = ThisGPT2Config(n_layer=2, n_embd=H, n_head=4, vocab_size=V, n_positions=64,
                         add_cross_attention=True, is_decoder=True, cross_attention_reduce_factor=1)
    cfg._attn_implementation = impl
    return ThisGPT2LMHeadModel(cfg).eval()


def batch():
    torch.manual_seed(1)
    ids = torch.randint(1, V, (2, T))
    ids[1, 7:] = 0                      # row 2 right-padded, like RecapCollator
    mask = torch.ones_like(ids); mask[1, 7:] = 0
    return ids, mask, torch.randn(2, S, H)


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
@pytest.mark.parametrize("use_mask", [True, False])
def test_no_future_leak(impl, use_mask):
    m = make(impl)
    ids, mask, enc = batch()
    ids2 = ids.clone(); ids2[0, -1] = (ids[0, -1] + 1) % V   # change only the last real token of row 0
    am = mask if use_mask else None
    with torch.no_grad():
        a = m(input_ids=ids, attention_mask=am, encoder_hidden_states=enc).logits
        b = m(input_ids=ids2, attention_mask=am, encoder_hidden_states=enc).logits
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6), "earlier positions saw a future token"


def test_sdpa_matches_eager():
    e, s = make("eager"), make("sdpa")
    s.load_state_dict(e.state_dict())
    ids, mask, enc = batch()
    with torch.no_grad():
        le = e(input_ids=ids, attention_mask=mask, encoder_hidden_states=enc).logits
        ls = s(input_ids=ids, attention_mask=mask, encoder_hidden_states=enc).logits
    real = mask.bool()
    assert torch.allclose(le[real], ls[real], atol=1e-5)


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_padding_does_not_change_real_tokens(impl):
    m = make(impl)
    ids, mask, enc = batch()
    with torch.no_grad():
        padded = m(input_ids=ids, attention_mask=mask, encoder_hidden_states=enc).logits[1, :7]
        alone = m(input_ids=ids[1:, :7], encoder_hidden_states=enc[1:]).logits[0]
    assert torch.allclose(padded, alone, atol=1e-5)


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_kv_cache_with_mask_matches_full_pass(impl):
    m = make(impl)
    ids, _, enc = batch()
    ids, enc = ids[:1], enc[:1]
    full_mask = torch.ones_like(ids)
    with torch.no_grad():
        full = m(input_ids=ids, attention_mask=full_mask, encoder_hidden_states=enc).logits[0, -1]
        pre = m(input_ids=ids[:, :-1], attention_mask=full_mask[:, :-1], encoder_hidden_states=enc, use_cache=True)
        step = m(input_ids=ids[:, -1:], attention_mask=full_mask, past_key_values=pre.past_key_values,
                 encoder_hidden_states=enc, use_cache=True).logits[0, -1]
    assert torch.allclose(full, step, atol=1e-5)


def test_block_signature_matches_stock_block():
    """The stock GPT2Model.forward drives ThisGPT2Block, so the two must take the
    same arguments. If a transformers upgrade changes GPT2Block.forward, this
    fails before the KV cache or masks can go silently wrong."""
    ours = list(inspect.signature(ThisGPT2Block.forward).parameters)
    stock = list(inspect.signature(GPT2Block.forward).parameters)
    assert ours == stock, f"ThisGPT2Block.forward{ours} != GPT2Block.forward{stock}"


def test_block_rejects_unknown_arguments():
    blk = make("eager").transformer.h[0]
    with pytest.raises(TypeError):
        blk(torch.randn(1, 3, H), past_key_value=None)
