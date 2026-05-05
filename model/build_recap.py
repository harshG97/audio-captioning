"""
build_recap.py — Factory for a RECAP model + GPT-2 tokenizer with the
special-token IDs wired up correctly for cross-attention seq2seq training.

GPT-2 has no native pad token; we repurpose `!` as pad and `.` as EOS.
The decoder_start_token_id and pad_token_id are written onto both the
parent RECAPConfig (used by RECAP.forward / shift_tokens_right) and the
decoder config (used by GPT-2's generation utilities).

Default trainable scope: cross-attention sublayers + ln_cross_attn only
(the rest of the GPT-2 decoder is frozen). Pass train_decoder=True to
fine-tune the full decoder.

Public helpers:
  - build_recap:          construct from encoder/decoder names (mode 1).
  - load_recap_checkpoint: reload a saved RECAP + tokenizer from disk.
  - wire_special_tokens:  set pad/eos/bos IDs on tokenizer + model configs.
  - apply_freeze_policy:  toggle requires_grad on encoder / decoder params.
  - build_param_groups:   AdamW param groups with separate LRs for the
                          cross-attention sublayers and the GPT-2 backbone.
"""

from __future__ import annotations

from typing import List, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model.gpt2_xattn import ThisGPT2Config, ThisGPT2LMHeadModel
from model.recap import RECAP, RECAPConfig


# Cross-attention parameter budget (M) -> reduce_factor passed to RECAP.
# Higher reduce factor = smaller / cheaper cross-attention.
ATTENTION_SIZE_TO_REDUCE_FACTOR: dict[float, int] = {
    28.0: 1,
    14.0: 2,
    7.0: 4,
    3.5: 8,
    1.75: 16,
}


# Substrings that identify the cross-attention sublayer params. Used by
# both the freeze policy and the optimizer param-group split.
XATTN_PARAM_KEYS: tuple[str, ...] = ("crossattention", "ln_cross_attn")


def _register_auto_classes_once() -> None:
    """Register custom configs/models once so `AutoModel.from_pretrained(ckpt)`
    works without importing RECAP directly. Safe to call repeatedly."""
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM

    try:
        AutoConfig.register("this_gpt2", ThisGPT2Config)
    except ValueError:
        pass
    try:
        AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    except ValueError:
        pass
    try:
        AutoConfig.register("recap", RECAPConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(RECAPConfig, RECAP)
    except ValueError:
        pass


def wire_special_tokens(model: RECAP, tokenizer: PreTrainedTokenizerBase) -> None:
    """Set pad/eos/bos token IDs on the tokenizer and propagate to model
    configs. Idempotent: safe to call after either fresh construction or
    a from_pretrained reload."""
    tokenizer.pad_token = '!'
    tokenizer.eos_token = '.'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = None
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = None

    model.decoder.config.pad_token_id = tokenizer.pad_token_id
    model.decoder.config.bos_token_id = None
    model.decoder.config.eos_token_id = tokenizer.eos_token_id


def apply_freeze_policy(
    model: RECAP,
    *,
    freeze_encoder: bool,
    train_decoder: bool,
) -> None:
    """Toggle requires_grad on encoder and decoder params per the policy.

    - freeze_encoder=True  : encoder params frozen and put in eval mode.
    - train_decoder=False  : only cross-attn sublayers + ln_cross_attn are
                             trainable in the decoder; everything else frozen.
    - train_decoder=True   : every decoder param is trainable.
    """
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()
    else:
        for p in model.encoder.parameters():
            p.requires_grad = True

    if train_decoder:
        for p in model.decoder.parameters():
            p.requires_grad = True
    else:
        for name, p in model.decoder.named_parameters():
            p.requires_grad = any(k in name for k in XATTN_PARAM_KEYS)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[apply_freeze_policy] Trainable: {n_trainable:,} / {n_total:,} parameters "
          f"(train_decoder={train_decoder}, freeze_encoder={freeze_encoder})")


def build_param_groups(
    model: RECAP,
    *,
    lr_xattn: float,
    lr_decoder: float,
    weight_decay: float,
) -> List[dict]:
    """Build AdamW param groups split by (xattn vs decoder backbone) ×
    (decay vs no-decay).

    - xattn group:    params whose name contains 'crossattention' or
                      'ln_cross_attn' -> LR = lr_xattn.
    - decoder group:  every other trainable decoder param (e.g. wte,
                      lm_head, attn, mlp, ln_*) -> LR = lr_decoder.
    - no-decay:       biases, LayerNorm/ln_* weights, and any 1-D param
                      get weight_decay=0.0.

    Encoder params, if trainable, fall into the decoder LR group; the
    typical caller has the encoder frozen so this is a no-op.
    """
    def is_no_decay(name: str, param) -> bool:
        if param.ndim < 2:
            return True
        lname = name.lower()
        return ("bias" in lname) or ("layernorm" in lname) or (".ln_" in lname) or lname.endswith("ln_f.weight")

    xattn_decay, xattn_nodecay = [], []
    other_decay, other_nodecay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_xattn = any(k in name for k in XATTN_PARAM_KEYS)
        bucket_no_decay = is_no_decay(name, p)
        if is_xattn:
            (xattn_nodecay if bucket_no_decay else xattn_decay).append(p)
        else:
            (other_nodecay if bucket_no_decay else other_decay).append(p)

    groups: List[dict] = []
    if xattn_decay:
        groups.append({"params": xattn_decay, "lr": lr_xattn, "weight_decay": weight_decay})
    if xattn_nodecay:
        groups.append({"params": xattn_nodecay, "lr": lr_xattn, "weight_decay": 0.0})
    if other_decay:
        groups.append({"params": other_decay, "lr": lr_decoder, "weight_decay": weight_decay})
    if other_nodecay:
        groups.append({"params": other_nodecay, "lr": lr_decoder, "weight_decay": 0.0})

    n_xattn = sum(p.numel() for g in groups
                  for p in g["params"] if g["lr"] == lr_xattn)
    n_decoder = sum(p.numel() for g in groups
                    for p in g["params"] if g["lr"] == lr_decoder)
    print(f"[build_param_groups] xattn LR={lr_xattn} ({n_xattn:,} params), "
          f"decoder LR={lr_decoder} ({n_decoder:,} params), "
          f"weight_decay={weight_decay}")

    return groups


def build_recap(
    encoder_name: str = "laion/clap-htsat-fused",
    decoder_name: str = "gpt2",
    cross_attention_reduce_factor: int = 1,
    freeze_encoder: bool = True,
    train_decoder: bool = False,
) -> Tuple[RECAP, PreTrainedTokenizerBase]:
    _register_auto_classes_once()

    tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    model = RECAP.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=encoder_name,
        decoder_pretrained_model_name_or_path=decoder_name,
        cross_attention_reduce_factor=cross_attention_reduce_factor,
    )

    wire_special_tokens(model, tokenizer)
    apply_freeze_policy(model, freeze_encoder=freeze_encoder, train_decoder=train_decoder)

    return model, tokenizer


def load_recap_checkpoint(
    checkpoint_path: str,
) -> Tuple[RECAP, PreTrainedTokenizerBase]:
    """Reload a RECAP + tokenizer that was saved by `train.py`.

    Does NOT apply freeze policy or re-wire special tokens — the caller
    is responsible for both (so the same helpers work for fresh models)."""
    _register_auto_classes_once()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = RECAP.from_pretrained(checkpoint_path)
    return model, tokenizer
