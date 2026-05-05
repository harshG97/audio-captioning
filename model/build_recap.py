"""
build_recap.py — Factory for a RECAP model + GPT-2 tokenizer with the
special-token IDs wired up correctly for cross-attention seq2seq training.

GPT-2 has no native pad token; we reuse `<|endoftext|>` for pad/bos/eos.
The decoder_start_token_id and pad_token_id are written onto both the
parent RECAPConfig (used by RECAP.forward / shift_tokens_right) and the
decoder config (used by GPT-2's generation utilities).

Default trainable scope: cross-attention sublayers + ln_cross_attn only
(the rest of the GPT-2 decoder is frozen). Pass train_decoder=True to
fine-tune the full decoder.
"""

from __future__ import annotations

from typing import Tuple

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


def build_recap(
    encoder_name: str = "laion/clap-htsat-fused",
    decoder_name: str = "gpt2",
    cross_attention_reduce_factor: int = 1,
    freeze_encoder: bool = True,
    train_decoder: bool = False,
) -> Tuple[RECAP, PreTrainedTokenizerBase]:
    _register_auto_classes_once()

    tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    # Repurpose existing punctuation as special tokens
    tokenizer.pad_token = '!' ##
    tokenizer.eos_token = '.' ##

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = RECAP.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=encoder_name,
        decoder_pretrained_model_name_or_path=decoder_name,
        cross_attention_reduce_factor=cross_attention_reduce_factor,
    )

    ## bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = None ##
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = None  ## 

    model.decoder.config.pad_token_id = tokenizer.pad_token_id
    model.decoder.config.bos_token_id = None ##
    model.decoder.config.eos_token_id = tokenizer.eos_token_id

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()

    # Selective decoder freezing: by default only the randomly-initialized
    # cross-attention sublayers (and their LayerNorms) are trainable.
    if not train_decoder:
        for name, p in model.decoder.named_parameters():
            if "crossattention" not in name and "ln_cross_attn" not in name:
                p.requires_grad = False
    
    # if not train_decoder:
    #     for name, p in model.decoder.named_parameters():
    #         # Keep wte (word token embeddings) and lm_head trainable
    #         if not any(k in name for k in ["crossattention", "ln_cross_attn", "wte", "lm_head"]):
    #             p.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[build_recap] Trainable: {n_trainable:,} / {n_total:,} parameters "
          f"(train_decoder={train_decoder}, freeze_encoder={freeze_encoder})")

    return model, tokenizer
