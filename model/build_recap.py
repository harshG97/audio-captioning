"""
build_recap.py — Factory for a RECAP model + GPT-2 tokenizer with the
special-token IDs wired up correctly for cross-attention seq2seq training.

GPT-2 has no native pad token; we reuse `<|endoftext|>` for pad/bos/eos.
The decoder_start_token_id and pad_token_id are written onto both the
parent RECAPConfig (used by RECAP.forward / shift_tokens_right) and the
decoder config (used by GPT-2's generation utilities).
"""

from __future__ import annotations

from typing import Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model.recap import RECAP


def build_recap(
    encoder_name: str = "laion/clap-htsat-fused",
    decoder_name: str = "gpt2",
    cross_attention_reduce_factor: int = 1,
    freeze_encoder: bool = True,
) -> Tuple[RECAP, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = RECAP.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=encoder_name,
        decoder_pretrained_model_name_or_path=decoder_name,
        cross_attention_reduce_factor=cross_attention_reduce_factor,
    )

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = bos_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = bos_id

    model.decoder.config.pad_token_id = tokenizer.pad_token_id
    model.decoder.config.bos_token_id = bos_id
    model.decoder.config.eos_token_id = tokenizer.eos_token_id

    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()

    return model, tokenizer
