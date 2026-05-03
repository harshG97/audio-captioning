"""
cross_attention.py

Standalone cross-attention module for RECAP.
Extends HuggingFace GPT2Attention to support cross-attention with a
cross_attention_reduce_factor that shrinks the key/value projection
dimensions when attending to encoder hidden states.

Usage:
    xattn = CrossAttention(config, layer_idx=0)
    out, = xattn(hidden_states, encoder_hidden_states=enc_hs)
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.pytorch_utils import Conv1D


class CrossAttention(GPT2Attention):
    """
    Cross-attention layer for RECAP's GPT-2 decoder.

    Inherits GPT2Attention and overrides:
      - __init__: replaces c_attn / q_attn / c_proj with reduced-dimension
                  projections controlled by cross_attention_reduce_factor.
      - forward:  routes to cross-attention path when encoder_hidden_states
                  is provided; self-attention path otherwise (for safety,
                  though in practice this layer is always called with
                  encoder_hidden_states).

    Args:
        config: ThisGPT2Config, must expose:
            - embed_dim / hidden_size
            - num_attention_heads (n_head)
            - cross_attention_reduce_factor (int >= 1)
            - attention_dropout / resid_pdrop
        layer_idx (int, optional): used by the parent for caching.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        # Initialise as cross-attention from the start so the parent sets
        # self.is_cross_attention = True and creates q_attn.
        super().__init__(config, is_cross_attention=True, layer_idx=layer_idx)

        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor

        # embed_dim is set by the parent; hidden_size is an alias in some configs.
        embed_dim = self.embed_dim  # e.g. 768 for GPT-2 small

        r = self.cross_attention_reduce_factor  # shorthand

        # Key/value projection (applied to encoder hidden states):
        #   output dim = 2 * (embed_dim / r)
        self.c_attn = Conv1D(int(2 * embed_dim / r), embed_dim)

        # Query projection (applied to decoder hidden states):
        #   output dim = embed_dim / r
        self.q_attn = Conv1D(int(embed_dim / r), embed_dim)

        # Output projection: maps reduced head outputs back to embed_dim.
        # Zero-init both weight and bias so cross-attention initially adds 0
        # to the residual stream — it acts as a no-op at step 0 and only
        # contributes once training has had a chance to learn useful audio
        # conditioning. Random init causes the decoder to learn to attenuate
        # cross-attn output before it can become useful, which (combined with
        # an informative retrieval prompt) leads to the decoder ignoring audio
        # entirely. See scripts/ablate_audio.py.
        self.c_proj = Conv1D(embed_dim, int(embed_dim / r))
        nn.init.zeros_(self.c_proj.weight)
        nn.init.zeros_(self.c_proj.bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor, # (B, T, C)
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None, # (B, S, C_enc)
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """
        Args:
            hidden_states: Decoder hidden states  (B, T, embed_dim)
            encoder_hidden_states: Encoder output (B, S, embed_dim)
            encoder_attention_mask: Additive mask for encoder positions (B, 1, 1, S)
            layer_past, use_cache: KV-cache support (inherited behaviour).
            head_mask: Per-head scaling mask.
            output_attentions: Return attention weights when True.

        Returns:
            (attn_output, present [, attn_weights])
        """
        if encoder_hidden_states is not None:
            # Cross-attention path 
            r = self.cross_attention_reduce_factor
            reduced_head_dim = int(self.head_dim / r)
            reduced_split_size = int(self.split_size / r) # num_heads * reduced_head_dim

            # Queries come from the decoder
            query = self.q_attn(hidden_states) # (B, T, C/r)

            # Keys and values come from the encoder
            key, value = self.c_attn(encoder_hidden_states).split(
                reduced_split_size, dim=2
            ) # each (B, S, C/r)

            # Use the encoder mask for this sublayer
            attention_mask = encoder_attention_mask

            # Split into heads
            query = self._split_heads(query, self.num_heads, reduced_head_dim)
            key   = self._split_heads(key, self.num_heads, reduced_head_dim)
            value = self._split_heads(value, self.num_heads, reduced_head_dim)

        else:
            # Self-attention fallback (standard GPT-2)
            query, key, value = self.c_attn(hidden_states).split(
                self.split_size, dim=2
            )
            query = self._split_heads(query, self.num_heads, self.head_dim)
            key   = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

        # KV-cache: concatenate with cached keys/values if present
        if layer_past is not None:
            past_key, past_value = layer_past
            key   = torch.cat((past_key, key),   dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        # Scaled dot-product attention
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

        # Merge heads and project back to embed_dim
        effective_head_dim = (
            int(self.head_dim / self.cross_attention_reduce_factor)
            if encoder_hidden_states is not None
            else self.head_dim
        )
        attn_output = self._merge_heads(attn_output, self.num_heads, effective_head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs # (attn_output, present [, attn_weights])