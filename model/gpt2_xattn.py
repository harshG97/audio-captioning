# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
gpt2_xattn.py

PyTorch OpenAI GPT-2 model modified to support cross-attention to encoder hidden states.

Provides:
  - ThisGPT2Config:       adds cross_attention_reduce_factor to GPT2Config.
  - ThisGPT2Block:        GPT2Block + cross-attention sublayer.
  - ThisGPT2Model:        GPT2Model rebuilt with ThisGPT2Block layers.
  - ThisGPT2LMHeadModel:  GPT2LMHeadModel wrapping ThisGPT2Model.

Cross-attention is wired in by:
  1. Setting `add_cross_attention=True` on the config (RECAP does this).
  2. ThisGPT2Block.__init__ instantiates `crossattention` (a CrossAttention
     instance) and `ln_cross_attn` only when `add_cross_attention` is set.
  3. ThisGPT2Block.forward inserts the cross-attention sublayer between
     self-attention and the MLP.
  4. ThisGPT2Model swaps in ThisGPT2Block layers and otherwise uses the stock
     GPT2Model.forward, which passes `encoder_hidden_states` to every block and
     builds the causal + padding self-attention mask for eager and SDPA alike.

A hand-written copy of GPT2Model.forward used to live here. It built a
padding-only mask that switched off causal masking under GPT2SdpaAttention
(transformers >= 4.42), so it was removed in favour of the stock forward.
tests/test_decoder_causality.py guards against that regression.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block,
    GPT2LMHeadModel,
    GPT2Model,
)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.utils import logging

from .cross_attention import CrossAttention

logger = logging.get_logger(__name__)

class ThisGPT2Config(GPT2Config):
    model_type = "this_gpt2"

    def __init__(self, cross_attention_reduce_factor: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor

class ThisGPT2Block(GPT2Block):
    """
    GPT-2 transformer block with optional cross-attention sublayer.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size

        if config.add_cross_attention:
            self.crossattention = CrossAttention(config, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,                                  # absorb cache_position etc. from newer HF
    ) -> Tuple[torch.Tensor, ...]:

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]                   # (B, T, C)
        outputs = attn_outputs[1:]                      # (present, [attn_weights])
        hidden_states = attn_output + residual

        # Cross-attention sublayer
        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"{self.__class__.__name__} received `encoder_hidden_states` but has no "
                    "`crossattention` layer. Set `config.add_cross_attention=True` "
                    "before constructing the model."
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            cross_attn_output = cross_attn_outputs[0]
            hidden_states = cross_attn_output + residual

            # cross_attn_outputs = (output, present, [attn_weights])
            # We append cross-attn weights (index 2) when requested,
            # matching HuggingFace convention.
            outputs = outputs + cross_attn_outputs[2:]

        # MLP sublayer
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs        # (h, present, [self-attn], [cross-attn])
        else:
            outputs = (hidden_states,) + outputs[1:]    # drop `present` slot

        return outputs

class ThisGPT2Model(GPT2Model):
    """GPT2Model with ThisGPT2Block layers; forward is inherited unchanged."""

    config_class = ThisGPT2Config

    def __init__(self, config):
        super().__init__(config)
        # Replace the block list with our cross-attention-aware blocks.
        self.h = nn.ModuleList(
            [ThisGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        # Re-run weight init / device placement hooks set by the parent.
        self.post_init()


class ThisGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = ThisGPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ThisGPT2Model(config)
        # Re-tie / re-init weights for the swapped transformer.
        self.post_init()