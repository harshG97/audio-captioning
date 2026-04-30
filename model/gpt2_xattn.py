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
  4. ThisGPT2Model.forward threads `encoder_hidden_states` /
     `encoder_attention_mask` through every block.
"""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block,
    GPT2LMHeadModel,
    GPT2Model,
)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
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
    """GPT2Model with ThisGPT2Block layers and an encoder-aware forward."""

    config_class = ThisGPT2Config

    def __init__(self, config):
        super().__init__(config)
        # Replace the block list with our cross-attention-aware blocks.
        self.h = nn.ModuleList(
            [ThisGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        # Re-run weight init / device placement hooks set by the parent.
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        # Resolve flags
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Resolve input shape
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        # Position ids and past length
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Self-attention mask
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # [B, 1, 1, T_to] for broadcasting over heads / from-positions
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Cross-attention mask
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Head mask
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        # Gradient-checkpoint compatibility
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`..."
            )
            use_cache = False

        # Containers for outputs
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None

        # Block loop
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallelism support (kept for compatibility with HF base class)
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(p.to(hidden_states.device) for p in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, output_attentions)
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)

            if output_attentions:
                # Index of self-attn weights in `outputs` shifts by 1 if `present` was emitted
                self_attn_idx = 2 if use_cache else 1
                all_self_attentions = all_self_attentions + (outputs[self_attn_idx],)
                if self.config.add_cross_attention:
                    cross_attn_idx = 3 if use_cache else 2
                    all_cross_attentions = all_cross_attentions + (outputs[cross_attn_idx],)

            # Move tensors to the next device shard if needed
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # Final norm + reshape
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ] if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ThisGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = ThisGPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ThisGPT2Model(config)
        # Re-tie / re-init weights for the swapped transformer.
        self.post_init()