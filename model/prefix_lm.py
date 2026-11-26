"""
prefix_lm.py — Audio captioning with a decoder-only LLM, audio given as a prefix.

The precomputed audio features (BEATs or CLAP, from the same HDF5 files as
RECAP) are average-pooled in time and mapped by a small trainable MLP into
the LLM's embedding space. The LLM then reads

    [audio tokens] [start token] [RECAP prompt with retrieved captions] [caption] [end token]

and learns to write the caption. The LLM itself stays frozen; LoRA adapters
(peft) on its attention and MLP layers are trained with the projector.

The model takes the same batches as RECAP (data/recap_dataset.py):
decoder_input_ids = start token + prompt + caption (shifted by one), labels
= prompt masked out + caption + end token, and the features as
encoder_outputs. Logits at each text position predict the label at that
position, as in RECAP, so datasets, collator, training loop and the audio
ablation are shared.

Checkpoints hold only the trained weights (projector and LoRA, ~20M
parameters) plus the config; loading rebuilds the base LLM from its name.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _IncompatibleKeys
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PretrainedConfig,
                          PreTrainedModel)
from transformers.modeling_outputs import CausalLMOutput

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class AudioPrefixConfig(PretrainedConfig):
    model_type = "audio_prefix_lm"

    def __init__(self, llm_name: str = "Qwen/Qwen2.5-1.5B", feature_width: int = 768,
                 audio_pool: int = 4, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, lora_targets=LORA_TARGETS,
                 llm_revision: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm_name = llm_name
        # Hub commit the LLM was trained from, so a checkpoint's adapters are
        # never loaded onto updated base weights (None for a local folder).
        self.llm_revision = llm_revision
        self.feature_width = feature_width
        self.audio_pool = audio_pool
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = list(lora_targets)


class AudioProjector(nn.Module):
    """(B, T, feature_width) -> (B, T / pool, llm_width): average-pool groups
    of `pool` consecutive feature vectors, then LayerNorm and a 2-layer MLP."""

    def __init__(self, feature_width: int, llm_width: int, pool: int):
        super().__init__()
        self.pool = pool
        self.norm = nn.LayerNorm(feature_width)
        self.mlp = nn.Sequential(nn.Linear(feature_width, llm_width), nn.GELU(),
                                 nn.Linear(llm_width, llm_width))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        b, t, d = feats.shape
        if t % self.pool:
            raise ValueError(f"{t} feature vectors per clip do not split into groups of {self.pool}")
        pooled = feats.view(b, t // self.pool, self.pool, d).mean(dim=2)
        return self.mlp(self.norm(pooled))


class AudioPrefixLM(PreTrainedModel):
    config_class = AudioPrefixConfig
    main_input_name = "decoder_input_ids"

    def __init__(self, config: AudioPrefixConfig, llm: Optional[nn.Module] = None):
        super().__init__(config)
        from peft import LoraConfig, get_peft_model   # only this model needs peft

        base = llm if llm is not None else AutoModelForCausalLM.from_pretrained(
            config.llm_name, revision=config.llm_revision, torch_dtype=torch.float32)
        for p in base.parameters():
            p.requires_grad_(False)
        lora = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,
                          lora_dropout=config.lora_dropout, target_modules=list(config.lora_targets),
                          bias="none", task_type="CAUSAL_LM")
        self.llm = get_peft_model(base, lora)
        self.projector = AudioProjector(config.feature_width, base.config.hidden_size, config.audio_pool)

    # ---- inputs -------------------------------------------------------------

    @staticmethod
    def _features(encoder_outputs=None, encoder_hidden_states=None) -> torch.Tensor:
        if encoder_hidden_states is not None:
            return encoder_hidden_states
        if isinstance(encoder_outputs, torch.Tensor):
            return encoder_outputs
        return encoder_outputs[0]          # tuple or BaseModelOutput

    def build_inputs(self, feats: torch.Tensor, token_ids: torch.Tensor,
                     token_mask: Optional[torch.Tensor] = None):
        """Embeddings and attention mask for [audio tokens][token_ids]."""
        audio = self.projector(feats.to(self.projector.mlp[0].weight.dtype))
        text = self.llm.get_input_embeddings()(token_ids)
        embeds = torch.cat([audio.to(text.dtype), text], dim=1)
        if token_mask is None:
            token_mask = torch.ones_like(token_ids)
        audio_mask = torch.ones(audio.shape[:2], dtype=token_mask.dtype, device=token_mask.device)
        return embeds, torch.cat([audio_mask, token_mask], dim=1), audio.shape[1]

    def forward(self, encoder_outputs=None, decoder_input_ids=None, decoder_attention_mask=None,
                labels=None, encoder_hidden_states=None, return_dict=True, **kwargs):
        feats = self._features(encoder_outputs, encoder_hidden_states)
        embeds, mask, n_audio = self.build_inputs(feats, decoder_input_ids, decoder_attention_mask)
        out = self.llm(inputs_embeds=embeds, attention_mask=mask, use_cache=False, return_dict=True)
        logits = out.logits[:, n_audio:, :]        # one per text position, as in RECAP
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   labels.reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, encoder_hidden_states: torch.Tensor, input_ids: torch.Tensor, **kwargs):
        """New tokens only: the LLM gets embeddings, not ids, so its output
        holds no prompt tokens. Decoding follows only the given arguments: the
        LLM's own generation_config.json may sample or add penalties (Llama
        3.2's samples at temperature 0.6), so it is replaced by a neutral one."""
        embeds, mask, _ = self.build_inputs(encoder_hidden_states, input_ids)
        return self.llm.generate(inputs_embeds=embeds, attention_mask=mask,
                                 generation_config=GenerationConfig(), **kwargs)

    # ---- checkpoints: trained weights only ----------------------------------

    def _trained_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def state_dict(self, *args, **kwargs):
        """Only the trained weights (projector and LoRA). The frozen LLM is
        rebuilt from config.llm_name, so checkpoints stay small."""
        params = dict(self.named_parameters())
        return OrderedDict((n, params[n].detach()) for n in self._trained_names())

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Always strict about the trained weights, whatever `strict` says:
        a checkpoint with missing or extra entries would silently leave
        random adapters in place."""
        expected, got = set(self._trained_names()), set(state_dict)
        if expected != got:
            raise RuntimeError(f"checkpoint does not match the model: missing {sorted(expected - got)[:3]}, "
                               f"unexpected {sorted(got - expected)[:3]}")
        params = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in state_dict.items():
                if params[name].shape != value.shape:
                    raise RuntimeError(f"{name}: checkpoint {tuple(value.shape)} vs model {tuple(params[name].shape)}")
                params[name].copy_(value)
        return _IncompatibleKeys([], [])

    @classmethod
    def from_pretrained(cls, path: str, *args, **kwargs) -> "AudioPrefixLM":
        from safetensors.torch import load_file
        config = AudioPrefixConfig.from_pretrained(path)
        model = cls(config)
        weights = os.path.join(path, "model.safetensors")
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"no model.safetensors in {path}")
        model.load_state_dict(load_file(weights))
        return model.eval()


def build_prefix_lm(llm_name: str, feature_width: int, audio_pool: int = 4, lora_r: int = 16,
                    lora_alpha: int = 32, lora_dropout: float = 0.05):
    """Model and tokenizer for training. The start token (fed after the audio)
    and the end token are the tokenizer's EOS, as for a document boundary."""
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AudioPrefixConfig(llm_name=llm_name, feature_width=feature_width, audio_pool=audio_pool,
                               lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                               decoder_start_token_id=tokenizer.eos_token_id,
                               eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = AudioPrefixLM(config)
    model.config.llm_revision = getattr(model.llm.get_base_model().config, "_commit_hash", None)
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[prefix_lm] {llm_name}: trainable {trained:,} / {total:,} parameters "
          f"(projector + LoRA r={lora_r}); {audio_pool}x pooled audio; "
          f"revision {model.config.llm_revision or 'local'}")
    return model, tokenizer


def load_captioner(path: str):
    """RECAP or AudioPrefixLM, from the model_type in path/config.json."""
    import json
    with open(os.path.join(path, "config.json")) as f:
        model_type = json.load(f).get("model_type")
    if model_type == AudioPrefixConfig.model_type:
        return AudioPrefixLM.from_pretrained(path)
    from model.recap import RECAP
    return RECAP.from_pretrained(path)
