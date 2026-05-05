"""
recap.py

Top-level encoder-decoder for RECAP.

Combines:
  - An audio encoder (CLAP's audio_model), loaded via AutoModel.
  - A GPT-2 decoder with cross-attention (ThisGPT2LMHeadModel).

The decoder attends to the encoder's hidden states via the cross-attention
layers.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import (
    VisionEncoderDecoderConfig,
)
from transformers.utils import logging
from transformers import AutoConfig, AutoModelForCausalLM

from model.gpt2_xattn import ThisGPT2Config, ThisGPT2LMHeadModel

# Register custom model types with Hugging Face registries
AutoConfig.register("this_gpt2", ThisGPT2Config)
AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)

logger = logging.get_logger(__name__)

def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
) -> torch.Tensor:
    """
    Shift `input_ids` one position to the right and prepend `decoder_start_token_id`.
    Used to build decoder inputs from labels during training.
    """
    if decoder_start_token_id is None:
        raise ValueError(
            "Set `decoder_start_token_id` on the model config before training."
        )
    if pad_token_id is None:
        raise ValueError("Set `pad_token_id` on the model config before training.")

    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id

    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted

class RECAPConfig(VisionEncoderDecoderConfig):
    """
    Config wrapper. Inherits VisionEncoderDecoderConfig so we get
    from_encoder_decoder_configs(...) for free.
    """
    model_type = "recap"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class RECAP(PreTrainedModel):
    """
    Audio-Encoder + Text-Decoder model for retrieval-augmented audio captioning.

    The encoder is the audio_model from a CLAP checkpoint; the decoder is
    GPT-2 with cross-attention to the encoder's hidden states.
    """
    config_class = RECAPConfig
    base_model_prefix = "recap"
    main_input_name = "pixel_values"  # CLAP-style audio inputs come in as `pixel_values`

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError(
                "Provide either `config` or both `encoder` and `decoder`."
            )
        if config is None:
            config = RECAPConfig.from_encoder_decoder_configs(
                encoder.config, decoder.config
            )
        elif not isinstance(config, self.config_class):
            raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # Sanity check: if the decoder declares a cross_attention_hidden_size,
        # it must match the encoder's hidden_size.
        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    f"`config.decoder.cross_attention_hidden_size` "
                    f"({config.decoder.cross_attention_hidden_size}) must match "
                    f"`config.encoder.hidden_size` ({config.encoder.hidden_size})."
                )

        # Don't tie encoder and decoder embeddings — they live in different vocabularies.
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)
        if decoder is None:
            decoder = ThisGPT2LMHeadModel(config.decoder)

        # CLAP wraps audio + text models; we only need the audio side.
        # If `encoder` is already an audio_model (no .audio_model attr), use it directly.
        self.encoder = encoder.audio_model if hasattr(encoder, "audio_model") else encoder
        self.encoder.main_input_name = "pixel_values"
        self.decoder = decoder

        # Keep the wrapped configs in sync with the parent config.
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

    def tie_weights(self):
        # Delegate to the decoder so GPT-2's lm_head ↔ wte tie is
        # re-established after from_pretrained loads the state dict.
        # (RECAP sets tie_word_embeddings=False on the *parent* config,
        # which would otherwise skip re-tying.)
        self.decoder.tie_weights()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Composite models don't support fast init.
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        cross_attention_reduce_factor: int = 1,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        Build a RECAP from pretrained encoder and decoder checkpoints.

        Encoder kwargs use the `encoder_` prefix, decoder kwargs use `decoder_`.
        Anything else flows into the parent RECAPConfig.

        Args:
            encoder_pretrained_model_name_or_path:
                HF model id or local path for the audio encoder (e.g. a CLAP checkpoint).
            decoder_pretrained_model_name_or_path:
                HF model id or local path for the GPT-2 decoder.
            cross_attention_reduce_factor:
                Bottleneck factor for cross-attention's Q/K/V projections.
                1 = no reduction. Higher = smaller (cheaper) cross-attention.
        """
        # Split kwargs by prefix
        kwargs_encoder = {
            k[len("encoder_"):]: v
            for k, v in kwargs.items()
            if k.startswith("encoder_")
        }
        kwargs_decoder = {
            k[len("decoder_"):]: v
            for k, v in kwargs.items()
            if k.startswith("decoder_")
        }
        for k in list(kwargs_encoder.keys()):
            del kwargs["encoder_" + k]
        for k in list(kwargs_decoder.keys()):
            del kwargs["decoder_" + k]

        # Encoder
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "Provide `encoder_model` or `encoder_pretrained_model_name_or_path`."
                )
            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path,
                    **kwargs_encoder,
                    return_unused_kwargs=True,
                )
                # The encoder must NOT be set up as a decoder.
                if encoder_config.is_decoder or encoder_config.add_cross_attention:
                    logger.info(
                        f"Disabling decoder/cross-attention flags on "
                        f"{encoder_pretrained_model_name_or_path} (encoder role)."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )

        # Decoder (GPT-2 only)
        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "Provide `decoder_model` or `decoder_pretrained_model_name_or_path`."
                )
            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = ThisGPT2Config.from_pretrained(
                    decoder_pretrained_model_name_or_path,
                    **kwargs_decoder,
                    return_unused_kwargs=True,
                )

                # Set decoder + cross-attention flags so HF wires things up correctly.
                if not decoder_config.is_decoder or not decoder_config.add_cross_attention:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a "
                        f"decoder with cross-attention layers (randomly initialized)."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                # Tell the decoder how big the encoder hidden states will be.
                # CLAP nests the audio config under `audio_config`; fall back to
                # `hidden_size` for plain encoders.
                enc_cfg = encoder.config
                if hasattr(enc_cfg, "audio_config"):
                    decoder_config.encoder_hidden_size = enc_cfg.audio_config.hidden_size
                else:
                    decoder_config.encoder_hidden_size = enc_cfg.hidden_size

                decoder_config.cross_attention_reduce_factor = cross_attention_reduce_factor
                kwargs_decoder["config"] = decoder_config

            decoder = ThisGPT2LMHeadModel.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder
            )

        # Assemble
        config = RECAPConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config, **kwargs
        )
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[tuple] = None,
        past_key_values: Optional[tuple] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Run audio through the encoder, then condition the decoder on the
        encoder's hidden states via cross-attention.

        Loss is computed externally (not inside the decoder) so we don't
        double-shift labels.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Split kwargs by prefix for forwarding to encoder/decoder.
        kwargs_encoder = {
            k: v for k, v in kwargs.items() if not k.startswith("decoder_")
        }
        kwargs_decoder = {
            k[len("decoder_"):]: v
            for k, v in kwargs.items()
            if k.startswith("decoder_")
        }

        # Encoder
        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("Provide `pixel_values` or precomputed `encoder_outputs`.")
            encoder_outputs = self.encoder(
                input_features=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)
        else:
            encoder_outputs = BaseModelOutput(encoder_outputs, None)

        encoder_hidden_states = encoder_outputs[0]
        encoder_attention_mask = None  # CLAP audio outputs don't carry a padding mask

        # Build decoder inputs from labels if needed
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Loss (computed here, not inside the decoder)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.reshape(-1, self.decoder.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     past=None,
    #     attention_mask=None,
    #     use_cache=None,
    #     encoder_outputs=None,
    #     **kwargs,
    # ):
    #     decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
    #     decoder_attention_mask = decoder_inputs.get("attention_mask")
    #     return {
    #         "attention_mask": attention_mask,
    #         "decoder_attention_mask": decoder_attention_mask,
    #         "decoder_input_ids": decoder_inputs["input_ids"],
    #         "encoder_outputs": encoder_outputs,
    #         "past_key_values": decoder_inputs["past_key_values"],
    #         "use_cache": use_cache,
    #     }

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # Fallback for older HF calling conventions
        if past_key_values is None:
            past_key_values = kwargs.get("past", None)

        decoder_inputs = self.decoder.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )
        decoder_attention_mask = decoder_inputs.get("attention_mask")
        
        return {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resize through the decoder directly: "
            "model.decoder.resize_token_embeddings(...)"
        )

    def _reorder_cache(self, past, beam_idx):
        return self.decoder._reorder_cache(past, beam_idx)