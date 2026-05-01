"""
train.py

Training entry point for RECAP on AudioCaps.
Adapted from Sreyan88/RECAP/train.py for our model layout and AudioCaps data format.

Typical usage (baseline, no retrieval):
    python train.py --features_dir data/features/ --disable_rag

Typical usage (RAG mode):
    python train.py \
        --features_dir data/features/ \
        --captions_path data/retrieved_caps.json \
        --template_path data/template.txt \
        --k 3
"""

import os
import argparse

import numpy as np

os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoTokenizer, AutoFeatureExtractor
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM

from model.recap import RECAP, RECAPConfig
from model.gpt2_xattn import ThisGPT2Config, ThisGPT2LMHeadModel
from data.dataset import TrainDataset, load_data_for_training

# Maps desired cross-attention size (in millions of parameters) to the
# reduce factor passed to RECAP.from_encoder_decoder_pretrained().
# Higher factor = smaller/cheaper cross-attention.
PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}

PAD_TOKEN = "!"
EOS_TOKEN = "."


def get_model_and_auxiliaries(args):
    # Register custom config/model types so HF AutoModel can load them.
    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoConfig.register("recap", RECAPConfig)
    AutoModel.register(RECAPConfig, RECAP)

    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    model = RECAP.from_encoder_decoder_pretrained(
        args.encoder_name,
        args.decoder_name,
        cross_attention_reduce_factor=cross_attention_reduce_factor,
    )
    model.config.vocab_size = model.config.decoder.vocab_size
    # Use BOS as decoder_start_token_id so shift_tokens_right works if ever called
    # (e.g. eval paths where only `labels` is passed). Training itself provides
    # decoder_input_ids directly so this is just defensive.
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = 60
    model.config.rag = not args.disable_rag

    if not args.disable_rag:
        model.config.k = args.k
        model.config.retrieval_encoder = args.retrieval_encoder

    # Encoder is a pretrained CLAP model — keep it frozen throughout training.
    for param in model.encoder.parameters():
        param.requires_grad = False

    # By default only train the cross-attention layers added to GPT-2.
    # `ln_cross_attn` is the new LayerNorm that pairs with cross-attention;
    # it must also be trainable (it's randomly initialized alongside the
    # cross-attention projections). Pass --train_decoder to also fine-tune
    # the rest of the GPT-2 decoder.
    if not args.train_decoder:
        for name, param in model.decoder.named_parameters():
            if "crossattention" not in name and "ln_cross_attn" not in name:
                param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training a model with {n_trainable:,} trainable parameters.")

    return model, tokenizer, feature_extractor


def get_data(tokenizer, max_length, args):
    data = load_data_for_training(
        args.annotations_path,
        caps_path=None if args.disable_rag else args.captions_path,
    )

    train_dataset = TrainDataset(
        df=data["train"],
        features_path=os.path.join(args.features_dir, "audiocaps_train.hdf5"),
        tokenizer=tokenizer,
        rag=not args.disable_rag,
        template_path=args.template_path if not args.disable_rag else None,
        k=args.k if not args.disable_rag else None,
        max_caption_length=max_length,
    )
    return train_dataset


def main(args):
    model, tokenizer, feature_extractor = get_model_and_auxiliaries(args)
    train_dataset = get_data(tokenizer, model.config.max_length, args)

    model_type = "recap_rag" if not args.disable_rag else "recap"
    run_name = f"{model_type}_{args.attention_size}M_{args.decoder_name.replace('/', '_')}"
    output_dir = os.path.join(args.experiments_dir, run_name)

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate=args.lr,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=3,
        logging_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        tokenizer=feature_extractor,
    )

    trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RECAP AudioCaps Training")

    # Data paths
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/features/",
        help="Directory containing audiocaps_train.hdf5 (precomputed CLAP features)",
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="data/audiocaps_annotations",
        help="Directory containing train.csv and val.csv",
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments/",
        help="Root directory where model checkpoints are saved",
    )

    # Model
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="laion/clap-htsat-fused",
        help="HuggingFace model ID or local path for the CLAP audio encoder",
    )
    parser.add_argument(
        "--decoder_name",
        type=str,
        default="gpt2",
        help="HuggingFace model ID or local path for the GPT-2 decoder",
    )
    parser.add_argument(
        "--attention_size",
        type=float,
        default=7,
        help="Cross-attention parameter budget in millions: 28, 14, 7, 3.5, or 1.75",
    )
    parser.add_argument(
        "--train_decoder",
        action="store_true",
        default=False,
        help="Also fine-tune the full GPT-2 decoder (default: cross-attention layers only)",
    )

    # RAG
    parser.add_argument(
        "--disable_rag",
        action="store_true",
        default=False,
        help="Train without retrieval-augmented prefix (baseline mode)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of retrieved captions to prepend as context (RAG mode)",
    )
    parser.add_argument(
        "--retrieval_encoder",
        type=str,
        default="laion/clap-htsat-unfused",
        help="CLAP encoder used to build the caption retrieval index",
    )
    parser.add_argument(
        "--captions_path",
        type=str,
        default="data/retrieved_caps.json",
        help="JSON mapping str(audiocap_id) -> list[str] of retrieved captions (RAG mode)",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default="data/template.txt",
        help="TXT file with the RAG prompt template; use || as the caption placeholder",
    )

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from latest checkpoint in output_dir if present",
    )

    args = parser.parse_args()
    main(args)