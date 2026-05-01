"""
train.py — Train RECAP on AudioCaps (or Clotho) using cached CLAP audio
features and a precomputed retrieval cache.

Encoder is frozen and we feed cached encoder hidden states directly via
`encoder_outputs`, so the CLAP audio tower is never re-run during training.
"""

from __future__ import annotations

import argparse

import torch
from transformers import Trainer, TrainingArguments

from data.access_id import DATASETS
from data.recap_dataset import RecapCollator, RecapDataset
from model.build_recap import build_recap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RECAP.")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--train_hdf5", type=str, required=True)
    p.add_argument("--train_retrieval_cache", type=str, required=True)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--val_hdf5", type=str, default=None)
    p.add_argument("--val_retrieval_cache", type=str, default=None)
    p.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--encoder_name", type=str, default="laion/clap-htsat-fused")
    p.add_argument("--decoder_name", type=str, default="gpt2")
    p.add_argument("--cross_attention_reduce_factor", type=int, default=1)
    p.add_argument("--max_length", type=int, default=128)

    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_train_epochs", type=float, default=10.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model, tokenizer = build_recap(
        encoder_name=args.encoder_name,
        decoder_name=args.decoder_name,
        cross_attention_reduce_factor=args.cross_attention_reduce_factor,
        freeze_encoder=True,
    )

    train_ds = RecapDataset(
        csv_path=args.train_csv,
        hdf5_path=args.train_hdf5,
        retrieval_cache_path=args.train_retrieval_cache,
        tokenizer=tokenizer,
        dataset=args.dataset,
        max_length=args.max_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    eval_ds = None
    if args.val_csv and args.val_hdf5 and args.val_retrieval_cache:
        eval_ds = RecapDataset(
            csv_path=args.val_csv,
            hdf5_path=args.val_hdf5,
            retrieval_cache_path=args.val_retrieval_cache,
            tokenizer=tokenizer,
            dataset=args.dataset,
            max_length=args.max_length,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )

    collator = RecapCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
