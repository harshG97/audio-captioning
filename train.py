"""
train.py — Train RECAP on AudioCaps (or Clotho) using cached CLAP audio
features and a precomputed retrieval cache.

Encoder is frozen and we feed cached encoder hidden states directly via
`encoder_outputs`, so the CLAP audio tower is never re-run during training.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os

import h5py
import numpy as np
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from data.access_id import DATASETS
from data.recap_dataset import RecapCollator, RecapDataset
from model.build_recap import ATTENTION_SIZE_TO_REDUCE_FACTOR, build_recap


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
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument(
        "--prompt_dropout", type=float, default=0.0,
        help="Probability of replacing the retrieved-captions prompt with the "
             "baseline prompt during training. Forces the decoder to rely on "
             "cross-attention to the audio rather than copying from retrieved "
             "captions. 0.0 disables (default); 0.3-0.5 is a reasonable range.")

    # Cross-attention size: pick ONE of these (or neither for the default,
    # attention_size=7 / reduce_factor=4, as in the RECAP paper).
    attn = p.add_mutually_exclusive_group()
    attn.add_argument(
        "--cross_attention_reduce_factor", type=int, default=None,
        help="Bottleneck factor for cross-attention Q/K/V projections. "
             "Higher = smaller and cheaper cross-attention. Default: 4 "
             "(same as --attention_size 7).")
    attn.add_argument(
        "--attention_size", type=float, default=None,
        choices=list(ATTENTION_SIZE_TO_REDUCE_FACTOR.keys()),
        help="Cross-attention parameter budget in millions; maps to a reduce factor "
             "(28->1, 14->2, 7->4, 3.5->8, 1.75->16). Default: 7, which is "
             "~7.1M trainable params with GPT-2 small, matching the RECAP paper "
             "and its reference code.")

    p.add_argument(
        "--train_decoder", action="store_true",
        help="Fine-tune the entire GPT-2 decoder. Default: train cross-attention "
             "sublayers + ln_cross_attn only.")
    p.add_argument(
        "--xattn_zero_init", action="store_true",
        help="Zero the cross-attention output projections after loading so the "
             "decoder starts identical to plain GPT-2. Default: random init, as "
             "in the RECAP paper.")

    # Retrieval metadata recorded onto the saved checkpoint config so reloads
    # remember how it was trained. Strict: required when --rag is set.
    p.add_argument(
        "--rag", action=argparse.BooleanOptionalAction, default=True,
        help="Whether retrieval-augmented prefixes are in use. Pass --no-rag for "
             "the empty-cache baseline.")
    p.add_argument("--retrieval_k", type=int, default=None,
                   help="Number of retrieved captions per query (required with --rag).")
    p.add_argument("--retrieval_strategy", type=str, default=None,
                   choices=["topk", "mmr"],
                   help="Retrieval strategy used to build the cache (required with --rag).")
    p.add_argument("--retrieval_mmr_lambda", type=float, default=None,
                   help="MMR relevance/diversity tradeoff (required when "
                        "--retrieval_strategy mmr).")

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
    p.add_argument("--save_total_limit", type=int, default=None,
                   help="Maximum number of checkpoints to keep. Deletes older ones.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep dataloader workers alive across epochs to avoid "
                        "respawn overhead. Default: enabled.")
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="Number of batches each worker prefetches in advance. "
                        "Higher = more memory, less GPU starvation. Default: 4.")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument(
        "--load_best_model_at_end", action="store_true",
        help="At end of training, reload the checkpoint with the lowest "
             "eval_loss instead of keeping the last training step. Requires "
             "a validation set (--val_csv/--val_hdf5/--val_retrieval_cache).")
    p.add_argument(
        "--eval_save_strategy", choices=["steps", "epoch"], default="steps",
        help="When to evaluate and save checkpoints: every --eval_steps / "
             "--save_steps steps (default), or once per epoch.")
    p.add_argument(
        "--early_stopping_patience", type=int, default=0,
        help="Stop after this many evaluations without a lower eval_loss. "
             "0 disables (default). Needs --load_best_model_at_end.")
    p.add_argument(
        "--resume", action="store_true",
        help="If --output_dir already holds checkpoint-* folders, continue "
             "from the latest one (optimizer, scheduler, RNG and step count "
             "included). Otherwise start fresh. Safe to pass on every launch, "
             "so a resubmitted job picks up where it stopped.")
    args = p.parse_args()

    if args.early_stopping_patience > 0 and not args.load_best_model_at_end:
        p.error("--early_stopping_patience needs --load_best_model_at_end")

    # Resolve attention-size flags.
    if args.attention_size is not None:
        args.cross_attention_reduce_factor = ATTENTION_SIZE_TO_REDUCE_FACTOR[args.attention_size]
    elif args.cross_attention_reduce_factor is None:
        args.cross_attention_reduce_factor = ATTENTION_SIZE_TO_REDUCE_FACTOR[7.0]

    # Strict retrieval-metadata validation.
    if args.rag:
        missing = []
        if args.retrieval_k is None:
            missing.append("--retrieval_k")
        if args.retrieval_strategy is None:
            missing.append("--retrieval_strategy")
        if args.retrieval_strategy == "mmr" and args.retrieval_mmr_lambda is None:
            missing.append("--retrieval_mmr_lambda (required with --retrieval_strategy mmr)")
        if missing:
            p.error("--rag is set; the following are required: " + ", ".join(missing))

    return args


def checkpoint_safe_globals() -> list:
    """numpy types that transformers 4.45 pickles into checkpoint rng_state.pth.

    torch >= 2.6 loads checkpoints with weights_only=True by default and refuses
    these, so resuming fails with an UnpicklingError. Allowlisting exactly these
    types (only while resuming from our own checkpoints) keeps the safe loader.
    """
    core = getattr(np, "_core", None) or np.core
    return [core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]


def resume_checkpoint(output_dir: str) -> str | None:
    """Latest checkpoint-* folder in output_dir, or None to start fresh."""
    if not os.path.isdir(output_dir):
        return None
    return get_last_checkpoint(output_dir)


HISTORY_FIELDS = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]


def write_history(log_history: list[dict], path: str) -> int:
    """Write the Trainer's log history as a CSV, one row per logging step
    (training loss) or evaluation (eval_loss), for plotting loss curves.
    Returns the number of rows written."""
    rows = [e for e in log_history if "loss" in e or "eval_loss" in e]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def feature_width(hdf5_path: str) -> int:
    """Last dimension of the first clip's features in an HDF5 feature file."""
    with h5py.File(hdf5_path, "r") as f:
        first = next(iter(f.keys()))
        return int(f[first].shape[-1])


def check_feature_width(hdf5_path: str, decoder_width: int) -> None:
    """Cross-attention reads encoder features at the decoder's width, with no
    projection in between, so the two must match (768 for CLAP HTSAT-base and
    GPT-2 small)."""
    width = feature_width(hdf5_path)
    if width != decoder_width:
        raise SystemExit(
            f"{hdf5_path} holds {width}-wide features but the decoder is "
            f"{decoder_width} wide. Cross-attention has no projection between "
            f"them, so pick an encoder/decoder pair of the same width.")


def main() -> None:
    args = parse_args()

    model, tokenizer = build_recap(
        encoder_name=args.encoder_name,
        decoder_name=args.decoder_name,
        cross_attention_reduce_factor=args.cross_attention_reduce_factor,
        freeze_encoder=True,
        train_decoder=args.train_decoder,
        xattn_zero_init=args.xattn_zero_init,
    )

    decoder_width = model.decoder.config.hidden_size
    check_feature_width(args.train_hdf5, decoder_width)
    if args.val_hdf5:
        check_feature_width(args.val_hdf5, decoder_width)

    # Record retrieval metadata onto the config so the checkpoint remembers it.
    model.config.rag = args.rag
    if args.rag:
        model.config.retrieval_k = args.retrieval_k
        model.config.retrieval_strategy = args.retrieval_strategy
        if args.retrieval_strategy == "mmr":
            model.config.retrieval_mmr_lambda = args.retrieval_mmr_lambda

    train_ds = RecapDataset(
        csv_path=args.train_csv,
        hdf5_path=args.train_hdf5,
        retrieval_cache_path=args.train_retrieval_cache,
        tokenizer=tokenizer,
        dataset=args.dataset,
        max_length=args.max_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
        prompt_dropout=args.prompt_dropout,
        retrieval_k=args.retrieval_k if args.rag else None,
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
            retrieval_k=args.retrieval_k if args.rag else None,
        )

    collator = RecapCollator(pad_token_id=tokenizer.pad_token_id)

    if args.load_best_model_at_end and eval_ds is None:
        raise SystemExit(
            "--load_best_model_at_end requires --val_csv, --val_hdf5, and "
            "--val_retrieval_cache so eval can run during training."
        )

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
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        evaluation_strategy=args.eval_save_strategy if eval_ds is not None else "no",
        save_strategy=args.eval_save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss" if args.load_best_model_at_end else None,
        greater_is_better=False if args.load_best_model_at_end else None,
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.persistent_workers and args.num_workers > 0,
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
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
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
            if args.early_stopping_patience > 0 else None
        ),
    )

    resume_from = resume_checkpoint(args.output_dir) if args.resume else None
    if resume_from:
        print(f"[resume] continuing from {resume_from}")
        safe = getattr(torch.serialization, "safe_globals", None)
        ctx = safe(checkpoint_safe_globals()) if safe else contextlib.nullcontext()
        with ctx:
            trainer.train(resume_from_checkpoint=resume_from)
    else:
        if args.resume:
            print(f"[resume] no checkpoint in {args.output_dir}; starting fresh")
        trainer.train()

    # Keep the full loss history next to the final model (checkpoints get
    # pruned by --save_total_limit). It survives --resume: the Trainer restores
    # earlier entries from the checkpoint's trainer_state.json.
    trainer.save_state()
    n = write_history(trainer.state.log_history, os.path.join(args.output_dir, "history.csv"))
    print(f"[history] {n} rows -> {os.path.join(args.output_dir, 'history.csv')}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
