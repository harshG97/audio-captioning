"""
finetune.py — Stage-2 fine-tuning of a RECAP checkpoint.

Loads a RECAP model that was trained with cross-attention only (e.g. from
`train.py`), unfreezes the GPT-2 decoder, and continues training with two
learning rates: one for the cross-attention sublayers and a smaller one
for the GPT-2 backbone (incl. wte / lm_head). The encoder stays frozen
and is fed via cached `encoder_outputs`, same as `train.py`.

Architecture and retrieval metadata are inherited from the checkpoint
config; passing architecture flags (encoder/decoder/xattn size) or
retrieval flags here is a hard error.
"""

from __future__ import annotations

import argparse

import torch
from transformers import Trainer, TrainingArguments

from data.access_id import DATASETS
from data.recap_dataset import RecapCollator, RecapDataset
from model.build_recap import (
    apply_freeze_policy,
    build_param_groups,
    load_recap_checkpoint,
    wire_special_tokens,
)


class _RaiseOnUse(argparse.Action):
    """Argparse action that errors out if the flag is supplied at all."""
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            f"{option_string} is not accepted by finetune.py; this value is "
            f"inherited from the pretrained checkpoint. Use train.py to change it."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-2 fine-tune RECAP: unfreeze GPT-2 + dual LR.",
    )
    p.add_argument("--pretrained_checkpoint", type=str, required=True,
                   help="Path to a RECAP checkpoint directory written by train.py.")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--train_hdf5", type=str, required=True)
    p.add_argument("--train_retrieval_cache", type=str, required=True)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--val_hdf5", type=str, default=None)
    p.add_argument("--val_retrieval_cache", type=str, default=None)
    p.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)

    p.add_argument("--max_length", type=int, default=128)
    p.add_argument(
        "--prompt_dropout", type=float, default=0.3,
        help="Probability of replacing the retrieved-captions prompt with the "
             "baseline prompt during training. Higher than train.py's default "
             "because the unfrozen decoder can otherwise re-learn to copy "
             "from retrieved captions.")

    # Optimization
    p.add_argument("--learning_rate", type=float, default=5e-5,
                   help="LR for cross-attention sublayers (xattn group).")
    p.add_argument("--decoder_lr", type=float, default=5e-6,
                   help="LR for GPT-2 backbone params (incl. wte, lm_head). "
                        "Typically 10x lower than --learning_rate.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--num_train_epochs", type=float, default=4.0)

    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument(
        "--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True,
        help="When a val set is provided, reload the lowest-eval_loss checkpoint "
             "at end of training. Default: enabled. Pass --no-load_best_model_at_end "
             "to keep the last-step weights instead.")

    # Hard-fail flags: anything below would silently relabel/reshape the
    # checkpoint, which is almost always a mistake.
    forbidden = p.add_argument_group(
        "forbidden (inherited from checkpoint)",
        "These flags are NOT accepted by finetune.py. The architecture and "
        "retrieval metadata come from the pretrained checkpoint. Re-running "
        "train.py is the right tool if any of these need to change.",
    )
    for flag in (
        "--encoder_name", "--decoder_name",
        "--cross_attention_reduce_factor", "--attention_size",
        "--rag", "--no-rag",
        "--retrieval_k", "--retrieval_strategy", "--retrieval_mmr_lambda",
    ):
        forbidden.add_argument(flag, action=_RaiseOnUse, nargs="?", default=argparse.SUPPRESS)

    return p.parse_args()


def _xattn_weight_norm(model) -> float:
    """L2 norm of the first decoder block's cross-attn output projection.
    Trained checkpoints have non-zero values; fresh-init has c_proj=0
    (see commit 1315b32). Used as a sanity check that weights loaded."""
    block = model.decoder.transformer.h[0]
    return float(block.crossattention.c_proj.weight.detach().norm().item())


def main() -> None:
    args = parse_args()

    print(f"[finetune] loading checkpoint: {args.pretrained_checkpoint}")
    model, tokenizer = load_recap_checkpoint(args.pretrained_checkpoint)
    wire_special_tokens(model, tokenizer)
    apply_freeze_policy(model, freeze_encoder=True, train_decoder=True)

    # Sanity: confirm cross-attn weights actually loaded (non-zero) and
    # surface the resolved special-token IDs from the new EOS/pad changes.
    xattn_norm = _xattn_weight_norm(model)
    print(f"[finetune] cross-attn c_proj norm (layer 0): {xattn_norm:.4f} "
          f"({'OK' if xattn_norm > 1e-4 else 'WARNING: looks zero-init'})")
    print(f"[finetune] pad_token_id={model.config.pad_token_id} "
          f"eos_token_id={model.config.eos_token_id} "
          f"decoder_start_token_id={model.config.decoder_start_token_id}")
    rag = getattr(model.config, "rag", None)
    print(f"[finetune] inherited retrieval metadata: rag={rag}, "
          f"k={getattr(model.config, 'retrieval_k', None)}, "
          f"strategy={getattr(model.config, 'retrieval_strategy', None)}")

    train_ds = RecapDataset(
        csv_path=args.train_csv,
        hdf5_path=args.train_hdf5,
        retrieval_cache_path=args.train_retrieval_cache,
        tokenizer=tokenizer,
        dataset=args.dataset,
        max_length=args.max_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
        prompt_dropout=args.prompt_dropout,
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

    load_best = args.load_best_model_at_end and eval_ds is not None
    if args.load_best_model_at_end and eval_ds is None:
        print("[finetune] --load_best_model_at_end set but no val set provided; "
              "ignoring (last-step weights will be saved).")

    # Custom optimizer with two LR groups (xattn vs GPT-2 backbone).
    # Trainer builds the LR scheduler over this optimizer.
    param_groups = build_param_groups(
        model,
        lr_xattn=args.learning_rate,
        lr_decoder=args.decoder_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,    # used only by the auto-built scheduler floor; per-group LRs come from the optimizer.
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
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
        optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
