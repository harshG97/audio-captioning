"""
train.py — Train RECAP on AudioCaps (or Clotho) using cached CLAP audio
features and a precomputed retrieval cache.

Encoder is frozen and we feed cached encoder hidden states directly via
`encoder_outputs`, so the CLAP audio tower is never re-run during training.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import csv
import json
import os
import re
import shutil

import h5py
import numpy as np
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed

from data.access_id import DATASETS
from data.recap_dataset import RecapCollator, RecapDataset
from model.build_recap import ATTENTION_SIZE_TO_REDUCE_FACTOR, build_recap
from model.prefix_lm import AUDIO_MERGES, LORA_TARGET_SETS, AudioPrefixLM, build_prefix_lm


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

    p.add_argument("--arch", choices=["xattn", "prefix"], default="xattn",
                   help="xattn (default): RECAP, GPT-2 with cross-attention to the audio. "
                        "prefix: a decoder-only LLM (--decoder_name, e.g. Qwen/Qwen2.5-1.5B) "
                        "reading the audio as prefix tokens, trained with LoRA "
                        "(model/prefix_lm.py).")
    p.add_argument("--encoder_name", type=str, default="laion/clap-htsat-fused")
    p.add_argument("--decoder_name", type=str, default="gpt2")
    p.add_argument("--audio_pool", type=int, default=4,
                   help="prefix: average this many feature vectors per audio token (default 4).")
    p.add_argument("--audio_merge", choices=list(AUDIO_MERGES), default="mean",
                   help="prefix: how each group of --audio_pool vectors becomes one audio token: "
                        "mean (default) averages them, concat places them side by side.")
    p.add_argument("--lora_r", type=int, default=16, help="prefix: LoRA rank (default 16).")
    p.add_argument("--lora_alpha", type=int, default=None,
                   help="prefix: LoRA alpha (default: twice the rank).")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="prefix: LoRA dropout (default 0.05).")
    p.add_argument("--lora_targets", choices=list(LORA_TARGET_SETS), default="all",
                   help="prefix: LLM layers that get LoRA adapters: all (default; attention and "
                        "MLP) or attn (attention only).")
    p.add_argument("--projector_lr", type=float, default=None,
                   help="prefix: learning rate of the audio projector, which starts from "
                        "scratch (default: --learning_rate, which then applies to LoRA only).")
    p.add_argument("--lora_warmup_epochs", type=float, default=0.0,
                   help="prefix: keep LoRA's learning rate at 0 for this many epochs, so only "
                        "the projector trains at first (default 0).")
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
        "--snapshot_every", type=int, default=0,
        help="Also keep every N-th epoch's model in OUTPUT_DIR/snapshots/epochNNN, "
             "safe from --save_total_limit, for evaluate.py. Model and tokenizer "
             "only, unless the epoch is also a --checkpoint_every epoch. "
             "0 disables (default). Needs --eval_save_strategy epoch.")
    p.add_argument(
        "--checkpoint_every", type=int, default=0,
        help="Keep every N-th epoch as a full checkpoint (optimizer, scheduler "
             "and RNG state too) in OUTPUT_DIR/snapshots/epochNNN, so "
             "--resume_from can continue from it. 0 disables (default). "
             "Needs --eval_save_strategy epoch.")
    p.add_argument(
        "--resume", action="store_true",
        help="If --output_dir already holds checkpoint-* folders, continue "
             "from the latest one (optimizer, scheduler, RNG and step count "
             "included). Otherwise start fresh. Safe to pass on every launch, "
             "so a resubmitted job picks up where it stopped.")
    p.add_argument(
        "--resume_from", type=str, default=None,
        help="Start from this checkpoint folder (another run's checkpoint-N or "
             "snapshots/epochNNN|best), with its optimizer, scheduler and step "
             "count. Once OUTPUT_DIR has checkpoints of its own, --resume "
             "continues from those instead.")
    p.add_argument(
        "--track_best", action="store_true",
        help="Track the checkpoint with the lowest eval_loss without reloading "
             "it at the end, and copy it to OUTPUT_DIR/snapshots/best. For runs "
             "that keep the last epoch.")
    args = p.parse_args()

    if args.early_stopping_patience > 0 and not args.load_best_model_at_end:
        p.error("--early_stopping_patience needs --load_best_model_at_end")
    for flag in ("snapshot_every", "checkpoint_every"):
        if getattr(args, flag) < 0:
            p.error(f"--{flag} must be 0 or more")
        if getattr(args, flag) > 0 and args.eval_save_strategy != "epoch":
            p.error(f"--{flag} needs --eval_save_strategy epoch")
    if args.resume_from and not is_full_checkpoint(args.resume_from):
        p.error(f"--resume_from {args.resume_from} is not a full checkpoint "
                f"(needs {', '.join(FULL_CHECKPOINT_FILES)}); snapshots between "
                f"--checkpoint_every epochs hold the model only")
    if args.track_best and not args.val_csv:
        p.error("--track_best needs a validation set (--val_csv)")

    # Resolve attention-size flags.
    if args.arch == "prefix" and (args.attention_size is not None or args.cross_attention_reduce_factor
                                  is not None or args.train_decoder or args.xattn_zero_init):
        p.error("--arch prefix has no cross-attention and trains LoRA, not the decoder: "
                "drop --attention_size / --cross_attention_reduce_factor / --train_decoder / --xattn_zero_init")
    prefix_only = {"--audio_pool": args.audio_pool != 4, "--audio_merge": args.audio_merge != "mean",
                   "--lora_r": args.lora_r != 16, "--lora_alpha": args.lora_alpha is not None,
                   "--lora_dropout": args.lora_dropout != 0.05, "--lora_targets": args.lora_targets != "all",
                   "--projector_lr": args.projector_lr is not None,
                   "--lora_warmup_epochs": args.lora_warmup_epochs != 0}
    if args.arch != "prefix" and any(prefix_only.values()):
        p.error(f"{', '.join(k for k, v in prefix_only.items() if v)} only apply to --arch prefix")
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    if args.lora_warmup_epochs < 0 or args.lora_warmup_epochs >= args.num_train_epochs:
        p.error("--lora_warmup_epochs must be 0 or more and below --num_train_epochs")
    if args.projector_lr is not None and args.projector_lr <= 0:
        p.error("--projector_lr must be positive")
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
    """Newest complete checkpoint-* folder in output_dir, or None to start
    fresh. A checkpoint left half-written (the job was killed while saving)
    is skipped with a message, so the run resumes from the one before."""
    if not os.path.isdir(output_dir):
        return None
    found = []
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m and os.path.isdir(os.path.join(output_dir, name)):
            found.append((int(m.group(1)), name))
    for _, name in sorted(found, reverse=True):
        path = os.path.join(output_dir, name)
        if is_full_checkpoint(path):
            return path
        print(f"[resume] skipping incomplete {path}")
    return None


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


def _own_generator(loader, seed: int):
    """Give a (prepared) DataLoader a generator of its own. Starting its
    workers draws a seed from loader.generator, or from the global RNG when it
    has none; drawing from the global RNG would shift the dropout stream of
    training depending on when the loader is created (a resumed run creates
    its loaders at a different point than an uninterrupted one).
    accelerate's wrapper iterates an inner base_dataloader, so set both."""
    inner = getattr(loader, "base_dataloader", loader)
    for dl in {id(loader): loader, id(inner): inner}.values():
        dl.generator = torch.Generator().manual_seed(seed)
    return loader


def linear_schedule(step: int, start: int, ramp: int, total: int) -> float:
    """Learning-rate factor of the Trainer's default schedule (linear warm-up
    over `ramp` steps, then linear decay to 0 at `total`), delayed so that it
    is 0 until step `start`."""
    if step < start:
        return 0.0
    if step < start + ramp:
        return (step - start) / max(1, ramp)
    return max(0.0, (total - step) / max(1, total - start - ramp))


class RecapTrainer(Trainer):
    """Trainer that starts the evaluation dataloader's workers once and keeps
    dataloader worker seeding out of the global RNG.

    With persistent workers, transformers 4.45 caches the evaluation
    DataLoader but passes it through accelerator.prepare() at every
    evaluation, which builds a new loader with a new set of worker processes
    while the accelerator keeps the old ones alive. Over a long run that is
    num_workers more processes per evaluation (about 570 after 70 epochs with
    8 workers), until the machine runs out of memory. Reusing the prepared
    loader keeps one set of workers for the whole run.

    For the prefix model it can also give the audio projector its own
    learning rate (projector_lr) and hold LoRA's learning rate at 0 for the
    first lora_warmup_epochs, so the projector learns useful audio tokens
    before the LLM starts to adapt. Both live in the optimizer's parameter
    groups and the scheduler, so a resumed run continues them exactly."""

    def __init__(self, *args, projector_lr=None, lora_warmup_epochs=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.projector_lr = projector_lr
        self.lora_warmup_epochs = lora_warmup_epochs

    def _split_rates(self) -> bool:
        return self.projector_lr is not None or self.lora_warmup_epochs > 0

    def create_optimizer(self):
        if not self._split_rates() or self.optimizer is not None:
            return super().create_optimizer()
        decay = set(self.get_decay_parameter_names(self.model))
        groups = []          # each tagged with its part, which create_scheduler reads
        for part, lr in (("projector", self.projector_lr or self.args.learning_rate),
                         ("lora", self.args.learning_rate)):
            for decays in (True, False):
                params = [p for n, p in self.model.named_parameters() if p.requires_grad
                          and n.startswith("projector.") == (part == "projector") and (n in decay) == decays]
                if params:
                    groups.append({"params": params, "lr": lr, "part": part,
                                   "weight_decay": self.args.weight_decay if decays else 0.0})
        n_grouped = sum(len(g["params"]) for g in groups)
        assert n_grouped == sum(p.requires_grad for p in self.model.parameters())
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)     # the Trainer's own optimizer
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lora_warmup_epochs <= 0 or self.lr_scheduler is not None:
            return super().create_scheduler(num_training_steps, optimizer)
        optimizer = self.optimizer if optimizer is None else optimizer
        ramp = self.args.get_warmup_steps(num_training_steps)
        hold = round(self.lora_warmup_epochs * num_training_steps / self.args.num_train_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [
            functools.partial(linear_schedule, start=hold if g["part"] == "lora" else 0,
                              ramp=ramp, total=num_training_steps)
            for g in optimizer.param_groups])
        self._created_lr_scheduler = True
        return self.lr_scheduler

    def get_eval_dataloader(self, eval_dataset=None):
        # evaluate() passes self.eval_dataset itself, not None.
        if eval_dataset is None or eval_dataset is self.eval_dataset:
            key = "eval"
        elif isinstance(eval_dataset, str):
            key = eval_dataset
        else:
            key = None                      # a one-off dataset: no caching
        if not self.args.dataloader_persistent_workers or key is None:
            return super().get_eval_dataloader(eval_dataset)
        prepared = self.__dict__.setdefault("_prepared_eval_dataloaders", {})
        if key not in prepared:
            prepared[key] = _own_generator(super().get_eval_dataloader(eval_dataset), self.args.seed)
        return prepared[key]

    def get_train_dataloader(self):
        # Data order comes from the sampler, which the Trainer restores on
        # resume; only the workers' seed is moved off the global RNG here.
        return _own_generator(super().get_train_dataloader(), self.args.seed + 1)


FULL_CHECKPOINT_FILES = ("model weights", "optimizer.pt", "scheduler.pt", "trainer_state.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin",
                "model.safetensors.index.json", "pytorch_model.bin.index.json")


def is_full_checkpoint(path: str) -> bool:
    """True if PATH holds what the Trainer needs to resume training exactly:
    model weights, optimizer and scheduler state, and a readable
    trainer_state.json. The Trainer writes trainer_state.json last, so a
    checkpoint whose save was interrupted fails this check."""
    def has(name):
        return os.path.isfile(os.path.join(path, name))
    if not (any(has(w) for w in WEIGHT_FILES) and has("optimizer.pt") and has("scheduler.pt")):
        return False
    try:
        with open(os.path.join(path, "trainer_state.json")) as f:
            return "global_step" in json.load(f)
    except (OSError, ValueError, TypeError):
        return False


class SnapshotCallback(TrainerCallback):
    """Copies the checkpoint saved at chosen epochs to
    OUTPUT_DIR/snapshots/epochNNN so --save_total_limit cannot prune it:
    every `every`-th epoch with the model and tokenizer only (for
    evaluate.py), and every `full_every`-th epoch as a full checkpoint that
    --resume_from can continue from. 0 turns either off."""

    MODEL_ONLY_SKIP = ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state*.pth")

    def __init__(self, every: int = 0, full_every: int = 0):
        self.every = every
        self.full_every = full_every

    def on_save(self, args, state, control, **kwargs):
        epoch = round(state.epoch or 0)
        if not state.is_world_process_zero or epoch == 0:
            return
        full = self.full_every > 0 and epoch % self.full_every == 0
        if not full and not (self.every > 0 and epoch % self.every == 0):
            return
        src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        dst = os.path.join(args.output_dir, "snapshots", f"epoch{epoch:03d}")
        if os.path.isdir(src) and not os.path.exists(dst):
            ignore = None if full else shutil.ignore_patterns(*self.MODEL_ONLY_SKIP)
            shutil.copytree(src, dst, ignore=ignore)
            print(f"[snapshot] epoch {epoch} -> {dst} ({'full checkpoint' if full else 'model only'})")


def feature_width(hdf5_path: str) -> int:
    """Last dimension of the first clip's features in an HDF5 feature file."""
    with h5py.File(hdf5_path, "r") as f:
        first = next(iter(f.keys()))
        return int(f[first].shape[-1])


def model_input_width(model) -> int:
    """Width of the encoder features the model reads: the projection's input
    when the encoder and decoder widths differ, otherwise the decoder's (for
    the prefix model, the projector's input)."""
    if isinstance(model, AudioPrefixLM):
        return model.config.feature_width
    proj = getattr(model, "enc_to_dec_proj", None)
    return proj.in_features if proj is not None else model.decoder.config.hidden_size


def check_feature_width(hdf5_path: str, expected: int) -> None:
    """The HDF5 features must have the width of the encoder the model was
    built with (768 for CLAP HTSAT-base); a mismatch means they were made
    with a different CLAP than --encoder_name."""
    width = feature_width(hdf5_path)
    if width != expected:
        raise SystemExit(
            f"{hdf5_path} holds {width}-wide features but the model reads "
            f"{expected}-wide encoder features. Were they made with a different "
            f"CLAP than --encoder_name?")


def main() -> None:
    args = parse_args()
    # Seed before building the model: the cross-attention layers are randomly
    # initialised there, and the Trainer only seeds when it is created.
    set_seed(args.seed)

    if args.arch == "prefix":
        model, tokenizer = build_prefix_lm(
            llm_name=args.decoder_name,
            feature_width=feature_width(args.train_hdf5),
            audio_pool=args.audio_pool,
            audio_merge=args.audio_merge,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_targets=args.lora_targets,
        )
    else:
        model, tokenizer = build_recap(
            encoder_name=args.encoder_name,
            decoder_name=args.decoder_name,
            cross_attention_reduce_factor=args.cross_attention_reduce_factor,
            freeze_encoder=True,
            train_decoder=args.train_decoder,
            xattn_zero_init=args.xattn_zero_init,
        )

    input_width = model_input_width(model)
    check_feature_width(args.train_hdf5, input_width)
    if args.val_hdf5:
        check_feature_width(args.val_hdf5, input_width)

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
        metric_for_best_model="eval_loss" if args.load_best_model_at_end or args.track_best else None,
        greater_is_better=False if args.load_best_model_at_end or args.track_best else None,
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

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    if args.snapshot_every > 0 or args.checkpoint_every > 0:
        callbacks.append(SnapshotCallback(args.snapshot_every, args.checkpoint_every))

    trainer = RecapTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks or None,
        projector_lr=args.projector_lr,
        lora_warmup_epochs=args.lora_warmup_epochs,
    )

    resume_from = resume_checkpoint(args.output_dir) if args.resume else None
    if resume_from is None and args.resume_from:
        resume_from = args.resume_from
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
    if args.track_best:
        keep_best_checkpoint(trainer.state.best_model_checkpoint, args.output_dir)


def keep_best_checkpoint(best: str | None, output_dir: str) -> str | None:
    """Copy the lowest-eval_loss checkpoint to OUTPUT_DIR/snapshots/best,
    replacing an older copy (a resumed run may have found a better one)."""
    if not best or not os.path.isdir(best):
        print(f"[best] no best checkpoint to keep ({best})")
        return None
    dst = os.path.join(output_dir, "snapshots", "best")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(best, dst)
    print(f"[best] {best} -> {dst}")
    return dst


if __name__ == "__main__":
    main()
