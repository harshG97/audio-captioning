"""
ablate_audio.py — Diagnostic for "is the model actually using the audio?"

Computes teacher-forced cross-entropy loss on a held-out split under three
conditions:
  intact  : encoder_hidden_states as produced by the encoder.
  zeroed  : encoder_hidden_states replaced with zeros (same shape).
  shuffled: encoder_hidden_states permuted across the batch (each example
            gets some other example's audio).

If loss(zeroed) ~= loss(intact), the decoder is ignoring audio entirely.
If loss(shuffled) ~= loss(intact), audio info is treated as ~noise.

Usage:
    python scripts/ablate_audio.py \
        --checkpoint $RUNS_DIR/audiocaps-recap-topk \
        --csv_path data/val.csv \
        --hdf5_path $FEAT_DIR/audiocaps_val.hdf5 \
        --retrieval_cache $FEAT_DIR/retrieved_topk_val.json \
        --batch_size 16
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# Make repo modules importable when invoked as `python scripts/ablate_audio.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.access_id import DATASETS
from data.recap_dataset import RecapCollator, RecapDataset
from model.device import best_device
from model.recap import RECAP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--hdf5_path", type=str, required=True)
    p.add_argument("--retrieval_cache", type=str, required=True)
    p.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_batches", type=int, default=0,
                   help="Cap number of batches (0 = full split).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _shuffle_batch(t: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permute tensor along dim 0 with no fixed points (when batch >= 2)."""
    bsz = t.shape[0]
    if bsz < 2:
        return t.clone()
    perm = torch.randperm(bsz, generator=generator, device="cpu")
    # Ensure no fixed points (each example gets someone else's audio).
    fixed = (perm == torch.arange(bsz)).nonzero(as_tuple=True)[0]
    for i in fixed.tolist():
        j = (i + 1) % bsz
        perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    return t[perm.to(t.device)]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else best_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = RECAP.from_pretrained(args.checkpoint).to(device)
    model.eval()

    dataset = RecapDataset(
        csv_path=args.csv_path,
        hdf5_path=args.hdf5_path,
        retrieval_cache_path=args.retrieval_cache,
        tokenizer=tokenizer,
        dataset=args.dataset,
        max_length=args.max_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    collator = RecapCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    print(f"val examples: {len(dataset)}  batches: {len(loader)}")

    gen = torch.Generator().manual_seed(args.seed)

    # Token-weighted loss accumulators (sum of per-token NLL / total tokens).
    totals = {"intact": 0.0, "zeroed": 0.0, "shuffled": 0.0}
    token_counts = {"intact": 0, "zeroed": 0, "shuffled": 0}

    n_batches = 0
    for batch in tqdm(loader, desc="ablate"):
        if args.max_batches and n_batches >= args.max_batches:
            break
        n_batches += 1

        encoder_hidden = batch["encoder_outputs"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        n_tok = (labels != -100).sum().item()

        variants = {
            "intact": encoder_hidden,
            "zeroed": torch.zeros_like(encoder_hidden),
            "shuffled": _shuffle_batch(encoder_hidden, gen),
        }
        for name, eh in variants.items():
            out = model(
                encoder_outputs=(eh,),
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                return_dict=True,
            )
            totals[name] += float(out.loss.item()) * n_tok
            token_counts[name] += n_tok

    print()
    print(f"batches evaluated: {n_batches}")
    print()
    print(f"{'condition':<10}  {'mean loss':>10}  {'perplexity':>12}")
    for name in ("intact", "zeroed", "shuffled"):
        mean_loss = totals[name] / max(1, token_counts[name])
        ppl = float(torch.tensor(mean_loss).exp())
        print(f"{name:<10}  {mean_loss:>10.4f}  {ppl:>12.2f}")

    intact = totals["intact"] / max(1, token_counts["intact"])
    zeroed = totals["zeroed"] / max(1, token_counts["zeroed"])
    shuffled = totals["shuffled"] / max(1, token_counts["shuffled"])
    print()
    print(f"delta zeroed-intact   : {zeroed - intact:+.4f}  "
          f"({'IGNORING audio' if abs(zeroed - intact) < 0.05 else 'audio contributes'})")
    print(f"delta shuffled-intact : {shuffled - intact:+.4f}")


if __name__ == "__main__":
    main()
