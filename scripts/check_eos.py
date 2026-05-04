"""
check_eos.py — Diagnostics for EOS supervision in RECAP training.

Runs three checks against a freshly-built (or checkpointed) model:

  1. Print labels[0] of one collated batch and confirm the last non--100
     value equals the tokenizer's EOS id.
  2. Forward-pass the batch and report the model's predicted probability
     for EOS at the gold EOS position (with top-5 predicted token ids /
     pieces for context). If this stays near zero across training, EOS
     supervision is too weak.
  3. Verify tokenizer.eos_token_id, model.config.eos_token_id,
     model.decoder.config.eos_token_id, and the eos_id evaluate.py
     would use all agree.

Pass --checkpoint to load a trained model; otherwise a fresh
build_recap() is used (useful as a pre-training sanity check).
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from data.access_id import DATASETS
from data.recap_dataset import RecapCollator, RecapDataset
from model.build_recap import _register_auto_classes_once, build_recap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EOS supervision diagnostics.")
    p.add_argument("--csv", type=str, required=True, help="Train CSV path.")
    p.add_argument("--hdf5", type=str, required=True, help="Cached encoder HDF5.")
    p.add_argument("--retrieval_cache", type=str, required=True)
    p.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional path to a trained RECAP checkpoint. "
                        "If omitted, a fresh model is built.")
    p.add_argument("--encoder_name", type=str, default="laion/clap-htsat-fused")
    p.add_argument("--decoder_name", type=str, default="gpt2")
    return p.parse_args()


def _load_model_and_tokenizer(args):
    if args.checkpoint:
        _register_auto_classes_once()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(args.checkpoint)
        return model, tokenizer
    return build_recap(
        encoder_name=args.encoder_name,
        decoder_name=args.decoder_name,
    )


def check_labels_end_with_eos(batch, eos_id: int, tokenizer) -> None:
    print("\n=== [1] labels[0] sanity check ===")
    labels0 = batch["labels"][0]
    print(f"labels[0] shape: {tuple(labels0.shape)}")
    print(f"labels[0] tensor:\n{labels0.tolist()}")

    nonmasked = (labels0 != -100).nonzero(as_tuple=False).flatten()
    if nonmasked.numel() == 0:
        print("ERROR: every position in labels[0] is -100.")
        return

    last_idx = nonmasked[-1].item()
    last_id = labels0[last_idx].item()
    print(f"Last non--100 position: {last_idx}, token id: {last_id}, "
          f"piece: {tokenizer.decode([last_id])!r}")
    print(f"Tokenizer EOS id: {eos_id}, piece: {tokenizer.decode([eos_id])!r}")

    if last_id == eos_id:
        print("PASS: caption is terminated with EOS.")
    else:
        print("FAIL: last supervised token is NOT EOS.")

    sup_count = nonmasked.numel()
    print(f"Supervised tokens in labels[0]: {sup_count}  "
          f"(EOS contributes 1/{sup_count} = {1.0 / sup_count:.3f} of the per-example loss)")


def check_eos_probability(model, batch, eos_id: int, tokenizer, device) -> None:
    print("\n=== [2] Predicted P(EOS) at gold EOS position ===")
    model.eval()
    with torch.no_grad():
        out = model(
            encoder_outputs=batch["encoder_outputs"].to(device),
            decoder_input_ids=batch["decoder_input_ids"].to(device),
            decoder_attention_mask=batch["decoder_attention_mask"].to(device),
            labels=batch["labels"].to(device),
            return_dict=True,
        )
    logits = out.logits  # (B, L, V)
    probs = logits.softmax(dim=-1)
    labels = batch["labels"].to(device)

    bsz = labels.shape[0]
    eos_probs = []
    for b in range(bsz):
        nonmasked = (labels[b] != -100).nonzero(as_tuple=False).flatten()
        if nonmasked.numel() == 0:
            continue
        last_idx = nonmasked[-1].item()
        if labels[b, last_idx].item() != eos_id:
            print(f"  sample {b}: gold last token is not EOS, skipping")
            continue
        p_eos = probs[b, last_idx, eos_id].item()
        eos_probs.append(p_eos)
        topk = torch.topk(probs[b, last_idx], k=5)
        top_ids = topk.indices.tolist()
        top_ps = [round(x, 4) for x in topk.values.tolist()]
        top_pieces = [tokenizer.decode([i]) for i in top_ids]
        print(f"  sample {b}: P(EOS)={p_eos:.4f}  top5={list(zip(top_ids, top_pieces, top_ps))}")

    if eos_probs:
        avg = sum(eos_probs) / len(eos_probs)
        print(f"Mean P(EOS) at gold EOS positions over batch: {avg:.4f}")
        print("Interpretation: at convergence this should be high (>0.5). "
              "If it stays <0.05 across training, EOS supervision is being "
              "drowned out — up-weight EOS in the loss or duplicate EOS in cap_ids.")


def check_id_consistency(model, tokenizer) -> None:
    print("\n=== [3] EOS id consistency across model + tokenizer ===")
    tok_eos = tokenizer.eos_token_id
    model_eos = getattr(model.config, "eos_token_id", None)
    dec_eos = getattr(model.decoder.config, "eos_token_id", None)
    eval_eos = tokenizer.eos_token_id  # evaluate.py:151 sets eos_id this way

    print(f"tokenizer.eos_token_id        : {tok_eos}")
    print(f"model.config.eos_token_id     : {model_eos}")
    print(f"model.decoder.config.eos_token_id: {dec_eos}")
    print(f"evaluate.py would use         : {eval_eos}")

    if tok_eos == model_eos == dec_eos == eval_eos:
        print("PASS: all four EOS ids agree.")
    else:
        print("FAIL: EOS ids disagree — generation will not stop on the "
              "token the model was trained to emit.")

    tok_pad = tokenizer.pad_token_id
    print(f"\ntokenizer.pad_token_id        : {tok_pad}")
    if tok_pad == tok_eos:
        print("WARNING: pad_token_id == eos_token_id (alias). This is the "
              "GPT-2 default; consider distinguishing them so EOS is a true "
              "stop signal (see model/build_recap.py).")
    else:
        print("OK: pad and eos are distinct.")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = _load_model_and_tokenizer(args)
    model.to(device)
    eos_id = tokenizer.eos_token_id

    ds = RecapDataset(
        csv_path=args.csv,
        hdf5_path=args.hdf5,
        retrieval_cache_path=args.retrieval_cache,
        tokenizer=tokenizer,
        dataset=args.dataset,
        max_length=args.max_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    collator = RecapCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    check_labels_end_with_eos(batch, eos_id, tokenizer)
    check_eos_probability(model, batch, eos_id, tokenizer, device)
    check_id_consistency(model, tokenizer)


if __name__ == "__main__":
    main()
