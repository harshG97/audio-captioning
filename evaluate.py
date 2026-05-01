"""
evaluate.py — Generate captions on a held-out split with a trained RECAP
checkpoint, write predictions to JSON.

Reads cached encoder hidden states (no CLAP at eval time) and the precomputed
retrieval cache built against a *train-only* datastore (see
retrieval/build_retrieval_cache.py). Self-exclusion is irrelevant here
because val/test access_ids do not appear in the training datastore.

Generation: greedy by default; beam search if --num_beams > 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from data.access_id import DATASETS, build_access_id_column
from model.device import best_device
from model.recap import RECAP
from retrieval.prompt_builder import build_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate RECAP captions and save predictions.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--hdf5_path", type=str, required=True)
    p.add_argument("--retrieval_cache", type=str, required=True)
    p.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_prompt_length", type=int, default=96)
    p.add_argument("--length_penalty", type=float, default=1.0,
                   help="Beam-search length normalization exponent. score / n**alpha. "
                        "1.0 = pure mean log-prob (default); <1 favors shorter, >1 favors longer.")
    p.add_argument("--no_repeat_ngram_size", type=int, default=0,
                   help="Forbid generating n-grams of this size that already appeared in the "
                        "generated suffix. 0 disables blocking (default).")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _build_prompt_ids(tokenizer, prompt: str, max_prompt_length: int) -> List[int]:
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > max_prompt_length:
        ids = ids[-max_prompt_length:]
    return ids


def _ban_repeated_ngrams(
    logits: torch.Tensor,            # (vocab,)
    sequence: torch.Tensor,          # (1, L) -- single hypothesis
    n: int,
    suffix_start: int,               # only block n-grams generated AFTER this index
) -> torch.Tensor:
    """In-place return of logits with banned next-token ids set to -inf so that
    no n-gram of size `n` from the *generated suffix* repeats. n <= 0 is a no-op."""
    if n <= 0:
        return logits
    seq = sequence[0].tolist()[suffix_start:]
    if len(seq) < n - 1:
        return logits
    prefix = tuple(seq[-(n - 1):]) if n > 1 else tuple()
    banned: set[int] = set()
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            banned.add(seq[i + n - 1])
    if not banned:
        return logits
    logits = logits.clone()
    for tok in banned:
        logits[tok] = float("-inf")
    return logits


@torch.no_grad()
def generate_caption(
    model: RECAP,
    tokenizer,
    encoder_hidden: torch.Tensor,         # (1, T, D)
    decoder_start_id: int,
    prompt_ids: List[int],
    eos_id: int,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden.to(device))
    initial = torch.tensor([[decoder_start_id, *prompt_ids]], dtype=torch.long, device=device)
    suffix_start = 1 + len(prompt_ids)            # generated tokens begin here

    if num_beams <= 1:
        cur = initial
        for _ in range(max_new_tokens):
            out = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=cur,
                return_dict=True,
            )
            logits = out.logits[0, -1]
            logits = _ban_repeated_ngrams(logits, cur, no_repeat_ngram_size, suffix_start)
            next_id = int(logits.argmax().item())
            cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == eos_id:
                break
        gen_ids = cur[0, suffix_start:].tolist()
        if gen_ids and gen_ids[-1] == eos_id:
            gen_ids = gen_ids[:-1]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Beam search with length-normalized log-probs: score / n**length_penalty.
    beams = [(0.0, initial.clone(), False)]            # (score, sequence, is_done)
    for _ in range(max_new_tokens):
        if all(done for _, _, done in beams):
            break
        candidates: list[tuple[float, torch.Tensor, bool]] = []
        for score, seq, done in beams:
            if done:
                candidates.append((score, seq, True))
                continue
            out = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=seq,
                return_dict=True,
            )
            logits = out.logits[0, -1]
            logits = _ban_repeated_ngrams(logits, seq, no_repeat_ngram_size, suffix_start)
            log_probs = torch.log_softmax(logits, dim=-1)
            top_lp, top_idx = log_probs.topk(num_beams)
            for lp, tok in zip(top_lp.tolist(), top_idx.tolist()):
                new_seq = torch.cat([seq, torch.tensor([[tok]], device=device)], dim=1)
                new_done = tok == eos_id
                candidates.append((score + lp, new_seq, new_done))

        def length_norm(item: tuple[float, torch.Tensor, bool]) -> float:
            s, sq, _ = item
            n = max(1, sq.shape[1] - suffix_start)
            return s / (n ** length_penalty)

        candidates.sort(key=length_norm, reverse=True)
        beams = candidates[:num_beams]

    def final_score(item: tuple[float, torch.Tensor, bool]) -> float:
        s, sq, _ = item
        n = max(1, sq.shape[1] - suffix_start)
        return s / (n ** length_penalty)

    best = max(beams, key=final_score)
    seq = best[1]
    gen_ids = seq[0, suffix_start:].tolist()
    if gen_ids and gen_ids[-1] == eos_id:
        gen_ids = gen_ids[:-1]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else best_device()
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = RECAP.from_pretrained(args.checkpoint).to(device)
    model.eval()

    decoder_start_id = (
        model.config.decoder_start_token_id
        if model.config.decoder_start_token_id is not None
        else (tokenizer.bos_token_id or tokenizer.eos_token_id)
    )
    eos_id = tokenizer.eos_token_id

    df = pd.read_csv(args.csv_path)
    df = build_access_id_column(df, args.dataset)
    if "caption" not in df.columns:
        raise ValueError("CSV must contain a 'caption' column.")

    cache = json.loads(Path(args.retrieval_cache).read_text())

    # Group references by access_id (val/test clips have multiple gold captions)
    references_by_id: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        references_by_id.setdefault(str(row["access_id"]), []).append(str(row["caption"]))

    predictions: dict[str, dict] = {}
    with h5py.File(args.hdf5_path, "r") as h5f:
        available = set(h5f.keys())
        unique_ids = [aid for aid in references_by_id if aid in available and aid in cache]
        for access_id in tqdm(unique_ids, desc="generate"):
            retrieved = cache[access_id]
            prompt = build_prompt(retrieved)
            prompt_ids = _build_prompt_ids(tokenizer, prompt, args.max_prompt_length)

            feats = np.asarray(h5f[access_id][()], dtype=np.float32)
            encoder_hidden = torch.from_numpy(feats).unsqueeze(0)

            pred = generate_caption(
                model=model,
                tokenizer=tokenizer,
                encoder_hidden=encoder_hidden,
                decoder_start_id=decoder_start_id,
                prompt_ids=prompt_ids,
                eos_id=eos_id,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                device=device,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

            predictions[access_id] = {
                "references": references_by_id[access_id],
                "prediction": pred,
            }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"Wrote {len(predictions)} predictions to {out}")


if __name__ == "__main__":
    main()
