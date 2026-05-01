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
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _build_prompt_ids(tokenizer, prompt: str, max_prompt_length: int) -> List[int]:
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > max_prompt_length:
        ids = ids[-max_prompt_length:]
    return ids


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
) -> str:
    encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden.to(device))
    initial = torch.tensor([[decoder_start_id, *prompt_ids]], dtype=torch.long, device=device)

    if num_beams <= 1:
        cur = initial
        for _ in range(max_new_tokens):
            out = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=cur,
                return_dict=True,
            )
            next_id = int(out.logits[0, -1].argmax().item())
            cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == eos_id:
                break
        gen_ids = cur[0, 1 + len(prompt_ids):].tolist()
        if gen_ids and gen_ids[-1] == eos_id:
            gen_ids = gen_ids[:-1]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Beam search (length-normalized log-probs).
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
            log_probs = torch.log_softmax(out.logits[0, -1], dim=-1)
            top_lp, top_idx = log_probs.topk(num_beams)
            for lp, tok in zip(top_lp.tolist(), top_idx.tolist()):
                new_seq = torch.cat([seq, torch.tensor([[tok]], device=device)], dim=1)
                new_done = tok == eos_id
                candidates.append((score + lp, new_seq, new_done))

        def length_norm(item: tuple[float, torch.Tensor, bool]) -> float:
            s, sq, _ = item
            n = sq.shape[1] - 1 - len(prompt_ids)
            return s / max(1, n)

        candidates.sort(key=length_norm, reverse=True)
        beams = candidates[:num_beams]

    best = max(beams, key=lambda item: (item[0] / max(1, item[1].shape[1] - 1 - len(prompt_ids))))
    seq = best[1]
    gen_ids = seq[0, 1 + len(prompt_ids):].tolist()
    if gen_ids and gen_ids[-1] == eos_id:
        gen_ids = gen_ids[:-1]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
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

    predictions: dict[str, dict] = {}
    with h5py.File(args.hdf5_path, "r") as h5f:
        available = set(h5f.keys())
        for _, row in tqdm(df.iterrows(), total=len(df), desc="generate"):
            access_id = str(row["access_id"])
            if access_id not in available or access_id not in cache:
                continue
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
            )

            entry = predictions.setdefault(access_id, {"references": [], "prediction": pred})
            entry["references"].append(str(row["caption"]))

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"Wrote {len(predictions)} predictions to {out}")


if __name__ == "__main__":
    main()
