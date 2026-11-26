"""
evaluate.py — Generate captions on a held-out split with a trained RECAP
checkpoint, write predictions to JSON.

Reads cached encoder hidden states (no CLAP at eval time) and the precomputed
retrieval cache built against a *train-only* datastore (see
retrieval/build_retrieval_cache.py). Self-exclusion is irrelevant here
because val/test access_ids do not appear in the training datastore.

Generation defaults follow the RECAP reference inference (infer.py):
beam search with 3 beams, up to 60 new tokens, length_penalty 0, no n-gram
blocking. Pass --num_beams 1 for greedy decoding.
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
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutput

from data.access_id import DATASETS, build_access_id_column
from data.caption_filter import warn_if_cache_has_placeholders
from model.device import best_device
from model.prefix_lm import AudioPrefixLM, load_captioner
from model.recap import RECAP
from retrieval.prompt_builder import GENERATION_STOP_STRINGS, build_prompt, checkpoint_retrieval_k, first_k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate RECAP captions and save predictions.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--hdf5_path", type=str, required=True)
    p.add_argument("--retrieval_cache", type=str, required=True)
    p.add_argument("--retrieval_k", type=int, default=None,
                   help="Captions per prompt. Default: the k the checkpoint was "
                        "trained with (config.retrieval_k), else the whole cache entry.")
    p.add_argument("--dataset", type=str, default="audiocaps", choices=DATASETS)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--num_beams", type=int, default=3, help="1 = greedy. Default: 3.")
    p.add_argument("--max_prompt_length", type=int, default=126,
                   help="Prompts longer than this are trimmed from the left. 126 matches the "
                        "prompt budget train.py gets with its default --max_length 128.")
    p.add_argument("--length_penalty", type=float, default=0.0,
                   help="Beam-search length normalization exponent: score / n**alpha. "
                        "0.0 = summed log-prob (default, as in the RECAP reference code); "
                        "1.0 = mean log-prob; >1 favors longer. Ignored when --num_beams 1.")
    p.add_argument("--no_repeat_ngram_size", type=int, default=0,
                   help="Forbid generating n-grams of this size that already appeared in the "
                        "generated suffix. 0 disables blocking (default).")
    p.add_argument("--min_new_tokens", type=int, default=0,
                   help="Do not end a caption before it has this many tokens: EOS and "
                        "tokens containing '.' or a newline, where captions are cut, are "
                        "blocked until then (GPT-2 uses about 1.3 tokens per word). 0 "
                        "disables it (default).")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    if not 0 <= args.min_new_tokens < args.max_new_tokens:
        p.error("--min_new_tokens must be 0 or more and below --max_new_tokens")
    return args


def _build_prompt_ids(tokenizer, prompt: str, max_prompt_length: int) -> List[int]:
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > max_prompt_length:
        ids = ids[-max_prompt_length:]
    return ids


class SuffixNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Block n-gram repeats within the *generated suffix only* (ignoring the
    prompt prefix), matching the original evaluate.py semantics. HF's built-in
    NoRepeatNGramLogitsProcessor blocks across the entire input including the
    prompt, which would forbid the model from echoing useful phrases from the
    retrieved context."""

    def __init__(self, ngram_size: int, suffix_start: int):
        self.n = ngram_size
        self.suffix_start = suffix_start

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.n <= 0:
            return scores
        for i in range(input_ids.shape[0]):
            seq = input_ids[i, self.suffix_start:].tolist()
            if len(seq) < self.n - 1:
                continue
            prefix = tuple(seq[-(self.n - 1):]) if self.n > 1 else ()
            banned: set[int] = set()
            for j in range(len(seq) - self.n + 1):
                if tuple(seq[j:j + self.n - 1]) == prefix:
                    banned.add(seq[j + self.n - 1])
            if banned:
                scores[i, list(banned)] = float("-inf")
        return scores

# Stop strings a single token can contain; clean_prediction cuts the caption at
# the first of them, so a minimum length must hold these back as well as EOS.
TOKEN_STOP_STRINGS = tuple(s for s in GENERATION_STOP_STRINGS if len(s) <= 2)


def caption_end_token_ids(tokenizer, eos_id: int) -> List[int]:
    """EOS (for RECAP the "." token, see wire_special_tokens) plus every token
    whose text contains a stop string, such as "...", ".," or a newline: each
    ends the caption, by stopping generation or by the cut in
    clean_prediction. Cached on the tokenizer."""
    cache = tokenizer.__dict__.setdefault("_caption_end_token_ids", {})
    if eos_id not in cache:
        ids = {eos_id}
        for tid in range(len(tokenizer)):
            text = tokenizer.decode([tid]).lower()
            if any(stop in text for stop in TOKEN_STOP_STRINGS):
                ids.add(tid)
        cache[eos_id] = sorted(ids)
    return cache[eos_id]


class SuffixMinLengthLogitsProcessor(LogitsProcessor):
    """Forbid the tokens that end a caption until the generated suffix has
    min_new_tokens tokens. Hugging Face's min_new_tokens blocks only EOS, so
    the model could still end early with a token such as "..." or a newline,
    where clean_prediction cuts the caption."""

    def __init__(self, min_new_tokens: int, suffix_start: int, end_ids: List[int]):
        self.min_new_tokens = min_new_tokens
        self.suffix_start = suffix_start
        self.end_ids = end_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] - self.suffix_start < self.min_new_tokens:
            scores[:, self.end_ids] = float("-inf")
        return scores


def clean_prediction(text, stop_strings):
    """
    Truncates the string at the first occurrence of any stop string 
    and removes the prompt-based repetitions.
    """
    # 1. Find the earliest occurrence of any stop string
    earliest_stop = len(text)
    for stop_str in stop_strings:
        idx = text.find(stop_str)
        if idx != -1 and idx < earliest_stop:
            earliest_stop = idx
    
    # 2. Slice the text at that point
    text = text[:earliest_stop]
    
    # 3. Clean up whitespace
    return text.strip()

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
    return_stop_reason: bool = False,
    min_new_tokens: int = 0,
):
    """Generate one caption. Returns the decoded text, or (text, stop_reason)
    when return_stop_reason=True, where stop_reason is "eos" (the model ended
    the caption itself) or "max_new_tokens" (it ran into the token limit).
    min_new_tokens > 0 holds back every token that would end the caption (EOS,
    ".", newline) until that many tokens follow the prompt.

    The prefix model (AudioPrefixLM) gets the same start token and prompt
    after its audio tokens; its output holds only the new tokens."""
    decoder_input_ids = torch.tensor(
        [[decoder_start_id, *prompt_ids]], dtype=torch.long, device=device
    )
    if isinstance(model, AudioPrefixLM):
        inputs = dict(encoder_hidden_states=encoder_hidden.to(device), input_ids=decoder_input_ids)
        suffix_start = 0
    else:
        inputs = dict(encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden.to(device)),
                      decoder_input_ids=decoder_input_ids)
        suffix_start = decoder_input_ids.shape[1]

    logits_processor = LogitsProcessorList()
    if no_repeat_ngram_size > 0:
        logits_processor.append(
            SuffixNoRepeatNGramLogitsProcessor(no_repeat_ngram_size, suffix_start)
        )
    if min_new_tokens > 0:
        logits_processor.append(SuffixMinLengthLogitsProcessor(
            min_new_tokens, suffix_start, caption_end_token_ids(tokenizer, eos_id)))
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=(num_beams > 1),
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=logits_processor,
        use_cache=True,
    )

    gen_ids = out[0, suffix_start:].tolist()
    stop_reason = "max_new_tokens"
    if gen_ids and gen_ids[-1] == eos_id:
        gen_ids = gen_ids[:-1]
        stop_reason = "eos"
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return (text, stop_reason) if return_stop_reason else text


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else best_device()
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_captioner(args.checkpoint).to(device)
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
    warn_if_cache_has_placeholders(cache, source=args.retrieval_cache)
    k = args.retrieval_k if args.retrieval_k is not None else checkpoint_retrieval_k(model.config)
    print(f"Retrieved captions per prompt: {k if k is not None else 'all in cache'}")
    print(f"Decoding: num_beams={args.num_beams} length_penalty={args.length_penalty} "
          f"no_repeat_ngram_size={args.no_repeat_ngram_size} "
          f"min_new_tokens={args.min_new_tokens} max_new_tokens={args.max_new_tokens}")

    # Group references by access_id (val/test clips have multiple gold captions)
    references_by_id: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        references_by_id.setdefault(str(row["access_id"]), []).append(str(row["caption"]))

    predictions: dict[str, dict] = {}
    with h5py.File(args.hdf5_path, "r") as h5f:
        available = set(h5f.keys())
        unique_ids = [aid for aid in references_by_id if aid in available and aid in cache]
        for access_id in tqdm(unique_ids, desc="generate"):
            retrieved = first_k(cache[access_id], k)
            prompt = build_prompt(retrieved)
            prompt_ids = _build_prompt_ids(tokenizer, prompt, args.max_prompt_length)

            feats = np.asarray(h5f[access_id][()], dtype=np.float32)
            encoder_hidden = torch.from_numpy(feats).unsqueeze(0)

            pred_raw, stop_reason = generate_caption(
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
                return_stop_reason=True,
                min_new_tokens=args.min_new_tokens,
            )
            pred_raw = pred_raw.strip().lower()

            # Cut the continuation at the first stop string (newline, prompt echo, ".").
            processed_pred = clean_prediction(pred_raw, GENERATION_STOP_STRINGS)

            predictions[access_id] = {
                "references": references_by_id[access_id],
                "prediction": processed_pred,
                "raw_prediction": pred_raw,
                # Diagnostics read by score_predictions.py (generation health).
                "stop_reason": stop_reason,
                "cut_by_postprocessing": processed_pred != pred_raw,
            }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"Wrote {len(predictions)} predictions to {out}")


if __name__ == "__main__":
    main()
