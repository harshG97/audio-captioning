"""
recap_dataset.py — Torch Dataset + collator for RECAP training/eval.

Each example yields:
  encoder_hidden_states: (T_audio, D_audio) cached CLAP audio encoder output.
  decoder_input_ids   : prompt + GT caption shifted right by one
                        (prepended with decoder_start_token_id).
  labels              : same length as decoder_input_ids, shift-aligned for
                        next-token prediction. Prompt tokens and pad
                        positions are masked to -100.

The dataset reads cached encoder features from an HDF5 written by
`data/audio_preprocess.py`, plus a precomputed retrieval cache JSON
{access_id: [retrieved_captions]} written by
`retrieval/build_retrieval_cache.py`. No CLAP / no retrieval is run at
training time.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.access_id import build_access_id_column
from data.caption_filter import drop_placeholder_captions, warn_if_cache_has_placeholders
from retrieval.prompt_builder import build_prompt, first_k


class RecapDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        hdf5_path: str,
        retrieval_cache_path: str,
        tokenizer,
        dataset: str = "audiocaps",
        max_length: int = 128,
        decoder_start_token_id: Optional[int] = None,
        prompt_dropout: float = 0.0,
        retrieval_k: Optional[int] = None,
    ) -> None:
        df = pd.read_csv(csv_path)
        df = build_access_id_column(df, dataset)
        if "caption" not in df.columns:
            raise ValueError("CSV must contain a 'caption' column.")
        # Placeholder/empty captions are not valid training targets.
        df = drop_placeholder_captions(df, source=csv_path)

        cache = json.loads(Path(retrieval_cache_path).read_text())
        warn_if_cache_has_placeholders(cache, source=str(retrieval_cache_path))

        with h5py.File(hdf5_path, "r") as f:
            available = set(f.keys())

        ids = df["access_id"].astype(str)
        keep = ids.isin(available) & ids.isin(cache.keys())
        dropped = (~keep).sum()
        if dropped:
            print(f"[RecapDataset] Skipping {dropped} rows missing audio features or retrieval.")
        self.df = df.loc[keep].reset_index(drop=True)
        self.cache = cache
        # Use the first retrieval_k cached captions (None = all of them).
        self.retrieval_k = retrieval_k
        if retrieval_k is not None:
            short = sum(len(v) < retrieval_k for v in cache.values())
            if short:
                print(f"[WARN] {short} cache entries in {retrieval_cache_path} have fewer "
                      f"than retrieval_k={retrieval_k} captions; those prompts are shorter.")
        self.hdf5_path = hdf5_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dec_start = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else (tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id)
        )
        self.eos_id = tokenizer.eos_token_id
        if not 0.0 <= prompt_dropout <= 1.0:
            raise ValueError("prompt_dropout must be in [0, 1]")
        self.prompt_dropout = prompt_dropout
        self._h5: Optional[h5py.File] = None

    def _h5_handle(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        access_id = str(row["access_id"])
        gt_caption = str(row["caption"])
        retrieved = first_k(self.cache[access_id], self.retrieval_k)
        if self.prompt_dropout > 0.0 and random.random() < self.prompt_dropout:
            # Forces the decoder to rely on cross-attention rather than the
            # retrieved-caption text, by collapsing the prompt to BASELINE_PROMPT.
            retrieved = []

        prompt = build_prompt(retrieved)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        cap_ids = self.tokenizer(" " + gt_caption, add_special_tokens=False)["input_ids"]
        cap_ids = cap_ids + [self.eos_id]

        # Reserve 1 slot for the leading decoder_start_token in decoder_input_ids.
        budget = self.max_length - 1
        if len(prompt_ids) + len(cap_ids) > budget:
            # Caption is sacred; trim the prompt from the left first.
            keep_prompt = max(0, budget - len(cap_ids))
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
            if len(cap_ids) > budget:
                cap_ids = cap_ids[: budget - 1] + [self.eos_id]

        full = prompt_ids + cap_ids
        decoder_input_ids = [self.dec_start] + full[:-1]
        labels = list(full)
        for i in range(len(prompt_ids)):
            labels[i] = -100

        feats = np.asarray(self._h5_handle()[access_id][()], dtype=np.float32)

        return {
            "encoder_hidden_states": torch.from_numpy(feats),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "access_id": access_id,
        }


class RecapCollator:
    """Right-pads decoder_input_ids/labels and stacks encoder hidden states."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["decoder_input_ids"]) for b in batch)
        bsz = len(batch)

        encoder_hidden = torch.stack([b["encoder_hidden_states"] for b in batch])
        decoder_input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        decoder_attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)

        for i, b in enumerate(batch):
            n = b["decoder_input_ids"].shape[0]
            decoder_input_ids[i, :n] = b["decoder_input_ids"]
            decoder_attention_mask[i, :n] = 1
            labels[i, :n] = b["labels"]

        return {
            "encoder_outputs": encoder_hidden,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
