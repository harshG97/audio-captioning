"""
dataset.py

PyTorch Dataset and data-loading utilities for AudioCaps/RECAP training.
Adapted from Sreyan88/RECAP/src/utils.py for the AudioCaps column layout.

Key differences from the Clotho reference:
  - HDF5 key is "{youtube_id}_{start_time}" (not file_name)
  - CSVs have columns: audiocap_id, youtube_id, start_time, caption
  - One caption per row (no grouping by audio file needed)
"""

import json
import os

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

CAPTION_LENGTH = 60
SIMPLE_PREFIX = "This audio sounds like "


def prep_strings(
    text,
    tokenizer,
    template=None,
    retrieved_caps=None,
    k=None,
    is_test=False,
    max_length=None,
):
    """
    Build decoder input_ids and labels for one sample.

    Training (is_test=False):
        input_ids  = [prefix tokens] + [caption tokens] + [pad to max_length]
        labels     = [-100]*(len_prefix-1) + [caption tokens] + [eos] + [-100 pad]
        The -100 mask causes CrossEntropyLoss to ignore prefix positions.

    Inference (is_test=True):
        Returns only the prefix token ids (no padding, no truncation).
    """
    if retrieved_caps is not None:
        infix = "\n\n".join(retrieved_caps[:k])
        prefix = template.replace("||", infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if not is_test:
        text_ids = text_ids[:CAPTION_LENGTH]

    input_ids = prefix_ids + text_ids if not is_test else prefix_ids
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id]

    if not is_test:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))

    if is_test:
        return input_ids
    return input_ids, label_ids


def postprocess_preds(pred, tokenizer):
    """Strip the prefix and special tokens from a generated caption string."""
    if SIMPLE_PREFIX in pred:
        pred = pred.split(SIMPLE_PREFIX)[-1]
    else:
        pred = pred.split(SIMPLE_PREFIX.strip())[-1]
    pred = pred.strip()
    pred = pred.replace(tokenizer.pad_token, "")
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[: -len(tokenizer.eos_token)]
    return pred


class TrainDataset(Dataset):
    """
    Dataset for training RECAP on AudioCaps.

    Each item returns a dict with:
        encoder_outputs:    (64, 768) float32 — precomputed CLAP audio features
        decoder_input_ids:  (max_target_length,) long  — tokenized prefix + caption
        labels:             (max_target_length,) long  — -100 masked prefix, caption + eos
    """

    def __init__(
        self,
        df,
        features_path,
        tokenizer,
        rag=False,
        template_path=None,
        k=None,
        max_caption_length=CAPTION_LENGTH,
    ):
        """
        Args:
            df:                 DataFrame with columns: audio_id, caption[, caps]
                                audio_id must match the HDF5 key (youtube_id_start_time).
            features_path:      Path to HDF5 file keyed by audio_id strings.
            tokenizer:          GPT-2 tokenizer (pad/eos tokens already configured).
            rag:                Whether to prepend retrieved captions as a prefix.
            template_path:      Path to template .txt file; required when rag=True.
                                Use "||" as the placeholder for retrieved captions.
            k:                  Number of retrieved captions to include; required when rag=True.
            max_caption_length: Max tokens for the ground-truth caption portion.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, "r")
        self.rag = rag

        if rag:
            assert template_path is not None, "template_path required when rag=True"
            assert k is not None, "k required when rag=True"
            self.template = open(template_path).read().strip() + " "
            self.k = k
            self.max_target_length = (
                max_caption_length
                + max_caption_length * k
                + len(tokenizer.encode(self.template))
                + len(tokenizer.encode("\n\n")) * (k - 1)
            )
        else:
            self.max_target_length = max_caption_length + len(
                tokenizer.encode(SIMPLE_PREFIX)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["caption"]
        audio_id = str(row["audio_id"])

        if self.rag:
            caps = row["caps"]
            decoder_input_ids, labels = prep_strings(
                text,
                self.tokenizer,
                template=self.template,
                retrieved_caps=caps,
                k=self.k,
                max_length=self.max_target_length,
            )
        else:
            decoder_input_ids, labels = prep_strings(
                text,
                self.tokenizer,
                max_length=self.max_target_length,
            )

        encoder_outputs = self.features[audio_id][()]

        return {
            "encoder_outputs": torch.tensor(encoder_outputs, dtype=torch.float32),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_data_for_training(annotations_path, caps_path=None):
    """
    Load AudioCaps train and val DataFrames.
    
    Applies the same row filtering as audio_preprocess.py so that every
    surviving row has a corresponding HDF5 entry.
    """
    train_df = pd.read_csv(os.path.join(annotations_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(annotations_path, "val.csv"))

    for df_name, df in [("train", train_df), ("val", val_df)]:
        # Match audio_preprocess.py's filtering exactly.
        n_before = len(df)
        df.dropna(subset=["youtube_id", "start_time"], inplace=True)
        n_after = len(df)
        if n_before != n_after:
            print(f"[load_data] Dropped {n_before - n_after} NaN rows from {df_name}.csv "
                  f"({n_before} -> {n_after})")
        df["start_time"] = df["start_time"].astype(int)
        df["audio_id"] = df["youtube_id"] + "_" + df["start_time"].astype(str)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    if caps_path is not None:
        retrieved = json.load(open(caps_path))
        train_df["caps"] = train_df["audiocap_id"].apply(
            lambda aid: retrieved.get(str(aid), [])
        )
        val_df["caps"] = val_df["audiocap_id"].apply(
            lambda aid: retrieved.get(str(aid), [])
        )

    return {"train": train_df, "val": val_df}
