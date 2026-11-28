"""
build_sound_labels.py — Sound labels for each clip, from BEATs' own classifier.

A BEATs checkpoint fine-tuned on AudioSet ends in a classifier over
AudioSet's 527 sound labels ("Speech", "Dog", "Rain", ...). This writes each
clip's most probable labels to

    <feat_dir>/labels_<split>.json     {clip id: [[label, probability], ...]}

so they can be shown to the decoder as text in the prompt ("Sounds: speech;
typing"): train with LABELS=5 (scripts/submit.sh), or --sound_labels in
train.py.

No audio pass is needed. BEATs applies the classifier to every patch of its
last layer and averages the results; the classifier is linear, so that
equals the classifier applied to the average patch, and the stored features
(the last layer, averaged over neighbouring frequency bands) have the same
average. The labels therefore match a full BEATs forward pass up to
rounding (tests/test_beats.py checks this against the model).

Usage:
    python data/build_sound_labels.py \\
        --checkpoint ~/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \\
        --feat_dir ~/scratch/recap/features_beats
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.extract_beats_features import label_names


def load_classifier(checkpoint_path: str):
    """(weight (labels, width), bias (labels,), label names by output index)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state, label_dict = ckpt.get("model", {}), ckpt.get("label_dict") or {}
    if "predictor.weight" not in state or not label_dict:
        raise SystemExit(f"{checkpoint_path} has no AudioSet classifier: "
                         "use a checkpoint fine-tuned on AudioSet")
    weight = state["predictor.weight"].float().numpy()
    bias = state["predictor.bias"].float().numpy()
    names_by_id = label_names()
    names = [names_by_id.get(label_dict[i], label_dict[i]).lower() for i in range(weight.shape[0])]
    return weight, bias, names


def check_features(feat_dir: str, checkpoint_path: str) -> None:
    """The features must be the last layer of this very checkpoint."""
    path = os.path.join(feat_dir, "encoder.txt")
    if not os.path.isfile(path):
        raise SystemExit(f"{feat_dir} has no encoder.txt: not a BEATs features folder")
    encoder = open(path).read().split()
    if encoder[:1] != ["beats"] or os.path.basename(checkpoint_path) not in encoder:
        raise SystemExit(f"{feat_dir} was built with '{' '.join(encoder)}', not {os.path.basename(checkpoint_path)}")
    if any(part.startswith("layers=") for part in encoder):
        raise SystemExit(f"{feat_dir} holds several layers; build the labels from the "
                         "single-layer features folder and point LABELS_DIR at it")


def clip_probabilities(feats: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Label probabilities of one clip from its (vectors, width) features."""
    logits = weight @ feats.astype(np.float64).mean(axis=0) + bias
    return 1.0 / (1.0 + np.exp(-logits))


def build_labels(hdf5_path: str, weight, bias, names, top: int) -> dict:
    out = {}
    with h5py.File(hdf5_path, "r") as f:
        for clip in tqdm(f.keys(), desc=os.path.basename(hdf5_path)):
            feats = f[clip][()]
            if feats.shape[-1] != weight.shape[1]:
                raise SystemExit(f"{hdf5_path}: features are {feats.shape[-1]} wide, "
                                 f"the classifier expects {weight.shape[1]}")
            probs = clip_probabilities(feats, weight, bias)
            best = np.argsort(-probs)[:top]
            out[clip] = [[names[i], round(float(probs[i]), 4)] for i in best]
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, help="the BEATs .pt the features were made with")
    p.add_argument("--feat_dir", required=True, help="BEATs features folder (single layer)")
    p.add_argument("--dataset", default="audiocaps")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--top", type=int, default=10, help="labels kept per clip (default 10)")
    args = p.parse_args(argv)

    check_features(args.feat_dir, args.checkpoint)
    weight, bias, names = load_classifier(args.checkpoint)
    for split in args.splits:
        labels = build_labels(os.path.join(args.feat_dir, f"{args.dataset}_{split}.hdf5"),
                              weight, bias, names, args.top)
        out = os.path.join(args.feat_dir, f"labels_{split}.json")
        with open(out, "w") as f:
            json.dump(labels, f)
        example = next(iter(labels.items()))
        print(f"{out}: {len(labels)} clips, e.g. {example[0]}: {example[1][:3]}")


if __name__ == "__main__":
    main()
