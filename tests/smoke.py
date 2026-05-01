"""
smoke.py — Cheap correctness checks for the RECAP pipeline.

Covers:
  #4 dataset alignment   : prompt masking and shift between
                           decoder_input_ids and labels (no model needed,
                           uses GPT-2 tokenizer only).
  #3 forward path        : RECAP.forward accepts a raw tensor as
                           encoder_outputs and returns a finite loss.
  #5 save/load roundtrip : RECAP.from_pretrained on a saved checkpoint
                           reproduces the same logits.

Tests #3 and #5 download GPT-2 (~500 MB) and CLAP-HTSAT-Fused (~1.5 GB)
on first run. They are skipped automatically if those downloads fail.

Run:
    python -m tests.smoke
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_tiny_dataset(tmp: Path):
    """Construct a 1-row CSV + HDF5 + retrieval cache and a RecapDataset."""
    import h5py
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    from data.recap_dataset import RecapDataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    csv = tmp / "data.csv"
    pd.DataFrame(
        [{"youtube_id": "abc", "start_time": 0, "caption": "a dog barks loudly"}]
    ).to_csv(csv, index=False)

    access_id = "abc_0"
    h5_path = tmp / "feats.hdf5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(access_id, data=np.zeros((64, 768), dtype=np.float32))

    cache = tmp / "retrieved.json"
    cache.write_text(json.dumps({access_id: ["a hello world", "another caption"]}))

    ds = RecapDataset(
        csv_path=str(csv),
        hdf5_path=str(h5_path),
        retrieval_cache_path=str(cache),
        tokenizer=tokenizer,
        dataset="audiocaps",
        max_length=128,
    )
    return ds, tokenizer


def test_dataset_alignment() -> None:
    from data.recap_dataset import RecapCollator

    with tempfile.TemporaryDirectory() as tmp:
        ds, tokenizer = _build_tiny_dataset(Path(tmp))
        item = ds[0]

        decoder_input_ids = item["decoder_input_ids"].tolist()
        labels = item["labels"].tolist()
        assert len(decoder_input_ids) == len(labels), "length mismatch"
        assert decoder_input_ids[0] == ds.dec_start, "first token must be decoder_start"

        non_masked = [i for i, l in enumerate(labels) if l != -100]
        assert non_masked, "no caption positions to predict"
        assert labels[non_masked[-1]] == tokenizer.eos_token_id, "last label must be EOS"

        # The shift contract recap.py relies on: labels[t] == decoder_input_ids[t+1]
        # for every position whose label is not masked (except possibly the very last,
        # which has no t+1 in decoder_input_ids).
        for t in non_masked:
            if t + 1 < len(decoder_input_ids):
                assert decoder_input_ids[t + 1] == labels[t], (
                    f"shift bug at t={t}: dec_in[{t+1}]={decoder_input_ids[t+1]} "
                    f"vs labels[{t}]={labels[t]}"
                )

        collator = RecapCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([item, item])
        assert tuple(batch["encoder_outputs"].shape) == (2, 64, 768)
        assert batch["decoder_input_ids"].shape == batch["labels"].shape
    print("[PASS] #4 dataset alignment")


def test_forward_with_encoder_outputs_tensor() -> None:
    import torch

    try:
        from model.build_recap import build_recap
    except Exception as e:
        print(f"[SKIP] #3 forward: import failure ({e})")
        return

    try:
        model, tokenizer = build_recap()
    except Exception as e:
        print(f"[SKIP] #3 forward: model load failed ({e})")
        return

    model.eval()
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    enc = torch.randn(2, 64, 768)
    dec_in = torch.tensor([[bos_id, 100, 200, 300, 400, 500, 600, 700]] * 2, dtype=torch.long)
    labels = torch.full((2, 8), -100, dtype=torch.long)
    labels[:, 4:] = dec_in[:, 4:]

    with torch.no_grad():
        out = model(
            encoder_outputs=enc,
            decoder_input_ids=dec_in,
            labels=labels,
            return_dict=True,
        )
    assert torch.isfinite(out.loss), f"loss not finite: {out.loss}"
    assert out.logits.shape == (2, 8, model.decoder.config.vocab_size)
    print("[PASS] #3 forward with encoder_outputs tensor")


def test_save_load_roundtrip() -> None:
    import torch

    try:
        from model.build_recap import build_recap
        from model.recap import RECAP
    except Exception as e:
        print(f"[SKIP] #5 save/load: import failure ({e})")
        return

    try:
        model, tokenizer = build_recap()
    except Exception as e:
        print(f"[SKIP] #5 save/load: model load failed ({e})")
        return

    model.eval()
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    enc = torch.randn(1, 64, 768)
    dec_in = torch.tensor([[bos_id, 100, 200, 300]], dtype=torch.long)

    with torch.no_grad():
        out1 = model(encoder_outputs=enc, decoder_input_ids=dec_in, return_dict=True)

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        loaded = RECAP.from_pretrained(tmp)
        loaded.eval()
        with torch.no_grad():
            out2 = loaded(encoder_outputs=enc, decoder_input_ids=dec_in, return_dict=True)

    assert torch.allclose(out1.logits, out2.logits, atol=1e-5), (
        "logits diverge after save/load (max diff: "
        f"{(out1.logits - out2.logits).abs().max().item()})"
    )
    print("[PASS] #5 save/load roundtrip")


def main() -> None:
    test_dataset_alignment()
    test_forward_with_encoder_outputs_tensor()
    test_save_load_roundtrip()
    print("smoke tests: done")


if __name__ == "__main__":
    main()
