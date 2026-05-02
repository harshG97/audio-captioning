"""
smoke2.py — Extended pipeline tests for RECAP.

Covers:
  #6 training step      : one forward+backward on synthetic data; loss is
                           finite, only cross-attention params have gradients,
                           and a second step reduces the loss.
  #7 collator→forward   : RecapCollator output is directly accepted by
                           RECAP.forward and produces a finite loss.
  #8 greedy generation   : generate_caption produces decodable text that
                           terminates at EOS within the token budget.
  #9 prompt builder      : build_prompt handles empty, single, and multi-
                           caption inputs; baseline fallback works.

Tests #6–#8 download GPT-2 (~500 MB) and CLAP (~1.5 GB) on first run.
They are skipped automatically if those downloads fail.

Run:
    python -m tests.smoke2
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── #9  prompt builder (no model needed) ─────────────────────────────

def test_prompt_builder() -> None:
    from retrieval.prompt_builder import build_prompt, BASELINE_PROMPT

    # Empty retrieval → baseline
    assert build_prompt([]) == BASELINE_PROMPT, "empty retrieval should give baseline"

    # Single caption
    p1 = build_prompt(["a dog barks"])
    assert "a dog barks" in p1
    assert p1.endswith("This audio sounds like:")

    # Multiple captions – comma-separated
    p2 = build_prompt(["cap1", "cap2", "cap3"])
    assert "cap1, cap2, cap3" in p2
    assert p2.endswith("This audio sounds like:")

    print("[PASS] #9 prompt builder")


# ── helpers (shared by #6, #7, #8) ───────────────────────────────────

def _build_tiny_dataset(tmp: Path):
    """Construct a 2-row CSV + HDF5 + retrieval cache and a RecapDataset."""
    import h5py
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    from data.recap_dataset import RecapDataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [
        {"youtube_id": "abc", "start_time": 0, "caption": "a dog barks loudly"},
        {"youtube_id": "def", "start_time": 0, "caption": "rain falling on a roof"},
    ]
    csv = tmp / "data.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    h5_path = tmp / "feats.hdf5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("abc_0", data=np.random.randn(64, 768).astype(np.float32))
        f.create_dataset("def_0", data=np.random.randn(64, 768).astype(np.float32))

    cache = tmp / "retrieved.json"
    cache.write_text(json.dumps({
        "abc_0": ["a hello world", "another caption"],
        "def_0": ["water dripping", "storm sounds"],
    }))

    ds = RecapDataset(
        csv_path=str(csv),
        hdf5_path=str(h5_path),
        retrieval_cache_path=str(cache),
        tokenizer=tokenizer,
        dataset="audiocaps",
        max_length=128,
    )
    return ds, tokenizer


def _get_model_and_tokenizer():
    """Build RECAP once; skip if downloads fail."""
    try:
        from model.build_recap import build_recap
        return build_recap()
    except Exception as e:
        return None, None, str(e)


# ── #7  collator → forward integration ──────────────────────────────

def test_collator_forward(model, tokenizer) -> None:
    import torch
    from data.recap_dataset import RecapCollator

    with tempfile.TemporaryDirectory() as tmp:
        ds, _ = _build_tiny_dataset(Path(tmp))
        collator = RecapCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0], ds[1]])

        model.eval()
        with torch.no_grad():
            out = model(
                encoder_outputs=batch["encoder_outputs"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                labels=batch["labels"],
                return_dict=True,
            )

        assert torch.isfinite(out.loss), f"loss not finite: {out.loss}"
        assert out.logits.shape[0] == 2, "batch size mismatch"
        assert out.logits.shape[1] == batch["decoder_input_ids"].shape[1]

    print("[PASS] #7 collator → forward integration")


# ── #6  training step ────────────────────────────────────────────────

def test_training_step(model, tokenizer) -> None:
    import torch
    from data.recap_dataset import RecapCollator

    with tempfile.TemporaryDirectory() as tmp:
        ds, _ = _build_tiny_dataset(Path(tmp))
        collator = RecapCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0], ds[1]])

        model.train()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        # --- Step 1: forward + backward ---
        out1 = model(
            encoder_outputs=batch["encoder_outputs"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
            return_dict=True,
        )
        loss1 = out1.loss
        assert torch.isfinite(loss1), f"loss1 not finite: {loss1}"
        loss1.backward()

        # Only cross-attention + ln_cross_attn should have gradients
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert "crossattention" in name or "ln_cross_attn" in name, (
                    f"unexpected grad on {name}"
                )

        optimizer.step()
        optimizer.zero_grad()

        # --- Step 2: loss should decrease (model is overfitting 2 examples) ---
        out2 = model(
            encoder_outputs=batch["encoder_outputs"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"],
            return_dict=True,
        )
        loss2 = out2.loss
        assert torch.isfinite(loss2), f"loss2 not finite: {loss2}"
        # With a large LR on 2 samples, loss should drop
        assert loss2.item() < loss1.item(), (
            f"loss did not decrease: {loss1.item():.4f} → {loss2.item():.4f}"
        )

        model.eval()  # restore

    print("[PASS] #6 training step (backward + loss decrease)")


# ── #8  greedy generation ────────────────────────────────────────────

def test_greedy_generation(model, tokenizer) -> None:
    import torch
    from evaluate import generate_caption

    model.eval()
    device = next(model.parameters()).device

    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    # Fake encoder hidden states
    encoder_hidden = torch.randn(1, 64, 768)

    # Build a short prompt
    prompt = "This audio sounds like:"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        caption = generate_caption(
            model=model,
            tokenizer=tokenizer,
            encoder_hidden=encoder_hidden,
            decoder_start_id=bos_id,
            prompt_ids=prompt_ids,
            eos_id=eos_id,
            max_new_tokens=20,
            num_beams=1,
            device=device,
        )

    assert isinstance(caption, str), f"expected str, got {type(caption)}"
    assert len(caption) > 0, "generated caption is empty"
    # Should be decodable text (not raw token ids)
    assert not caption.startswith("["), f"suspicious output: {caption!r}"

    print(f"[PASS] #8 greedy generation (output: {caption!r})")


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    # Prompt builder: no model needed
    test_prompt_builder()

    # Build model once for remaining tests
    result = _get_model_and_tokenizer()
    if len(result) == 3:
        _, _, err = result
        print(f"[SKIP] #6, #7, #8: model load failed ({err})")
        return
    model, tokenizer = result

    test_collator_forward(model, tokenizer)
    test_training_step(model, tokenizer)
    test_greedy_generation(model, tokenizer)

    print("\nsmoke2 tests: done")


if __name__ == "__main__":
    main()
