"""
score_predictions.py — Compute caption-quality metrics on the JSON written
by evaluate.py.

Reads {access_id: {references: [...], prediction: "..."}} and reports
BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE, SPIDEr.

References and predictions are first run through the PTB tokenizer
(lowercasing, splitting and dropping punctuation), as in the standard
COCO-caption / DCASE evaluation that published numbers use. Without it,
"background." and "background" (or "A" and "a") count as different words.
The tokenizer, METEOR and SPICE need Java on PATH (pycocoevalcap shells out
to JARs). --no_tokenize scores the raw strings instead; those numbers are
not comparable with published results.

If METEOR or SPICE fail for any reason, the script warns and continues with
the metrics it can compute. SPIDEr is reported only if both SPICE and CIDEr
succeeded. SPICE downloads Stanford CoreNLP models on its first run.

It also reports generation-health rates (no Java needed), which flag a model
that ignores the audio or copies retrieved captions even when the caption
metrics look plausible:
  unique_rate           distinct predictions / clips
  empty_rate            empty predictions
  mean_words            average prediction length
  eos_stop_rate         generation ended on EOS (needs evaluate.py stop_reason)
  max_tokens_rate       generation hit --max_new_tokens instead
  postprocess_cut_rate  evaluate.py had to cut the text at a stop string
  train_copy_rate       prediction is verbatim a training caption (--train_csv)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional


def _has_java() -> bool:
    return shutil.which("java") is not None


def _build_io(preds: dict) -> tuple[dict, dict]:
    gts = {k: list(v["references"]) for k, v in preds.items()}
    res = {k: [v["prediction"]] for k, v in preds.items()}
    return gts, res


def _norm_caption(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower()).rstrip(".!?,;: ")


def generation_health(preds: dict, train_captions: Optional[Iterable[str]] = None) -> dict:
    """Rates describing how the captions were generated (see module docstring)."""
    items = list(preds.values())
    n = len(items)
    texts = [str(v["prediction"]) for v in items]
    rate = lambda flags: sum(1 for f in flags if f) / n
    health = {
        "unique_rate": len(set(texts)) / n,
        "empty_rate": rate(t.strip() == "" for t in texts),
        "mean_words": sum(len(t.split()) for t in texts) / n,
    }
    if all("stop_reason" in v for v in items):
        health["eos_stop_rate"] = rate(v["stop_reason"] == "eos" for v in items)
        health["max_tokens_rate"] = rate(v["stop_reason"] == "max_new_tokens" for v in items)
    if all("cut_by_postprocessing" in v for v in items):
        health["postprocess_cut_rate"] = rate(v["cut_by_postprocessing"] for v in items)
    if train_captions is not None:
        train_set = {_norm_caption(c) for c in train_captions}
        health["train_copy_rate"] = rate(_norm_caption(t) in train_set for t in texts)
    return health


def _ptb_tokenize(gts: dict, res: dict) -> tuple[dict, dict]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    tok = PTBTokenizer()
    wrap = lambda d: {k: [{"caption": c} for c in v] for k, v in d.items()}
    return tok.tokenize(wrap(gts)), tok.tokenize(wrap(res))


def _safe_score(name: str, scorer_factory, gts: dict, res: dict, java_required: bool):
    """Run one scorer, returning (score_dict, ok). Never raises."""
    if java_required and not _has_java():
        print(f"[WARN] {name}: skipped (no `java` on PATH; install a JRE to enable).")
        return None, False
    try:
        scorer = scorer_factory()
        score, _ = scorer.compute_score(gts, res)
    except Exception as e:                            # noqa: BLE001
        print(f"[WARN] {name}: skipped due to error: {e!r}")
        return None, False
    return score, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Score RECAP predictions JSON.")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON from evaluate.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to write a metrics JSON.")
    parser.add_argument("--skip_java", action="store_true",
                        help="Skip METEOR and SPICE (Java-based) to prevent hangs/errors. "
                             "PTB tokenization still runs (it is fast).")
    parser.add_argument("--skip_spice", action="store_true",
                        help="Skip SPICE (and therefore SPIDEr); SPICE is slow and "
                             "memory-hungry.")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Training caption CSV; enables train_copy_rate.")
    parser.add_argument("--no_tokenize", action="store_true",
                        help="Score raw strings without PTB tokenization (previous "
                             "behaviour). Not comparable with published numbers.")
    args = parser.parse_args()

    preds = json.loads(Path(args.predictions).read_text())
    if not preds:
        raise SystemExit("Predictions JSON is empty.")
    gts, res = _build_io(preds)
    print(f"Scoring {len(gts)} clips.")

    train_captions = None
    if args.train_csv:
        import pandas as pd
        train_captions = pd.read_csv(args.train_csv)["caption"].dropna().astype(str).tolist()
    health = generation_health(preds, train_captions)

    if args.no_tokenize:
        print("[WARN] --no_tokenize: scoring raw strings; not comparable with published numbers.")
        tokenizer_name = "none"
    else:
        if not _has_java():
            raise SystemExit(
                "PTB tokenization needs `java` on PATH (install a JRE), or pass "
                "--no_tokenize to score raw strings."
            )
        gts, res = _ptb_tokenize(gts, res)
        tokenizer_name = "ptb"

    metrics: dict[str, float] = {}

    # BLEU-1..4 (pure Python)
    try:
        from pycocoevalcap.bleu.bleu import Bleu
    except Exception as e:                            # noqa: BLE001
        print(f"[WARN] BLEU: pycocoevalcap not installed ({e!r}). Run `pip install pycocoevalcap`.")
    else:
        bleu_scores, ok = _safe_score("BLEU", lambda: Bleu(4), gts, res, java_required=False)
        if ok:
            for i, s in enumerate(bleu_scores, start=1):
                metrics[f"BLEU-{i}"] = float(s)

    # ROUGE-L (pure Python)
    try:
        from pycocoevalcap.rouge.rouge import Rouge
    except Exception:
        pass
    else:
        rouge_score, ok = _safe_score("ROUGE-L", Rouge, gts, res, java_required=False)
        if ok:
            metrics["ROUGE-L"] = float(rouge_score)

    # CIDEr (pure Python)
    cider_score: Optional[float] = None
    try:
        from pycocoevalcap.cider.cider import Cider
    except Exception:
        pass
    else:
        c, ok = _safe_score("CIDEr", Cider, gts, res, java_required=False)
        if ok:
            cider_score = float(c)
            metrics["CIDEr"] = cider_score

    # METEOR (Java)
    try:
        from pycocoevalcap.meteor.meteor import Meteor
    except Exception:
        pass
    else:
        if args.skip_java:
            print("[INFO] METEOR: skipped via --skip_java.")
            ok = False
        else:
            m, ok = _safe_score("METEOR", Meteor, gts, res, java_required=True)
        if ok:
            metrics["METEOR"] = float(m)

    # SPICE (Java)
    spice_score: Optional[float] = None
    try:
        from pycocoevalcap.spice.spice import Spice
    except Exception:
        pass
    else:
        if args.skip_java or args.skip_spice:
            print("[INFO] SPICE: skipped via --skip_java/--skip_spice.")
            ok = False
        else:
            s, ok = _safe_score("SPICE", Spice, gts, res, java_required=True)
        if ok:
            spice_score = float(s)
            metrics["SPICE"] = spice_score

    # SPIDEr = (SPICE + CIDEr) / 2, only when both are available.
    if spice_score is not None and cider_score is not None:
        metrics["SPIDEr"] = (spice_score + cider_score) / 2.0
    else:
        print("[WARN] SPIDEr: skipped (requires both SPICE and CIDEr).")

    print(f"\n=== Metrics (tokenizer: {tokenizer_name}) ===")
    for name, val in metrics.items():
        print(f"  {name:8s}: {val:.4f}")
    print("\n=== Generation health ===")
    for name, val in health.items():
        print(f"  {name:20s}: {val:.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**metrics, **health, "tokenizer": tokenizer_name}, indent=2))
        print(f"\nWrote metrics to {out}")


if __name__ == "__main__":
    main()
