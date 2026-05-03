"""
score_predictions.py — Compute caption-quality metrics on the JSON written
by evaluate.py.

Reads {access_id: {references: [...], prediction: "..."}} and reports
BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE, SPIDEr.

METEOR and SPICE require Java on PATH (pycocoevalcap shells out to JARs).
If Java is unavailable or those scorers fail for any reason, the script
warns and continues with the metrics it can compute. SPIDEr is reported
only if both SPICE and CIDEr succeeded.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional


def _has_java() -> bool:
    return shutil.which("java") is not None


def _build_io(preds: dict) -> tuple[dict, dict]:
    gts = {k: list(v["references"]) for k, v in preds.items()}
    res = {k: [v["prediction"]] for k, v in preds.items()}
    return gts, res


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
                        help="Skip METEOR and SPICE (Java-based) to prevent hangs/errors.")
    args = parser.parse_args()

    preds = json.loads(Path(args.predictions).read_text())
    if not preds:
        raise SystemExit("Predictions JSON is empty.")
    gts, res = _build_io(preds)
    print(f"Scoring {len(gts)} clips.")

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
        if args.skip_java:
            print("[INFO] SPICE: skipped via --skip_java.")
            ok = False
        else:
            s, ok = _safe_score("SPICE", Spice, gts, res, java_required=True)
        if ok:
            spice_score = float(s)
            metrics["SPICE"] = spice_score

    # SPIDEr = (SPICE + CIDEr) / 2 — only when both are available.
    if spice_score is not None and cider_score is not None:
        metrics["SPIDEr"] = (spice_score + cider_score) / 2.0
    else:
        print("[WARN] SPIDEr: skipped (requires both SPICE and CIDEr).")

    print("\n=== Metrics ===")
    for name, val in metrics.items():
        print(f"  {name:8s}: {val:.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"\nWrote metrics to {out}")


if __name__ == "__main__":
    main()
