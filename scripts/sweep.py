"""
sweep.py — Train / evaluate / score RECAP across a cartesian product
of training, generation, and scoring hyperparameters.

Layout:
  runs_dir/
    top{k}_pd{pd}_wd{wd}_ep{ep}_lr{lr}/        # one per train config
      <trained model files>
      predictions/
        test_preds_nb{nb}_nrngram{nrn}_lp{lp}.json
      metrics/
        metrics_skip_java.json
        metrics_with_java.json

Retrieval caches are expected at:
  retrieval_cache_dir/retrieved_top{k}_{train,val,test}.json

For k=0 the cache should contain empty lists per access_id; --no-rag is
passed to train.py so the model trains on the baseline (no retrieval)
prompt. Eval still uses the k=0 cache.

Subprocesses are invoked rather than imports so each training run gets a
fresh CUDA process.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------- formatting helpers --------------------------------------------

def _fmt(v) -> str:
    """Stable, human-readable string for use in filenames/dirs.
    Whole-number floats render without the trailing '.0' (10.0 -> '10'),
    fractions keep all digits ('0.05' -> '0.05'), small floats use
    scientific notation (5e-5 -> '5e-05')."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def run_name(k: int, pd: float, wd: float, ep: float, lr: float) -> str:
    return (
        f"top{_fmt(k)}_pd{_fmt(pd)}_wd{_fmt(wd)}"
        f"_ep{_fmt(ep)}_lr{_fmt(lr)}"
    )


def pred_name(nb: int, nrn: int, lp: float) -> str:
    return f"test_preds_nb{_fmt(nb)}_nrngram{_fmt(nrn)}_lp{_fmt(lp)}.json"


def metrics_name(skip_java: bool) -> str:
    return "metrics_skip_java.json" if skip_java else "metrics_with_java.json"


def cache_path(retrieval_cache_dir: Path, k: int, split: str) -> Path:
    return retrieval_cache_dir / f"retrieved_top{k}_{split}.json"


# ---------- subprocess helpers --------------------------------------------

def _run(cmd: list[str], dry_run: bool) -> None:
    pretty = " ".join(cmd)
    print(f"\n$ {pretty}", flush=True)
    if dry_run:
        return
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    print(f"[done in {time.time() - t0:.1f}s]", flush=True)


# ---------- stages --------------------------------------------------------

def stage_train(
    *, run_dir: Path, k: int, pd: float, wd: float, ep: float, lr: float,
    train_csv: str, train_hdf5: str,
    val_csv: Optional[str], val_hdf5: Optional[str],
    retrieval_cache_dir: Path, dataset: str,
    no_best_model: bool,
    skip_existing: bool, dry_run: bool,
) -> None:
    if skip_existing and (run_dir / "config.json").exists():
        print(f"[skip train] {run_dir.name} (config.json already present)")
        return

    cmd = [
        sys.executable, "train.py",
        "--train_csv", train_csv,
        "--train_hdf5", train_hdf5,
        "--train_retrieval_cache", str(cache_path(retrieval_cache_dir, k, "train")),
        "--dataset", dataset,
        "--output_dir", str(run_dir),
        "--prompt_dropout", _fmt(pd),
        "--weight_decay", _fmt(wd),
        "--num_train_epochs", _fmt(ep),
        "--learning_rate", _fmt(lr),
    ]
    if val_csv and val_hdf5:
        cmd += [
            "--val_csv", val_csv,
            "--val_hdf5", val_hdf5,
            "--val_retrieval_cache", str(cache_path(retrieval_cache_dir, k, "val")),
        ]
        if not no_best_model:
            cmd.append("--load_best_model_at_end")
    if k == 0:
        cmd.append("--no-rag")
    else:
        cmd += ["--rag", "--retrieval_k", _fmt(k), "--retrieval_strategy", "topk"]

    _run(cmd, dry_run)


def stage_eval(
    *, run_dir: Path, k: int, nb: int, nrn: int, lp: float,
    test_csv: str, test_hdf5: str,
    retrieval_cache_dir: Path, dataset: str,
    skip_existing: bool, dry_run: bool,
) -> Path:
    out_path = run_dir / "predictions" / pred_name(nb, nrn, lp)
    if skip_existing and out_path.exists():
        print(f"[skip eval]  {run_dir.name}/{out_path.name} (exists)")
        return out_path

    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", str(run_dir),
        "--csv_path", test_csv,
        "--hdf5_path", test_hdf5,
        "--retrieval_cache", str(cache_path(retrieval_cache_dir, k, "test")),
        "--dataset", dataset,
        "--output_path", str(out_path),
        "--num_beams", _fmt(nb),
        "--no_repeat_ngram_size", _fmt(nrn),
        "--length_penalty", _fmt(lp),
    ]
    _run(cmd, dry_run)
    return out_path


def stage_score(
    *, run_dir: Path, pred_path: Path, skip_java: bool,
    skip_existing: bool, dry_run: bool,
) -> None:
    out_path = run_dir / "metrics" / metrics_name(skip_java)
    if skip_existing and out_path.exists():
        print(f"[skip score] {run_dir.name}/{out_path.name} (exists)")
        return

    cmd = [
        sys.executable, "score_predictions.py",
        "--predictions", str(pred_path),
        "--output", str(out_path),
    ]
    if skip_java:
        cmd.append("--skip_java")
    _run(cmd, dry_run)


# ---------- CLI -----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep RECAP train/eval/score.")

    # Paths
    p.add_argument("--feature_dir", type=str, required=True,
                   help="Directory holding cached encoder HDF5s.")
    p.add_argument("--runs_dir", type=str, required=True,
                   help="Output directory; one subdir per run is created here.")
    p.add_argument("--retrieval_cache_dir", type=str, required=True,
                   help="Directory with retrieved_top{k}_{split}.json files.")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--train_hdf5", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--test_hdf5", type=str, required=True)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--val_hdf5", type=str, default=None)
    p.add_argument("--dataset", type=str, default="audiocaps")

    # Train sweep
    p.add_argument("--ks", type=int, nargs="+", required=True,
                   help="Top-k retrieval values. 0 means --no-rag.")
    p.add_argument("--prompt_dropouts", type=float, nargs="+", default=[0.0])
    p.add_argument("--weight_decays", type=float, nargs="+", default=[0.0])
    p.add_argument("--epochs", type=float, nargs="+", default=[10.0])
    p.add_argument("--lrs", type=float, nargs="+", default=[5e-5])

    # Eval sweep
    p.add_argument("--num_beams", type=int, nargs="+", default=[1])
    p.add_argument("--no_repeat_ngram_sizes", type=int, nargs="+", default=[0])
    p.add_argument("--length_penalties", type=float, nargs="+", default=[1.0])

    # Score sweep
    p.add_argument("--score_skip_java", action="store_true",
                   help="Produce metrics_skip_java.json (no METEOR/SPICE).")
    p.add_argument("--score_with_java", action="store_true",
                   help="Produce metrics_with_java.json (full metric set).")

    # Stage selection / control
    p.add_argument("--stages", type=str, nargs="+",
                   default=["train", "eval", "score"],
                   choices=["train", "eval", "score"])
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip a stage when its output already exists.")
    p.add_argument("--no_best_model", action="store_true",
                   help="When a val set is provided, do NOT pass "
                        "--load_best_model_at_end to train.py. Periodic eval "
                        "still runs and is logged, but the saved model is the "
                        "final training-step weights instead of the lowest-"
                        "eval_loss checkpoint.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing.")

    return p.parse_args()


def _validate(args) -> None:
    if (args.val_csv is None) ^ (args.val_hdf5 is None):
        raise SystemExit("Provide both --val_csv and --val_hdf5, or neither.")
    if "score" in args.stages and not (args.score_skip_java or args.score_with_java):
        raise SystemExit(
            "Stage 'score' requested but neither --score_skip_java nor "
            "--score_with_java was set; nothing to do."
        )
    rdir = Path(args.retrieval_cache_dir)
    splits = ["train", "test"] + (["val"] if args.val_csv else [])
    missing = [
        cache_path(rdir, k, s) for k in args.ks for s in splits
        if not cache_path(rdir, k, s).exists()
    ]
    if missing:
        msg = "\n  ".join(str(p) for p in missing)
        raise SystemExit(f"Missing retrieval cache files:\n  {msg}")


def main() -> None:
    args = parse_args()
    _validate(args)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    rcache_dir = Path(args.retrieval_cache_dir)

    train_combos = list(itertools.product(
        args.ks, args.prompt_dropouts, args.weight_decays,
        args.epochs, args.lrs,
    ))
    eval_combos = list(itertools.product(
        args.num_beams, args.no_repeat_ngram_sizes, args.length_penalties,
    ))
    score_modes: list[bool] = []
    if args.score_skip_java:
        score_modes.append(True)
    if args.score_with_java:
        score_modes.append(False)

    print(
        f"[sweep] {len(train_combos)} train configs × {len(eval_combos)} eval "
        f"configs × {len(score_modes)} score modes"
    )

    for k, pd, wd, ep, lr in train_combos:
        rd = runs_dir / run_name(k, pd, wd, ep, lr)
        rd.mkdir(parents=True, exist_ok=True)
        print(f"\n========== {rd.name} ==========")

        if "train" in args.stages:
            stage_train(
                run_dir=rd, k=k, pd=pd, wd=wd, ep=ep, lr=lr,
                train_csv=args.train_csv, train_hdf5=args.train_hdf5,
                val_csv=args.val_csv, val_hdf5=args.val_hdf5,
                retrieval_cache_dir=rcache_dir, dataset=args.dataset,
                no_best_model=args.no_best_model,
                skip_existing=args.skip_existing, dry_run=args.dry_run,
            )

        if "eval" not in args.stages and "score" not in args.stages:
            continue

        for nb, nrn, lp in eval_combos:
            pred_path = rd / "predictions" / pred_name(nb, nrn, lp)
            if "eval" in args.stages:
                pred_path = stage_eval(
                    run_dir=rd, k=k, nb=nb, nrn=nrn, lp=lp,
                    test_csv=args.test_csv, test_hdf5=args.test_hdf5,
                    retrieval_cache_dir=rcache_dir, dataset=args.dataset,
                    skip_existing=args.skip_existing, dry_run=args.dry_run,
                )

            if "score" in args.stages:
                if not args.dry_run and not pred_path.exists():
                    print(f"[skip score] {pred_path.name} not found "
                          "(run eval stage first)")
                    continue
                for skip_java in score_modes:
                    stage_score(
                        run_dir=rd, pred_path=pred_path, skip_java=skip_java,
                        skip_existing=args.skip_existing, dry_run=args.dry_run,
                    )


if __name__ == "__main__":
    main()
