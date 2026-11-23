#!/bin/bash
# Submit a RECAP job to Slurm with the right resources and settings.
#
#   scripts/submit.sh <job> [extra sbatch options]
#   scripts/submit.sh list
#
# Cluster settings come from your environment (put them in ~/.bashrc), so no
# account names or email addresses live in the repo:
#   RECAP_ACCOUNT   charge account (required), e.g. from pace-quota
#   RECAP_QOS       inferno (paid, default) or embers (free, preemptible)
#   RECAP_GPU       GPU type as listed by `sinfo -o "%G"` (default a100)
#   RECAP_CPUS      CPU cores (default 8; capped at 4 on l40s nodes)
#   RECAP_MEM       memory (default 64G)
#   NOTIFY_EMAIL    if set, email on job start, end and failure
#
# Any run setting can be overridden for one submission, for example
#   SEED=1 scripts/submit.sh run-a
#   K=2 EPOCHS=5 scripts/submit.sh g2
# DRY_RUN=1 prints the sbatch command instead of submitting.

set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
    cat <<'EOF'
jobs:
  preprocess    CLAP features, retrieval embeddings, empty and top-4 caches (full splits)
  paper-caches  top-4 caches with the paper split's train captions as the datastore
                (reuses the preprocess features; run after preprocess)
  g1            sanity gate 1: k=0, 1 epoch on the full train set, checked on val
  g2            sanity gate 2: k=4, 3 epochs on the full train set, checked on val
  run-a         paper-faithful: paper split, k=4, up to 100 epochs, early stopping
  run-b         same settings on the full 91k train set, tested on both test sets
  run-c         no-retrieval baseline: paper split, k=0, same settings as run-a
EOF
}

job=${1:-}
[ -n "$job" ] || { usage; exit 1; }
shift
[ "$job" = list ] && { usage; exit 0; }

: "${RECAP_ACCOUNT:?set RECAP_ACCOUNT (your charge account) in ~/.bashrc}"
QOS=${RECAP_QOS:-inferno}
GPU=${RECAP_GPU:-a100}
CPUS=${RECAP_CPUS:-8}
MEM=${RECAP_MEM:-64G}
if [[ $GPU == l40s* ]] && [ "$CPUS" -gt 4 ]; then
    echo "note: l40s nodes allow at most 4 CPUs per GPU; using 4"
    CPUS=4
fi

SCRATCH_RECAP=${SCRATCH_RECAP:-$HOME/scratch/recap}
export FEAT_DIR=${FEAT_DIR:-$SCRATCH_RECAP/features}
export RUNS_DIR=${RUNS_DIR:-$SCRATCH_RECAP/runs}
export SEED=${SEED:-42}
PAPER_FEAT=$SCRATCH_RECAP/features_paper

case $job in
    preprocess)
        script=scripts/phoenix_preprocess.sbatch; time=08:00:00 ;;
    paper-caches)
        script=scripts/phoenix_preprocess.sbatch; time=01:00:00
        export REUSE_FROM=$FEAT_DIR FEAT_DIR=$PAPER_FEAT DATA_DIR=data/paper_split ;;
    g1)
        script=scripts/phoenix_train.sbatch; time=03:00:00
        export RUN_NAME=${RUN_NAME:-g1_k0_s$SEED} K=${K:-0} EPOCHS=${EPOCHS:-1} \
               DATA_DIR=data SKIP_SPICE=${SKIP_SPICE:-1} ;;
    g2)
        script=scripts/phoenix_train.sbatch; time=06:00:00
        export RUN_NAME=${RUN_NAME:-g2_k4_s$SEED} K=${K:-4} EPOCHS=${EPOCHS:-3} \
               DATA_DIR=data SKIP_SPICE=${SKIP_SPICE:-1} ;;
    run-a)
        script=scripts/phoenix_train.sbatch; time=24:00:00
        export RUN_NAME=${RUN_NAME:-a_paper_k4_s$SEED} K=${K:-4} EPOCHS=${EPOCHS:-100} \
               PATIENCE=${PATIENCE:-5} DATA_DIR=data/paper_split CACHE_DIR=$PAPER_FEAT \
               SKIP_SPICE=${SKIP_SPICE:-0} \
               EVAL_SETS="val:data/paper_split/val.csv:val test:data/paper_split/test.csv:test" ;;
    run-b)
        script=scripts/phoenix_train.sbatch; time=24:00:00
        export RUN_NAME=${RUN_NAME:-b_full_k4_s$SEED} K=${K:-4} EPOCHS=${EPOCHS:-100} \
               PATIENCE=${PATIENCE:-5} DATA_DIR=data SKIP_SPICE=${SKIP_SPICE:-0} \
               EVAL_SETS="val:data/val.csv:val test:data/test.csv:test papertest:data/paper_split/test.csv:test" ;;
    run-c)
        script=scripts/phoenix_train.sbatch; time=24:00:00
        export RUN_NAME=${RUN_NAME:-c_paper_k0_s$SEED} K=${K:-0} EPOCHS=${EPOCHS:-100} \
               PATIENCE=${PATIENCE:-5} DATA_DIR=data/paper_split SKIP_SPICE=${SKIP_SPICE:-0} \
               EVAL_SETS="val:data/paper_split/val.csv:val test:data/paper_split/test.csv:test" ;;
    *)
        echo "unknown job: $job" >&2; usage >&2; exit 1 ;;
esac

args=(--account="$RECAP_ACCOUNT" --qos="$QOS" --gres="gpu:$GPU:1"
      --cpus-per-task="$CPUS" --mem="$MEM" --time="$time" --job-name="recap-$job")
if [ -n "${NOTIFY_EMAIL:-}" ]; then
    args+=("--mail-type=BEGIN,END,FAIL" --mail-user="$NOTIFY_EMAIL")
fi

mkdir -p logs
echo "job: $job  script: $script  ${RUN_NAME:+run: $RUNS_DIR/$RUN_NAME}"
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'sbatch'; printf ' %q' "${args[@]}" "$@" "$script"; echo
    env | grep -E '^(RUN_NAME|K|EPOCHS|PATIENCE|SEED|DATA_DIR|FEAT_DIR|CACHE_DIR|RUNS_DIR|REUSE_FROM|EVAL_SETS|SKIP_SPICE)=' | sort
    exit 0
fi
sbatch "${args[@]}" "$@" "$script"
