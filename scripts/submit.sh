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
# K sets the captions per prompt (1 to 4, taken from the top-4 caches) and
# appears in the run name (a_paper_k2_s42), so each k gets its own run folder.
# DEDUP=1 retrieves from distinct captions only (repeated captions merged);
# the run name gets a _dedup suffix so both versions sit side by side.
# DRY_RUN=1 prints the sbatch command instead of submitting.
#
# LOCAL=1 runs the job right here instead of submitting it, for a single GPU
# machine without Slurm (a cloud instance, say). Activate the conda env
# first and set AUDIO_DIR for the preprocessing jobs; RECAP_ACCOUNT is not
# needed and extra sbatch options are ignored. The log is tee'd to logs/.
#   LOCAL=1 AUDIO_DIR=~/data/audiocaps_raw_audio scripts/submit.sh preprocess
#   LOCAL=1 K=2 DEDUP=1 scripts/submit.sh g2

set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
    cat <<'EOF'
jobs:
  preprocess    CLAP features, retrieval embeddings, empty and top-4 caches (full splits),
                top-4 caches both as-is and over distinct captions (_dedup)
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

LOCAL=${LOCAL:-0}
if [ "$LOCAL" != 1 ]; then
    : "${RECAP_ACCOUNT:?set RECAP_ACCOUNT (your charge account) in ~/.bashrc}"
fi
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
DEDUP_SUFFIX=""
if [ "${DEDUP:-0}" = 1 ]; then
    export CACHE_VARIANT=_dedup
    DEDUP_SUFFIX=_dedup
fi

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
        export RUN_NAME=${RUN_NAME:-g2_k${K:-4}${DEDUP_SUFFIX}_s$SEED} K=${K:-4} EPOCHS=${EPOCHS:-3} \
               DATA_DIR=data SKIP_SPICE=${SKIP_SPICE:-1} ;;
    run-a)
        script=scripts/phoenix_train.sbatch; time=24:00:00
        export RUN_NAME=${RUN_NAME:-a_paper_k${K:-4}${DEDUP_SUFFIX}_s$SEED} K=${K:-4} EPOCHS=${EPOCHS:-100} \
               PATIENCE=${PATIENCE:-5} DATA_DIR=data/paper_split CACHE_DIR=$PAPER_FEAT \
               SKIP_SPICE=${SKIP_SPICE:-0} \
               EVAL_SETS="val:data/paper_split/val.csv:val test:data/paper_split/test.csv:test" ;;
    run-b)
        script=scripts/phoenix_train.sbatch; time=24:00:00
        export RUN_NAME=${RUN_NAME:-b_full_k${K:-4}${DEDUP_SUFFIX}_s$SEED} K=${K:-4} EPOCHS=${EPOCHS:-100} \
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

mkdir -p logs
echo "job: $job  script: $script  ${RUN_NAME:+run: $RUNS_DIR/$RUN_NAME}"

if [ "$LOCAL" = 1 ]; then
    # No module system or cluster paths here: use the active env, keep the
    # Hugging Face cache in $HOME, and require the audio location explicitly.
    export SKIP_CONDA=1 PROJ=${PROJ:-$SCRATCH_RECAP}
    export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
    if [ "$script" = scripts/phoenix_preprocess.sbatch ] && [ -z "${REUSE_FROM:-}" ]; then
        : "${AUDIO_DIR:?set AUDIO_DIR to the raw audio folder}"
    fi
    [ $# -eq 0 ] || echo "note: LOCAL=1 ignores the sbatch options: $*"
    log=logs/recap-$job-$(date +%Y%m%d-%H%M%S).log
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "bash $script 2>&1 | tee $log"
        env | grep -E '^(RUN_NAME|K|EPOCHS|PATIENCE|SEED|DATA_DIR|FEAT_DIR|CACHE_DIR|CACHE_VARIANT|RUNS_DIR|REUSE_FROM|EVAL_SETS|SKIP_SPICE|AUDIO_DIR|HF_HOME)=' | sort
        exit 0
    fi
    echo "log: $log"
    bash "$script" 2>&1 | tee "$log"   # pipefail: exits with the job's status
    exit
fi

args=(--account="$RECAP_ACCOUNT" --qos="$QOS" --gres="gpu:$GPU:1"
      --cpus-per-task="$CPUS" --mem="$MEM" --time="$time" --job-name="recap-$job")
if [ -n "${NOTIFY_EMAIL:-}" ]; then
    args+=("--mail-type=BEGIN,END,FAIL" --mail-user="$NOTIFY_EMAIL")
fi
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'sbatch'; printf ' %q' "${args[@]}" "$@" "$script"; echo
    env | grep -E '^(RUN_NAME|K|EPOCHS|PATIENCE|SEED|DATA_DIR|FEAT_DIR|CACHE_DIR|CACHE_VARIANT|RUNS_DIR|REUSE_FROM|EVAL_SETS|SKIP_SPICE)=' | sort
    exit 0
fi
sbatch "${args[@]}" "$@" "$script"
