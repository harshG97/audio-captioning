#!/bin/bash
# Train one RECAP configuration, then evaluate, score and run the health checks.
# Submit through scripts/submit.sh, which sets the account, GPU, time limit,
# email and the run settings below. Run `scripts/submit.sh list` for presets.
#
# Steps (each leaves a marker in $RUN_DIR/.done/, so a resubmitted job skips
# finished steps; training itself resumes from its last checkpoint):
#   1. train            train.py, evaluating and saving once per epoch
#   2. per eval set     evaluate.py -> score_predictions.py (health metrics,
#                       train-copy rate)
#   3. ablate           ablate_audio.py on the first eval set: loss with the
#                       audio intact, zeroed and shuffled
#   4. copy baseline    (K > 0) output the top retrieved caption, scored the
#                       same way, so the model can be compared against copying
#   5. summary          one table with the numbers that decide pass or fail

#SBATCH --job-name=recap-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# ---------- settings (submit.sh exports these; defaults shown) ----------
ANACONDA_MODULE=${ANACONDA_MODULE:-anaconda3}
CONDA_ENV=${CONDA_ENV:-audio}
PROJ=${PROJ:-/storage/project/r-amadabhushi9-ext_em/hgupta312}
FEAT_DIR=${FEAT_DIR:-$HOME/scratch/recap/features}        # HDF5 features
CACHE_DIR=${CACHE_DIR:-$FEAT_DIR}                          # retrieval caches
RUNS_DIR=${RUNS_DIR:-$HOME/scratch/recap/runs}
DATA_DIR=${DATA_DIR:-data}                                 # train/val CSVs
RUN_NAME=${RUN_NAME:?set RUN_NAME}
K=${K:-4}                     # captions in the prompt; 0 = no retrieval
CACHE_K=${CACHE_K:-4}         # k the top-k caches were built with (>= K)
CACHE_VARIANT=${CACHE_VARIANT:-}   # "" or "_dedup" (distinct captions only)
EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-32}
LR=${LR:-5e-5}
PROMPT_DROPOUT=${PROMPT_DROPOUT:-0.0}
PATIENCE=${PATIENCE:-0}       # early stopping patience in epochs; 0 = off
SEED=${SEED:-42}
SKIP_SPICE=${SKIP_SPICE:-1}
ENCODER_NAME=${ENCODER_NAME:-laion/clap-htsat-fused}
DECODER_NAME=${DECODER_NAME:-gpt2}
# Space-separated name:csv:split triples; split picks the HDF5 and cache files.
EVAL_SETS=${EVAL_SETS:-val:$DATA_DIR/val.csv:val}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=${HF_HOME:-$PROJ/.cache/huggingface}

RUN_DIR=$RUNS_DIR/$RUN_NAME

# ---------- environment ----------
if [ "${SKIP_CONDA:-0}" != 1 ]; then
    module load "$ANACONDA_MODULE"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$RUN_DIR/.done"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-local}  start: $(date)"
echo "run: $RUN_DIR"
echo "K=$K cache=top${CACHE_K}${CACHE_VARIANT} epochs=$EPOCHS batch=$BATCH lr=$LR prompt_dropout=$PROMPT_DROPOUT patience=$PATIENCE seed=$SEED"
echo "train csv: $DATA_DIR/train.csv  features: $FEAT_DIR  caches: $CACHE_DIR"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
if [ "${ALLOW_CPU:-0}" != 1 ]; then
    python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available: compare the nvidia-smi CUDA version with the torch build')"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

step() {  # step <name> <command...>
    local name=$1; shift
    if [ -f "$RUN_DIR/.done/$name" ]; then
        echo "=== skip $name (done)"
        return
    fi
    echo "=== $name  $(date +%T)"
    "$@"
    touch "$RUN_DIR/.done/$name"
}

cache() {  # cache <split>
    if [ "$K" -gt 0 ]; then
        echo "$CACHE_DIR/retrieved_top${CACHE_K}${CACHE_VARIANT}_$1.json"
    else
        echo "$FEAT_DIR/empty_$1.json"
    fi
}

if [ "$K" -gt 0 ]; then
    RAG_ARGS=(--rag --retrieval_k "$K" --retrieval_strategy topk)
else
    RAG_ARGS=(--no-rag)
fi
EARLY_ARGS=()
[ "$PATIENCE" -gt 0 ] && EARLY_ARGS=(--early_stopping_patience "$PATIENCE")

# 1. Train (resumes from the last checkpoint if the job was cut short)
step train python train.py \
    --train_csv "$DATA_DIR/train.csv" --train_hdf5 "$FEAT_DIR/audiocaps_train.hdf5" \
    --train_retrieval_cache "$(cache train)" \
    --val_csv "$DATA_DIR/val.csv" --val_hdf5 "$FEAT_DIR/audiocaps_val.hdf5" \
    --val_retrieval_cache "$(cache val)" \
    --output_dir "$RUN_DIR" \
    --encoder_name "$ENCODER_NAME" --decoder_name "$DECODER_NAME" \
    "${RAG_ARGS[@]}" --prompt_dropout "$PROMPT_DROPOUT" \
    --num_train_epochs "$EPOCHS" --per_device_train_batch_size "$BATCH" \
    --per_device_eval_batch_size "$BATCH" --learning_rate "$LR" --seed "$SEED" \
    --eval_save_strategy epoch --load_best_model_at_end --save_total_limit 2 \
    "${EARLY_ARGS[@]}" --resume --bf16 --num_workers "$WORKERS"

SPICE_ARGS=()
[ "$SKIP_SPICE" = 1 ] && SPICE_ARGS=(--skip_spice)

# 2. Generate and score on each eval set
for spec in $EVAL_SETS; do
    IFS=: read -r name csv split <<< "$spec"
    step "eval_$name" python evaluate.py \
        --checkpoint "$RUN_DIR" --csv_path "$csv" \
        --hdf5_path "$FEAT_DIR/audiocaps_$split.hdf5" --retrieval_cache "$(cache "$split")" \
        --output_path "$RUN_DIR/preds_$name.json"
    step "score_$name" python score_predictions.py \
        --predictions "$RUN_DIR/preds_$name.json" --output "$RUN_DIR/metrics_$name.json" \
        --train_csv "$DATA_DIR/train.csv" "${SPICE_ARGS[@]}"
done

# 3. Audio ablation on the first eval set
IFS=: read -r name csv split <<< "${EVAL_SETS%% *}"
step ablate bash -c "set -o pipefail; python scripts/ablate_audio.py --checkpoint '$RUN_DIR' --csv_path '$csv' \
    --hdf5_path '$FEAT_DIR/audiocaps_$split.hdf5' --retrieval_cache '$(cache "$split")' \
    --batch_size '$BATCH' --num_workers '$WORKERS' | tee '$RUN_DIR/ablate_$name.txt'"

# 4. Copy baseline: output the top retrieved caption for every clip
if [ "$K" -gt 0 ]; then
    for spec in $EVAL_SETS; do
        IFS=: read -r name csv split <<< "$spec"
        step "copy_$name" python scripts/copy_retrieval_baseline.py \
            --csv_path "$csv" --retrieval_cache "$(cache "$split")" --rank 1 \
            --hdf5_path "$FEAT_DIR/audiocaps_$split.hdf5" --output_path "$RUN_DIR/copy_preds_$name.json"
        step "copy_score_$name" python score_predictions.py \
            --predictions "$RUN_DIR/copy_preds_$name.json" --output "$RUN_DIR/copy_metrics_$name.json" \
            --train_csv "$DATA_DIR/train.csv" "${SPICE_ARGS[@]}"
    done
fi

# 5. Summary
echo "=== summary  $(date +%T)"
python - "$RUN_DIR" $EVAL_SETS <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
cols = ["CIDEr", "SPIDEr", "METEOR", "eos_stop_rate", "max_tokens_rate",
        "unique_rate", "train_copy_rate", "mean_words"]
print(f"{'':22s}" + "".join(f"{c:>16s}" for c in cols))
for spec in sys.argv[2:]:
    name = spec.split(":")[0]
    for label, f in [(f"model / {name}", f"metrics_{name}.json"),
                     (f"copy top-1 / {name}", f"copy_metrics_{name}.json")]:
        p = run / f
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        cells = "".join(f"{m[c]:>16.3f}" if isinstance(m.get(c), (int, float)) else f"{'-':>16s}"
                        for c in cols)
        print(f"{label:22s}{cells}")
for p in sorted(run.glob("ablate_*.txt")):
    print()
    print(f"audio ablation ({p.stem.removeprefix('ablate_')}):")
    lines = p.read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("condition")), 0)
    print("\n".join(lines[start:]))
EOF
echo "=== all done  $(date)"
