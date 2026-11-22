#!/bin/bash
# One-time preprocessing on PACE Phoenix: CLAP features for cross-attention,
# CLAP retrieval embeddings, and the retrieval caches (baseline + top-k).
#
# Fill in the three lines marked FILL IN, then from the repo root:
#   mkdir -p logs && sbatch scripts/phoenix_preprocess.sbatch
#
# Each step leaves a marker in $FEAT_DIR/.done/, so resubmitting after a
# crash, timeout or preemption skips the steps that already finished.
# Delete $FEAT_DIR/.done/<step> to force a step to run again.

#SBATCH --job-name=recap-preprocess
#SBATCH --account=FILL_IN            # FILL IN: charge account (sacctmgr show assoc user=$USER format=account,qos)
#SBATCH --qos=embers                 # FILL IN: embers (free, can be preempted) or inferno (paid)
#SBATCH --gres=gpu:A100:1            # FILL IN: GPU type name as listed by: sinfo -o "%G" | sort -u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# ---------- settings (override by exporting before sbatch) ----------
ANACONDA_MODULE=${ANACONDA_MODULE:-anaconda3}   # e.g. anaconda3/2023.03 (module avail anaconda3)
CONDA_ENV=${CONDA_ENV:-audio}
PROJ=${PROJ:-/storage/project/r-amadabhushi9-ext_em/hgupta312}
AUDIO_DIR=${AUDIO_DIR:-$PROJ/audiocaps_raw_audio}
FEAT_DIR=${FEAT_DIR:-$HOME/scratch/recap/features}
DATA_DIR=${DATA_DIR:-data}                      # CSVs: data or data/paper_split
K=${K:-4}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=${HF_HOME:-$PROJ/.cache/huggingface}

# ---------- environment ----------
if [ "${SKIP_CONDA:-0}" != 1 ]; then
    module load "$ANACONDA_MODULE"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p "$FEAT_DIR/.done"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-local}  start: $(date)"
echo "audio: $AUDIO_DIR"
echo "features: $FEAT_DIR"
echo "csvs: $DATA_DIR"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

step() {  # step <name> <command...>
    local name=$1; shift
    if [ -f "$FEAT_DIR/.done/$name" ]; then
        echo "=== skip $name (done)"
        return
    fi
    echo "=== $name  $(date +%T)"
    "$@"
    touch "$FEAT_DIR/.done/$name"
}

# 1. CLAP encoder features for cross-attention: (64, 768) per clip
for split in train val test; do
    step "features_$split" python data/audio_preprocess.py \
        --audio_dir "$AUDIO_DIR" --captions_csv "$DATA_DIR/$split.csv" \
        --output_dir "$FEAT_DIR" --dataset audiocaps --split "$split" \
        --batch_size 64 --num_workers "$WORKERS"
done

# 2. Retrieval datastore: CLAP text embeddings of the training captions
step text_train python -m retrieval.precompute_text_embeddings \
    --csv_path "$DATA_DIR/train.csv" --output_path "$FEAT_DIR/text_train.npy" \
    --dataset audiocaps --batch_size 64

# 3. Retrieval queries: CLAP audio embeddings per split
for split in train val test; do
    step "audio_emb_$split" python -m retrieval.precompute_audio_embeddings \
        --csv_path "$DATA_DIR/$split.csv" --audio_dir "$AUDIO_DIR" --dataset audiocaps \
        --output_path "$FEAT_DIR/audio_$split.npy" --ids_path "$FEAT_DIR/audio_${split}_ids.csv" \
        --batch_size 16 --num_workers "$WORKERS"
done

# 4. Caches: empty (no-retrieval baseline) and top-k
for split in train val test; do
    step "empty_$split" python -m retrieval.build_baseline_cache \
        --csv_path "$DATA_DIR/$split.csv" --dataset audiocaps \
        --output_path "$FEAT_DIR/empty_$split.json"
done
for split in train val test; do
    role="eval"; [ "$split" = train ] && role="train"
    step "top${K}_$split" python -m retrieval.build_retrieval_cache \
        --query_emb_path "$FEAT_DIR/audio_$split.npy" --query_ids_path "$FEAT_DIR/audio_${split}_ids.csv" \
        --datastore_emb_path "$FEAT_DIR/text_train.npy" \
        --datastore_csv_path "$DATA_DIR/train.csv" --datastore_dataset audiocaps \
        --output_path "$FEAT_DIR/retrieved_top${K}_$split.json" \
        --k "$K" --strategy topk --role "$role"
done

echo "=== all done  $(date)"
ls -lh "$FEAT_DIR"
