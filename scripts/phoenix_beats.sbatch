#!/bin/bash
# BEATs features for cross-attention, in place of CLAP's. Retrieval stays on
# CLAP: train with FEAT_DIR pointing here and CACHE_DIR at a CLAP features
# folder that holds the top-k caches (for example features_unfused).
#
# Submit through scripts/submit.sh:
#   BEATS_CKPT=~/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
#   scripts/submit.sh beats-features
#
# Writes $FEAT_DIR/audiocaps_{train,val,test}.hdf5 for the clips in DATA_DIR
# (default the paper split) and the empty caches that run-c (no retrieval)
# reads. The checkpoint and pooling are recorded in $FEAT_DIR/encoder.txt; a
# run with a different checkpoint or POOL for the same FEAT_DIR stops, so
# features never mix. Each step leaves a marker in $FEAT_DIR/.done/.

#SBATCH --job-name=recap-beats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

ANACONDA_MODULE=${ANACONDA_MODULE:-anaconda3}
CONDA_ENV=${CONDA_ENV:-audio}
PROJ=${PROJ:-/storage/project/r-amadabhushi9-ext_em/hgupta312}
AUDIO_DIR=${AUDIO_DIR:-$PROJ/audiocaps_raw_audio}
FEAT_DIR=${FEAT_DIR:-$HOME/scratch/recap/features_beats}
DATA_DIR=${DATA_DIR:-data/paper_split}
BEATS_CKPT=${BEATS_CKPT:?set BEATS_CKPT to the BEATs .pt checkpoint}
POOL=${POOL:-4}             # frequency bands averaged per time step: 4 -> 124 vectors, 1 -> 496
WORKERS=${SLURM_CPUS_PER_TASK:-8}

if [ "${SKIP_CONDA:-0}" != 1 ]; then
    module load "$ANACONDA_MODULE"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f "$BEATS_CKPT" ] || { echo "BEATS_CKPT=$BEATS_CKPT not found" >&2; exit 1; }
mkdir -p "$FEAT_DIR/.done"

# A CLAP features folder must not receive BEATs features, and a BEATs folder
# must not mix checkpoints or pooling.
if [ -f "$FEAT_DIR/clap.txt" ]; then
    echo "$FEAT_DIR holds CLAP features ($(cat "$FEAT_DIR/clap.txt")); use another FEAT_DIR" >&2
    exit 1
fi
encoder="beats $(basename "$BEATS_CKPT") pool=$POOL"
if [ -f "$FEAT_DIR/encoder.txt" ] && [ "$(cat "$FEAT_DIR/encoder.txt")" != "$encoder" ]; then
    echo "$FEAT_DIR was built with '$(cat "$FEAT_DIR/encoder.txt")', not '$encoder'; use another FEAT_DIR" >&2
    exit 1
fi
if [ ! -f "$FEAT_DIR/encoder.txt" ] && [ -n "$(ls -A "$FEAT_DIR/.done")" ]; then
    echo "$FEAT_DIR has finished steps but no encoder.txt; use another FEAT_DIR" >&2
    exit 1
fi
echo "$encoder" > "$FEAT_DIR/encoder.txt"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-local}  start: $(date)"
echo "audio: $AUDIO_DIR"
echo "features: $FEAT_DIR ($encoder)"
echo "csvs: $DATA_DIR"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if [ "${ALLOW_CPU:-0}" != 1 ]; then
    python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available: compare the nvidia-smi CUDA version with the torch build')"
fi
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

# 0. The weights must load and recognise sounds: top AudioSet labels for a
#    few val clips, next to their captions (fine-tuned checkpoints only).
python data/extract_beats_features.py --checkpoint "$BEATS_CKPT" --audio_dir "$AUDIO_DIR" \
    --captions_csv "$DATA_DIR/val.csv" --output_dir "$FEAT_DIR" --dataset audiocaps --check 3

# 1. BEATs features for cross-attention
for split in train val test; do
    step "features_$split" python data/extract_beats_features.py --checkpoint "$BEATS_CKPT" \
        --audio_dir "$AUDIO_DIR" --captions_csv "$DATA_DIR/$split.csv" --output_dir "$FEAT_DIR" \
        --dataset audiocaps --split "$split" --pool "$POOL" --batch_size 32 --num_workers "$WORKERS"
done

# 2. Empty caches for the no-retrieval baseline (run-c reads them from FEAT_DIR)
for split in train val test; do
    step "empty_$split" python -m retrieval.build_baseline_cache \
        --csv_path "$DATA_DIR/$split.csv" --dataset audiocaps \
        --output_path "$FEAT_DIR/empty_$split.json"
done

echo "=== all done  $(date)"
ls -lh "$FEAT_DIR"
