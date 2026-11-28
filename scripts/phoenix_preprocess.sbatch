#!/bin/bash
# One-time preprocessing on PACE Phoenix: CLAP features for cross-attention,
# CLAP retrieval embeddings, and the retrieval caches (baseline + top-k).
#
# Submit through scripts/submit.sh, which adds the account, QoS, GPU type,
# time limit and email notifications:
#   scripts/submit.sh preprocess       # full AudioCaps splits
#   scripts/submit.sh paper-caches     # paper split: reuses the features above
#
# Each step leaves a marker in $FEAT_DIR/.done/, so resubmitting after a
# crash, timeout or preemption skips the steps that already finished.
# Delete $FEAT_DIR/.done/<step> to force a step to run again.
#
# CLAP=<checkpoint> picks the CLAP model for the features and the retrieval
# embeddings (default laion/clap-htsat-fused). The checkpoint is recorded in
# $FEAT_DIR/clap.txt, and a run with a different CLAP for the same FEAT_DIR
# stops, so features from two checkpoints never mix.
#
# REUSE_FROM=<features dir> links that directory's HDF5 features and audio
# embeddings instead of recomputing them (they are keyed by clip, so a
# subset split like data/paper_split can share them) and only builds the
# text embeddings and caches for DATA_DIR.

#SBATCH --job-name=recap-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
CLAP=${CLAP:-laion/clap-htsat-fused}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=${HF_HOME:-$PROJ/.cache/huggingface}

# ---------- environment ----------
if [ "${SKIP_CONDA:-0}" != 1 ]; then
    module load "$ANACONDA_MODULE"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# torch compiles some GPU kernels on first use (Triton) and caches them, by
# default under ~/.triton, which every node shares. Two jobs compiling the
# same kernel at once then overwrite each other's files ("Stale file
# handle"). Under Slurm each job gets its own cache on the node's local disk.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${TRITON_CACHE_DIR:-}" ]; then
    export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/triton-cache-$SLURM_JOB_ID
    mkdir -p "$TRITON_CACHE_DIR"
fi
mkdir -p "$FEAT_DIR/.done"

# clap_of <features dir>: the CLAP checkpoint it was built with. Folders made
# before clap.txt existed were all built with the default checkpoint.
clap_of() {
    if [ -f "$1/clap.txt" ]; then cat "$1/clap.txt"
    elif [ -n "$(ls -A "$1/.done" 2>/dev/null)" ]; then echo laion/clap-htsat-fused
    fi
}
if [ -f "$FEAT_DIR/encoder.txt" ]; then
    echo "$FEAT_DIR holds other features ($(cat "$FEAT_DIR/encoder.txt")); use another FEAT_DIR" >&2
    exit 1
fi
existing=$(clap_of "$FEAT_DIR")
if [ -n "$existing" ] && [ "$existing" != "$CLAP" ]; then
    echo "$FEAT_DIR was built with CLAP=$existing, not $CLAP; use another FEAT_DIR" >&2
    exit 1
fi
if [ -n "${REUSE_FROM:-}" ]; then
    reused=$(clap_of "$REUSE_FROM")
    if [ "${reused:-laion/clap-htsat-fused}" != "$CLAP" ]; then
        echo "REUSE_FROM=$REUSE_FROM was built with CLAP=${reused:-laion/clap-htsat-fused}, not $CLAP" >&2
        exit 1
    fi
fi
echo "$CLAP" > "$FEAT_DIR/clap.txt"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-local}  start: $(date)"
echo "audio: $AUDIO_DIR"
echo "features: $FEAT_DIR"
echo "clap: $CLAP"
echo "csvs: $DATA_DIR"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
if [ "${ALLOW_CPU:-0}" != 1 ]; then
    python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available: compare the nvidia-smi CUDA version with the torch build')"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ -n "${REUSE_FROM:-}" ]; then
    echo "reusing features and audio embeddings from $REUSE_FROM"
    for split in train val test; do
        for f in "audiocaps_$split.hdf5" "audio_$split.npy" "audio_${split}_ids.csv" "empty_$split.json"; do
            [ -e "$REUSE_FROM/$f" ] || { echo "missing $REUSE_FROM/$f" >&2; exit 1; }
            ln -sfn "$REUSE_FROM/$f" "$FEAT_DIR/$f"
        done
        touch "$FEAT_DIR/.done/features_$split" "$FEAT_DIR/.done/audio_emb_$split" "$FEAT_DIR/.done/empty_$split"
    done
fi

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
        --output_dir "$FEAT_DIR" --dataset audiocaps --split "$split" --encoder_name "$CLAP" \
        --batch_size 64 --num_workers "$WORKERS"
done

# 2. Retrieval datastore: CLAP text embeddings of the training captions
step text_train python -m retrieval.precompute_text_embeddings \
    --csv_path "$DATA_DIR/train.csv" --output_path "$FEAT_DIR/text_train.npy" \
    --dataset audiocaps --batch_size 64 --clap_model "$CLAP"

# 3. Retrieval queries: CLAP audio embeddings per split
for split in train val test; do
    step "audio_emb_$split" python -m retrieval.precompute_audio_embeddings \
        --csv_path "$DATA_DIR/$split.csv" --audio_dir "$AUDIO_DIR" --dataset audiocaps \
        --output_path "$FEAT_DIR/audio_$split.npy" --ids_path "$FEAT_DIR/audio_${split}_ids.csv" \
        --batch_size 16 --num_workers "$WORKERS" --clap_model "$CLAP"
done

# 4. Caches: empty (no-retrieval baseline), top-k, and top-k over distinct
#    captions (_dedup: repeated captions merged so the k are all different)
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
    step "top${K}_dedup_$split" python -m retrieval.build_retrieval_cache \
        --query_emb_path "$FEAT_DIR/audio_$split.npy" --query_ids_path "$FEAT_DIR/audio_${split}_ids.csv" \
        --datastore_emb_path "$FEAT_DIR/text_train.npy" \
        --datastore_csv_path "$DATA_DIR/train.csv" --datastore_dataset audiocaps \
        --output_path "$FEAT_DIR/retrieved_top${K}_dedup_$split.json" \
        --k "$K" --strategy topk --role "$role" --dedupe_captions
done

echo "=== all done  $(date)"
ls -lh "$FEAT_DIR"
