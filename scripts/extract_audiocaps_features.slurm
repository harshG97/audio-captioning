#!/bin/bash
#SBATCH --job-name=nocap-preprocess
#SBATCH --account=cse
#SBATCH --partition=ice-gpu
#SBATCH -N 1                         
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=4         
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G            
#SBATCH --time=03:00:00                  
#SBATCH --output=$SCRATCH/DLProject/logs/%j.out
#SBATCH --error=$SCRATCH/DLProject/logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=#YOUREMAIL

# ==============================================================
# ENVIRONMENT
# ==============================================================
export SCRATCH=/path/to/scratch

module load cuda/12.1.1
module load anaconda3    

conda init
conda activate recap

# HuggingFace cache — avoids re-downloading models on every job
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface
mkdir -p $HF_HOME

# ==============================================================
# PATHS
# ==============================================================

PROJECT_DIR=$SCRATCH/DLProject
DATA_DIR=/storage/ice-shared/cs7643/shared-group-project-data/nocap/data
AUDIO_DIR=$DATA_DIR/audiocaps_raw_audio
OUTPUT_DIR=$PROJECT_DIR/data

cd $PROJECT_DIR

echo "=== Running preprocessing script for train split at $(date) ==="

python data/audio_preprocess.py \
    --audio_dir $AUDIO_DIR \
    --captions_csv $DATA_DIR/train.csv \
    --output_dir $OUTPUT_DIR \
    --dataset audiocaps \
    --split train \
    --batch_size 32

if [ $? -ne 0 ]; then
    echo "FAILED: train preprocessing failed."
    exit 1
fi

echo "=== Running preprocessing script for val split at $(date) ==="

python data/audio_preprocess.py \
    --audio_dir $AUDIO_DIR \
    --captions_csv $DATA_DIR/val.csv \
    --output_dir $OUTPUT_DIR \
    --dataset audiocaps \
    --split val \
    --batch_size 32


if [ $? -ne 0 ]; then
    echo "FAILED: val preprocessing failed."
    exit 1
fi

echo "=== Running preprocessing script for test split at $(date) ==="

python data/audio_preprocess.py \
    --audio_dir $AUDIO_DIR \
    --captions_csv $DATA_DIR/test.csv \
    --output_dir $OUTPUT_DIR \
    --dataset audiocaps \
    --split test \
    --batch_size 32


if [ $? -ne 0 ]; then
    echo "FAILED: test preprocessing failed."
    exit 1
fi