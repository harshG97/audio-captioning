#!/bin/bash
#SBATCH --job-name=recap-train
#SBATCH --account=cse
#SBATCH --partition=ice-gpu
#SBATCH -N 1                         
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=4        
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-gpu=32G            
#SBATCH --time=12:00:00                  
#SBATCH --output=$SCRATCH/DLProject/logs/%j.out
#SBATCH --error=$SCRATCH/DLProject/logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=#YOUREMAIL


export SCRATCH=/path/to/your/scratch

module load cuda/12.1.1
module load anaconda3    

conda init
conda activate recap

# Cache HF models in scratch, not home (home quota is small)
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface
mkdir -p $HF_HOME

# Reproducibility
export PYTHONHASHSEED=42

# Save tokenizers from a deadlock warning when used with multi-worker dataloaders
export TOKENIZERS_PARALLELISM=false
# ==============================================================
# PATHS
# ==============================================================

PROJECT_DIR=$SCRATCH/DLProject
DATA_DIR=/storage/ice-shared/cs7643/shared-group-project-data/nocap/data
# AUDIO_DIR=$DATA_DIR/audiocaps_raw_audio
# OUTPUT_DIR=$PROJECT_DIR/data

cd $PROJECT_DIR

# Run smoke test first
echo "=== Smoke test at $(date) ==="
python -m tests.test_train \
    --disable_rag \
    --features_dir $DATA_DIR/features \
    --annotations_path $DATA_DIR/audiocaps_annotations
if [ $? -ne 0 ]; then
    echo "FAILED: smoke test failed; aborting training"
    exit 1
fi

echo "=== Training at $(date) ==="
python -m train \
    --disable_rag \
    --features_dir $DATA_DIR/features \
    --annotations_path $DATA_DIR/audiocaps_annotations \
    --experiments_dir $SCRATCH/DLProject/experiments/ \
    --batch_size 32 \
    --n_epochs 10 \
    --lr 5e-5 
if [ $? -ne 0 ]; then
    echo "FAILED: training failed"
    exit 1
fi
echo "=== Done at $(date) ==="