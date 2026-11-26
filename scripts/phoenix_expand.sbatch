#!/bin/bash
# Expanded training data: paper train plus the other AudioCaps 2.0 train
# clips, capped at CAP copies per caption, with and without the worst-fitting
# DROP_PCT% of extra clips (data/build_expanded_split.py). Then retrieval
# caches for the extra clips, from the same datastore as the paper runs (the
# paper train captions in DATASTORE_DIR), so only the training data changes.
#
# Submit through scripts/submit.sh:
#   scripts/submit.sh expand-data
#
# Writes data/expanded/cap$CAP and data/expanded/cap${CAP}_agree$DROP_PCT
# (train, val and test csvs; val and test are the paper split's) and
# $CACHE_DIR/retrieved_top4_{train,val,test}.json covering both. The train
# cache holds DATASTORE_DIR's entries for paper clips unchanged, plus new
# entries for the extra clips. Each step leaves a marker in $CACHE_DIR/.done/.

#SBATCH --job-name=recap-expand
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
EMB_DIR=${EMB_DIR:-$HOME/scratch/recap/features}              # CLAP embeddings of data/train.csv
DATASTORE_DIR=${DATASTORE_DIR:-$HOME/scratch/recap/features_unfused}  # paper-split retrieval
CACHE_DIR=${CACHE_DIR:-$HOME/scratch/recap/features_unfused_plus}
CAP=${CAP:-3}
DROP_PCT=${DROP_PCT:-10}
K=${K:-4}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
OUT=data/expanded

if [ "${SKIP_CONDA:-0}" != 1 ]; then
    module load "$ANACONDA_MODULE"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

for f in "$EMB_DIR/text_train.npy" "$EMB_DIR/audio_train.npy" "$EMB_DIR/audio_train_ids.csv" \
         "$DATASTORE_DIR/text_train.npy" "$DATASTORE_DIR/clap.txt"; do
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done
for split in train val test; do
    [ -f "$DATASTORE_DIR/retrieved_top${K}_$split.json" ] || { echo "missing $DATASTORE_DIR/retrieved_top${K}_$split.json" >&2; exit 1; }
done
CLAP=$(cat "$DATASTORE_DIR/clap.txt")
mkdir -p "$CACHE_DIR/.done"
settings="cap=$CAP drop_pct=$DROP_PCT k=$K datastore=$DATASTORE_DIR emb=$EMB_DIR clap=$CLAP"
if [ -f "$CACHE_DIR/settings.txt" ] && [ "$(cat "$CACHE_DIR/settings.txt")" != "$settings" ]; then
    echo "$CACHE_DIR was built with '$(cat "$CACHE_DIR/settings.txt")', not '$settings'; use another CACHE_DIR" >&2
    exit 1
fi
echo "$settings" > "$CACHE_DIR/settings.txt"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-local}  start: $(date)"
echo "caches: $CACHE_DIR ($settings)"

step() {  # step <name> <command...>
    local name=$1; shift
    if [ -f "$CACHE_DIR/.done/$name" ]; then
        echo "=== skip $name (done)"
        return
    fi
    echo "=== $name  $(date +%T)"
    "$@"
    touch "$CACHE_DIR/.done/$name"
}

A=$OUT/cap$CAP
B=$OUT/cap${CAP}_agree$DROP_PCT

# 1. The two training splits
step splits python data/build_expanded_split.py --emb_dir "$EMB_DIR" --out_dir "$OUT" \
    --cap "$CAP" --drop_frac "$(python -c "print($DROP_PCT / 100)")"

# 2. CLAP audio embeddings (the retrieval model) for the extra clips only
step extra_csv python - "$A/train.csv" data/paper_split/train.csv "$CACHE_DIR/extra_train.csv" <<'EOF'
import sys, pandas as pd
sys.path.insert(0, ".")
from data.access_id import build_access_id_column
a, paper, out = sys.argv[1:]
a = build_access_id_column(pd.read_csv(a, dtype={"youtube_id": str}), "audiocaps")
paper_ids = set(build_access_id_column(pd.read_csv(paper, dtype={"youtube_id": str}), "audiocaps")["access_id"])
extra = a[~a["access_id"].isin(paper_ids)]
extra[["youtube_id", "start_time", "caption"]].to_csv(out, index=False)
print(f"{len(extra)} extra clips")
EOF
step audio_emb_extra python -m retrieval.precompute_audio_embeddings \
    --csv_path "$CACHE_DIR/extra_train.csv" --audio_dir "$AUDIO_DIR" --dataset audiocaps \
    --output_path "$CACHE_DIR/audio_extra.npy" --ids_path "$CACHE_DIR/audio_extra_ids.csv" \
    --batch_size 16 --num_workers "$WORKERS" --clap_model "$CLAP"

# 3. Their top-k captions from the paper train datastore (they are not in it,
#    so nothing to exclude)
step "top${K}_extra" python -m retrieval.build_retrieval_cache \
    --query_emb_path "$CACHE_DIR/audio_extra.npy" --query_ids_path "$CACHE_DIR/audio_extra_ids.csv" \
    --datastore_emb_path "$DATASTORE_DIR/text_train.npy" \
    --datastore_csv_path data/paper_split/train.csv --datastore_dataset audiocaps \
    --output_path "$CACHE_DIR/retrieved_top${K}_extra.json" --k "$K" --strategy topk --role eval

# 4. Train cache = the paper clips' entries from DATASTORE_DIR (unchanged) +
#    the new entries for the extra clips; val/test as before
step "top${K}_train" python - "$DATASTORE_DIR/retrieved_top${K}_train.json" \
    "$CACHE_DIR/retrieved_top${K}_extra.json" data/paper_split/train.csv "$A/train.csv" \
    "$CACHE_DIR/retrieved_top${K}_train.json" <<'EOF'
import json, sys, pandas as pd
sys.path.insert(0, ".")
from data.access_id import build_access_id_column
old, new, paper_csv, train_csv, out = sys.argv[1:]
ids = lambda p: set(build_access_id_column(pd.read_csv(p, dtype={"youtube_id": str}), "audiocaps")["access_id"])
paper_ids, train_ids = ids(paper_csv), ids(train_csv)
old, new = json.load(open(old)), json.load(open(new))
paper = {k: v for k, v in old.items() if k in paper_ids}
if set(new) & paper_ids:
    sys.exit("extra clips overlap the paper train clips")
# Paper clips without audio have no entry, as in the paper runs (training
# skips them); every extra clip was chosen for having audio, so it must.
print(f"paper train clips without an entry (no audio, as before): {len(paper_ids - set(paper))}")
missing = (train_ids - paper_ids) - set(new)
if missing:
    sys.exit(f"{len(missing)} extra clips have no cache entry, e.g. {sorted(missing)[:3]}")
merged = {**paper, **new}
json.dump(merged, open(out, "w"))
print(f"{len(paper)} paper + {len(new)} extra = {len(merged)} entries")
EOF
for split in val test; do
    step "top${K}_$split" cp "$DATASTORE_DIR/retrieved_top${K}_$split.json" "$CACHE_DIR/"
done

echo "=== all done  $(date)"
for d in "$A" "$B"; do echo "$d: $(cat "$d/info.json" | tr -d '\n ')"; done
