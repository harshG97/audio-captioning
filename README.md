# RECAP — Retrieval-Augmented Audio Captioning

Audio-encoder + GPT-2 decoder with cross-attention, optionally conditioned on
retrieved captions from a CLAP-built datastore. This README covers the three
**AudioCaps-only** configurations end to end.

| Config | Retrieval | Prompt fed to decoder |
|---|---|---|
| 1. Baseline | none | `This audio sounds like:` |
| 2. RECAP top-K | k=4 nearest captions by CLAP cosine | `Audios similar to this audio sounds like: c1, c2, c3, c4. This audio sounds like:` |
| 3. RECAP MMR | k=4 with maximal marginal relevance (λ=0.7) | same template, different captions (more diverse) |

All three share the same architecture, training loop, evaluation loop, and
audio-feature cache. The only thing that changes between them is the contents
of the retrieval cache JSON fed to `train.py` / `evaluate.py`.

---

## 1. Setup

### Environment

```bash
conda env create -f env.yml
conda activate audio
```

`torch` is installed via pip inside the env. On CUDA hardware you may want
to reinstall against your CUDA version, e.g.:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch
```

GPU/MPS auto-detected by `model/device.py`. Drop `--bf16` from the training
commands below if your hardware doesn't support it.

### Data

The repo ships AudioCaps captions (`data/{train,val,test,train_small}.csv`)
but **not** the raw audio. Place WAVs at `${AUDIO_DIR}/{youtube_id}_{start_time}.wav`,
then export paths:

```bash
export AUDIO_DIR=/path/to/audiocaps_raw_audio
export FEAT_DIR=$PWD/features
export RUNS_DIR=$PWD/runs
mkdir -p $FEAT_DIR $RUNS_DIR
```

### Smoke test

Before launching anything heavy:

```bash
python -m tests.smoke
```

Three asserts run in seconds: dataset alignment (no model needed), forward
path with cached `encoder_outputs`, save/load round-trip. The model-touching
ones download GPT-2 (~500 MB) and CLAP-HTSAT-Fused (~1.5 GB) to the HF cache
on first run.

---

## 2. Shared preprocessing (one-time)

Cache CLAP encoder hidden states for cross-attention. These are
`(64, 768)` per clip — *not* the projected retrieval embeddings.

```bash
for split in train val test; do
  python data/audio_preprocess.py \
      --audio_dir $AUDIO_DIR \
      --captions_csv data/${split}.csv \
      --output_dir $FEAT_DIR \
      --dataset audiocaps --split $split --batch_size 32
done
```

Outputs: `$FEAT_DIR/audiocaps_{train,val,test}.hdf5`. All three configs
below read from these.

---

## 3. Config 1 — Baseline (no retrieval)

**What it does.** Trains the same encoder–decoder architecture but feeds an
empty retrieval list, so the prompt collapses to `"This audio sounds like:"`.
Establishes how much of the captioning quality comes from the model itself
versus the retrieval signal.

```bash
# Build empty caches (one entry per access_id, no retrieved captions)
for split in train val test; do
  python -m retrieval.build_baseline_cache \
      --csv_path data/${split}.csv \
      --dataset audiocaps \
      --output_path $FEAT_DIR/empty_${split}.json
done

# Train. --no-rag tells train.py to skip the strict retrieval-metadata check.
python train.py \
    --train_csv data/train.csv \
    --train_hdf5 $FEAT_DIR/audiocaps_train.hdf5 \
    --train_retrieval_cache $FEAT_DIR/empty_train.json \
    --val_csv data/val.csv \
    --val_hdf5 $FEAT_DIR/audiocaps_val.hdf5 \
    --val_retrieval_cache $FEAT_DIR/empty_val.json \
    --output_dir $RUNS_DIR/recap-baseline \
    --no-rag \
    --num_train_epochs 10 --per_device_train_batch_size 16 --bf16

# Generate predictions
python evaluate.py \
    --checkpoint $RUNS_DIR/recap-baseline \
    --csv_path data/test.csv \
    --hdf5_path $FEAT_DIR/audiocaps_test.hdf5 \
    --retrieval_cache $FEAT_DIR/empty_test.json \
    --output_path $RUNS_DIR/recap-baseline/preds_test.json \
    --num_beams 5

# Score
python score_predictions.py \
    --predictions $RUNS_DIR/recap-baseline/preds_test.json \
    --output $RUNS_DIR/recap-baseline/metrics_test.json
```

---

## 4. Config 2 — RECAP with top-K retrieval (k=4)

**What it does.** For every clip, finds the 4 nearest training-caption
embeddings in CLAP space, prepends them to the prompt, then trains the
decoder to produce the gold caption while attending to both the audio
features (cross-attention) and the retrieved-caption prefix (self-attention).
Self-exclusion is built into the cache so a clip can never retrieve its own
gold caption.

### Build the datastore (one-time, shared with Config 3)

```bash
# Datastore: CLAP text embeddings of every training caption (~91k × 512 ≈ 186 MB)
python -m retrieval.precompute_text_embeddings \
    --csv_path data/train.csv \
    --output_path $FEAT_DIR/text_train.npy \
    --dataset audiocaps --batch_size 64

# Queries: CLAP audio retrieval embeddings per split. Shared with Config 3.
for split in train val test; do
  python -m retrieval.precompute_audio_embeddings \
      --csv_path data/${split}.csv \
      --audio_dir $AUDIO_DIR \
      --dataset audiocaps \
      --output_path $FEAT_DIR/audio_${split}.npy \
      --ids_path $FEAT_DIR/audio_${split}_ids.csv \
      --batch_size 16 --num_workers 8
done
```

### Build top-K caches

```bash
# Training cache: --role train allows query/datastore overlap (self-exclusion handles it)
python -m retrieval.build_retrieval_cache \
    --query_emb_path $FEAT_DIR/audio_train.npy \
    --query_ids_path $FEAT_DIR/audio_train_ids.csv \
    --datastore_emb_path $FEAT_DIR/text_train.npy \
    --datastore_csv_path data/train.csv --datastore_dataset audiocaps \
    --output_path $FEAT_DIR/retrieved_topk_train.json \
    --k 4 --strategy topk --role train

# Eval caches: --role eval refuses to build if any query access_id appears in the
# datastore (leakage guard). Default is --role eval.
for split in val test; do
  python -m retrieval.build_retrieval_cache \
      --query_emb_path $FEAT_DIR/audio_${split}.npy \
      --query_ids_path $FEAT_DIR/audio_${split}_ids.csv \
      --datastore_emb_path $FEAT_DIR/text_train.npy \
      --datastore_csv_path data/train.csv --datastore_dataset audiocaps \
      --output_path $FEAT_DIR/retrieved_topk_${split}.json \
      --k 4 --strategy topk --role eval
done
```

### Train + eval + score

```bash
python train.py \
    --train_csv data/train.csv \
    --train_hdf5 $FEAT_DIR/audiocaps_train.hdf5 \
    --train_retrieval_cache $FEAT_DIR/retrieved_topk_train.json \
    --val_csv data/val.csv \
    --val_hdf5 $FEAT_DIR/audiocaps_val.hdf5 \
    --val_retrieval_cache $FEAT_DIR/retrieved_topk_val.json \
    --output_dir $RUNS_DIR/recap-topk \
    --rag --retrieval_k 4 --retrieval_strategy topk \
    --num_train_epochs 10 --per_device_train_batch_size 16 --bf16

python evaluate.py \
    --checkpoint $RUNS_DIR/recap-topk \
    --csv_path data/test.csv \
    --hdf5_path $FEAT_DIR/audiocaps_test.hdf5 \
    --retrieval_cache $FEAT_DIR/retrieved_topk_test.json \
    --output_path $RUNS_DIR/recap-topk/preds_test.json \
    --num_beams 5

python score_predictions.py \
    --predictions $RUNS_DIR/recap-topk/preds_test.json \
    --output $RUNS_DIR/recap-topk/metrics_test.json
```

---

## 5. Config 3 — RECAP with MMR retrieval (k=4, λ=0.7)

**What it does.** Same as Config 2 but uses Maximal Marginal Relevance to
pick the 4 captions: at each step, the next pick maximizes
`λ · relevance(query) − (1−λ) · max_similarity(already_picked)`. The intent
is that for compositional audio (multiple distinct sound events) MMR retrieves
captions covering different events rather than near-duplicate captions about
the dominant one.

`λ = 1.0` reduces to top-K; `λ < 1.0` trades relevance for diversity.

### Reuses

The text and audio embedding `.npy` files from Config 2 — only the cache
JSONs change.

```bash
# MMR caches
python -m retrieval.build_retrieval_cache \
    --query_emb_path $FEAT_DIR/audio_train.npy \
    --query_ids_path $FEAT_DIR/audio_train_ids.csv \
    --datastore_emb_path $FEAT_DIR/text_train.npy \
    --datastore_csv_path data/train.csv --datastore_dataset audiocaps \
    --output_path $FEAT_DIR/retrieved_mmr_train.json \
    --k 4 --strategy mmr --mmr_lambda 0.7 --role train

for split in val test; do
  python -m retrieval.build_retrieval_cache \
      --query_emb_path $FEAT_DIR/audio_${split}.npy \
      --query_ids_path $FEAT_DIR/audio_${split}_ids.csv \
      --datastore_emb_path $FEAT_DIR/text_train.npy \
      --datastore_csv_path data/train.csv --datastore_dataset audiocaps \
      --output_path $FEAT_DIR/retrieved_mmr_${split}.json \
      --k 4 --strategy mmr --mmr_lambda 0.7 --role eval
done

# Train + eval + score
python train.py \
    --train_csv data/train.csv \
    --train_hdf5 $FEAT_DIR/audiocaps_train.hdf5 \
    --train_retrieval_cache $FEAT_DIR/retrieved_mmr_train.json \
    --val_csv data/val.csv \
    --val_hdf5 $FEAT_DIR/audiocaps_val.hdf5 \
    --val_retrieval_cache $FEAT_DIR/retrieved_mmr_val.json \
    --output_dir $RUNS_DIR/recap-mmr \
    --rag --retrieval_k 4 --retrieval_strategy mmr --retrieval_mmr_lambda 0.7 \
    --num_train_epochs 10 --per_device_train_batch_size 16 --bf16

python evaluate.py \
    --checkpoint $RUNS_DIR/recap-mmr \
    --csv_path data/test.csv \
    --hdf5_path $FEAT_DIR/audiocaps_test.hdf5 \
    --retrieval_cache $FEAT_DIR/retrieved_mmr_test.json \
    --output_path $RUNS_DIR/recap-mmr/preds_test.json \
    --num_beams 5

python score_predictions.py \
    --predictions $RUNS_DIR/recap-mmr/preds_test.json \
    --output $RUNS_DIR/recap-mmr/metrics_test.json
```

---

## 6. Output structure

After all three configs finish:

```
runs/
├── recap-baseline/
│   ├── checkpoint-*/      ← Trainer checkpoints
│   ├── preds_test.json    ← {access_id: {references, prediction}}
│   └── metrics_test.json  ← BLEU/METEOR/ROUGE/CIDEr/SPICE/SPIDEr
├── recap-topk/
│   └── ...
└── recap-mmr/
    └── ...
```

Three `metrics_test.json` files give you the comparison table:

| Config | CIDEr | SPIDEr |
|---|---|---|
| Baseline (no retrieval) | … | … |
| RECAP top-K (k=4) | … | … |
| RECAP MMR (k=4, λ=0.7) | … | … |

---

## 7. Notes and pitfalls

- **Smoke-train first.** Before a 10-epoch run, do a 1-epoch run on
  `data/train_small.csv` (500 rows) with the same flags to confirm the loss
  decreases and checkpointing works on your hardware.
- **Trainable parameter count** prints at the top of each training run.
  Default is selective freezing (~7 M trainable: cross-attention layers +
  LayerNorms). Pass `--train_decoder` to fine-tune the full GPT-2 stack
  (~120 M trainable) — slower, more memory, more risk of overfitting on
  AudioCaps-scale data.
- **`--bf16` requires Ampere or newer NVIDIA, or recent Apple Silicon.**
  Use `--fp16` on older CUDA, or omit both for fp32.
- **Leakage guard.** `build_retrieval_cache.py` defaults to `--role eval`
  and aborts if query and datastore access_ids overlap. Pass `--role train`
  only when you're building the in-domain training cache.
- **Generation knobs** in `evaluate.py`:
  `--num_beams`, `--length_penalty` (default 1.0; <1 favors shorter
  captions), `--no_repeat_ngram_size` (default 0; try 3 to suppress GPT-2
  repetition).
