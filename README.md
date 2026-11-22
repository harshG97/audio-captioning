# RECAP — Retrieval-Augmented Audio Captioning

Audio-encoder + GPT-2 decoder with cross-attention, optionally conditioned on
retrieved captions from a CLAP-built datastore. This README covers the three
**AudioCaps-only** configurations end to end.

| Config | Retrieval | Prompt fed to decoder |
|---|---|---|
| 1. Baseline | none | `This audio sounds like` |
| 2. RECAP top-K | k=4 nearest captions by CLAP cosine | `Similar audios sound like` + the 4 captions, one per paragraph + `This audio sounds like` (the RECAP reference template) |
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

### Tests

Unit tests run on CPU in a few seconds and need no downloads:

```bash
python -m pytest tests -q
```

They cover the decoder's causal masking (`tests/test_decoder_causality.py`),
CLAP encoder loading and input preparation, caption filtering, the prompt
template, cross-attention initialisation and the generation-health metrics.
Run them after any change to `transformers`: the decoder masking depends on
GPT-2 internals that changed within the 4.4x series.

### Smoke test

Before launching anything heavy:

```bash
python -m tests.smoke
python -m tests.smoke2
```

These exercise dataset alignment, the forward path with cached
`encoder_outputs`, a training step, generation and save/load. The
model-touching ones download GPT-2 (~500 MB) and CLAP-HTSAT-Fused (~1.5 GB)
to the HF cache on first run.

### Rebuilding artifacts from earlier runs

Checkpoints, features and caches produced before the fixes on this branch
must be rebuilt:

- **Encoder HDF5 features** (`data/audio_preprocess.py`): earlier ones came
  from a randomly initialised CLAP audio encoder, with zero-padding and a
  batch-dependent `is_longer` flag.
- **Audio retrieval embeddings** (`retrieval.precompute_audio_embeddings`):
  same padding and `is_longer` issues.
- **Text embeddings and retrieval caches**: placeholder captions such as
  "video unvailable" are now excluded from the datastore.
- **Checkpoints**: earlier ones were trained with a non-causal decoder
  (future tokens visible) and the old prompt template.

---

## 2. Shared preprocessing (one-time)

Cache CLAP encoder hidden states for cross-attention. These are
`(64, 768)` per clip — *not* the projected retrieval embeddings. The script
loads the pretrained audio tower from the full CLAP checkpoint (and refuses
to run if any encoder weight is missing), and prepares each clip on its own
so features do not depend on `--batch_size`.

```bash
for split in train val test; do
  python data/audio_preprocess.py \
      --audio_dir $AUDIO_DIR \
      --captions_csv data/${split}.csv \
      --output_dir $FEAT_DIR \
      --dataset audiocaps --split $split \
      --batch_size 64 --num_workers 8
done
```

Outputs: `$FEAT_DIR/audiocaps_{train,val,test}.hdf5`. All three configs
below read from these.

---

## 3. Config 1 — Baseline (no retrieval)

**What it does.** Trains the same encoder–decoder architecture but feeds an
empty retrieval list, so the prompt collapses to `"This audio sounds like"`.
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
    --output_path $RUNS_DIR/recap-baseline/preds_test.json

# Score
python score_predictions.py \
    --predictions $RUNS_DIR/recap-baseline/preds_test.json \
    --output $RUNS_DIR/recap-baseline/metrics_test.json \
    --train_csv data/train.csv
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
# Datastore: CLAP text embeddings of every training caption (~91k × 512 ≈ 186 MB).
# Placeholder/empty captions are dropped here and, identically, in
# build_retrieval_cache.py, so embedding rows and datastore rows stay aligned.
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
    --output_path $RUNS_DIR/recap-topk/preds_test.json

python score_predictions.py \
    --predictions $RUNS_DIR/recap-topk/preds_test.json \
    --output $RUNS_DIR/recap-topk/metrics_test.json \
    --train_csv data/train.csv
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
    --output_path $RUNS_DIR/recap-mmr/preds_test.json

python score_predictions.py \
    --predictions $RUNS_DIR/recap-mmr/preds_test.json \
    --output $RUNS_DIR/recap-mmr/metrics_test.json \
    --train_csv data/train.csv
```

---

## 6. Comparing with the paper and with plain retrieval

### Paper splits

The RECAP paper trains on 42,797 AudioCaps clips and tests on 748. Those
splits are in `data/paper_split/` (see its README). Every clip is also in
`data/{train,val,test}.csv`, so the HDF5 features and audio embeddings
built above cover them. Rebuild only the datastore and caches from the
paper's training captions:

```bash
python -m retrieval.precompute_text_embeddings \
    --csv_path data/paper_split/train.csv \
    --output_path $FEAT_DIR/text_paper_train.npy --dataset audiocaps

# then build_retrieval_cache.py as in Config 2, with
#   --datastore_emb_path $FEAT_DIR/text_paper_train.npy
#   --datastore_csv_path data/paper_split/train.csv
```

Train on `data/paper_split/train.csv` and evaluate on
`data/paper_split/test.csv`. A model trained on the full split can also be
evaluated on `data/paper_split/test.csv` directly.

### Copy-the-retrieved-caption baseline

A RECAP model should beat simply outputting each clip's nearest retrieved
caption:

```bash
python scripts/copy_retrieval_baseline.py \
    --csv_path data/test.csv \
    --retrieval_cache $FEAT_DIR/retrieved_topk_test.json \
    --hdf5_path $FEAT_DIR/audiocaps_test.hdf5 \
    --output_path $RUNS_DIR/copy-top1/preds_test.json

python score_predictions.py \
    --predictions $RUNS_DIR/copy-top1/preds_test.json \
    --output $RUNS_DIR/copy-top1/metrics_test.json \
    --train_csv data/train.csv
```

---

## 7. Output structure

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

## 8. Notes and pitfalls

- **Smoke-train first.** Before a 10-epoch run, do a 1-epoch run on
  `data/train_small.csv` (500 rows) with the same flags to confirm the loss
  decreases and checkpointing works on your hardware.
- **Trainable parameter count** prints at the top of each training run.
  Default is selective freezing (~7 M trainable: cross-attention layers +
  LayerNorms, `--attention_size 7` as in the paper; `--attention_size 28`
  gives the larger 28 M variant). Pass `--train_decoder` to fine-tune the
  full GPT-2 stack (~120 M trainable) — slower, more memory, more risk of
  overfitting on AudioCaps-scale data.
- **Cross-attention init.** Cross-attention layers start from random init,
  as in the paper. `--xattn_zero_init` zeroes their output projections
  instead, so training starts from plain GPT-2 behaviour.
- **`--bf16` requires Ampere or newer NVIDIA, or recent Apple Silicon.**
  Use `--fp16` on older CUDA, or omit both for fp32.
- **Leakage guard.** `build_retrieval_cache.py` defaults to `--role eval`
  and aborts if query and datastore access_ids overlap. Pass `--role train`
  only when you're building the in-domain training cache.
- **Generation knobs** in `evaluate.py` default to the RECAP reference
  inference: `--num_beams 3`, `--max_new_tokens 60`, `--length_penalty 0`
  (summed log-prob; ignored for greedy), `--no_repeat_ngram_size 0`.
  `--num_beams 1` gives greedy decoding.
- **Scoring.** `score_predictions.py` PTB-tokenizes references and
  predictions (needs Java), as in the standard COCO/DCASE evaluation, and
  reports SPICE/SPIDEr unless `--skip_spice`. `--no_tokenize` reproduces the
  older raw-string scores, which are not comparable with published numbers.
- **Generation health.** The metrics JSON also has `eos_stop_rate`,
  `max_tokens_rate`, `postprocess_cut_rate`, `unique_rate`, `empty_rate` and
  (with `--train_csv`) `train_copy_rate`. A model that uses the audio should
  stop on EOS almost always, produce mostly unique captions and not just
  copy training captions; check these before reading the caption metrics.
