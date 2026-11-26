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

With conda (includes Java):

```bash
conda env create -f env.yml
conda activate audio
```

Or with pip in a virtualenv (`requirements.txt` has the same packages):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

With pip, install Java separately (JDK 11; on macOS `brew install openjdk@11`).
Scoring needs it for PTB tokenization, METEOR and SPICE, and
`score_predictions.py` stops with an error if it is missing. SPICE 1.0
crashes on recent Java releases, so `env.yml` pins `openjdk=11`.

SPICE also needs Stanford CoreNLP 3.6.0. On its first run it tries to
download it from nlp.stanford.edu, which is often unreachable; if SPICE is
skipped with a `URLError`, fetch the two jars from Maven Central instead:

```bash
cd "$(python -c 'import pycocoevalcap.spice.spice as s, os; print(os.path.dirname(s.__file__))')/lib"
for f in stanford-corenlp-3.6.0.jar stanford-corenlp-3.6.0-models.jar; do
  curl -fL -o $f https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.6.0/$f
done
```

Both setups pin `transformers==4.45.2`; keep it exact (see the comment in
`env.yml`). `torch` comes from pip in both. On CUDA hardware you may want
to reinstall it against your CUDA version, e.g.:

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
`tests/test_pipeline.py` runs the whole path on tiny models built in the
test: dataset prompt masking and label shift, collation, a training step
(only cross-attention moves), save/load, and greedy generation checked
against a cache-free argmax loop. Nothing is downloaded, so run the suite
before launching anything heavy and after any change to `transformers`:
the decoder masking depends on GPT-2 internals that changed within the
4.4x series.

### Running on Slurm (PACE Phoenix)

`scripts/submit.sh` submits every stage as a Slurm job with the right GPU,
time limit and settings. Cluster details come from your environment rather
than the repo; put them in `~/.bashrc`:

```bash
export RECAP_ACCOUNT=<charge account>    # pace-quota lists yours
export RECAP_QOS=inferno                 # or embers (free, preemptible)
export RECAP_GPU=a100                    # as listed by: sinfo -o "%G"
export NOTIFY_EMAIL=<you>@gatech.edu     # optional: mail on start/end/failure
```

Then, from the repo root:

```bash
scripts/submit.sh list           # all jobs
scripts/submit.sh preprocess     # features, embeddings, caches (about 2.5 h)
scripts/submit.sh paper-caches   # caches for the paper split (reuses features)
scripts/submit.sh g1             # sanity gate 1: k=0, 1 epoch
scripts/submit.sh g2             # sanity gate 2: k=4, 3 epochs
scripts/submit.sh run-a          # paper-faithful run (also run-b, run-c)
```

Any setting can be overridden for one submission, e.g.
`SEED=1 scripts/submit.sh run-a` or `K=2 EPOCHS=5 scripts/submit.sh g2`;
`DRY_RUN=1` prints the `sbatch` command instead. `DEDUP=1` retrieves from
distinct captions only (`build_retrieval_cache.py --dedupe_captions`: the
full 2.0 datastore repeats some captions hundreds of times, so 18% of its
rows merge; 8% for the paper split), and the run name gets a `_dedup`
suffix so both versions can be compared. Preprocessing builds both caches. `K=1`, `K=2` or `K=3` uses the first k
captions of the top-4 caches and names the run accordingly (`a_paper_k2_s42`). Training jobs evaluate and
score on their eval sets, run the audio ablation and (for k > 0) the copy
baseline, and end the log with a summary table. Every step leaves a marker
in its output folder and training resumes from its last checkpoint
(`train.py --resume`), so resubmitting after a timeout or preemption picks
up where the job stopped. On `l40s` nodes PACE allows at most 4 CPUs per
GPU; `submit.sh` caps it.

#### Experiment settings

Training jobs take these settings; each non-default value adds a suffix to
the run name, so a variant never reuses an existing run's folder:

| Setting | Effect | Run-name suffix |
|---|---|---|
| `XATTN=28`, `14`, `7`, `3.5` or `1.75` | cross-attention size in millions of parameters (default 7, as in RECAP) | `_x28` |
| `TRAIN_DECODER=1` | fine-tune all of GPT-2, not just the cross-attention | `_ftdec` |
| `PROMPT_DROPOUT=0.3` | train without the retrieved captions 30% of the time | `_pd0.3` |
| `KEEP=last` | train all `EPOCHS` (100) without early stopping and keep the last epoch, as RECAP's own `train.py` does; every `CHECKPOINT_EVERY`-th (20) epoch and the lowest-val-loss epoch are kept as full checkpoints and scored on the first eval set | `_last` |
| `TAG=<name>` | free-form label | `_<name>` |

Pointing a training job at other features, caches or models (`FEAT_DIR`,
`CACHE_DIR`, `ENCODER_NAME`, `DECODER_NAME`) requires a `TAG`. For example,
to train on the paper split but retrieve from the full 2.0 caption list:

```bash
CACHE_DIR=$HOME/scratch/recap/features TAG=ds2 scripts/submit.sh run-a
```

By default (`KEEP=best`) a run keeps the epoch with the lowest val loss and
stops after `PATIENCE` epochs without improvement. `KEEP=last` keeps the
final epoch instead (it cannot be combined with `PATIENCE`). It also keeps
`RUN_DIR/snapshots/epoch020`, `epoch040`, ... and `snapshots/best` (the
lowest val loss), each a full checkpoint with optimizer, scheduler and RNG
state plus its own `preds_*.json` and `metrics_*.json`; the summary lists
them, so you can see at which epoch caption quality peaks. `CHECKPOINT_EVERY`
changes the interval; `SNAPSHOT_EVERY=10` adds model-only snapshots in
between (scored, but not resumable):

```bash
KEEP=last scripts/submit.sh run-a     # a_paper_k4_last_s42
```

`RESUME_FROM=<checkpoint folder>` starts a new run from any full checkpoint
(or a `checkpoint-N` folder) with its optimizer, scheduler and step count, and
needs a `TAG`. With the same settings it continues exactly where that run
was; the learning-rate schedule follows `EPOCHS`.

```bash
RESUME_FROM=$HOME/scratch/recap/runs/a_paper_k4_last_s42/snapshots/epoch040 \
    TAG=from40 KEEP=last scripts/submit.sh run-a
```

A resubmitted job resumes from the newest complete checkpoint; one left
half-written by a killed job is skipped (`[resume] skipping incomplete ...`).
Resuming, with `--resume` or `RESUME_FROM`, continues the same random
streams as the uninterrupted run (dataloader workers included), so it ends
where that run would have, up to the GPU's floating-point variation.

`SEED` sets the cross-attention initialisation and the data order. Two runs
with the same seed match exactly on one CPU thread; on a GPU or with
several threads they can differ slightly in the last digits.

Each run folder records its settings in `settings.txt`; resubmitting with
different settings stops with a diff instead of reusing finished steps.

Preprocessing takes `CLAP=<checkpoint>` (default `laion/clap-htsat-fused`)
for both the cross-attention features and the retrieval embeddings, and
records it in `$FEAT_DIR/clap.txt`; mixing checkpoints in one folder is
refused, and training jobs read the encoder name from that file. To build
another checkpoint's features for the paper split only (about half the
clips of the full set):

```bash
CLAP=laion/clap-htsat-unfused FEAT_DIR=$HOME/scratch/recap/features_unfused \
    DATA_DIR=data/paper_split scripts/submit.sh preprocess
FEAT_DIR=$HOME/scratch/recap/features_unfused CACHE_DIR=$HOME/scratch/recap/features_unfused \
    TAG=unfused scripts/submit.sh run-a
```

Cross-attention reads the encoder features at the decoder's width. When the
two differ, e.g. CLAP's 768 and GPT-2 medium's 1024, the model adds a
trainable linear projection between them (`enc_to_dec_proj`); with equal
widths (GPT-2 small) there is none and the model is unchanged. `train.py`
stops early if the features are not as wide as the encoder the model was
built with. A larger GPT-2 needs only a `DECODER_NAME` and a `TAG`:

```bash
DECODER_NAME=gpt2-medium TAG=medium scripts/submit.sh run-a     # a_paper_k4_medium_s42
```

`XATTN` names the cross-attention sizes of GPT-2 small; it sets the reduce
factor (7 = 4), so on GPT-2 medium `XATTN=7` is about 25M parameters plus
0.8M for the projection. The run log prints the actual trainable count.

**BEATs features.** [BEATs](https://arxiv.org/abs/2212.09058) (Microsoft,
trained on AudioSet) can replace CLAP as the features the decoder attends
to; retrieval stays on CLAP, since it needs CLAP's shared audio-text space.
Its code is vendored in `model/beats/` (MIT, from `microsoft/unilm`, with
torchaudio's filterbank so no torchaudio install is needed). Checkpoints are
plain `.pt` files outside Hugging Face; the one captioning systems usually
use is `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`, linked from the
[BEATs README](https://github.com/microsoft/unilm/tree/master/beats).

```bash
BEATS_CKPT=~/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt scripts/submit.sh beats-features
B=$HOME/scratch/recap/features_beats U=$HOME/scratch/recap/features_unfused
K=3 XATTN=28 FEAT_DIR=$B CACHE_DIR=$U TAG=beats-official scripts/submit.sh run-a
XATTN=28 FEAT_DIR=$B TAG=beats scripts/submit.sh run-c
```

`beats-features` writes the paper split's features to `features_beats`,
with the empty caches run-c needs. Clips are loaded at 16 kHz and made 10 s
long (cut, or repeated if shorter). BEATs gives 62 time steps x 8 frequency
bands of 768-wide vectors; `POOL=4` (default) averages 4 neighbouring bands
per time step, giving 124 vectors per clip (about 17 GB for the paper
split), and `POOL=1` keeps all 496. The checkpoint and pooling are recorded
in `encoder.txt`, and neither preprocessing job writes into a folder made
by the other. The job first prints the top AudioSet labels of a few val
clips beside their captions, to show the weights loaded. It stops if any
clip that the CLAP features in `MATCH_FEAT_DIR` (default `features`) hold
has no audio, since training skips clips without features and the two
encoders would then be compared on different clips. The trained model
still carries CLAP's encoder config, which only gives the feature width
(768, as for BEATs): new audio must go through `extract_beats_features.py`.
`data/audioset_labels.csv` is the AudioSet label list (Google, CC BY 4.0).

**More training data.** The paper split trains on 42,797 of AudioCaps 2.0's
91,254 train clips. `scripts/submit.sh expand-data` (after `preprocess`)
adds the other 48,457, filtered by `data/build_expanded_split.py`:

- `data/expanded/cap3`: paper train plus the extra clips, each caption kept
  at most 3 times counting the paper copies (the extra captions repeat
  stock phrasings, e.g. one caption 355 times), and only clips with audio.
- `data/expanded/cap3_agree10`: the same, minus the 10% of extra clips whose
  caption fits their audio worst (CLAP audio-text cosine similarity, from
  the `features` embeddings).

Extra captions are written in the paper split's style (lowercase, ending
with "."). Val and test are the paper split's, unchanged; paper train rows
are kept as they are. The job also writes retrieval caches for the extra
clips to `features_unfused_plus`, from the same datastore as the paper runs
(paper train captions, unfused CLAP), keeping the paper clips' entries
unchanged. Train with `TRAIN_DIR`, which only run-a and run-c take, needs a
`TAG`, and must keep the paper split's val and test:

```bash
scripts/submit.sh expand-data
DATA_DIR=data/expanded/cap3 FEAT_DIR=$HOME/scratch/recap/features_beats_plus \
  BEATS_CKPT=... scripts/submit.sh beats-features
TRAIN_DIR=data/expanded/cap3 K=3 DECODER_NAME=gpt2-medium FEAT_DIR=$HOME/scratch/recap/features_beats_plus \
  CACHE_DIR=$HOME/scratch/recap/features_unfused_plus TAG=beats-medium-cap3 scripts/submit.sh run-a
```

Before training, the job stops if any clip beyond the paper train clips has
no features in `FEAT_DIR` or no entry in the train cache: the dataset skips
such clips, so a folder left at its paper default would otherwise train on
the paper clips alone. The summary's `train_copy_rate` then counts copies of
the expanded training captions, so it is not directly comparable with
paper-split runs.

### Running on a single GPU machine (no Slurm)

On a machine without Slurm, such as a cloud GPU instance, `LOCAL=1` runs the
same jobs in the foreground instead of submitting them. Activate the env
first; the log is also written to `logs/`. Run it inside `tmux` so a dropped
SSH connection does not stop the job.

```bash
conda activate audio
export AUDIO_DIR=~/data/audiocaps_raw_audio   # only preprocessing reads it
LOCAL=1 scripts/submit.sh preprocess
LOCAL=1 scripts/submit.sh paper-caches
LOCAL=1 K=2 DEDUP=1 scripts/submit.sh g2
```

Outputs go to `~/scratch/recap` (set `SCRATCH_RECAP` to change it) and the
Hugging Face cache to `~/.cache/huggingface`. `RECAP_ACCOUNT` is not needed,
and sbatch options such as `--time` are ignored.

On AWS Deep Learning AMIs, `LD_LIBRARY_PATH` points at the system CUDA, whose
cuDNN clashes with the one bundled with pip's torch (convolutions fail with
`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`). Add `unset LD_LIBRARY_PATH` to the
end of `~/.bashrc`; torch does not need the system CUDA.

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

On a Slurm cluster, `scripts/submit.sh preprocess` runs this step plus the
retrieval embeddings and caches from sections 3 and 4 (empty and top-k) as
one job; see "Running on Slurm" below.

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

`--retrieval_k` sets how many cached captions go into each prompt (the first
k, most similar first), so a top-4 cache also serves k=1 to 3. The value is
saved in the checkpoint config, and `evaluate.py` and `ablate_audio.py` use
it by default, so evaluation always matches training (`evaluate.py
--retrieval_k` overrides it).

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
  `--num_beams 1` gives greedy decoding. Beam search with length penalty 0
  favors short captions (about 6 words, against about 10 in the
  references): `--length_penalty` above 0 and `--min_new_tokens N` lengthen them. The
  minimum holds back every token the caption would end or be cut at (".",
  the end token here, and tokens such as "..." or a newline) for N tokens,
  about 1.3 tokens per word; pair a
  minimum with `--no_repeat_ngram_size 3` to avoid repetition.
- **Scoring.** `score_predictions.py` PTB-tokenizes references and
  predictions (needs Java), as in the standard COCO/DCASE evaluation, and
  reports METEOR and SPICE/SPIDEr unless `--skip_meteor` or `--skip_spice`.
  `scripts/sweep.py` scores each prediction file once and passes these on
  as `--score_skip_meteor` and `--score_skip_spice`. `--no_tokenize`
  reproduces the older raw-string scores, which are not comparable with
  published numbers.
- **Generation health.** The metrics JSON also has `eos_stop_rate`,
  `max_tokens_rate`, `postprocess_cut_rate`, `unique_rate`, `empty_rate` and
  (with `--train_csv`) `train_copy_rate`. A model that uses the audio should
  stop on EOS almost always, produce mostly unique captions and not just
  copy training captions; check these before reading the caption metrics.
