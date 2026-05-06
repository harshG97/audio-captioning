# Sweep results

This directory contains predictions, per-config metrics, and roll-up
summary CSVs from training and evaluating RECAP across a hyperparameter
grid.

- `summary1.csv` (64 rows) — k=0 and k=2 configs.
- `summary2.csv` (32 rows) — k=4 configs.
- `top<k>_pd<pd>_wd<wd>_ep<ep>_lr<lr>/` — one subdir per training config,
  containing `predictions/` and `metrics/` subfolders for each generation
  config evaluated.
- `finetuned/` — stage-2 fine-tune of the best stage-1 checkpoint
  (top4_pd0_wd0_ep12_lr5e-05) at two epoch budgets (10, 20).

All metrics below are from `score_mode = with_java` (full metric set
including METEOR; SPICE/SPIDEr were not produced).


## Tuned hyperparameters

### Training (used by `train.py` / `sweep.py`)

| Flag | Description | Values explored |
|---|---|---|
| `k` (`--retrieval_k`) | Number of retrieved captions in the prompt. `k=0` is the no-retrieval baseline (`--no-rag`). | **0, 2, 4** |
| `pd` (`--prompt_dropout`) | Probability of replacing the retrieved-captions prompt with the baseline prompt during training. Forces the decoder to rely on cross-attention to audio rather than copying. | **0.0, 0.25** |
| `wd` (`--weight_decay`) | AdamW L2 weight decay coefficient. | **0.0, 0.01** |
| `ep` (`--num_train_epochs`) | Total training epochs. | **12** for k=2/4; **15** for k=0 |
| `lr` (`--learning_rate`) | Peak learning rate (linear warmup + decay). | **5e-5** (fixed) |

### Generation (used by `evaluate.py`)

| Flag | Description | Values explored |
|---|---|---|
| `nb` (`--num_beams`) | Beam-search width. `1` = greedy. | **1, 3** |
| `nrngram` (`--no_repeat_ngram_size`) | Block any n-gram from being repeated within a generated caption. `0` = disabled. | **0, 3** |
| `lp` (`--length_penalty`) | Sequence-length normalization for beam search. `<1` favors shorter, `>1` favors longer. | **0.8, 1.0** |


## Top-performing configurations

The same stage-1 training config (`k=4, pd=0, wd=0, ep=12, lr=5e-5`)
wins every metric, but with different generation settings.

### Best BLEU-1: 0.4842

| | |
|---|---|
| Training | k=4, pd=0, wd=0, ep=12, lr=5e-5 |
| Generation | num_beams=1, no_repeat_ngram_size=3, length_penalty=0.8 |
| BLEU-1 / 2 / 3 / 4 | 0.4842 / 0.2961 / 0.1760 / 0.0987 |
| ROUGE-L | 0.3440 |
| CIDEr | 0.3078 |
| METEOR | 0.1829 |

### Best ROUGE-L: 0.3440

Same row as the BLEU-1 winner (greedy + n-gram blocking + lp=0.8).

### Best CIDEr: 0.3109

| | |
|---|---|
| Training | k=4, pd=0, wd=0, ep=12, lr=5e-5 |
| Generation | num_beams=3, no_repeat_ngram_size=3, length_penalty=1.0 |
| BLEU-1 / 2 / 3 / 4 | 0.4842 / 0.2974 / 0.1780 / 0.0967 |
| ROUGE-L | 0.3387 |
| CIDEr | 0.3109 |
| METEOR | 0.1861 |


## Stage-2 fine-tune results (`finetuned/`)

Both runs continue training from the best stage-1 checkpoint above with
GPT-2 unfrozen via `finetune.py` (xattn LR 5e-5, decoder LR 5e-6,
`pd=0, wd=0.01`). Evaluated only at `nb=3, nrngram=3, lp=1.0`.

| Run | epochs | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | METEOR |
|---|---|---|---|---|---|---|
| `..._finetuned_pd0_wd0.01_ep10` | 10 | 0.4815 | 0.1021 | 0.3383 | 0.3040 | 0.1893 |
| `..._finetuned_pd0_wd0.01_ep20` | 20 | 0.4686 | 0.0969 | 0.3350 | 0.2914 | 0.1910 |
| stage-1 best (same eval cfg)    | —  | 0.4842 | 0.0967 | 0.3387 | 0.3109 | 0.1861 |

Stage-2 fine-tuning did not improve BLEU-1, ROUGE-L, or CIDEr at this
eval config — BLEU-4 ticks up slightly at ep10 (+0.005), and longer
fine-tuning (ep20) hurts every metric except METEOR. Consistent with
overfitting once GPT-2 is unfrozen for many extra epochs.
