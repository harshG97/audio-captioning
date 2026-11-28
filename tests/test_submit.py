"""Tests for scripts/submit.sh run naming and its guards.

Runs submit.sh with LOCAL=1 DRY_RUN=1, which prints the settings a job would
get without running it, so these need no GPU, Slurm or data.

Run: python -m pytest tests/test_submit.py -q
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMIT = REPO_ROOT / "scripts" / "submit.sh"

# Variables a developer's shell may have exported that would change the result.
CLEARED = ["FEAT_DIR", "CACHE_DIR", "ENCODER_NAME", "DECODER_NAME", "RUN_NAME",
           "K", "SEED", "DEDUP", "XATTN", "TRAIN_DECODER", "PROMPT_DROPOUT", "TAG",
           "CLAP", "RUNS_DIR", "EVAL_SETS", "RECAP_ACCOUNT", "KEEP", "PATIENCE",
           "SNAPSHOT_EVERY", "EPOCHS", "RESUME_FROM", "CHECKPOINT_EVERY",
           "BEATS_CKPT", "POOL", "DATA_DIR", "MATCH_FEAT_DIR",
           "TRAIN_DIR", "EMB_DIR", "DATASTORE_DIR", "CAP", "DROP_PCT", "ARCH", "LR",
           "AUDIO_POOL", "AUDIO_MERGE", "LORA_R", "LORA_TARGETS", "LORA_DROPOUT", "PROJ_LR", "LORA_WARMUP",
           "LABELS", "LABELS_DIR", "ZERO_AUDIO", "LAYERS"]


def submit(job, tmp_path, **env_vars):
    env = {k: v for k, v in os.environ.items() if k not in CLEARED}
    env.update(LOCAL="1", DRY_RUN="1", SCRATCH_RECAP=str(tmp_path / "recap"),
               AUDIO_DIR=str(tmp_path / "audio"))
    env.update({k: str(v) for k, v in env_vars.items()})
    return subprocess.run(["bash", str(SUBMIT), job], env=env,
                          capture_output=True, text=True)


def settings(result):
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines()
                if "=" in line and not line.startswith(("bash ", "job:")))


@pytest.mark.parametrize("job, env_vars, name", [
    ("g1", {}, "g1_k0_s42"),
    ("g2", {}, "g2_k4_s42"),
    ("g2", {"K": 2, "DEDUP": 1}, "g2_k2_dedup_s42"),
    ("run-a", {}, "a_paper_k4_s42"),
    ("run-b", {}, "b_full_k4_s42"),
    ("run-c", {}, "c_paper_k0_s42"),
    ("run-a", {"SEED": 1}, "a_paper_k4_s1"),
])
def test_default_run_names_unchanged(tmp_path, job, env_vars, name):
    assert settings(submit(job, tmp_path, **env_vars))["RUN_NAME"] == name


@pytest.mark.parametrize("env_vars, name", [
    ({"XATTN": 28}, "a_paper_k4_x28_s42"),
    ({"XATTN": 3.5}, "a_paper_k4_x3.5_s42"),
    ({"XATTN": 7}, "a_paper_k4_s42"),            # the default adds nothing
    ({"TRAIN_DECODER": 1}, "a_paper_k4_ftdec_s42"),
    ({"TRAIN_DECODER": 0}, "a_paper_k4_s42"),
    ({"PROMPT_DROPOUT": 0.3}, "a_paper_k4_pd0.3_s42"),
    ({"PROMPT_DROPOUT": "0.0"}, "a_paper_k4_s42"),
    ({"DEDUP": 1, "XATTN": 14, "TAG": "try1"}, "a_paper_k4_dedup_x14_try1_s42"),
    ({"KEEP": "last"}, "a_paper_k4_last_s42"),
    ({"KEEP": "best"}, "a_paper_k4_s42"),
    ({"KEEP": "last", "XATTN": 28}, "a_paper_k4_x28_last_s42"),
])
def test_variant_suffixes(tmp_path, env_vars, name):
    assert settings(submit("run-a", tmp_path, **env_vars))["RUN_NAME"] == name


@pytest.mark.parametrize("env_vars", [
    {"XATTN": 5}, {"TRAIN_DECODER": "yes"}, {"KEEP": "first"},
    {"KEEP": "last", "PATIENCE": 5}, {"PROMPT_DROPOUT": "abc"},
    {"PROMPT_DROPOUT": 2}, {"TAG": "has space"}, {"TAG": "a/b"},
])
def test_bad_values_are_refused(tmp_path, env_vars):
    result = submit("run-a", tmp_path, **env_vars)
    assert result.returncode != 0


def test_cache_dir_override_needs_a_tag(tmp_path):
    full = tmp_path / "recap" / "features"
    refused = submit("run-a", tmp_path, CACHE_DIR=full)
    assert refused.returncode != 0 and "TAG" in refused.stderr
    s = settings(submit("run-a", tmp_path, CACHE_DIR=full, TAG="ds2"))
    assert s["RUN_NAME"] == "a_paper_k4_ds2_s42"
    assert s["CACHE_DIR"] == str(full)


def test_run_a_defaults_to_paper_caches(tmp_path):
    s = settings(submit("run-a", tmp_path))
    assert s["CACHE_DIR"] == str(tmp_path / "recap" / "features_paper")


@pytest.mark.parametrize("var, value", [
    ("FEAT_DIR", "/elsewhere/features"),
    ("ENCODER_NAME", "laion/clap-htsat-unfused"),
    ("DECODER_NAME", "gpt2-medium"),
])
def test_other_overrides_need_a_tag(tmp_path, var, value):
    result = submit("run-c", tmp_path, **{var: value})
    assert result.returncode != 0 and var in result.stderr
    assert submit("run-c", tmp_path, TAG="t", **{var: value}).returncode == 0


def test_overrides_equal_to_the_defaults_need_no_tag(tmp_path):
    recap = tmp_path / "recap"
    assert submit("run-a", tmp_path, FEAT_DIR=f"{recap}/features/",
                  CACHE_DIR=recap / "features_paper",
                  ENCODER_NAME="laion/clap-htsat-fused",
                  DECODER_NAME="gpt2").returncode == 0


def test_preprocessing_takes_a_feat_dir_without_a_tag(tmp_path):
    s = settings(submit("preprocess", tmp_path, FEAT_DIR=tmp_path / "f2",
                        CLAP="laion/clap-htsat-unfused"))
    assert s["FEAT_DIR"] == str(tmp_path / "f2")
    assert s["CLAP"] == "laion/clap-htsat-unfused"


def test_keep_last_turns_off_early_stopping(tmp_path):
    assert settings(submit("run-a", tmp_path))["PATIENCE"] == "5"
    s = settings(submit("run-a", tmp_path, KEEP="last"))
    assert s["PATIENCE"] == "0" and s["KEEP"] == "last" and s["EPOCHS"] == "100"
    assert settings(submit("run-a", tmp_path, KEEP="last", PATIENCE=0))["PATIENCE"] == "0"


def test_resume_from_needs_a_tag_and_a_checkpoint(tmp_path):
    ckpt = tmp_path / "run" / "snapshots" / "epoch050"
    ckpt.mkdir(parents=True)
    refused = submit("run-a", tmp_path, RESUME_FROM=ckpt, TAG="from50")
    assert refused.returncode != 0 and "not a full checkpoint" in refused.stderr
    (ckpt / "trainer_state.json").write_text("{}")
    model_only = submit("run-a", tmp_path, RESUME_FROM=ckpt, TAG="from50")
    assert model_only.returncode != 0 and "not a full checkpoint" in model_only.stderr
    (ckpt / "optimizer.pt").write_text("")
    (ckpt / "scheduler.pt").write_text("")
    no_tag = submit("run-a", tmp_path, RESUME_FROM=ckpt)
    assert no_tag.returncode != 0 and "RESUME_FROM" in no_tag.stderr
    s = settings(submit("run-a", tmp_path, RESUME_FROM=ckpt, TAG="from50", KEEP="last"))
    assert s["RUN_NAME"] == "a_paper_k4_last_from50_s42" and s["RESUME_FROM"] == str(ckpt)


def test_beats_features_defaults(tmp_path):
    got = settings(submit("beats-features", tmp_path, BEATS_CKPT=tmp_path / "b.pt"))
    assert got["FEAT_DIR"] == str(tmp_path / "recap" / "features_beats")
    assert got["DATA_DIR"] == "data/paper_split"
    assert got["BEATS_CKPT"] == str(tmp_path / "b.pt")
    assert got["MATCH_FEAT_DIR"] == str(tmp_path / "recap" / "features")
    got = settings(submit("beats-features", tmp_path, BEATS_CKPT="b.pt", POOL=1,
                          FEAT_DIR=tmp_path / "fb"))
    assert got["FEAT_DIR"] == str(tmp_path / "fb") and got["POOL"] == "1"


def test_beats_features_needs_a_checkpoint_and_audio(tmp_path):
    assert submit("beats-features", tmp_path).returncode != 0
    assert submit("beats-features", tmp_path, BEATS_CKPT="b.pt", AUDIO_DIR="").returncode != 0


def _train_dir(tmp_path, val_text=None):
    d = tmp_path / "expanded"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.csv").write_text("youtube_id,start_time,caption\n")
    for split in ("val", "test"):
        text = (REPO_ROOT / "data" / "paper_split" / f"{split}.csv").read_text()
        (d / f"{split}.csv").write_text(val_text if (val_text and split == "val") else text)
    return d


@pytest.mark.parametrize("job", ["run-a", "run-c"])
def test_train_dir_replaces_only_the_training_data(tmp_path, job):
    d = _train_dir(tmp_path)
    got = settings(submit(job, tmp_path, TRAIN_DIR=d, TAG="plus"))
    assert got["DATA_DIR"] == str(d)
    assert "data/paper_split/test.csv" in got["EVAL_SETS"] and got["RUN_NAME"].endswith("_plus_s42")
    assert submit(job, tmp_path, TRAIN_DIR=d).returncode != 0          # needs a TAG


def test_train_dir_must_keep_the_paper_val_and_test(tmp_path):
    d = _train_dir(tmp_path, val_text="youtube_id,start_time,caption\nx,0,other.\n")
    result = submit("run-a", tmp_path, TRAIN_DIR=d, TAG="plus")
    assert result.returncode != 0 and "differs" in result.stderr
    assert submit("g2", tmp_path, TRAIN_DIR=_train_dir(tmp_path / "ok"), TAG="x").returncode != 0


def test_expand_data_defaults(tmp_path):
    got = settings(submit("expand-data", tmp_path))
    assert got["EMB_DIR"] == str(tmp_path / "recap" / "features")
    assert got["DATASTORE_DIR"] == str(tmp_path / "recap" / "features_unfused")
    assert got["CACHE_DIR"] == str(tmp_path / "recap" / "features_unfused_plus")
    assert submit("expand-data", tmp_path, AUDIO_DIR="").returncode != 0


@pytest.mark.parametrize("job, name", [("run-a", "a_paper_k4_prefix_s42"), ("run-c", "c_paper_k0_prefix_s42")])
def test_prefix_arch_defaults(tmp_path, job, name):
    got = settings(submit(job, tmp_path, ARCH="prefix"))
    assert got["RUN_NAME"] == name and got["ARCH"] == "prefix"
    assert got["DECODER_NAME"] == "Qwen/Qwen2.5-1.5B"
    assert (got["EPOCHS"], got["PATIENCE"], got["LR"]) == ("20", "3", "1e-4")
    got = settings(submit(job, tmp_path, ARCH="prefix", EPOCHS=5, LR="2e-4"))
    assert (got["EPOCHS"], got["LR"]) == ("5", "2e-4")


def test_prefix_arch_rejects_cross_attention_settings(tmp_path):
    assert submit("run-a", tmp_path, ARCH="prefix", XATTN=28).returncode != 0
    assert submit("run-a", tmp_path, ARCH="prefix", TRAIN_DECODER=1).returncode != 0
    assert submit("run-a", tmp_path, ARCH="nope").returncode != 0
    # another LLM is a different model: it needs a TAG like any DECODER_NAME
    assert submit("run-a", tmp_path, ARCH="prefix", DECODER_NAME="meta-llama/Llama-3.2-1B").returncode != 0
    assert settings(submit("run-a", tmp_path, ARCH="prefix", DECODER_NAME="meta-llama/Llama-3.2-1B",
                           TAG="llama"))["RUN_NAME"] == "a_paper_k4_prefix_llama_s42"


def test_default_runs_keep_the_old_learning_rate(tmp_path):
    assert "LR" not in settings(submit("run-a", tmp_path))


@pytest.mark.parametrize("env, suffix", [
    (dict(AUDIO_POOL=2, AUDIO_MERGE="concat"), "_prefix_ap2c"),
    (dict(AUDIO_POOL=1), "_prefix_ap1"),
    (dict(AUDIO_MERGE="concat"), "_prefix_ap4c"),
    (dict(PROJ_LR="1e-3"), "_prefix_plr1e-3"),
    (dict(LORA_WARMUP=1), "_prefix_wu1"),
    (dict(LORA_TARGETS="attn", LORA_R=8, LORA_DROPOUT=0.1), "_prefix_r8_attn_ld0.1"),
    (dict(AUDIO_POOL=2, AUDIO_MERGE="concat", PROJ_LR="1e-3", LORA_WARMUP=1), "_prefix_ap2c_plr1e-3_wu1"),
    (dict(AUDIO_POOL=4, AUDIO_MERGE="mean", LORA_R=16, LORA_TARGETS="all", LORA_DROPOUT=0.05, LORA_WARMUP=0),
     "_prefix"),                                              # the defaults add nothing
])
def test_prefix_settings_show_in_the_run_name(tmp_path, env, suffix):
    got = settings(submit("run-c", tmp_path, ARCH="prefix", **env))
    assert got["RUN_NAME"] == f"c_paper_k0{suffix}_s42"
    for k, v in env.items():
        assert got[k] == str(v)


def test_prefix_settings_are_checked(tmp_path):
    for bad in (dict(AUDIO_POOL=3), dict(AUDIO_MERGE="max"), dict(LORA_TARGETS="mlp"), dict(LORA_R=0),
                dict(LORA_R="8.5"), dict(PROJ_LR="fast"), dict(LORA_WARMUP="one"), dict(LORA_DROPOUT="x")):
        assert submit("run-c", tmp_path, ARCH="prefix", **bad).returncode != 0, bad
    for only_prefix in (dict(AUDIO_POOL=2), dict(PROJ_LR="1e-3"), dict(LORA_WARMUP=1), dict(LORA_TARGETS="attn")):
        result = submit("run-c", tmp_path, **only_prefix)                 # GPT-2 run
        assert result.returncode != 0 and "ARCH=prefix" in result.stderr, only_prefix


@pytest.mark.parametrize("env, name", [
    (dict(LABELS=5), "c_paper_k0_lab5_s42"),
    (dict(LABELS=0), "c_paper_k0_s42"),
    (dict(ZERO_AUDIO=1), "c_paper_k0_noaudio_s42"),
    (dict(LABELS=5, ZERO_AUDIO=1, ARCH="prefix"), "c_paper_k0_lab5_noaudio_prefix_s42"),
    (dict(LABELS=3, ARCH="prefix", AUDIO_POOL=2, AUDIO_MERGE="concat"), "c_paper_k0_lab3_prefix_ap2c_s42"),
])
def test_labels_and_no_audio_show_in_the_run_name(tmp_path, env, name):
    got = settings(submit("run-c", tmp_path, **env))
    assert got["RUN_NAME"] == name
    for k, v in env.items():
        assert got[k] == str(v)


def test_labels_settings_are_checked_and_passed_on(tmp_path):
    for bad in (dict(LABELS="five"), dict(LABELS=-1), dict(ZERO_AUDIO=2)):
        assert submit("run-c", tmp_path, **bad).returncode != 0, bad
    got = settings(submit("run-a", tmp_path, LABELS=5, LABELS_DIR=tmp_path / "lab"))
    assert got["LABELS_DIR"] == str(tmp_path / "lab") and got["RUN_NAME"] == "a_paper_k4_lab5_s42"
    got = settings(submit("beats-features", tmp_path, BEATS_CKPT="b.pt", LAYERS="4,8,12",
                          FEAT_DIR=tmp_path / "fl"))
    assert got["LAYERS"] == "4,8,12" and got["FEAT_DIR"] == str(tmp_path / "fl")
    # LAYERS is for the features job; a training job would silently ignore it.
    result = submit("run-c", tmp_path, ARCH="prefix", LAYERS="4,8,12")
    assert result.returncode != 0 and "only applies to beats-features" in result.stderr
    result = submit("beats-features", tmp_path, BEATS_CKPT="b.pt", LAYERS="4-12", FEAT_DIR=tmp_path / "fl")
    assert result.returncode != 0 and "LAYERS must be" in result.stderr
