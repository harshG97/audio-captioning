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
           "CLAP", "RUNS_DIR", "EVAL_SETS", "RECAP_ACCOUNT"]


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
])
def test_variant_suffixes(tmp_path, env_vars, name):
    assert settings(submit("run-a", tmp_path, **env_vars))["RUN_NAME"] == name


@pytest.mark.parametrize("env_vars", [
    {"XATTN": 5}, {"TRAIN_DECODER": "yes"}, {"PROMPT_DROPOUT": "abc"},
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
