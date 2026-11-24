"""Tests for retrieval/build_retrieval_cache.py: ranking, self-exclusion and
--dedupe_captions.

Run: python -m pytest tests/test_retrieval_cache.py -q
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.build_retrieval_cache import build_cache, caption_key  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# Datastore: "dog barks" appears on three clips (in two spellings), the rest once.
DS_CAPTIONS = ["a dog barks", "A dog barks.", "a dog barks", "a cat meows", "rain falls", "a car passes"]
DS_IDS = np.array(["d1", "d2", "d3", "c1", "r1", "v1"])
DOG, CAT, RAIN, CAR = np.eye(4)
DS_EMB = _unit([DOG, DOG, DOG, CAT + 0.2 * DOG, RAIN + 0.1 * DOG, CAR])
QUERY = _unit([DOG + 0.01 * CAT])          # closest to dog, then cat, then rain


def test_caption_key_ignores_case_punctuation_spacing():
    assert caption_key("A dog  barks.") == caption_key("a dog barks") == "a dog barks"
    assert caption_key("a dog barks loudly") != caption_key("a dog barks")


def test_without_dedupe_duplicates_fill_the_prompt():
    cache = build_cache(["q"], QUERY, DS_EMB, DS_IDS, DS_CAPTIONS, k=4)
    assert [caption_key(c) for c in cache["q"]] == ["a dog barks"] * 3 + ["a cat meows"]


def test_dedupe_returns_k_distinct_captions_in_similarity_order():
    cache = build_cache(["q"], QUERY, DS_EMB, DS_IDS, DS_CAPTIONS, k=4, dedupe=True)
    assert cache["q"] == ["a dog barks", "a cat meows", "rain falls", "a car passes"]


def test_dedupe_with_mmr_is_also_distinct():
    cache = build_cache(["q"], QUERY, DS_EMB, DS_IDS, DS_CAPTIONS, k=3, strategy="mmr", dedupe=True)
    assert len({caption_key(c) for c in cache["q"]}) == 3


def test_self_exclusion_without_dedupe():
    cache = build_cache(["c1"], _unit([CAT]), DS_EMB, DS_IDS, DS_CAPTIONS, k=2)
    assert "a cat meows" not in cache["c1"]


def test_self_exclusion_with_dedupe():
    # c1 is the only clip saying "a cat meows": masked for its own query.
    cache = build_cache(["c1"], _unit([CAT]), DS_EMB, DS_IDS, DS_CAPTIONS, k=2, dedupe=True)
    assert "a cat meows" not in cache["c1"]
    # d1 shares "a dog barks" with d2 and d3, so other clips still provide it,
    # exactly as without dedupe (where d2's and d3's rows stay available).
    cache = build_cache(["d1"], _unit([DOG]), DS_EMB, DS_IDS, DS_CAPTIONS, k=2, dedupe=True)
    assert caption_key(cache["d1"][0]) == "a dog barks"


def test_dedupe_is_a_no_op_without_duplicates():
    caps = ["a", "b", "c", "d"]
    emb = _unit(np.eye(4) + 0.1)
    q = _unit(np.random.default_rng(0).standard_normal((5, 4)))
    ids = [f"q{i}" for i in range(5)]
    assert build_cache(ids, q, emb, np.array(caps), caps, k=3) == \
        build_cache(ids, q, emb, np.array(caps), caps, k=3, dedupe=True)


def test_k_larger_than_available_captions():
    cache = build_cache(["c1"], _unit([CAT]), DS_EMB, DS_IDS, DS_CAPTIONS, k=10, dedupe=True)
    assert len(cache["c1"]) == 3          # 4 distinct captions minus its own
