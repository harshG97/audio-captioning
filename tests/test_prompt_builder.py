"""Tests for retrieval/prompt_builder.py and the stop strings used by evaluate.py.

Run: python -m pytest tests/test_prompt_builder.py -q
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import clean_prediction
from retrieval.prompt_builder import BASELINE_PROMPT, GENERATION_STOP_STRINGS, build_prompt


def test_reference_template():
    p = build_prompt(["a dog barks", "rain falls on a roof"])
    assert p == ("Similar audios sound like\n\na dog barks\n\nrain falls on a roof"
                 "\n\nThis audio sounds like")


def test_baseline_prompt():
    assert build_prompt([]) == BASELINE_PROMPT == "This audio sounds like"


def test_captions_are_stripped():
    assert build_prompt([" a dog barks \n"]) == build_prompt(["a dog barks"])


def test_stop_strings_cut_continuations():
    # The model starting another retrieved-style item or echoing the prompt.
    assert clean_prediction("a dog barks\n\na cat meows", GENERATION_STOP_STRINGS) == "a dog barks"
    assert clean_prediction("a dog barks similar audios sound like", GENERATION_STOP_STRINGS) == "a dog barks"
    assert clean_prediction("this audio sounds like a dog", GENERATION_STOP_STRINGS) == ""
    assert clean_prediction("a man speaks while birds chirp", GENERATION_STOP_STRINGS) == \
        "a man speaks while birds chirp"


def test_first_k():
    from retrieval.prompt_builder import first_k
    caps = ["a", "b", "c", "d"]
    assert first_k(caps, None) == caps
    assert first_k(caps, 2) == ["a", "b"]
    assert first_k(caps, 9) == caps
    assert first_k([], 2) == []


def test_checkpoint_retrieval_k():
    from types import SimpleNamespace
    from retrieval.prompt_builder import checkpoint_retrieval_k
    assert checkpoint_retrieval_k(SimpleNamespace(rag=True, retrieval_k=2)) == 2
    assert checkpoint_retrieval_k(SimpleNamespace(rag=False, retrieval_k=4)) is None
    assert checkpoint_retrieval_k(SimpleNamespace(rag=True)) is None     # older checkpoints
    assert checkpoint_retrieval_k(SimpleNamespace()) is None
