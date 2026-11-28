"""
prompt_builder.py

Constructs the text prompt for GPT-2 from retrieved captions, using the
template from the RECAP reference code (src/template.txt in
github.com/Sreyan88/RECAP):

    Similar audios sound like

    <caption 1>

    <caption 2>
    ...

    This audio sounds like

With no retrieved captions it falls back to BASELINE_PROMPT
("This audio sounds like"), the reference code's no-retrieval prefix.

Whitespace: the reference template ends with a trailing space and appends
the caption without a leading one. Here the prompt has no trailing space
and RecapDataset appends " " + caption, which gives the same text with
GPT-2's natural tokenization (" like", " a", ...) instead of a lone space
token before the caption.
"""

from typing import Optional, Sequence

RETRIEVAL_HEADER = "Similar audios sound like"
CAPTION_SEPARATOR = "\n\n"
BASELINE_PROMPT = "This audio sounds like"
PROMPT_TEMPLATE = RETRIEVAL_HEADER + "\n\n{retrieved}\n\n" + BASELINE_PROMPT

# Lowercase strings at which a decoded continuation is cut in evaluate.py:
# a newline means the model has started another list item or prompt line.
GENERATION_STOP_STRINGS = (
    "\n",
    "\\n",
    RETRIEVAL_HEADER.lower(),
    BASELINE_PROMPT.lower(),
    ".",
)


SOUNDS_HEADER = "Sounds:"
SOUND_LABEL_SEPARATOR = "; "      # AudioSet names contain commas ("Male speech, man speaking")


def select_sound_labels(entries: Sequence, k: int, min_prob: float = 0.1) -> list[str]:
    """Names of a clip's sound labels to show in the prompt: up to `k` of the
    most probable ones with probability >= min_prob, and always the top one.
    `entries` are [name, probability] pairs, most probable first."""
    if k <= 0 or not entries:
        return []
    names = [name for name, prob in entries[:k] if prob >= min_prob]
    return names or [entries[0][0]]


def build_prompt(retrieved_captions: list[str], sound_labels: Optional[Sequence[str]] = None) -> str:
    """
    Build the RECAP prompt string from retrieved captions.

    Args:
        retrieved_captions: list of caption strings from retriever.
            If empty, returns the no-retrieval baseline prompt
            ("This audio sounds like") so the same training/eval code path
            can be used for an ablation without retrieval.
        sound_labels: optional sound labels of this clip (from the audio
            encoder's own classifier). They go on a line of their own,
            "Sounds: speech; typing", right before "This audio sounds like":
            when a long prompt is trimmed from the left, the retrieved
            captions go first and the labels stay.
    Returns:
        prompt string ready to feed into GPT-2 tokenizer
    """
    tail = BASELINE_PROMPT
    if sound_labels:
        tail = (SOUNDS_HEADER + " " + SOUND_LABEL_SEPARATOR.join(n.strip() for n in sound_labels)
                + CAPTION_SEPARATOR + BASELINE_PROMPT)
    if not retrieved_captions:
        return tail
    joined = CAPTION_SEPARATOR.join(c.strip() for c in retrieved_captions)
    return RETRIEVAL_HEADER + CAPTION_SEPARATOR + joined + CAPTION_SEPARATOR + tail


def build_prompt_with_metadata(
    retrieved_captions: list[str],
    strategy: str = "topk",
    k: int = 4,
) -> dict:
    """
    Returns prompt string plus metadata for logging/debugging.

    Args:
        retrieved_captions: list of caption strings
        strategy: "topk" or "mmr"
        k: number retrieved
    Returns:
        dict with keys: prompt, retrieved_captions, strategy, k
    """
    return {
        "prompt": build_prompt(retrieved_captions),
        "retrieved_captions": retrieved_captions,
        "strategy": strategy,
        "k": k,
    }


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    dummy_captions = [
        "a dog is barking loudly",
        "people are talking in the background",
        "an animal is making noise",
        "voices can be heard outside",
    ]

    prompt = build_prompt(dummy_captions)
    print("=== Prompt Output ===")
    print(prompt)

    print("\n=== With Metadata ===")
    result = build_prompt_with_metadata(dummy_captions, strategy="mmr", k=4)
    for key, val in result.items():
        print(f"{key}: {val}")


def first_k(captions: Sequence[str], k: Optional[int]) -> list[str]:
    """The first k retrieved captions (all of them when k is None).

    Caches list captions most similar first, and every strategy we use picks
    greedily, so the first k of a top-4 cache are exactly the top-k: one cache
    serves every k up to the k it was built with.
    """
    return list(captions) if k is None else list(captions[:k])


def checkpoint_sound_labels(config) -> tuple[int, float]:
    """(number of sound labels in the prompt, their minimum probability) a
    checkpoint was trained with; (0, ...) for checkpoints without labels."""
    return int(getattr(config, "sound_labels", 0) or 0), float(getattr(config, "sound_label_min_prob", 0.1))


def checkpoint_retrieval_k(config) -> Optional[int]:
    """The k a checkpoint was trained with, or None for no-retrieval or
    checkpoints that predate the setting (they then use the whole cache entry,
    which is how they were trained)."""
    if not getattr(config, "rag", False):
        return None
    k = getattr(config, "retrieval_k", None)
    return int(k) if k is not None else None
