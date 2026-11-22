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


def build_prompt(retrieved_captions: list[str]) -> str:
    """
    Build the RECAP prompt string from retrieved captions.

    Args:
        retrieved_captions: list of caption strings from retriever.
            If empty, returns the no-retrieval baseline prompt
            ("This audio sounds like") so the same training/eval code path
            can be used for an ablation without retrieval.
    Returns:
        prompt string ready to feed into GPT-2 tokenizer
    """
    if not retrieved_captions:
        return BASELINE_PROMPT
    joined = CAPTION_SEPARATOR.join(c.strip() for c in retrieved_captions)
    return PROMPT_TEMPLATE.format(retrieved=joined)


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
