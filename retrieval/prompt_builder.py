"""
prompt_builder.py

Constructs the text prompt for GPT-2 from retrieved captions.
Follows the exact template from the RECAP paper:

    "Audios similar to this audio sounds like:
     caption1, caption2, ..., captionk. This audio sounds like:"
"""


PROMPT_TEMPLATE = (
    "Audios similar to this audio sounds like: {retrieved}. "
    "This audio sounds like:"
)


def build_prompt(retrieved_captions: list[str]) -> str:
    """
    Build the RECAP prompt string from retrieved captions.

    Args:
        retrieved_captions: list of caption strings from retriever
    Returns:
        prompt string ready to feed into GPT-2 tokenizer
    """
    assert len(retrieved_captions) > 0, "Need at least one retrieved caption."
    joined = ", ".join(retrieved_captions)
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
