"""
caption_filter.py

Detect placeholder and empty captions in the caption CSVs.

The AudioCaps training captions contain annotator placeholders such as
"video unvailable", "not available" or "no audio/video available", plus a
few empty rows. They describe no sound, so they must not be:
  - retrievable from the datastore (they end up copied into predictions), or
  - used as training targets.

Matching is done on the whole (normalised) caption against explicit
patterns, so real descriptions that merely mention a video, such as
"silence occurs throughout the entire video" or "video game sounds", are
kept.

Both precompute_text_embeddings.py and build_retrieval_cache.py call
drop_placeholder_captions() on the same CSV, so datastore rows and
embedding rows stay aligned.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

import pandas as pd

# Full-caption patterns (matched after lowercasing, trimming and stripping
# trailing punctuation).
_PLACEHOLDER_PATTERNS = (
    # "unavailable" and its misspellings (unvailable, uanvailable, unavailabel), unplayable
    r"(the )?video (is )?(u\w*(?:vail|play)\w*|not available|will not play|does not play|won'?t play)",
    r"not available",
    r"no sounds? available",
    r"no audio\b.*\bvideo\b.*",
)
_PLACEHOLDER_RE = re.compile(r"^(?:" + "|".join(_PLACEHOLDER_PATTERNS) + r")$")
# Placeholders are short; never treat a long description as one.
_MAX_PLACEHOLDER_WORDS = 6


def _normalise(caption: str) -> str:
    text = re.sub(r"\s+", " ", caption.strip().lower())
    return text.rstrip(".!?,;: ")


def is_placeholder_caption(caption) -> bool:
    """True for missing/empty captions and known annotator placeholders."""
    if caption is None or (not isinstance(caption, str) and pd.isna(caption)):
        return True
    text = _normalise(str(caption))
    if text in ("", "nan"):
        return True
    if len(text.split()) > _MAX_PLACEHOLDER_WORDS:
        return False
    return bool(_PLACEHOLDER_RE.match(text))


def drop_placeholder_captions(
    df: pd.DataFrame, column: str = "caption", source: str = ""
) -> pd.DataFrame:
    """Return `df` without placeholder/empty captions (index reset)."""
    mask = df[column].map(is_placeholder_caption).astype(bool)
    n = int(mask.sum())
    if n:
        examples = sorted({str(c) for c in df.loc[mask, column].head(50)})[:5]
        where = f" from {source}" if source else ""
        print(f"[caption_filter] Dropped {n} placeholder/empty captions{where}, e.g. {examples}")
    return df.loc[~mask].reset_index(drop=True)


def count_placeholders(captions: Iterable) -> int:
    return sum(1 for c in captions if is_placeholder_caption(c))


def warn_if_cache_has_placeholders(cache: Mapping[str, list], source: str = "") -> int:
    """Warn when a retrieval cache was built from an unfiltered datastore."""
    n = count_placeholders(c for caps in cache.values() for c in caps)
    if n:
        where = f" {source}" if source else ""
        print(
            f"[caption_filter] WARNING: retrieval cache{where} contains {n} "
            f"placeholder/empty captions. It was built from an unfiltered datastore; "
            f"rebuild it with precompute_text_embeddings.py + build_retrieval_cache.py."
        )
    return n
