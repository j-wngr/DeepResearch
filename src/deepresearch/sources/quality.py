"""Lightweight content-quality filter for extracted web sources.

Two fast, deterministic signals — no LLM calls:
  word_count    — prose words after stripping markdown syntax and code blocks.
  link_density  — fraction of words that are hyperlink anchor text.

A source is acceptable when word_count >= MIN_SOURCE_WORDS and
link_density <= MAX_LINK_DENSITY.  Both thresholds are configurable.
"""

from __future__ import annotations

import re

# Matches fenced code blocks (``` ... ```)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# Matches inline code (`...`)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
# Matches markdown links: [anchor text](url)
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Strips remaining markdown punctuation after link/code removal
_MD_PUNCT_RE = re.compile(r"[#*_~>|\\]")


def word_count(markdown: str) -> int:
    """Count prose words in *markdown*, ignoring code blocks and markdown syntax."""
    text = _CODE_BLOCK_RE.sub(" ", markdown)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _LINK_RE.sub(r" \1 ", text)   # keep anchor text, drop URL
    text = _MD_PUNCT_RE.sub(" ", text)
    return len(text.split())


def link_density(markdown: str) -> float:
    """Return the fraction of words that are hyperlink anchor text (0.0–1.0).

    A high ratio (> 0.5) indicates a directory or index page rather than prose.
    Returns 0.0 for empty documents.
    """
    total = word_count(markdown)
    if total == 0:
        return 0.0
    linked = sum(len(m.group(1).split()) for m in _LINK_RE.finditer(markdown))
    return linked / total


def is_acceptable(markdown: str, *, min_words: int, max_link_density: float) -> bool:
    """Return True when the source meets both quality thresholds."""
    if word_count(markdown) < min_words:
        return False
    if link_density(markdown) > max_link_density:
        return False
    return True
