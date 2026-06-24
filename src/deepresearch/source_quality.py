"""LLM-based source quality gate.

Topic-agnostic: decides whether a document is substantive research content
(paper, article, report) or low-quality garbage (navigation page, link farm,
cookie wall, boilerplate).  Separate from ``gate.py`` which judges relevance
to a specific sub-topic.

Results are cached under ``state_dir/quality_cache.json`` keyed by source_id
so each document is judged at most once across all runs.
"""

import fcntl
import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

from deepresearch.config import get_config
from deepresearch.llm import chat as real_chat
from deepresearch.llm import extract_json
from deepresearch.models import SourceRef
from deepresearch.sources.pool import get as pool_get

logger = logging.getLogger("deepresearch.source_quality")
_CACHE_LOCK = threading.Lock()


def _cache_path(state_dir: Path) -> Path:
    return state_dir / "quality_cache.json"


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse quality cache; starting fresh.")
        return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(cache_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with os.fdopen(fd, "w+", encoding="utf-8") as f:
            fd = -1
            f.seek(0)
            json.dump(cache, f, indent=2)
            f.truncate()
    finally:
        if fd >= 0:
            os.close(fd)


def _build_prompt(markdown: str) -> str:
    return (
        "You are a document quality assessor for a research pipeline.\n"
        "Decide if the document below is substantive research content worth keeping\n"
        "in a bibliography (academic paper, detailed article, technical report, or\n"
        "long-form journalism with sourced claims), OR low-quality garbage that should\n"
        "be discarded (navigation page, link farm, cookie wall, login wall, error page,\n"
        "boilerplate/legal text, thin aggregation page, or stub with no real content).\n\n"
        "Return a JSON object with exactly these fields:\n"
        '  "quality": bool  (true = keep, false = discard),\n'
        '  "reason": string  (one short sentence explaining the verdict).\n\n'
        "Document:\n"
        "---\n"
        f"{markdown}\n"
        "---\n"
    )


def _parse_response(response: str) -> tuple[bool, str]:
    try:
        data = json.loads(extract_json(response))
    except json.JSONDecodeError:
        logger.error("Failed to parse quality gate LLM response as JSON.")
        return True, "parse error — defaulting to keep"
    quality = bool(data.get("quality", True))
    reason = str(data.get("reason", ""))
    return quality, reason


def assess(
    source_ref: SourceRef,
    bibliography_dir: Path,
    state_dir: Path,
    *,
    chat_fn: Callable[[str, list], str] | None = None,
) -> tuple[bool, str]:
    """Return ``(is_quality, reason)`` for ``source_ref``.

    Phase 1 — heuristic pre-check (free, no LLM): applies ``min_source_words``
    and ``max_link_density`` thresholds from config.  Clear garbage is rejected
    immediately without an LLM call.

    Phase 2 — LLM judgment: loads the full document, truncates to
    ``doc_size_cap``, and asks the ``"quality"`` role model.

    Results are cached in ``state_dir/quality_cache.json`` keyed by
    ``source_ref.id`` so each source is assessed at most once.
    """
    from deepresearch.sources.quality import is_acceptable

    chat_fn = chat_fn or real_chat
    cfg = get_config()
    cache_path = _cache_path(state_dir)

    # Cache read
    with _CACHE_LOCK:
        cache = _load_cache(cache_path)
        cached = cache.get(source_ref.id)
    if cached is not None:
        return bool(cached["quality"]), str(cached["reason"])

    # Heuristic pre-check — load body once for both phases
    try:
        markdown = pool_get(source_ref.id, bibliography_dir)
    except FileNotFoundError:
        logger.warning("Source %s not found in pool; skipping quality check.", source_ref.id)
        return True, "source not found — defaulting to keep"

    if not is_acceptable(markdown, min_words=cfg.min_source_words, max_link_density=cfg.max_link_density):
        quality, reason = False, "heuristic: too short or link-heavy"
        _write_cache(cache_path, source_ref.id, quality, reason)
        return quality, reason

    # LLM judgment
    cap = cfg.doc_size_cap
    if len(markdown) > cap:
        markdown = markdown[:cap]

    prompt = _build_prompt(markdown)
    response = chat_fn("quality", [{"role": "user", "content": prompt}])
    quality, reason = _parse_response(response)

    _write_cache(cache_path, source_ref.id, quality, reason)
    return quality, reason


def _write_cache(cache_path: Path, source_id: str, quality: bool, reason: str) -> None:
    if "parse error" in reason:
        return
    with _CACHE_LOCK:
        cache = _load_cache(cache_path)
        if source_id not in cache:
            cache[source_id] = {"quality": quality, "reason": reason}
            _save_cache(cache_path, cache)
