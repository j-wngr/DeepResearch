"""Full-document relevance gate + verdict cache.

`judge()` decides whether a source is relevant to a sub-topic, distills the
supporting evidence as verbatim quotes, and caches the result under a fully
qualified key so verdicts never leak across runs or sub-topics.
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
from deepresearch.models import EvidenceExtract, EvidencePoint, SourceRef, SubTopic, Verdict
from deepresearch.sources.pool import get as pool_get

logger = logging.getLogger("deepresearch.gate")
_CACHE_LOCK = threading.Lock()


def _cache_path(state_dir: Path) -> Path:
    return state_dir / "relevance_cache.json"


def _cache_key(super_slug: str, sub_slug: str, source_id: str) -> str:
    return f"{super_slug}::{sub_slug}::{source_id}"


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse relevance cache; starting fresh.")
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


def _build_prompt(sub: SubTopic, markdown: str) -> str:
    questions = "\n".join(f"- {q}" for q in sub.guiding_questions)
    return (
        f"Sub-topic: {sub.title}\n"
        f"Scope: {sub.scope}\n"
        f"Guiding questions:\n{questions}\n\n"
        "You are judging whether the following document is relevant to the "
        "sub-topic above. If relevant, extract the supporting claims as "
        "verbatim quotes from the document.\n\n"
        "Return a JSON object with exactly these fields:\n"
        '  "relevant": bool,\n'
        '  "reason": string,\n'
        '  "evidence": list of {claim: string, quote: string} objects.\n'
        "If relevant is false, evidence must be an empty list.\n\n"
        "Document:\n"
        "---\n"
        f"{markdown}\n"
        "---\n"
    )


def _parse_response(response: str, super_slug: str, sub_slug: str, source_id: str) -> Verdict:
    try:
        data = json.loads(extract_json(response))
    except json.JSONDecodeError:
        logger.error("Failed to parse gate LLM response as JSON.")
        return Verdict(
            super_slug=super_slug,
            sub_slug=sub_slug,
            source_id=source_id,
            relevant=False,
            reason="LLM response parse error",
            evidence=None,
        )

    relevant = bool(data.get("relevant", False))
    reason = data.get("reason", "")
    evidence = None
    if relevant:
        points = [
            EvidencePoint(claim=item.get("claim", ""), quote=item.get("quote", ""))
            for item in data.get("evidence", [])
        ]
        evidence = EvidenceExtract(source_id=source_id, points=points)

    return Verdict(
        super_slug=super_slug,
        sub_slug=sub_slug,
        source_id=source_id,
        relevant=relevant,
        reason=reason,
        evidence=evidence,
    )


def judge(
    super_slug: str,
    sub: SubTopic,
    source_ref: SourceRef,
    bibliography_dir: Path,
    state_dir: Path,
    *,
    chat_fn: Callable[[str, list], str] | None = None,
) -> Verdict:
    """Return a relevance Verdict for `source_ref` against `sub`.

    Uses a cache keyed by ``(super_slug, sub.slug, source_ref.id)`` stored under
    ``state_dir / relevance_cache.json``. On a cache miss the full document is
    loaded from the pool, possibly truncated to ``config.doc_size_cap``, and
    judged by the ``gate`` role chat model.
    """
    chat_fn = chat_fn or real_chat
    key = _cache_key(super_slug, sub.slug, source_ref.id)
    cache_path = _cache_path(state_dir)

    # Phase 1: cache read (lock held briefly, no I/O beyond the JSON file)
    with _CACHE_LOCK:
        cache = _load_cache(cache_path)
        cached = cache.get(key)
    if cached is not None:
        return Verdict.model_validate(cached)

    # Phase 2: load document and call LLM (no lock — allows parallel judgments)
    markdown = pool_get(source_ref.id, bibliography_dir)
    cap = get_config().doc_size_cap
    if len(markdown) > cap:
        logger.warning(
            "Document %s exceeds doc_size_cap (%d > %d); truncating.",
            source_ref.id,
            len(markdown),
            cap,
        )
        markdown = markdown[:cap]

    prompt = _build_prompt(sub, markdown)
    response = chat_fn("gate", [{"role": "user", "content": prompt}])
    verdict = _parse_response(response, super_slug, sub.slug, source_ref.id)

    # Phase 3: cache write (lock held briefly; re-read to merge concurrent writes)
    if verdict.reason != "LLM response parse error":
        with _CACHE_LOCK:
            cache = _load_cache(cache_path)
            if key not in cache:
                cache[key] = verdict.model_dump()
                _save_cache(cache_path, cache)

    return verdict
