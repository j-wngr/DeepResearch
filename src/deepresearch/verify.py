"""Resolution and groundedness verification."""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from deepresearch.citations import extract_inline_citations
from deepresearch.config import get_config
from deepresearch.llm import extract_json
from deepresearch.models import SourceRef
from deepresearch.sources import pool as pool_module

logger = logging.getLogger("deepresearch.verify")


@dataclass
class ClaimVerdict:
    n: int
    claim: str
    source_id: str
    supported: bool
    reason: str = ""


@dataclass
class ResolutionResult:
    ok: bool
    dangling: list[int] = field(default_factory=list)


@dataclass
class VerifyResult:
    body: str
    references: list[SourceRef]
    ok: bool
    attempts: int
    unsupported: list[ClaimVerdict] = field(default_factory=list)
    dangling: list[int] = field(default_factory=list)


def resolve(body: str, references: list[SourceRef]) -> ResolutionResult:
    """Every [n] in body must be 1 <= n <= len(references)."""
    decimal_re = re.compile(r"\[(\d+)\]")
    used = [int(m.group(1)) for m in decimal_re.finditer(body)]
    dangling = [n for n in used if n < 1 or n > len(references)]
    return ResolutionResult(ok=not dangling, dangling=dangling)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_claims(body: str) -> list[str]:
    """Split a body into approximate sentences."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]


def _claim_for_n(body: str, n: int) -> str | None:
    """Return the first sentence in `body` that contains `[n]`, or None."""
    needle = f"[{n}]"
    for sentence in _split_claims(body):
        if needle in sentence:
            return sentence
    return None


def _parse_supported(response: str) -> tuple[bool, str]:
    try:
        data = json.loads(extract_json(response))
        return bool(data.get("supported", False)), str(data.get("reason", ""))
    except json.JSONDecodeError:
        compact = re.sub(r"\s+", "", response.lower())
        if '"supported":true' in compact or "supported:true" in compact:
            return True, ""
        if '"supported":false' in compact or "supported:false" in compact:
            return False, ""
        logger.warning("Failed to parse groundedness response: %s", response)
        return False, "unparseable groundedness response"


def ground_claims(
    body: str,
    references: list[SourceRef],
    bibliography_dir: Path,
    *,
    chat_fn: Callable[[str, list], str],
) -> list[ClaimVerdict]:
    """Judge each distinct cited claim against its full source document."""
    ordered: list[int] = []
    seen: set[int] = set()
    for token in extract_inline_citations(body):
        if token.isdecimal():
            n = int(token)
            if n not in seen:
                seen.add(n)
                ordered.append(n)

    markdown_cache: dict[str, str] = {}
    verdicts: list[ClaimVerdict] = []
    for n in ordered:
        if n < 1 or n > len(references):
            continue
        ref = references[n - 1]
        claim = _claim_for_n(body, n)
        if not claim:
            verdicts.append(ClaimVerdict(n, "", ref.id, False, "no claim sentence"))
            continue
        if ref.id not in markdown_cache:
            try:
                markdown_cache[ref.id] = pool_module.get(ref.id, bibliography_dir)
            except FileNotFoundError:
                logger.warning(
                    "Source %s missing during groundedness check; skipping claim", ref.id
                )
                verdicts.append(
                    ClaimVerdict(n, claim, ref.id, False, "source deleted during verification")
                )
                continue
        markdown = markdown_cache[ref.id]
        prompt = (
            f"Claim: {claim}\n\n"
            f"Source ({ref.id}):\n---\n{markdown}\n---\n\n"
            'Return JSON {"supported": bool, "reason": str}.'
        )
        response = chat_fn("writer", [{"role": "user", "content": prompt}])
        supported, reason = _parse_supported(response)
        verdicts.append(ClaimVerdict(n, claim, ref.id, supported, reason))
    return verdicts


def _remove_sentence(body: str, sentence: str) -> str:
    if not sentence:
        return body
    return body.replace(sentence, "", 1).replace("\n\n\n", "\n\n").strip() + "\n"


def _revise(
    body: str,
    unsupported: list[ClaimVerdict],
    *,
    chat_fn: Callable[[str, list], str],
) -> str:
    """Ask the writer model to drop or revise each unsupported claim."""
    if not unsupported:
        return body
    prompt_lines = [
        "Revise unsupported claims. For each claim, return a replacement sentence",
        "with the citation number preserved, or return DROP uppercase alone on a line.",
        "Unsupported claims:",
    ]
    for verdict in unsupported:
        prompt_lines.append(f"[{verdict.n}] {verdict.claim}\nReason: {verdict.reason}")
    prompt_lines.append("\nCurrent body:\n---\n" + body + "\n---")
    response = chat_fn("writer", [{"role": "user", "content": "\n".join(prompt_lines)}]).strip()
    if not response:
        return body
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return body
    revised = body
    if len(lines) == 1 and lines[0] == "DROP":
        for verdict in unsupported:
            revised = _remove_sentence(revised, verdict.claim)
        return revised
    if len(lines) < len(unsupported):
        logger.warning(
            "_revise: LLM returned %d lines for %d unsupported claims; applying partial revision",
            len(lines),
            len(unsupported),
        )
    for verdict, line in zip(unsupported, lines, strict=False):
        if line == "DROP":
            revised = _remove_sentence(revised, verdict.claim)
        elif verdict.claim:
            revised = revised.replace(verdict.claim, line, 1)
    return revised


def check(
    body: str,
    references: list[SourceRef],
    bibliography_dir: Path,
    *,
    chat_fn: Callable[[str, list], str],
    max_revisions: int | None = None,
) -> VerifyResult:
    """Top-level verification: resolve + groundedness, with bounded revision."""
    resolution = resolve(body, references)
    if not resolution.ok:
        return VerifyResult(body, references, False, 0, dangling=resolution.dangling)

    cfg = get_config()
    cap = max_revisions if max_revisions is not None else cfg.writer_max_revisions
    attempts = 0
    current = body

    for _ in range(cap):
        verdicts = ground_claims(current, references, bibliography_dir, chat_fn=chat_fn)
        unsupported = [v for v in verdicts if not v.supported]
        if not unsupported:
            return VerifyResult(current, references, True, attempts, unsupported=[])
        attempts += 1
        current = _revise(current, unsupported, chat_fn=chat_fn)
        resolution = resolve(current, references)
        if not resolution.ok:
            return VerifyResult(current, references, False, attempts, dangling=resolution.dangling)

    resolution = resolve(current, references)
    if not resolution.ok:
        return VerifyResult(current, references, False, attempts, dangling=resolution.dangling)
    verdicts = ground_claims(current, references, bibliography_dir, chat_fn=chat_fn)
    unsupported = [v for v in verdicts if not v.supported]
    return VerifyResult(current, references, not unsupported, attempts, unsupported=unsupported)
