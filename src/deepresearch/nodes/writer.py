"""Writer node: merge subreports, renumber citations, synthesize, verify, persist."""

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from deepresearch import citations
from deepresearch import verify as verify_mod
from deepresearch.paths import output_path
from deepresearch.state import ResearchState

logger = logging.getLogger("deepresearch.nodes.writer")

_CITATION_RE = re.compile(r"\[\d+\]")


def _build_references_json(slug: str, references) -> dict:
    """Serialize the references list to the on-disk references.json schema."""
    sources = []
    for i, ref in enumerate(references, start=1):
        sources.append(
            {
                "index": i,
                "source_id": ref.id,
                "title": ref.title,
                "url": ref.url,
                "source_path": ref.source_path,
                "type": ref.type,
                "retrieved_at": ref.retrieved_at,
                "content_hash": ref.content_hash,
            }
        )
    return {"slug": slug, "sources": sources}


def _synthesize(body: str, question: str, chat_fn: Callable) -> str:
    """Ask the writer LLM to add an executive summary and reconcile cross-topic overlaps.

    Falls back to the concatenated body if the LLM returns empty output or
    drops all citation markers (which would break downstream verification).
    """
    if not body.strip():
        return body
    prompt = (
        "You are synthesizing a research report from sub-topic sections.\n\n"
        f"Research question: {question}\n\n"
        "Current draft (IMPORTANT: do NOT remove or renumber any [n] citation markers):\n"
        "---\n"
        f"{body}\n"
        "---\n\n"
        "Tasks:\n"
        "1. Write a 2-4 sentence executive summary at the very top.\n"
        "2. Merge duplicate claims across sections (keep the better-supported one with its [n]).\n"
        "3. Where sections genuinely contradict each other, note both views in one sentence.\n"
        "4. Return the full revised report. Every [n] marker must be preserved exactly.\n"
    )
    response = chat_fn("writer_synthesis", [{"role": "user", "content": prompt}])
    if not response or not response.strip():
        logger.warning("synthesis returned empty response; using concatenated body")
        return body
    # Safety guard: if the body had citations but the LLM stripped them all, fall back.
    if _CITATION_RE.search(body) and not _CITATION_RE.search(response):
        logger.warning("synthesis dropped all citation markers; using concatenated body")
        return body
    return response


def _ordered_subreports(state: ResearchState) -> list:
    subreports = state.get("subreports", {})
    brief = state.get("brief")
    if brief is None:
        return list(subreports.values())
    ordered = []
    for subtopic in brief.subtopics:
        report = subreports.get(subtopic.slug)
        if report is not None:
            ordered.append(report)
    return ordered


def writer(state: ResearchState, config: RunnableConfig) -> dict:
    """Synthesize the unified report and verify it."""
    configurable = config.get("configurable", {})
    chat_fn = configurable["chat_fn"]
    bibliography_dir = Path(configurable["bibliography_dir"])
    output_dir = Path(configurable["output_dir"])
    slug = state["slug"]

    subreports = state.get("subreports", {})
    if not subreports:
        return {
            "report": None,
            "report_references": [],
            "verify_ok": False,
            "verify_attempts": 0,
            "verify_unsupported": [],
            "verify_dangling": [],
        }

    ordered = _ordered_subreports(state)
    body, references = citations.merge(
        ordered,
        bibliography_dir,
        question=state.get("question", ""),
    )
    body = _synthesize(body, state.get("question", ""), chat_fn)
    result = verify_mod.check(body, references, bibliography_dir, chat_fn=chat_fn)
    final_body = result.body

    # Fail loud: citations are mandatory, so a report with zero whitelisted
    # sources is a failed research outcome even though groundedness has nothing
    # to check. Surface it rather than reporting verify_ok on an uncited report.
    verify_ok = result.ok and bool(references)

    report_path = output_path(output_dir, slug, "report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(final_body, encoding="utf-8")

    references_path = output_path(output_dir, slug, "references.json")
    references_payload = _build_references_json(slug, references)
    references_path.write_text(json.dumps(references_payload, indent=2), encoding="utf-8")

    return {
        "report": final_body,
        "report_references": references,
        "verify_ok": verify_ok,
        "verify_attempts": result.attempts,
        "verify_unsupported": result.unsupported,
        "verify_dangling": result.dangling,
    }
