"""Writer node: merge subreports, renumber citations, verify, persist."""

import json
import logging
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from deepresearch import citations
from deepresearch import verify as verify_mod
from deepresearch.paths import output_path
from deepresearch.state import ResearchState

logger = logging.getLogger("deepresearch.nodes.writer")


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
    result = verify_mod.check(body, references, bibliography_dir, chat_fn=chat_fn)
    final_body = result.body

    report_path = output_path(output_dir, slug, "report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(final_body, encoding="utf-8")

    references_path = output_path(output_dir, slug, "references.json")
    references_payload = _build_references_json(slug, references)
    references_path.write_text(json.dumps(references_payload, indent=2), encoding="utf-8")

    return {
        "report": final_body,
        "report_references": references,
        "verify_ok": result.ok,
        "verify_attempts": result.attempts,
        "verify_unsupported": result.unsupported,
        "verify_dangling": result.dangling,
    }
