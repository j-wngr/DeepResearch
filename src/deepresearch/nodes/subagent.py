"""Per-sub-topic research subagent subgraph.

A single sub-topic is researched end-to-end through a context-engineered
ReAct loop: plan -> acquire (RAG-first, web/PDF fallback) -> relevance gate
-> reflect -> synthesize draft -> quality gate.  The quality gate is the
exit criterion: pass emits a SubReport and per-sub `report.md`; fail loops
back to acquire (bounded by `subagent_max_iterations`); cap hit emits with
a `shortfall` flag.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from deepresearch.config import get_config
from deepresearch.models import (
    Citation,
    EvidenceExtract,
    QualityGateResult,
    SearchHit,
    SourceRef,
    SubReport,
    SubTopic,
)
from deepresearch.paths import output_path
from deepresearch.state import SubAgentState

logger = logging.getLogger("deepresearch.nodes.subagent")


def _plan_node(state: SubAgentState, config: RunnableConfig) -> dict:
    """Initialize the subagent scratchpad and iteration counter."""
    sub = state["subtopic"]
    lines = [
        f"Sub-topic: {sub.title}",
        f"Slug: {sub.slug}",
        f"Scope: {sub.scope}",
        "Guiding questions:",
    ]
    for question in sub.guiding_questions:
        lines.append(f"  - {question}")
    lines.append("Seed queries:")
    for query in sub.seed_queries:
        lines.append(f"  - {query}")
    lines.append("Open questions: all guiding questions are currently open.")

    scratchpad = "\n".join(lines)
    return {"scratchpad": scratchpad, "iteration": 0}


def _build_queries(sub: SubTopic, scratchpad: str) -> list[str]:
    """Combine guiding questions, seed queries, and any scratchpad queries."""
    queries: list[str] = []
    queries.extend(sub.guiding_questions)
    queries.extend(sub.seed_queries)
    # Pull out lines that look like added search queries from the scratchpad.
    for line in scratchpad.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and any(
            kw in stripped.lower() for kw in ("query:", "search:", "fetch")
        ):
            queries.append(stripped[2:].split(":", 1)[-1].strip())
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


def _acquire_node(state: SubAgentState, config: RunnableConfig) -> dict:
    """Find candidate sources: RAG first, then Tavily/PDF fallback if needed."""
    configurable = config.get("configurable", {})
    store = configurable.get("store")
    embeddings = configurable.get("embeddings")
    tavily_client = configurable.get("tavily_client")
    pdf_client = configurable.get("pdf_client")
    bibliography_dir = Path(configurable["bibliography_dir"])

    from deepresearch.rag import retrieve
    from deepresearch.sources import pool, web

    sub = state["subtopic"]
    super_slug = state["slug"]
    whitelisted_ids = {ref.id for ref in state["whitelisted"]}

    candidates: list[SourceRef] = []
    seen_ids: set[str] = set()
    retrieve_fn = configurable.get("retrieve_fn")

    queries = _build_queries(sub, state["scratchpad"])
    if not queries:
        queries = [sub.title]

    for query in queries:
        source_ids: list[str]
        if retrieve_fn is not None:
            source_ids = retrieve_fn(query, super_slug)
        else:
            source_ids = retrieve.candidates(query, super_slug, store, embeddings)
        for source_id in source_ids:
            if source_id in seen_ids or source_id in whitelisted_ids:
                continue
            try:
                source_ref = pool.get_ref(source_id, bibliography_dir)
            except FileNotFoundError:
                logger.warning("RAG returned missing source id %s; skipping", source_id)
                continue
            seen_ids.add(source_id)
            candidates.append(source_ref)

    # Gap fill: if no local candidates, search the web.
    if not candidates and tavily_client is not None:
        for query in queries:
            hits = web.search(query, tavily_client)
            for hit in hits:
                source_id = _source_id_from_hit(hit)
                if source_id in seen_ids or source_id in whitelisted_ids:
                    continue
                source_ref = _fetch_from_search_hit(
                    hit, bibliography_dir, store, embeddings, tavily_client, pdf_client
                )
                if source_ref is None:
                    continue
                seen_ids.add(source_ref.id)
                candidates.append(source_ref)

    return {"candidates": candidates}


def _source_id_from_hit(hit: SearchHit) -> str:
    """Derive a deterministic source id from a search hit URL."""
    from deepresearch.paths import hash_url

    return hash_url(hit.url)


def _fetch_from_search_hit(
    hit: SearchHit,
    bibliography_dir: Path,
    store,
    embeddings,
    tavily_client,
    pdf_client,
) -> SourceRef | None:
    """Fetch/convert/save/ingest a single search hit, returning its SourceRef."""
    from deepresearch.rag import index as rag_index
    from deepresearch.sources import pdf as pdf_source
    from deepresearch.sources import pool, web

    if hit.url.endswith(".pdf"):
        if pdf_client is None:
            return None
        fetch_result = pdf_source.fetch(hit.url, pdf_client, bibliography_dir)
        if isinstance(fetch_result, Path):
            markdown = pdf_source.convert(fetch_result, pdf_client)
            source_ref = pool.save_pdf(
                fetch_result.read_bytes(),
                markdown,
                hit.url,
                hit.title,
                bibliography_dir,
            )
        else:
            # Blocked PDFs are handled by acquisition interrupts in Phase 8.
            logger.warning("Blocked PDF %s skipped in Phase 4", hit.url)
            return None
    else:
        markdown = web.extract(hit.url, tavily_client)
        source_ref = pool.save_web(markdown, hit.url, hit.title, bibliography_dir)

    rag_index.ingest(source_ref, store, embeddings, bibliography_dir)
    return source_ref


def _gate_node(state: SubAgentState, config: RunnableConfig) -> dict:
    """Run the relevance gate over each candidate; keep whitelisted + evidence."""
    configurable = config.get("configurable", {})
    chat_fn: Callable[[str, list], str] = configurable["chat_fn"]
    bibliography_dir = Path(configurable["bibliography_dir"])
    state_dir = Path(configurable["state_dir"])

    from deepresearch import gate

    sub = state["subtopic"]
    super_slug = state["slug"]

    whitelisted = list(state["whitelisted"])
    evidence = list(state["evidence"])

    for candidate in state["candidates"]:
        verdict = gate.judge(
            super_slug,
            sub,
            candidate,
            bibliography_dir,
            state_dir,
            chat_fn=chat_fn,
        )
        if verdict.relevant and verdict.evidence is not None:
            whitelisted.append(candidate)
            evidence.append(verdict.evidence)

    return {"whitelisted": whitelisted, "evidence": evidence, "candidates": []}


def _reflect_node(state: SubAgentState, config: RunnableConfig) -> dict:
    """Update the scratchpad with new findings and revised queries."""
    configurable = config.get("configurable", {})
    chat_fn: Callable[[str, list], str] = configurable["chat_fn"]

    sub = state["subtopic"]
    evidence = state["evidence"]
    scratchpad = state["scratchpad"]

    evidence_text = _format_evidence(evidence)
    prompt = (
        f"Sub-topic: {sub.title}\n"
        f"Scope: {sub.scope}\n\n"
        "Current scratchpad:\n"
        "---\n"
        f"{scratchpad}\n"
        "---\n\n"
        "Distilled evidence from newly whitelisted sources:\n"
        "---\n"
        f"{evidence_text}\n"
        "---\n\n"
        "Update the scratchpad. Summarize new findings, note which guiding "
        "questions are now addressed, identify remaining gaps, and formulate "
        "concise next search queries (prefix them with 'query:'). Return only "
        "the updated scratchpad text."
    )

    updated = chat_fn("synth", [{"role": "user", "content": prompt}])
    return {"scratchpad": updated}


def _format_evidence(evidence: list[EvidenceExtract]) -> str:
    lines: list[str] = []
    for extract in evidence:
        lines.append(f"Source: {extract.source_id}")
        for point in extract.points:
            lines.append(f"  - claim: {point.claim}")
            lines.append(f"    quote: {point.quote}")
    return "\n".join(lines)


def _synthesize_node(state: SubAgentState, config: RunnableConfig) -> dict:
    """Draft a cited sub-report from evidence and scratchpad."""
    configurable = config.get("configurable", {})
    chat_fn: Callable[[str, list], str] = configurable["chat_fn"]

    sub = state["subtopic"]
    evidence = state["evidence"]
    scratchpad = state["scratchpad"]

    evidence_text = _format_evidence(evidence)
    prompt = (
        f"Sub-topic: {sub.title}\n"
        f"Scope: {sub.scope}\n\n"
        "Guiding questions:\n" + "\n".join(f"- {q}" for q in sub.guiding_questions) + "\n\n"
        "Distilled evidence:\n"
        "---\n"
        f"{evidence_text}\n"
        "---\n\n"
        "Scratchpad:\n"
        "---\n"
        f"{scratchpad}\n"
        "---\n\n"
        "Draft a sub-report that answers the guiding questions. Every claim "
        "must carry an inline citation using the stable source id in the form "
        "[source_id]. Return only the report body."
    )

    draft = chat_fn("synth", [{"role": "user", "content": prompt}])
    return {"draft": draft}


def _quality_gate_node(state: SubAgentState, config: RunnableConfig) -> dict:
    """Score the draft; emit SubReport on pass or after iteration cap."""
    configurable = config.get("configurable", {})
    chat_fn: Callable[[str, list], str] = configurable["chat_fn"]
    output_dir = Path(configurable["output_dir"])

    sub = state["subtopic"]
    draft = state["draft"] or ""
    cfg = get_config()

    questions_text = "\n".join(f"- {q}" for q in sub.guiding_questions)
    evidence_text = _format_evidence(state["evidence"])
    prompt = (
        f"Sub-topic: {sub.title}\n\n"
        "Guiding questions:\n"
        f"{questions_text}\n\n"
        "Draft sub-report:\n"
        "---\n"
        f"{draft}\n"
        "---\n\n"
        "Distilled evidence:\n"
        "---\n"
        f"{evidence_text}\n"
        "---\n\n"
        "Score this draft. Return a JSON object with exactly these fields:\n"
        '  "passed": bool,\n'
        '  "coverage_score": float between 0.0 and 1.0,\n'
        '  "reason": string.\n'
        "A draft passes only if every guiding question is addressed and each "
        "claim has a backing quote in the distilled evidence."
    )

    response = chat_fn("synth", [{"role": "user", "content": prompt}])
    result = _parse_quality_gate_response(response)

    updates: dict[str, Any] = {"quality_gate_result": result}

    if result.passed:
        subreport = SubReport(
            subtopic_slug=sub.slug,
            body=draft,
            citations=_build_citations(state["evidence"], state["whitelisted"]),
            shortfall=None,
        )
        _write_sub_report(subreport, output_dir, state["slug"])
        updates["subreport"] = subreport
        # Also emit to parent's subreports channel (keyed by subtopic slug)
        # so the merge_subreports reducer accumulates results from all subagents.
        updates["subreports"] = {sub.slug: subreport}
    elif state["iteration"] < cfg.subagent_max_iterations:
        # Increment the iteration counter so the next acquire cycle is tracked.
        updates["iteration"] = state["iteration"] + 1
    else:
        # Iteration cap reached while the draft still fails the quality gate.
        # Emit the best effort with the shortfall recorded.
        subreport = SubReport(
            subtopic_slug=sub.slug,
            body=draft,
            citations=_build_citations(state["evidence"], state["whitelisted"]),
            shortfall=result.reason,
        )
        _write_sub_report(subreport, output_dir, state["slug"])
        updates["subreport"] = subreport
        updates["subreports"] = {sub.slug: subreport}

    return updates


def _parse_quality_gate_response(response: str) -> QualityGateResult:
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Failed to parse quality gate response as JSON: %s", response)
        return QualityGateResult(passed=False, coverage_score=0.0, reason="JSON parse error")

    return QualityGateResult(
        passed=bool(data.get("passed", False)),
        coverage_score=float(data.get("coverage_score", 0.0)),
        reason=str(data.get("reason", "")),
    )


# Build the citation list from distilled evidence.
def _build_citations(
    evidence: list[EvidenceExtract],
    whitelisted: list[SourceRef],
) -> list[Citation]:
    """Map each evidence point to a Citation carrying its supporting quote."""
    citations: list[Citation] = []
    for extract in evidence:
        for point in extract.points:
            citations.append(
                Citation(
                    source_id=extract.source_id,
                    claim=point.claim,
                    supporting_quote=point.quote,
                )
            )
    return citations


def _write_sub_report(subreport: SubReport, output_dir: Path, slug: str) -> None:
    report_path = output_path(output_dir, slug, subreport.subtopic_slug, "report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(subreport.body, encoding="utf-8")


def _route_after_quality_gate(state: SubAgentState) -> str:
    """Return the next node: 'end' or 'loop' back to acquire."""
    cfg = get_config()
    result = state.get("quality_gate_result")
    if result is None:
        return "end"
    if result.passed:
        return "end"
    # A failing gate either increments iteration to loop (no subreport yet) or
    # emits a shortfall SubReport when the cap is hit.  Loop only when no report
    # has been emitted and the cap has not been reached.
    if state.get("subreport") is None and state["iteration"] <= cfg.subagent_max_iterations:
        return "loop"
    return "end"


def build_subagent_subgraph():
    """Build and compile the subagent subgraph for one sub-topic."""
    builder = StateGraph(SubAgentState)

    builder.add_node("plan", _plan_node)
    builder.add_node("acquire", _acquire_node)
    builder.add_node("gate", _gate_node)
    builder.add_node("reflect", _reflect_node)
    builder.add_node("synthesize", _synthesize_node)
    builder.add_node("quality_gate", _quality_gate_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "acquire")
    builder.add_edge("acquire", "gate")
    builder.add_edge("gate", "reflect")
    builder.add_edge("reflect", "synthesize")
    builder.add_edge("synthesize", "quality_gate")
    builder.add_conditional_edges(
        "quality_gate",
        _route_after_quality_gate,
        {"loop": "acquire", "end": END},
    )

    return builder.compile()
