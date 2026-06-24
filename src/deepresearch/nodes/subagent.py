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
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from deepresearch import acquisition
from deepresearch.config import get_config
from deepresearch.sources import quality
from deepresearch.llm import extract_json
from deepresearch.models import (
    AcquisitionRequest,
    AcquisitionResponse,
    Blocked,
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
    return {
        "scratchpad": scratchpad,
        "iteration": 0,
        "pending_acquisitions": [],
        "acquisition_gaps": [],
    }


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

    from deepresearch.rag import index as rag_index
    from deepresearch.rag import retrieve
    from deepresearch.sources import pool, web

    sub = state["subtopic"]
    super_slug = state["slug"]
    whitelisted_ids = {ref.id for ref in state["whitelisted"]}

    candidates: list[SourceRef] = []
    seen_ids: set[str] = set()
    pending_acquisitions: list[AcquisitionRequest] = list(state.get("pending_acquisitions", []))
    pending_ids = {req.source_id for req in pending_acquisitions}
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

    # Gap fill via web search. RAG stays first: on the initial pass we only
    # search the web when the pool yielded nothing. But once the subagent has
    # looped (the quality gate failed and sent us back here), RAG keeps
    # returning the same already-rejected candidates, so we must supplement with
    # a web search whenever too few sources have actually been whitelisted --
    # otherwise a pool holding a single off-topic source starves every sub-topic.
    target = get_config().min_sources_per_subtopic
    need_web = (not candidates) or (
        state["iteration"] > 0 and len(whitelisted_ids) < target
    )
    if need_web and tavily_client is not None:
        for query in queries:
            hits = web.search(query, tavily_client)
            for hit in hits:
                source_id = _source_id_from_hit(hit)
                if source_id in seen_ids or source_id in whitelisted_ids:
                    continue
                source_ref = _fetch_from_search_hit(
                    hit,
                    bibliography_dir,
                    store,
                    embeddings,
                    tavily_client,
                    pdf_client,
                    super_topic=super_slug,
                    sub_topic=sub.slug,
                )
                if isinstance(source_ref, Blocked):
                    request = acquisition.build_request(source_ref.url, hit.title, bibliography_dir)
                    if request.source_id not in pending_ids:
                        pending_acquisitions.append(request)
                        pending_ids.add(request.source_id)
                    continue
                if source_ref is None:
                    continue
                seen_ids.add(source_ref.id)
                candidates.append(source_ref)

    if pending_acquisitions:
        resume_value = interrupt(
            {
                "type": "acquire",
                "requests": [request.model_dump() for request in pending_acquisitions],
            }
        )
        responses = _parse_acquisition_responses(resume_value)
        requests_by_id = {request.source_id: request for request in pending_acquisitions}
        gaps = list(state.get("acquisition_gaps", []))
        saved_responses = [response for response in responses if response.kind == "saved"]
        for response in saved_responses:
            save_path = Path(response.save_path or "")
            if not save_path.exists():
                raise FileNotFoundError(f"acquired PDF not found at save_path: {save_path}")
        if saved_responses:
            # Reconcile-on-resume folds all user-dropped PDFs into the pool
            # under their already-named ids before the subagent gates them.
            rag_index.reconcile(bibliography_dir, store, embeddings, pdf_converter=pdf_client)
        for response in responses:
            request = requests_by_id.get(response.source_id)
            if request is None:
                continue
            if response.kind == "saved":
                source_ref = pool.get_ref(response.source_id, bibliography_dir)
                gap = None
            else:
                source_ref, gap = acquisition.apply_response(
                    response,
                    original_url=request.url,
                    original_title=request.title,
                    bibliography_dir=bibliography_dir,
                    chat_fn=configurable.get("chat_fn"),
                    tavily_client=tavily_client,
                    pdf_client=pdf_client,
                    store=store,
                    embeddings=embeddings,
                )
            if source_ref is not None and source_ref.id not in seen_ids:
                candidates.append(source_ref)
                seen_ids.add(source_ref.id)
            if gap is not None:
                gaps.append(gap)
        return {"candidates": candidates, "pending_acquisitions": [], "acquisition_gaps": gaps}

    return {"candidates": candidates, "pending_acquisitions": pending_acquisitions}


def _parse_acquisition_responses(value) -> list[AcquisitionResponse]:
    if isinstance(value, dict):
        raw_values = value.values()
    else:
        raw_values = value or []
    return [
        item if isinstance(item, AcquisitionResponse) else AcquisitionResponse.model_validate(item)
        for item in raw_values
    ]


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
    super_topic: str = "",
    sub_topic: str = "",
) -> SourceRef | Blocked | None:
    """Fetch, convert, and save a single search hit to the pool.

    Does NOT index in RAG — that happens in _gate_node only for sources that
    pass the relevance gate, so the RAG stays free of irrelevant content.
    """
    from deepresearch.sources import pdf as pdf_source
    from deepresearch.sources import pool, web

    if hit.url.endswith(".pdf"):
        if pdf_client is None:
            return None
        fetch_result = pdf_source.fetch(hit.url, pdf_client, bibliography_dir)
        if isinstance(fetch_result, Path):
            markdown = pdf_source.convert(fetch_result, pdf_client)
            if not markdown or not markdown.strip():
                # A PDF that converts to nothing is not a usable source; skip it
                # rather than poison the pool with an empty-body entry.
                logger.warning("Empty conversion for %s; skipping", hit.url)
                return None
            source_ref = pool.save_pdf(
                fetch_result.read_bytes(),
                markdown,
                hit.url,
                hit.title,
                bibliography_dir,
            )
        else:
            return fetch_result
    else:
        markdown = web.extract(hit.url, tavily_client)
        if not markdown or not markdown.strip():
            # An empty extract (paywall, JS-only page, extractor miss) must not
            # be saved -- an empty-body source is retrieved by RAG yet rejected
            # by the gate, and it suppresses the web-search gap-fill.
            logger.warning("Empty extract for %s; skipping", hit.url)
            return None
        cfg = get_config()
        if not quality.is_acceptable(
            markdown,
            min_words=cfg.min_source_words,
            max_link_density=cfg.max_link_density,
        ):
            logger.warning(
                "Low-quality extract for %s (words=%d, link_density=%.2f); skipping",
                hit.url,
                quality.word_count(markdown),
                quality.link_density(markdown),
            )
            return None
        source_ref = pool.save_web(markdown, hit.url, hit.title, bibliography_dir)

    return source_ref


def _gate_node(state: SubAgentState, config: RunnableConfig) -> dict:
    """Run the relevance gate over each candidate; keep whitelisted + evidence.

    RAG indexing happens here — only sources that pass the gate are indexed,
    so the vector store stays free of irrelevant content.
    """
    configurable = config.get("configurable", {})
    chat_fn: Callable[[str, list], str] = configurable["chat_fn"]
    bibliography_dir = Path(configurable["bibliography_dir"])
    state_dir = Path(configurable["state_dir"])
    store = configurable["store"]
    embeddings = configurable["embeddings"]

    from deepresearch import gate
    from deepresearch.rag import index as rag_index
    from deepresearch.sources import pool

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
            rag_index.ingest(
                candidate,
                store,
                embeddings,
                bibliography_dir,
                super_topic=super_slug,
                sub_topic=sub.slug,
            )
            whitelisted.append(candidate)
            evidence.append(verdict.evidence)
        else:
            logger.info(
                "Gate rejected %s (%s); removing from pool",
                candidate.id,
                verdict.reason,
            )
            pool.remove(candidate.id, bibliography_dir)
            store.delete(candidate.id)

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

    # Build an explicit citation key table so the LLM can emit citations that
    # directly reference the stable source ids, improving downstream resolution.
    citation_keys: list[str] = []
    for idx, ref in enumerate(state["whitelisted"], start=1):
        citation_keys.append(f"[{ref.id}] — {ref.title} (index {idx})")
    keys_text = "\n".join(citation_keys) if citation_keys else "(no whitelisted sources yet)"

    prompt = (
        f"Sub-topic: {sub.title}\n"
        f"Scope: {sub.scope}\n\n"
        "CITATION KEYS — use these exact source identifiers in [brackets] for every claim:\n"
        f"{keys_text}\n\n"
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
        "must carry an inline citation using one of the exact citation keys "
        "listed above, in the form [source_id]. Return only the report body."
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
    # Fail loud: a draft with no distilled evidence cannot pass, regardless of
    # what the scorer says. Forcing a non-pass lets the subagent loop (and the
    # web gap-fill in ``_acquire_node`` run) before emitting a shortfall report.
    if result.passed and not state["evidence"]:
        result = QualityGateResult(
            passed=False,
            coverage_score=result.coverage_score,
            reason="no evidence gathered; cannot pass quality gate",
        )
    gaps = list(state.get("acquisition_gaps", []))

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
    elif gaps and not state.get("candidates"):
        shortfall = _shortfall_with_gaps(result.reason, gaps)
        body = _append_gap_paragraph(draft, gaps)
        subreport = SubReport(
            subtopic_slug=sub.slug,
            body=body,
            citations=_build_citations(state["evidence"], state["whitelisted"]),
            shortfall=shortfall,
        )
        _write_sub_report(subreport, output_dir, state["slug"])
        updates["subreport"] = subreport
        updates["subreports"] = {sub.slug: subreport}
    elif state["iteration"] < cfg.subagent_max_iterations:
        # Increment the iteration counter so the next acquire cycle is tracked.
        updates["iteration"] = state["iteration"] + 1
    else:
        # Iteration cap reached while the draft still fails the quality gate.
        # Emit the best effort with the shortfall recorded.
        shortfall = _shortfall_with_gaps(result.reason, gaps) if gaps else result.reason
        body = _append_gap_paragraph(draft, gaps) if gaps else draft
        subreport = SubReport(
            subtopic_slug=sub.slug,
            body=body,
            citations=_build_citations(state["evidence"], state["whitelisted"]),
            shortfall=shortfall,
        )
        _write_sub_report(subreport, output_dir, state["slug"])
        updates["subreport"] = subreport
        updates["subreports"] = {sub.slug: subreport}

    return updates


def _shortfall_with_gaps(reason: str, gaps: list[str]) -> str:
    if not gaps:
        return reason
    if not reason:
        return acquisition.format_gaps(gaps)
    return f"{reason} | {acquisition.format_gaps(gaps)}"


def _append_gap_paragraph(body: str, gaps: list[str]) -> str:
    if not gaps:
        return body
    lines = [body.rstrip(), "", "Acquisition gaps:"]
    for gap in gaps:
        url = gap.removeprefix("source unobtainable: ")
        if gap.startswith("source unobtainable: "):
            lines.append(f"Could not obtain: {url}")
        else:
            lines.append(f"Could not obtain: {gap}")
    return "\n".join(lines).strip() + "\n"


def _parse_quality_gate_response(response: str) -> QualityGateResult:
    try:
        data = json.loads(extract_json(response))
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


def _build_isolated_state(state: SubAgentState, exc: Exception, config: RunnableConfig) -> dict:
    """Build an isolated SubReport after a hard subagent failure."""
    output_dir = Path(config.get("configurable", {}).get("output_dir", get_config().output_dir))
    sub = state["subtopic"]
    isolated_sub = sub.model_copy(update={"status": "isolated"})
    shortfall = f"subagent isolated: {type(exc).__name__}: {exc}"
    body = (
        f"## {sub.title}\n\n"
        f"Could not obtain: this sub-topic was isolated after an internal failure.\n\n"
        f"isolated: {type(exc).__name__}: {exc}\n"
    )
    subreport = SubReport(
        subtopic_slug=sub.slug,
        body=body,
        citations=[],
        shortfall=shortfall,
    )
    _write_sub_report(subreport, output_dir, state["slug"])
    logger.exception(
        "subagent isolated", extra={"run_slug": state["slug"], "subtopic_slug": sub.slug}
    )
    return {
        "subtopic": isolated_sub,
        "subreport": subreport,
        "subreports": {sub.slug: subreport},
        "acquisition_gaps": [],
        "pending_acquisitions": [],
        "isolated": True,
    }


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
    if (
        state.get("subreport") is None
        and not state.get("acquisition_gaps")
        and state["iteration"] <= cfg.subagent_max_iterations
    ):
        return "loop"
    return "end"


def _build_inner_subagent_subgraph(checkpointer=None):
    """Build and compile the unsafe inner subagent subgraph."""
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

    return builder.compile(checkpointer=checkpointer)


def build_subagent_subgraph(checkpointer=None):
    """Build the subagent subgraph wrapped with failure isolation."""
    compiled = _build_inner_subagent_subgraph(checkpointer=checkpointer)

    def _isolating_subagent(state: SubAgentState, config: RunnableConfig) -> dict:
        try:
            result = compiled.invoke(state, config)
            if isinstance(result, dict):
                result.setdefault("isolated", False)
            return result
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - isolation boundary catches all hard failures.
            return _build_isolated_state(state, exc, config)

    builder = StateGraph(SubAgentState)
    builder.add_node("research_subagent", _isolating_subagent)
    builder.add_edge(START, "research_subagent")
    builder.add_edge("research_subagent", END)
    return builder.compile(checkpointer=checkpointer)
