"""TypedDict graph states and channel reducers."""

from typing import Annotated, Literal, TypedDict

from deepresearch.models import (
    AcquisitionRequest,
    Brief,
    CoverageReport,
    EvidenceExtract,
    QualityGateResult,
    RoundRecord,
    SourceRef,
    SubReport,
    SubTopic,
)


def merge_subreports(
    current: dict[str, SubReport],
    incoming: dict[str, SubReport],
) -> dict[str, SubReport]:
    """Upsert by subtopic_slug. Latest value wins; no duplicates."""
    return {**current, **incoming}


def prune_subreports(
    subreports: dict[str, SubReport],
    brief: Brief,
) -> dict[str, SubReport]:
    """Remove subreports whose subtopic_slug is no longer in the brief.

    Called by the gather node before the merged subreports are returned; the
    reducer itself only upserts. This separation keeps the reducer simple while
    still pruning dropped sub-topics using brief context.
    """
    live_slugs = {st.slug for st in brief.subtopics}
    return {slug: report for slug, report in subreports.items() if slug in live_slugs}


def add(
    current: list[RoundRecord],
    incoming: list[RoundRecord],
) -> list[RoundRecord]:
    """Append incoming round records to the history list."""
    return current + incoming


class ResearchState(TypedDict):
    """Top-level graph state."""

    question: str
    slug: Annotated[str, lambda a, b: a or b]
    brief: Brief | None
    round: int
    auto_round: int
    mode: Literal["autonomous", "user_facing"]
    pending_handoff: bool
    subreports: Annotated[dict[str, SubReport], merge_subreports]
    report: str | None
    # Plain list[SourceRef]; no reducer because the writer overwrites it in one step.
    report_references: list[SourceRef]
    verify_ok: bool
    verify_attempts: int
    verify_unsupported: list
    verify_dangling: list
    coverage: CoverageReport | None
    history: Annotated[list[RoundRecord], add]
    user_approved: bool


class SubAgentState(TypedDict):
    """Per-subagent subgraph state."""

    slug: str
    subtopic: SubTopic
    iteration: int
    scratchpad: str
    whitelisted: list[SourceRef]
    evidence: list[EvidenceExtract]
    draft: str | None
    candidates: list[SourceRef]
    pending_acquisitions: list[AcquisitionRequest]
    acquisition_gaps: list[str]
    subreport: SubReport | None
    subreports: dict[str, SubReport]
    quality_gate_result: QualityGateResult | None
    isolated: bool
