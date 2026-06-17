"""Pydantic domain models."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from deepresearch.paths import hash_url


class SearchHit(BaseModel):
    """A single Tavily-style search result."""

    url: str
    title: str
    snippet: str


class EvidencePoint(BaseModel):
    """A distilled evidence point: a claim and its backing verbatim quote."""

    claim: str
    quote: str


class EvidenceExtract(BaseModel):
    """Distilled evidence from a whitelisted source."""

    source_id: str
    points: list[EvidencePoint]


class Citation(BaseModel):
    """An inline citation linking a claim to a source."""

    source_id: str
    claim: str
    supporting_quote: str | None = None


class SubTopic(BaseModel):
    """One bounded facet of a super-topic and mandate for one subagent."""

    slug: str
    title: str
    scope: str
    guiding_questions: list[str] = Field(default_factory=list)
    seed_queries: list[str] = Field(default_factory=list)
    status: Literal["pending", "active", "done", "isolated"] = "pending"
    dirty: bool = False


class Brief(BaseModel):
    """Approved, structured brief: super-topic plus its sub-topics."""

    question: str
    slug: str
    thread_id: str
    subtopics: list[SubTopic]
    approved: bool = False


class SourceRef(BaseModel):
    """Reference to a source in the central pool.

    - `id` is `hash(url)` when a URL exists (web pages and PDFs-by-URL).
    - `id` is `hash(bytes)` for URL-less drop-ins.
    - `content_hash` is always stored separately for dedup/change-detection.
    """

    id: str
    type: Literal["web", "pdf"]
    url: str | None = None
    title: str
    source_path: str
    retrieved_at: str
    content_hash: str

    @model_validator(mode="after")
    def validate_id_consistency(self) -> "SourceRef":
        if self.url is not None:
            expected_id = hash_url(self.url)
            if self.id != expected_id:
                msg = f"SourceRef id {self.id!r} does not match hash(url)={expected_id!r}"
                raise ValueError(msg)
        elif self.type != "pdf":
            raise ValueError("URL-less SourceRef must have type='pdf'")
        return self


class SubReport(BaseModel):
    """A per-sub-topic report."""

    subtopic_slug: str
    body: str
    citations: list[Citation]
    shortfall: str | None = None


class QualityGateResult(BaseModel):
    """Subagent quality gate pass/fail result."""

    passed: bool
    coverage_score: float
    reason: str


class Verdict(BaseModel):
    """Relevance verdict + distilled evidence for a source under a sub-topic."""

    super_slug: str
    sub_slug: str
    source_id: str
    relevant: bool
    reason: str
    evidence: EvidenceExtract | None = None


class QuestionScore(BaseModel):
    """Coverage/support score for one guiding question."""

    subtopic_slug: str
    question: str
    status: Literal["answered", "partial", "unanswered"]
    supported: bool


class CoverageReport(BaseModel):
    """Evaluator output: coverage analysis and proposed follow-ups."""

    per_question: list[QuestionScore]
    gaps: list[str]
    followups: list[SubTopic]
    queued_additions: list[SubTopic]


class RoundRecord(BaseModel):
    """One refinement round in the run history."""

    round: int
    mode: Literal["autonomous", "user_facing"]
    coverage_score: float
    fully_covered: bool


class Blocked:
    """Sentinel returned by pdf.fetch when a URL is paywalled/bot-walled."""

    def __init__(self, url: str, reason: str = "paywall/login wall"):
        self.url = url
        self.reason = reason
