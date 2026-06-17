"""Tests for Pydantic model contracts."""

import pytest

from deepresearch.models import (
    Brief,
    Citation,
    CoverageReport,
    EvidenceExtract,
    EvidencePoint,
    QuestionScore,
    RoundRecord,
    SearchHit,
    SourceRef,
    SubReport,
    SubTopic,
    Verdict,
)
from deepresearch.paths import hash_url


def _roundtrip(model_cls, instance):
    dumped = instance.model_dump()
    restored = model_cls(**dumped)
    assert restored == instance


@pytest.mark.parametrize(
    "model_cls,instance",
    [
        (SubTopic, SubTopic(slug="s", title="T", scope="S", guiding_questions=["Q"])),
        (
            Brief,
            Brief(
                question="Q?",
                slug="q-a1b2c3d4",
                thread_id="q-a1b2c3d4",
                subtopics=[SubTopic(slug="s", title="T", scope="S")],
            ),
        ),
        (
            SourceRef,
            SourceRef(
                id=hash_url("https://example.com/x"),
                type="web",
                url="https://example.com/x",
                title="Example",
                source_path="_sources/abc.md",
                retrieved_at="2024-01-01T00:00:00Z",
                content_hash="def",
            ),
        ),
        (
            Citation,
            Citation(source_id="abc", claim="claim"),
        ),
        (
            EvidenceExtract,
            EvidenceExtract(
                source_id="abc",
                points=[EvidencePoint(claim="c", quote="q")],
            ),
        ),
        (
            SubReport,
            SubReport(subtopic_slug="s", body="body", citations=[]),
        ),
        (
            Verdict,
            Verdict(
                super_slug="sup",
                sub_slug="sub",
                source_id="abc",
                relevant=True,
                reason="good",
            ),
        ),
        (
            QuestionScore,
            QuestionScore(
                subtopic_slug="s",
                question="Q?",
                status="answered",
                supported=True,
            ),
        ),
        (
            CoverageReport,
            CoverageReport(
                per_question=[],
                gaps=[],
                followups=[],
                queued_additions=[],
            ),
        ),
        (
            RoundRecord,
            RoundRecord(
                round=1,
                mode="autonomous",
                coverage_score=0.8,
                fully_covered=False,
            ),
        ),
        (
            SearchHit,
            SearchHit(url="https://x.com", title="X", snippet="..."),
        ),
    ],
)
def test_all_models_roundtrip(model_cls, instance):
    _roundtrip(model_cls, instance)


def test_sourceref_valid_url_id():
    url = "https://example.com/x"
    ref = SourceRef(
        id=hash_url(url),
        type="web",
        url=url,
        title="Example",
        source_path="_sources/abc.md",
        retrieved_at="2024-01-01T00:00:00Z",
        content_hash="def",
    )
    assert ref.id == hash_url(url)


def test_sourceref_invalid_url_id():
    with pytest.raises(ValueError):
        SourceRef(
            id="not-the-hash",
            type="web",
            url="https://example.com/x",
            title="Example",
            source_path="_sources/abc.md",
            retrieved_at="2024-01-01T00:00:00Z",
            content_hash="def",
        )


def test_sourceref_urlless_must_be_pdf():
    with pytest.raises(ValueError):
        SourceRef(
            id="somehash",
            type="web",
            url=None,
            title="Example",
            source_path="_sources/abc.md",
            retrieved_at="2024-01-01T00:00:00Z",
            content_hash="def",
        )


def test_sourceref_urlless_pdf_ok():
    ref = SourceRef(
        id="somehash",
        type="pdf",
        url=None,
        title="Example",
        source_path="_sources/abc.md",
        retrieved_at="2024-01-01T00:00:00Z",
        content_hash="def",
    )
    assert ref.url is None


def test_subtopic_defaults():
    st = SubTopic(slug="s", title="T", scope="S", guiding_questions=[])
    assert st.seed_queries == []
    assert st.dirty is False
    assert st.status == "pending"


def test_citation_optional_quote():
    c = Citation(source_id="abc", claim="claim")
    assert c.supporting_quote is None


def test_verdict_optional_evidence():
    v = Verdict(
        super_slug="sup",
        sub_slug="sub",
        source_id="abc",
        relevant=False,
        reason="bad",
    )
    assert v.evidence is None
