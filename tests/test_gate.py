"""Tests for the relevance gate and verdict cache."""

import json
import logging
from pathlib import Path

import pytest
from fakes.chat import FakeChat

from deepresearch.config import get_config
from deepresearch.gate import judge
from deepresearch.models import EvidenceExtract, EvidencePoint, SourceRef, SubTopic, Verdict
from deepresearch.sources.pool import save_web


def _subtopic() -> SubTopic:
    return SubTopic(
        slug="fasting",
        title="Intermittent fasting and insulin sensitivity",
        scope="Effects of intermittent fasting on insulin sensitivity in humans.",
        guiding_questions=["Does fasting improve insulin sensitivity?"],
    )


def _make_source_ref(bibliography_dir: Path) -> SourceRef:
    url = "https://example.com/fasting-insulin"
    markdown = (
        "Intermittent fasting has been shown to improve insulin sensitivity "
        "by 30% in clinical trials.\n\n"
        "Other studies confirm metabolic benefits in adults."
    )
    return save_web(
        markdown=markdown,
        url=url,
        title="Fasting and insulin sensitivity",
        bibliography_dir=bibliography_dir,
    )


@pytest.mark.unit
def test_relevant_source_produces_evidence(tmp_workspace, caplog):
    bibliography_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    source_ref = _make_source_ref(bibliography_dir)
    sub = _subtopic()

    quote = (
        "Intermittent fasting has been shown to improve insulin sensitivity "
        "by 30% in clinical trials."
    )
    response = {
        "relevant": True,
        "reason": "Directly addresses the sub-topic",
        "evidence": [
            {
                "claim": "Fasting improves insulin sensitivity",
                "quote": quote,
            }
        ],
    }
    fake = FakeChat({"gate": [json.dumps(response)]})

    verdict = judge(
        super_slug="run-a",
        sub=sub,
        source_ref=source_ref,
        bibliography_dir=bibliography_dir,
        state_dir=state_dir,
        chat_fn=fake.chat,
    )

    assert isinstance(verdict, Verdict)
    assert verdict.relevant is True
    assert verdict.evidence is not None
    assert isinstance(verdict.evidence, EvidenceExtract)
    assert len(verdict.evidence.points) == 1
    assert isinstance(verdict.evidence.points[0], EvidencePoint)
    assert verdict.evidence.points[0].quote == quote
    assert fake.call_count("gate") == 1


@pytest.mark.unit
def test_irrelevant_source_no_evidence(tmp_workspace):
    bibliography_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    source_ref = _make_source_ref(bibliography_dir)
    sub = _subtopic()

    response = {
        "relevant": False,
        "reason": "Not about cardiovascular health",
        "evidence": [],
    }
    fake = FakeChat({"gate": [json.dumps(response)]})

    verdict = judge(
        super_slug="run-a",
        sub=sub,
        source_ref=source_ref,
        bibliography_dir=bibliography_dir,
        state_dir=state_dir,
        chat_fn=fake.chat,
    )

    assert verdict.relevant is False
    assert verdict.evidence is None


@pytest.mark.unit
def test_cache_hit_avoids_llm_call(tmp_workspace):
    bibliography_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    source_ref = _make_source_ref(bibliography_dir)
    sub = _subtopic()

    first_response = {
        "relevant": True,
        "reason": "Directly addresses the sub-topic",
        "evidence": [
            {
                "claim": "Fasting improves insulin sensitivity",
                "quote": (
                    "Intermittent fasting has been shown to improve insulin sensitivity "
                    "by 30% in clinical trials."
                ),
            }
        ],
    }
    fake_first = FakeChat({"gate": [json.dumps(first_response)]})
    verdict1 = judge(
        super_slug="run-a",
        sub=sub,
        source_ref=source_ref,
        bibliography_dir=bibliography_dir,
        state_dir=state_dir,
        chat_fn=fake_first.chat,
    )

    second_response = {
        "relevant": False,
        "reason": "Would be returned if LLM were called again",
        "evidence": [],
    }
    fake_second = FakeChat({"gate": [json.dumps(second_response)]})
    verdict2 = judge(
        super_slug="run-a",
        sub=sub,
        source_ref=source_ref,
        bibliography_dir=bibliography_dir,
        state_dir=state_dir,
        chat_fn=fake_second.chat,
    )

    assert verdict2 == verdict1
    assert fake_first.call_count("gate") == 1
    assert fake_second.call_count("gate") == 0


@pytest.mark.unit
def test_cache_key_isolation(tmp_workspace):
    bibliography_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    source_ref = _make_source_ref(bibliography_dir)
    sub = _subtopic()

    response_a = {
        "relevant": True,
        "reason": "Relevant in run-a context",
        "evidence": [
            {
                "claim": "Fasting improves insulin sensitivity",
                "quote": (
                    "Intermittent fasting has been shown to improve insulin sensitivity "
                    "by 30% in clinical trials."
                ),
            }
        ],
    }
    fake_a = FakeChat({"gate": [json.dumps(response_a)]})
    verdict_a = judge(
        super_slug="run-a",
        sub=sub,
        source_ref=source_ref,
        bibliography_dir=bibliography_dir,
        state_dir=state_dir,
        chat_fn=fake_a.chat,
    )
    assert verdict_a.relevant is True
    assert fake_a.call_count("gate") == 1

    response_b = {
        "relevant": False,
        "reason": "Not about cardiovascular health",
        "evidence": [],
    }
    fake_b = FakeChat({"gate": [json.dumps(response_b)]})
    verdict_b = judge(
        super_slug="run-b",
        sub=sub,
        source_ref=source_ref,
        bibliography_dir=bibliography_dir,
        state_dir=state_dir,
        chat_fn=fake_b.chat,
    )

    assert verdict_b.relevant is False
    assert verdict_b.evidence is None
    assert fake_b.call_count("gate") == 1


@pytest.mark.unit
def test_oversized_doc_truncated_with_warning(tmp_workspace, caplog):
    bibliography_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    cap = get_config().doc_size_cap

    url = "https://example.com/long-doc"
    marker = "RELEVANT-PREFIX "
    filler = "word " * ((cap // 5) + 10_000)
    markdown = marker + filler
    source_ref = save_web(
        markdown=markdown,
        url=url,
        title="Very long document",
        bibliography_dir=bibliography_dir,
    )
    sub = _subtopic()

    response = {
        "relevant": True,
        "reason": "Contains relevant information",
        "evidence": [
            {
                "claim": "The document is relevant",
                "quote": marker,
            }
        ],
    }
    fake = FakeChat({"gate": [json.dumps(response)]})

    with caplog.at_level(logging.WARNING, logger="deepresearch.gate"):
        verdict = judge(
            super_slug="run-a",
            sub=sub,
            source_ref=source_ref,
            bibliography_dir=bibliography_dir,
            state_dir=state_dir,
            chat_fn=fake.chat,
        )

    assert any("truncat" in record.message.lower() for record in caplog.records)
    assert verdict.relevant is True
    assert verdict.evidence is not None


@pytest.mark.unit
def test_llm_parse_error_returns_safe_default(tmp_workspace, caplog):
    bibliography_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    source_ref = _make_source_ref(bibliography_dir)
    sub = _subtopic()

    fake = FakeChat({"gate": ["this is not json {{}"]})

    with caplog.at_level(logging.ERROR, logger="deepresearch.gate"):
        verdict = judge(
            super_slug="run-a",
            sub=sub,
            source_ref=source_ref,
            bibliography_dir=bibliography_dir,
            state_dir=state_dir,
            chat_fn=fake.chat,
        )

    assert verdict.relevant is False
    assert "parse error" in verdict.reason.lower()
    assert verdict.evidence is None

    cache_path = state_dir / "relevance_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        key = f"run-a::{sub.slug}::{source_ref.id}"
        assert key not in cache
