"""Regression tests for brief node JSON parsing.

Real LLMs (deepseek-v4-*:cloud) wrap JSON responses in ```json fences even
when the prompt asks for raw JSON. The parsing helpers must strip those
fences before calling ``json.loads``; otherwise every LLM call silently
returns empty defaults (0 subtopics, no approved flag, etc.).
"""

from langchain_core.runnables import RunnableConfig

from deepresearch.models import Brief
from deepresearch.nodes.brief import _parse_json_response, decompose_node
from deepresearch.state import ResearchState


def _configurable(chat_fn):
    return {"configurable": {"chat_fn": chat_fn}}


def _state():
    return ResearchState(
        question="What are the health benefits of green tea?",
        slug="green-tea-benefits",
        brief=None,
        round=0,
        auto_round=0,
        mode="user_facing",
        pending_handoff=False,
        subreports={},
        report=None,
        report_references=[],
        coverage=None,
        history=[],
        verify_ok=False,
        verify_attempts=0,
        verify_unsupported=[],
        verify_dangling=[],
        user_approved=False,
    )


def test_parse_json_response_handles_fenced_array():
    """Regression: fenced JSON arrays from real LLMs must parse as lists."""
    fenced = '```json\n[{"slug": "x", "title": "X", "scope": "S", "guiding_questions": []}]\n```'
    result = _parse_json_response(fenced)
    assert isinstance(result, list)
    assert result[0]["slug"] == "x"


def test_parse_json_response_handles_fenced_object():
    """Regression: fenced JSON objects from real LLMs must parse as dicts."""
    fenced = '```json\n{"approved": true}\n```'
    result = _parse_json_response(fenced)
    assert isinstance(result, dict)
    assert result["approved"] is True


def test_parse_json_response_keeps_unfenced_behavior():
    """Sanity: unfenced JSON still works (as the hermetic suite assumes)."""
    result = _parse_json_response('{"needs_clarification": false}')
    assert result == {"needs_clarification": False}


def test_decompose_node_survives_fenced_subtopics(monkeypatch):
    """End-to-end: decompose_node must return a Brief with real subtopics
    when the LLM fences its JSON array response (the deepseek behavior)."""
    monkeypatch.setenv("SUBTOPICS_CEILING", "12")

    fenced = (
        "```json\n"
        "[\n"
        '  {"slug": "alpha", "title": "Alpha", "scope": "A", '
        '"guiding_questions": ["Q1?"], "seed_queries": ["q1"]},\n'
        '  {"slug": "beta", "title": "Beta", "scope": "B", '
        '"guiding_questions": ["Q2?"], "seed_queries": ["q2"]}\n'
        "]\n"
        "```"
    )

    def fake_chat(role, messages):
        assert role == "synth"
        return fenced

    state = _state()
    config = RunnableConfig(configurable={"chat_fn": fake_chat})
    result = decompose_node(state, config)

    brief = result["brief"]
    assert isinstance(brief, Brief)
    assert len(brief.subtopics) == 2
    assert {s.slug for s in brief.subtopics} == {"alpha", "beta"}
