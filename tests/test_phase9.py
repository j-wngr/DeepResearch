"""Phase 9 hardening tests."""

from __future__ import annotations

import logging

import pytest
from fakes.chat import FakeChat
from fakes.embeddings import FakeEmbeddings

from deepresearch.logging_setup import configure_logging
from deepresearch.models import Brief, SubTopic
from deepresearch.nodes.subagent import build_subagent_subgraph
from deepresearch.persistence import create_checkpointer
from deepresearch.rag.store import ChromaStore
from deepresearch.retry import TransientError, retry_call
from deepresearch.run_state import load_run, load_runs


class RaisingSynthChat(FakeChat):
    def chat(self, role: str, messages: list) -> str:
        if role == "synth":
            raise RuntimeError("boom")
        return super().chat(role, messages)


def _subagent_config(tmp_workspace, chat_fn):
    embeddings = FakeEmbeddings()
    store = ChromaStore(tmp_workspace["state_dir"], embeddings.embed_query)
    return {
        "configurable": {
            "chat_fn": chat_fn,
            "embeddings": embeddings,
            "store": store,
            "bibliography_dir": str(tmp_workspace["bibliography_dir"]),
            "state_dir": str(tmp_workspace["state_dir"]),
            "output_dir": str(tmp_workspace["output_dir"]),
            "retrieve_fn": lambda _query, _super_slug: [],
        },
        "thread_id": "run",
    }


@pytest.mark.unit
def test_retry_call_retries_three_times_before_giving_up():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise TransientError("temporary")

    with pytest.raises(TransientError):
        retry_call(
            flaky,
            max_attempts=3,
            initial_interval=0.1,
            backoff_factor=2.0,
            max_interval=1.0,
            sleep_fn=lambda _seconds: None,
        )
    assert calls["n"] == 3


@pytest.mark.integration
def test_subagent_tool_failure_isolates_not_aborts(tmp_workspace):
    subtopic = SubTopic(
        slug="broken",
        title="Broken",
        scope="Broken tools",
        guiding_questions=["What fails?"],
    )
    state = {
        "slug": "run",
        "subtopic": subtopic,
        "iteration": 0,
        "scratchpad": "",
        "whitelisted": [],
        "evidence": [],
        "draft": None,
        "candidates": [],
        "pending_acquisitions": [],
        "acquisition_gaps": ["old gap"],
        "subreport": None,
        "subreports": {},
        "quality_gate_result": None,
        "isolated": False,
    }
    graph = build_subagent_subgraph()
    result = graph.invoke(state, _subagent_config(tmp_workspace, RaisingSynthChat().chat))
    report = result["subreport"]
    assert result["isolated"] is True
    assert report.shortfall.startswith("subagent isolated")
    assert "isolated" in report.body
    assert "[1]" not in report.body
    assert result["acquisition_gaps"] == []
    assert (tmp_workspace["output_dir"] / "run" / "broken" / "report.md").exists()


@pytest.mark.integration
def test_load_runs_lists_threads(tmp_workspace):
    with create_checkpointer(tmp_workspace["state_dir"]) as checkpointer:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class S(TypedDict):
            question: str
            slug: str

        builder = StateGraph(S)
        builder.add_node("noop", lambda s: {})
        builder.add_edge(START, "noop")
        builder.add_edge("noop", END)
        graph = builder.compile(checkpointer=checkpointer)
        graph.invoke({"question": "Q1", "slug": "one"}, {"configurable": {"thread_id": "one"}})
        graph.invoke({"question": "Q2", "slug": "two"}, {"configurable": {"thread_id": "two"}})
    assert {
        run.thread_id for run in load_runs(tmp_workspace["state_dir"], tmp_workspace["output_dir"])
    } == {
        "one",
        "two",
    }


@pytest.mark.integration
def test_load_run_returns_summary_with_file_presence(tmp_workspace):
    with create_checkpointer(tmp_workspace["state_dir"]) as checkpointer:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class S(TypedDict):
            question: str
            slug: str
            brief: Brief
            round: int
            auto_round: int
            pending_handoff: bool

        builder = StateGraph(S)
        builder.add_node("noop", lambda s: {})
        builder.add_edge(START, "noop")
        builder.add_edge("noop", END)
        graph = builder.compile(checkpointer=checkpointer)
        brief = Brief(question="Q", slug="one", thread_id="one", subtopics=[], approved=True)
        graph.invoke(
            {
                "question": "Q",
                "slug": "one",
                "brief": brief,
                "round": 2,
                "auto_round": 1,
                "pending_handoff": True,
            },
            {"configurable": {"thread_id": "one"}},
        )
    run_dir = tmp_workspace["output_dir"] / "one"
    run_dir.mkdir()
    (run_dir / "brief.md").write_text("brief", encoding="utf-8")
    (run_dir / "report.md").write_text("report", encoding="utf-8")
    summary = load_run(tmp_workspace["state_dir"], tmp_workspace["output_dir"], "one")
    assert summary is not None
    assert summary.has_brief is True
    assert summary.has_report is True
    assert summary.brief_approved is True


@pytest.mark.integration
def test_load_run_missing_returns_none(tmp_workspace):
    assert load_run(tmp_workspace["state_dir"], tmp_workspace["output_dir"], "missing") is None


@pytest.mark.unit
def test_configure_logging_emits_structured_lines(capsys):
    configure_logging("INFO")
    logging.getLogger("deepresearch.test").info(
        "hello", extra={"run_slug": "abc", "node": "writer"}
    )
    captured = capsys.readouterr()
    assert "run_slug=abc" in captured.err
    assert "node=writer" in captured.err
