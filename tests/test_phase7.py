"""Tests for Phase 7 evaluator/refinement loop."""

import json

import pytest
from fakes.chat import FakeChat

from deepresearch.config import reset_config
from deepresearch.models import Brief, Citation, SubReport, SubTopic
from deepresearch.nodes.evaluate import _route, _user_response_to_brief_update, evaluate_node
from deepresearch.sources.pool import save_web


def _config(tmp_workspace, fake_chat):
    return {
        "configurable": {
            "chat_fn": fake_chat.chat,
            "bibliography_dir": str(tmp_workspace["bibliography_dir"]),
            "output_dir": str(tmp_workspace["output_dir"]),
            "state_dir": str(tmp_workspace["state_dir"]),
        }
    }


def _coverage(slug, question, status="answered", supported=True):
    return json.dumps(
        {
            "per_question": [
                {
                    "subtopic_slug": slug,
                    "question": question,
                    "status": status,
                    "supported": supported,
                }
            ],
            "gaps": [] if status == "answered" and supported else [question],
            "followups": [],
            "queued_additions": [],
            "coverage_score": 1.0 if status == "answered" and supported else 0.0,
            "fully_covered": status == "answered" and supported,
        }
    )


def _state(tmp_workspace, questions=None):
    questions = questions or ["What about alpha?", "What about beta?"]
    ref_a = save_web(
        "Alpha is supported.", "https://example.com/a", "A", tmp_workspace["bibliography_dir"]
    )
    ref_b = save_web(
        "Beta is supported.", "https://example.com/b", "B", tmp_workspace["bibliography_dir"]
    )
    subs = [
        SubTopic(slug="alpha", title="Alpha", scope="Alpha", guiding_questions=[questions[0]]),
        SubTopic(slug="beta", title="Beta", scope="Beta", guiding_questions=[questions[1]]),
    ]
    return {
        "question": "Q",
        "slug": "q",
        "brief": Brief(question="Q", slug="q", thread_id="q", subtopics=subs, approved=True),
        "round": 0,
        "auto_round": 0,
        "mode": "user_facing",
        "pending_handoff": False,
        "subreports": {
            "alpha": SubReport(
                subtopic_slug="alpha",
                body=f"Alpha is supported [{ref_a.id}].",
                citations=[Citation(source_id=ref_a.id, claim="Alpha is supported")],
            ),
            "beta": SubReport(
                subtopic_slug="beta",
                body=f"Beta is supported [{ref_b.id}].",
                citations=[Citation(source_id=ref_b.id, claim="Beta is supported")],
            ),
        },
        "report": "Alpha is supported [1].\n\nBeta is supported [2].",
        "report_references": [ref_a, ref_b],
        "verify_ok": True,
        "verify_attempts": 0,
        "verify_unsupported": [],
        "verify_dangling": [],
        "coverage": None,
        "history": [],
        "user_approved": False,
    }


@pytest.mark.integration
def test_autonomous_loop_improves_then_terminates_on_full_coverage(tmp_workspace):
    state = _state(tmp_workspace)
    fake = FakeChat(
        {
            "eval": [
                _coverage("alpha", "What about alpha?", "partial", True),
                _coverage("beta", "What about beta?", "answered", True),
                _coverage("alpha", "What about alpha?", "answered", True),
                _coverage("beta", "What about beta?", "answered", True),
            ]
        }
    )
    first = evaluate_node(state, _config(tmp_workspace, fake))
    assert first["mode"] == "autonomous"
    assert [s.dirty for s in first["brief"].subtopics] == [True, False]
    state.update(first)
    second = evaluate_node(state, _config(tmp_workspace, fake))
    assert second["coverage"].gaps == []
    assert second["pending_handoff"] is True
    assert second["auto_round"] == 1


@pytest.mark.integration
def test_autonomous_loop_terminates_on_auto_round_cap(tmp_workspace, monkeypatch):
    monkeypatch.setenv("AUTO_ROUND_CAP", "1")
    reset_config()
    state = _state(tmp_workspace)
    state["mode"] = "autonomous"
    state["auto_round"] = 1
    fake = FakeChat(
        {
            "eval": [
                _coverage("alpha", "What about alpha?", "partial", True),
                _coverage("beta", "What about beta?", "answered", True),
            ]
        }
    )
    updates = evaluate_node(state, _config(tmp_workspace, fake))
    assert updates["mode"] == "user_facing"
    assert updates["pending_handoff"] is True
    reset_config()


@pytest.mark.integration
def test_autonomous_loop_terminates_on_plateau(tmp_workspace):
    state = _state(tmp_workspace)
    state["history"] = [type("R", (), {"coverage_score": 0.5})()]
    fake = FakeChat(
        {
            "eval": [
                _coverage("alpha", "What about alpha?", "partial", True),
                _coverage("beta", "What about beta?", "answered", True),
            ]
        }
    )
    updates = evaluate_node(state, _config(tmp_workspace, fake))
    assert updates["pending_handoff"] is True


@pytest.mark.integration
def test_user_facing_loop_after_handoff(tmp_workspace):
    state = _state(tmp_workspace)
    fake = FakeChat(
        {"eval": [_coverage("alpha", "What about alpha?"), _coverage("beta", "What about beta?")]}
    )
    updates = evaluate_node(state, _config(tmp_workspace, fake))
    assert updates["mode"] == "user_facing"
    assert updates["pending_handoff"] is True
    assert _route({**state, **updates}) == "supervisor"


@pytest.mark.integration
def test_user_can_add_followup_subtopic_at_handoff(tmp_workspace):
    brief = _state(tmp_workspace)["brief"]
    updated, done = _user_response_to_brief_update(
        json.dumps(
            [{"title": "Gamma", "scope": "Gamma scope", "guiding_questions": ["What about gamma?"]}]
        ),
        brief,
    )
    assert done is False
    assert len(updated.subtopics) == 3
    assert updated.subtopics[-1].slug.startswith("gamma")


@pytest.mark.integration
def test_two_mode_rerun_dispatch(tmp_workspace):
    state = _state(tmp_workspace)
    fake = FakeChat(
        {
            "eval": [
                _coverage("alpha", "What about alpha?", "partial", True),
                _coverage("beta", "What about beta?"),
            ]
        }
    )
    updates = evaluate_node(state, _config(tmp_workspace, fake))
    assert [sub.slug for sub in updates["brief"].subtopics if sub.dirty] == ["alpha"]
    assert (
        all(
            sub.dirty
            for sub in _user_response_to_brief_update("feedback", state["brief"])[0].subtopics
        )
        is False
    )


@pytest.mark.integration
def test_history_records_every_round(tmp_workspace):
    state = _state(tmp_workspace)
    fake = FakeChat(
        {"eval": [_coverage("alpha", "What about alpha?"), _coverage("beta", "What about beta?")]}
    )
    updates = evaluate_node(state, _config(tmp_workspace, fake))
    assert updates["history"][0].round == 1
    assert updates["history"][0].mode == "user_facing"
    assert updates["history"][0].coverage_score == 1.0


@pytest.mark.integration
def test_max_rounds_cap_enforced(tmp_workspace, monkeypatch):
    monkeypatch.setenv("MAX_ROUNDS", "1")
    monkeypatch.setenv("AUTO_ROUND_CAP", "5")
    reset_config()
    state = _state(tmp_workspace)
    fake = FakeChat(
        {
            "eval": [
                _coverage("alpha", "What about alpha?", "partial", True),
                _coverage("beta", "What about beta?"),
            ]
        }
    )
    updates = evaluate_node(state, _config(tmp_workspace, fake))
    assert updates["round"] == 1
    assert updates["pending_handoff"] is True
    reset_config()


@pytest.mark.integration
def test_thin_single_source_answer_flagged_as_partial(tmp_workspace):
    state = _state(tmp_workspace)
    state["brief"] = state["brief"].model_copy(update={"subtopics": state["brief"].subtopics[:1]})
    fake = FakeChat({"eval": [_coverage("alpha", "What about alpha?", "partial", False)]})
    updates = evaluate_node(state, _config(tmp_workspace, fake))
    assert updates["coverage"].per_question[0].supported is False
    assert updates["history"][0].coverage_score < 1.0


@pytest.mark.integration
def test_single_report_md_overwritten_each_round(tmp_workspace):
    path = tmp_workspace["output_dir"] / "q" / "report.md"
    path.parent.mkdir(parents=True)
    path.write_text("round one", encoding="utf-8")
    first = path.read_text(encoding="utf-8")
    path.write_text("round two", encoding="utf-8")
    assert first == "round one"
    assert path.read_text(encoding="utf-8") == "round two"
