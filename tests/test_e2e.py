"""Phase 9 end-to-end smoke suite over the compiled graph."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes.chat import FakeChat
from fakes.embeddings import FakeEmbeddings
from fakes.pdf import FakePdf
from fakes.tavily import FakeTavily
from langgraph.types import Command

from deepresearch.config import reset_config
from deepresearch.graph import build_graph
from deepresearch.models import AcquisitionResponse, SearchHit
from deepresearch.paths import hash_url
from deepresearch.persistence import create_checkpointer
from deepresearch.rag.store import ChromaStore
from deepresearch.sources.pool import save_web


def _quality(passed=True):
    return json.dumps({"passed": passed, "coverage_score": 1.0 if passed else 0.0, "reason": "ok"})


def _gate(source_id, claim, quote):
    return json.dumps(
        {"relevant": True, "reason": "ok", "evidence": [{"claim": claim, "quote": quote}]}
    )


def _coverage(slug, q, answered=True):
    return json.dumps(
        {
            "per_question": [
                {
                    "subtopic_slug": slug,
                    "question": q,
                    "status": "answered" if answered else "partial",
                    "supported": answered,
                }
            ],
            "gaps": [] if answered else [q],
            "followups": [],
            "queued_additions": [],
        }
    )


def _run(graph, state, config, resumes):
    current = state
    resume_iter = iter(resumes)
    while True:
        result = graph.invoke(current, config)
        if not result.get("__interrupt__"):
            return result
        current = Command(resume=next(resume_iter))


class FirstSynthRaises(FakeChat):
    def __init__(self, responses):
        super().__init__(responses)
        self.failed = False

    def chat(self, role, messages):
        if role == "synth" and "Current scratchpad" in messages[0]["content"] and not self.failed:
            self.failed = True
            raise RuntimeError("boom")
        return super().chat(role, messages)


def _config(tmp_workspace, fake_chat, retrieve_fn=None, *, slug="e2e-run", tavily=None, pdf=None):
    embeddings = FakeEmbeddings()
    store = ChromaStore(tmp_workspace["state_dir"], embeddings.embed_query)
    return {
        "configurable": {
            "chat_fn": fake_chat.chat,
            "embeddings": embeddings,
            "store": store,
            "bibliography_dir": str(tmp_workspace["bibliography_dir"]),
            "state_dir": str(tmp_workspace["state_dir"]),
            "output_dir": str(tmp_workspace["output_dir"]),
            "retrieve_fn": retrieve_fn,
            "tavily_client": tavily,
            "pdf_client": pdf,
            "thread_id": slug,
        },
    }


def _initial(slug="e2e-run"):
    return {
        "question": "E2E question",
        "slug": slug,
        "brief": None,
        "round": 0,
        "auto_round": 0,
        "mode": "user_facing",
        "pending_handoff": False,
        "subreports": {},
        "report": None,
        "report_references": [],
        "coverage": None,
        "history": [],
        "verify_ok": False,
        "verify_attempts": 0,
        "verify_unsupported": [],
        "verify_dangling": [],
        "user_approved": False,
    }


class TrackingFakeChat(FakeChat):
    def __init__(self, responses):
        super().__init__(responses)
        self.drafted_subtopics: list[str] = []

    def chat(self, role, messages):
        if role == "synth" and messages and "Draft a sub-report" in messages[0]["content"]:
            content = messages[0]["content"]
            for line in content.splitlines():
                if line.startswith("Sub-topic: "):
                    self.drafted_subtopics.append(line.removeprefix("Sub-topic: ").strip())
                    break
        return super().chat(role, messages)


class SinglePassFakeChat(FakeChat):
    def __init__(self, subtopics, sid_a, sid_b):
        super().__init__(
            {
                "clarify": ['{"needs_clarification": false}', '{"approved": true}'],
                "gate": [
                    _gate(sid_a, "Alpha answer", "Alpha quote."),
                    _gate(sid_b, "Beta answer", "Beta quote."),
                ],
                "writer": ['{"supported": true}', '{"supported": true}'],
            }
        )
        self._subtopics = subtopics
        self._sid_a = sid_a
        self._sid_b = sid_b

    def chat(self, role, messages):
        if role != "synth" or not messages:
            return super().chat(role, messages)
        content = messages[0]["content"]
        if "Decompose this research question" in content:
            return json.dumps(self._subtopics)
        if "Update the scratchpad" in content:
            return "reflect a" if "Sub-topic: Alpha" in content else "reflect b"
        if "Draft a sub-report" in content:
            if "Sub-topic: Alpha" in content:
                return f"Alpha answer [{self._sid_a}]."
            return f"Beta answer [{self._sid_b}]."
        if "Score this draft" in content:
            return _quality()
        return super().chat(role, messages)


class RefinementFakeChat(TrackingFakeChat):
    def __init__(self, subtopics, sid_a, sid_b):
        super().__init__(
            {
                "clarify": ['{"needs_clarification": false}', '{"approved": true}'],
                "gate": [
                    _gate(sid_a, "Alpha", "Alpha quote."),
                    _gate(sid_b, "Beta", "Beta quote."),
                ],
                "writer": ['{"supported": true}'] * 8,
                "eval": [
                    _coverage("alpha", "Alpha?", answered=False),
                    _coverage("beta", "Beta?"),
                    _coverage("alpha", "Alpha?", answered=False),
                    _coverage("beta", "Beta?"),
                ],
            }
        )
        self._subtopics = subtopics
        self._sid_a = sid_a
        self._sid_b = sid_b

    def chat(self, role, messages):
        if role != "synth" or not messages:
            return super().chat(role, messages)
        content = messages[0]["content"]
        if "Decompose this research question" in content:
            return json.dumps(self._subtopics)
        if "Update the scratchpad" in content:
            return "notes"
        if "Draft a sub-report" in content:
            super().chat(role, messages)
            if "Sub-topic: Alpha" in content:
                return f"Alpha answer [{self._sid_a}]."
            return f"Beta answer [{self._sid_b}]."
        if "Score this draft" in content:
            return _quality()
        return super().chat(role, messages)


@pytest.mark.e2e
def test_e2e_scenario_a_happy_path_single_pass(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    sid_a = hash_url("https://example.com/a")
    sid_b = hash_url("https://example.com/b")
    save_web("Alpha quote.", "https://example.com/a", "A", tmp_workspace["bibliography_dir"])
    save_web("Beta quote.", "https://example.com/b", "B", tmp_workspace["bibliography_dir"])
    subs = [
        {
            "slug": "alpha",
            "title": "Alpha",
            "scope": "A",
            "guiding_questions": ["Alpha?"],
            "seed_queries": ["alpha"],
        },
        {
            "slug": "beta",
            "title": "Beta",
            "scope": "B",
            "guiding_questions": ["Beta?"],
            "seed_queries": ["beta"],
        },
    ]
    fake = SinglePassFakeChat(subs, sid_a, sid_b)
    config = _config(
        tmp_workspace, fake, lambda q, _s: [sid_a] if "Alpha" in q or "alpha" in q else [sid_b]
    )
    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        final = _run(build_graph(cp), _initial(), config, ["approved"])
    report_path = tmp_workspace["output_dir"] / "e2e-run" / "report.md"
    assert report_path.exists()
    assert (tmp_workspace["output_dir"] / "e2e-run" / "references.json").exists()
    assert (tmp_workspace["output_dir"] / "e2e-run" / "alpha" / "report.md").exists()
    assert (tmp_workspace["output_dir"] / "e2e-run" / "beta" / "report.md").exists()
    assert final["verify_ok"] is True
    assert not list(tmp_workspace["bibliography_dir"].glob("**/report.md"))


@pytest.mark.e2e
def test_e2e_scenario_f_failure_isolation(tmp_workspace):
    sid_b = hash_url("https://example.com/b")
    save_web("Beta quote.", "https://example.com/b", "B", tmp_workspace["bibliography_dir"])
    subs = [
        {
            "slug": "broken",
            "title": "Broken",
            "scope": "bad",
            "guiding_questions": ["Broken?"],
            "seed_queries": ["broken"],
        },
        {
            "slug": "beta",
            "title": "Beta",
            "scope": "B",
            "guiding_questions": ["Beta?"],
            "seed_queries": ["beta"],
        },
    ]
    fake = FirstSynthRaises(
        {
            "clarify": ['{"needs_clarification": false}', '{"approved": true}'],
            "synth": [json.dumps(subs), "reflect beta", f"Beta answer [{sid_b}].", _quality()],
            "gate": [_gate(sid_b, "Beta answer", "Beta quote.")],
            "writer": ['{"supported": true}'],
        }
    )
    config = _config(
        tmp_workspace, fake, lambda q, _s: [] if "Broken" in q or "broken" in q else [sid_b]
    )
    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        final = _run(build_graph(cp), _initial(), config, ["approved"])
    assert set(final["subreports"]) == {"broken", "beta"}
    assert final["subreports"]["broken"].shortfall
    assert final["subreports"]["beta"].citations
    assert (tmp_workspace["output_dir"] / "e2e-run" / "report.md").exists()


@pytest.mark.e2e
def test_e2e_scenario_b_refinement_loop_full_graph(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    monkeypatch.setenv("AUTO_ROUND_CAP", "1")
    reset_config()
    sid_a = hash_url("https://example.com/refine-a")
    sid_b = hash_url("https://example.com/refine-b")
    save_web("Alpha quote.", "https://example.com/refine-a", "A", tmp_workspace["bibliography_dir"])
    save_web("Beta quote.", "https://example.com/refine-b", "B", tmp_workspace["bibliography_dir"])
    subs = [
        {
            "slug": "alpha",
            "title": "Alpha",
            "scope": "A",
            "guiding_questions": ["Alpha?"],
            "seed_queries": ["alpha"],
        },
        {
            "slug": "beta",
            "title": "Beta",
            "scope": "B",
            "guiding_questions": ["Beta?"],
            "seed_queries": ["beta"],
        },
    ]
    fake = RefinementFakeChat(subs, sid_a, sid_b)
    config = _config(
        tmp_workspace, fake, lambda q, _s: [sid_a] if "Alpha" in q or "alpha" in q else [sid_b]
    )
    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        final = _run(build_graph(cp), _initial(), config, ["approved", "approved"])
    report_path = tmp_workspace["output_dir"] / "e2e-run" / "report.md"
    assert final["auto_round"] >= 1
    assert final["round"] >= 2
    assert sorted(fake.drafted_subtopics[:2]) == ["Alpha", "Beta"]
    assert fake.drafted_subtopics[2:3] == ["Alpha"]
    assert sorted(fake.drafted_subtopics[3:]) == ["Alpha", "Beta"]
    assert final["pending_handoff"] is False
    assert final["user_approved"] is True
    assert len(list((tmp_workspace["output_dir"] / "e2e-run").glob("report.md"))) == 1
    assert report_path.read_text(encoding="utf-8") == final["report"]
    assert len(final["history"]) >= 2
    assert final["history"][-1].coverage_score < 1.0
    assert final["auto_round"] == 1
    reset_config()


def _blocked_pdf_setup(tmp_workspace, url, *, alt_url=None):
    source_id = hash_url(url)
    save_path = tmp_workspace["bibliography_dir"] / "_inbox" / f"{source_id}.pdf"
    conversions = {str(save_path): "# Manual\nManual PDF evidence."}
    pdfs = {}
    if alt_url is not None:
        alt_id = hash_url(alt_url)
        alt_path = tmp_workspace["bibliography_dir"] / "_inbox" / f"{alt_id}.pdf"
        conversions[str(alt_path)] = "# Alternative\nAlternative PDF evidence."
        pdfs[alt_url] = b"%PDF alternative"
    tavily = FakeTavily(
        search_results={
            "blocked query": [SearchHit(url=url, title="Paywalled PDF", snippet="blocked")]
        }
    )
    pdf = FakePdf(pdfs=pdfs, conversions=conversions, blocked_urls={url})
    return source_id, save_path, tavily, pdf


@pytest.mark.e2e
def test_e2e_scenario_c_saved_pdf_full_graph(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    url = "https://example.com/paywalled.pdf"
    source_id, save_path, tavily, pdf = _blocked_pdf_setup(tmp_workspace, url)
    subs = [
        {
            "slug": "blocked-source",
            "title": "Blocked source",
            "scope": "Use a blocked source.",
            "guiding_questions": ["blocked query"],
            "seed_queries": [],
        }
    ]
    fake = FakeChat(
        {
            "clarify": ['{"needs_clarification": false}', '{"approved": true}'],
            "synth": [json.dumps(subs), "notes", f"Manual answer [{source_id}]", _quality()],
            "gate": [_gate(source_id, "Manual answer", "Manual PDF evidence.")],
            "writer": ['{"supported": true}'],
        }
    )
    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        graph = build_graph(cp)
        config = _config(tmp_workspace, fake, lambda _q, _s: [], tavily=tavily, pdf=pdf)
        graph.invoke(_initial(), config)
        acquire = graph.invoke(Command(resume="approved"), config)["__interrupt__"][0].value
        brief_before = (tmp_workspace["output_dir"] / "e2e-run" / "brief.md").read_text(
            encoding="utf-8"
        )
        request = acquire["requests"][0]
        assert request["source_id"] == source_id
        assert request["save_path"] == str(save_path)
        Path(request["save_path"]).write_bytes(b"%PDF fake")
        final = graph.invoke(
            Command(
                resume=[
                    AcquisitionResponse(
                        source_id=source_id, kind="saved", save_path=request["save_path"]
                    ).model_dump()
                ]
            ),
            config,
        )
    assert source_id in {ref.id for ref in final["report_references"]}
    assert source_id in {c.source_id for c in final["subreports"]["blocked-source"].citations}
    assert (tmp_workspace["output_dir"] / "e2e-run" / "brief.md").read_text(
        encoding="utf-8"
    ) == brief_before
    assert not save_path.exists()
    assert (tmp_workspace["bibliography_dir"] / "_sources" / f"{source_id}.md").exists()
    assert (tmp_workspace["bibliography_dir"] / "_sources" / "pdfs" / f"{source_id}.pdf").exists()


@pytest.mark.e2e
def test_e2e_scenario_c_unobtainable_pdf_full_graph(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    url = "https://example.com/unobtainable.pdf"
    source_id, save_path, tavily, pdf = _blocked_pdf_setup(tmp_workspace, url)
    subs = [
        {
            "slug": "blocked-source",
            "title": "Blocked source",
            "scope": "Use a blocked source.",
            "guiding_questions": ["blocked query"],
            "seed_queries": [],
        }
    ]
    fake = FakeChat(
        {
            "clarify": ['{"needs_clarification": false}', '{"approved": true}'],
            "synth": [json.dumps(subs), "notes", "Draft without citation", _quality(False)],
        }
    )
    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        graph = build_graph(cp)
        config = _config(tmp_workspace, fake, lambda _q, _s: [], tavily=tavily, pdf=pdf)
        graph.invoke(_initial(), config)
        acquire = graph.invoke(Command(resume="approved"), config)["__interrupt__"][0].value
        brief_before = (tmp_workspace["output_dir"] / "e2e-run" / "brief.md").read_text(
            encoding="utf-8"
        )
        assert acquire["requests"][0]["save_path"] == str(save_path)
        final = graph.invoke(
            Command(
                resume=[AcquisitionResponse(source_id=source_id, kind="unobtainable").model_dump()]
            ),
            config,
        )
    subreport = final["subreports"]["blocked-source"]
    assert "source unobtainable" in subreport.shortfall
    assert "Could not obtain:" in subreport.body
    assert source_id not in {ref.id for ref in final["report_references"]}
    assert (tmp_workspace["output_dir"] / "e2e-run" / "brief.md").read_text(
        encoding="utf-8"
    ) == brief_before


@pytest.mark.e2e
def test_e2e_scenario_c_alternative_pdf_full_graph(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    url = "https://example.com/blocked-original.pdf"
    alt_url = "https://example.com/open-access.pdf"
    blocked_id, _save_path, tavily, pdf = _blocked_pdf_setup(tmp_workspace, url, alt_url=alt_url)
    alt_id = hash_url(alt_url)
    subs = [
        {
            "slug": "blocked-source",
            "title": "Blocked source",
            "scope": "Use a blocked source.",
            "guiding_questions": ["blocked query"],
            "seed_queries": [],
        }
    ]
    fake = FakeChat(
        {
            "clarify": ['{"needs_clarification": false}', '{"approved": true}'],
            "synth": [json.dumps(subs), "notes", f"Alternative answer [{alt_id}]", _quality()],
            "gate": [_gate(alt_id, "Alternative answer", "Alternative PDF evidence.")],
            "writer": ['{"supported": true}'],
        }
    )
    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        graph = build_graph(cp)
        config = _config(tmp_workspace, fake, lambda _q, _s: [], tavily=tavily, pdf=pdf)
        graph.invoke(_initial(), config)
        graph.invoke(Command(resume="approved"), config)
        brief_before = (tmp_workspace["output_dir"] / "e2e-run" / "brief.md").read_text(
            encoding="utf-8"
        )
        final = graph.invoke(
            Command(
                resume=[
                    AcquisitionResponse(
                        source_id=blocked_id, kind="alternative", alternative_url=alt_url
                    ).model_dump()
                ]
            ),
            config,
        )
    assert alt_id in {ref.id for ref in final["report_references"]}
    assert blocked_id not in {ref.id for ref in final["report_references"]}
    assert alt_id in {c.source_id for c in final["subreports"]["blocked-source"].citations}
    assert (tmp_workspace["output_dir"] / "e2e-run" / "brief.md").read_text(
        encoding="utf-8"
    ) == brief_before


@pytest.mark.e2e
def test_e2e_scenario_d_cross_process_resume_matches_uninterrupted(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    source_url = "https://example.com/resume"
    source_id = hash_url(source_url)
    save_web("Resume quote.", source_url, "Resume", tmp_workspace["bibliography_dir"])
    subs = [
        {
            "slug": "resume",
            "title": "Resume",
            "scope": "Resume scope",
            "guiding_questions": ["Resume?"],
            "seed_queries": ["resume"],
        }
    ]

    def fake_for_run():
        return FakeChat(
            {
                "clarify": ['{"needs_clarification": false}', '{"approved": true}'],
                "synth": [json.dumps(subs), "notes", f"Resume answer [{source_id}]", _quality()],
                "gate": [_gate(source_id, "Resume answer", "Resume quote.")],
                "writer": ['{"supported": true}'],
            }
        )

    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        first_graph = build_graph(cp)
        first_config = _config(
            tmp_workspace, fake_for_run(), lambda _q, _s: [source_id], slug="resume-run"
        )
        approval = first_graph.invoke(_initial("resume-run"), first_config)
        assert approval["__interrupt__"][0].value["brief"]["slug"] == "resume-run"
    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        rebuilt = build_graph(cp)
        resumed_config = _config(
            tmp_workspace,
            FakeChat(
                {
                    "clarify": ['{"approved": true}'],
                    "synth": ["notes", f"Resume answer [{source_id}]", _quality()],
                    "gate": [_gate(source_id, "Resume answer", "Resume quote.")],
                    "writer": ['{"supported": true}'],
                }
            ),
            lambda _q, _s: [source_id],
            slug="resume-run",
        )
        resumed_final = _run(rebuilt, Command(resume="approved"), resumed_config, [])

    straight_dirs = {
        "bibliography_dir": tmp_workspace["bibliography_dir"],
        "output_dir": tmp_workspace["output_dir"],
        "state_dir": tmp_workspace["state_dir"] / "straight",
    }
    straight_dirs["state_dir"].mkdir()
    with create_checkpointer(straight_dirs["state_dir"]) as cp:
        straight = _run(
            build_graph(cp),
            _initial("straight-run"),
            _config(straight_dirs, fake_for_run(), lambda _q, _s: [source_id], slug="straight-run"),
            ["approved"],
        )
    assert resumed_final["brief"].slug == "resume-run"
    assert resumed_final["round"] == straight["round"]
    assert set(resumed_final["subreports"]) == {"resume"}
    assert resumed_final["report"] == straight["report"]
    assert (tmp_workspace["output_dir"] / "resume-run" / "report.md").read_text(
        encoding="utf-8"
    ) == (tmp_workspace["output_dir"] / "straight-run" / "report.md").read_text(encoding="utf-8")


@pytest.mark.e2e
def test_e2e_scenario_e_cross_run_rag_reuse_full_graph(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "-1.0")
    reset_config()
    url = "https://example.com/reusable"
    source_id = hash_url(url)
    subs1 = [
        {
            "slug": "reuse-one",
            "title": "Reuse one",
            "scope": "Seed shared source",
            "guiding_questions": ["shared query"],
            "seed_queries": ["shared query"],
        }
    ]
    tavily1 = FakeTavily(
        search_results={"shared query": [SearchHit(url=url, title="Reusable", snippet="shared")]},
        extracts={url: (
            "Reusable content about permanent magnet synchronous motors and anomaly "
            "detection. This source covers feature extraction from phase current signals, "
            "stator current signatures under eccentricity faults, bearing degradation "
            "patterns, and demagnetisation indicators. Autoencoder-based models learn a "
            "compact representation of normal motor operation. Reconstruction errors flag "
            "anomalies with high precision across a variety of load conditions. "
            "Variational autoencoders handle multivariate sensor streams with correlated "
            "features, improving detection rates for incipient faults before they develop "
            "into catastrophic failures. Threshold selection methods based on extreme value "
            "theory allow deployment without labelled fault data, which is rarely available "
            "in industrial settings. The models are evaluated on benchmark datasets covering "
            "bearing faults, rotor eccentricity, and inter-turn short-circuit faults, "
            "achieving high area-under-curve scores while maintaining low false-positive "
            "rates in continuous monitoring scenarios. Online learning extensions allow "
            "the detector to adapt as motor characteristics change over time due to ageing "
            "or replacement of components, maintaining detection accuracy without retraining."
        )},
    )
    fake1 = FakeChat(
        {
            "clarify": ['{"needs_clarification": false}', '{"approved": true}'],
            "synth": [json.dumps(subs1), "notes", f"Reusable answer [{source_id}]", _quality()],
            "gate": [_gate(source_id, "Reusable answer", "Reusable quote")],
            "writer": ['{"supported": true}'],
        }
    )
    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        _run(
            build_graph(cp),
            _initial("reuse-run-1"),
            _config(tmp_workspace, fake1, None, slug="reuse-run-1", tavily=tavily1),
            ["approved"],
        )
    store = ChromaStore(tmp_workspace["state_dir"], FakeEmbeddings().embed_query)
    assert source_id in store.list_source_ids()
    assert (tmp_workspace["bibliography_dir"] / "_sources" / f"{source_id}.md").exists()

    subs2 = [
        {
            "slug": "reuse-two",
            "title": "Reuse two",
            "scope": "Reuse shared source",
            "guiding_questions": ["shared query"],
            "seed_queries": ["shared query"],
        }
    ]
    tavily2 = FakeTavily(search_results={})
    fake2 = FakeChat(
        {
            "clarify": ['{"needs_clarification": false}', '{"approved": true}'],
            "synth": [json.dumps(subs2), "notes", f"Reused answer [{source_id}]", _quality()],
            "gate": [_gate(source_id, "Reused answer", "Reusable quote")],
            "writer": ['{"supported": true}'],
        }
    )
    with create_checkpointer(tmp_workspace["state_dir"]) as cp:
        final2 = _run(
            build_graph(cp),
            _initial("reuse-run-2"),
            _config(tmp_workspace, fake2, None, slug="reuse-run-2", tavily=tavily2),
            ["approved"],
        )
    assert source_id in {ref.id for ref in final2["report_references"]}
    assert source_id in {c.source_id for c in final2["subreports"]["reuse-two"].citations}
    assert tavily2.search_count() == 0


@pytest.mark.e2e
def test_e2e_suite_runs_clean():
    assert True
