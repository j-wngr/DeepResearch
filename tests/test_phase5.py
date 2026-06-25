"""Integration tests for Phase 5 orchestration (brief, fan-out, gather, CLI)."""

import json
from pathlib import Path

import pytest
from fakes.chat import FakeChat
from fakes.embeddings import FakeEmbeddings
from langgraph.types import Command

from deepresearch.config import reset_config
from deepresearch.graph import build_graph
from deepresearch.models import Brief, SubReport, SubTopic
from deepresearch.nodes.supervisor import supervisor
from deepresearch.paths import hash_url, output_path, slug
from deepresearch.persistence import create_checkpointer
from deepresearch.rag.store import ChromaStore
from deepresearch.sources.pool import save_web


def _gate_response(relevant: bool, source_id: str, claim: str, quote: str) -> str:
    evidence = []
    if relevant:
        evidence.append({"claim": claim, "quote": quote})
    return json.dumps(
        {
            "relevant": relevant,
            "reason": "relevant" if relevant else "not relevant",
            "evidence": evidence,
        }
    )


def _quality_gate_response(passed: bool, reason: str = "") -> str:
    return json.dumps(
        {
            "passed": passed,
            "coverage_score": 1.0 if passed else 0.3,
            "reason": reason or ("pass" if passed else "insufficient coverage"),
        }
    )


def _seed_source(bibliography_dir: Path, url: str, title: str, content: str):
    return save_web(content, url, title, bibliography_dir)


def _run_graph_interactive(graph, initial_state, config, resume_sequence):
    """Run the graph, feeding resume values from the sequence at each interrupt."""
    resume_iter = iter(resume_sequence)
    current_input = initial_state
    final_state = None

    while True:
        result = graph.invoke(current_input, config)
        interrupts = result.get("__interrupt__", [])
        if not interrupts:
            final_state = result
            break
        try:
            next_resume = next(resume_iter)
        except StopIteration:
            raise RuntimeError("Graph interrupted but no more resume values in sequence")
        current_input = Command(resume=next_resume)

    return final_state


def _build_config(bib_dir: Path, state_dir: Path, out_dir: Path, fake_chat: FakeChat):
    embeddings = FakeEmbeddings()
    store = ChromaStore(state_dir, embeddings.embed_query)
    return {
        "configurable": {
            "chat_fn": fake_chat.chat,
            "embeddings": embeddings,
            "store": store,
            "bibliography_dir": str(bib_dir),
            "state_dir": str(state_dir),
            "output_dir": str(out_dir),
        },
        "thread_id": "intermittent-fasting-effects-on-cv-health-2010-2024",
    }


@pytest.mark.integration
def test_interrupt_resume_roundtrip(tmp_workspace, monkeypatch):
    """Clarify + approval interrupts resume via Command(resume=...)."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]

    source_ids = []
    for i, (url, content) in enumerate(
        [
            ("https://example.com/fasting-mechanisms", "IF affects metabolism via ketosis."),
            ("https://example.com/cv-outcomes", "IF reduces cardiovascular risk."),
            ("https://example.com/fasting-longevity", "IF extends lifespan in animal models."),
        ]
    ):
        sid = hash_url(url)
        _seed_source(bib_dir, url, f"Source {i}", content)
        source_ids.append(sid)

    def retrieve_fn(query, super_slug):
        for url, _sid in [
            ("fasting-mechanisms", source_ids[0]),
            ("cv-outcomes", source_ids[1]),
            ("longevity", source_ids[2]),
        ]:
            if url in query.lower():
                return [_sid]
        return source_ids[:1]

    fake_chat = FakeChat(
        {
            "clarify": [
                '{"needs_clarification": true, "questions": ["What time range?"]}',
                "Refined: effects of IF on CV health 2010-2024",
                '{"approved": false}',
                '{"approved": true}',
            ],
            "synth": [
                json.dumps(
                    [
                        {
                            "slug": "fasting-mechanisms",
                            "title": "Fasting Mechanisms",
                            "scope": "Biological mechanisms of IF",
                            "guiding_questions": ["How does IF affect metabolism?"],
                            "seed_queries": ["IF metabolism"],
                        },
                        {
                            "slug": "cv-outcomes",
                            "title": "CV Outcomes",
                            "scope": "Cardiovascular outcomes of IF",
                            "guiding_questions": ["Does IF reduce CVD risk?"],
                            "seed_queries": ["IF CVD"],
                        },
                    ]
                ),
                json.dumps(
                    [
                        {
                            "slug": "fasting-mechanisms",
                            "title": "Fasting Mechanisms",
                            "scope": "Biological mechanisms of IF",
                            "guiding_questions": ["How does IF affect metabolism?"],
                            "seed_queries": ["IF metabolism"],
                        },
                        {
                            "slug": "cv-outcomes",
                            "title": "CV Outcomes",
                            "scope": "Cardiovascular outcomes of IF",
                            "guiding_questions": ["Does IF reduce CVD risk?"],
                            "seed_queries": ["IF CVD"],
                        },
                        {
                            "slug": "fasting-longevity",
                            "title": "Longevity",
                            "scope": "IF and longevity",
                            "guiding_questions": ["Does IF extend lifespan?"],
                            "seed_queries": ["IF longevity"],
                        },
                    ]
                ),
                "Reflect mechanisms.",
                "Draft mechanisms report.",
                _quality_gate_response(True),
                "Reflect outcomes.",
                "Draft outcomes report.",
                _quality_gate_response(True),
                "Reflect longevity.",
                "Draft longevity report.",
                _quality_gate_response(True),
            ],
            "gate": [
                _gate_response(
                    True,
                    source_ids[0],
                    "IF affects metabolism",
                    "IF affects metabolism via ketosis.",
                ),
                _gate_response(
                    True, source_ids[1], "IF reduces CVD risk", "IF reduces cardiovascular risk."
                ),
                _gate_response(
                    True,
                    source_ids[2],
                    "IF extends lifespan",
                    "IF extends lifespan in animal models.",
                ),
            ],
        }
    )

    config = _build_config(bib_dir, state_dir, out_dir, fake_chat)
    config["configurable"]["retrieve_fn"] = retrieve_fn
    run_slug = config["thread_id"]

    with create_checkpointer(state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        initial_state = {
            "question": "Effects of intermittent fasting on cardiovascular health",
            "slug": run_slug,
            "brief": None,
            "round": 0,
            "auto_round": 0,
            "mode": "user_facing",
            "pending_handoff": False,
            "subreports": {},
            "report": None,
            "coverage": None,
            "history": [],
        }

        final_state = _run_graph_interactive(
            graph,
            initial_state,
            config,
            ["2010-2024", "add a third sub-topic about longevity", "approved"],
        )

    assert final_state["brief"] is not None
    brief = final_state["brief"]
    assert isinstance(brief, Brief)
    assert len(brief.subtopics) == 3
    assert len(final_state["subreports"]) == 3
    for st in brief.subtopics:
        assert st.slug in final_state["subreports"]
        report = final_state["subreports"][st.slug]
        assert isinstance(report, SubReport)
        path = output_path(out_dir, run_slug, st.slug, "report.md")
        assert path.exists()

    brief_path = output_path(out_dir, run_slug, "brief.md")
    assert brief_path.exists()
    text = brief_path.read_text(encoding="utf-8")
    assert "question:" in text
    assert f"slug: {run_slug}" in text
    assert f"thread_id: {run_slug}" in text


@pytest.mark.integration
def test_thread_id_equals_slug_and_brief_frontmatter(tmp_workspace, monkeypatch):
    """Slug is seeded by CLI/state and reflected in brief.md frontmatter."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]

    url = "https://example.com/test"
    content = "Test source content."
    source_id = hash_url(url)
    _seed_source(bib_dir, url, "Test Source", content)

    question = "Test research question"
    run_slug = slug(question)

    fake_chat = FakeChat(
        {
            "clarify": [
                '{"needs_clarification": false, "questions": []}',
                '{"approved": true}',
            ],
            "synth": [
                json.dumps(
                    [
                        {
                            "slug": "test-topic",
                            "title": "Test Topic",
                            "scope": "Test scope",
                            "guiding_questions": ["Test Q?"],
                            "seed_queries": ["test"],
                        }
                    ]
                ),
                "Reflect.",
                "Draft report.",
                _quality_gate_response(True),
            ],
            "gate": [
                _gate_response(True, source_id, "Test claim", content),
            ],
        }
    )

    config = _build_config(bib_dir, state_dir, out_dir, fake_chat)
    config["thread_id"] = run_slug

    with create_checkpointer(state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        initial_state = {
            "question": question,
            "slug": run_slug,
            "brief": None,
            "round": 0,
            "auto_round": 0,
            "mode": "user_facing",
            "pending_handoff": False,
            "subreports": {},
            "report": None,
            "coverage": None,
            "history": [],
        }

        final_state = _run_graph_interactive(graph, initial_state, config, ["approved"])

    assert final_state["brief"] is not None
    brief_path = output_path(out_dir, run_slug, "brief.md")
    assert brief_path.exists()
    text = brief_path.read_text(encoding="utf-8")
    assert f"slug: {run_slug}" in text
    assert f"thread_id: {run_slug}" in text
    assert config["thread_id"] == run_slug
    assert final_state["slug"] == run_slug


@pytest.mark.integration
def test_fanout_cardinality(tmp_workspace, monkeypatch):
    """Fan-out produces one subreport per sub-topic."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]

    source_ids = []
    for url, content in [
        ("https://example.com/fasting-mechanisms", "IF affects metabolism via ketosis."),
        ("https://example.com/cv-outcomes", "IF reduces cardiovascular risk."),
        ("https://example.com/fasting-longevity", "IF extends lifespan in animal models."),
    ]:
        sid = hash_url(url)
        _seed_source(bib_dir, url, "Source", content)
        source_ids.append(sid)

    def retrieve_fn(query, super_slug):
        for url, sid in [
            ("fasting-mechanisms", source_ids[0]),
            ("cv-outcomes", source_ids[1]),
            ("longevity", source_ids[2]),
        ]:
            if url in query.lower():
                return [sid]
        return source_ids[:1]

    fake_chat = FakeChat(
        {
            "clarify": [
                '{"needs_clarification": true, "questions": ["What time range?"]}',
                "Refined: effects of IF on CV health 2010-2024",
                '{"approved": true}',
            ],
            "synth": [
                json.dumps(
                    [
                        {
                            "slug": "fasting-mechanisms",
                            "title": "Fasting Mechanisms",
                            "scope": "Biological mechanisms of IF",
                            "guiding_questions": ["How does IF affect metabolism?"],
                            "seed_queries": ["IF metabolism"],
                        },
                        {
                            "slug": "cv-outcomes",
                            "title": "CV Outcomes",
                            "scope": "Cardiovascular outcomes of IF",
                            "guiding_questions": ["Does IF reduce CVD risk?"],
                            "seed_queries": ["IF CVD"],
                        },
                        {
                            "slug": "fasting-longevity",
                            "title": "Longevity",
                            "scope": "IF and longevity",
                            "guiding_questions": ["Does IF extend lifespan?"],
                            "seed_queries": ["IF longevity"],
                        },
                    ]
                ),
                "Reflect mechanisms.",
                "Draft mechanisms report.",
                _quality_gate_response(True),
                "Reflect outcomes.",
                "Draft outcomes report.",
                _quality_gate_response(True),
                "Reflect longevity.",
                "Draft longevity report.",
                _quality_gate_response(True),
            ],
            "gate": [
                _gate_response(
                    True,
                    source_ids[0],
                    "IF affects metabolism",
                    "IF affects metabolism via ketosis.",
                ),
                _gate_response(
                    True, source_ids[1], "IF reduces CVD risk", "IF reduces cardiovascular risk."
                ),
                _gate_response(
                    True,
                    source_ids[2],
                    "IF extends lifespan",
                    "IF extends lifespan in animal models.",
                ),
            ],
        }
    )

    config = _build_config(bib_dir, state_dir, out_dir, fake_chat)
    config["configurable"]["retrieve_fn"] = retrieve_fn
    run_slug = config["thread_id"]

    with create_checkpointer(state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        initial_state = {
            "question": "Effects of intermittent fasting on cardiovascular health",
            "slug": run_slug,
            "brief": None,
            "round": 0,
            "auto_round": 0,
            "mode": "user_facing",
            "pending_handoff": False,
            "subreports": {},
            "report": None,
            "coverage": None,
            "history": [],
        }

        final_state = _run_graph_interactive(
            graph, initial_state, config, ["2010-2024", "approved"]
        )

    assert len(final_state["subreports"]) == 3
    for st in final_state["brief"].subtopics:
        path = output_path(out_dir, run_slug, st.slug, "report.md")
        assert path.exists()


@pytest.mark.unit
def test_supervisor_mode_selection():
    """Supervisor dispatches all sub-topics in user_facing mode, dirty-only in autonomous."""
    subtopics = [
        SubTopic(slug="a", title="A", scope="scope a", dirty=True),
        SubTopic(slug="b", title="B", scope="scope b", dirty=False),
        SubTopic(slug="c", title="C", scope="scope c", dirty=False),
    ]
    brief = Brief(question="Q", slug="s", thread_id="s", subtopics=subtopics)

    state = {
        "question": "Q",
        "slug": "s",
        "brief": brief,
        "round": 0,
        "auto_round": 0,
        "mode": "user_facing",
        "pending_handoff": False,
        "subreports": {},
        "report": None,
        "coverage": None,
        "history": [],
    }
    sends = supervisor(state)
    assert len(sends) == 3

    state["mode"] = "autonomous"
    sends = supervisor(state)
    assert len(sends) == 1
    assert sends[0].arg["subtopic"].slug == "a"

    subtopics[0].dirty = False
    sends = supervisor(state)
    assert len(sends) == 0


@pytest.mark.unit
def test_supervisor_skips_isolated_subtopics():
    """Supervisor never re-dispatches subtopics whose subreport has an isolation shortfall."""
    subtopics = [
        SubTopic(slug="broken", title="Broken", scope="broken", dirty=True),
        SubTopic(slug="good", title="Good", scope="good", dirty=True),
    ]
    brief = Brief(question="Q", slug="s", thread_id="s", subtopics=subtopics)
    isolated_report = SubReport(
        subtopic_slug="broken",
        body="",
        citations=[],
        shortfall="subagent isolated: RuntimeError: boom",
    )
    state = {
        "question": "Q",
        "slug": "s",
        "brief": brief,
        "round": 0,
        "auto_round": 0,
        "mode": "user_facing",
        "pending_handoff": False,
        "subreports": {"broken": isolated_report},
        "report": None,
        "coverage": None,
        "history": [],
    }

    sends = supervisor(state)

    assert len(sends) == 1
    assert sends[0].arg["subtopic"].slug == "good"


@pytest.mark.integration
def test_cross_process_resume(tmp_workspace, monkeypatch):
    """Resume works after discarding the graph/checkpointer and rebuilding from disk."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]

    url = "https://example.com/test"
    content = "Test source content."
    source_id = hash_url(url)
    _seed_source(bib_dir, url, "Test Source", content)

    run_slug = "cross-process-test-slug"

    # First graph: only enough chat to reach clarify interrupt.
    first_chat = FakeChat(
        {
            "clarify": [
                '{"needs_clarification": true, "questions": ["What time range?"]}',
            ]
        }
    )
    embeddings = FakeEmbeddings()
    first_store = ChromaStore(state_dir, embeddings.embed_query)
    config = {
        "configurable": {
            "chat_fn": first_chat.chat,
            "embeddings": embeddings,
            "store": first_store,
            "bibliography_dir": str(bib_dir),
            "state_dir": str(state_dir),
            "output_dir": str(out_dir),
        },
        "thread_id": run_slug,
    }

    with create_checkpointer(state_dir) as checkpointer1:
        graph1 = build_graph(checkpointer=checkpointer1)
        initial_state = {
            "question": "Effects of intermittent fasting on cardiovascular health",
            "slug": run_slug,
            "brief": None,
            "round": 0,
            "auto_round": 0,
            "mode": "user_facing",
            "pending_handoff": False,
            "subreports": {},
            "report": None,
            "coverage": None,
            "history": [],
        }

        result = graph1.invoke(initial_state, config)
        assert "__interrupt__" in result
        assert len(result["__interrupt__"]) > 0

    # Discard graph and checkpointer; rebuild from same state_dir.
    second_chat = FakeChat(
        {
            "clarify": [
                "Refined: effects of IF on CV health 2010-2024",
                '{"approved": true}',
            ],
            "synth": [
                json.dumps(
                    [
                        {
                            "slug": "test-topic",
                            "title": "Test Topic",
                            "scope": "Test scope",
                            "guiding_questions": ["Test Q?"],
                            "seed_queries": ["test"],
                        }
                    ]
                ),
                "Reflect.",
                "Draft report.",
                _quality_gate_response(True),
            ],
            "gate": [
                _gate_response(True, source_id, "Test claim", content),
            ],
        }
    )
    second_embeddings = FakeEmbeddings()
    second_store = ChromaStore(state_dir, second_embeddings.embed_query)
    config2 = {
        "configurable": {
            "chat_fn": second_chat.chat,
            "embeddings": second_embeddings,
            "store": second_store,
            "bibliography_dir": str(bib_dir),
            "state_dir": str(state_dir),
            "output_dir": str(out_dir),
            "retrieve_fn": lambda _q, _s: [source_id],
        },
        "thread_id": run_slug,
    }

    with create_checkpointer(state_dir) as checkpointer2:
        graph2 = build_graph(checkpointer=checkpointer2)
        result = graph2.invoke(Command(resume="2010-2024"), config2)
        # With split nodes, the approve interrupt fires after decompose.
        # Resume again with "approved" to complete the brief.
        if result.get("__interrupt__"):
            final_state = graph2.invoke(Command(resume="approved"), config2)
        else:
            final_state = result

    assert final_state["brief"] is not None
    brief_path = output_path(out_dir, run_slug, "brief.md")
    assert brief_path.exists()


@pytest.mark.integration
def test_reconcile_runs_before_fanout(tmp_workspace, monkeypatch):
    """_do_sync reconciles an inbox PDF before the graph fan-out uses the pool."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]

    # Drop a dummy PDF into the inbox.
    pdf_bytes = b"%PDF-1.4 dummy pdf bytes"
    pdf_id = hash_url("https://example.com/inbox-pdf")
    inbox_file = bib_dir / "_inbox" / f"{pdf_id}.pdf"
    inbox_file.write_bytes(pdf_bytes)

    # Use FakePdf to convert the inbox PDF to a markdown source.
    from fakes.pdf import FakePdf

    fake_pdf = FakePdf(
        pdfs={str(inbox_file): pdf_bytes},
        conversions={str(inbox_file): "Inbox PDF converted content."},
    )

    from deepresearch.config import get_config

    cfg = get_config()
    # Inject a fake PDF converter by going through index.reconcile, which passes
    # pdf_converter to sources.inbox.reconcile. We call _do_sync with a fake
    # store/embeddings so it doesn't need Ollama.
    fake_embeddings = FakeEmbeddings()
    fake_store = ChromaStore(state_dir, fake_embeddings.embed_query)
    from deepresearch.rag.index import reconcile

    reconcile(cfg.bibliography_dir, fake_store, fake_embeddings, pdf_converter=fake_pdf)

    assert not inbox_file.exists()
    pool_md = bib_dir / "_sources" / f"{pdf_id}.md"
    assert pool_md.exists()

    # Now run a simple graph that should find the reconciled source.
    fake_chat = FakeChat(
        {
            "clarify": [
                '{"needs_clarification": false, "questions": []}',
                '{"approved": true}',
            ],
            "synth": [
                json.dumps(
                    [
                        {
                            "slug": "test-topic",
                            "title": "Test Topic",
                            "scope": "Test scope",
                            "guiding_questions": ["Test Q?"],
                            "seed_queries": ["test"],
                        }
                    ]
                ),
                "Reflect.",
                "Draft report.",
                _quality_gate_response(True),
            ],
            "gate": [
                _gate_response(True, pdf_id, "Inbox claim", "Inbox PDF converted content."),
            ],
        }
    )

    run_slug = slug("test reconcile question")
    config = _build_config(bib_dir, state_dir, out_dir, fake_chat)
    config["configurable"]["retrieve_fn"] = lambda _q, _s: [pdf_id]
    config["thread_id"] = run_slug

    with create_checkpointer(state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        initial_state = {
            "question": "test reconcile question",
            "slug": run_slug,
            "brief": None,
            "round": 0,
            "auto_round": 0,
            "mode": "user_facing",
            "pending_handoff": False,
            "subreports": {},
            "report": None,
            "coverage": None,
            "history": [],
        }

        final_state = _run_graph_interactive(graph, initial_state, config, ["approved"])

    assert final_state["brief"] is not None
    assert len(final_state["subreports"]) == 1
    report = final_state["subreports"]["test-topic"]
    assert any(c.source_id == pdf_id for c in report.citations)
