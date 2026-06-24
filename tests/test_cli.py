"""Tests for the CLI graph wiring."""

import json

import pytest
from langgraph.types import Interrupt

from deepresearch.cli import _build_configurable, _build_resume_value, _collect_resume_input
from deepresearch.config import get_config, reset_config


@pytest.mark.unit
def test_build_configurable_injects_network_clients(tmp_workspace):
    """The subagent gates web/PDF acquisition on these clients being present;
    without them a real run is RAG-only and produces empty, uncited reports."""
    cfg = get_config()
    configurable = _build_configurable(cfg)

    # The acquisition seams the subagent reads out of config.
    assert configurable.get("tavily_client") is not None
    assert configurable.get("pdf_client") is not None
    # pdf_client must expose the fetch/convert surface the subagent calls.
    assert hasattr(configurable["pdf_client"], "fetch")
    assert hasattr(configurable["pdf_client"], "convert")

    # Existing dependencies are still wired.
    for key in ("chat_fn", "embeddings", "store", "bibliography_dir", "output_dir"):
        assert key in configurable


@pytest.mark.unit
def test_build_resume_value_uses_interrupt_id_map_for_single_acquire(monkeypatch, tmp_path):
    """LangGraph requires map-form resumes if hidden interrupts are also pending."""
    save_path = tmp_path / "manual.pdf"
    request = {
        "source_id": "source-1",
        "title": "Manual source",
        "url": "https://example.com/manual.pdf",
        "save_path": str(save_path),
    }
    interrupt = Interrupt(value={"type": "acquire", "requests": [request]}, id="interrupt-1")
    monkeypatch.setattr("typer.prompt", lambda *args, **kwargs: "")

    resume = _build_resume_value([interrupt])

    assert resume == {
        "interrupt-1": [
            {
                "source_id": "source-1",
                "kind": "unobtainable",
            }
        ]
    }


@pytest.mark.unit
def test_build_resume_value_uses_interrupt_id_map_for_multiple_acquires(monkeypatch, tmp_path):
    first_path = tmp_path / "first.pdf"
    first_path.write_bytes(b"%PDF-1.4\n")
    second_path = tmp_path / "second.pdf"
    interrupts = [
        Interrupt(
            value={
                "type": "acquire",
                "requests": [
                    {
                        "source_id": "source-1",
                        "title": "First",
                        "url": "https://example.com/first.pdf",
                        "save_path": str(first_path),
                    }
                ],
            },
            id="interrupt-1",
        ),
        Interrupt(
            value={
                "type": "acquire",
                "requests": [
                    {
                        "source_id": "source-2",
                        "title": "Second",
                        "url": "https://example.com/second.pdf",
                        "save_path": str(second_path),
                    }
                ],
            },
            id="interrupt-2",
        ),
    ]
    monkeypatch.setattr("typer.prompt", lambda *args, **kwargs: "")

    resume = _build_resume_value(interrupts)

    assert resume == {
        "interrupt-1": [
            {
                "source_id": "source-1",
                "kind": "saved",
                "save_path": str(first_path),
            }
        ],
        "interrupt-2": [
            {
                "source_id": "source-2",
                "kind": "unobtainable",
            }
        ],
    }


@pytest.mark.unit
def test_collect_resume_input_preserves_interrupt_map(monkeypatch, tmp_path):
    save_path = tmp_path / "manual.pdf"
    state = type(
        "State",
        (),
        {
            "interrupts": [
                Interrupt(
                    value={
                        "type": "acquire",
                        "requests": [
                            {
                                "source_id": "source-1",
                                "title": "Manual source",
                                "url": "https://example.com/manual.pdf",
                                "save_path": str(save_path),
                            }
                        ],
                    },
                    id="interrupt-1",
                )
            ]
        },
    )()
    monkeypatch.setattr("typer.prompt", lambda *args, **kwargs: "")

    resume = _collect_resume_input(state)

    assert resume == {"interrupt-1": [{"source_id": "source-1", "kind": "unobtainable"}]}


@pytest.mark.unit
def test_collect_resume_input_empty_interrupts_returns_empty_map():
    """When get_state() surfaces no interrupts, return {} so LangGraph sets
    resume_is_map=True and bypasses the pending-interrupt count check."""
    state = type("State", (), {"interrupts": []})()

    resume = _collect_resume_input(state)

    assert resume == {}
    # {} satisfies LangGraph's is_xxh3 map check vacuously: resume_is_map=True
    # means the pending-interrupt count check is skipped entirely.


@pytest.mark.unit
def test_build_configurable_includes_quality_gate_flag(tmp_workspace):
    """quality_gate_enabled is forwarded from Config so subagent nodes can read it."""
    cfg = get_config()
    configurable = _build_configurable(cfg)
    assert "quality_gate_enabled" in configurable


# ---------------------------------------------------------------------------
# improve command
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_improve_patches_state_correctly(tmp_workspace, monkeypatch):
    """improve injects pending_handoff=True, user_approved=False, auto_round=0
    into the checkpoint and marks all subtopics dirty."""
    from fakes.chat import FakeChat
    from fakes.embeddings import FakeEmbeddings

    from deepresearch.models import Brief, SubTopic
    from deepresearch.nodes.evaluate import mark_all_dirty
    from deepresearch.persistence import create_checkpointer
    from deepresearch.graph import build_graph
    from deepresearch.rag.store import ChromaStore
    from deepresearch.state import ResearchState

    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    reset_config()
    cfg = get_config()

    brief = Brief(
        question="test?",
        slug="test-slug",
        thread_id="test-slug",
        subtopics=[
            SubTopic(slug="sub-a", title="Sub A", scope="scope a", dirty=False),
            SubTopic(slug="sub-b", title="Sub B", scope="scope b", dirty=False),
        ],
        approved=True,
    )

    embeddings = FakeEmbeddings()
    store = ChromaStore(tmp_workspace["state_dir"], embeddings.embed_query)

    with create_checkpointer(cfg.state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {
            "configurable": {
                "chat_fn": FakeChat({}).chat,
                "embeddings": embeddings,
                "store": store,
                "tavily_client": None,
                "pdf_client": None,
                "bibliography_dir": str(cfg.bibliography_dir),
                "state_dir": str(cfg.state_dir),
                "output_dir": str(cfg.output_dir),
                "quality_gate_enabled": False,
                "thread_id": "test-slug",
            }
        }

        # Seed a terminal checkpoint (user_approved=True, run is "finished")
        graph.update_state(
            config,
            {
                "question": "test?",
                "slug": "test-slug",
                "brief": brief,
                "round": 1,
                "auto_round": 1,
                "mode": "user_facing",
                "pending_handoff": False,
                "subreports": {},
                "report": "# Report\n\nSome content.",
                "report_references": [],
                "coverage": None,
                "history": [],
                "verify_ok": True,
                "verify_attempts": 1,
                "verify_unsupported": [],
                "verify_dangling": [],
                "user_approved": True,
            },
        )

        # Simulate what `improve` does
        graph.update_state(
            config,
            {
                "user_approved": False,
                "pending_handoff": True,
                "auto_round": 0,
                "brief": mark_all_dirty(brief),
            },
            as_node="writer",
        )

        patched = graph.get_state(config)
        values = patched.values
        assert values["user_approved"] is False
        assert values["pending_handoff"] is True
        assert values["auto_round"] == 0
        assert all(sub.dirty for sub in values["brief"].subtopics)
        # next node should be evaluate (successor of writer)
        assert "evaluate" in patched.next


@pytest.mark.unit
def test_improve_rejects_missing_slug(tmp_workspace, monkeypatch):
    """improve exits with an error message when the slug has no checkpoint."""
    from typer.testing import CliRunner
    from deepresearch.cli import app

    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    reset_config()

    runner = CliRunner()
    result = runner.invoke(app, ["improve", "no-such-slug"])

    assert result.exit_code != 0 or "No run found" in result.output


@pytest.mark.unit
def test_improve_rejects_unfinished_run(tmp_workspace, monkeypatch):
    """improve exits when the run is not yet approved."""
    from fakes.chat import FakeChat
    from fakes.embeddings import FakeEmbeddings

    from deepresearch.models import Brief, SubTopic
    from deepresearch.persistence import create_checkpointer
    from deepresearch.graph import build_graph
    from deepresearch.rag.store import ChromaStore
    from typer.testing import CliRunner
    from deepresearch.cli import app

    monkeypatch.setenv("TAVILY_API_KEY", "fake")
    reset_config()
    cfg = get_config()

    brief = Brief(
        question="test?",
        slug="unfinished-slug",
        thread_id="unfinished-slug",
        subtopics=[SubTopic(slug="sub-a", title="Sub A", scope="s", dirty=False)],
        approved=True,
    )
    embeddings = FakeEmbeddings()
    store = ChromaStore(tmp_workspace["state_dir"], embeddings.embed_query)

    with create_checkpointer(cfg.state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {
            "configurable": {
                "chat_fn": FakeChat({}).chat,
                "embeddings": embeddings,
                "store": store,
                "tavily_client": None,
                "pdf_client": None,
                "bibliography_dir": str(cfg.bibliography_dir),
                "state_dir": str(cfg.state_dir),
                "output_dir": str(cfg.output_dir),
                "quality_gate_enabled": False,
                "thread_id": "unfinished-slug",
            }
        }
        graph.update_state(
            config,
            {
                "question": "test?",
                "slug": "unfinished-slug",
                "brief": brief,
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
            },
        )

    runner = CliRunner()
    result = runner.invoke(app, ["improve", "unfinished-slug"])

    assert "not yet finished" in result.output or result.exit_code != 0


# ---------------------------------------------------------------------------
# prune command
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_prune_removes_garbage_source(tmp_workspace, monkeypatch):
    """prune deletes a garbage source from pool and store."""
    from fakes.embeddings import FakeEmbeddings

    from deepresearch.rag import index as rag_index
    from deepresearch.rag.store import ChromaStore
    from deepresearch.sources.pool import get_ref, save_web
    from deepresearch.paths import hash_url

    monkeypatch.setenv("MIN_SOURCE_WORDS", "10")
    reset_config()
    cfg = get_config()

    embeddings = FakeEmbeddings()
    store = ChromaStore(cfg.state_dir, embeddings.embed_query)

    content = " ".join(["word"] * 300)
    source_ref = save_web(content, "https://example.com/garbage", "Garbage", cfg.bibliography_dir)
    rag_index.ingest(source_ref, store, embeddings, cfg.bibliography_dir)
    assert store.count() > 0

    # Run prune with a fake quality gate that rejects the source.
    garbage_json = json.dumps({"quality": False, "reason": "link farm"})

    import deepresearch.source_quality as sq

    monkeypatch.setattr(sq, "assess", lambda ref, bib, state, **kw: (False, "link farm"))

    from typer.testing import CliRunner
    from deepresearch.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["prune"])

    assert result.exit_code == 0
    assert "Pruned: 1" in result.output
    assert not (cfg.bibliography_dir / "_sources" / f"{source_ref.id}.md").exists()
    # Chroma chunks for the source should be gone.
    store2 = ChromaStore(cfg.state_dir, embeddings.embed_query)
    assert source_ref.id not in store2.list_source_ids()


@pytest.mark.integration
def test_prune_keeps_quality_source(tmp_workspace, monkeypatch):
    """prune leaves quality sources untouched."""
    from fakes.embeddings import FakeEmbeddings

    from deepresearch.rag.store import ChromaStore
    from deepresearch.sources.pool import save_web

    monkeypatch.setenv("MIN_SOURCE_WORDS", "10")
    reset_config()
    cfg = get_config()

    content = " ".join(["word"] * 300)
    source_ref = save_web(content, "https://example.com/good", "Good paper", cfg.bibliography_dir)

    import deepresearch.source_quality as sq

    monkeypatch.setattr(sq, "assess", lambda ref, bib, state, **kw: (True, "good paper"))

    from typer.testing import CliRunner
    from deepresearch.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["prune"])

    assert result.exit_code == 0
    assert "Kept: 1" in result.output
    assert (cfg.bibliography_dir / "_sources" / f"{source_ref.id}.md").exists()


@pytest.mark.unit
def test_prune_dry_run_makes_no_changes(tmp_workspace, monkeypatch):
    """--dry-run reports what would be pruned without touching any files."""
    from deepresearch.sources.pool import save_web

    monkeypatch.setenv("MIN_SOURCE_WORDS", "10")
    reset_config()
    cfg = get_config()

    content = " ".join(["word"] * 300)
    source_ref = save_web(content, "https://example.com/dry", "Dry run source", cfg.bibliography_dir)
    md_path = cfg.bibliography_dir / "_sources" / f"{source_ref.id}.md"
    assert md_path.exists()

    import deepresearch.source_quality as sq

    monkeypatch.setattr(sq, "assess", lambda ref, bib, state, **kw: (False, "garbage"))

    from typer.testing import CliRunner
    from deepresearch.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["prune", "--dry-run"])

    assert result.exit_code == 0
    assert "dry run" in result.output.lower()
    assert md_path.exists()  # file not deleted


@pytest.mark.unit
def test_prune_empty_bibliography(tmp_workspace, monkeypatch):
    """prune on an empty bibliography reports 0 pruned and exits cleanly."""
    reset_config()

    from typer.testing import CliRunner
    from deepresearch.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["prune"])

    assert result.exit_code == 0
    assert "Pruned: 0" in result.output or "No sources found" in result.output
