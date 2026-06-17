"""Tests for Phase 6 writer, citation merge, and verification."""

import json
import re
from pathlib import Path

import pytest
from fakes.chat import FakeChat
from fakes.embeddings import FakeEmbeddings
from langgraph.types import Command

from deepresearch import citations, verify
from deepresearch.config import reset_config
from deepresearch.graph import build_graph
from deepresearch.models import Citation, SourceRef, SubReport
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


def _build_config(
    bib_dir: Path,
    state_dir: Path,
    out_dir: Path,
    fake_chat: FakeChat,
    retrieve_fn=None,
    tavily_client=None,
    pdf_client=None,
    store=None,
):
    embeddings = FakeEmbeddings()
    if store is None:
        store = ChromaStore(state_dir, embeddings.embed_query)
    configurable = {
        "chat_fn": fake_chat.chat,
        "embeddings": embeddings,
        "store": store,
        "bibliography_dir": str(bib_dir),
        "state_dir": str(state_dir),
        "output_dir": str(out_dir),
    }
    if retrieve_fn is not None:
        configurable["retrieve_fn"] = retrieve_fn
    if tavily_client is not None:
        configurable["tavily_client"] = tavily_client
    if pdf_client is not None:
        configurable["pdf_client"] = pdf_client
    return {"configurable": configurable, "thread_id": "phase-six-test-slug"}


def _run_graph_interactive(graph, initial_state, config, resume_sequence):
    resume_iter = iter(resume_sequence)
    current_input = initial_state
    while True:
        result = graph.invoke(current_input, config)
        if not result.get("__interrupt__", []):
            return result
        try:
            next_resume = next(resume_iter)
        except StopIteration as exc:
            raise RuntimeError("Graph interrupted but no more resume values") from exc
        current_input = Command(resume=next_resume)


def _initial_state(question: str, run_slug: str) -> dict:
    return {
        "question": question,
        "slug": run_slug,
        "brief": None,
        "round": 0,
        "auto_round": 0,
        "mode": "user_facing",
        "pending_handoff": False,
        "subreports": {},
        "report": None,
        "report_references": [],
        "verify_ok": False,
        "verify_attempts": 0,
        "verify_unsupported": [],
        "verify_dangling": [],
        "coverage": None,
        "history": [],
    }


@pytest.mark.unit
def test_citations_merge_collapses_by_source_id(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    ref_x = _seed_source(bib_dir, "https://example.com/x", "X", "X content")
    ref_y = _seed_source(bib_dir, "https://example.com/y", "Y", "Y content")
    sub_a = SubReport(
        subtopic_slug="sub-a",
        body=f"X claim [{ref_x.id}].",
        citations=[Citation(source_id=ref_x.id, claim="X claim")],
    )
    sub_b = SubReport(
        subtopic_slug="sub-b",
        body=f"More X [{ref_x.id}]. Y claim [{ref_y.id}].",
        citations=[
            Citation(source_id=ref_x.id, claim="More X"),
            Citation(source_id=ref_y.id, claim="Y claim"),
        ],
    )

    body, references = citations.merge([sub_a, sub_b], bib_dir, question="Q")

    assert len(references) == 2
    assert references[0].id == ref_x.id
    assert references[1].id == ref_y.id
    assert body.count("[1]") == 2
    assert "[2]" in body


@pytest.mark.unit
def test_citations_merge_preserves_subreport_order(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    ref = _seed_source(bib_dir, "https://example.com/source", "Source", "content")
    reports = [
        SubReport(
            subtopic_slug=slug,
            body=f"Body [{ref.id}].",
            citations=[Citation(source_id=ref.id, claim="Body")],
        )
        for slug in ["first-topic", "second-topic", "third-topic"]
    ]

    body, _ = citations.merge(reports, bib_dir)

    assert (
        body.index("## First Topic") < body.index("## Second Topic") < body.index("## Third Topic")
    )


@pytest.mark.unit
def test_verify_resolution_dangling_fails():
    ref = SourceRef(
        id=hash_url("https://example.com/one"),
        type="web",
        url="https://example.com/one",
        title="One",
        source_path="_sources/one.md",
        retrieved_at="now",
        content_hash="hash",
    )
    result = verify.resolve("blah [1] [3] [99]", [ref])

    assert result.ok is False
    assert result.dangling == [3, 99]


@pytest.mark.unit
def test_verify_groundedness_against_full_doc(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    ref = _seed_source(
        bib_dir,
        "https://example.com/france",
        "France",
        "The capital of France is Paris.",
    )
    body = "The capital of France is London [1]."
    fake_chat = FakeChat({"writer": [json.dumps({"supported": False, "reason": "claim wrong"})]})

    verdicts = verify.ground_claims(body, [ref], bib_dir, chat_fn=fake_chat.chat)

    assert len(verdicts) == 1
    assert verdicts[0].supported is False

    fake_chat = FakeChat(
        {
            "writer": [
                json.dumps({"supported": False, "reason": "claim wrong"}),
                "",
                json.dumps({"supported": False, "reason": "claim wrong"}),
            ]
        }
    )
    result = verify.check(body, [ref], bib_dir, chat_fn=fake_chat.chat, max_revisions=1)

    assert result.ok is False
    assert result.attempts <= 1


@pytest.mark.unit
def test_verify_revise_drops_unsupported_claim(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    ref = _seed_source(
        bib_dir,
        "https://example.com/france",
        "France",
        "The capital of France is Paris.",
    )
    body = "The capital of France is London [1]."
    fake_chat = FakeChat(
        {
            "writer": [
                json.dumps({"supported": False, "reason": "claim wrong"}),
                "DROP",
            ]
        }
    )

    result = verify.check(body, [ref], bib_dir, chat_fn=fake_chat.chat, max_revisions=1)

    assert "London" not in result.body
    assert result.ok is True


def _run_happy_graph(tmp_workspace):
    reset_config()
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    ref_a = _seed_source(bib_dir, "https://example.com/a", "A", "Alpha is supported.")
    ref_b = _seed_source(bib_dir, "https://example.com/b", "B", "Beta is supported.")

    def retrieve_fn(query, super_slug):
        del super_slug
        if "alpha" in query.lower():
            return [ref_a.id]
        if "beta" in query.lower():
            return [ref_b.id]
        return []

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
                            "slug": "alpha-topic",
                            "title": "Alpha Topic",
                            "scope": "Alpha scope",
                            "guiding_questions": ["What about alpha?"],
                            "seed_queries": ["alpha"],
                        },
                        {
                            "slug": "beta-topic",
                            "title": "Beta Topic",
                            "scope": "Beta scope",
                            "guiding_questions": ["What about beta?"],
                            "seed_queries": ["beta"],
                        },
                    ]
                ),
                "Reflect alpha.",
                f"Alpha is supported [{ref_a.id}].",
                _quality_gate_response(True),
                "Reflect beta.",
                f"Beta is supported [{ref_b.id}].",
                _quality_gate_response(True),
            ],
            "gate": [
                _gate_response(True, ref_a.id, "Alpha is supported", "Alpha is supported."),
                _gate_response(True, ref_b.id, "Beta is supported", "Beta is supported."),
            ],
            "writer": [
                json.dumps({"supported": True, "reason": "ok"}),
                json.dumps({"supported": True, "reason": "ok"}),
            ],
        }
    )
    config = _build_config(
        bib_dir, state_dir, out_dir, fake_chat, retrieve_fn=retrieve_fn, store={}
    )
    run_slug = slug("Phase six happy path")
    config["thread_id"] = run_slug

    with create_checkpointer(state_dir) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        final_state = _run_graph_interactive(
            graph,
            _initial_state("Phase six happy path", run_slug),
            config,
            ["approved"],
        )
    return final_state, out_dir, bib_dir, run_slug


@pytest.mark.integration
def test_writer_persists_report_and_references(tmp_workspace):
    final_state, out_dir, _, run_slug = _run_happy_graph(tmp_workspace)
    del final_state
    report_path = output_path(out_dir, run_slug, "report.md")
    references_path = output_path(out_dir, run_slug, "references.json")

    assert report_path.exists()
    body = report_path.read_text(encoding="utf-8")
    assert "[1]" in body and "[2]" in body
    payload = json.loads(references_path.read_text(encoding="utf-8"))
    refs = [source["source_id"] for source in payload["sources"]]
    assert len(refs) == 2
    assert len(set(refs)) == 2
    assert [source["index"] for source in payload["sources"]] == [1, 2]


@pytest.mark.integration
def test_writer_references_match_cited_only(tmp_workspace):
    _, out_dir, _, run_slug = _run_happy_graph(tmp_workspace)
    body = output_path(out_dir, run_slug, "report.md").read_text(encoding="utf-8")
    payload = json.loads(
        output_path(out_dir, run_slug, "references.json").read_text(encoding="utf-8")
    )
    cited_numbers = {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", body)}
    cited_source_ids = {payload["sources"][n - 1]["source_id"] for n in cited_numbers}
    reference_source_ids = {source["source_id"] for source in payload["sources"]}

    assert reference_source_ids == cited_source_ids


@pytest.mark.integration
def test_writer_full_pipeline_end_to_end(tmp_workspace):
    final_state, out_dir, bib_dir, run_slug = _run_happy_graph(tmp_workspace)

    assert final_state["verify_ok"] is True
    assert final_state["report"]
    assert len(final_state["report_references"]) >= 1
    assert output_path(out_dir, run_slug, "alpha-topic", "report.md").exists()
    assert output_path(out_dir, run_slug, "beta-topic", "report.md").exists()
    assert not list(bib_dir.rglob("report.md"))


@pytest.mark.integration
def test_bibliography_holds_no_reports(tmp_workspace):
    _, _, bib_dir, _ = _run_happy_graph(tmp_workspace)

    assert not list((bib_dir / "_sources").rglob("report.md"))
