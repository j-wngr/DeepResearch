"""Phase 8 acquisition UX tests."""

import json
from pathlib import Path

import pytest
from fakes.chat import FakeChat
from fakes.embeddings import FakeEmbeddings
from fakes.pdf import FakePdf
from fakes.tavily import FakeTavily
from langgraph.types import Command
from pydantic import ValidationError

from deepresearch import acquisition
from deepresearch.config import reset_config
from deepresearch.models import AcquisitionResponse, SearchHit, SubTopic
from deepresearch.nodes.subagent import build_subagent_subgraph
from deepresearch.paths import hash_url, output_path
from deepresearch.persistence import create_checkpointer
from deepresearch.rag.index import reconcile
from deepresearch.rag.store import ChromaStore
from deepresearch.sources.pool import save_pdf


def _quality(passed: bool, reason: str = "needs source") -> str:
    return json.dumps(
        {"passed": passed, "coverage_score": 1.0 if passed else 0.2, "reason": reason}
    )


def _gate(quote: str) -> str:
    return json.dumps(
        {
            "relevant": True,
            "reason": "relevant",
            "evidence": [{"claim": "Acquired source supports the answer", "quote": quote}],
        }
    )


def _subtopic(query: str = "blocked query") -> SubTopic:
    return SubTopic(
        slug="blocked-source",
        title="Blocked source",
        scope="Use a blocked source.",
        guiding_questions=[query],
        seed_queries=[],
    )


def _initial_state(subtopic: SubTopic | None = None) -> dict:
    return {
        "slug": "phase8-super",
        "subtopic": subtopic or _subtopic(),
        "iteration": 0,
        "scratchpad": "",
        "whitelisted": [],
        "evidence": [],
        "draft": None,
        "candidates": [],
        "pending_acquisitions": [],
        "acquisition_gaps": [],
        "subreport": None,
        "subreports": {},
        "quality_gate_result": None,
    }


def _config(tmp_workspace, fake_chat, fake_tavily=None, fake_pdf=None):
    embeddings = FakeEmbeddings()
    state_dir = tmp_workspace["state_dir"]
    return {
        "configurable": {
            "chat_fn": fake_chat.chat,
            "tavily_client": fake_tavily,
            "pdf_client": fake_pdf,
            "embeddings": embeddings,
            "store": ChromaStore(state_dir, embeddings.embed_query),
            "bibliography_dir": str(tmp_workspace["bibliography_dir"]),
            "state_dir": str(state_dir),
            "output_dir": str(tmp_workspace["output_dir"]),
            "retrieve_fn": lambda query, slug: [],
        },
        "thread_id": "phase8-thread",
    }


def _blocked_tavily(url: str, title: str = "Paywalled PDF") -> FakeTavily:
    return FakeTavily(
        search_results={"blocked query": [SearchHit(url=url, title=title, snippet="blocked")]}
    )


class RetryTavily(FakeTavily):
    def __init__(self, blocked_url: str, open_url: str, open_markdown: str):
        super().__init__(extracts={open_url: open_markdown})
        self.blocked_url = blocked_url
        self.open_url = open_url

    def search(self, query: str) -> list[SearchHit]:
        self._search_count += 1
        if self._search_count <= 2:
            return [SearchHit(url=self.blocked_url, title="Blocked", snippet="blocked")]
        return [SearchHit(url=self.open_url, title="Open", snippet="open")]


@pytest.mark.integration
def test_blocked_fetch_raises_acquisition_interrupt_with_save_path(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    url = "https://example.com/paywalled.pdf"
    source_id = hash_url(url)
    fake_chat = FakeChat()
    config = _config(
        tmp_workspace,
        fake_chat,
        _blocked_tavily(url),
        FakePdf(blocked_urls={url}),
    )

    with create_checkpointer(tmp_workspace["state_dir"]) as checkpointer:
        subgraph = build_subagent_subgraph(checkpointer=checkpointer)
        result = subgraph.invoke(_initial_state(), config)

    interrupt_payload = result["__interrupt__"][0].value
    request = interrupt_payload["requests"][0]
    assert interrupt_payload["type"] == "acquire"
    assert request["source_id"] == source_id
    assert request["save_path"] == str(
        tmp_workspace["bibliography_dir"] / "_inbox" / f"{source_id}.pdf"
    )
    assert request["url"] == url
    assert request["title"] == "Paywalled PDF"


@pytest.mark.integration
def test_saved_at_path_resume_ingests_and_cites(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    url = "https://example.com/paywalled.pdf"
    source_id = hash_url(url)
    save_path = tmp_workspace["bibliography_dir"] / "_inbox" / f"{source_id}.pdf"
    fake_pdf = FakePdf(
        blocked_urls={url}, conversions={str(save_path): "# Acquired\nManual PDF evidence."}
    )
    fake_chat = FakeChat(
        {
            "gate": [_gate("Manual PDF evidence.")],
            "synth": ["notes", f"Draft [{source_id}]", _quality(True)],
        }
    )
    config = _config(tmp_workspace, fake_chat, _blocked_tavily(url), fake_pdf)

    with create_checkpointer(tmp_workspace["state_dir"]) as checkpointer:
        subgraph = build_subagent_subgraph(checkpointer=checkpointer)
        first = subgraph.invoke(_initial_state(), config)
        request = first["__interrupt__"][0].value["requests"][0]
        Path(request["save_path"]).write_bytes(b"%PDF fake")
        result = subgraph.invoke(
            Command(
                resume=[
                    AcquisitionResponse(
                        source_id=source_id, kind="saved", save_path=request["save_path"]
                    ).model_dump()
                ]
            ),
            config,
        )

    assert source_id in {ref.id for ref in result["whitelisted"]}
    assert source_id in {citation.source_id for citation in result["subreport"].citations}
    assert output_path(
        tmp_workspace["output_dir"], "phase8-super", "blocked-source", "report.md"
    ).exists()
    assert not save_path.exists()
    assert (tmp_workspace["bibliography_dir"] / "_sources" / "pdfs" / f"{source_id}.pdf").exists()
    assert (tmp_workspace["bibliography_dir"] / "_sources" / f"{source_id}.md").exists()


@pytest.mark.integration
def test_unobtainable_creates_permanent_gap_no_whitelist(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    url = "https://example.com/paywalled.pdf"
    source_id = hash_url(url)
    fake_chat = FakeChat({"synth": ["notes", "Draft without citation", _quality(False, "missing")]})
    config = _config(tmp_workspace, fake_chat, _blocked_tavily(url), FakePdf(blocked_urls={url}))

    with create_checkpointer(tmp_workspace["state_dir"]) as checkpointer:
        subgraph = build_subagent_subgraph(checkpointer=checkpointer)
        subgraph.invoke(_initial_state(), config)
        result = subgraph.invoke(
            Command(
                resume=[AcquisitionResponse(source_id=source_id, kind="unobtainable").model_dump()]
            ),
            config,
        )

    assert "source unobtainable" in result["subreport"].shortfall
    assert "Could not obtain:" in result["subreport"].body
    assert source_id not in {citation.source_id for citation in result["subreport"].citations}
    assert result["acquisition_gaps"]


@pytest.mark.integration
def test_unobtainable_gap_does_not_prevent_retry_with_open_source(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    monkeypatch.setenv("MIN_SOURCE_WORDS", "1")
    reset_config()
    blocked_url = "https://example.com/paywalled.pdf"
    open_url = "https://example.com/open"
    open_id = hash_url(open_url)
    blocked_id = hash_url(blocked_url)
    fake_tavily = RetryTavily(blocked_url, open_url, "# Open\n\nRetry evidence.")
    fake_pdf = FakePdf(blocked_urls={blocked_url})
    fake_chat = FakeChat(
        {
            "gate": [_gate("Retry evidence.")],
            "synth": [
                "notes before retry",
                "Draft without citation",
                _quality(False, "missing"),
                "notes after retry",
                f"Retry answer [{open_id}]",
                _quality(True),
            ],
        }
    )
    config = _config(tmp_workspace, fake_chat, fake_tavily, fake_pdf)

    with create_checkpointer(tmp_workspace["state_dir"]) as checkpointer:
        subgraph = build_subagent_subgraph(checkpointer=checkpointer)
        subgraph.invoke(_initial_state(), config)
        result = subgraph.invoke(
            Command(
                resume=[AcquisitionResponse(source_id=blocked_id, kind="unobtainable").model_dump()]
            ),
            config,
        )

    assert open_id in {ref.id for ref in result["whitelisted"]}
    assert result["subreport"].shortfall is None
    assert result["acquisition_gaps"]


@pytest.mark.integration
def test_alternative_url_fetches_and_ingests(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    url = "https://example.com/paywalled.pdf"
    alt_url = "https://example.com/open.pdf"
    blocked_id = hash_url(url)
    alt_id = hash_url(alt_url)
    alt_path = tmp_workspace["bibliography_dir"] / "_inbox" / f"{alt_id}.pdf"
    fake_pdf = FakePdf(
        pdfs={alt_url: b"%PDF open"},
        conversions={str(alt_path): "# Open\nAlternative evidence."},
        blocked_urls={url},
    )
    fake_chat = FakeChat(
        {
            "gate": [_gate("Alternative evidence.")],
            "synth": ["notes", f"Draft [{alt_id}]", _quality(True)],
        }
    )
    config = _config(tmp_workspace, fake_chat, _blocked_tavily(url), fake_pdf)

    with create_checkpointer(tmp_workspace["state_dir"]) as checkpointer:
        subgraph = build_subagent_subgraph(checkpointer=checkpointer)
        subgraph.invoke(_initial_state(), config)
        result = subgraph.invoke(
            Command(
                resume=[
                    AcquisitionResponse(
                        source_id=blocked_id, kind="alternative", alternative_url=alt_url
                    ).model_dump()
                ]
            ),
            config,
        )

    whitelist_ids = {ref.id for ref in result["whitelisted"]}
    assert alt_id in whitelist_ids
    assert blocked_id not in whitelist_ids
    assert alt_id in {citation.source_id for citation in result["subreport"].citations}


@pytest.mark.integration
def test_concurrent_blocked_fetches_batch_into_single_interrupt(tmp_workspace, monkeypatch):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    urls = ["https://example.com/a.pdf", "https://example.com/b.pdf"]
    fake_tavily = FakeTavily(
        search_results={
            "blocked query": [
                SearchHit(url=urls[0], title="A", snippet="blocked"),
                SearchHit(url=urls[1], title="B", snippet="blocked"),
            ]
        }
    )
    fake_chat = FakeChat(
        {
            "gate": [_gate("A evidence."), _gate("B evidence.")],
            "synth": ["notes", "Draft with both sources", _quality(True)],
        }
    )
    fake_pdf = FakePdf(
        blocked_urls=set(urls),
        conversions={
            str(
                tmp_workspace["bibliography_dir"] / "_inbox" / f"{hash_url(urls[0])}.pdf"
            ): "# A\nA evidence.",
            str(
                tmp_workspace["bibliography_dir"] / "_inbox" / f"{hash_url(urls[1])}.pdf"
            ): "# B\nB evidence.",
        },
    )
    config = _config(tmp_workspace, fake_chat, fake_tavily, fake_pdf)

    with create_checkpointer(tmp_workspace["state_dir"]) as checkpointer:
        subgraph = build_subagent_subgraph(checkpointer=checkpointer)
        first = subgraph.invoke(_initial_state(), config)
        requests = first["__interrupt__"][0].value["requests"]
        for request in requests:
            path = Path(request["save_path"])
            path.write_bytes(f"%PDF {request['title']}".encode())
        result = subgraph.invoke(
            Command(
                resume=[
                    AcquisitionResponse(
                        source_id=request["source_id"],
                        kind="saved",
                        save_path=request["save_path"],
                    ).model_dump()
                    for request in requests
                ]
            ),
            config,
        )

    assert len(requests) == 2
    assert {request["source_id"] for request in requests} == {hash_url(url) for url in urls}
    assert {hash_url(url) for url in urls}.issubset({ref.id for ref in result["whitelisted"]})


@pytest.mark.integration
def test_acquisition_interrupt_during_autonomous_round_does_not_change_brief(
    tmp_workspace, monkeypatch
):
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()
    url = "https://example.com/autonomous.pdf"
    source_id = hash_url(url)
    save_path = tmp_workspace["bibliography_dir"] / "_inbox" / f"{source_id}.pdf"
    fake_pdf = FakePdf(
        blocked_urls={url}, conversions={str(save_path): "# Auto\nAutonomous evidence."}
    )
    fake_chat = FakeChat(
        {
            "gate": [_gate("Autonomous evidence.")],
            "synth": ["notes", f"Autonomous draft [{source_id}]", _quality(True)],
        }
    )
    config = _config(tmp_workspace, fake_chat, _blocked_tavily(url), fake_pdf)
    brief_subtopics = [_subtopic().model_dump()]
    mode = "autonomous"
    pending_handoff = False

    with create_checkpointer(tmp_workspace["state_dir"]) as checkpointer:
        subgraph = build_subagent_subgraph(checkpointer=checkpointer)
        first = subgraph.invoke(_initial_state(), config)
        request = first["__interrupt__"][0].value["requests"][0]
        Path(request["save_path"]).write_bytes(b"%PDF auto")
        result = subgraph.invoke(
            Command(
                resume=[
                    AcquisitionResponse(
                        source_id=source_id, kind="saved", save_path=request["save_path"]
                    ).model_dump()
                ]
            ),
            config,
        )

    assert brief_subtopics == [_subtopic().model_dump()]
    assert mode == "autonomous"
    assert pending_handoff is False
    assert fake_chat.call_count("clarify") == 0
    assert result["subreport"] is not None


@pytest.mark.unit
def test_blocked_url_id_is_deterministic_and_namable_before_bytes(tmp_workspace):
    url = "https://example.com/nameable.pdf"
    request = acquisition.build_request(url, "Title", tmp_workspace["bibliography_dir"])
    assert request.source_id == hash_url(url)
    assert request.save_path.endswith(f"{request.source_id}.pdf")
    assert not Path(request.save_path).exists()


@pytest.mark.unit
def test_acquisition_response_validator_enforces_required_fields():
    with pytest.raises(ValidationError):
        AcquisitionResponse(source_id="x", kind="saved")
    with pytest.raises(ValidationError):
        AcquisitionResponse(source_id="x", kind="alternative")
    with pytest.raises(ValidationError):
        AcquisitionResponse(source_id="x", kind="unobtainable", save_path="/tmp/x.pdf")


@pytest.mark.unit
def test_reconcile_folds_acquired_pdf_under_known_id(tmp_workspace):
    url = "https://example.com/acquired.pdf"
    source_id = hash_url(url)
    path = tmp_workspace["bibliography_dir"] / "_inbox" / f"{source_id}.pdf"
    path.write_bytes(b"%PDF fake")
    fake_pdf = FakePdf(conversions={str(path): "# Folded\nKnown id."})
    embeddings = FakeEmbeddings()
    store = ChromaStore(tmp_workspace["state_dir"], embeddings.embed_query)

    reconcile(tmp_workspace["bibliography_dir"], store, embeddings, pdf_converter=fake_pdf)

    assert not path.exists()
    assert (tmp_workspace["bibliography_dir"] / "_sources" / "pdfs" / f"{source_id}.pdf").exists()
    assert (tmp_workspace["bibliography_dir"] / "_sources" / f"{source_id}.md").exists()


@pytest.mark.unit
def test_apply_response_alternative_blocked_gap_includes_url(tmp_workspace):
    """apply_response embeds the alternative URL (not the reason) in the blocked gap string."""
    bib_dir = tmp_workspace["bibliography_dir"]
    alt_url = "https://example.com/alternative.pdf"
    response = AcquisitionResponse(source_id="deadbeef", kind="alternative", alternative_url=alt_url)
    pdf = FakePdf(pdfs={}, conversions={}, blocked_urls={alt_url})

    source_ref, gap = acquisition.apply_response(
        response,
        original_url="https://example.com/original.pdf",
        original_title="Original",
        bibliography_dir=bib_dir,
        pdf_client=pdf,
        store=None,
        embeddings=None,
    )

    assert source_ref is None
    assert gap is not None
    assert gap.startswith("alternative also blocked: ")
    assert alt_url in gap


@pytest.mark.unit
def test_reconcile_skips_acquired_pdf_already_pooled(tmp_workspace):
    url = "https://example.com/already.pdf"
    source_id = hash_url(url)
    save_pdf(
        b"%PDF existing",
        "# Existing",
        None,
        "Existing",
        tmp_workspace["bibliography_dir"],
        source_id=source_id,
    )
    path = tmp_workspace["bibliography_dir"] / "_inbox" / f"{source_id}.pdf"
    path.write_bytes(b"%PDF duplicate")
    fake_pdf = FakePdf(conversions={str(path): "# Duplicate"})
    embeddings = FakeEmbeddings()
    store = ChromaStore(tmp_workspace["state_dir"], embeddings.embed_query)

    reconcile(tmp_workspace["bibliography_dir"], store, embeddings, pdf_converter=fake_pdf)

    assert not path.exists()
    assert len(list((tmp_workspace["bibliography_dir"] / "_sources").glob(f"{source_id}.md"))) == 1
