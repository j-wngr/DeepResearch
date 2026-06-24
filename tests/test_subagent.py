"""Integration tests for the subagent subgraph (Phase 4)."""

import json
from pathlib import Path

import pytest
from fakes.chat import FakeChat
from fakes.embeddings import FakeEmbeddings
from fakes.pdf import FakePdf
from fakes.tavily import FakeTavily

from deepresearch.config import reset_config
from deepresearch.models import EvidenceExtract, SearchHit, SourceRef, SubReport, SubTopic
from deepresearch.nodes.subagent import build_subagent_subgraph
from deepresearch.paths import hash_url, output_path
from deepresearch.sources.pool import save_web


def _subtopic() -> SubTopic:
    return SubTopic(
        slug="fasting-insulin",
        title="Intermittent fasting and insulin sensitivity",
        scope="Effects of intermittent fasting on insulin sensitivity in humans.",
        guiding_questions=["intermittent fasting"],
        seed_queries=["insulin sensitivity"],
    )


def _seed_source(
    bibliography_dir: Path,
    url: str,
    title: str,
    content: str,
) -> SourceRef:
    """Save a web source to the pool."""
    return save_web(content, url, title, bibliography_dir)


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


def _gate_response_for_test(relevant: bool, quote: str = "") -> str:
    """Return a gate response; the source_id is inferred from the quote."""
    evidence = []
    if relevant and quote:
        evidence.append({"claim": "Relevant claim", "quote": quote})
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


# Build a seed query that FakeEmbeddings will retrieve all given contents.
# Because FakeEmbeddings vectors are deterministic hashes of the text,
# a query equal to the space-joined contents reliably returns every source.
_find_seed_query = lambda *contents: " ".join(contents)  # noqa: E731


def _run_subagent(
    bibliography_dir: Path,
    state_dir: Path,
    output_dir: Path,
    embeddings: FakeEmbeddings,
    fake_chat: FakeChat,
    fake_tavily: FakeTavily | None = None,
    fake_pdf: FakePdf | None = None,
    retrieve_fn=None,
) -> dict:
    return _run_subagent_with_subtopic(
        bibliography_dir,
        state_dir,
        output_dir,
        embeddings,
        fake_chat,
        _subtopic(),
        fake_tavily,
        fake_pdf,
        retrieve_fn=retrieve_fn,
    )


def _run_subagent_with_subtopic(
    bibliography_dir: Path,
    state_dir: Path,
    output_dir: Path,
    embeddings: FakeEmbeddings,
    fake_chat: FakeChat,
    subtopic: SubTopic,
    fake_tavily: FakeTavily | None = None,
    fake_pdf: FakePdf | None = None,
    retrieve_fn=None,
) -> dict:
    from deepresearch.rag.store import ChromaStore

    subgraph = build_subagent_subgraph()
    store = ChromaStore(state_dir, embeddings.embed_query)
    initial_state = {
        "slug": "test-super-slug",
        "subtopic": subtopic,
        "iteration": 0,
        "scratchpad": "",
        "whitelisted": [],
        "evidence": [],
        "draft": None,
        "candidates": [],
        "subreport": None,
        "subreports": {},
        "quality_gate_result": None,
    }
    config = {
        "configurable": {
            "chat_fn": fake_chat.chat,
            "tavily_client": fake_tavily,
            "pdf_client": fake_pdf,
            "embeddings": embeddings,
            "store": store,
            "bibliography_dir": str(bibliography_dir),
            "state_dir": str(state_dir),
            "output_dir": str(output_dir),
        }
    }
    if retrieve_fn is not None:
        config["configurable"]["retrieve_fn"] = retrieve_fn
    return subgraph.invoke(initial_state, config)


@pytest.mark.integration
def test_rag_before_web(tmp_workspace, monkeypatch):
    """Local RAG candidates are consulted before any Tavily search happens."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    url = "https://example.com/fasting-insulin"
    content = "Intermittent fasting improves insulin sensitivity by 30% in trials."
    source_id = hash_url(url)
    _seed_source(bib_dir, url, "Fasting and insulin", content)

    fake_tavily = FakeTavily(
        search_results={
            "intermittent fasting": [
                SearchHit(
                    url="https://example.com/backup",
                    title="Backup",
                    snippet="Backup snippet",
                )
            ]
        },
        extracts={"https://example.com/backup": "Backup content."},
    )
    fake_chat = FakeChat(
        {
            "gate": [
                _gate_response(True, source_id, "Fasting improves insulin sensitivity", content)
            ],
            "synth": [
                "Updated scratchpad.",
                "Draft report body citing [" + source_id + "].",
                _quality_gate_response(True),
            ],
        }
    )

    result = _run_subagent(
        bib_dir,
        state_dir,
        out_dir,
        embeddings,
        fake_chat,
        fake_tavily,
        retrieve_fn=lambda query, slug: [source_id],
    )

    assert any(ref.id == source_id for ref in result["whitelisted"])
    assert fake_tavily.search_count() == 0


@pytest.mark.integration
def test_whitelist_gating(tmp_workspace, monkeypatch):
    """Only sources judged relevant enter the whitelist and evidence list."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    relevant_url = "https://example.com/relevant"
    relevant_content = "Intermittent fasting improves insulin sensitivity."
    irrelevant_url = "https://example.com/irrelevant"
    irrelevant_content = "Solar panels are efficient renewable energy sources."

    relevant_id = hash_url(relevant_url)
    irrelevant_id = hash_url(irrelevant_url)

    _seed_source(bib_dir, relevant_url, "Relevant", relevant_content)
    _seed_source(bib_dir, irrelevant_url, "Irrelevant", irrelevant_content)

    subtopic = SubTopic(
        slug="fasting-insulin",
        title="Intermittent fasting and insulin sensitivity",
        scope="Effects of intermittent fasting on insulin sensitivity in humans.",
        guiding_questions=["Does intermittent fasting improve insulin sensitivity?"],
        seed_queries=["insulin sensitivity"],
    )

    fake_chat = FakeChat(
        {
            "gate": [
                _gate_response_for_test(False),
                _gate_response_for_test(True, relevant_content),
            ],
            "synth": [
                "Updated scratchpad.",
                "Draft report body citing [" + relevant_id + "].",
                _quality_gate_response(True),
            ],
        }
    )
    fake_tavily = FakeTavily()

    result = _run_subagent_with_subtopic(
        bib_dir,
        state_dir,
        out_dir,
        embeddings,
        fake_chat,
        subtopic,
        fake_tavily,
        retrieve_fn=lambda query, slug: [irrelevant_id, relevant_id],
    )

    whitelisted_ids = {ref.id for ref in result["whitelisted"]}
    assert relevant_id in whitelisted_ids
    assert irrelevant_id not in whitelisted_ids

    evidence_ids = {extract.source_id for extract in result["evidence"]}
    assert relevant_id in evidence_ids
    assert irrelevant_id not in evidence_ids
    assert len(evidence_ids) == len(whitelisted_ids)
    assert result["candidates"] == []


@pytest.mark.integration
def test_quality_gate_pass_emits_report(tmp_workspace, monkeypatch):
    """A passing quality gate emits a SubReport and writes the per-sub report.md."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    url = "https://example.com/fasting-insulin"
    content = "Intermittent fasting improves insulin sensitivity."
    _seed_source(bib_dir, url, "Fasting", content)

    draft_body = "Draft report body citing [" + hash_url(url) + "]."
    fake_chat = FakeChat(
        {
            "gate": [
                _gate_response(True, hash_url(url), "Fasting improves insulin sensitivity", content)
            ],
            "synth": ["Updated scratchpad.", draft_body, _quality_gate_response(True)],
        }
    )

    result = _run_subagent(
        bib_dir,
        state_dir,
        out_dir,
        embeddings,
        fake_chat,
        retrieve_fn=lambda query, slug: [hash_url(url)],
    )

    subreport = result["subreport"]
    assert subreport is not None
    assert isinstance(subreport, SubReport)
    assert subreport.shortfall is None
    assert subreport.subtopic_slug == "fasting-insulin"

    report_path = output_path(out_dir, "test-super-slug", "fasting-insulin", "report.md")
    assert report_path.exists()
    assert draft_body in report_path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_quality_gate_fail_loops(tmp_workspace, monkeypatch):
    """A failing quality gate loops back to acquire, bounded, and eventually passes."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    url = "https://example.com/fasting-insulin"
    content = "Intermittent fasting improves insulin sensitivity."
    _seed_source(bib_dir, url, "Fasting", content)

    source_id = hash_url(url)
    draft1 = "First draft citing [" + source_id + "]."
    draft2 = "Second draft citing [" + source_id + "]."
    fake_chat = FakeChat(
        {
            "gate": [
                _gate_response(True, source_id, "Fasting improves insulin sensitivity", content),
                _gate_response(True, source_id, "Fasting improves insulin sensitivity", content),
            ],
            "synth": [
                "First reflect.",
                draft1,
                _quality_gate_response(False, "needs more detail"),
                "Second reflect.",
                draft2,
                _quality_gate_response(True),
            ],
        }
    )

    result = _run_subagent(
        bib_dir,
        state_dir,
        out_dir,
        embeddings,
        fake_chat,
        retrieve_fn=lambda query, slug: [source_id],
    )

    assert result["subreport"] is not None
    assert result["subreport"].shortfall is None
    # Two acquire cycles occurred (iteration 0 and iteration 1) before the
    # passing quality gate emitted the SubReport.
    assert result["iteration"] == 1
    assert fake_chat.call_count("synth") == 6


@pytest.mark.integration
def test_quality_gate_cap_hit_emits_with_shortfall(tmp_workspace, monkeypatch):
    """If quality gate never passes before the iteration cap, emit with shortfall."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    monkeypatch.setenv("SUBAGENT_MAX_ITERATIONS", "2")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    url = "https://example.com/fasting-insulin"
    content = "Intermittent fasting improves insulin sensitivity."
    _seed_source(bib_dir, url, "Fasting", content)

    source_id = hash_url(url)
    draft = "Draft citing [" + source_id + "]."
    fake_chat = FakeChat(
        {
            "gate": [
                _gate_response(True, source_id, "Fasting improves insulin sensitivity", content),
                _gate_response(True, source_id, "Fasting improves insulin sensitivity", content),
                _gate_response(True, source_id, "Fasting improves insulin sensitivity", content),
            ],
            "synth": [
                "Reflect 1",
                draft,
                _quality_gate_response(False),
                "Reflect 2",
                draft,
                _quality_gate_response(False),
                "Reflect 3",
                draft,
                _quality_gate_response(False),
            ],
        }
    )

    result = _run_subagent(
        bib_dir,
        state_dir,
        out_dir,
        embeddings,
        fake_chat,
        retrieve_fn=lambda query, slug: [source_id],
    )

    assert result["iteration"] == 2
    subreport = result["subreport"]
    assert subreport is not None
    assert subreport.shortfall is not None
    assert subreport.shortfall != ""

    report_path = output_path(out_dir, "test-super-slug", "fasting-insulin", "report.md")
    assert report_path.exists()


@pytest.mark.integration
def test_trajectory_compaction(tmp_workspace, monkeypatch):
    """After the gate, raw candidate bodies are dropped; only refs and extracts remain."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    url = "https://example.com/fasting-insulin"
    content = "Intermittent fasting improves insulin sensitivity."
    _seed_source(bib_dir, url, "Fasting", content)

    source_id = hash_url(url)
    fake_chat = FakeChat(
        {
            "gate": [
                _gate_response(True, source_id, "Fasting improves insulin sensitivity", content)
            ],
            "synth": ["Updated scratchpad.", "Draft body.", _quality_gate_response(True)],
        }
    )

    result = _run_subagent(
        bib_dir,
        state_dir,
        out_dir,
        embeddings,
        fake_chat,
        retrieve_fn=lambda query, slug: [source_id],
    )

    assert result["candidates"] == []
    assert all(isinstance(ref, SourceRef) for ref in result["whitelisted"])
    assert all(isinstance(extract, EvidenceExtract) for extract in result["evidence"])
    for extract in result["evidence"]:
        assert isinstance(extract.source_id, str)
        for point in extract.points:
            assert isinstance(point.claim, str)
            assert isinstance(point.quote, str)


@pytest.mark.integration
def test_per_sub_report_written_at_expected_path(tmp_workspace, monkeypatch):
    """The subagent writes its per-sub-topic report to the expected outputs path."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    url = "https://example.com/fasting-insulin"
    content = "Intermittent fasting improves insulin sensitivity."
    _seed_source(bib_dir, url, "Fasting", content)

    source_id = hash_url(url)
    draft = "Final sub-report draft citing [" + source_id + "]."
    fake_chat = FakeChat(
        {
            "gate": [
                _gate_response(True, source_id, "Fasting improves insulin sensitivity", content)
            ],
            "synth": ["Updated scratchpad.", draft, _quality_gate_response(True)],
        }
    )

    _run_subagent(
        bib_dir,
        state_dir,
        out_dir,
        embeddings,
        fake_chat,
        retrieve_fn=lambda query, slug: [source_id],
    )

    report_path = output_path(out_dir, "test-super-slug", "fasting-insulin", "report.md")
    assert report_path.exists()
    assert draft in report_path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_web_gap_fill_when_rag_only_returns_irrelevant(tmp_workspace, monkeypatch):
    """Regression: a pool holding only an off-topic source must not starve a
    sub-topic. RAG keeps returning the irrelevant source (so the old
    ``if not candidates`` web-search guard never fired); once the quality gate
    loops, the subagent must supplement with a web search and whitelist the
    relevant hit it finds."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    # Pool holds one off-topic source that RAG returns for every query.
    irrelevant_url = "https://example.com/off-topic"
    _seed_source(bib_dir, irrelevant_url, "Off topic", "Solar panels and renewable energy.")
    irrelevant_id = hash_url(irrelevant_url)

    # The web search yields a relevant source.
    web_url = "https://example.com/fasting"
    web_content = (
        "Intermittent fasting improves insulin sensitivity in randomised controlled trials. "
        "Time-restricted eating protocols reduce fasting glucose levels and improve markers "
        "of metabolic health across diverse study populations. Mechanisms include enhanced "
        "autophagy, reduced oxidative stress, and improved mitochondrial efficiency. "
        "Studies comparing sixteen-hour fasting windows against control diets report "
        "statistically significant reductions in HbA1c and fasting insulin after twelve "
        "weeks of adherence, with benefits sustained at six-month follow-up. "
        "Caloric restriction combined with time-restricted feeding further amplifies these "
        "metabolic improvements, particularly in individuals with pre-diabetes or metabolic "
        "syndrome. The hormonal response to fasting includes increased glucagon secretion, "
        "decreased insulin levels, and elevated growth hormone, collectively promoting "
        "lipolysis and ketogenesis. These adaptations improve cellular energy metabolism "
        "and reduce chronic low-grade inflammation, which is a key driver of insulin "
        "resistance. Long-term adherence studies indicate that intermittent fasting is as "
        "effective as continuous caloric restriction for weight management and metabolic "
        "health improvement, with comparable dropout rates between the two approaches."
    )
    web_id = hash_url(web_url)

    fake_tavily = FakeTavily(
        search_results={
            "intermittent fasting": [
                SearchHit(url=web_url, title="Fasting study", snippet="snippet")
            ]
        },
        extracts={web_url: web_content},
    )
    draft = "Draft citing [" + web_id + "]."
    fake_chat = FakeChat(
        {
            # iter 0: judge the irrelevant source (False). iter 1: the cached
            # False is reused, then the web source is judged True.
            "gate": [
                _gate_response(False, irrelevant_id, "", ""),
                _gate_response(True, web_id, "Fasting improves insulin sensitivity", web_content),
            ],
            "synth": [
                "Reflect 1",
                "Empty draft.",
                _quality_gate_response(False, "no evidence"),
                "Reflect 2",
                draft,
                _quality_gate_response(True),
            ],
        }
    )

    result = _run_subagent(
        bib_dir,
        state_dir,
        out_dir,
        embeddings,
        fake_chat,
        fake_tavily,
        retrieve_fn=lambda query, slug: [irrelevant_id],
    )

    whitelisted_ids = {ref.id for ref in result["whitelisted"]}
    assert web_id in whitelisted_ids
    assert irrelevant_id not in whitelisted_ids
    assert fake_tavily.search_count() > 0  # the gap-fill search actually ran


@pytest.mark.integration
def test_quality_gate_cannot_pass_without_evidence(tmp_workspace, monkeypatch):
    """Fail loud: even if the scorer says 'pass', a draft with no distilled
    evidence must not emit a clean sub-report -- it loops and then records a
    shortfall with no citations."""
    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    monkeypatch.setenv("SUBAGENT_MAX_ITERATIONS", "1")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    # Nothing in the pool and no web client -> nothing ever whitelisted.
    fake_chat = FakeChat(
        {
            "synth": [
                "Reflect 1",
                "Draft with no citations.",
                _quality_gate_response(True),  # scorer lies: claims pass
                "Reflect 2",
                "Draft with no citations.",
                _quality_gate_response(True),
            ],
        }
    )

    result = _run_subagent(
        bib_dir,
        state_dir,
        out_dir,
        embeddings,
        fake_chat,
        retrieve_fn=lambda query, slug: [],
    )

    subreport = result["subreport"]
    assert subreport is not None
    assert subreport.shortfall  # not a clean pass
    assert subreport.citations == []


@pytest.mark.unit
def test_fetch_skips_empty_web_extract(tmp_workspace):
    """An empty extract must not be saved as a source: an empty-body entry gets
    retrieved by RAG yet rejected by the gate, and it suppresses the gap-fill."""
    from deepresearch.nodes.subagent import _fetch_from_search_hit
    from deepresearch.rag.store import ChromaStore

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    embeddings = FakeEmbeddings()
    store = ChromaStore(state_dir, embeddings.embed_query)

    url = "http://ex.com/paywalled"
    hit = SearchHit(url=url, title="Paywalled", snippet="")
    fake_tavily = FakeTavily(extracts={})  # extract(url) -> ""

    result = _fetch_from_search_hit(hit, bib_dir, fake_tavily, None)

    assert result is None
    assert not (bib_dir / "_sources" / f"{hash_url(url)}.md").exists()
    assert store.count() == 0
