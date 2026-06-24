"""Tests for the LLM-based source quality gate."""

import json
from pathlib import Path

import pytest

from deepresearch.config import reset_config
from deepresearch.models import SourceRef
from deepresearch.source_quality import _cache_path, assess
from deepresearch.sources.pool import save_web


def _make_source(bibliography_dir: Path, url: str, content: str, title: str = "Test") -> SourceRef:
    return save_web(content, url, title, bibliography_dir)


def _quality_response(quality: bool, reason: str = "") -> str:
    return json.dumps({"quality": quality, "reason": reason or ("ok" if quality else "garbage page")})


@pytest.mark.unit
def test_assess_returns_true_for_quality_content(tmp_workspace):
    """A substantive article passes the LLM quality gate."""
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    content = " ".join(["word"] * 300)  # > min_source_words
    source_ref = _make_source(bib_dir, "https://example.com/paper", content)

    call_count = {"n": 0}

    def fake_chat(role, messages):
        call_count["n"] += 1
        return _quality_response(True, "substantive article")

    ok, reason = assess(source_ref, bib_dir, state_dir, chat_fn=fake_chat)

    assert ok is True
    assert "substantive" in reason
    assert call_count["n"] == 1


@pytest.mark.unit
def test_assess_returns_false_for_garbage(tmp_workspace):
    """A garbage page fails the LLM quality gate."""
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    content = " ".join(["word"] * 300)
    source_ref = _make_source(bib_dir, "https://example.com/nav", content, "Nav page")

    def fake_chat(role, messages):
        return _quality_response(False, "navigation page with no content")

    ok, reason = assess(source_ref, bib_dir, state_dir, chat_fn=fake_chat)

    assert ok is False
    assert "navigation" in reason


@pytest.mark.unit
def test_assess_heuristic_shortcircuits_llm_for_short_content(tmp_workspace, monkeypatch):
    """Content below min_source_words is rejected without an LLM call."""
    monkeypatch.setenv("MIN_SOURCE_WORDS", "100")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    content = "Too short."
    source_ref = _make_source(bib_dir, "https://example.com/stub", content, "Stub")

    call_count = {"n": 0}

    def fake_chat(role, messages):
        call_count["n"] += 1
        return _quality_response(True)

    ok, reason = assess(source_ref, bib_dir, state_dir, chat_fn=fake_chat)

    assert ok is False
    assert call_count["n"] == 0


@pytest.mark.unit
def test_assess_cache_hit_skips_llm(tmp_workspace):
    """Second call for the same source returns the cached verdict without an LLM call."""
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    content = " ".join(["word"] * 300)
    source_ref = _make_source(bib_dir, "https://example.com/cached", content, "Cached")

    call_count = {"n": 0}

    def fake_chat(role, messages):
        call_count["n"] += 1
        return _quality_response(True, "first call")

    assess(source_ref, bib_dir, state_dir, chat_fn=fake_chat)
    ok, reason = assess(source_ref, bib_dir, state_dir, chat_fn=fake_chat)

    assert ok is True
    assert call_count["n"] == 1  # second call used cache


@pytest.mark.unit
def test_assess_cache_written_to_disk(tmp_workspace):
    """A verdict is persisted in the JSON cache file."""
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    content = " ".join(["word"] * 300)
    source_ref = _make_source(bib_dir, "https://example.com/persist", content)

    assess(source_ref, bib_dir, state_dir, chat_fn=lambda r, m: _quality_response(False, "link farm"))

    cache = json.loads(_cache_path(state_dir).read_text())
    assert source_ref.id in cache
    assert cache[source_ref.id]["quality"] is False


@pytest.mark.unit
def test_assess_json_parse_error_defaults_to_keep(tmp_workspace):
    """An unparseable LLM response defaults to keeping the source."""
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    content = " ".join(["word"] * 300)
    source_ref = _make_source(bib_dir, "https://example.com/bad-json", content)

    ok, reason = assess(source_ref, bib_dir, state_dir, chat_fn=lambda r, m: "not valid json")

    assert ok is True
    assert "parse error" in reason


@pytest.mark.unit
def test_assess_missing_source_defaults_to_keep(tmp_workspace):
    """A source missing from the pool is not rejected (safe default)."""
    from deepresearch.paths import hash_url

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    ghost_url = "https://example.com/ghost-never-saved"

    ghost_ref = SourceRef(
        id=hash_url(ghost_url),
        type="web",
        url=ghost_url,
        title="Ghost",
        source_path=f"_sources/{hash_url(ghost_url)}.md",
        retrieved_at="2024-01-01T00:00:00+00:00",
        content_hash="abc",
    )

    ok, _ = assess(ghost_ref, bib_dir, state_dir, chat_fn=lambda r, m: _quality_response(False))

    assert ok is True


@pytest.mark.integration
def test_quality_gate_inline_rejects_garbage_from_pool(tmp_workspace, monkeypatch):
    """When a newly fetched source fails the quality gate, it is removed from
    the pool and not passed to the relevance gate as a candidate."""
    import json as _json

    from fakes.chat import FakeChat
    from fakes.embeddings import FakeEmbeddings
    from fakes.tavily import FakeTavily

    from deepresearch.models import SearchHit
    from deepresearch.nodes.subagent import build_subagent_subgraph
    from deepresearch.paths import hash_url
    from deepresearch.rag.store import ChromaStore
    from deepresearch.sources.pool import save_web

    monkeypatch.setenv("SIMILARITY_FLOOR", "0.0")
    reset_config()

    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    out_dir = tmp_workspace["output_dir"]
    embeddings = FakeEmbeddings()

    # A garbage source fetched from the web.
    garbage_url = "https://example.com/garbage"
    garbage_content = " ".join(["word"] * 300)  # passes heuristic; fails LLM gate

    fake_tavily = FakeTavily(
        search_results={
            "fasting-insulin": [
                SearchHit(url=garbage_url, title="Garbage", snippet="s")
            ],
            "intermittent fasting": [
                SearchHit(url=garbage_url, title="Garbage", snippet="s")
            ],
            "insulin sensitivity": [
                SearchHit(url=garbage_url, title="Garbage", snippet="s")
            ],
        },
        extracts={garbage_url: garbage_content},
    )

    fake_chat = FakeChat(
        {
            "quality": [_quality_response(False, "navigation page")] * 20,
            "synth": [
                "Reflect.",
                "Draft no citations.",
                _json.dumps({"passed": False, "coverage_score": 0.0, "reason": "no sources"}),
                "Reflect 2.",
                "Draft no citations.",
                _json.dumps({"passed": False, "coverage_score": 0.0, "reason": "still none"}),
            ],
        }
    )

    from deepresearch.models import SubTopic

    subtopic = SubTopic(
        slug="fasting-insulin",
        title="Intermittent fasting and insulin sensitivity",
        scope="Effects.",
        guiding_questions=["intermittent fasting"],
        seed_queries=["insulin sensitivity"],
    )

    subgraph = build_subagent_subgraph()
    store = ChromaStore(state_dir, embeddings.embed_query)
    monkeypatch.setenv("SUBAGENT_MAX_ITERATIONS", "1")
    reset_config()

    result = subgraph.invoke(
        {
            "slug": "test-super",
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
        },
        {
            "configurable": {
                "chat_fn": fake_chat.chat,
                "tavily_client": fake_tavily,
                "pdf_client": None,
                "embeddings": embeddings,
                "store": store,
                "bibliography_dir": str(bib_dir),
                "state_dir": str(state_dir),
                "output_dir": str(out_dir),
                "quality_gate_enabled": True,
            }
        },
    )

    # Garbage source must not appear in whitelisted
    garbage_id = hash_url(garbage_url)
    assert all(ref.id != garbage_id for ref in result["whitelisted"])
    # Pool file must be deleted
    assert not (bib_dir / "_sources" / f"{garbage_id}.md").exists()
