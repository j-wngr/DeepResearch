"""Tests for the RAG core subsystem."""

import random
import string
import threading
from pathlib import Path

import pytest
from fakes.embeddings import FakeEmbeddings

from deepresearch.models import SourceRef
from deepresearch.paths import hash_url
from deepresearch.rag.index import chunk_markdown, ingest, reconcile
from deepresearch.rag.retrieve import candidates
from deepresearch.rag.store import ChromaStore
from deepresearch.sources import pool


def seed_source(bib_dir: Path, source_id: str, content: str) -> Path:
    path = bib_dir / "_sources" / f"{source_id}.md"
    path.write_text(content)
    return path


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.mark.unit
def test_chunk_header_aware():
    text = (
        "# Title\n\nIntro sentence.\n"
        "## Section 1\n\nContent for section one.\n"
        "## Section 2\n\nContent for section two.\n"
    )
    chunks = chunk_markdown(text, "src1")
    assert len(chunks) >= 3

    header_paths = [c["metadata"]["header_path"] for c in chunks]
    assert "Title" in header_paths
    assert "Title > Section 1" in header_paths
    assert "Title > Section 2" in header_paths


@pytest.mark.unit
def test_chunk_overlap():
    long_body = " ".join(["lorem"] * 500)  # ~2500 chars
    text = f"# Title\n\n{long_body}\n"
    chunks = chunk_markdown(text, "src2")

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        tail = chunks[i]["document"][-200:]
        head = chunks[i + 1]["document"][:200]
        # Consecutive chunks should share at least ~150 chars.
        shared = sum(1 for a, b in zip(tail, head) if a == b)
        assert shared >= 150, f"insufficient overlap between chunk {i} and {i + 1}"


@pytest.mark.unit
def test_chunk_id_deterministic():
    text = "# Title\n\nBody text here.\n## Section A\n\nMore text.\n"
    first = chunk_markdown(text, "src3")
    second = chunk_markdown(text, "src3")
    assert [c["id"] for c in first] == [c["id"] for c in second]


@pytest.mark.unit
def test_idempotent_upsert(tmp_workspace, embeddings):
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]

    source_id = "idempotent-source"
    seed_source(bib_dir, source_id, "# Title\n\nThis is the source body.\n")

    source_ref = SourceRef(
        id=source_id,
        type="pdf",
        url=None,
        title="Title",
        source_path=f"_sources/{source_id}.md",
        retrieved_at="2024-01-01T00:00:00Z",
        content_hash="abc123",
    )

    store = ChromaStore(state_dir, embeddings.embed_query)
    ingest(source_ref, store, embeddings, bib_dir)
    count_after_first = store.count()
    ingest(source_ref, store, embeddings, bib_dir)
    count_after_second = store.count()

    assert count_after_second == count_after_first
    assert count_after_second > 0


@pytest.mark.unit
def test_ingest_tags_provenance_metadata(tmp_workspace, embeddings):
    """Per-run ingest tags chunks with the discovering super/sub topic so the
    provenance re-rank boost can actually fire on later runs."""
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]

    source_id = "provenance-source"
    seed_source(bib_dir, source_id, "# Title\n\nThe quick brown fox.\n")
    source_ref = SourceRef(
        id=source_id,
        type="pdf",
        url=None,
        title="Title",
        source_path=f"_sources/{source_id}.md",
        retrieved_at="2024-01-01T00:00:00Z",
        content_hash="abc123",
    )

    store = ChromaStore(state_dir, embeddings.embed_query)
    ingest(
        source_ref,
        store,
        embeddings,
        bib_dir,
        super_topic="my-research",
        sub_topic="my-subtopic",
    )

    meta = store.collection.get(ids=[f"{source_id}:0"], include=["metadatas"])["metadatas"][0]
    assert meta["super_topic"] == "my-research"
    assert meta["sub_topic"] == "my-subtopic"


@pytest.mark.unit
def test_ingest_indexes_body_not_frontmatter(tmp_workspace, embeddings):
    """Indexing must use the frontmatter-stripped body the gate grounds on, not
    the raw file -- otherwise embeddings carry YAML noise and an empty-body
    source becomes a retrievable 'frontmatter chunk'."""
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    ref = pool.save_web(
        "# Heading\n\nGreen tea lowers blood pressure in adults.",
        "http://ex.com/article",
        "Green Tea and Blood Pressure",
        bib_dir,
    )
    ingest(ref, store, embeddings, bib_dir)

    doc = store.collection.get(ids=[f"{ref.id}:0"], include=["documents"])["documents"][0]
    assert "Green tea lowers blood pressure in adults." in doc
    # Frontmatter keys must not leak into the indexed/embedded text.
    assert "content_hash:" not in doc
    assert "source_path:" not in doc
    assert "retrieved_at:" not in doc


@pytest.mark.unit
def test_ingest_skips_empty_body_source(tmp_workspace, embeddings):
    """A source whose body is empty (e.g. a failed extract saved as frontmatter
    only) must index zero chunks so it is never retrieved as a candidate."""
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    ref = pool.save_web("", "http://ex.com/empty", "Green Tea Empty Page", bib_dir)
    ingest(ref, store, embeddings, bib_dir)

    assert store.count() == 0


@pytest.mark.unit
def test_reconcile_skips_empty_body_source(tmp_workspace, embeddings):
    """Reconcile must not index empty-body pool sources either."""
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    pool.save_web("", "http://ex.com/empty", "Empty", bib_dir)
    pool.save_web("# Real\n\nGreen tea has catechins.", "http://ex.com/real", "Real", bib_dir)

    indexed = reconcile(bib_dir, store, embeddings)
    assert indexed == 1  # only the non-empty source counts
    assert store.count() > 0


@pytest.mark.unit
def test_retrieval_floor(tmp_workspace, embeddings):
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    relevant_id = "relevant-source"
    unrelated_id = "unrelated-source"

    repeated_query = " ".join(["quick brown fox"] * 15)
    # Use content without headers so the single chunk is almost exactly the
    # query phrase repeated, yielding a high cosine similarity under the
    # deterministic FakeEmbeddings hashing scheme.
    seed_source(bib_dir, relevant_id, f"{repeated_query}\n")
    seed_source(bib_dir, unrelated_id, "Alpha beta gamma delta epsilon zeta eta theta iota kappa\n")

    for sid in (relevant_id, unrelated_id):
        ref = SourceRef(
            id=sid,
            type="pdf",
            url=None,
            title=sid,
            source_path=f"_sources/{sid}.md",
            retrieved_at="2024-01-01T00:00:00Z",
            content_hash=sid,
        )
        ingest(ref, store, embeddings, bib_dir)

    # floor=0.0 should include both; floor high enough should drop unrelated.
    all_ids = candidates("quick brown fox", "test-super", store, embeddings, k=10, floor=0.0)
    assert relevant_id in all_ids
    assert unrelated_id in all_ids

    ids_strict = candidates("quick brown fox", "test-super", store, embeddings, k=10, floor=0.2)
    assert relevant_id in ids_strict
    assert unrelated_id not in ids_strict


@pytest.mark.unit
def test_retrieval_provenance_boost(tmp_workspace, embeddings):
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    mine_id = "my-research-source"
    other_id = "other-source"

    seed_source(bib_dir, mine_id, "# Mine\n\nThe quick brown fox jumps over the lazy dog.\n")
    seed_source(bib_dir, other_id, "# Other\n\nThe quick brown fox sleeps all day.\n")

    for sid in (mine_id, other_id):
        ref = SourceRef(
            id=sid,
            type="pdf",
            url=None,
            title=sid,
            source_path=f"_sources/{sid}.md",
            retrieved_at="2024-01-01T00:00:00Z",
            content_hash=sid,
        )
        ingest(ref, store, embeddings, bib_dir)

    # Patch metadata after ingest to set super_topic values.
    # Easiest path: read back, modify, and re-upsert through the collection.
    data = store.collection.get(
        ids=[f"{mine_id}:0"],
        include=["embeddings", "metadatas", "documents"],
    )
    meta = data["metadatas"][0]
    meta["super_topic"] = "my-research"
    store.collection.upsert(
        ids=[f"{mine_id}:0"],
        embeddings=data["embeddings"],
        metadatas=[meta],
        documents=data["documents"],
    )

    ids = candidates(
        "quick brown fox", "my-research", store, embeddings, k=10, floor=0.0, boost=0.5
    )
    assert ids[0] == mine_id


@pytest.mark.unit
def test_retrieval_cross_topic_eligible(tmp_workspace, embeddings):
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    other_id = "other-topic-source"
    seed_source(bib_dir, other_id, "The quick brown fox jumps over the lazy dog.\n")

    ref = SourceRef(
        id=other_id,
        type="pdf",
        url=None,
        title=other_id,
        source_path=f"_sources/{other_id}.md",
        retrieved_at="2024-01-01T00:00:00Z",
        content_hash=other_id,
    )
    ingest(ref, store, embeddings, bib_dir)

    data = store.collection.get(
        ids=[f"{other_id}:0"],
        include=["embeddings", "metadatas", "documents"],
    )
    meta = data["metadatas"][0]
    meta["super_topic"] = "other"
    store.collection.upsert(
        ids=[f"{other_id}:0"],
        embeddings=data["embeddings"],
        metadatas=[meta],
        documents=data["documents"],
    )

    ids = candidates("quick brown fox", "my-research", store, embeddings, k=10, floor=0.0)
    assert other_id in ids


@pytest.mark.unit
def test_retrieval_dedup_to_source_id(tmp_workspace, embeddings):
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    source_id = "multi-chunk-source"
    seed_source(
        bib_dir,
        source_id,
        "# Title\n\nThe quick brown fox jumps over the lazy dog. "
        + " ".join(["It runs through the forest"] * 50)
        + "\n",
    )

    ref = SourceRef(
        id=source_id,
        type="pdf",
        url=None,
        title=source_id,
        source_path=f"_sources/{source_id}.md",
        retrieved_at="2024-01-01T00:00:00Z",
        content_hash=source_id,
    )
    ingest(ref, store, embeddings, bib_dir)

    assert store.count() > 1
    ids = candidates("quick brown fox", "test-super", store, embeddings, k=10, floor=0.0)
    assert ids == [source_id]


@pytest.mark.integration
def test_concurrent_ingest(tmp_workspace, embeddings):
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    sources = ["concurrent-a", "concurrent-b", "concurrent-c"]
    for sid in sources:
        content = "# " + sid + "\n\n" + " ".join(random.choices(string.ascii_letters, k=300))
        seed_source(bib_dir, sid, content)

    refs = [
        SourceRef(
            id=sid,
            type="pdf",
            url=None,
            title=sid,
            source_path=f"_sources/{sid}.md",
            retrieved_at="2024-01-01T00:00:00Z",
            content_hash=sid,
        )
        for sid in sources
    ]

    expected_chunks = sum(
        len(chunk_markdown((bib_dir / "_sources" / f"{sid}.md").read_text(), sid))
        for sid in sources
    )

    errors: list[Exception] = []
    # Ensure every source is ingested at least once, plus two random repeats.
    chosen = refs + [random.choice(refs) for _ in range(2)]

    def worker(ref: SourceRef) -> None:
        try:
            ingest(ref, store, embeddings, bib_dir)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(ref,)) for ref in chosen]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert store.count() == expected_chunks


@pytest.mark.unit
def test_reconcile_indexes_unindexed(tmp_workspace, embeddings):
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    seed_source(bib_dir, "rec-a", "# Rec A\n\nSome interesting content here.\n")
    seed_source(bib_dir, "rec-b", "# Rec B\n\nDifferent content over there.\n")

    first = reconcile(bib_dir, store, embeddings)
    assert first == 2

    expected_count = store.count()
    second = reconcile(bib_dir, store, embeddings)
    assert second == 0
    assert store.count() == expected_count


@pytest.mark.unit
def test_store_delete_removes_chunks(tmp_workspace, embeddings):
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)

    url = "http://example.com/delete-me"
    src_id = hash_url(url)
    ref = SourceRef(
        id=src_id,
        type="web",
        url=url,
        title="Delete Me",
        source_path=f"_sources/{src_id}.md",
        retrieved_at="2024-01-01T00:00:00+00:00",
        content_hash="abc",
    )
    seed_source(bib_dir, src_id, "# Delete Me\n\nContent that will be removed.\n")
    ingest(ref, store, embeddings, bib_dir)
    assert src_id in store.list_source_ids()

    store.delete(src_id)
    assert src_id not in store.list_source_ids()


@pytest.mark.unit
def test_store_delete_nonexistent_is_noop(tmp_workspace, embeddings):
    """delete() on a source not in the store should not raise."""
    state_dir = tmp_workspace["state_dir"]
    store = ChromaStore(state_dir, embeddings.embed_query)
    store.delete("does-not-exist")
