"""Tests for the Sources subsystem (Phase 2)."""

import hashlib
from pathlib import Path

import pytest
from fakes.embeddings import FakeEmbeddings
from fakes.pdf import FakePdf
from fakes.tavily import FakeTavily

from deepresearch.models import Blocked, SearchHit
from deepresearch.paths import hash_bytes, hash_url
from deepresearch.rag.index import reconcile as rag_reconcile
from deepresearch.rag.store import ChromaStore
from deepresearch.sources import extract, fetch, get, inbox_reconcile, pool, save_pdf, save_web
from deepresearch.sources import search as web_search


@pytest.mark.unit
def test_save_web_id_from_url(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    url = "http://example.com/page"
    markdown = "# Page\n\nBody.\n"
    title = "Example Page"

    ref = save_web(markdown, url, title, bib_dir)

    assert ref.id == hash_url(url)
    assert ref.type == "web"
    assert ref.url == url

    md_path = bib_dir / "_sources" / f"{ref.id}.md"
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert f"id: {ref.id}" in text
    assert "type: web" in text
    assert f"url: {url}" in text
    assert f"title: {title}" in text
    assert "retrieved_at:" in text
    assert f"content_hash: {ref.content_hash}" in text
    assert markdown in text


@pytest.mark.unit
def test_save_pdf_id_from_url(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    url = "http://example.com/paper.pdf"
    pdf_bytes = b"fake pdf bytes"
    markdown = "# Paper\n\nAbstract.\n"
    title = "Paper"

    ref = save_pdf(pdf_bytes, markdown, url, title, bib_dir)

    assert ref.id == hash_url(url)
    assert ref.type == "pdf"
    assert ref.url == url

    assert (bib_dir / "_sources" / "pdfs" / f"{ref.id}.pdf").exists()
    assert (bib_dir / "_sources" / f"{ref.id}.md").exists()


@pytest.mark.unit
def test_save_pdf_id_from_bytes(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    pdf_bytes = b"url-less pdf bytes"
    markdown = "# Drop-in\n\nContent.\n"
    title = "Drop-in"

    ref = save_pdf(pdf_bytes, markdown, None, title, bib_dir)

    assert ref.id == hash_bytes(pdf_bytes)
    assert ref.url is None
    assert ref.type == "pdf"


@pytest.mark.unit
def test_content_hash_stored(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    url = "http://example.com/page"
    markdown = "Content for hashing.\n"
    expected = hashlib.sha256(markdown.encode()).hexdigest()

    ref = save_web(markdown, url, "Title", bib_dir)

    assert ref.content_hash == expected

    md_path = bib_dir / "_sources" / f"{ref.id}.md"
    text = md_path.read_text(encoding="utf-8")
    assert f"content_hash: {expected}" in text


@pytest.mark.unit
def test_refetch_updates_in_place(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    url = "http://example.com/page"

    first = save_web("old content", url, "Title", bib_dir)
    second = save_web("new content", url, "Title", bib_dir)

    assert first.id == second.id
    assert second.retrieved_at > first.retrieved_at
    assert second.content_hash != first.content_hash

    md_path = bib_dir / "_sources" / f"{first.id}.md"
    text = md_path.read_text(encoding="utf-8")
    assert "new content" in text
    assert "old content" not in text
    assert first.content_hash not in text
    assert f"content_hash: {second.content_hash}" in text


@pytest.mark.unit
def test_dedup_before_save(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    url = "http://example.com/page"

    first = save_web("content", url, "Title", bib_dir)
    second = save_web("content", url, "Title", bib_dir)

    md_path = bib_dir / "_sources" / f"{first.id}.md"
    assert md_path.exists()
    assert len(list((bib_dir / "_sources").glob("*.md"))) == 1
    assert first.id == second.id


@pytest.mark.unit
def test_save_pdf_writes_both_files(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    url = "http://example.com/paper.pdf"
    pdf_bytes = b"real pdf bytes"
    markdown = "# Converted Paper\n\nText.\n"
    title = "Paper"

    ref = save_pdf(pdf_bytes, markdown, url, title, bib_dir)

    pdf_path = bib_dir / "_sources" / "pdfs" / f"{ref.id}.pdf"
    md_path = bib_dir / "_sources" / f"{ref.id}.md"

    assert pdf_path.read_bytes() == pdf_bytes
    assert md_path.exists()
    body = pool._read_markdown_stripping_frontmatter(md_path)
    assert body == markdown


@pytest.mark.unit
def test_get_returns_markdown(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    url = "http://example.com/page"
    markdown = "# Title\n\nBody.\n"

    ref = save_web(markdown, url, "Title", bib_dir)
    body = get(ref.id, bib_dir)

    assert body == markdown


@pytest.mark.unit
def test_get_raises_on_missing(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    with pytest.raises(FileNotFoundError):
        get("nonexistent-id", bib_dir)


@pytest.mark.unit
def test_web_search_returns_hits():
    hit = SearchHit(url="http://ex.com", title="Ex", snippet="Snippet text")
    fake_tavily = FakeTavily(search_results={"test query": [hit]})

    results = web_search("test query", tavily_client=fake_tavily)

    assert len(results) == 1
    assert results[0].url == "http://ex.com"
    assert results[0].title == "Ex"
    assert results[0].snippet == "Snippet text"


@pytest.mark.unit
def test_web_extract_returns_markdown():
    fake_tavily = FakeTavily(extracts={"http://ex.com": "# Extracted\n\nContent here."})

    result = extract("http://ex.com", tavily_client=fake_tavily)

    assert result == "# Extracted\n\nContent here."


@pytest.mark.unit
def test_pdf_fetch_returns_path(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    fake_pdf = FakePdf(pdfs={"http://ex.com/paper.pdf": b"pdf bytes"})

    result = fetch("http://ex.com/paper.pdf", client=fake_pdf, bibliography_dir=bib_dir)

    assert isinstance(result, Path)
    assert not isinstance(result, Blocked)


@pytest.mark.unit
def test_pdf_fetch_blocked(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    fake_pdf = FakePdf(blocked_urls={"http://paywall.com/paper.pdf"})

    result = fetch("http://paywall.com/paper.pdf", client=fake_pdf, bibliography_dir=bib_dir)

    assert isinstance(result, Blocked)
    assert result.url == "http://paywall.com/paper.pdf"


@pytest.mark.unit
def test_pdf_convert_returns_markdown():
    path = Path("/fake/path/hash.pdf")
    fake_pdf = FakePdf(conversions={str(path): "# Converted\n\nPDF content."})

    from deepresearch.sources import convert as pdf_convert

    result = pdf_convert(path, converter=fake_pdf)

    assert result == "# Converted\n\nPDF content."


@pytest.mark.unit
def test_inbox_reconcile_folds_pdf(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    pdf_bytes = b"inbox pdf bytes"
    inbox_path = bib_dir / "_inbox" / "test.pdf"
    inbox_path.write_bytes(pdf_bytes)

    expected_id = hash_bytes(pdf_bytes)
    fake_pdf = FakePdf(conversions={str(inbox_path): "# Converted\n\nPDF content."})

    count = inbox_reconcile(bib_dir, pdf_converter=fake_pdf)

    assert count == 1
    assert not inbox_path.exists()
    assert not any((bib_dir / "_inbox").iterdir())
    assert (bib_dir / "_sources" / "pdfs" / f"{expected_id}.pdf").exists()
    assert (bib_dir / "_sources" / f"{expected_id}.md").exists()


@pytest.mark.unit
def test_inbox_reconcile_dedup(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    url = "http://example.com/paper.pdf"
    pdf_bytes = b"same pdf bytes"
    markdown = "# Same\n\nContent.\n"

    ref = save_pdf(pdf_bytes, markdown, url, "Same", bib_dir)

    inbox_path = bib_dir / "_inbox" / f"{ref.id}.pdf"
    inbox_path.write_bytes(pdf_bytes)

    fake_pdf = FakePdf(conversions={str(inbox_path): markdown})
    count = inbox_reconcile(bib_dir, pdf_converter=fake_pdf)

    assert count == 0
    assert not inbox_path.exists()


@pytest.mark.unit
def test_inbox_reconcile_hash_url_filename(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    url = "http://example.com/paper.pdf"
    expected_id = hash_url(url)
    pdf_bytes = b"url-named pdf bytes"

    inbox_path = bib_dir / "_inbox" / f"{expected_id}.pdf"
    inbox_path.write_bytes(pdf_bytes)

    fake_pdf = FakePdf(conversions={str(inbox_path): "# Converted\n\nContent."})
    count = inbox_reconcile(bib_dir, pdf_converter=fake_pdf)

    assert count == 1
    assert not inbox_path.exists()
    assert (bib_dir / "_sources" / "pdfs" / f"{expected_id}.pdf").exists()
    assert (bib_dir / "_sources" / f"{expected_id}.md").exists()


@pytest.mark.unit
def test_rag_reconcile_with_inbox(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    state_dir = tmp_workspace["state_dir"]

    # Seed an un-indexed web source in the pool.
    web_url = "http://example.com/page"
    web_id = hash_url(web_url)
    save_web("# Web Source\n\nSome interesting content.\n", web_url, "Web", bib_dir)

    # Drop a PDF in the inbox.
    pdf_bytes = b"rag inbox pdf bytes"
    inbox_path = bib_dir / "_inbox" / "rag-inbox.pdf"
    inbox_path.write_bytes(pdf_bytes)
    pdf_id = hash_bytes(pdf_bytes)
    fake_pdf = FakePdf(conversions={str(inbox_path): "# PDF Source\n\nMore content.\n"})

    embeddings = FakeEmbeddings()
    store = ChromaStore(state_dir, embeddings.embed_query)

    newly_indexed = rag_reconcile(bib_dir, store, embeddings, pdf_converter=fake_pdf)

    assert newly_indexed >= 1
    assert not inbox_path.exists()
    assert (bib_dir / "_sources" / "pdfs" / f"{pdf_id}.pdf").exists()
    assert (bib_dir / "_sources" / f"{pdf_id}.md").exists()
    assert web_id in store.list_source_ids()
    assert pdf_id in store.list_source_ids()
