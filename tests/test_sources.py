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
from deepresearch.sources import (
    extract,
    extract_doi,
    fetch,
    get,
    inbox_reconcile,
    pool,
    remove,
    save_pdf,
    save_web,
)
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
def test_remove_web_source(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    ref = save_web("# Page\n\nContent.\n", "http://example.com/page", "Page", bib_dir)
    md_path = bib_dir / "_sources" / f"{ref.id}.md"

    assert md_path.exists()
    remove(ref.id, bib_dir)
    assert not md_path.exists()


@pytest.mark.unit
def test_remove_pdf_source(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    ref = save_pdf(
        b"pdf bytes",
        "# Paper\n\nText.\n",
        "http://example.com/paper.pdf",
        "Paper",
        bib_dir,
    )
    md_path = bib_dir / "_sources" / f"{ref.id}.md"
    pdf_path = bib_dir / "_sources" / "pdfs" / f"{ref.id}.pdf"

    assert md_path.exists()
    assert pdf_path.exists()
    remove(ref.id, bib_dir)
    assert not md_path.exists()
    assert not pdf_path.exists()


@pytest.mark.unit
def test_remove_nonexistent_is_noop(tmp_workspace):
    """remove() on a source that doesn't exist should not raise."""
    remove("nonexistent-id", tmp_workspace["bibliography_dir"])


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
def test_web_extract_parses_real_tavily_results_shape():
    """The real Tavily extract API returns the content inside a ``results`` list,
    not at the top level. Regression: the old parser fell through to "" for this
    shape, so every fetched web source was saved with an empty body and the
    relevance gate then rejected them all as "empty"."""

    class ResultsShapeTavily:
        def extract(self, url):
            return {
                "results": [
                    {"url": url, "raw_content": "# Green Tea\n\nReal extracted body."}
                ],
                "failed_results": [],
                "response_time": 0.5,
            }

    result = extract("http://ex.com/page", tavily_client=ResultsShapeTavily())

    assert result == "# Green Tea\n\nReal extracted body."


@pytest.mark.unit
def test_throttle_sleeps_when_called_too_soon(monkeypatch):
    """_throttle sleeps for the remaining interval when called before the delay expires."""
    import deepresearch.sources.web as web_mod

    slept: list[float] = []
    t = 0.0

    def fake_monotonic():
        return t

    def fake_sleep(secs):
        nonlocal t
        slept.append(secs)
        t += secs

    # Simulate a call that happened 0.3s ago with a 1.0s delay required
    monkeypatch.setattr(web_mod, "_throttle_last_call", t - 0.3)
    web_mod._throttle(1.0, _sleep=fake_sleep, _monotonic=fake_monotonic)

    assert len(slept) == 1
    assert abs(slept[0] - 0.7) < 0.01


@pytest.mark.unit
def test_throttle_no_sleep_when_delay_elapsed(monkeypatch):
    """_throttle does not sleep when enough time has already passed."""
    import deepresearch.sources.web as web_mod

    slept: list[float] = []
    t = 10.0

    monkeypatch.setattr(web_mod, "_throttle_last_call", t - 2.0)  # 2s ago, delay=1s
    web_mod._throttle(1.0, _sleep=lambda s: slept.append(s), _monotonic=lambda: t)

    assert slept == []


@pytest.mark.unit
def test_transient_tavily_call_retries_on_rate_limit(monkeypatch):
    """A 429-style exception is caught, converted to TavilyRateLimitError, and retried."""
    from deepresearch.config import reset_config
    from deepresearch.sources.web import _transient_tavily_call

    monkeypatch.setenv("TAVILY_MAX_RETRIES", "3")
    monkeypatch.setenv("TAVILY_REQUEST_DELAY", "0")
    reset_config()

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Exception("HTTP 429: too many requests")
        return "ok"

    result = _transient_tavily_call(flaky, _sleep=lambda _: None, _monotonic=lambda: 0.0)
    assert result == "ok"
    assert attempts["n"] == 3
    reset_config()


@pytest.mark.unit
def test_transient_tavily_call_retries_on_usage_limit_class(monkeypatch):
    """Exceptions whose class name contains 'usagelimitexceeded' are treated as rate limits."""
    from deepresearch.config import reset_config
    from deepresearch.sources.web import _transient_tavily_call

    monkeypatch.setenv("TAVILY_MAX_RETRIES", "2")
    monkeypatch.setenv("TAVILY_REQUEST_DELAY", "0")
    reset_config()

    class UsageLimitExceededError(Exception):
        pass

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise UsageLimitExceededError("quota exceeded")
        return "done"

    result = _transient_tavily_call(flaky, _sleep=lambda _: None, _monotonic=lambda: 0.0)
    assert result == "done"
    reset_config()


@pytest.mark.unit
def test_transient_tavily_call_does_not_swallow_other_errors(monkeypatch):
    """Non-rate-limit exceptions propagate immediately without retry."""
    from deepresearch.config import reset_config
    from deepresearch.sources.web import _transient_tavily_call

    monkeypatch.setenv("TAVILY_MAX_RETRIES", "3")
    monkeypatch.setenv("TAVILY_REQUEST_DELAY", "0")
    reset_config()

    attempts = {"n": 0}

    def broken():
        attempts["n"] += 1
        raise ValueError("bad api key")

    with pytest.raises(ValueError, match="bad api key"):
        _transient_tavily_call(broken, _sleep=lambda _: None, _monotonic=lambda: 0.0)

    assert attempts["n"] == 1  # no retries
    reset_config()


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
def test_convert_dispatches_to_pymupdf(monkeypatch):
    """convert() calls _convert_pymupdf when PDF_CONVERTER=pymupdf (default)."""
    from deepresearch.config import reset_config
    from deepresearch.sources.pdf import convert

    monkeypatch.setenv("PDF_CONVERTER", "pymupdf")
    reset_config()

    called_with = []

    monkeypatch.setattr(
        "deepresearch.sources.pdf._convert_pymupdf",
        lambda path: called_with.append(path) or "pymupdf markdown",
    )

    result = convert(Path("/fake/doc.pdf"))
    assert result == "pymupdf markdown"
    assert called_with == [Path("/fake/doc.pdf")]
    reset_config()


@pytest.mark.unit
def test_convert_dispatches_to_marker(monkeypatch):
    """convert() calls _convert_marker when PDF_CONVERTER=marker."""
    from deepresearch.config import reset_config
    from deepresearch.sources.pdf import convert

    monkeypatch.setenv("PDF_CONVERTER", "marker")
    reset_config()

    called_with = []

    monkeypatch.setattr(
        "deepresearch.sources.pdf._convert_marker",
        lambda path: called_with.append(path) or "marker markdown",
    )

    result = convert(Path("/fake/doc.pdf"))
    assert result == "marker markdown"
    assert called_with == [Path("/fake/doc.pdf")]
    reset_config()


@pytest.mark.unit
def test_convert_marker_missing_extra(monkeypatch):
    """_convert_marker raises a clear ImportError when marker-pdf is not installed."""
    import sys

    from deepresearch.sources.pdf import _convert_marker

    # Simulate marker not being installed
    monkeypatch.setitem(sys.modules, "marker", None)
    monkeypatch.setitem(sys.modules, "marker.config.parser", None)
    monkeypatch.setitem(sys.modules, "marker.converters.pdf", None)
    monkeypatch.setitem(sys.modules, "marker.models", None)

    with pytest.raises(ImportError, match="uv sync --extra ocr"):
        _convert_marker(Path("/fake/doc.pdf"))


@pytest.mark.unit
def test_convert_dispatches_to_remote(monkeypatch):
    """convert() calls _convert_remote when PDF_CONVERTER=remote."""
    from deepresearch.config import reset_config
    from deepresearch.sources.pdf import convert

    monkeypatch.setenv("PDF_CONVERTER", "remote")
    monkeypatch.setenv("PDF_CONVERTER_URL", "http://marker-server:8080/convert")
    reset_config()

    called_with = []
    monkeypatch.setattr(
        "deepresearch.sources.pdf._convert_remote",
        lambda path: called_with.append(path) or "remote markdown",
    )

    result = convert(Path("/fake/doc.pdf"))
    assert result == "remote markdown"
    assert called_with == [Path("/fake/doc.pdf")]
    reset_config()


@pytest.mark.unit
def test_convert_remote_missing_url(monkeypatch):
    """_convert_remote raises RuntimeError when PDF_CONVERTER_URL is not set."""
    from deepresearch.config import reset_config
    from deepresearch.sources.pdf import _convert_remote

    monkeypatch.setenv("PDF_CONVERTER_URL", "")
    reset_config()

    with pytest.raises(RuntimeError, match="PDF_CONVERTER_URL"):
        _convert_remote(Path("/fake/doc.pdf"))
    reset_config()


@pytest.mark.unit
def test_convert_remote_posts_pdf_and_returns_markdown(monkeypatch, tmp_path):
    """_convert_remote POSTs the file and extracts markdown from the JSON response."""
    import httpx

    from deepresearch.config import reset_config
    from deepresearch.sources.pdf import _convert_remote

    monkeypatch.setenv("PDF_CONVERTER_URL", "http://marker-server:8080/convert")
    monkeypatch.setenv("PDF_CONVERTER_API_KEY", "")
    reset_config()

    calls = []

    class _MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"markdown": "# Remote\n\nConverted."}

    def fake_post(url, *, files, headers, timeout):
        calls.append({"url": url, "headers": headers})
        return _MockResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")

    result = _convert_remote(pdf)

    assert result == "# Remote\n\nConverted."
    assert calls[0]["url"] == "http://marker-server:8080/convert"
    assert "Authorization" not in calls[0]["headers"]
    reset_config()


@pytest.mark.unit
def test_convert_remote_sends_auth_header(monkeypatch, tmp_path):
    """_convert_remote includes Authorization header when PDF_CONVERTER_API_KEY is set."""
    import httpx

    from deepresearch.config import reset_config
    from deepresearch.sources.pdf import _convert_remote

    monkeypatch.setenv("PDF_CONVERTER_URL", "http://marker-server:8080/convert")
    monkeypatch.setenv("PDF_CONVERTER_API_KEY", "supersecret")
    reset_config()

    captured = {}

    class _MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"markdown": "md"}

    def fake_post(url, *, files, headers, timeout):
        captured["headers"] = headers
        return _MockResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    _convert_remote(pdf)

    assert captured["headers"].get("Authorization") == "Bearer supersecret"
    reset_config()


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


class _MockHttpxResponse:
    """Minimal httpx response stand-in for PDF fetch tests."""

    def __init__(
        self,
        status_code: int = 200,
        content: bytes = b"",
        content_type: str = "application/pdf",
    ):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}
        self.text = content.decode("utf-8", errors="replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("error", request=None, response=None)


@pytest.mark.unit
def test_pdf_client_fetch_empty_response_is_blocked(monkeypatch):
    """HttpxPdfClient.fetch returns Blocked when the server sends an empty body."""
    import httpx

    from deepresearch.sources.pdf import HttpxPdfClient

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _MockHttpxResponse(content=b""))
    result = HttpxPdfClient().fetch("http://ex.com/paper.pdf")
    assert isinstance(result, Blocked)


@pytest.mark.unit
def test_pdf_client_fetch_non_pdf_magic_is_blocked(monkeypatch):
    """HttpxPdfClient.fetch returns Blocked when response lacks the %PDF- header."""
    import httpx

    from deepresearch.sources.pdf import HttpxPdfClient

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: _MockHttpxResponse(
            content=b"<html>error</html>",
            content_type="text/html",
        ),
    )
    result = HttpxPdfClient().fetch("http://ex.com/paper.pdf")
    assert isinstance(result, Blocked)


@pytest.mark.unit
def test_pdf_fetch_no_client_empty_response_is_blocked(tmp_workspace, monkeypatch):
    """Module-level fetch (client=None) returns Blocked for an empty response body."""
    import httpx

    from deepresearch.sources.pdf import fetch as pdf_fetch

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _MockHttpxResponse(content=b""))
    result = pdf_fetch(
        "http://ex.com/paper.pdf",
        bibliography_dir=tmp_workspace["bibliography_dir"],
    )
    assert isinstance(result, Blocked)


@pytest.mark.unit
def test_pdf_fetch_no_client_non_pdf_magic_is_blocked(tmp_workspace, monkeypatch):
    """Module-level fetch (client=None) returns Blocked when bytes lack %PDF- magic."""
    import httpx

    from deepresearch.sources.pdf import fetch as pdf_fetch

    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **kw: _MockHttpxResponse(
            content=b"<html>not a pdf</html>",
            content_type="text/html",
        ),
    )
    result = pdf_fetch(
        "http://ex.com/paper.pdf",
        bibliography_dir=tmp_workspace["bibliography_dir"],
    )
    assert isinstance(result, Blocked)


@pytest.mark.unit
def test_pdf_client_fetch_valid_pdf_passes(monkeypatch):
    """HttpxPdfClient.fetch returns bytes for a valid PDF response."""
    import httpx

    from deepresearch.sources.pdf import HttpxPdfClient

    valid = b"%PDF-1.4 fake content"
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _MockHttpxResponse(content=valid))
    result = HttpxPdfClient().fetch("http://ex.com/paper.pdf")
    assert result == valid


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


@pytest.mark.unit
def test_extract_doi_finds_doi():
    text = "See https://doi.org/10.1038/s41586-021-03819-2 for details."
    assert extract_doi(text) == "10.1038/s41586-021-03819-2"


@pytest.mark.unit
def test_extract_doi_strips_trailing_punctuation():
    assert extract_doi("Published (10.1145/3292500.3330919).") == "10.1145/3292500.3330919"


@pytest.mark.unit
def test_extract_doi_returns_none_when_absent():
    assert extract_doi("No DOI here at all.") is None


@pytest.mark.unit
def test_save_web_extracts_doi(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    markdown = "# Paper\n\nDOI: 10.1109/CVPR.2020.00022\n\nBody text.\n"
    ref = save_web(markdown, "https://example.com/paper", "Paper", bib_dir)
    assert ref.doi == "10.1109/CVPR.2020.00022"


@pytest.mark.unit
def test_save_web_doi_none_when_absent(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    ref = save_web("# Blog\n\nNo doi here.\n", "https://example.com/blog", "Blog", bib_dir)
    assert ref.doi is None


@pytest.mark.unit
def test_save_pdf_extracts_doi(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    markdown = "# Study\n\nhttps://doi.org/10.1016/j.neuron.2021.01.001\n\nContent.\n"
    ref = save_pdf(b"%PDF-fake", markdown, "https://example.com/study.pdf", "Study", bib_dir)
    assert ref.doi == "10.1016/j.neuron.2021.01.001"


@pytest.mark.unit
def test_doi_persisted_in_frontmatter(tmp_workspace):
    bib_dir = tmp_workspace["bibliography_dir"]
    markdown = "# Paper\n\n10.1145/3292500.3330919\n\nBody.\n"
    ref = save_web(markdown, "https://example.com/doi-paper", "Paper", bib_dir)
    assert ref.doi is not None
    loaded = pool.get_ref(ref.id, bib_dir)
    assert loaded.doi == ref.doi
