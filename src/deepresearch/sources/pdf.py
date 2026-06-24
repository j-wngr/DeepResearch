"""PDF fetch and convert via httpx, with pymupdf4llm (default), marker (ocr extra), or remote server."""

from pathlib import Path

import httpx

from deepresearch.config import get_config
from deepresearch.models import Blocked
from deepresearch.paths import hash_url

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class HttpxPdfClient:
    """Real PDF client backed by httpx + pymupdf4llm (or marker when PDF_CONVERTER=marker).

    Injected into the graph via ``config["configurable"]["pdf_client"]`` so the
    subagent can fetch and convert PDFs. ``fetch`` returns raw bytes (or
    ``Blocked``); ``pdf.fetch`` then saves them to the inbox. ``convert``
    delegates to the module-level ``convert()``, which dispatches on config.
    Tests inject a fake with the same surface instead.
    """

    _HEADERS = {"User-Agent": _BROWSER_USER_AGENT}

    def fetch(self, url: str) -> bytes | Blocked:
        try:
            response = httpx.get(url, headers=self._HEADERS, follow_redirects=True, timeout=30)
        except Exception as exc:  # noqa: BLE001 - any transport failure blocks the fetch.
            return Blocked(url, reason=f"download failed: {exc}")
        if response.status_code in (401, 403):
            return Blocked(url)
        if response.status_code >= 400:
            return Blocked(url, reason=f"HTTP {response.status_code}")

        content_type = response.headers.get("content-type", "")
        text = response.text if not content_type.startswith("application/pdf") else ""
        if text and is_login_wall(text):
            return Blocked(url)
        data = response.content
        if not data or not data.startswith(b"%PDF-"):
            return Blocked(url, reason="empty or invalid PDF response")
        return data

    def convert(self, path: Path) -> str:
        return convert(path, converter=None)


def is_login_wall(text: str) -> bool:
    """Heuristic: common login-wall markers in response bodies."""
    lowered = text.lower()
    markers = [
        "sign in",
        "log in",
        "login",
        "subscribe to view",
        "please subscribe",
        "create account",
        "register to",
        "membership required",
    ]
    return any(marker in lowered for marker in markers)


# Backwards-compatible alias for code/tests that used the private name.
_is_login_wall = is_login_wall


def fetch(
    url: str,
    client=None,
    bibliography_dir: Path | None = None,
) -> Path | Blocked:
    """Download a PDF via httpx.

    When client is None, uses real httpx.
    When client is provided (FakePdf in tests), uses it directly.
    On 403/401/login-wall responses, returns Blocked(url).
    On success, saves to bibliography_dir/_inbox/<hash_url(url)>.pdf and returns that Path.
    If bibliography_dir is None, gets it from config.
    """
    if client is None:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        if response.status_code in (401, 403):
            return Blocked(url)
        if response.status_code >= 400:
            return Blocked(url, reason=f"HTTP {response.status_code}")

        content_type = response.headers.get("content-type", "")
        text = response.text if not content_type.startswith("application/pdf") else ""
        if text and is_login_wall(text):
            return Blocked(url)

        pdf_bytes = response.content
        if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
            return Blocked(url, reason="empty or invalid PDF response")
    else:
        result = client.fetch(url)
        if isinstance(result, Blocked):
            return result
        pdf_bytes = result.read_bytes() if isinstance(result, Path) else result

    if bibliography_dir is None:
        bibliography_dir = get_config().bibliography_dir

    source_id = hash_url(url)
    inbox_dir = bibliography_dir / "_inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    save_path = inbox_dir / f"{source_id}.pdf"
    save_path.write_bytes(pdf_bytes)
    return save_path


def convert(path: Path, converter=None) -> str:
    """Convert a PDF to markdown.

    When converter is provided (FakePdf in tests), delegates to it directly.
    Otherwise dispatches on PDF_CONVERTER config:
      ``pymupdf``  — default; lightweight, no model downloads, digital PDFs only.
      ``marker``   — local OCR via marker-pdf; requires ``uv sync --extra ocr``.
      ``remote``   — POST to PDF_CONVERTER_URL; marker (or any converter) runs on a
                     remote server; requires PDF_CONVERTER_URL to be set.
    """
    if converter is not None:
        return converter.convert(path)

    cfg = get_config()
    if cfg.pdf_converter == "marker":
        return _convert_marker(path)
    if cfg.pdf_converter == "remote":
        return _convert_remote(path)
    return _convert_pymupdf(path)


def _convert_pymupdf(path: Path) -> str:
    import pymupdf4llm

    return pymupdf4llm.to_markdown(str(path))


def _convert_remote(path: Path) -> str:
    cfg = get_config()
    if not cfg.pdf_converter_url:
        raise RuntimeError(
            "PDF_CONVERTER=remote requires PDF_CONVERTER_URL to be set in your .env."
        )
    headers: dict[str, str] = {}
    if cfg.pdf_converter_api_key:
        headers["Authorization"] = f"Bearer {cfg.pdf_converter_api_key}"
    with open(path, "rb") as f:
        response = httpx.post(
            cfg.pdf_converter_url,
            files={"file": (path.name, f, "application/pdf")},
            headers=headers,
            timeout=300.0,  # marker on a cold server can be slow for large PDFs
        )
    response.raise_for_status()
    return response.json()["markdown"]


def _convert_marker(path: Path) -> str:
    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as exc:
        raise ImportError(
            "marker-pdf is not installed. "
            "Install the ocr extra:  uv sync --extra ocr"
        ) from exc

    config = {
        "use_llm": False,
        "force_ocr": False,
        "languages": "en",
        "output_format": "markdown",
        "output_dir": str(path.parent),
        "processors": None,
        "config_json": None,
        "disable_multiprocessing": False,
        "disable_image_extraction": True,
        "page_range": None,
        "converter_cls": None,
        "llm_service": None,
    }
    config_parser = ConfigParser(config)
    artifact_dict = create_model_dict()
    marker_converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=artifact_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = marker_converter(str(path))
    if hasattr(rendered, "markdown"):
        return rendered.markdown
    return str(rendered)
