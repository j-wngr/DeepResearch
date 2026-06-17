"""Manual acquisition helpers for blocked PDF fetches."""

import logging
from pathlib import Path

from deepresearch.models import AcquisitionRequest, AcquisitionResponse, Blocked, SourceRef
from deepresearch.paths import hash_url

logger = logging.getLogger("deepresearch.acquisition")


def build_request(url: str, title: str, bibliography_dir: Path) -> AcquisitionRequest:
    """Build a nameable-before-bytes acquisition request for a blocked URL."""
    source_id = hash_url(url)
    save_path = str(bibliography_dir / "_inbox" / f"{source_id}.pdf")
    return AcquisitionRequest(url=url, title=title, source_id=source_id, save_path=save_path)


def format_gaps(gaps: list[str]) -> str:
    """Format acquisition gaps for a subreport shortfall."""
    return "gaps: " + "; ".join(gaps)


def apply_response(
    response: AcquisitionResponse,
    *,
    original_url: str,
    original_title: str,
    bibliography_dir: Path,
    chat_fn,
    tavily_client,
    pdf_client,
    store,
    embeddings,
) -> tuple[SourceRef | None, str | None]:
    """Apply a resume response and return either a SourceRef or a permanent gap."""
    del chat_fn, tavily_client

    from deepresearch.rag import index as rag_index
    from deepresearch.sources import pdf, pool

    if response.kind == "unobtainable":
        return None, f"source unobtainable: {original_url}"

    if response.kind == "saved":
        save_path = Path(response.save_path or "")
        if not save_path.exists():
            raise FileNotFoundError(f"acquired PDF not found at save_path: {save_path}")
        pdf_bytes = save_path.read_bytes()
        markdown = pdf.convert(save_path, converter=pdf_client)
        source_ref = pool.save_pdf(
            pdf_bytes,
            markdown,
            original_url,
            original_title,
            bibliography_dir,
        )
        save_path.unlink(missing_ok=True)
        rag_index.ingest(source_ref, store, embeddings, bibliography_dir)
        return source_ref, None

    fetch_result = pdf.fetch(response.alternative_url or "", pdf_client, bibliography_dir)
    if isinstance(fetch_result, Blocked):
        return None, f"alternative also blocked: {fetch_result.reason}"

    markdown = pdf.convert(fetch_result, converter=pdf_client)
    source_ref = pool.save_pdf(
        fetch_result.read_bytes(),
        markdown,
        response.alternative_url,
        original_title,
        bibliography_dir,
    )
    fetch_result.unlink(missing_ok=True)
    rag_index.ingest(source_ref, store, embeddings, bibliography_dir)
    return source_ref, None
