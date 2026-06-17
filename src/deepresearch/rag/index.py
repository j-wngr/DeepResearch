"""Chunking, ingest, and reconcile for the global source pool."""

import re
from pathlib import Path

from deepresearch.models import SourceRef
from deepresearch.rag.store import ChromaStore

CHUNK_SIZE = 1000
OVERLAP = 200


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split markdown on #/##/### boundaries, returning (header_path, body) pieces."""
    lines = text.splitlines(keepends=True)
    pieces: list[tuple[str, str]] = []
    current_header_path: list[str] = []
    current_body_lines: list[str] = []

    def level(line: str) -> int | None:
        match = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if not match:
            return None
        return len(match.group(1))

    def flush() -> None:
        if current_body_lines:
            body = "".join(current_body_lines).strip()
            if body:
                pieces.append((" > ".join(current_header_path), body))
            current_body_lines.clear()

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        lvl = level(line)
        if lvl is not None:
            flush()
            title = re.sub(r"^#{1,3}\s+", "", line).strip()
            if lvl == 1:
                current_header_path = [title]
            elif lvl == 2:
                current_header_path = current_header_path[:1] + [title]
            elif lvl == 3:
                current_header_path = current_header_path[:2] + [title]
            # Keep the header in the body so chunks retain context.
            current_body_lines.append(raw_line)
        else:
            current_body_lines.append(raw_line)

    flush()

    if not pieces:
        pieces.append(("", text.strip()))

    return pieces


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Overlap-split a plain text string into chunks of ``size`` chars."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_markdown(text: str, source_id: str) -> list[dict]:
    """Header-aware markdown chunking with deterministic ids.

    Returned dict keys: ``id``, ``document``, ``metadata``.
    Metadata contains ``header_path`` and ``chunk_index``.
    """
    text = text.strip()
    if not text:
        return []

    header_pieces = _split_by_headers(text)
    chunks: list[dict] = []
    chunk_index = 0

    for header_path, section_text in header_pieces:
        if len(section_text) <= CHUNK_SIZE:
            sub_chunks = [section_text]
        else:
            sub_chunks = _split_text(section_text, CHUNK_SIZE, OVERLAP)

        for sub_chunk in sub_chunks:
            chunks.append(
                {
                    "id": f"{source_id}:{chunk_index}",
                    "document": sub_chunk,
                    "metadata": {
                        "header_path": header_path,
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1

    return chunks


def _enrich_metadata(
    chunk: dict,
    source_id: str,
    content_hash: str,
    source_path: str,
    source_url: str,
    title: str,
    source_type: str,
    super_topic: str,
    sub_topic: str,
) -> dict:
    metadata = dict(chunk["metadata"])
    metadata.update(
        {
            "source_id": source_id,
            "content_hash": content_hash,
            "source_path": source_path,
            "source_url": source_url,
            "title": title,
            "type": source_type,
            "super_topic": super_topic,
            "sub_topic": sub_topic,
        }
    )
    return metadata


def ingest(
    source_ref: SourceRef,
    store: ChromaStore,
    embeddings,
    bibliography_dir: Path,
) -> None:
    """Chunk, embed, and index a source already stored in the pool."""
    full_path = bibliography_dir / source_ref.source_path
    text = full_path.read_text(encoding="utf-8")

    chunks = chunk_markdown(text, source_ref.id)
    vectors = embeddings.embed([c["document"] for c in chunks])

    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["embedding"] = vector
        chunk["metadata"] = _enrich_metadata(
            chunk,
            source_id=source_ref.id,
            content_hash=source_ref.content_hash,
            source_path=source_ref.source_path,
            source_url=source_ref.url or "",
            title=source_ref.title,
            source_type=source_ref.type,
            super_topic="",
            sub_topic="",
        )

    store.upsert(chunks)


def reconcile(
    bibliography_dir: Path,
    store: ChromaStore,
    embeddings,
    pdf_converter=None,
) -> int:
    """Reconcile sources: inbox first, then index un-indexed pool markdown.

    Returns the number of newly indexed sources. Inbox processing is performed
    first via ``sources.inbox.reconcile`` but does not count toward this total.

    ``pdf_converter`` is passed through to inbox reconciliation and is used by
    tests to keep the suite hermetic.
    """
    from deepresearch.sources.inbox import reconcile as inbox_reconcile

    inbox_reconcile(bibliography_dir, pdf_converter=pdf_converter)

    sources_dir = bibliography_dir / "_sources"
    indexed = store.list_source_ids()
    newly_indexed = 0

    if not sources_dir.exists():
        return 0

    for md_file in sorted(sources_dir.glob("*.md")):
        source_id = md_file.stem
        if source_id in indexed:
            continue

        text = md_file.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source_id)
        vectors = embeddings.embed([c["document"] for c in chunks])

        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk["embedding"] = vector
            chunk["metadata"] = _enrich_metadata(
                chunk,
                source_id=source_id,
                content_hash="",
                source_path=f"_sources/{source_id}.md",
                source_url="",
                title=md_file.name,
                source_type="web",
                super_topic="",
                sub_topic="",
            )

        store.upsert(chunks)
        newly_indexed += 1

    return newly_indexed
