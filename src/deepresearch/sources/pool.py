"""Central content-addressed source pool."""

import datetime
import hashlib
from pathlib import Path

import yaml

from deepresearch.models import SourceRef
from deepresearch.paths import hash_bytes, hash_url

_FRONTMATTER_DELIM = "---"


def _compute_content_hash(markdown: str) -> str:
    return hashlib.sha256(markdown.encode()).hexdigest()


def _write_markdown_with_frontmatter(path: Path, source_ref: SourceRef, markdown: str) -> None:
    frontmatter = {
        "id": source_ref.id,
        "type": source_ref.type,
        "url": source_ref.url,
        "title": source_ref.title,
        "source_path": source_ref.source_path,
        "retrieved_at": source_ref.retrieved_at,
        "content_hash": source_ref.content_hash,
    }
    yaml_text = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    body = markdown.strip("\n") + "\n"
    path.write_text(
        f"{_FRONTMATTER_DELIM}\n{yaml_text}{_FRONTMATTER_DELIM}\n\n{body}", encoding="utf-8"
    )


def _read_markdown_stripping_frontmatter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(_FRONTMATTER_DELIM):
        return text
    parts = text.split(_FRONTMATTER_DELIM, 2)
    if len(parts) < 3:
        return text
    # parts[0] is empty lead-in; parts[2] starts with the body (after closing delim).
    body = parts[2].lstrip("\n")
    return body


def _source_ref(
    source_id: str,
    source_type: str,
    url: str | None,
    title: str,
    source_path: str,
    content_hash: str,
) -> SourceRef:
    return SourceRef(
        id=source_id,
        type=source_type,
        url=url,
        title=title,
        source_path=source_path,
        retrieved_at=datetime.datetime.now(datetime.UTC).isoformat(),
        content_hash=content_hash,
    )


def save_web(markdown: str, url: str, title: str, bibliography_dir: Path) -> SourceRef:
    """Save a web page to the central pool.

    - id = hash_url(url)
    - content_hash = sha256(markdown.encode())
    - Write _sources/<id>.md with YAML frontmatter
    - Dedup: if file already exists, overwrite (re-fetch updates in place)
    - Returns SourceRef
    """
    source_id = hash_url(url)
    content_hash = _compute_content_hash(markdown)
    md_path = bibliography_dir / "_sources" / f"{source_id}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    source_ref = _source_ref(
        source_id=source_id,
        source_type="web",
        url=url,
        title=title,
        source_path=f"_sources/{source_id}.md",
        content_hash=content_hash,
    )
    _write_markdown_with_frontmatter(md_path, source_ref, markdown)
    return source_ref


def save_pdf(
    pdf_bytes: bytes,
    markdown: str,
    url: str | None,
    title: str,
    bibliography_dir: Path,
    source_id: str | None = None,
) -> SourceRef:
    """Save a PDF to the central pool.

    - id = hash_url(url) if url else hash_bytes(pdf_bytes)
    - content_hash = sha256(markdown.encode())
    - Write _sources/pdfs/<id>.pdf AND _sources/<id>.md under the same id
    - Dedup: if files already exist, overwrite
    - Returns SourceRef

    ``source_id`` may be supplied (e.g. by inbox reconciliation for hex-named
    URL-derived ids) to override the computed id.
    """
    if source_id is None:
        source_id = hash_url(url) if url else hash_bytes(pdf_bytes)
    content_hash = _compute_content_hash(markdown)

    pdf_dir = bibliography_dir / "_sources" / "pdfs"
    md_dir = bibliography_dir / "_sources"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / f"{source_id}.pdf"
    md_path = md_dir / f"{source_id}.md"

    pdf_path.write_bytes(pdf_bytes)

    source_ref = _source_ref(
        source_id=source_id,
        source_type="pdf",
        url=url,
        title=title,
        source_path=f"_sources/{source_id}.md",
        content_hash=content_hash,
    )
    _write_markdown_with_frontmatter(md_path, source_ref, markdown)
    return source_ref


def get(source_id: str, bibliography_dir: Path) -> str:
    """Return full markdown body for a source by id.

    Reads _sources/<id>.md, strips YAML frontmatter, returns body.
    Raises FileNotFoundError if source doesn't exist.
    """
    md_path = bibliography_dir / "_sources" / f"{source_id}.md"
    if not md_path.exists():
        raise FileNotFoundError(md_path)
    return _read_markdown_stripping_frontmatter(md_path)


def get_ref(source_id: str, bibliography_dir: Path) -> SourceRef:
    """Return a SourceRef by reading the frontmatter of a pooled source."""
    md_path = bibliography_dir / "_sources" / f"{source_id}.md"
    if not md_path.exists():
        raise FileNotFoundError(md_path)

    text = md_path.read_text(encoding="utf-8")
    if not text.startswith(_FRONTMATTER_DELIM):
        raise ValueError(f"Source {source_id} has no YAML frontmatter")

    parts = text.split(_FRONTMATTER_DELIM, 2)
    if len(parts) < 3:
        raise ValueError(f"Source {source_id} frontmatter is malformed")

    frontmatter = yaml.safe_load(parts[1])
    return SourceRef.model_validate(frontmatter)
