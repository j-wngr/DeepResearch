"""Inbox reconciliation: convert, dedup, and fold drop-ins into the pool."""

import re
from pathlib import Path

from deepresearch.paths import hash_bytes
from deepresearch.sources import pdf
from deepresearch.sources.pool import save_pdf

_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def reconcile(bibliography_dir: Path, pdf_converter=None) -> int:
    """Process _inbox/: convert PDFs to markdown, dedup, fold into pool.

    For each file in _inbox/:
      - If .pdf: convert to markdown via pdf.convert()
      - Determine id: if filename (stem) is 64 hex chars, use as id (hash(url) case);
        otherwise compute hash_bytes(pdf_bytes) from the file content
      - Check if source already exists in pool (_sources/<id>.md); if so, skip
      - Call pool.save_pdf() to write both pdf and md to pool
      - Remove the file from _inbox/

    Returns number of files processed (folded into pool).
    """
    inbox_dir = bibliography_dir / "_inbox"
    if not inbox_dir.exists():
        return 0

    processed = 0
    for file_path in sorted(inbox_dir.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() != ".pdf":
            continue

        pdf_bytes = file_path.read_bytes()
        markdown = pdf.convert(file_path, converter=pdf_converter)

        stem = file_path.stem
        if _HEX_64_RE.match(stem):
            source_id = stem
        else:
            source_id = hash_bytes(pdf_bytes)

        md_path = bibliography_dir / "_sources" / f"{source_id}.md"
        if md_path.exists():
            file_path.unlink()
            continue

        save_pdf(
            pdf_bytes=pdf_bytes,
            markdown=markdown,
            url=None,
            title=file_path.name,
            bibliography_dir=bibliography_dir,
            source_id=source_id,
        )
        file_path.unlink()
        processed += 1

    return processed
