"""Citation resolution, merge, and global renumber."""

import re
from collections.abc import Iterable
from pathlib import Path

from deepresearch.models import SourceRef, SubReport
from deepresearch.sources import pool as pool_module

_CITATION_RE = re.compile(r"\[([0-9a-f]{8,}|\d+)\]")


def extract_inline_citations(body: str) -> list[str]:
    """Return the citation tokens (in order) found in `body`."""
    return [m.group(1) for m in _CITATION_RE.finditer(body)]


def merge(
    subreports: Iterable[SubReport],
    bibliography_dir: Path,
    *,
    question: str = "",
) -> tuple[str, list[SourceRef]]:
    """Merge multiple SubReports into one body with global [n] renumbering."""
    global_numbers: dict[str, int] = {}
    global_source_ids: list[str] = []
    sections: list[str] = []

    def number_for(source_id: str) -> int:
        if source_id not in global_numbers:
            global_source_ids.append(source_id)
            global_numbers[source_id] = len(global_source_ids)
        return global_numbers[source_id]

    for subreport in subreports:
        citations = list(subreport.citations)

        def source_id_for(token: str) -> str:
            if token.isdecimal():
                index = int(token)
                if 1 <= index <= len(citations):
                    return citations[index - 1].source_id
            return token

        def replace(match: re.Match[str]) -> str:
            token = match.group(1)
            source_id = source_id_for(token)
            return f"[{number_for(source_id)}]"

        renumbered = _CITATION_RE.sub(replace, subreport.body)
        title = subreport.subtopic_slug.replace("-", " ").title()
        sections.append(f"## {title}\n\n{renumbered.strip()}")

    parts: list[str] = []
    if question:
        parts.append(f"# Research Report: {question}\n\n=====")
    parts.extend(sections)
    body = "\n\n".join(parts).strip() + ("\n" if parts else "")

    references = [
        pool_module.get_ref(source_id, bibliography_dir) for source_id in global_source_ids
    ]
    return body, references
