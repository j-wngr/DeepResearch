"""Slug derivation, path resolution, and source identity hashing."""

import hashlib
import re
from pathlib import Path


def slug(question: str) -> str:
    """Derive a stable, collision-resistant slug from the research question.

    Algorithm:
      1. Lowercase and replace non-alphanumerics with hyphens.
      2. Collapse multiple hyphens and trim to a maximum base length.
      3. Append the first 8 hex characters of the SHA-256 hash of the
         original question as a collision-resistant suffix.
    """
    base = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")
    base = base[:60]
    suffix = hashlib.sha256(question.encode()).hexdigest()[:8]
    return f"{base}-{suffix}"


def hash_url(url: str) -> str:
    """Canonical source id for URL-bearing sources."""
    return hashlib.sha256(url.encode()).hexdigest()


def hash_bytes(data: bytes) -> str:
    """Canonical source id / content hash for URL-less drop-ins."""
    return hashlib.sha256(data).hexdigest()


def output_path(output_dir: Path, slug: str, *parts: str) -> Path:
    """Resolve a path under the outputs tree for a super-topic."""
    return output_dir / slug / Path(*parts)


def bibliography_path(bibliography_dir: Path, *parts: str) -> Path:
    """Resolve a path under the Bibliography tree."""
    return bibliography_dir / Path(*parts)


def state_path(state_dir: Path, *parts: str) -> Path:
    """Resolve a path under the run-state tree."""
    return state_dir / Path(*parts)
