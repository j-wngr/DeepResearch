"""PDF fetch/convert double for testing."""

import tempfile
from pathlib import Path

from deepresearch.models import Blocked
from deepresearch.paths import hash_url


class FakePdf:
    """Serve canned PDF bytes, conversions, and blocked URLs."""

    def __init__(
        self,
        pdfs: dict[str, bytes] | None = None,
        conversions: dict[str, str] | None = None,
        blocked_urls: set[str] | None = None,
    ) -> None:
        self._pdfs = pdfs or {}
        self._conversions = conversions or {}
        self._blocked = blocked_urls or set()
        self._temp_paths: list[Path] = []

    def fetch(self, url: str) -> Path | Blocked:
        """Return a fake path for non-blocked URLs, or Blocked sentinel."""
        if url in self._blocked:
            return Blocked(url)
        if url in self._pdfs:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(self._pdfs[url])
                path = Path(tmp.name)
            self._temp_paths.append(path)
            return path
        return Path(f"/fake/path/{hash_url(url)}.pdf")

    def convert(self, path: Path) -> str:
        """Return canned markdown for a path."""
        return self._conversions.get(str(path), "")

    def cleanup(self) -> None:
        """Remove any temporary files created by ``fetch``."""
        for path in self._temp_paths:
            if path.exists():
                path.unlink()
        self._temp_paths.clear()
