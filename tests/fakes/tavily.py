"""Tavily search/extract double for testing."""

from deepresearch.models import SearchHit


class FakeTavily:
    """Serve canned search hits and extracts."""

    def __init__(
        self,
        search_results: dict[str, list[SearchHit]] | None = None,
        extracts: dict[str, str] | None = None,
    ) -> None:
        self._search = search_results or {}
        self._extracts = extracts or {}
        self._search_count = 0

    def search(self, query: str) -> list[SearchHit]:
        """Return canned hits for a query."""
        self._search_count += 1
        return self._search.get(query, [])

    def extract(self, url: str) -> str:
        """Return canned markdown for a URL."""
        return self._extracts.get(url, "")

    def search_count(self) -> int:
        """Return how many times ``search`` has been called."""
        return self._search_count
