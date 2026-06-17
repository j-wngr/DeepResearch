"""Tavily search and extract wrapper."""

from deepresearch.config import get_config
from deepresearch.models import SearchHit


def search(query: str, tavily_client=None) -> list[SearchHit]:
    """Search the web via Tavily.

    When tavily_client is None, creates a real TavilyClient using the API key from config.
    When tavily_client is provided (FakeTavily in tests), uses it directly.
    Returns list of SearchHit.
    """
    client = tavily_client
    if client is None:
        from tavily import TavilyClient

        cfg = get_config()
        client = TavilyClient(api_key=cfg.tavily_api_key)

    raw = client.search(query)
    results = raw.get("results", raw) if isinstance(raw, dict) else raw

    hits: list[SearchHit] = []
    for item in results:
        if isinstance(item, SearchHit):
            hits.append(item)
            continue
        if isinstance(item, dict):
            hits.append(
                SearchHit(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", item.get("snippet", "")),
                )
            )
        else:
            raise TypeError(f"Unexpected Tavily search result type: {type(item)}")
    return hits


def extract(url: str, tavily_client=None) -> str:
    """Extract markdown from a URL via Tavily extract.

    When tavily_client is None, creates a real TavilyClient.
    When tavily_client is provided, uses it directly.
    Returns markdown string.
    """
    client = tavily_client
    if client is None:
        from tavily import TavilyClient

        cfg = get_config()
        client = TavilyClient(api_key=cfg.tavily_api_key)

    raw = client.extract(url)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        # Tavily extract may return a dict keyed by URL or contain 'content'/'raw_content'.
        if url in raw:
            value = raw[url]
            return value if isinstance(value, str) else value.get("content", "")
        return raw.get("content", raw.get("raw_content", ""))
    raise TypeError(f"Unexpected Tavily extract result type: {type(raw)}")
