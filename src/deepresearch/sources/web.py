"""Tavily search and extract wrapper."""

import threading
import time

from deepresearch.config import get_config
from deepresearch.models import SearchHit
from deepresearch.retry import TavilyRateLimitError, retry_call

# ---------------------------------------------------------------------------
# Process-wide throttle — shared across all concurrent subagent threads so
# that every Tavily call (search or extract) respects TAVILY_REQUEST_DELAY.
# ---------------------------------------------------------------------------

_throttle_lock = threading.Lock()
_throttle_last_call: float = 0.0


def _throttle(min_interval: float, *, _sleep=time.sleep, _monotonic=time.monotonic) -> None:
    """Block until at least ``min_interval`` seconds have elapsed since the last call."""
    if min_interval <= 0:
        return
    global _throttle_last_call
    with _throttle_lock:
        elapsed = _monotonic() - _throttle_last_call
        wait = min_interval - elapsed
        if wait > 0:
            _sleep(wait)
        _throttle_last_call = _monotonic()


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True for HTTP 429s and Tavily usage-limit exceptions."""
    msg = str(exc).lower()
    if any(k in msg for k in ("429", "rate limit", "too many requests", "usage limit")):
        return True
    # Tavily SDK raises UsageLimitExceededError; match by name so we don't need
    # to import it (it may not exist in all SDK versions).
    return type(exc).__name__.lower() in ("usagelimitexceedederror", "ratelimiterror")


def _transient_tavily_call(fn, *, _sleep=time.sleep, _monotonic=time.monotonic):
    cfg = get_config()

    def _call():
        _throttle(cfg.tavily_request_delay, _sleep=_sleep, _monotonic=_monotonic)
        try:
            return fn()
        except (ConnectionError, TimeoutError) as exc:
            raise TavilyRateLimitError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit_error(exc):
                raise TavilyRateLimitError(str(exc)) from exc
            raise

    return retry_call(
        _call,
        max_attempts=cfg.tavily_max_retries,
        initial_interval=cfg.tavily_retry_initial_interval,
        backoff_factor=2.0,
        max_interval=30.0,  # rate limits need longer recovery than transient errors
    )


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

    raw = _transient_tavily_call(lambda: client.search(query))
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

    raw = _transient_tavily_call(lambda: client.extract(url))
    return _content_from_extract(raw, url)


def _content_from_extract(raw, url: str) -> str:
    """Pull the extracted markdown out of a Tavily extract response.

    The real Tavily extract API returns
    ``{"results": [{"url": ..., "raw_content": ...}], "failed_results": [...]}``;
    the content lives in the ``results`` list, not at the top level. Older or
    alternate shapes (a bare string, a dict keyed by URL, or a dict with a
    top-level ``content``/``raw_content``) are handled too so the parser is
    robust across SDK versions and test doubles.
    """
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected Tavily extract result type: {type(raw)}")

    results = raw.get("results")
    if isinstance(results, list) and results:
        # Prefer the entry matching the requested URL; fall back to the first.
        chosen = next(
            (r for r in results if isinstance(r, dict) and r.get("url") == url),
            None,
        )
        if chosen is None and isinstance(results[0], dict):
            chosen = results[0]
        if chosen is not None:
            return chosen.get("raw_content") or chosen.get("content") or ""

    # Dict keyed by URL, or a flat content payload.
    if url in raw:
        value = raw[url]
        return value if isinstance(value, str) else value.get("content", "")
    return raw.get("content") or raw.get("raw_content") or ""
