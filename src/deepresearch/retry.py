"""Bounded retry helpers for transient dependency failures."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("deepresearch.retry")


class TransientError(Exception):
    """Marker for retryable transient failures."""


class OllamaTransientError(TransientError):
    """Retryable Ollama/chat transport failure."""


class TavilyRateLimitError(TransientError):
    """Retryable Tavily throttling/transport failure."""


class ChromaTransientError(TransientError):
    """Retryable Chroma/vector-store failure."""


def retry_call(
    fn: Callable[[], Any],
    *,
    max_attempts: int,
    initial_interval: float,
    backoff_factor: float,
    max_interval: float,
    retry_on: tuple[type[BaseException], ...] = (TransientError,),
    logger: logging.Logger | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``fn`` with bounded deterministic backoff for retryable errors."""
    log = logger or globals()["logger"]
    attempts = max(1, max_attempts)
    interval = initial_interval
    name = getattr(fn, "__name__", fn.__class__.__name__)
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            log.warning("retry %d/%d for %s: %s", attempt, attempts, name, exc)
            sleep_fn(interval)
            interval = min(interval * backoff_factor, max_interval)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_call exhausted without result or exception")
