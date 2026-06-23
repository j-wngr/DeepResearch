"""Runtime resolution of Ollama endpoints with a localhost fallback.

The configured Ollama hosts (chat and embeddings) may live on a LAN that is not
always reachable -- e.g. when running from a different network than the one the
``.env`` was written for. Rather than fail the whole run, we probe each
configured host once at startup and, when it is unreachable but a local Ollama
is up, transparently fall back to ``http://localhost:11434``.

Probing happens only at explicit runtime entry points (the CLI callback and the
live e2e script), never at import or ``Config`` construction, so the test suite
stays hermetic (no network at import time).
"""

import logging
from collections.abc import Callable

import httpx

from deepresearch.config import Config

LOCALHOST_OLLAMA = "http://localhost:11434"

log = logging.getLogger(__name__)

# A probe takes (base_url, headers) and returns whether the host answered.
Probe = Callable[[str, dict | None], bool]


def _http_reachable(base_url: str, headers: dict | None = None, timeout: float = 2.0) -> bool:
    """True when *any* HTTP response comes back from ``base_url``.

    A 4xx/5xx still means the server is up (e.g. an auth-gated cloud endpoint),
    so only connect/timeout/transport failures count as unreachable.
    """
    try:
        httpx.get(base_url, headers=headers or {}, timeout=timeout)
        return True
    except httpx.HTTPError:
        return False


def resolve_ollama_endpoints(cfg: Config, *, probe: Probe = _http_reachable) -> Config:
    """Swap unreachable configured Ollama hosts for a reachable localhost.

    Mutates ``cfg`` in place and returns it. A host is only swapped when it is
    both unreachable *and* a local Ollama is reachable; otherwise the configured
    value is kept so any later error names the intended target. ``localhost`` is
    probed at most once, lazily, and only if some configured host is down.
    """
    headers: dict | None = None
    if cfg.ollama_api_key:
        headers = {"Authorization": f"Bearer {cfg.ollama_api_key}"}

    local_up: bool | None = None  # tri-state: None = not yet probed

    for attr in ("ollama_base_url", "embed_base_url"):
        configured = getattr(cfg, attr)
        if configured.rstrip("/") == LOCALHOST_OLLAMA:
            continue
        if probe(configured, headers):
            continue
        if local_up is None:
            local_up = probe(LOCALHOST_OLLAMA, None)
        if local_up:
            log.warning(
                "Ollama host %s unreachable; falling back to %s for %s",
                configured,
                LOCALHOST_OLLAMA,
                attr,
            )
            setattr(cfg, attr, LOCALHOST_OLLAMA)
        else:
            log.warning(
                "Ollama host %s unreachable and no local Ollama at %s; keeping configured host",
                configured,
                LOCALHOST_OLLAMA,
            )

    return cfg
