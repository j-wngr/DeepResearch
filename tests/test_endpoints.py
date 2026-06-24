"""Tests for the localhost Ollama fallback resolver and embed model pre-flight check."""

import pytest

from deepresearch.config import get_config, reset_config
from deepresearch.endpoints import LOCALHOST_OLLAMA, check_embed_model, resolve_ollama_endpoints


def _fake_probe(up_urls: set[str]):
    """Build a probe that reports the given URLs as reachable."""

    def probe(base_url: str, headers: dict | None = None) -> bool:
        return base_url in up_urls

    return probe


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://lan-chat:11434")
    monkeypatch.setenv("EMBED_BASE_URL", "http://lan-embed:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "")
    reset_config()
    yield get_config()
    reset_config()


@pytest.mark.unit
def test_unreachable_host_falls_back_to_localhost(cfg):
    """A configured host that is down is swapped for a reachable local Ollama."""
    probe = _fake_probe({LOCALHOST_OLLAMA})  # only localhost answers
    resolve_ollama_endpoints(cfg, probe=probe)
    assert cfg.ollama_base_url == LOCALHOST_OLLAMA
    assert cfg.embed_base_url == LOCALHOST_OLLAMA


@pytest.mark.unit
def test_reachable_host_is_kept(cfg):
    """When the configured hosts answer, they are left untouched."""
    probe = _fake_probe({"http://lan-chat:11434", "http://lan-embed:11434"})
    resolve_ollama_endpoints(cfg, probe=probe)
    assert cfg.ollama_base_url == "http://lan-chat:11434"
    assert cfg.embed_base_url == "http://lan-embed:11434"


@pytest.mark.unit
def test_no_local_ollama_keeps_configured_host(cfg):
    """With nothing reachable, the configured host is kept so errors name it."""
    probe = _fake_probe(set())  # nothing answers
    resolve_ollama_endpoints(cfg, probe=probe)
    assert cfg.ollama_base_url == "http://lan-chat:11434"
    assert cfg.embed_base_url == "http://lan-embed:11434"


@pytest.mark.unit
def test_only_unreachable_host_swapped(cfg):
    """Only the down host is swapped; the reachable one is left alone."""
    probe = _fake_probe({"http://lan-chat:11434", LOCALHOST_OLLAMA})
    resolve_ollama_endpoints(cfg, probe=probe)
    assert cfg.ollama_base_url == "http://lan-chat:11434"  # was up, kept
    assert cfg.embed_base_url == LOCALHOST_OLLAMA  # was down, swapped


@pytest.mark.unit
def test_already_localhost_not_probed(monkeypatch):
    """A host already pointing at localhost is never probed or changed."""
    monkeypatch.setenv("OLLAMA_BASE_URL", LOCALHOST_OLLAMA)
    monkeypatch.setenv("EMBED_BASE_URL", LOCALHOST_OLLAMA)
    reset_config()
    cfg = get_config()

    calls: list[str] = []

    def probe(base_url: str, headers: dict | None = None) -> bool:
        calls.append(base_url)
        return False

    resolve_ollama_endpoints(cfg, probe=probe)
    assert calls == []  # nothing probed
    assert cfg.ollama_base_url == LOCALHOST_OLLAMA
    assert cfg.embed_base_url == LOCALHOST_OLLAMA
    reset_config()


# ---------------------------------------------------------------------------
# check_embed_model
# ---------------------------------------------------------------------------


@pytest.fixture
def embed_cfg(monkeypatch):
    monkeypatch.setenv("EMBED_BASE_URL", "http://lan-embed:11434")
    monkeypatch.setenv("EMBED_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("OLLAMA_API_KEY", "")
    reset_config()
    yield get_config()
    reset_config()


@pytest.mark.unit
def test_check_embed_model_passes_when_present(embed_cfg):
    """No exception when the model is listed."""
    check_embed_model(embed_cfg, _tags=lambda url, h: {"mxbai-embed-large", "llama3.2:3b"})


@pytest.mark.unit
def test_check_embed_model_raises_when_missing(embed_cfg):
    """RuntimeError with a pull hint when the model is absent."""
    with pytest.raises(RuntimeError, match="ollama pull mxbai-embed-large"):
        check_embed_model(embed_cfg, _tags=lambda url, h: {"llama3.2:3b"})


@pytest.mark.unit
def test_check_embed_model_skips_when_unreachable(embed_cfg):
    """No exception when the server is unreachable (returns None)."""
    check_embed_model(embed_cfg, _tags=lambda url, h: None)


@pytest.mark.unit
def test_check_embed_model_skips_cloud_endpoint(monkeypatch):
    """Cloud (ollama.com) endpoints are never checked."""
    monkeypatch.setenv("EMBED_BASE_URL", "https://ollama.com")
    monkeypatch.setenv("EMBED_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("OLLAMA_API_KEY", "")
    reset_config()
    cfg = get_config()

    called = []
    check_embed_model(cfg, _tags=lambda url, h: called.append(url) or set())
    assert called == [], "cloud endpoint should not be probed"
    reset_config()
