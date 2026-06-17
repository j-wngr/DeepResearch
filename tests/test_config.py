"""Tests for layered configuration."""

import tempfile
from pathlib import Path

from deepresearch.config import Config, get_config, reset_config


def test_defaults_are_floor(tmp_workspace):
    reset_config()
    cfg = get_config()
    assert cfg.max_concurrency == 2
    assert cfg.ollama_base_url == "http://localhost:11434"
    assert cfg.tavily_api_key == "fake-test-key"


def test_env_var_overrides_default(monkeypatch):
    reset_config()
    monkeypatch.setenv("MAX_CONCURRENCY", "8")
    cfg = Config()
    assert cfg.max_concurrency == 8


def test_env_var_beats_dotenv(monkeypatch):
    reset_config()
    monkeypatch.setenv("MAX_CONCURRENCY", "8")
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / ".env"
        env_file.write_text("MAX_CONCURRENCY=3\n")
        cfg = Config(env_file=env_file)
        # Environment variables have higher precedence than .env file.
        assert cfg.max_concurrency == 8


def test_dotenv_overrides_default():
    reset_config()
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / ".env"
        env_file.write_text("MAX_CONCURRENCY=3\n")
        cfg = Config(env_file=env_file)
        # .env file value wins over hard-coded default when no env var is set.
        assert cfg.max_concurrency == 3


def test_tavily_api_key_no_default(monkeypatch):
    reset_config()
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    cfg = Config()
    assert cfg.tavily_api_key == ""


def test_path_settings_resolved(tmp_workspace):
    reset_config()
    cfg = get_config()
    assert cfg.bibliography_dir == tmp_workspace["bibliography_dir"]
    assert cfg.output_dir == tmp_workspace["output_dir"]
    assert cfg.state_dir == tmp_workspace["state_dir"]


def test_model_fast_from_env(monkeypatch):
    reset_config()
    monkeypatch.setenv("MODEL_FAST", "qwen2.5:7b")
    cfg = Config()
    assert cfg.model_fast == "qwen2.5:7b"


def test_embed_base_url_default():
    reset_config()
    cfg = Config()
    assert cfg.embed_base_url == "http://localhost:11434"


def test_embed_base_url_from_env(monkeypatch):
    reset_config()
    monkeypatch.setenv("EMBED_BASE_URL", "http://embed-host:11434")
    cfg = Config()
    assert cfg.embed_base_url == "http://embed-host:11434"
