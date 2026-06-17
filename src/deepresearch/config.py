"""Layered project settings.

Precedence: environment variables > .env file > hard-coded defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


class Config:
    """Layered configuration, rebuilt on first access."""

    _instance: "Config | None" = None

    def __init__(self, env_file: Path | str | None = None) -> None:
        # Load .env if present, but do NOT override already-exported env vars.
        # This makes the precedence environment variables > .env file > defaults.
        load_dotenv(dotenv_path=env_file, override=False)

        self.ollama_base_url = self._get_str("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embed_base_url = self._get_str("EMBED_BASE_URL", "http://localhost:11434")
        self.model_fast = self._get_str("MODEL_FAST", "llama3.2:3b")
        self.model_long = self._get_str("MODEL_LONG", "llama3.2:3b")
        self.model_writer = self._get_str("MODEL_WRITER", "llama3.2:3b")
        self.embed_model = self._get_str("EMBED_MODEL", "mxbai-embed-large")
        self.tavily_api_key = self._get_str("TAVILY_API_KEY", "")
        self.bibliography_dir = self._get_path("BIBLIOGRAPHY_DIR", "./Bibliography")
        self.output_dir = self._get_path("OUTPUT_DIR", "./research")
        self.state_dir = self._get_path("STATE_DIR", "./.deepresearch")
        self.max_concurrency = self._get_int("MAX_CONCURRENCY", 2)
        # Phase 9 tuning: keep subagent loops bounded while allowing one retry pass.
        self.subagent_max_iterations = self._get_int("SUBAGENT_MAX_ITERATIONS", 3)
        self.auto_round_cap = self._get_int("AUTO_ROUND_CAP", 2)
        self.max_rounds = self._get_int("MAX_ROUNDS", 5)
        self.subtopics_target = self._get_int("SUBTOPICS_TARGET", 5)
        self.subtopics_ceiling = self._get_int("SUBTOPICS_CEILING", 12)
        self.retrieval_k = self._get_int("RETRIEVAL_K", 10)
        self.similarity_floor = self._get_float("SIMILARITY_FLOOR", 0.5)
        self.provenance_boost = self._get_float("PROVENANCE_BOOST", 0.05)
        self.doc_size_cap = self._get_int("DOC_SIZE_CAP", 50000)
        self.writer_max_revisions = self._get_int("WRITER_MAX_REVISIONS", 3)
        # Bounded transient dependency retries: 3 attempts, 0.5s initial, 2x backoff.
        self.llm_max_retries = self._get_int("LLM_MAX_RETRIES", 3)
        self.llm_retry_initial_interval = self._get_float("LLM_RETRY_INITIAL_INTERVAL", 0.5)
        self.tavily_max_retries = self._get_int("TAVILY_MAX_RETRIES", 3)

    @classmethod
    def get(cls) -> "Config":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @staticmethod
    def _get_str(key: str, default: str) -> str:
        return os.environ.get(key, default)

    @staticmethod
    def _get_int(key: str, default: int) -> int:
        value = os.environ.get(key)
        return int(value) if value is not None else default

    @staticmethod
    def _get_float(key: str, default: float) -> float:
        value = os.environ.get(key)
        return float(value) if value is not None else default

    @staticmethod
    def _get_path(key: str, default: str) -> Path:
        value = os.environ.get(key, default)
        return Path(value).resolve()


def get_config() -> Config:
    """Return the singleton config instance."""
    return Config.get()


def reset_config() -> None:
    """Reset the singleton so the next call re-reads env vars."""
    Config.reset()
