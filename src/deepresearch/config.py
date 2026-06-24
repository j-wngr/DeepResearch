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
        # Load .env only when an env_file is explicitly provided.  This keeps the
        # Config class testable: tests set env vars via monkeypatch and then call
        # Config() without risking the real .env file leaking in.  The CLI and
        # standalone scripts explicitly opt in by passing env_file=".env" (or by
        # calling Config.load_env()).
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=False)

        self.ollama_base_url = self._get_str("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embed_base_url = self._get_str("EMBED_BASE_URL", "http://localhost:11434")
        self.ollama_api_key = self._get_str("OLLAMA_API_KEY", "")
        self.model_fast = self._get_str("MODEL_FAST", "llama3.2:3b")
        self.model_long = self._get_str("MODEL_LONG", "llama3.2:3b")
        self.model_writer = self._get_str("MODEL_WRITER", "llama3.2:3b")
        self.embed_model = self._get_str("EMBED_MODEL", "mxbai-embed-large")
        self.pdf_converter = self._get_str("PDF_CONVERTER", "pymupdf")
        self.pdf_converter_url = self._get_str("PDF_CONVERTER_URL", "")
        self.pdf_converter_api_key = self._get_str("PDF_CONVERTER_API_KEY", "")
        self.tavily_api_key = self._get_str("TAVILY_API_KEY", "")
        self.bibliography_dir = self._get_path("BIBLIOGRAPHY_DIR", "./Bibliography")
        self.output_dir = self._get_path("OUTPUT_DIR", "./Research")
        self.state_dir = self._get_path("STATE_DIR", "./.deepresearch")
        self.max_concurrency = self._get_int("MAX_CONCURRENCY", 2)
        # Phase 9 tuning: keep subagent loops bounded while allowing one retry pass.
        self.subagent_max_iterations = self._get_int("SUBAGENT_MAX_ITERATIONS", 3)
        # Below this many whitelisted sources, a looping subagent supplements RAG
        # candidates with a fresh web search instead of starving on the pool.
        self.min_sources_per_subtopic = self._get_int("MIN_SOURCES_PER_SUBTOPIC", 3)
        self.auto_round_cap = self._get_int("AUTO_ROUND_CAP", 2)
        self.max_rounds = self._get_int("MAX_ROUNDS", 5)
        self.subtopics_target = self._get_int("SUBTOPICS_TARGET", 5)
        self.subtopics_ceiling = self._get_int("SUBTOPICS_CEILING", 12)
        self.retrieval_k = self._get_int("RETRIEVAL_K", 10)
        self.similarity_floor = self._get_float("SIMILARITY_FLOOR", 0.5)
        self.provenance_boost = self._get_float("PROVENANCE_BOOST", 0.05)
        self.doc_size_cap = self._get_int("DOC_SIZE_CAP", 50000)
        self.min_source_words = self._get_int("MIN_SOURCE_WORDS", 150)
        self.max_link_density = self._get_float("MAX_LINK_DENSITY", 0.5)
        self.writer_max_revisions = self._get_int("WRITER_MAX_REVISIONS", 3)
        # Bounded transient dependency retries: 3 attempts, 0.5s initial, 2x backoff.
        self.llm_max_retries = self._get_int("LLM_MAX_RETRIES", 3)
        self.llm_retry_initial_interval = self._get_float("LLM_RETRY_INITIAL_INTERVAL", 0.5)
        self.tavily_max_retries = self._get_int("TAVILY_MAX_RETRIES", 3)
        self.tavily_retry_initial_interval = self._get_float("TAVILY_RETRY_INITIAL_INTERVAL", 1.0)
        self.tavily_request_delay = self._get_float("TAVILY_REQUEST_DELAY", 1.0)
        # When True, blocked PDF acquisitions are silently skipped instead of
        # raising an interrupt that pauses the run for manual download.
        self.skip_acquire_interrupt = self._get_bool("SKIP_ACQUIRE_INTERRUPT", False)
        # When True, only sources already present in the Bibliography are used;
        # web searches and PDF fetches are skipped entirely.
        self.bibliography_only = self._get_bool("BIBLIOGRAPHY_ONLY", False)
        # When False, the LLM source quality gate is disabled for newly fetched
        # sources; heuristic checks in sources/quality.py still apply via prune.
        self.quality_gate_enabled = self._get_bool("QUALITY_GATE_ENABLED", True)

    @classmethod
    def get(cls, env_file: Path | str | None = None) -> "Config":
        if cls._instance is None:
            cls._instance = cls(env_file=env_file)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @classmethod
    def load_env(cls, env_file: Path | str = ".env") -> "Config":
        """Load the specified .env file and return the singleton config."""
        cls.reset()
        return cls.get(env_file=env_file)

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
    def _get_bool(key: str, default: bool) -> bool:
        value = os.environ.get(key)
        if value is None:
            return default
        return value.lower() not in ("0", "false", "no", "")

    @staticmethod
    def _get_path(key: str, default: str) -> Path:
        value = os.environ.get(key, default)
        return Path(value).resolve()


def get_config(env_file: Path | str | None = None) -> Config:
    """Return the singleton config instance.

    Pass ``env_file=".env"`` to opt in to loading a .env file.  When called
    without arguments the singleton is created from already-exported environment
    variables only.
    """
    return Config.get(env_file=env_file)


def reset_config() -> None:
    """Reset the singleton so the next call re-reads env vars."""
    Config.reset()
