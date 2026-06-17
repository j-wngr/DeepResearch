"""Ollama embeddings client wrapper."""

from langchain_ollama import OllamaEmbeddings

from deepresearch.config import get_config


def _make_embeddings() -> OllamaEmbeddings:
    cfg = get_config()
    return OllamaEmbeddings(base_url=str(cfg.embed_base_url), model=cfg.embed_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a list of texts via Ollama."""
    return _make_embeddings().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Return a single embedding vector for a query text via Ollama."""
    return _make_embeddings().embed_query(text)
