"""RAG subsystem: embeddings, store, indexing, and retrieval."""

from deepresearch.rag.embeddings import embed, embed_query
from deepresearch.rag.index import chunk_markdown, ingest, reconcile
from deepresearch.rag.retrieve import candidates
from deepresearch.rag.store import ChromaStore

__all__ = [
    "embed",
    "embed_query",
    "ChromaStore",
    "chunk_markdown",
    "ingest",
    "reconcile",
    "candidates",
]
