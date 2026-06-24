"""Chroma vector store wrapper with a serialized writer."""

import threading
from collections.abc import Callable
from pathlib import Path

import chromadb

COLLECTION_NAME = "deepresearch_sources"


class ChromaStore:
    """Persistent Chroma collection wrapper.

    Writes are serialized through ``self._write_lock`` so concurrent ingest
    calls share one safe writer. The ``embedding_fn`` is used at query time;
    upserts receive pre-computed embeddings.
    """

    def __init__(self, state_dir: Path, embedding_fn: Callable) -> None:
        self._client = chromadb.PersistentClient(path=str(state_dir / "chroma"))
        self._embedding_fn = embedding_fn
        self._write_lock = threading.Lock()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        """Expose the underlying Chroma collection (mostly for tests)."""
        return self._collection

    def upsert(self, chunks: list[dict]) -> None:
        """Upsert pre-embedded chunks through the serialized writer."""
        ids = [c["id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        documents = [c["document"] for c in chunks]
        with self._write_lock:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )

    def replace(self, source_id: str, chunks: list[dict]) -> None:
        """Atomically replace all chunks for ``source_id`` with ``chunks``."""
        ids = [c["id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        documents = [c["document"] for c in chunks]
        with self._write_lock:
            self._collection.delete(where={"source_id": source_id})
            if not chunks:
                return
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )

    def query(self, text: str, k: int) -> list[dict]:
        """Query the store and return matched chunks with distances."""
        vec = self._embedding_fn(text)
        results = self._collection.query(
            query_embeddings=[vec],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
        hits: list[dict] = []
        for i, sid in enumerate(results["ids"][0]):
            hits.append(
                {
                    "id": sid,
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return hits

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._collection.count()

    def list_source_ids(self) -> set[str]:
        """Return all distinct ``source_id`` values stored in chunk metadata."""
        data = self._collection.get(include=["metadatas"])
        source_ids: set[str] = set()
        for meta in data["metadatas"]:
            sid = meta.get("source_id")
            if sid:
                source_ids.add(sid)
        return source_ids

    def list_source_hashes(self) -> dict[str, str]:
        """Return the stored ``content_hash`` for each distinct source_id.

        Only the first chunk seen per source_id is used (all chunks for a source
        share the same hash). Sources indexed before this field was tracked will
        have an empty string, which always differs from the file's real hash and
        therefore triggers a re-index on the next reconcile.
        """
        data = self._collection.get(include=["metadatas"])
        hashes: dict[str, str] = {}
        for meta in data["metadatas"]:
            sid = meta.get("source_id")
            if sid and sid not in hashes:
                hashes[sid] = meta.get("content_hash", "")
        return hashes

    def delete(self, source_id: str) -> None:
        """Remove all chunks for ``source_id`` from the collection."""
        with self._write_lock:
            self._collection.delete(where={"source_id": source_id})

    def reset(self) -> None:
        """Delete and recreate the collection (for tests)."""
        self._client.delete_collection(name=COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
