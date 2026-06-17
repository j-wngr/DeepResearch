"""Deterministic embedding model double for testing."""

import hashlib


class FakeEmbeddings:
    """Return deterministic vectors derived from each text's SHA-256 hash."""

    DIMENSION = 384

    def _vector(self, text: str) -> list[float]:
        """Return a deterministic float vector derived from ``text``."""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        vector = []
        for i in range(self.DIMENSION):
            # Consume hash bytes cyclically; scale a byte to a small float.
            byte = hash_bytes[i % len(hash_bytes)]
            # Deterministic pseudo-randomish float in [-1, 1].
            vector.append((byte / 255.0) * 2.0 - 1.0)
        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic vector per text."""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)
