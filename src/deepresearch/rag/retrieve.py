"""RAG candidate retrieval with floor, provenance re-rank, and dedup."""

from deepresearch.config import get_config
from deepresearch.rag.store import ChromaStore


def candidates(
    query: str,
    super_slug: str,
    store: ChromaStore,
    embeddings,
    k: int | None = None,
    floor: float | None = None,
    boost: float | None = None,
) -> list[str]:
    """Return ordered, deduplicated candidate source ids for a sub-topic query.

    Pipeline:
      1. Query the store for ``k`` nearest chunk neighbours.
      2. Convert Chroma cosine distance to similarity.
      3. Drop results below ``floor``.
      4. Boost similarity for chunks whose ``super_topic`` matches ``super_slug``.
      5. Sort descending by adjusted similarity.
      6. Deduplicate to distinct ``source_id`` values, keeping first occurrence.
    """
    cfg = get_config()
    k = k if k is not None else cfg.retrieval_k
    floor = floor if floor is not None else cfg.similarity_floor
    boost = boost if boost is not None else cfg.provenance_boost

    hits = store.query(query, k=k)
    scored: list[tuple[float, dict]] = []
    for hit in hits:
        similarity = 1.0 - float(hit["distance"])
        if similarity < floor:
            continue
        if hit["metadata"].get("super_topic") == super_slug:
            similarity += boost
        scored.append((similarity, hit))

    scored.sort(key=lambda item: item[0], reverse=True)

    seen: set[str] = set()
    source_ids: list[str] = []
    for _, hit in scored:
        source_id = hit["metadata"].get("source_id")
        if source_id and source_id not in seen:
            seen.add(source_id)
            source_ids.append(source_id)

    return source_ids
