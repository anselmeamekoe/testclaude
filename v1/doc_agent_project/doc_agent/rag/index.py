"""FAISS vector index over repository chunks, with a pure-numpy fallback.

This is the retrieval core. It builds an inner-product (cosine, since vectors are
normalized) index over chunk embeddings and answers nearest-neighbour queries. If
``faiss`` is unavailable it falls back to a brute-force numpy search so behaviour
is identical (just slower) without the native dependency.

Retrieval results come back as typed :class:`RetrievedChunk` objects with a
normalized ``score`` in ``[0, 1]`` so downstream confidence calibration can use
them directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search.

    Attributes:
        chunk: The matched :class:`~doc_agent.rag.chunking.Chunk`.
        score: Similarity in ``[0, 1]`` (1.0 == identical direction). Already
            clamped/rescaled from raw inner product for safe downstream use.
    """

    chunk: Chunk
    score: float


class VectorIndex:
    """A searchable index of repository chunks.

    Wraps an embedder and a (FAISS or numpy) nearest-neighbour structure. Build it
    once per repository and query it many times across a question batch.
    """

    def __init__(self, embedder: Embedder) -> None:
        """Create an empty index bound to an embedder.

        Args:
            embedder: Backend used to vectorize both chunks and queries. The
                index dimension is taken from ``embedder.dim``.
        """
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # used by the numpy fallback
        self._faiss_index = None
        self._use_faiss = False

    def build(self, chunks: list[Chunk]) -> "VectorIndex":
        """Embed and index a collection of chunks.

        Attempts to use FAISS (``IndexFlatIP``); on import failure it stores the
        embedding matrix for brute-force numpy search instead.

        Args:
            chunks: The chunks to index. An empty list yields an empty (no-op) index.

        Returns:
            This instance, for chaining.
        """
        self._chunks = list(chunks)
        if not chunks:
            return self

        vectors = self._embedder.encode([c.text for c in chunks])
        try:
            import faiss

            index = faiss.IndexFlatIP(self._embedder.dim)
            index.add(vectors)
            self._faiss_index = index
            self._use_faiss = True
        except Exception:  # pragma: no cover - env dependent
            self._matrix = vectors
            self._use_faiss = False
        return self

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``query``.

        Args:
            query: Natural-language or code-like search string.
            top_k: Maximum number of results to return.

        Returns:
            Up to ``top_k`` :class:`RetrievedChunk` objects sorted by descending
            score. Empty if the index has no chunks.
        """
        if not self._chunks:
            return []
        q = self._embedder.encode([query])

        if self._use_faiss and self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q, min(top_k, len(self._chunks)))
            pairs = zip(idxs[0].tolist(), scores[0].tolist())
        else:
            sims = (self._matrix @ q[0]).tolist()  # type: ignore[operator]
            order = np.argsort(sims)[::-1][:top_k]
            pairs = ((int(i), float(sims[i])) for i in order)

        results: list[RetrievedChunk] = []
        for idx, raw in pairs:
            if idx < 0:
                continue
            # Inner product of unit vectors is in [-1, 1]; rescale to [0, 1].
            score = max(0.0, min(1.0, (raw + 1.0) / 2.0))
            results.append(RetrievedChunk(self._chunks[idx], score))
        return results

    def __len__(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)
