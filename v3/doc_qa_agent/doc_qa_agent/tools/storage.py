"""Imposed Qdrant ("sparrow") vector-storage backend.

This wraps the organizer-provided ``get_qdrant_client`` helper and exposes a
small :class:`QdrantVectorStore` that matches the same search interface as the
local FAISS store, so the agent can use either backend interchangeably.

FAISS is the default (zero infra, fast for a single repo). Qdrant is the
imposed option for when an index must be persisted/shared across runs.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..models import Evidence, EvidenceKind


def get_qdrant_client(sparrow_token: str) -> Any:
    """Return an authenticated Qdrant client for the sparrow cluster.

    Args:
        sparrow_token: Auth token for the internal sparrow Qdrant deployment.

    Returns:
        A configured ``qdrant_client.QdrantClient``.
    """
    from qdrant_client import QdrantClient  # imported lazily to keep import cost low
    from sparrow_flow.auth import AuthConfig

    return QdrantClient(
        "qdrant.sparrow.cloud.net.intra",
        auth=AuthConfig(sparrow_token=sparrow_token).auth,
        check_compatibility=False,
        https=True,
        port=443,
    )


class QdrantVectorStore:
    """A persistent vector store backed by the imposed Qdrant cluster.

    Mirrors the search surface of :class:`~doc_qa_agent.tools.retrieval.FaissStore`
    so it is a drop-in replacement when persistence is required.

    Args:
        sparrow_token: Token used to build the underlying client.
        collection: Qdrant collection name.
        dim: Embedding dimensionality of the vectors stored.
    """

    def __init__(self, sparrow_token: str, collection: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client = get_qdrant_client(sparrow_token)
        self._collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Upsert vectors with their metadata payloads.

        Args:
            vectors: Embedding vectors.
            payloads: Per-vector metadata (``kind``, ``source``, ``content``).
        """
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=v, payload=p)
            for v, p in zip(vectors, payloads, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: list[float], top_k: int) -> list[Evidence]:
        """Return the ``top_k`` nearest payloads as :class:`Evidence`.

        Args:
            query_vector: The embedded query.
            top_k: Number of neighbours to return.

        Returns:
            Evidence objects with cosine similarity mapped into ``[0, 1]``.
        """
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
        )
        results: list[Evidence] = []
        for h in hits:
            payload = h.payload or {}
            results.append(
                Evidence(
                    kind=EvidenceKind(payload.get("kind", "doc")),
                    source=payload.get("source", "qdrant"),
                    content=payload.get("content", ""),
                    score=max(0.0, min(1.0, (h.score + 1.0) / 2.0)),
                )
            )
        return results
