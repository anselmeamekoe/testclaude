"""Embedding client for the imposed **OpenAI embedding model**.

Wraps the OpenAI-compatible embeddings endpoint, batches requests, and returns
L2-normalised float32 vectors so that a FAISS inner-product index behaves as a
cosine-similarity index.
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from sparrow_agent.config import Settings


class EmbeddingClient:
    """Batched, normalised embeddings."""

    def __init__(self, settings: Settings) -> None:
        """Create the client.

        Args:
            settings: Global configuration. Uses the ``embedding_*`` fields.
        """
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.embedding_base_url, api_key=settings.embedding_api_key
        )

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: Strings to embed. Empty list returns an empty array.
            normalize: If True (default), L2-normalise each vector so that inner
                product equals cosine similarity.

        Returns:
            A ``(len(texts), dim)`` float32 numpy array.
        """
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        vectors: list[list[float]] = []
        batch = self._settings.embedding_batch_size
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            vectors.extend(self._embed_batch(chunk))

        matrix = np.asarray(vectors, dtype="float32")
        if normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        return matrix

    def embed_one(self, text: str, normalize: bool = True) -> np.ndarray:
        """Embed a single string and return a 1-D vector.

        Args:
            text: The string to embed.
            normalize: Whether to L2-normalise the vector.

        Returns:
            A 1-D float32 vector of shape ``(dim,)``.
        """
        return self.embed([text], normalize=normalize)[0]

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the embeddings endpoint for one batch, with retries.

        Args:
            texts: A single batch of strings (<= ``embedding_batch_size``).

        Returns:
            A list of raw embedding vectors, in input order.
        """
        response = self._client.embeddings.create(
            model=self._settings.embedding_model,
            input=texts,
        )
        # The API preserves order; sort defensively on index anyway.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
