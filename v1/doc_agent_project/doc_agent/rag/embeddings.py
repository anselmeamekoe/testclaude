"""Embedding backends for RAG, with a graceful no-dependency fallback.

Embeddings turn chunks and queries into vectors so FAISS can find semantically
similar text. The preferred backend is SentenceTransformers (small, fast, good
quality). To keep the whole system runnable in constrained environments (and in
unit tests) without heavy ML wheels, a deterministic hashing embedder is provided
as a fallback — it is far weaker semantically but keeps the pipeline functional.

Both backends conform to the :class:`Embedder` protocol so the index code never
needs to know which one is active.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Anything that can turn a list of strings into an array of float vectors."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` into an ``(len(texts), dim)`` float32 array."""
        ...


class SentenceTransformerEmbedder:
    """High-quality embeddings via the ``sentence-transformers`` library.

    Loaded lazily so importing this module is cheap and side-effect-free.
    """

    def __init__(self, model_name: str) -> None:
        """Load the model and record its embedding dimensionality.

        Args:
            model_name: Hugging Face / SentenceTransformers model id.

        Raises:
            RuntimeError: If ``sentence-transformers`` is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "sentence-transformers not installed; install it or use "
                "HashingEmbedder."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed and L2-normalize ``texts`` for cosine/IP similarity.

        Args:
            texts: Strings to embed.

        Returns:
            A normalized ``(len(texts), dim)`` float32 array.
        """
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")


class HashingEmbedder:
    """Deterministic, dependency-free fallback embedder (bag-of-hashed-tokens).

    Maps tokens into a fixed-size vector via hashing and L2-normalizes the result.
    This captures lexical overlap only (no real semantics), so retrieval quality is
    modest — but it guarantees the RAG pipeline runs anywhere, with no model
    download, which matters for reproducible hackathon evaluation harnesses.
    """

    def __init__(self, dim: int = 512) -> None:
        """Initialize with a target vector dimensionality.

        Args:
            dim: Size of the hashed feature space (and of every output vector).
        """
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        """Hash a single string's tokens into one normalized vector.

        Args:
            text: Input string.

        Returns:
            A length-``dim`` float32 vector (zero vector if no tokens).
        """
        vec = np.zeros(self.dim, dtype="float32")
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(float((vec * vec).sum()))
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed each string with :meth:`_embed_one`.

        Args:
            texts: Strings to embed.

        Returns:
            A normalized ``(len(texts), dim)`` float32 array.
        """
        return np.vstack([self._embed_one(t) for t in texts]).astype("float32")


def build_embedder(model_name: str) -> Embedder:
    """Return the best available embedder, preferring quality over the fallback.

    Tries SentenceTransformers first; if that import fails, transparently returns
    the :class:`HashingEmbedder` so callers always get a working embedder.

    Args:
        model_name: SentenceTransformers model id to attempt.

    Returns:
        An object satisfying the :class:`Embedder` protocol.
    """
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception:  # pragma: no cover - env dependent
        return HashingEmbedder()
