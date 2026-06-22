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


class OpenAIEmbedder:
    """Embeddings via any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Use this when the hackathon provider hosts an embeddings model behind an
    OpenAI-compatible API (set ``base_url``). Note that gpt-oss-120b is a *chat*
    model, not an embeddings model, so the embedding ``model`` here is typically a
    separate model id offered by the same provider.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Connect lazily and probe the embedding dimensionality on first encode.

        Args:
            model: Embedding model id as the server expects it.
            api_key: API key/token (defaults to ``OPENAI_API_KEY`` or a placeholder).
            base_url: OpenAI-compatible base URL for the embeddings endpoint.

        Raises:
            RuntimeError: If the ``openai`` package is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "The 'openai' package is required for OpenAIEmbedder. Install it "
                "with `poetry add openai`."
            ) from exc
        self._model = model
        self._client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
        self.dim = 0  # discovered on first call

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed and L2-normalize ``texts`` via the remote embeddings endpoint.

        Args:
            texts: Strings to embed.

        Returns:
            A normalized ``(len(texts), dim)`` float32 array.
        """
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype="float32")
        if self.dim == 0 and vecs.size:
            self.dim = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype("float32")


def build_embedder(settings) -> Embedder:
    """Return the best embedder for the given :class:`~doc_agent.config.Settings`.

    Selection by ``settings.embedding_provider``:

    * ``"openai"``  -> :class:`OpenAIEmbedder` against ``embedding_base_url``.
    * ``"hashing"`` -> the dependency-free :class:`HashingEmbedder`.
    * ``"sentence_transformers"`` / ``"auto"`` (default) -> SentenceTransformers,
      transparently falling back to :class:`HashingEmbedder` if unavailable.

    This keeps embeddings provider-flexible (mirroring the LLM layer) while always
    yielding a working embedder so the RAG pipeline never hard-fails.

    Args:
        settings: The active settings object.

    Returns:
        An object satisfying the :class:`Embedder` protocol.
    """
    provider = getattr(settings, "embedding_provider", "auto")

    if provider == "openai":
        try:
            return OpenAIEmbedder(
                model=settings.embedding_model,
                api_key=getattr(settings, "embedding_api_key", None) or settings.api_key,
                base_url=getattr(settings, "embedding_base_url", None) or settings.base_url,
            )
        except Exception:  # pragma: no cover - env dependent
            return HashingEmbedder()

    if provider == "hashing":
        return HashingEmbedder()

    # "sentence_transformers" or "auto"
    try:
        return SentenceTransformerEmbedder(settings.embedding_model)
    except Exception:  # pragma: no cover - env dependent
        return HashingEmbedder()
