"""Retrieval-augmented generation over a cloned repository.

Responsibilities:

* Walk a repository, separating **code** files from **documentation** files.
* Chunk them by token budget with overlap.
* Embed the chunks (Azure embeddings, with a deterministic hashing fallback so
  the pipeline is testable offline).
* Index them in FAISS (cosine via inner-product on normalised vectors).
* Expose ``search_code`` / ``search_docs`` returning :class:`Evidence`.

The :class:`RepoIndex` produced here also carries ``venv_python_path`` so it can
be handed directly to the execution tools, matching the imposed convention
(``RepoIndex.venv_python_path``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..config import Settings
from ..models import Evidence, EvidenceKind

# File-type partitioning -----------------------------------------------------
CODE_EXTENSIONS = {
    ".py", ".pyi", ".ipynb", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
    ".rs", ".cpp", ".cc", ".c", ".h", ".hpp", ".rb", ".scala", ".sql",
}
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc", ".yaml", ".yml", ".toml", ".cfg"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", "dist", "build"}
MAX_FILE_BYTES = 1_000_000  # skip very large/binary-ish files


class Embedder(Protocol):
    """Anything that turns texts into fixed-length vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class HashingEmbedder:
    """Deterministic, network-free embedder for offline development and tests.

    Produces unit-norm vectors from token hashing. Quality is far below a real
    embedding model, but it keeps the retrieval pipeline fully runnable without
    Azure access (useful for CI and local smoke tests).

    Args:
        dim: Output dimensionality.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via a hashed bag-of-tokens projection."""
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # noqa: S324
                vec[h % self.dim] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


class AzureEmbedder:
    """Adapter exposing :class:`~doc_qa_agent.llm.LLMClient` as an :class:`Embedder`."""

    def __init__(self, llm: "object") -> None:
        self._llm = llm

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Delegate to the LLM client's embeddings endpoint."""
        return self._llm.embed(texts)  # type: ignore[attr-defined]


class FaissStore:
    """A small cosine-similarity FAISS index over text chunks.

    Args:
        dim: Embedding dimensionality.
    """

    def __init__(self, dim: int) -> None:
        import faiss

        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Evidence] = []

    def add(self, vectors: list[list[float]], evidences: list[Evidence]) -> None:
        """Add normalised vectors and their associated evidence records."""
        if not vectors:
            return
        arr = _normalise(np.asarray(vectors, dtype=np.float32))
        self._index.add(arr)
        self._chunks.extend(evidences)

    def search(self, query_vector: list[float], top_k: int) -> list[Evidence]:
        """Return the ``top_k`` most similar evidence chunks with scores in [0, 1]."""
        if self._index.ntotal == 0:
            return []
        q = _normalise(np.asarray([query_vector], dtype=np.float32))
        scores, idx = self._index.search(q, min(top_k, self._index.ntotal))
        results: list[Evidence] = []
        for score, i in zip(scores[0], idx[0], strict=True):
            if i < 0:
                continue
            base = self._chunks[i]
            results.append(base.model_copy(update={"score": float(max(0.0, min(1.0, score)))}))
        return results


class RepoIndex(BaseModel):
    """Everything the agent needs to reason about one repository.

    Attributes:
        repo_path: Absolute path to the cloned repository on disk.
        venv_python_path: Python executable inside the repo's venv (passed to
            the execution tools). ``None`` if no venv was created.
        embedding_dim: Dimensionality of the indexed vectors.

    Note:
        The FAISS code/doc stores and the embedder are held as plain attributes
        (``arbitrary_types_allowed``) rather than serialised fields, because
        they are runtime objects, not data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_path: str
    venv_python_path: str | None = None
    embedding_dim: int = 0
    code_store: object | None = Field(default=None, exclude=True)
    doc_store: object | None = Field(default=None, exclude=True)
    embedder: object | None = Field(default=None, exclude=True)

    def search_code(self, query: str, top_k: int) -> list[Evidence]:
        """Semantic search over source-code chunks."""
        if self.code_store is None or self.embedder is None:
            return []
        vec = self.embedder.embed([query])[0]  # type: ignore[attr-defined]
        return self.code_store.search(vec, top_k)  # type: ignore[attr-defined]

    def search_docs(self, query: str, top_k: int) -> list[Evidence]:
        """Semantic search over documentation chunks."""
        if self.doc_store is None or self.embedder is None:
            return []
        vec = self.embedder.embed([query])[0]  # type: ignore[attr-defined]
        return self.doc_store.search(vec, top_k)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Indexing                                                                    #
# --------------------------------------------------------------------------- #
def index_repository(
    repo_path: str,
    embedder: Embedder,
    settings: Settings,
    extra_files: list[str] | None = None,
    venv_python_path: str | None = None,
) -> RepoIndex:
    """Build code and documentation FAISS indices for a repository.

    Args:
        repo_path: Absolute path to the cloned repo.
        embedder: Embedder used to vectorise chunks.
        settings: Provides chunk sizes, top-k and embedding dimension.
        extra_files: Additional file paths (external docs) to index as docs.
        venv_python_path: Stored on the returned index for the execution tools.

    Returns:
        A populated :class:`RepoIndex`.
    """
    code_chunks: list[Evidence] = []
    doc_chunks: list[Evidence] = []

    for path in _iter_files(Path(repo_path)):
        ext = path.suffix.lower()
        if ext in CODE_EXTENSIONS:
            target, kind = code_chunks, EvidenceKind.CODE
        elif ext in DOC_EXTENSIONS:
            target, kind = doc_chunks, EvidenceKind.DOC
        else:
            continue
        target.extend(_chunk_file(path, kind, settings))

    for fp in extra_files or []:
        p = Path(fp)
        if p.is_file():
            doc_chunks.extend(_chunk_file(p, EvidenceKind.DOC, settings))

    dim = settings.embedding_dim
    code_store = FaissStore(dim)
    doc_store = FaissStore(dim)

    _embed_into(code_chunks, embedder, code_store)
    _embed_into(doc_chunks, embedder, doc_store)

    return RepoIndex(
        repo_path=repo_path,
        venv_python_path=venv_python_path,
        embedding_dim=dim,
        code_store=code_store,
        doc_store=doc_store,
        embedder=embedder,
    )


def _embed_into(chunks: list[Evidence], embedder: Embedder, store: FaissStore) -> None:
    """Embed chunk contents in batches and add them to ``store``."""
    batch = 64
    for start in range(0, len(chunks), batch):
        sub = chunks[start : start + batch]
        vectors = embedder.embed([c.content for c in sub])
        store.add(vectors, sub)


def _iter_files(root: Path):
    """Yield candidate files, skipping VCS/build dirs and oversized files."""
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _chunk_file(path: Path, kind: EvidenceKind, settings: Settings) -> list[Evidence]:
    """Split a single file into overlapping token-bounded evidence chunks."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if not text.strip():
        return []

    pieces = _token_windows(text, settings.chunk_tokens, settings.chunk_overlap_tokens)
    out: list[Evidence] = []
    for n, piece in enumerate(pieces):
        out.append(
            Evidence(
                kind=kind,
                source=f"{path}#chunk{n}",
                content=piece,
            )
        )
    return out


def _token_windows(text: str, size: int, overlap: int) -> list[str]:
    """Slice text into overlapping windows by token count.

    Uses ``tiktoken`` when available for accurate token boundaries, otherwise
    falls back to whitespace-delimited words.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        step = max(1, size - overlap)
        return [
            enc.decode(toks[i : i + size]) for i in range(0, len(toks), step) if toks[i : i + size]
        ]
    except Exception:  # noqa: BLE001
        words = text.split()
        step = max(1, size - overlap)
        return [" ".join(words[i : i + size]) for i in range(0, len(words), step) if words[i : i + size]]


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so inner product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
