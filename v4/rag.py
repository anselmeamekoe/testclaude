"""RAG layer: code-aware chunking, a FAISS index, and a retriever.

Design notes
------------
* **Code-aware chunking.** Python files are split along top-level ``def``/``class``
  boundaries via the ``ast`` module so that a function and its docstring stay
  together. Everything else (Markdown, RST, txt, config, other languages) is
  split with a token-budgeted sliding window.
* **One index, two views.** All chunks live in a single FAISS
  ``IndexFlatIP`` (inner product over normalised vectors = cosine similarity).
  ``search_code`` and ``search_docs`` over-fetch and then filter by ``kind`` so
  the agent can target source vs. prose without maintaining two indices.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import faiss
import numpy as np
import tiktoken

from sparrow_agent.config import Settings
from sparrow_agent.embeddings import EmbeddingClient
from sparrow_agent.models import Chunk, RetrievalHit

# File-extension taxonomy used to tag chunks and to skip noise.
_CODE_EXT = {
    ".py", ".pyi", ".ipynb", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
    ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".rb", ".php", ".scala",
    ".sh", ".sql", ".yaml", ".yml", ".toml", ".cfg", ".ini",
}
_DOC_EXT = {".md", ".rst", ".txt", ".adoc"}
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", "dist", "build"}
_BINARY_HINT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz",
                ".whl", ".so", ".dylib", ".bin", ".pyc", ".parquet", ".npy"}

_ENC = None  # lazily initialised; may stay None if the BPE file is unavailable
_ENC_READY = False


def _get_encoder():
    """Lazily load the ``cl100k_base`` encoder, tolerating offline environments.

    ``tiktoken`` downloads its BPE file on first use; if that is blocked, we fall
    back to a character heuristic rather than crashing the whole pipeline.

    Returns:
        A tiktoken encoder, or ``None`` if it could not be loaded.
    """
    global _ENC, _ENC_READY
    if not _ENC_READY:
        _ENC_READY = True
        try:
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENC = None
    return _ENC


def _count_tokens(text: str) -> int:
    """Return an approximate token count for sizing chunks.

    Args:
        text: Text to measure.

    Returns:
        Number of tokens via ``cl100k_base`` when available, else a ~4 chars per
        token heuristic.
    """
    enc = _get_encoder()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text, disallowed_special=()))


class CodeChunker:
    """Turns a repository directory tree into a flat list of :class:`Chunk`."""

    def __init__(self, settings: Settings) -> None:
        """Initialise the chunker.

        Args:
            settings: Global configuration (chunk sizes, file-size cap).
        """
        self._settings = settings

    def chunk_repo(self, root: str | os.PathLike[str]) -> list[Chunk]:
        """Walk ``root`` and produce chunks for every indexable file.

        Args:
            root: Absolute path to the repository root.

        Returns:
            A list of chunks across all files (order is deterministic by path).
        """
        root_path = Path(root)
        chunks: list[Chunk] = []
        for file_path in sorted(self._iter_files(root_path)):
            rel = str(file_path.relative_to(root_path))
            ext = file_path.suffix.lower()
            kind = "code" if ext in _CODE_EXT else "doc" if ext in _DOC_EXT else "other"
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if ext == ".py":
                chunks.extend(self._chunk_python(rel, text))
            else:
                chunks.extend(self._chunk_windowed(rel, text, kind))
        return chunks

    def _iter_files(self, root: Path):
        """Yield indexable files under ``root``, skipping noise and binaries.

        Args:
            root: Repository root.

        Yields:
            ``Path`` objects for files worth embedding.
        """
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                path = Path(dirpath) / name
                if path.suffix.lower() in _BINARY_HINT:
                    continue
                try:
                    if path.stat().st_size > self._settings.max_file_bytes:
                        continue
                except OSError:
                    continue
                yield path

    def _chunk_python(self, rel_path: str, source: str) -> list[Chunk]:
        """Split a Python file by top-level functions/classes, with a fallback.

        Args:
            rel_path: Repository-relative file path.
            source: Full file contents.

        Returns:
            One chunk per top-level definition; the module preamble (imports,
            constants) becomes its own chunk. Falls back to windowing if the
            file does not parse.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._chunk_windowed(rel_path, source, "code")

        lines = source.splitlines()
        chunks: list[Chunk] = []
        spans: list[tuple[int, int]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", node.lineno) or node.lineno
                spans.append((start, end))

        # Module-level preamble before the first definition.
        first_def = spans[0][0] if spans else len(lines) + 1
        if first_def > 1:
            preamble = "\n".join(lines[: first_def - 1]).strip()
            if preamble:
                chunks.append(self._make_chunk(rel_path, 1, first_def - 1, preamble, "code"))

        for start, end in spans:
            body = "\n".join(lines[start - 1 : end])
            # Very large definitions get windowed so a single chunk never explodes.
            if _count_tokens(body) > self._settings.chunk_max_tokens * 3:
                chunks.extend(self._chunk_windowed(rel_path, body, "code", line_offset=start - 1))
            else:
                chunks.append(self._make_chunk(rel_path, start, end, body, "code"))
        return chunks

    def _chunk_windowed(
        self, rel_path: str, text: str, kind: str, line_offset: int = 0
    ) -> list[Chunk]:
        """Split arbitrary text into overlapping, token-budgeted line windows.

        Args:
            rel_path: Repository-relative file path.
            text: Text to split.
            kind: Chunk category (``code``/``doc``/``other``).
            line_offset: Added to line numbers (used when re-windowing a slice).

        Returns:
            A list of windowed chunks with overlap to preserve context.
        """
        lines = text.splitlines()
        if not lines:
            return []

        max_tokens = self._settings.chunk_max_tokens
        overlap = self._settings.chunk_overlap_tokens
        chunks: list[Chunk] = []
        window: list[str] = []
        window_start = 0  # 0-based index of first line in current window

        def flush(end_idx: int) -> None:
            if not window:
                return
            body = "\n".join(window).strip()
            if body:
                chunks.append(
                    self._make_chunk(
                        rel_path,
                        window_start + 1 + line_offset,
                        end_idx + 1 + line_offset,
                        body,
                        kind,
                    )
                )

        running_tokens = 0
        for idx, line in enumerate(lines):
            line_tokens = _count_tokens(line) + 1
            if running_tokens + line_tokens > max_tokens and window:
                flush(idx - 1)
                # Build overlap tail for the next window.
                tail: list[str] = []
                tail_tokens = 0
                for prev in reversed(window):
                    t = _count_tokens(prev) + 1
                    if tail_tokens + t > overlap:
                        break
                    tail.insert(0, prev)
                    tail_tokens += t
                window = tail
                window_start = idx - len(tail)
                running_tokens = tail_tokens
            window.append(line)
            running_tokens += line_tokens
        flush(len(lines) - 1)
        return chunks

    @staticmethod
    def _make_chunk(rel_path: str, start: int, end: int, text: str, kind: str) -> Chunk:
        """Construct a :class:`Chunk` with a stable id and token count.

        Args:
            rel_path: Repository-relative file path.
            start: 1-based first line.
            end: 1-based last line.
            text: Chunk text.
            kind: Chunk category.

        Returns:
            A populated :class:`Chunk`.
        """
        return Chunk(
            id=f"{rel_path}:{start}-{end}",
            path=rel_path,
            start_line=start,
            end_line=end,
            kind=kind,  # type: ignore[arg-type]
            text=text,
            num_tokens=_count_tokens(text),
        )


class VectorIndex:
    """A FAISS inner-product index over chunk embeddings."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Build the index.

        Args:
            chunks: The chunks, parallel to ``embeddings`` rows.
            embeddings: ``(n, dim)`` L2-normalised float32 matrix.
        """
        self._chunks = chunks
        if embeddings.shape[0] == 0:
            self._index = None
            self._dim = 0
            return
        self._dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)

    def search(
        self, query_vec: np.ndarray, k: int, kind: str | None = None
    ) -> list[RetrievalHit]:
        """Return the ``k`` most similar chunks, optionally filtered by kind.

        Args:
            query_vec: 1-D normalised query embedding.
            k: Number of results to return after filtering.
            kind: If given (``code``/``doc``), restrict results to that category.

        Returns:
            A ranked list of :class:`RetrievalHit`.
        """
        if self._index is None or k <= 0:
            return []
        # Over-fetch when filtering so we still return k after the filter.
        fetch = k * 4 if kind else k
        fetch = min(fetch, len(self._chunks))
        scores, indices = self._index.search(query_vec.reshape(1, -1).astype("float32"), fetch)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if kind and chunk.kind != kind:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=float(score)))
            if len(hits) >= k:
                break
        return hits


class Retriever:
    """High-level semantic search over an indexed repository."""

    def __init__(self, settings: Settings, embedder: EmbeddingClient) -> None:
        """Initialise the retriever.

        Args:
            settings: Global configuration.
            embedder: Embedding client (for query encoding and indexing).
        """
        self._settings = settings
        self._embedder = embedder
        self._index: VectorIndex | None = None
        self._chunks: list[Chunk] = []

    def build(self, repo_root: str | os.PathLike[str]) -> int:
        """Chunk, embed, and index a repository.

        Args:
            repo_root: Absolute path to the cloned repository.

        Returns:
            The number of chunks indexed.
        """
        chunker = CodeChunker(self._settings)
        self._chunks = chunker.chunk_repo(repo_root)
        texts = [self._render_for_embedding(c) for c in self._chunks]
        embeddings = self._embedder.embed(texts, normalize=True)
        self._index = VectorIndex(self._chunks, embeddings)
        return len(self._chunks)

    def search_code(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        """Semantic search restricted to source-code chunks.

        Args:
            query: Natural-language or code query.
            k: Number of hits (defaults to ``settings.retrieval_k``).

        Returns:
            Ranked code hits.
        """
        return self._search(query, k, kind="code")

    def search_docs(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        """Semantic search restricted to documentation chunks.

        Args:
            query: Natural-language query.
            k: Number of hits (defaults to ``settings.retrieval_k``).

        Returns:
            Ranked doc hits.
        """
        return self._search(query, k, kind="doc")

    def search_all(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        """Semantic search across all chunk kinds.

        Args:
            query: Natural-language query.
            k: Number of hits (defaults to ``settings.retrieval_k``).

        Returns:
            Ranked hits regardless of kind.
        """
        return self._search(query, k, kind=None)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Look up a chunk by its id.

        Args:
            chunk_id: The chunk identifier (``path:start-end``).

        Returns:
            The chunk, or ``None`` if not found.
        """
        for chunk in self._chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def _search(self, query: str, k: int | None, kind: str | None) -> list[RetrievalHit]:
        """Shared search implementation.

        Args:
            query: Query string.
            k: Result count, or None for the configured default.
            kind: Optional kind filter.

        Returns:
            Ranked hits, or empty list if the index is not built.
        """
        if self._index is None:
            return []
        top_k = k or self._settings.retrieval_k
        query_vec = self._embedder.embed_one(query, normalize=True)
        return self._index.search(query_vec, top_k, kind=kind)

    @staticmethod
    def _render_for_embedding(chunk: Chunk) -> str:
        """Prefix a chunk with its path so the embedding captures location.

        Args:
            chunk: The chunk to render.

        Returns:
            Text to embed (``# path\n<text>``).
        """
        return f"# {chunk.path} (lines {chunk.start_line}-{chunk.end_line})\n{chunk.text}"
