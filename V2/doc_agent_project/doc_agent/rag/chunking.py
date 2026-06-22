"""Chunking strategies that turn a repository into retrievable units.

Good retrieval starts with good chunks. Code and prose want different treatment:
code is best split along symbol boundaries (functions/classes) so a retrieved
chunk is a self-contained, runnable-ish unit; prose is best split along headings
and paragraphs so a chunk is a coherent topic. This module provides both, with a
size-bounded sliding-window fallback for anything else.

Each chunk is returned as a :class:`Chunk` carrying enough metadata (path, symbol,
line span) for the agent to cite precise references in its evidence.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

# File extensions we treat as code vs. documentation.
CODE_EXTENSIONS = {".py"}
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".ipynb", ".cfg", ".toml", ".yaml", ".yml"}


@dataclass
class Chunk:
    """One retrievable unit of a repository.

    Attributes:
        text: The chunk's content (what gets embedded and shown to the model).
        path: Repository-relative file path the chunk came from.
        source_type: ``"code"`` or ``"doc"`` — drives evidence tagging.
        symbol: For code, the enclosing function/class name (else ``""``).
        start_line: 1-based first line of the chunk in its file.
        end_line: 1-based last line of the chunk in its file.
    """

    text: str
    path: str
    source_type: str
    symbol: str = ""
    start_line: int = 0
    end_line: int = 0

    @property
    def reference(self) -> str:
        """A compact, human-followable locator like ``pkg/mod.py:MyClass:40-88``."""
        ref = self.path
        if self.symbol:
            ref += f":{self.symbol}"
        if self.start_line:
            ref += f":{self.start_line}-{self.end_line}"
        return ref


def _sliding_window(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split arbitrary text into overlapping windows by line count proxy.

    Operates on characters but reports approximate line spans so even fallback
    chunks stay citable.

    Args:
        text: Raw text to split.
        chunk_size: Target characters per window.
        overlap: Characters of overlap between consecutive windows.

    Returns:
        A list of ``(chunk_text, start_line, end_line)`` tuples.
    """
    if not text.strip():
        return []
    chunks: list[tuple[str, int, int]] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        window = text[start : start + chunk_size]
        if not window.strip():
            continue
        start_line = text.count("\n", 0, start) + 1
        end_line = start_line + window.count("\n")
        chunks.append((window, start_line, end_line))
        if start + chunk_size >= len(text):
            break
    return chunks


def chunk_python(path: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk a Python file along top-level function/class boundaries.

    Uses the :mod:`ast` module to find top-level defs so each chunk is a coherent
    symbol. Oversized symbols are further split with the sliding window. Any
    module-level code outside symbols is captured as a preamble chunk. Falls back
    to plain windowing if the file does not parse (e.g. Python 2 syntax).

    Args:
        path: Repo-relative path (used for references).
        text: File contents.
        chunk_size: Target characters per chunk.
        overlap: Overlap for oversized-symbol splitting.

    Returns:
        A list of code :class:`Chunk` objects.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [
            Chunk(t, path, "code", "", s, e)
            for (t, s, e) in _sliding_window(text, chunk_size, overlap)
        ]

    lines = text.splitlines()
    chunks: list[Chunk] = []
    covered: set[int] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            covered.update(range(start, end + 1))
            body = "\n".join(lines[start - 1 : end])
            if len(body) <= chunk_size * 1.5:
                chunks.append(Chunk(body, path, "code", node.name, start, end))
            else:
                for (t, s, e) in _sliding_window(body, chunk_size, overlap):
                    chunks.append(
                        Chunk(t, path, "code", node.name, start + s - 1, start + e - 1)
                    )

    # Capture module-level lines not inside any symbol (imports, constants, etc.).
    preamble = "\n".join(
        ln for i, ln in enumerate(lines, start=1) if i not in covered
    ).strip()
    if preamble:
        chunks.append(Chunk(preamble, path, "code", "<module>", 1, len(lines)))
    return chunks


def chunk_document(path: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk prose/doc files by markdown-style sections, then by size.

    Splits on heading lines (``#`` / ``=====`` underlines are approximated by ``#``)
    so each chunk centers on one topic, then size-bounds each section.

    Args:
        path: Repo-relative path.
        text: File contents.
        chunk_size: Target characters per chunk.
        overlap: Overlap for oversized sections.

    Returns:
        A list of doc :class:`Chunk` objects.
    """
    # Greedy section split on markdown headings; keep the heading with its body.
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    chunks: list[Chunk] = []
    offset_line = 1
    for section in sections:
        if len(section) <= chunk_size:
            if section.strip():
                end = offset_line + section.count("\n")
                chunks.append(Chunk(section, path, "doc", "", offset_line, end))
        else:
            for (t, s, e) in _sliding_window(section, chunk_size, overlap):
                chunks.append(
                    Chunk(t, path, "doc", "", offset_line + s - 1, offset_line + e - 1)
                )
        offset_line += section.count("\n") + 1
    return chunks


def chunk_repository(
    root: str,
    chunk_size: int,
    overlap: int,
    max_file_bytes: int = 1_000_000,
) -> list[Chunk]:
    """Walk a repository tree and chunk every code/doc file it contains.

    Skips hidden directories, virtual environments, and obvious binary/vendored
    paths so the index stays focused on first-party source and docs.

    Args:
        root: Directory to walk (typically the cloned repo).
        chunk_size: Target characters per chunk.
        overlap: Overlap between chunks.
        max_file_bytes: Skip files larger than this to avoid indexing data blobs.

    Returns:
        A flat list of :class:`Chunk` objects across all eligible files.
    """
    ignore_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache"}
    chunks: list[Chunk] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs and not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in CODE_EXTENSIONS and ext not in DOC_EXTENSIONS:
                continue
            full = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(full) > max_file_bytes:
                    continue
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except OSError:
                continue
            rel = os.path.relpath(full, root)
            if ext in CODE_EXTENSIONS:
                chunks.extend(chunk_python(rel, text, chunk_size, overlap))
            else:
                chunks.extend(chunk_document(rel, text, chunk_size, overlap))
    return chunks
