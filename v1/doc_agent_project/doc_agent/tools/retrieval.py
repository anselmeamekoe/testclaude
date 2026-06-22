"""Retrieval tools: let the agent search code and docs over the FAISS index.

These tools turn the RAG layer into two model-callable capabilities — one biased
toward source code, one toward documentation — each returning the matched chunks
both as readable text (for the model) and as structured :class:`Evidence` (for
confidence calibration and final citations).

Splitting "search code" from "search docs" is deliberate: it lets the model express
*intent* ("I need the implementation" vs. "I need the documented behaviour"), and
it lets us filter results by source type so each tool stays high-signal.
"""

from __future__ import annotations

from ..models import Evidence, SourceType
from ..rag.index import VectorIndex
from .base import Tool, ToolResult

# How many characters of each retrieved chunk to show the model inline.
_SNIPPET_CHARS = 800


def _format_results(retrieved, source_filter: str | None) -> ToolResult:
    """Render retrieved chunks into a :class:`ToolResult` with evidence.

    Args:
        retrieved: List of :class:`~doc_agent.rag.index.RetrievedChunk`.
        source_filter: If given (``"code"``/``"doc"``), keep only chunks of that
            source type so each tool returns focused results.

    Returns:
        A :class:`ToolResult` whose content lists the matches and whose evidence
        captures each match with its relevance score.
    """
    kept = [r for r in retrieved if source_filter is None or r.chunk.source_type == source_filter]
    if not kept:
        return ToolResult(
            content="No relevant results found for this query.",
            success=True,
        )

    lines: list[str] = []
    evidence: list[Evidence] = []
    for i, r in enumerate(kept, start=1):
        snippet = r.chunk.text[:_SNIPPET_CHARS]
        lines.append(
            f"[{i}] {r.chunk.reference} (relevance={r.score:.2f})\n{snippet}"
        )
        evidence.append(
            Evidence(
                source_type=(
                    SourceType.CODE if r.chunk.source_type == "code" else SourceType.DOC
                ),
                reference=r.chunk.reference,
                snippet=snippet,
                score=r.score,
            )
        )
    return ToolResult(content="\n\n".join(lines), evidence=evidence, success=True)


def build_retrieval_tools(index: VectorIndex, top_k: int) -> list[Tool]:
    """Create the ``search_code`` and ``search_docs`` tools over ``index``.

    Args:
        index: A built :class:`VectorIndex` covering the repository.
        top_k: Number of chunks to retrieve per query.

    Returns:
        A list with the two retrieval :class:`Tool` objects. If the index is empty
        the tools still exist but will report no results.
    """

    def search_code(query: str) -> ToolResult:
        """Semantic search restricted to source-code chunks."""
        return _format_results(index.search(query, top_k), source_filter="code")

    def search_docs(query: str) -> ToolResult:
        """Semantic search restricted to documentation/prose chunks."""
        return _format_results(index.search(query, top_k), source_filter="doc")

    return [
        Tool(
            name="search_code",
            description=(
                "Search the repository's SOURCE CODE for relevant functions, "
                "classes, or implementation details. Use when the question is "
                "about how something is implemented, signatures, defaults, or "
                "internal behaviour."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for (concept, symbol, behaviour).",
                    }
                },
                "required": ["query"],
            },
            handler=search_code,
        ),
        Tool(
            name="search_docs",
            description=(
                "Search the repository's DOCUMENTATION (README, markdown, rst, "
                "config, notebooks' prose) for explanations, usage, and intended "
                "behaviour. Use when the question is about documented usage, setup, "
                "or conceptual descriptions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for in the docs.",
                    }
                },
                "required": ["query"],
            },
            handler=search_docs,
        ),
    ]
