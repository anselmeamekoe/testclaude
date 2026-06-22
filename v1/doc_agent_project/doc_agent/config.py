"""Runtime configuration for the documentation agent.

Centralizes every knob (model name, retrieval depth, calibration weights, step
budgets) into one validated :class:`Settings` object so behaviour is reproducible
and overridable from the environment without touching code. Pydantic validation
guards against silently invalid configs (e.g. weights out of range).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Tunable parameters for the agent, RAG layer, and confidence calibrator.

    Attributes:
        model: Anthropic model id used for reasoning and tool orchestration.
            Opus is the default because tool routing benefits from the strongest
            model; swap to ``claude-sonnet-4-6`` for cheaper/faster runs.
        max_tokens: Per-call output token ceiling (the Messages API requires it).
        max_agent_steps: Hard cap on tool-use turns per question, preventing
            runaway loops on ambiguous questions.
        temperature: Sampling temperature for the orchestration calls. Kept low
            for deterministic tool routing.

        embedding_model: SentenceTransformers model id for RAG embeddings. Ignored
            if the dependency is unavailable (a deterministic hash fallback is used).
        chunk_size: Target characters per document chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        top_k: Number of chunks retrieved per search query.

        w_verbalized / w_retrieval / w_execution: Blend weights for the confidence
            signals. They need not sum to 1; the calibrator normalizes them.
        self_consistency_samples: If > 1, the agent samples this many answers and
            uses their agreement as an extra calibration signal (costs more tokens).

        enable_rag: Master switch for the FAISS retrieval layer.
        workdir: Base directory for clones, venvs and indexes.
    """

    # --- LLM ---
    model: str = Field(default="claude-opus-4-8")
    max_tokens: int = Field(default=2048, gt=0)
    max_agent_steps: int = Field(default=10, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- RAG ---
    enable_rag: bool = True
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, gt=0)

    # --- Confidence calibration ---
    w_verbalized: float = Field(default=0.45, ge=0.0)
    w_retrieval: float = Field(default=0.30, ge=0.0)
    w_execution: float = Field(default=0.25, ge=0.0)
    self_consistency_samples: int = Field(default=1, ge=1)

    # --- Filesystem ---
    workdir: str = Field(default="./.doc_agent_work")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, falling back to defaults.

        Recognized variables: ``DOC_AGENT_MODEL``, ``DOC_AGENT_MAX_STEPS``,
        ``DOC_AGENT_ENABLE_RAG`` (``"0"``/``"false"`` to disable), ``DOC_AGENT_TOPK``,
        ``DOC_AGENT_WORKDIR``. Anything unset keeps its declared default.

        Returns:
            A validated :class:`Settings` instance.
        """
        kwargs: dict = {}
        if v := os.getenv("DOC_AGENT_MODEL"):
            kwargs["model"] = v
        if v := os.getenv("DOC_AGENT_MAX_STEPS"):
            kwargs["max_agent_steps"] = int(v)
        if v := os.getenv("DOC_AGENT_TOPK"):
            kwargs["top_k"] = int(v)
        if v := os.getenv("DOC_AGENT_WORKDIR"):
            kwargs["workdir"] = v
        if v := os.getenv("DOC_AGENT_ENABLE_RAG"):
            kwargs["enable_rag"] = v.lower() not in {"0", "false", "no"}
        return cls(**kwargs)
