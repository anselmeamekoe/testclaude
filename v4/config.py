"""Runtime configuration, loaded from environment variables (or a ``.env`` file).

All settings use the ``SPARROW_`` prefix. Two *separate* OpenAI-compatible
endpoints are configurable, because the organisers impose **gpt-oss-120b** for
chat and an **OpenAI embedding model** for vectors, and these are frequently
hosted on different base URLs with different keys.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed configuration for the whole agent.

    Attributes are grouped into: chat model, embedding model, and agent
    behaviour. Every value can be overridden by an environment variable named
    ``SPARROW_<UPPERCASE_FIELD>`` or by entries in a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="SPARROW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Chat model: gpt-oss-120b behind an OpenAI-compatible API ------------
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL of the OpenAI-compatible endpoint serving gpt-oss-120b.",
    )
    llm_api_key: str = Field(default="EMPTY", description="API key for the chat endpoint.")
    llm_model: str = Field(default="gpt-oss-120b", description="Chat model identifier.")
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="gpt-oss reasoning effort. Higher = more deliberate tool use.",
    )
    temperature: float = Field(
        default=0.2,
        description="Sampling temperature for the agent loop. Kept low for determinism.",
    )

    # ---- Embedding model: an OpenAI embedding model --------------------------
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL of the OpenAI-compatible endpoint serving embeddings.",
    )
    embedding_api_key: str = Field(default="EMPTY", description="API key for the embedding endpoint.")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model identifier.",
    )
    embedding_batch_size: int = Field(default=128, description="Texts per embedding request.")

    # ---- RAG / chunking ------------------------------------------------------
    chunk_max_tokens: int = Field(default=400, description="Max tokens per non-Python chunk.")
    chunk_overlap_tokens: int = Field(default=60, description="Token overlap between window chunks.")
    retrieval_k: int = Field(default=6, description="Default number of chunks returned per search.")
    max_file_bytes: int = Field(
        default=1_500_000, description="Skip files larger than this when indexing."
    )

    # ---- Agent behaviour -----------------------------------------------------
    max_agent_steps: int = Field(
        default=12, description="Hard cap on tool-calling iterations per question."
    )
    enable_verifier: bool = Field(
        default=True,
        description="Run a second LLM pass that grades evidence sufficiency before finalising.",
    )
    enable_self_consistency: bool = Field(
        default=False,
        description="Answer each question several times and use agreement as a calibration signal.",
    )
    self_consistency_samples: int = Field(
        default=3, description="Number of independent attempts when self-consistency is enabled."
    )

    # ---- Workspace -----------------------------------------------------------
    workspace_dir: str = Field(
        default="./.sparrow_workspace",
        description="Where cloned repos and virtual environments are created.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Caching avoids re-reading the environment on every call while keeping a
    single source of truth for configuration.
    """
    return Settings()
