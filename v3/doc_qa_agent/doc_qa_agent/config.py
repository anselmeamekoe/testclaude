"""Central configuration for the documentation-QA agent.

All runtime configuration (Azure endpoints, deployment names, retrieval
hyper-parameters, agent limits) is expressed as a single Pydantic ``Settings``
object so that the rest of the codebase never reads ``os.environ`` directly.

Environment variables are loaded from the process environment and, if present,
from a local ``.env`` file. The organizers impose **GPT-OSS-120 served on
Azure / an OpenAI-compatible gateway**, so the LLM section is modelled to cover
both the native ``AzureOpenAI`` style and a generic OpenAI-compatible base URL.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings, populated from the environment.

    Attributes:
        azure_endpoint: Base endpoint of the Azure deployment that serves the
            GPT-OSS-120 chat model (e.g. ``https://my-resource.openai.azure.com``
            or an ``https://...services.ai.azure.com`` Foundry endpoint).
        azure_api_key: API key / token for the Azure endpoint.
        azure_api_version: Azure OpenAI REST API version.
        chat_deployment: Deployment (model) name for GPT-OSS-120.
        embedding_deployment: Deployment name for the embeddings model used by
            the FAISS/Qdrant retrievers.
        use_openai_compatible_base_url: When True, talk to ``openai_base_url``
            with the plain OpenAI client instead of the ``AzureOpenAI`` client.
            Useful when GPT-OSS-120 is exposed behind an OpenAI-compatible
            gateway rather than the native Azure surface.
        openai_base_url: OpenAI-compatible base URL (only used when the flag
            above is True).
        request_timeout_s: Per-request timeout for LLM calls.
        chat_temperature: Sampling temperature for the *answering* pass.
        consistency_samples: Number of high-temperature samples drawn for the
            self-consistency confidence signal (set 0 to disable).
        consistency_temperature: Temperature used for those samples.
        embedding_dim: Dimensionality of the embedding model (used to size the
            fallback hashing embedder and to validate FAISS indices).
        chunk_tokens: Target token length of each retrieval chunk.
        chunk_overlap_tokens: Token overlap between consecutive chunks.
        top_k: Default number of chunks returned by a retrieval call.
        max_agent_iterations: Hard cap on tool-use loop iterations per question.
        gitlab_token: Default GitLab token for cloning (may be overridden per call).
        sparrow_token: Token for the imposed Qdrant ("sparrow") storage backend.
        workdir: Scratch directory for clones, venvs and indices.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCQA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Azure / GPT-OSS-120 ---
    azure_endpoint: str = Field(default="https://example.openai.azure.com")
    azure_api_key: str = Field(default="")
    azure_api_version: str = Field(default="2024-08-01-preview")
    chat_deployment: str = Field(default="gpt-oss-120b")
    embedding_deployment: str = Field(default="text-embedding-3-large")

    use_openai_compatible_base_url: bool = Field(default=False)
    openai_base_url: str = Field(default="")

    request_timeout_s: float = Field(default=120.0)
    chat_temperature: float = Field(default=0.1)

    # --- Confidence / self-consistency ---
    consistency_samples: int = Field(default=3, ge=0, le=10)
    consistency_temperature: float = Field(default=0.7)

    # --- Retrieval ---
    embedding_dim: int = Field(default=3072)
    chunk_tokens: int = Field(default=400)
    chunk_overlap_tokens: int = Field(default=60)
    top_k: int = Field(default=6)

    # --- Agent loop ---
    max_agent_iterations: int = Field(default=8, ge=1)

    # --- Secrets / infra ---
    gitlab_token: str = Field(default="")
    sparrow_token: str = Field(default="")
    workdir: str = Field(default="/tmp/docqa_workdir")


def get_settings() -> Settings:
    """Return a freshly-loaded :class:`Settings` instance.

    Kept as a function (rather than a module-level singleton) so tests can patch
    the environment and obtain settings reflecting those changes.
    """
    return Settings()
