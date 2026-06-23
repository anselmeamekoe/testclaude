"""Thin wrapper around the Azure-hosted GPT-OSS-120 chat model and embeddings.

The organizers impose **GPT-OSS-120 on Azure**. In practice that model is
reached either through the native ``AzureOpenAI`` surface or an OpenAI-compatible
gateway; this module supports both behind one interface so the rest of the code
never has to care which is in use.

Only two capabilities are exposed:

* :meth:`LLMClient.chat` — a chat-completions call that supports tool/function
  calling (the backbone of the agent loop) and can sample multiple choices.
* :meth:`LLMClient.embed` — batch embeddings for the FAISS/Qdrant retrievers.
"""

from __future__ import annotations

from typing import Any

from openai import AzureOpenAI, OpenAI

from .config import Settings


class LLMClient:
    """Unified client for GPT-OSS-120 chat + embeddings.

    Args:
        settings: Application settings carrying endpoint, keys and deployment
            names. When ``settings.use_openai_compatible_base_url`` is True the
            plain :class:`openai.OpenAI` client is used against
            ``settings.openai_base_url``; otherwise the native
            :class:`openai.AzureOpenAI` client is used.
    """

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        if settings.use_openai_compatible_base_url:
            self._client: OpenAI | AzureOpenAI = OpenAI(
                base_url=settings.openai_base_url,
                api_key=settings.azure_api_key,
                timeout=settings.request_timeout_s,
            )
        else:
            self._client = AzureOpenAI(
                azure_endpoint=settings.azure_endpoint,
                api_key=settings.azure_api_key,
                api_version=settings.azure_api_version,
                timeout=settings.request_timeout_s,
            )

    # ------------------------------------------------------------------ #
    # Chat                                                               #
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float | None = None,
        n: int = 1,
    ) -> Any:
        """Call the GPT-OSS-120 chat endpoint.

        Args:
            messages: OpenAI-style chat messages.
            tools: Optional list of tool/function schemas to expose for this call.
            tool_choice: ``"auto"``, ``"none"``, or a forced-tool spec.
            temperature: Override the configured default temperature.
            n: Number of independent completions to sample (used for the
                self-consistency confidence signal).

        Returns:
            The raw OpenAI ``ChatCompletion`` response. Callers read
            ``response.choices[i].message`` and ``...message.tool_calls``.
        """
        kwargs: dict[str, Any] = {
            "model": self._s.chat_deployment,
            "messages": messages,
            "temperature": self._s.chat_temperature if temperature is None else temperature,
            "n": n,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return self._client.chat.completions.create(**kwargs)

    # ------------------------------------------------------------------ #
    # Embeddings                                                         #
    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the configured embeddings deployment.

        Args:
            texts: Strings to embed. Empty strings are replaced with a single
                space so the API never rejects the batch.

        Returns:
            A list of embedding vectors, aligned with ``texts``.
        """
        safe = [t if t.strip() else " " for t in texts]
        resp = self._client.embeddings.create(
            model=self._s.embedding_deployment,
            input=safe,
        )
        return [item.embedding for item in resp.data]
