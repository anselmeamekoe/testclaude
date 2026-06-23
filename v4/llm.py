"""Thin wrapper around the OpenAI-compatible chat endpoint serving **gpt-oss-120b**.

The wrapper exposes exactly what the agent needs:

* :meth:`LLMClient.chat` — one tool-calling turn (returns the assistant message,
  which may contain ``tool_calls``).
* :meth:`LLMClient.complete_json` — a constrained call that returns parsed JSON,
  used by the calibration verifier.

gpt-oss exposes a ``reasoning_effort`` control. The official OpenAI Python SDK
does not have a first-class parameter for it on chat-completions, so we pass it
through ``extra_body`` — vLLM / Ollama / TGI style servers read it from there.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from sparrow_agent.config import Settings


class LLMClient:
    """Chat client for gpt-oss-120b."""

    def __init__(self, settings: Settings) -> None:
        """Create the client.

        Args:
            settings: Global configuration. Only the ``llm_*`` and
                ``reasoning_effort``/``temperature`` fields are used here.
        """
        self._settings = settings
        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Run one chat turn and return the raw assistant *message* object.

        Args:
            messages: Conversation so far, in OpenAI chat format.
            tools: Optional list of tool/function JSON schemas the model may call.
            tool_choice: ``"auto"``, ``"none"``, or a forced
                ``{"type": "function", "function": {"name": ...}}`` selector.
            temperature: Override the configured temperature for this call.

        Returns:
            The ``message`` field of the first choice. It has ``.content`` and,
            when the model decides to act, ``.tool_calls``.
        """
        kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": self._settings.temperature if temperature is None else temperature,
            # gpt-oss reads this; harmless on servers that ignore unknown keys.
            "extra_body": {"reasoning_effort": self._settings.reasoning_effort},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message

    def complete_json(self, system: str, user: str, temperature: float = 0.0) -> dict:
        """Ask the model for a single JSON object and return it parsed.

        Robust to models that wrap JSON in ``` fences. Used by the verifier in
        the calibration engine.

        Args:
            system: System instruction (should demand JSON-only output).
            user: User content.
            temperature: Sampling temperature (default deterministic).

        Returns:
            The parsed JSON object, or ``{}`` if parsing fails.
        """
        message = self.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return _safe_json(message.content or "")


def _safe_json(text: str) -> dict:
    """Parse JSON that may be wrapped in Markdown fences or have leading prose.

    Args:
        text: Raw model output.

    Returns:
        Parsed object, or ``{}`` when no valid JSON object is found.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # Drop an optional language tag on the first line (e.g. "json").
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}
