"""LLM client abstraction with Anthropic and OpenAI-compatible implementations.

The agent talks to a model through the small :class:`LLMClient` protocol so the
orchestration logic is provider-agnostic. Two concrete clients are provided:

* :class:`AnthropicLLM` — the Anthropic Messages API (tool use).
* :class:`OpenAILLM` — any OpenAI-*compatible* Chat Completions endpoint. This is
  what you point at a self-hosted open model such as **gpt-oss-120b** (served by
  vLLM/Ollama/Together/etc.) simply by setting ``base_url`` and ``model``.

To stay provider-neutral, the agent maintains a small neutral transcript made of
:class:`HistoryUser`, :class:`HistoryAssistant`, and :class:`HistoryToolResults`
records. Each client translates that transcript into its own wire format at call
time, so the agent loop never contains provider-specific message plumbing.

Robustness note (gpt-oss): tool-call parsing on the OpenAI Chat Completions
endpoint is inconsistent across open-model deployments — some emit the call into
``message.content`` as a JSON list instead of populating ``tool_calls``. The
OpenAI client therefore falls back to parsing tool calls out of the text content.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Union


# --------------------------------------------------------------------------- #
# Neutral transcript records (provider-independent conversation history)        #
# --------------------------------------------------------------------------- #
@dataclass
class HistoryUser:
    """A user-authored turn (the question prompt, or any user-side text).

    Attributes:
        content: The user text for this turn.
    """

    content: str


@dataclass
class HistoryAssistant:
    """An assistant turn: optional text plus any tool calls it requested.

    Attributes:
        text: Free-text the model produced (may be empty when it only called tools).
        tool_calls: List of ``{"id", "name", "input"}`` the model requested.
    """

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HistoryToolResults:
    """The results of executing the tool calls from the preceding assistant turn.

    Attributes:
        results: List of ``{"id", "content", "is_error"}``, one per executed call.
    """

    results: list[dict[str, Any]] = field(default_factory=list)


# A conversation is an ordered list of these neutral records.
HistoryItem = Union[HistoryUser, HistoryAssistant, HistoryToolResults]


@dataclass
class LLMTurn:
    """Normalized result of one model turn, consumed by the agent loop.

    Attributes:
        text: Concatenated assistant text for this turn.
        tool_calls: ``[{"id", "name", "input"}]`` the model requested (empty when
            it produced only a final text answer).
        stop_reason: Raw stop/finish reason from the provider (informational).
    """

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None


class LLMClient(Protocol):
    """Provider-neutral interface the agent depends on (enables fakes in tests)."""

    def complete(
        self,
        *,
        system: str,
        history: list[HistoryItem],
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        force_tool: Optional[str] = None,
    ) -> LLMTurn:
        """Run a single tool-use-enabled turn against the model.

        Args:
            system: System prompt text.
            history: Neutral transcript so far.
            tools: Tool specs as ``{"name", "description", "input_schema"}`` (the
                client converts these to its own wire format).
            max_tokens: Output token ceiling for this turn.
            temperature: Sampling temperature.
            force_tool: If set, force the model to call this specific tool (used to
                guarantee the terminal ``submit_answer`` is produced).

        Returns:
            A normalized :class:`LLMTurn`.
        """
        ...


# --------------------------------------------------------------------------- #
# Anthropic                                                                     #
# --------------------------------------------------------------------------- #
class AnthropicLLM:
    """:class:`LLMClient` backed by the official ``anthropic`` SDK (Messages API)."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Store credentials; defer SDK construction until first use.

        Args:
            api_key: Anthropic API key, or ``None`` to read ``ANTHROPIC_API_KEY``.
        """
        self._api_key = api_key
        self._client = None
        self._model = "claude-opus-4-8"

    def with_model(self, model: str) -> "AnthropicLLM":
        """Bind the model id used on subsequent calls and return self.

        Args:
            model: Anthropic model id.

        Returns:
            This instance (fluent style).
        """
        self._model = model
        return self

    def _ensure_client(self):
        """Lazily build and cache the ``anthropic.Anthropic`` client.

        Raises:
            RuntimeError: If the ``anthropic`` package is not installed.
        """
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "The 'anthropic' package is required for AnthropicLLM. "
                    "Install it with `poetry add anthropic`."
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _to_messages(history: list[HistoryItem]) -> list[dict[str, Any]]:
        """Convert the neutral transcript into Anthropic ``messages`` blocks.

        Args:
            history: Neutral transcript.

        Returns:
            A list of Anthropic-format message dicts.
        """
        messages: list[dict[str, Any]] = []
        for item in history:
            if isinstance(item, HistoryUser):
                messages.append({"role": "user", "content": item.content})
            elif isinstance(item, HistoryAssistant):
                content: list[dict[str, Any]] = []
                if item.text:
                    content.append({"type": "text", "text": item.text})
                for call in item.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["name"],
                            "input": call.get("input", {}),
                        }
                    )
                messages.append({"role": "assistant", "content": content})
            elif isinstance(item, HistoryToolResults):
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["id"],
                        "content": r["content"],
                        "is_error": bool(r.get("is_error", False)),
                    }
                    for r in item.results
                ]
                messages.append({"role": "user", "content": blocks})
        return messages

    def complete(
        self,
        *,
        system: str,
        history: list[HistoryItem],
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        force_tool: Optional[str] = None,
    ) -> LLMTurn:
        """Call Anthropic ``messages.create`` once and normalize the response.

        See :meth:`LLMClient.complete` for argument semantics. ``force_tool`` maps to
        Anthropic's ``tool_choice={"type": "tool", "name": ...}``.
        """
        client = self._ensure_client()
        kwargs: dict[str, Any] = dict(
            model=self._model,
            system=system,
            messages=self._to_messages(history),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools  # neutral spec == Anthropic schema
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

        response = client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        return LLMTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
        )


# --------------------------------------------------------------------------- #
# OpenAI-compatible (gpt-oss-120b, vLLM, Ollama, Together, Groq, ...)            #
# --------------------------------------------------------------------------- #
class OpenAILLM:
    """:class:`LLMClient` for any OpenAI-compatible Chat Completions endpoint.

    Point this at an open-source model by setting ``base_url`` and ``model``; the
    same agentic loop then runs unchanged. Tool calling uses the standard OpenAI
    ``tools``/``tool_choice`` shape, with a content-parsing fallback for servers
    whose tool parser emits calls into ``message.content`` instead of ``tool_calls``.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Store connection details; defer SDK construction until first use.

        Args:
            model: Model id as the server expects it (e.g. ``"openai/gpt-oss-120b"``).
            api_key: API key/token. Many self-hosted servers accept any non-empty
                string; defaults to ``OPENAI_API_KEY`` or a placeholder.
            base_url: OpenAI-compatible base URL, e.g.
                ``"http://my-host:8000/v1"``. ``None`` uses the OpenAI default.
        """
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def with_model(self, model: str) -> "OpenAILLM":
        """Bind the model id and return self (fluent, mirrors AnthropicLLM)."""
        self._model = model
        return self

    def _ensure_client(self):
        """Lazily build and cache the ``openai.OpenAI`` client.

        Raises:
            RuntimeError: If the ``openai`` package is not installed.
        """
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "The 'openai' package is required for OpenAILLM. Install it "
                    "with `poetry add openai`."
                ) from exc
            # Self-hosted servers often require a non-empty key even if unused.
            key = self._api_key or "EMPTY"
            self._client = OpenAI(api_key=key, base_url=self._base_url)
        return self._client

    @staticmethod
    def _tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Wrap neutral tool specs into OpenAI ``function`` tool definitions.

        Args:
            tools: Neutral ``{"name", "description", "input_schema"}`` specs.

        Returns:
            OpenAI-format tool definitions.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    def _to_messages(self, system: str, history: list[HistoryItem]) -> list[dict[str, Any]]:
        """Convert the neutral transcript into OpenAI chat ``messages``.

        Args:
            system: System prompt.
            history: Neutral transcript.

        Returns:
            OpenAI-format message list (system first, tool results as ``role=tool``).
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for item in history:
            if isinstance(item, HistoryUser):
                messages.append({"role": "user", "content": item.content})
            elif isinstance(item, HistoryAssistant):
                msg: dict[str, Any] = {"role": "assistant", "content": item.text or ""}
                if item.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("input", {})),
                            },
                        }
                        for call in item.tool_calls
                    ]
                messages.append(msg)
            elif isinstance(item, HistoryToolResults):
                for r in item.results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": r["id"],
                            "content": r["content"],
                        }
                    )
        return messages

    @staticmethod
    def _parse_tool_calls_from_content(text: str) -> list[dict[str, Any]]:
        """Best-effort recovery of tool calls embedded in plain text content.

        Some OpenAI-compatible servers (notably certain gpt-oss tool parsers) put
        the tool call into ``message.content`` as a JSON object or list of objects
        like ``[{"name": ..., "parameters"|"arguments": {...}}]`` instead of the
        structured ``tool_calls`` field. This parses that out so the agent loop can
        proceed regardless of server quirks.

        Args:
            text: The assistant message content.

        Returns:
            A list of ``{"id", "name", "input"}`` calls (empty if none parseable).
        """
        if not text:
            return []
        snippet = text.strip()
        start = snippet.find("[")
        brace = snippet.find("{")
        if start == -1 or (brace != -1 and brace < start):
            start = brace
        end = max(snippet.rfind("]"), snippet.rfind("}"))
        if start == -1 or end == -1 or end < start:
            return []
        try:
            data = json.loads(snippet[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return []

        items = data if isinstance(data, list) else [data]
        calls: list[dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict) or "name" not in it:
                continue
            args = it.get("parameters", it.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "name": it["name"], "input": args})
        return calls

    def _create(self, kwargs: dict[str, Any], force_tool: Optional[str]):
        """Call ``chat.completions.create`` with graceful ``tool_choice`` fallback.

        Named/forced tool choice is not honored uniformly by every OpenAI-compatible
        server. When forcing a tool, this tries the named form, then ``"required"``,
        then ``"auto"``, so a strict server can't break the loop.

        Args:
            kwargs: Base request kwargs (without ``tool_choice``).
            force_tool: Tool name to force, or ``None`` for auto.

        Returns:
            The provider response object.
        """
        client = self._ensure_client()
        if not force_tool:
            return client.chat.completions.create(**kwargs, tool_choice="auto")
        for choice in (
            {"type": "function", "function": {"name": force_tool}},
            "required",
            "auto",
        ):
            try:
                return client.chat.completions.create(**kwargs, tool_choice=choice)
            except Exception:  # pragma: no cover - server dependent
                continue
        # Last resort: no tool_choice at all.
        return client.chat.completions.create(**kwargs)

    def complete(
        self,
        *,
        system: str,
        history: list[HistoryItem],
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        force_tool: Optional[str] = None,
    ) -> LLMTurn:
        """Call the OpenAI-compatible endpoint once and normalize the response.

        Parses structured ``tool_calls`` when present, otherwise attempts to recover
        them from the text content. See :meth:`LLMClient.complete` for arguments.
        """
        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=self._to_messages(system, history),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = self._tools_to_openai(tools)

        response = self._create(kwargs, force_tool)
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""

        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(message, "tool_calls", None) or []:
            raw_args = tc.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                parsed = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "input": parsed})

        # Fallback: recover tool calls emitted into content by quirky parsers.
        if not tool_calls:
            tool_calls = self._parse_tool_calls_from_content(text)
            if tool_calls:
                text = ""  # the content was the call payload, not a real message

        return LLMTurn(
            text=text,
            tool_calls=tool_calls,
            stop_reason=getattr(choice, "finish_reason", None),
        )


# --------------------------------------------------------------------------- #
# Factory                                                                       #
# --------------------------------------------------------------------------- #
def build_llm(settings) -> LLMClient:
    """Construct the right :class:`LLMClient` from :class:`~doc_agent.config.Settings`.

    Selects the provider via ``settings.llm_provider`` (``"anthropic"`` or
    ``"openai"``). For the OpenAI provider, ``settings.base_url`` lets you target a
    self-hosted open model such as gpt-oss-120b.

    Args:
        settings: The active settings object.

    Returns:
        A ready-to-use client bound to the configured model.
    """
    if settings.llm_provider == "openai":
        return OpenAILLM(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
    return AnthropicLLM(api_key=settings.api_key).with_model(settings.model)
