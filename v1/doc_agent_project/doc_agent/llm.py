"""LLM client abstraction and the Anthropic tool-use implementation.

The agent talks to the model through the small :class:`LLMClient` protocol so the
orchestration logic stays provider-agnostic and unit-testable (you can drop in a
fake client in tests). The concrete :class:`AnthropicLLM` implements a single
round of the Messages API tool-use protocol: send messages + tool schemas, get
back either text or ``tool_use`` blocks.

The agentic *loop* (run the requested tool, append the ``tool_result``, call
again) lives in :mod:`doc_agent.agent`; this module only wraps one API turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class LLMTurn:
    """The model's reply for one Messages-API turn, normalized for the agent loop.

    Attributes:
        text: Concatenated text the model produced this turn (may be empty when it
            only requested tools).
        tool_calls: One entry per ``tool_use`` block: ``{"id", "name", "input"}``.
            Empty when the model is done and just produced a final text answer.
        stop_reason: Raw stop reason from the API (``"tool_use"``, ``"end_turn"`` …).
        raw_assistant_content: The exact assistant ``content`` blocks, so the caller
            can append them verbatim to the running message history (required for a
            valid tool-use transcript).
    """

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    raw_assistant_content: list[dict[str, Any]] = field(default_factory=list)


class LLMClient(Protocol):
    """Minimal interface the agent depends on (enables fakes/mocks in tests)."""

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tool_choice: Optional[dict[str, Any]] = None,
    ) -> LLMTurn:
        """Run a single tool-use-enabled completion turn.

        Args:
            system: Top-level system prompt (Anthropic passes this outside ``messages``).
            messages: Running conversation history in Messages-API format.
            tools: Tool schemas (``name``/``description``/``input_schema``).
            max_tokens: Output token ceiling for this turn.
            temperature: Sampling temperature.
            tool_choice: Optional forcing of a particular tool, e.g.
                ``{"type": "tool", "name": "submit_answer"}``.

        Returns:
            A normalized :class:`LLMTurn`.
        """
        ...


class AnthropicLLM:
    """Concrete :class:`LLMClient` backed by the official ``anthropic`` SDK.

    The client is created lazily so importing this module never requires the SDK
    or an API key — handy for environments where only the data models are used.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Store credentials; defer SDK construction until the first call.

        Args:
            api_key: Anthropic API key. If ``None`` the SDK reads
                ``ANTHROPIC_API_KEY`` from the environment.
        """
        self._api_key = api_key
        self._client = None  # built on first use

    def _ensure_client(self):
        """Instantiate and cache the ``anthropic.Anthropic`` client on first use.

        Raises:
            RuntimeError: If the ``anthropic`` package is not installed, with a
                hint to run ``poetry add anthropic``.
        """
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "The 'anthropic' package is required. Install it with "
                    "`poetry add anthropic`."
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tool_choice: Optional[dict[str, Any]] = None,
    ) -> LLMTurn:
        """Call ``messages.create`` once and normalize the response.

        Splits the returned content blocks into text vs. ``tool_use`` requests and
        preserves the raw assistant content so the agent can append it to history
        before sending tool results back (a requirement of the tool-use protocol).

        See :meth:`LLMClient.complete` for argument semantics.
        """
        client = self._ensure_client()
        kwargs: dict[str, Any] = dict(
            model=self._model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        raw_content: list[dict[str, Any]] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
                raw_content.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                call = {"id": block.id, "name": block.name, "input": block.input}
                tool_calls.append(call)
                raw_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        return LLMTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw_assistant_content=raw_content,
        )

    # The model id is injected by the agent (from Settings) to keep this class
    # free of configuration coupling.
    _model: str = "claude-opus-4-8"

    def with_model(self, model: str) -> "AnthropicLLM":
        """Return self after binding the model id used on subsequent calls.

        Args:
            model: Anthropic model id (e.g. ``"claude-opus-4-8"``).

        Returns:
            This instance (fluent style), now configured with ``model``.
        """
        self._model = model
        return self
