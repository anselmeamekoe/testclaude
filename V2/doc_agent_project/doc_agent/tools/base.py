"""Tool framework: a common interface and a registry that speaks Anthropic schemas.

Every capability the agent can invoke — searching code, searching docs, cloning a
repo, running a file, submitting the final answer — is a :class:`Tool`. Tools
declare a JSON-schema for their inputs so they can be advertised to the model in
the exact ``{"name", "description", "input_schema"}`` shape the Messages API
expects, and they return a uniform :class:`ToolResult` the agent loop can feed
back as a ``tool_result`` block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..models import Evidence


@dataclass
class ToolResult:
    """Uniform return type for every tool invocation.

    Attributes:
        content: Human/model-readable text summarizing what the tool did or found.
            This is what gets sent back to the model as the ``tool_result``.
        evidence: Structured evidence the tool produced (retrieved chunks, exec
            output), accumulated by the agent for the final answer and calibration.
        success: Whether the tool ran without error (distinct from "found nothing").
        metadata: Free-form extra data (e.g. raw execution result) for callers.
    """

    content: str
    evidence: list[Evidence] = field(default_factory=list)
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """A single callable capability advertised to the model.

    Attributes:
        name: Unique tool name (also the key the model uses to call it).
        description: When/why the model should use this tool. Written for the model,
            so it should be specific about preconditions and outputs.
        input_schema: JSON Schema (object) describing the tool's arguments.
        handler: Python callable ``(**input) -> ToolResult`` that performs the work.
        terminal: If ``True``, calling this tool ends the agent loop (used by
            ``submit_answer``).
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., ToolResult]
    terminal: bool = False

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Render this tool in the Messages-API tool format.

        Returns:
            A dict with ``name``, ``description`` and ``input_schema`` keys.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(self, **kwargs: Any) -> ToolResult:
        """Invoke the underlying handler, normalizing exceptions into a result.

        Returning a failed :class:`ToolResult` (rather than raising) lets the agent
        observe the error and recover (retry, switch tools, or lower confidence)
        instead of crashing the whole batch.

        Returns:
            The handler's :class:`ToolResult`, or a failure result on exception.
        """
        try:
            return self.handler(**kwargs)
        except Exception as exc:  # defensive: tools must never crash the loop
            return ToolResult(
                content=f"Tool '{self.name}' raised an error: {exc!r}",
                success=False,
            )


class ToolRegistry:
    """An ordered collection of tools exposed to the model for one question.

    The registry can be tailored per question (e.g. omit execution tools when the
    question is flagged read-only) and knows how to emit the combined schema list
    and dispatch a call by name.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add (or replace) a tool by name.

        Args:
            tool: The tool to register.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name, or ``None`` if absent."""
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in registration order for the API call."""
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        """Return the registered tool names."""
        return list(self._tools.keys())
