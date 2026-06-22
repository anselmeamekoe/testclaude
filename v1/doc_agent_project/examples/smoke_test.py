"""Offline smoke test of the full pipeline using a fake LLM (no API key needed).

This exercises chunking -> indexing -> retrieval -> the tool-use loop -> calibration
without calling Anthropic, so it can run in CI. It scripts a fake model that searches
code, executes a snippet, then submits an answer.

Run with:
    poetry run python examples/smoke_test.py
"""

from __future__ import annotations

import os
import tempfile

from doc_agent import DocAgent, LocalExecutionBackend, QuestionItem, QuestionSet, Settings
from doc_agent.llm import LLMTurn


class FakeLLM:
    """A deterministic stand-in for the LLM that replays a fixed tool-call script.

    Each call to :meth:`complete` returns the next scripted turn, letting us drive
    the agent loop through search -> execute -> submit_answer paths deterministically.
    """

    def __init__(self) -> None:
        """Initialize the scripted turn counter."""
        self._step = 0

    def with_model(self, model: str) -> "FakeLLM":
        """No-op model binding to match the real client's fluent API."""
        return self

    def complete(self, *, system, messages, tools, max_tokens, temperature, tool_choice=None):
        """Return the next scripted :class:`LLMTurn` regardless of inputs."""
        self._step += 1
        if self._step == 1:
            return LLMTurn(
                text="Let me search the code.",
                tool_calls=[{"id": "t1", "name": "search_code", "input": {"query": "add"}}],
                stop_reason="tool_use",
                raw_assistant_content=[
                    {"type": "text", "text": "Let me search the code."},
                    {"type": "tool_use", "id": "t1", "name": "search_code", "input": {"query": "add"}},
                ],
            )
        if self._step == 2:
            return LLMTurn(
                text="Now verify by running it.",
                tool_calls=[{
                    "id": "t2", "name": "execute_code",
                    "input": {"code": "from calc import add; print(add(2, 3))"},
                }],
                stop_reason="tool_use",
                raw_assistant_content=[
                    {"type": "text", "text": "Now verify by running it."},
                    {"type": "tool_use", "id": "t2", "name": "execute_code",
                     "input": {"code": "from calc import add; print(add(2, 3))"}},
                ],
            )
        return LLMTurn(
            text="",
            tool_calls=[{
                "id": "t3", "name": "submit_answer",
                "input": {
                    "answer": "add(2, 3) returns 5.",
                    "verbalized_confidence": 0.9,
                    "information_complete": True,
                    "reasoning": "Confirmed by executing add(2, 3).",
                },
            }],
            stop_reason="tool_use",
            raw_assistant_content=[
                {"type": "tool_use", "id": "t3", "name": "submit_answer", "input": {
                    "answer": "add(2, 3) returns 5.",
                    "verbalized_confidence": 0.9,
                    "information_complete": True,
                    "reasoning": "Confirmed by executing add(2, 3).",
                }},
            ],
        )


def main() -> None:
    """Create a tiny local repo, point the agent at it, and assert it answers."""
    repo = tempfile.mkdtemp(prefix="fake_repo_")
    with open(os.path.join(repo, "calc.py"), "w") as fh:
        fh.write("def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a + b\n")

    class InProcessBackend(LocalExecutionBackend):
        """LocalExecutionBackend that 'clones' by returning our temp repo as-is."""

        def clone_repo(self, repo_url: str) -> str:
            return repo

    settings = Settings(enable_rag=True, max_agent_steps=6)
    agent = DocAgent(backend=InProcessBackend(workdir=repo), settings=settings, llm=FakeLLM())

    result = agent.answer_questions(QuestionSet(
        repo_url="file://local",
        questions=[QuestionItem(id="q1", question="What does add(2,3) return?",
                                requires_code_execution=True)],
    ))

    ans = result.answers[0]
    print("Answer:", ans.answer)
    print("Confidence:", ans.confidence, ans.confidence_label.value)
    print("Tools used:", ans.tools_used)
    assert "5" in ans.answer
    assert ans.confidence > 0.7  # execution-verified -> high but calibrated
    assert "execute_code" in ans.tools_used
    print("\nSMOKE TEST PASSED ✔")


if __name__ == "__main__":
    main()
