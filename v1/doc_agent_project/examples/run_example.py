"""End-to-end example: answer a small question set against a repository.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    poetry run python examples/run_example.py

This uses the LocalExecutionBackend (real git/venv/subprocess). In the competition,
swap LocalExecutionBackend for an adapter around the organizer-provided sandbox —
the rest of the code is unchanged because both satisfy the ExecutionBackend protocol.
"""

from __future__ import annotations

import json

from doc_agent import DocAgent, LocalExecutionBackend, QuestionItem, QuestionSet, Settings


def main() -> None:
    """Build the agent, ask a mix of read-only and execution questions, print results."""
    settings = Settings(model="claude-opus-4-8", enable_rag=True, max_agent_steps=8)
    backend = LocalExecutionBackend(workdir=settings.workdir, timeout_seconds=120)
    agent = DocAgent(backend=backend, settings=settings)

    payload = QuestionSet(
        repo_url="https://github.com/psf/requests.git",
        questions=[
            QuestionItem(
                id="q1",
                question="What is the default timeout used by requests.get when none is passed?",
                requires_code_execution=False,  # answerable by reading the source/docs
            ),
            QuestionItem(
                id="q2",
                question="What status_code is returned when calling get() against httpbin /status/204?",
                requires_code_execution=True,  # must actually run a request to know
                context_hint="requests.api.get",
            ),
        ],
    )

    result = agent.answer_questions(payload)
    for ans in result.answers:
        print("=" * 70)
        print(f"Q: {ans.question_id}")
        print(f"Answer: {ans.answer}")
        print(f"Confidence: {ans.confidence} ({ans.confidence_label.value})")
        print(f"Information complete: {ans.information_complete}")
        if ans.missing_information:
            print(f"Missing: {ans.missing_information}")
        print(f"Tools used: {ans.tools_used}")
        print(f"Signals: {json.dumps(ans.signals.model_dump() if ans.signals else {}, indent=2)}")


if __name__ == "__main__":
    main()
