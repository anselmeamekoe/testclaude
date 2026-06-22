"""Prompt construction for the agent.

The prompts encode the *policy* the model follows: when to search code vs. docs,
when to execute, how to decide information is missing, and — crucially — how to
report a *calibrated* self-confidence rather than a reflexively high one. Keeping
this text in one module makes the agent's behaviour easy to audit and tune without
touching control flow.
"""

from __future__ import annotations

from .models import QuestionItem

SYSTEM_PROMPT = """\
You are a documentation-analysis agent. You answer questions about a software \
repository by reasoning over its source code and documentation, and by executing \
code when that is the only reliable way to know the answer.

Available tools:
- search_code: find relevant implementation (functions, classes, defaults).
- search_docs: find relevant documentation/usage/conceptual descriptions.
- setup_environment / run_python_file / run_notebook / execute_code: actually run \
code in the cloned repo to observe real behaviour.
- submit_answer: REQUIRED final step. You must call this exactly once with your \
answer, a calibrated confidence, and whether information was sufficient.

Operating principles:
1. Prefer EVIDENCE over recall. Do not answer from prior knowledge of popular \
libraries; ground every claim in this repo's code/docs or in execution output.
2. Choose tools by intent: implementation details -> search_code; documented \
usage/behaviour -> search_docs; behaviour that depends on runtime values, outputs, \
randomness, environment, or that you cannot determine by reading -> execute it.
3. If, after reasonable searching/execution, the needed information is genuinely \
absent, say so: set information_complete=false and explain what is missing rather \
than guessing.
4. CONFIDENCE MUST BE CALIBRATED. Report the probability that your answer is \
correct, not how fluent it sounds. Reserve high confidence (>0.8) for claims you \
verified by execution or by direct, unambiguous evidence. Use moderate confidence \
when evidence is partial. Use low confidence when you inferred or guessed. \
Overconfidence and underconfidence are both penalized.
5. Be efficient: a few well-chosen tool calls beat many redundant ones.
"""


def build_question_prompt(item: QuestionItem) -> str:
    """Build the per-question user message, injecting the routing policy.

    The ``requires_code_execution`` flag is surfaced explicitly so the model treats
    it as a strong prior: when ``True`` it is told to plan on running code and to
    flag missing information if execution is impossible; when ``False`` it is told to
    lead with retrieval and only execute if reading proves insufficient.

    Args:
        item: The question to answer (carries the binary execution flag and hints).

    Returns:
        The user-turn prompt string for this question.
    """
    if item.requires_code_execution:
        routing = (
            "This question is FLAGGED AS REQUIRING CODE EXECUTION. Plan to set up "
            "the environment and run the relevant file/notebook/snippet to verify "
            "the answer empirically. If execution is impossible or fails after a "
            "genuine attempt, set information_complete=false and lower your "
            "confidence accordingly — do not fall back to guessing from the source."
        )
    else:
        routing = (
            "This question is NOT flagged as requiring execution. Lead with "
            "search_code / search_docs. Only execute code if reading the source and "
            "docs cannot settle the answer."
        )

    hint = f"\nContext hint from the organizers: {item.context_hint}" if item.context_hint else ""

    return (
        f"Question id: {item.id}\n"
        f"Question: {item.question}\n"
        f"{routing}{hint}\n\n"
        "Work step by step using the tools, then call submit_answer exactly once."
    )


# The terminal tool's schema is defined alongside the prompts because its fields
# mirror the confidence policy described above.
SUBMIT_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The final, self-contained answer to the question.",
        },
        "verbalized_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "Your CALIBRATED probability that the answer is correct (0-1). "
                "High only if verified by execution or unambiguous evidence."
            ),
        },
        "information_complete": {
            "type": "boolean",
            "description": "False if you lacked sufficient information to answer reliably.",
        },
        "missing_information": {
            "type": "string",
            "description": "If incomplete, what specifically was missing (else empty).",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief justification grounded in the evidence you gathered.",
        },
    },
    "required": ["answer", "verbalized_confidence", "information_complete"],
}
