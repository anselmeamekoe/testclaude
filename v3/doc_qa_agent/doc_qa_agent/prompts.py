"""Prompts and tool schemas that drive the agentic reasoning loop.

The system prompt teaches the model the *decision policy* the challenge rewards:
search docs first for conceptual questions, search code for "how/where is X
implemented", execute code only when behaviour must be observed, and — crucially
— admit when information is missing instead of guessing. It also instructs the
model on honest verbalized-confidence reporting, which feeds the calibrator.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a documentation question-answering agent operating over a software \
repository. You answer by gathering evidence with tools, then giving a precise, \
grounded answer.

DECISION POLICY — choose tools deliberately, do not over-search:
- Conceptual / "what does X mean", "how do I use Y" -> search_docs first.
- "Where/how is X implemented", "what are the arguments of Z" -> search_code.
- Behaviour that must be observed (actual outputs, computed values, runtime \
errors, versions, shapes) -> execute code, but ONLY if execution tools are \
available to you in this turn.
- Read a specific file with read_file when search points you at it and you need \
full context.
- Stop searching once you can answer; extra calls waste budget.

HONESTY AND CALIBRATION — this is graded:
- If the evidence does not actually answer the question, set answerable=false \
and say what is missing. Do NOT fabricate.
- Report verbalized_confidence as your true probability the answer is correct \
(0-1). Be well-calibrated: ~0.9 only when evidence is direct and unambiguous; \
~0.5 when plausibly inferred; <0.3 when guessing. Both over- and under-\
confidence are penalised.
- Ground every claim in evidence you actually retrieved or executed; list the \
references you used in key_evidence.

When you are done, call the `finish` tool with your final structured answer. \
Never write the final answer as free text — always use `finish`.
"""


def build_tool_schemas(allow_execution: bool) -> list[dict]:
    """Return the JSON tool schemas exposed to the model for a question.

    The ``requires_code_execution`` flag on the input :class:`QuestionSet`
    controls ``allow_execution``: when False, the execution tools are omitted
    entirely so the model cannot choose to run code unnecessarily.

    Args:
        allow_execution: Whether to include the code-execution tools.

    Returns:
        A list of OpenAI-style tool schemas.
    """
    tools: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Semantic search over documentation files (README, .md, .rst, configs).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Semantic search over source-code files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file (optionally a line range) from the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        },
    ]

    if allow_execution:
        tools.extend(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "execute_python_snippet",
                        "description": "Run a short Python snippet inside the repo's venv and return stdout/stderr/exit code.",
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "string"}},
                            "required": ["code"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "execute_python_file",
                        "description": "Run an existing .py file from the repository inside its venv.",
                        "parameters": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "execute_notebook",
                        "description": "Execute a Jupyter notebook and return each cell's code and output.",
                        "parameters": {
                            "type": "object",
                            "properties": {"notebook_path": {"type": "string"}},
                            "required": ["notebook_path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "install_packages",
                        "description": "pip-install packages into the repo venv when an import fails.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "packages": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["packages"],
                        },
                    },
                },
            ]
        )

    tools.append(
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Provide the final structured answer. Always call this to finish.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "answerable": {"type": "boolean"},
                        "verbalized_confidence": {
                            "type": "number",
                            "description": "Your honest probability (0-1) the answer is correct.",
                        },
                        "reasoning": {"type": "string"},
                        "key_evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["answer", "verbalized_confidence"],
                },
            },
        }
    )
    return tools
