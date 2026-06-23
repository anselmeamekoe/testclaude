"""All data models for the system. **Everything is a Pydantic model** (no dataclasses).

This module contains two families of models:

* *Contract* models (``TemplateQuestion``, ``Input``, ``AnswerItem``, ``Output``)
  — the payloads exchanged with the leaderboard. These match the format the
  organisers send/expect and must not drift.
* *Internal* models (``Chunk``, ``RetrievalHit``, ``ToolInvocation``,
  ``Trajectory``, ``CalibratedConfidence``) — used to move structured state
  between the retriever, the agent loop, and the calibration engine.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Contract models (leaderboard payloads)
# ---------------------------------------------------------------------------


class TemplateQuestion(BaseModel):
    """One question entry that the leaderboard sends."""

    id: int = Field(..., description="Primary key of the question in the leaderboard DB")
    question: str = Field(..., description="The text of the question")


class Input(BaseModel):
    """Full payload received from the leaderboard when a submission is created."""

    submission_id: str = Field(..., description="UUID of the submission (must be echoed back later)")
    template_title: str = Field(..., description="Human-readable title of the chosen template")
    repo_url: str = Field(..., description="Git repository URL containing the starter code")
    code_execution: bool = Field(
        ..., description="Whether the agent may execute code to answer this set of questions"
    )
    token_gitlab: str = Field(..., description="Team-specific GitLab token")
    token_sparrow: str = Field(..., description="Team-specific Sparrow token")
    access_key: str = Field(..., description="Team-specific dataset access key")
    secret_key: str = Field(..., description="Team-specific dataset secret key")
    template: list[TemplateQuestion] = Field(
        ..., description="List of questions (id + question text) the agent must answer"
    )


class AnswerItem(BaseModel):
    """One answer the agent produces for one question, with calibrated confidence."""

    question: int = Field(..., description="The id of the question being answered")
    answer: str = Field(..., description="The agent's answer (or an explanation of what is missing)")
    confidence: Literal["low", "medium", "high"] | None = Field(
        default="high", description="Calibrated confidence bucket"
    )
    evidence: list[Literal["files", "execution"]] | None = Field(
        default_factory=lambda: ["files"],
        description="Which evidence sources actually contributed to the answer",
    )
    not_known: bool | None = Field(
        default=False, description="True when the agent abstains because evidence is insufficient"
    )


class Output(BaseModel):
    """Final response sent back to the leaderboard."""

    submission_id: str = Field(..., description="Echoes Input.submission_id")
    answers: list[AnswerItem] = Field(..., description="One AnswerItem per input question")


# ---------------------------------------------------------------------------
# Internal models (RAG + agent trajectory + calibration)
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A retrievable unit of text extracted from the repository."""

    id: str = Field(..., description="Stable identifier, e.g. 'path/to/file.py:120-180'")
    path: str = Field(..., description="Repository-relative path of the source file")
    start_line: int = Field(..., description="1-based first line covered by the chunk")
    end_line: int = Field(..., description="1-based last line covered by the chunk")
    kind: Literal["code", "doc", "other"] = Field(..., description="Coarse content category")
    text: str = Field(..., description="The chunk text that gets embedded")
    num_tokens: int = Field(default=0, description="Approximate token count of the text")


class RetrievalHit(BaseModel):
    """A chunk returned by similarity search, with its score."""

    chunk: Chunk
    score: float = Field(..., description="Cosine similarity in [-1, 1]; higher is closer")


class ToolInvocation(BaseModel):
    """A single tool call made during the agent loop, recorded for telemetry/calibration."""

    name: str = Field(..., description="Tool name, e.g. 'search_code' or 'execute_python_snippet'")
    arguments: dict = Field(default_factory=dict, description="Arguments passed to the tool")
    ok: bool = Field(default=True, description="Whether the tool ran without raising")
    result_preview: str = Field(default="", description="Truncated tool output for logging")


class Trajectory(BaseModel):
    """The full reasoning trace for one question.

    The calibration engine consumes this to turn raw, model-reported confidence
    into a calibrated bucket using *observable* signals (did we execute code?
    how strong was retrieval? how many steps did it take?).
    """

    question_id: int
    steps: list[ToolInvocation] = Field(default_factory=list)
    used_execution: bool = Field(default=False, description="Did any execution tool succeed?")
    used_files: bool = Field(default=False, description="Did any retrieval/file-read contribute?")
    max_retrieval_score: float = Field(
        default=0.0, description="Best cosine similarity seen across all searches"
    )
    num_steps: int = Field(default=0, description="Total tool calls made")


class CalibratedConfidence(BaseModel):
    """Output of the calibration engine."""

    bucket: Literal["low", "medium", "high"]
    score: float = Field(..., description="Continuous calibrated probability in [0, 1]")
    rationale: str = Field(default="", description="Short human-readable justification")
