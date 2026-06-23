"""Pydantic data models for the documentation-QA agent.

Everything that crosses a boundary in this system is a Pydantic model: the
imposed **input** schema (a set of questions plus the binary
``requires_code_execution`` flag), the **output** schema (per-question answer +
calibrated confidence), and the internal book-keeping types (evidence,
tool-call records, confidence signals). No ``dataclasses`` are used anywhere, as
required.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# --------------------------------------------------------------------------- #
# Input schema (imposed by organizers)                                        #
# --------------------------------------------------------------------------- #
class QuestionSet(BaseModel):
    """The unit of work handed to the agent.

    The organizers impose a single binary field that declares, for the **whole
    set**, whether answering will require executing code. The agent uses this
    flag to decide which tools to expose to the LLM (see ``agent.py``): when it
    is ``False`` the execution tools are withheld entirely, which keeps the
    model from wandering into unnecessary (and slower, riskier) code runs.

    Attributes:
        questions: The documentation questions to answer, in order.
        requires_code_execution: Binary flag — ``True`` if answering this set
            may require running code from the repository.
        repo_url: HTTPS GitLab URL of the repository the questions are about.
            Optional for purely conceptual questions with no repo.
        external_file_paths: Absolute paths to extra files (already on disk)
            that should be indexed alongside the repository.
        gitlab_token: Per-request override for the clone token; falls back to
            the configured default when omitted.
        set_id: Optional identifier echoed back in the result for traceability.
    """

    questions: list[str] = Field(min_length=1)
    requires_code_execution: bool
    repo_url: str | None = None
    external_file_paths: list[str] = Field(default_factory=list)
    gitlab_token: str | None = None
    set_id: str | None = None

    @field_validator("questions")
    @classmethod
    def _strip_questions(cls, value: list[str]) -> list[str]:
        """Reject empty/blank questions and trim surrounding whitespace."""
        cleaned = [q.strip() for q in value if q and q.strip()]
        if not cleaned:
            raise ValueError("`questions` must contain at least one non-empty string")
        return cleaned


# --------------------------------------------------------------------------- #
# Evidence & tool bookkeeping                                                  #
# --------------------------------------------------------------------------- #
class EvidenceKind(str, Enum):
    """Where a piece of supporting evidence came from."""

    CODE = "code"
    DOC = "doc"
    EXECUTION = "execution"
    FILE = "file"


class Evidence(BaseModel):
    """A single retrieved or produced fact that supports an answer.

    Attributes:
        kind: The category of evidence (code chunk, doc chunk, execution output…).
        source: A human-readable locator, e.g. ``"src/model.py:120-180"`` or
            ``"executed snippet"``.
        content: The actual text/snippet/output.
        score: Retrieval similarity in ``[0, 1]`` for searched evidence, or a
            success proxy for execution evidence. ``None`` when not applicable.
        success: For execution evidence, whether the run exited cleanly.
    """

    kind: EvidenceKind
    source: str
    content: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    success: bool | None = None


class ToolCallRecord(BaseModel):
    """An audit record of one tool invocation made by the agent."""

    name: str
    arguments: dict[str, Any]
    result_preview: str = Field(
        description="Truncated tool output, kept for transparency/debugging."
    )
    ok: bool = True


# --------------------------------------------------------------------------- #
# Confidence                                                                   #
# --------------------------------------------------------------------------- #
class ConfidenceSignals(BaseModel):
    """Raw, un-calibrated inputs to the confidence estimator.

    Keeping these explicit (rather than collapsing straight to a number) makes
    the final confidence auditable and lets the calibration weights be re-fit
    on a labelled dev set without touching the agent.

    Attributes:
        verbalized: The model's self-reported probability that its answer is
            correct, in ``[0, 1]``.
        self_consistency: Agreement rate across independent samples in
            ``[0, 1]`` (``None`` when self-consistency was disabled).
        evidence_strength: Aggregate quality of supporting evidence in
            ``[0, 1]`` (retrieval scores, execution success, corroboration).
        grounded: Whether the final answer is actually supported by collected
            evidence rather than asserted without it.
        answerable: Whether the agent believes the question is answerable from
            the available material at all.
    """

    verbalized: float = Field(ge=0.0, le=1.0)
    self_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_strength: float = Field(ge=0.0, le=1.0)
    grounded: bool = True
    answerable: bool = True


class AnswerConfidence(BaseModel):
    """The calibrated confidence plus the signals that produced it."""

    score: float = Field(ge=0.0, le=1.0, description="Final calibrated P(correct).")
    signals: ConfidenceSignals


# --------------------------------------------------------------------------- #
# Output schema (imposed by organizers)                                       #
# --------------------------------------------------------------------------- #
class QuestionAnswer(BaseModel):
    """The agent's answer to a single question.

    Attributes:
        question: The question being answered (echoed for alignment).
        answer: The natural-language answer. When ``answerable`` is False this
            should explicitly state that the information could not be found.
        confidence: Calibrated probability that ``answer`` is correct, ``[0, 1]``.
        answerable: False when required information was missing.
        reasoning: Short rationale describing how the answer was derived.
        evidence: The supporting evidence collected by the agent.
        tool_calls: Audit trail of tool usage for this question.
        confidence_detail: Full breakdown of the confidence computation.
    """

    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    answerable: bool = True
    reasoning: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    confidence_detail: AnswerConfidence | None = None

    @model_validator(mode="after")
    def _sync_confidence(self) -> "QuestionAnswer":
        """Keep the flat ``confidence`` field in lock-step with the detail block."""
        if self.confidence_detail is not None:
            self.confidence = self.confidence_detail.score
        return self


class AgentResult(BaseModel):
    """The full response for a :class:`QuestionSet`.

    Attributes:
        set_id: Echo of the input ``set_id``.
        answers: One :class:`QuestionAnswer` per input question, same order.
        repo_indexed: Whether a repository was successfully cloned and indexed.
        notes: Free-form diagnostics (errors during clone/index, etc.).
    """

    set_id: str | None = None
    answers: list[QuestionAnswer]
    repo_indexed: bool = False
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Internal: the structured object the LLM emits via the `finish` tool          #
# --------------------------------------------------------------------------- #
class FinalAnswerPayload(BaseModel):
    """Schema the LLM must satisfy when it calls the ``finish`` tool.

    Separating this from :class:`QuestionAnswer` lets the model report only what
    it can know (the answer + a self-assessed probability), while the system
    owns the *calibrated* confidence and the evidence ledger.
    """

    answer: str
    answerable: bool = True
    verbalized_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your honest probability that the answer is correct (0-1).",
    )
    reasoning: str = ""
    key_evidence: list[str] = Field(
        default_factory=list,
        description="Short references to the evidence you actually relied on.",
    )
