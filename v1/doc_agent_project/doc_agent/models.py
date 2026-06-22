"""Pydantic data models that define the agent's input/output contract.

This module is the single source of truth for the shapes that flow through the
system. Everything the agent consumes (questions) and produces (answers with a
calibrated confidence score) is described here so the rest of the codebase can
rely on validated, typed objects instead of loose dictionaries.

The most important field for routing is :attr:`QuestionItem.requires_code_execution`.
It is the binary signal the organizers attach to every question telling us whether
answering it correctly is expected to require *running* code (not just reading it).
The agent treats this as a strong prior when deciding which tools to reach for.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class SourceType(str, Enum):
    """Where a piece of supporting evidence came from.

    Used to tag every :class:`Evidence` object so the confidence calibrator can
    weight execution-derived facts differently from retrieved text.
    """

    CODE = "code"          # A chunk retrieved from the repository's source code.
    DOC = "doc"            # A chunk retrieved from docs / README / markdown / external files.
    EXECUTION = "execution"  # Output produced by actually running code.
    WEB = "web"            # (Optional) content fetched from the web.


class ConfidenceLabel(str, Enum):
    """Human-readable bucket derived from the numeric confidence score.

    The numeric ``confidence`` field is what gets scored for calibration; this
    label is a convenience for logging and UI display only.
    """

    VERY_LOW = "very_low"    # < 0.20 — essentially a guess / information missing.
    LOW = "low"              # 0.20–0.40
    MODERATE = "moderate"    # 0.40–0.60
    HIGH = "high"            # 0.60–0.80
    VERY_HIGH = "very_high"  # > 0.80 — directly verified (e.g. by execution).


class QuestionItem(BaseModel):
    """A single documentation-related question for the agent to answer.

    Attributes:
        id: Stable identifier for the question, echoed back on the answer so the
            grader can align predictions with ground truth.
        question: The natural-language question to answer.
        requires_code_execution: Binary flag provided by the organizers. ``True``
            means the question is expected to need code to be *executed* (run a
            file/notebook/snippet, install deps, inspect runtime behaviour) rather
            than merely read. The agent uses this to bias tool selection: when
            ``True`` it prioritizes the execution toolchain and will lower its
            confidence / flag missing information if execution is impossible.
        context_hint: Optional free-text hint from the organizers (e.g. a module
            name or file path) used to seed retrieval.
    """

    id: str = Field(..., description="Stable identifier echoed back on the answer.")
    question: str = Field(..., min_length=1, description="The question to answer.")
    requires_code_execution: bool = Field(
        ...,
        description=(
            "Binary flag: True if answering correctly is expected to require "
            "executing code (not just reading it). Drives tool routing."
        ),
    )
    context_hint: Optional[str] = Field(
        default=None,
        description="Optional hint (file path, module, symbol) to seed retrieval.",
    )


class QuestionSet(BaseModel):
    """The full payload handed to the agent: a repo plus a *list* of questions.

    The organizers send questions as a set/list, and each item carries its own
    ``requires_code_execution`` flag, so a single payload can mix read-only and
    execution-requiring questions. The agent answers each independently while
    reusing one cloned repo and one RAG index for the whole batch.

    Attributes:
        repo_url: GitLab repository to clone and analyze. Optional because some
            questions may be answerable purely from attached external files.
        questions: The list of questions to answer.
        external_files: Optional paths to extra files (already on disk) that should
            be indexed alongside the repo, e.g. supplementary docs.
    """

    repo_url: Optional[str] = Field(
        default=None, description="GitLab repo URL to clone and analyze."
    )
    questions: list[QuestionItem] = Field(
        ..., min_length=1, description="The list of questions to answer."
    )
    external_files: list[str] = Field(
        default_factory=list,
        description="Optional on-disk paths to additional files to index.",
    )


class Evidence(BaseModel):
    """A single piece of grounding the agent used to support its answer.

    Evidence is what separates a calibrated answer from a hallucination: the
    confidence calibrator inspects the quantity, relevance and *type* of evidence
    to decide how trustworthy the answer is.

    Attributes:
        source_type: Whether this came from code, docs, execution, or the web.
        reference: A locator the user can follow — a file path, symbol, or tool name.
        snippet: The actual supporting text/output (truncated for brevity).
        score: Relevance/quality in ``[0, 1]``. For retrieval this is the
            normalized similarity; for execution it is 1.0 on success.
    """

    source_type: SourceType
    reference: str = Field(..., description="File path, symbol, or tool name.")
    snippet: str = Field(..., description="Supporting text or execution output.")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance/quality 0–1.")


class ExecutionResult(BaseModel):
    """Structured result of running code via the organizer-provided backend.

    This is returned by every execution tool (run file, run notebook, run snippet,
    install packages) so the agent — and the calibrator — get a uniform, typed view
    of what happened at runtime.

    Attributes:
        success: ``True`` iff the process exited cleanly (return code 0) and no
            backend-level error occurred.
        return_code: Process exit code (``None`` if it never started).
        stdout: Captured standard output (may be truncated by the backend).
        stderr: Captured standard error.
        duration_seconds: Wall-clock execution time, useful for timeout reasoning.
        artifact_paths: Any files the run produced that may be worth inspecting.
    """

    success: bool
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: Optional[float] = None
    artifact_paths: list[str] = Field(default_factory=list)


class ConfidenceSignals(BaseModel):
    """The raw inputs to confidence calibration, gathered while answering.

    Keeping these as an explicit, inspectable object (rather than a single opaque
    number) is what makes the system *calibrated* and debuggable: each signal can
    be logged, weighted, and — if labeled data is available — fit to a calibration
    curve. See :mod:`doc_agent.confidence`.

    Attributes:
        verbalized_confidence: The model's own self-reported confidence in ``[0, 1]``
            from the ``submit_answer`` tool. Models are typically over-confident,
            so this is only one input, not the final score.
        retrieval_support: How well retrieved evidence covers the claim, in ``[0, 1]``.
            Derived from retrieval scores and how much evidence was actually found.
        execution_required: Mirror of the question's ``requires_code_execution`` flag.
        execution_attempted: Whether the agent actually ran code for this question.
        execution_succeeded: Whether that execution exited cleanly (``None`` if N/A).
        information_complete: The agent's judgment of whether it had enough
            information to answer at all. ``False`` forces low confidence.
        self_consistency: Optional agreement score in ``[0, 1]`` across multiple
            sampled answers (``None`` if self-consistency sampling was disabled).
    """

    verbalized_confidence: float = Field(..., ge=0.0, le=1.0)
    retrieval_support: float = Field(0.0, ge=0.0, le=1.0)
    execution_required: bool = False
    execution_attempted: bool = False
    execution_succeeded: Optional[bool] = None
    information_complete: bool = True
    self_consistency: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AgentAnswer(BaseModel):
    """The agent's complete, gradeable answer to one question.

    This is the object the hackathon grader consumes. It deliberately surfaces the
    reasoning, the evidence, and the calibration inputs alongside the final score so
    that the confidence number is auditable rather than magical.

    Attributes:
        question_id: Echoes :attr:`QuestionItem.id`.
        answer: The natural-language answer.
        confidence: Final *calibrated* confidence in ``[0, 1]``.
        confidence_label: Bucketed label derived from ``confidence``.
        information_complete: Whether the agent believed it had enough to answer.
        missing_information: If incomplete, what was missing (else ``None``).
        evidence: The grounding used, for auditability.
        tools_used: Names of tools the agent invoked, in order.
        signals: The raw confidence signals that produced ``confidence``.
        reasoning: Short rationale (kept concise to avoid leaking chain-of-thought).
    """

    question_id: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel
    information_complete: bool = True
    missing_information: Optional[str] = None
    evidence: list[Evidence] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    signals: Optional[ConfidenceSignals] = None
    reasoning: str = ""

    @model_validator(mode="after")
    def _sync_label(self) -> "AgentAnswer":
        """Keep ``confidence_label`` consistent with the numeric ``confidence``.

        Runs automatically after construction so callers can set just the number
        and trust the label, eliminating a class of inconsistency bugs.
        """
        c = self.confidence
        if c < 0.20:
            self.confidence_label = ConfidenceLabel.VERY_LOW
        elif c < 0.40:
            self.confidence_label = ConfidenceLabel.LOW
        elif c < 0.60:
            self.confidence_label = ConfidenceLabel.MODERATE
        elif c < 0.80:
            self.confidence_label = ConfidenceLabel.HIGH
        else:
            self.confidence_label = ConfidenceLabel.VERY_HIGH
        return self


class AgentResult(BaseModel):
    """Top-level container returned for a whole :class:`QuestionSet`.

    Attributes:
        answers: One :class:`AgentAnswer` per input question, in input order.
        repo_path: Local path the repo was cloned to (for debugging / reuse).
    """

    answers: list[AgentAnswer]
    repo_path: Optional[str] = None
