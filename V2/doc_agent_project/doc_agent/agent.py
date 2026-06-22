"""The agentic orchestrator: the reasoning system tying every piece together.

This is the component that makes the solution "an agentic reasoning system, not a
chatbot". For a :class:`~doc_agent.models.QuestionSet` it:

1. Clones the repo (once) and builds a FAISS RAG index over its code and docs.
2. For each question, assembles a tailored tool registry — retrieval always, plus
   the execution toolchain when the question is flagged ``requires_code_execution``
   (still available, but de-emphasized, otherwise).
3. Runs a bounded tool-use loop: the model decides when to search code, search
   docs, execute code, or declare information missing, finishing by calling the
   terminal ``submit_answer`` tool.
4. Collects evidence and runtime signals along the way and feeds them to the
   :class:`~doc_agent.confidence.ConfidenceCalibrator` to produce a calibrated
   score — never trusting the model's self-report alone.

The loop is defensive: tool errors become observations, step budgets are enforced,
and a missing ``submit_answer`` degrades gracefully into a low-confidence,
information-incomplete answer.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import Settings
from .confidence import ConfidenceCalibrator, PlattScaler
from .llm import (
    HistoryAssistant,
    HistoryItem,
    HistoryToolResults,
    HistoryUser,
    LLMClient,
    build_llm,
)
from .models import (
    AgentAnswer,
    AgentResult,
    ConfidenceLabel,
    ConfidenceSignals,
    Evidence,
    QuestionItem,
    QuestionSet,
)
from .prompts import SUBMIT_ANSWER_SCHEMA, SYSTEM_PROMPT, build_question_prompt
from .rag import VectorIndex, build_embedder, chunk_repository
from .tools.base import Tool, ToolRegistry, ToolResult
from .tools.execution import ExecutionBackend, build_execution_tools
from .tools.retrieval import build_retrieval_tools


class DocAgent:
    """Coordinates retrieval, execution, and calibration to answer questions.

    A single instance is reusable across many :class:`QuestionSet` payloads. State
    that is specific to one run (the cloned repo path, the built index) is held on
    the instance only for the duration of :meth:`answer_questions`.
    """

    def __init__(
        self,
        backend: ExecutionBackend,
        settings: Optional[Settings] = None,
        llm: Optional[LLMClient] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
    ) -> None:
        """Wire up the agent's collaborators.

        Args:
            backend: Execution backend (organizer sandbox adapter, or the local
                reference implementation).
            settings: Tunable configuration; defaults to :meth:`Settings.from_env`.
            llm: LLM client; defaults to the provider selected in ``settings``
                (Anthropic or OpenAI-compatible) via :func:`build_llm`. Inject a
                fake here in tests.
            calibrator: Confidence calibrator; defaults to one built from settings
                with an identity Platt map.
        """
        self.settings = settings or Settings.from_env()
        self.backend = backend
        self.llm = llm or build_llm(self.settings)
        self.calibrator = calibrator or ConfidenceCalibrator(self.settings, PlattScaler())
        self._repo_path: Optional[str] = None
        self._index: Optional[VectorIndex] = None

    # ------------------------------------------------------------------ setup ---
    def _prepare_repository(self, payload: QuestionSet) -> None:
        """Clone the repo (if any) and build the RAG index over it + external files.

        Runs once per :class:`QuestionSet` so the (potentially expensive) clone and
        embedding happen a single time for the whole batch.

        Args:
            payload: The incoming question set.
        """
        self._repo_path = None
        if payload.repo_url:
            self._repo_path = self.backend.clone_repo(payload.repo_url)

        if not self.settings.enable_rag:
            self._index = None
            return

        chunks = []
        if self._repo_path:
            chunks.extend(
                chunk_repository(
                    self._repo_path, self.settings.chunk_size, self.settings.chunk_overlap
                )
            )
        # Index any standalone external files as documents too.
        for path in payload.external_files:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                from .rag.chunking import chunk_document

                chunks.extend(
                    chunk_document(
                        path, text, self.settings.chunk_size, self.settings.chunk_overlap
                    )
                )
            except OSError:
                continue

        embedder = build_embedder(self.settings)
        self._index = VectorIndex(embedder).build(chunks)

    def _build_registry(self, item: QuestionItem) -> ToolRegistry:
        """Assemble the tools available for a single question.

        Retrieval tools are always present. Execution tools are always registered
        too (the model may discover it needs them), but the *prompt* — built from
        ``requires_code_execution`` — controls how strongly they are encouraged. The
        terminal ``submit_answer`` tool is added last.

        Args:
            item: The question being answered.

        Returns:
            A populated :class:`ToolRegistry`.
        """
        registry = ToolRegistry()

        if self._index is not None and len(self._index) > 0:
            for tool in build_retrieval_tools(self._index, self.settings.top_k):
                registry.register(tool)

        if self._repo_path:
            for tool in build_execution_tools(self.backend, lambda: self._repo_path):
                registry.register(tool)

        registry.register(self._make_submit_tool())
        return registry

    def _make_submit_tool(self) -> Tool:
        """Create the terminal ``submit_answer`` tool that ends the loop.

        Its handler simply echoes the structured input back via metadata; the agent
        loop detects the terminal tool and extracts the answer from there.

        Returns:
            The terminal :class:`Tool`.
        """

        def submit_answer(**kwargs: Any) -> ToolResult:
            """Record the model's structured final answer (no side effects)."""
            return ToolResult(
                content="Answer recorded.",
                success=True,
                metadata={"submission": kwargs},
            )

        return Tool(
            name="submit_answer",
            description=(
                "Submit your FINAL answer. Call exactly once when done. Provide a "
                "calibrated confidence and whether information was sufficient."
            ),
            input_schema=SUBMIT_ANSWER_SCHEMA,
            handler=submit_answer,
            terminal=True,
        )

    # -------------------------------------------------------------- main loop ---
    def _answer_one(self, item: QuestionItem) -> AgentAnswer:
        """Run the bounded tool-use loop for a single question and calibrate it.

        Args:
            item: The question to answer.

        Returns:
            A fully populated, calibrated :class:`AgentAnswer`.
        """
        registry = self._build_registry(item)
        history: list[HistoryItem] = [HistoryUser(content=build_question_prompt(item))]

        collected_evidence: list[Evidence] = []
        tools_used: list[str] = []
        execution_attempted = False
        execution_succeeded: Optional[bool] = None
        submission: Optional[dict[str, Any]] = None

        for _ in range(self.settings.max_agent_steps):
            turn = self.llm.complete(
                system=SYSTEM_PROMPT,
                history=history,
                tools=registry.schemas(),
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )

            # If the model produced prose without any tool call, force the terminal
            # submit_answer tool so we always end with a structured submission.
            if not turn.tool_calls:
                turn = self.llm.complete(
                    system=SYSTEM_PROMPT,
                    history=history,
                    tools=registry.schemas(),
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                    force_tool="submit_answer",
                )

            # Record the assistant turn (text + any tool calls) in the transcript.
            history.append(HistoryAssistant(text=turn.text, tool_calls=turn.tool_calls))

            if not turn.tool_calls:
                # Even forcing failed to elicit a tool call; stop and finalize.
                break

            # Execute each requested tool and stage the results for the next turn.
            result_records: list[dict[str, Any]] = []
            terminated = False
            for call in turn.tool_calls:
                tool = registry.get(call["name"])
                tools_used.append(call["name"])
                if tool is None:
                    result_records.append(
                        {
                            "id": call["id"],
                            "content": f"Unknown tool: {call['name']}",
                            "is_error": True,
                        }
                    )
                    continue

                result = tool.run(**call.get("input", {}))
                collected_evidence.extend(result.evidence)

                # Track execution outcomes for calibration.
                if call["name"] in {
                    "setup_environment",
                    "run_python_file",
                    "run_notebook",
                    "execute_code",
                }:
                    execution_attempted = True
                    execution_succeeded = bool(result.success) if execution_succeeded is None \
                        else (execution_succeeded or bool(result.success))

                if tool.terminal:
                    submission = result.metadata.get("submission")
                    terminated = True

                result_records.append(
                    {
                        "id": call["id"],
                        "content": result.content,
                        "is_error": not result.success,
                    }
                )

            if result_records:
                history.append(HistoryToolResults(results=result_records))

            if terminated:
                break

        return self._finalize(
            item=item,
            submission=submission,
            evidence=collected_evidence,
            tools_used=tools_used,
            execution_attempted=execution_attempted,
            execution_succeeded=execution_succeeded,
        )

    def _finalize(
        self,
        *,
        item: QuestionItem,
        submission: Optional[dict[str, Any]],
        evidence: list[Evidence],
        tools_used: list[str],
        execution_attempted: bool,
        execution_succeeded: Optional[bool],
    ) -> AgentAnswer:
        """Assemble signals, calibrate, and build the final :class:`AgentAnswer`.

        Handles the degraded path where the model never submitted: that becomes a
        low-confidence, information-incomplete answer rather than a crash.

        Args:
            item: The question answered.
            submission: The structured ``submit_answer`` input, or ``None``.
            evidence: All evidence gathered during the loop.
            tools_used: Tool names invoked, in order.
            execution_attempted: Whether any execution tool ran.
            execution_succeeded: Aggregate success of execution (``None`` if N/A).

        Returns:
            The calibrated :class:`AgentAnswer`.
        """
        if submission is None:
            submission = {
                "answer": "The agent could not produce a grounded answer within its step budget.",
                "verbalized_confidence": 0.1,
                "information_complete": False,
                "missing_information": "No final answer was submitted.",
                "reasoning": "Step budget exhausted before submit_answer was called.",
            }

        retrieval_scores = [
            e.score for e in evidence if e.source_type.value in {"code", "doc"}
        ]
        signals = ConfidenceSignals(
            verbalized_confidence=float(submission.get("verbalized_confidence", 0.5)),
            retrieval_support=ConfidenceCalibrator.retrieval_support_from_scores(
                retrieval_scores
            ),
            execution_required=item.requires_code_execution,
            execution_attempted=execution_attempted,
            execution_succeeded=execution_succeeded,
            information_complete=bool(submission.get("information_complete", True)),
        )
        confidence = self.calibrator.calibrate(signals)

        missing = submission.get("missing_information") or None
        if signals.information_complete:
            missing = None

        return AgentAnswer(
            question_id=item.id,
            answer=str(submission.get("answer", "")),
            confidence=confidence,
            confidence_label=ConfidenceLabel.MODERATE,  # corrected by model_validator
            information_complete=signals.information_complete,
            missing_information=missing,
            evidence=evidence,
            tools_used=tools_used,
            signals=signals,
            reasoning=str(submission.get("reasoning", "")),
        )

    # ------------------------------------------------------------------ public ---
    def answer_questions(self, payload: QuestionSet) -> AgentResult:
        """Answer an entire :class:`QuestionSet`, sharing one repo/index.

        This is the single entry point a grader calls. It prepares the repository
        once, then answers each question independently (so one failure cannot taint
        the rest), preserving input order.

        Args:
            payload: The repo + list of questions to answer.

        Returns:
            An :class:`AgentResult` with one calibrated answer per question.
        """
        self._prepare_repository(payload)
        answers = [self._answer_one(item) for item in payload.questions]
        return AgentResult(answers=answers, repo_path=self._repo_path)
