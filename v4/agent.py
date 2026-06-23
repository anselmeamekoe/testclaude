"""The agentic reasoning loop.

For each question the :class:`DocQAAgent` runs a bounded tool-calling loop with
gpt-oss-120b. The model decides, on its own, when to search code, search docs,
read a file, execute code, or abstain — then emits a final answer through the
``submit_answer`` tool. The agent never lets the model free-text its final
answer: structured output is enforced via that terminal tool, which maps 1:1 to
:class:`AnswerItem`.

The system prompt encodes the decision policy and the abstain-over-guess stance
that confidence calibration rewards.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from sparrow_agent.calibration import CalibrationEngine
from sparrow_agent.config import Settings
from sparrow_agent.embeddings import EmbeddingClient
from sparrow_agent.llm import LLMClient
from sparrow_agent.models import AnswerItem, RetrievalHit, TemplateQuestion, Trajectory
from sparrow_agent.tools import SUBMIT_ANSWER, ToolBox, parse_tool_arguments

_SYSTEM_TEMPLATE = """You are a meticulous documentation-QA agent working over a single \
code repository ("{title}").

Your job: answer the user's question about this repository as accurately as \
possible, and report HONEST confidence. You are graded on correctness AND on \
calibration — both overconfidence and underconfidence are penalised.

Decision policy (choose tools deliberately, do not call tools you do not need):
- Use `search_docs` for conceptual / "how do I" / purpose questions.
- Use `search_code` for "where / how is X implemented", defaults, signatures, behaviour.
- Use `read_file` to confirm exact wording, values, or line-level details after a search.
- Use `list_files` only to orient yourself when you do not know where to look.
{execution_policy}
- When the repository and files genuinely do not contain the answer, do NOT \
guess. Call `submit_answer` with not_known=true and confidence="low", and briefly \
say what is missing.

Confidence guidance for `submit_answer`:
- "high": the answer is directly shown by code/docs you read, or verified by execution.
- "medium": well-supported but requires some inference or you saw it only indirectly.
- "low": weak/partial/indirect evidence.

Always finish by calling `submit_answer` exactly once. Keep answers concise and \
specific (name files, functions, values). In `reasoning`, cite the concrete \
file:line or execution output you relied on."""

_EXEC_ON = (
    "- This question set ALLOWS code execution. Use `execute_python_snippet` to \
empirically verify a value when reading the code is ambiguous (e.g. compute a \
default, read a config at runtime, inspect an object). Use `execute_python_file` \
or `execute_notebook` to run existing artefacts. Install packages only on \
ModuleNotFoundError. Prefer cheap reads first; execute when it raises confidence."
)
_EXEC_OFF = (
    "- This question set does NOT allow code execution. Answer from code and docs \
only; never attempt to run anything."
)


class DocQAAgent:
    """Answers one question at a time via a tool-calling loop."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        embedder: EmbeddingClient,
        calibration: CalibrationEngine,
    ) -> None:
        """Initialise the agent.

        Args:
            settings: Global configuration.
            llm: Chat client for gpt-oss-120b.
            embedder: Embedding client (used for self-consistency clustering).
            calibration: Confidence calibration engine.
        """
        self._settings = settings
        self._llm = llm
        self._embedder = embedder
        self._calibration = calibration

    def answer(
        self, question: TemplateQuestion, toolbox: ToolBox, template_title: str
    ) -> AnswerItem:
        """Answer a question, optionally using self-consistency.

        Args:
            question: The question to answer.
            toolbox: The per-run toolbox (carries repo path, venv, exec flag).
            template_title: Title of the template, used to ground the system prompt.

        Returns:
            A fully-populated, calibration-adjusted :class:`AnswerItem`.
        """
        if self._settings.enable_self_consistency and self._settings.self_consistency_samples > 1:
            return self._answer_self_consistent(question, toolbox, template_title)

        item, trajectory, hits = self._run_once(question, toolbox, template_title)
        return self._finalize(question, item, trajectory, hits, agreement=None)

    # ---- single attempt ------------------------------------------------------

    def _run_once(
        self, question: TemplateQuestion, toolbox: ToolBox, template_title: str
    ) -> tuple[AnswerItem, Trajectory, list[RetrievalHit]]:
        """Run one full tool-calling loop for a question.

        Args:
            question: The question to answer.
            toolbox: The per-run toolbox.
            template_title: Template title for the system prompt.

        Returns:
            A tuple of (raw answer item, trajectory, retrieval hits gathered).
        """
        trajectory = Trajectory(question_id=question.id)
        collected_hits: dict[str, RetrievalHit] = {}

        system = _SYSTEM_TEMPLATE.format(
            title=template_title,
            execution_policy=_EXEC_ON if toolbox._code_execution else _EXEC_OFF,  # noqa: SLF001
        )
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question (id={question.id}): {question.question}"},
        ]
        tools = toolbox.schemas()

        for step in range(self._settings.max_agent_steps):
            force_submit = step == self._settings.max_agent_steps - 1
            message = self._llm.chat(
                messages=messages,
                tools=tools,
                tool_choice=(
                    {"type": "function", "function": {"name": SUBMIT_ANSWER}}
                    if force_submit
                    else "auto"
                ),
            )
            messages.append(_assistant_to_dict(message))

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                # Model answered in prose instead of submitting; nudge it.
                messages.append(
                    {"role": "user", "content": "Call submit_answer to finalise your answer."}
                )
                continue

            submitted: AnswerItem | None = None
            for call in tool_calls:
                name = call.function.name
                args = parse_tool_arguments(call.function.arguments)
                if name == SUBMIT_ANSWER:
                    submitted = self._build_answer(question.id, args)
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": "answer recorded"}
                    )
                    continue
                result = toolbox.dispatch(name, args, trajectory)
                self._collect_hits(name, args, toolbox, collected_hits)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

            if submitted is not None:
                return submitted, trajectory, list(collected_hits.values())

        # Safety net: never happens because the last step forces submit.
        fallback = AnswerItem(
            question=question.id,
            answer="Unable to determine the answer within the step budget.",
            confidence="low",
            evidence=["files"],
            not_known=True,
        )
        return fallback, trajectory, list(collected_hits.values())

    def _collect_hits(
        self, name: str, args: dict, toolbox: ToolBox, store: dict[str, RetrievalHit]
    ) -> None:
        """Re-run search tools to capture structured hits for calibration.

        The dispatch path returns text for the model; here we capture the typed
        hits so the verifier and evidence scorer can use similarity values.

        Args:
            name: The tool name that was called.
            args: The tool arguments.
            toolbox: The toolbox (to reach the retriever).
            store: Accumulator keyed by chunk id (dedupes across searches).
        """
        if name not in {"search_code", "search_docs"}:
            return
        query = str(args.get("query", "")).strip()
        if not query:
            return
        retriever = toolbox._retriever  # noqa: SLF001
        hits = (
            retriever.search_code(query) if name == "search_code" else retriever.search_docs(query)
        )
        for hit in hits:
            existing = store.get(hit.chunk.id)
            if existing is None or hit.score > existing.score:
                store[hit.chunk.id] = hit

    # ---- self-consistency ----------------------------------------------------

    def _answer_self_consistent(
        self, question: TemplateQuestion, toolbox: ToolBox, template_title: str
    ) -> AnswerItem:
        """Answer several times and use agreement as a calibration signal.

        Picks the majority answer (clustered by answer-embedding similarity) and
        passes the agreement ratio into calibration.

        Args:
            question: The question to answer.
            toolbox: The per-run toolbox.
            template_title: Template title for the system prompt.

        Returns:
            The finalised, calibrated answer for the majority cluster.
        """
        attempts: list[tuple[AnswerItem, Trajectory, list[RetrievalHit]]] = []
        for _ in range(self._settings.self_consistency_samples):
            attempts.append(self._run_once(question, toolbox, template_title))

        # If most attempts abstain, abstain.
        n = len(attempts)
        abstentions = sum(1 for item, _, _ in attempts if item.not_known)
        if abstentions > n / 2:
            item, traj, hits = next(a for a in attempts if a[0].not_known)
            return self._finalize(question, item, traj, hits, agreement=abstentions / n)

        answered = [a for a in attempts if not a[0].not_known]
        rep_index, agreement = self._majority_cluster([a[0].answer for a in answered])
        item, traj, hits = answered[rep_index]
        return self._finalize(question, item, traj, hits, agreement=agreement)

    def _majority_cluster(self, answers: list[str]) -> tuple[int, float]:
        """Cluster answers by embedding similarity and return the largest cluster.

        Args:
            answers: The answer strings from each attempt.

        Returns:
            A tuple ``(representative_index, agreement_ratio)`` where the index
            points into ``answers`` and agreement is cluster_size / total.
        """
        if len(answers) == 1:
            return 0, 1.0
        vectors = self._embedder.embed(answers, normalize=True)
        sim = vectors @ vectors.T  # cosine since normalised
        threshold = 0.85
        # Greedy: the answer agreeing with the most others wins.
        agree_counts = (sim >= threshold).sum(axis=1)
        rep = int(np.argmax(agree_counts))
        agreement = float(agree_counts[rep]) / len(answers)
        return rep, agreement

    # ---- finalisation --------------------------------------------------------

    def _finalize(
        self,
        question: TemplateQuestion,
        item: AnswerItem,
        trajectory: Trajectory,
        hits: list[RetrievalHit],
        agreement: float | None,
    ) -> AnswerItem:
        """Apply calibration and fill the evidence field from observed telemetry.

        Args:
            question: The question (text needed by the verifier).
            item: The raw answer item from the loop.
            trajectory: The reasoning trace.
            hits: Retrieval hits gathered during the loop.
            agreement: Self-consistency agreement, if computed.

        Returns:
            The final :class:`AnswerItem` with calibrated confidence and evidence.
        """
        calibrated = self._calibration.calibrate(
            question=question.question,
            answer=item,
            trajectory=trajectory,
            hits=hits,
            agreement=agreement,
        )
        item.confidence = calibrated.bucket

        # Derive evidence sources from what actually happened, intersected with
        # what the model claimed, defaulting sensibly.
        evidence: list[str] = []
        if trajectory.used_files or hits:
            evidence.append("files")
        if trajectory.used_execution:
            evidence.append("execution")
        if not evidence:
            evidence = item.evidence or ["files"]
        item.evidence = evidence  # type: ignore[assignment]
        return item

    # ---- helpers -------------------------------------------------------------

    @staticmethod
    def _build_answer(question_id: int, args: dict) -> AnswerItem:
        """Construct an :class:`AnswerItem` from ``submit_answer`` arguments.

        Args:
            question_id: The id to stamp on the answer.
            args: Parsed arguments from the terminal tool call.

        Returns:
            A validated answer item (pre-calibration).
        """
        confidence = args.get("confidence")
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        evidence = args.get("evidence") or ["files"]
        evidence = [e for e in evidence if e in {"files", "execution"}] or ["files"]
        return AnswerItem(
            question=question_id,
            answer=str(args.get("answer", "")).strip() or "(no answer produced)",
            confidence=confidence,  # type: ignore[arg-type]
            evidence=evidence,  # type: ignore[arg-type]
            not_known=bool(args.get("not_known", False)),
        )


def _assistant_to_dict(message) -> dict:
    """Convert an SDK assistant message into a plain dict for the next turn.

    Preserves tool calls so the follow-up ``tool`` messages are valid.

    Args:
        message: The assistant message object returned by the SDK.

    Returns:
        A chat-format dict with ``role``, ``content``, and optional ``tool_calls``.
    """
    out: dict = {"role": "assistant", "content": message.content or ""}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in tool_calls
        ]
    return out
