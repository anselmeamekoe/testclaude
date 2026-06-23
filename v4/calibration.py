"""Confidence calibration.

The leaderboard penalises **both** overconfidence and underconfidence, so the
model's raw self-reported confidence is not trusted on its own. The
:class:`CalibrationEngine` fuses several signals into a single probability and
maps it to a ``low``/``medium``/``high`` bucket:

* **Self-reported confidence** — the model's own ``high/medium/low``.
* **Evidence strength** — best retrieval similarity, and whether code execution
  actually produced the answer (execution is the strongest evidence).
* **Verifier** — a second, independent LLM pass that grades how well the gathered
  evidence supports the answer, and whether the agent should abstain.
* **Self-consistency** (optional) — agreement across several independent attempts.

This is deliberately transparent and tunable: every signal is a number in
``[0, 1]`` and the final score is a weighted average, with the bucket chosen by
fixed thresholds. Abstentions (``not_known``) are forced to ``low``.
"""

from __future__ import annotations

from sparrow_agent.config import Settings
from sparrow_agent.llm import LLMClient
from sparrow_agent.models import (
    AnswerItem,
    CalibratedConfidence,
    RetrievalHit,
    Trajectory,
)

# Map the discrete self-reported bucket to a prior probability.
_SELF_PRIOR = {"low": 0.30, "medium": 0.60, "high": 0.85}

# Thresholds for turning the fused score back into a bucket.
_HIGH_THRESHOLD = 0.72
_MEDIUM_THRESHOLD = 0.45

_VERIFIER_SYSTEM = (
    "You are a strict evidence auditor for a documentation-QA agent. "
    "Given a question, a proposed answer, and the evidence the agent gathered "
    "(retrieved snippets and/or execution output), judge how well the evidence "
    "supports the answer. Be skeptical: if the answer is plausible but not "
    "actually shown by the evidence, support should be low. "
    "Respond with ONLY a JSON object: "
    '{"support": <float 0..1>, "should_abstain": <bool>, "reason": "<short>"}'
)


class CalibrationEngine:
    """Turns raw answers and trajectories into calibrated confidence."""

    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        """Initialise the engine.

        Args:
            settings: Global configuration (verifier/consistency toggles).
            llm: Chat client used for the verifier pass.
        """
        self._settings = settings
        self._llm = llm

    def calibrate(
        self,
        question: str,
        answer: AnswerItem,
        trajectory: Trajectory,
        hits: list[RetrievalHit],
        agreement: float | None = None,
    ) -> CalibratedConfidence:
        """Compute a calibrated confidence for one answer.

        Args:
            question: The question text (for the verifier).
            answer: The model's proposed answer item.
            trajectory: The reasoning trace (execution/retrieval telemetry).
            hits: The most relevant retrieval hits gathered for this answer.
            agreement: Optional self-consistency agreement ratio in ``[0, 1]``.

        Returns:
            A :class:`CalibratedConfidence` with a continuous score, a bucket,
            and a short rationale.
        """
        # Abstention is, by construction, low confidence in *an answer*.
        if answer.not_known:
            return CalibratedConfidence(
                bucket="low",
                score=0.2,
                rationale="Agent abstained (insufficient evidence in repo/files).",
            )

        signals: list[tuple[float, float]] = []  # (value, weight)
        notes: list[str] = []

        # 1) Model self-report.
        self_prior = _SELF_PRIOR.get((answer.confidence or "medium"), 0.6)
        signals.append((self_prior, 1.0))

        # 2) Evidence strength from retrieval + execution.
        evidence_score = self._evidence_score(trajectory, hits)
        signals.append((evidence_score, 1.5))
        notes.append(f"evidence={evidence_score:.2f}")

        # 3) Verifier pass (independent grading).
        if self._settings.enable_verifier:
            support = self._verify(question, answer, hits, trajectory)
            signals.append((support, 2.0))
            notes.append(f"verifier={support:.2f}")

        # 4) Self-consistency agreement.
        if agreement is not None:
            signals.append((agreement, 1.5))
            notes.append(f"agreement={agreement:.2f}")

        score = _weighted_mean(signals)
        bucket = self._bucket(score)
        return CalibratedConfidence(
            bucket=bucket, score=round(score, 3), rationale="; ".join(notes)
        )

    def _evidence_score(self, trajectory: Trajectory, hits: list[RetrievalHit]) -> float:
        """Score how strong the gathered evidence is, in ``[0, 1]``.

        Execution that ran successfully is the strongest signal. Otherwise we
        lean on the best retrieval similarity, lightly rewarding having more than
        one corroborating snippet.

        Args:
            trajectory: The reasoning trace.
            hits: Retrieval hits.

        Returns:
            An evidence strength in ``[0, 1]``.
        """
        if trajectory.used_execution:
            base = 0.85
        else:
            base = 0.0

        best_sim = trajectory.max_retrieval_score
        if hits:
            best_sim = max(best_sim, max(h.score for h in hits))
        # Map cosine similarity (~0.2 weak .. ~0.6+ strong) onto [0, 1].
        sim_score = max(0.0, min(1.0, (best_sim - 0.20) / 0.40))

        corroboration = 0.05 if len([h for h in hits if h.score > 0.30]) >= 2 else 0.0
        return max(base, min(1.0, sim_score + corroboration))

    def _verify(
        self,
        question: str,
        answer: AnswerItem,
        hits: list[RetrievalHit],
        trajectory: Trajectory,
    ) -> float:
        """Run the independent verifier LLM pass.

        Args:
            question: The question text.
            answer: The proposed answer.
            hits: Retrieval hits used as evidence.
            trajectory: Trace (used to mention whether execution happened).

        Returns:
            The verifier's support score in ``[0, 1]`` (0.5 on failure/uncertain).
        """
        evidence_blocks = [
            f"[{h.chunk.path}:{h.chunk.start_line}-{h.chunk.end_line}] "
            f"(sim={h.score:.2f})\n{h.chunk.text[:600]}"
            for h in hits[:6]
        ]
        exec_note = (
            "Code WAS executed during answering." if trajectory.used_execution else "No code executed."
        )
        user = (
            f"QUESTION:\n{question}\n\n"
            f"PROPOSED ANSWER:\n{answer.answer}\n\n"
            f"{exec_note}\n\n"
            f"RETRIEVED EVIDENCE:\n" + ("\n\n".join(evidence_blocks) or "(none)")
        )
        result = self._llm.complete_json(_VERIFIER_SYSTEM, user)
        try:
            support = float(result.get("support", 0.5))
        except (TypeError, ValueError):
            support = 0.5
        return max(0.0, min(1.0, support))

    @staticmethod
    def _bucket(score: float) -> str:
        """Map a continuous score to a confidence bucket.

        Args:
            score: Fused probability in ``[0, 1]``.

        Returns:
            ``"high"``, ``"medium"``, or ``"low"``.
        """
        if score >= _HIGH_THRESHOLD:
            return "high"
        if score >= _MEDIUM_THRESHOLD:
            return "medium"
        return "low"


def _weighted_mean(signals: list[tuple[float, float]]) -> float:
    """Compute a weighted mean of ``(value, weight)`` pairs.

    Args:
        signals: List of ``(value, weight)`` with values in ``[0, 1]``.

    Returns:
        The weighted average, or ``0.5`` if there are no signals.
    """
    total_weight = sum(w for _, w in signals)
    if total_weight == 0:
        return 0.5
    return sum(v * w for v, w in signals) / total_weight
