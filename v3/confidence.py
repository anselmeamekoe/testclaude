"""Confidence calibration.

The scoring metric penalises **both** over- and under-confidence, so a naive
"ask the model how sure it is" is not enough — verbalized confidence from LLMs
is systematically over-confident. This module turns several weak signals into a
single calibrated probability:

1. **Verbalized** — the model's self-reported P(correct).
2. **Self-consistency** — agreement across independently sampled answers. Low
   agreement is a strong over-confidence detector.
3. **Evidence strength** — retrieval similarity, successful code execution and
   the amount of corroboration actually backing the answer.
4. **Grounding / answerability** — hard penalties when the answer is not
   supported by evidence or when the question is unanswerable.

The signals are combined in logit space with tunable weights, and the result
can be temperature-scaled. :meth:`ConfidenceCalibrator.fit_temperature` lets you
re-fit the temperature on a labelled dev set (Platt-style) to minimise expected
calibration error before the competition.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from .models import AnswerConfidence, ConfidenceSignals, Evidence


def _logit(p: float, eps: float = 1e-4) -> float:
    """Numerically-stable logit of a probability clamped away from 0/1."""
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Standard logistic function."""
    return 1.0 / (1.0 + math.exp(-x))


class CalibrationWeights(BaseModel):
    """Tunable weights for combining confidence signals (all in logit space).

    Defaults are hand-set to be conservative (pull confident-but-unsupported
    answers down). Re-fit ``temperature`` on labelled data for best results.
    """

    bias: float = Field(default=-0.4)
    w_verbalized: float = Field(default=0.7)
    w_consistency: float = Field(default=1.1)
    w_evidence: float = Field(default=1.4)
    temperature: float = Field(default=1.0, gt=0.0)
    unanswerable_cap: float = Field(default=0.15, ge=0.0, le=1.0)
    ungrounded_cap: float = Field(default=0.45, ge=0.0, le=1.0)


def evidence_strength(evidence: list[Evidence]) -> float:
    """Summarise a body of evidence into a single ``[0, 1]`` strength score.

    Combines (a) the best retrieval similarity, (b) whether any code execution
    succeeded, and (c) how much corroboration exists. Execution success is
    weighted heavily because a clean run is near-ground-truth for code questions.

    Args:
        evidence: The evidence collected while answering one question.

    Returns:
        A strength score in ``[0, 1]``.
    """
    if not evidence:
        return 0.0

    retrieval_scores = [e.score for e in evidence if e.score is not None]
    best_retrieval = max(retrieval_scores) if retrieval_scores else 0.0

    exec_items = [e for e in evidence if e.success is not None]
    exec_success = any(e.success for e in exec_items)
    exec_failed_only = bool(exec_items) and not exec_success

    # Corroboration: multiple decent sources agreeing raises strength.
    strong_sources = sum(1 for s in retrieval_scores if s >= 0.45)
    corroboration = min(1.0, strong_sources / 3.0)

    strength = 0.55 * best_retrieval + 0.45 * corroboration
    if exec_success:
        strength = max(strength, 0.85)
    if exec_failed_only:
        strength *= 0.5
    return float(max(0.0, min(1.0, strength)))


class ConfidenceCalibrator:
    """Turns raw signals into a calibrated probability.

    Args:
        weights: Combination weights / temperature. Defaults are reasonable but
            should be re-fit on a dev set via :meth:`fit_temperature`.
    """

    def __init__(self, weights: CalibrationWeights | None = None) -> None:
        self.weights = weights or CalibrationWeights()

    def calibrate(self, signals: ConfidenceSignals) -> AnswerConfidence:
        """Compute the calibrated confidence for a single answer.

        Args:
            signals: The raw confidence signals for the answer.

        Returns:
            An :class:`AnswerConfidence` containing the final score and the
            signals that produced it.
        """
        w = self.weights

        # Hard caps first — these dominate everything else.
        if not signals.answerable:
            return AnswerConfidence(score=min(w.unanswerable_cap, signals.verbalized),
                                    signals=signals)

        z = w.bias
        z += w.w_verbalized * _logit(signals.verbalized)
        z += w.w_evidence * (signals.evidence_strength - 0.5) * 2.0
        if signals.self_consistency is not None:
            z += w.w_consistency * (signals.self_consistency - 0.5) * 2.0

        p = _sigmoid(z / w.temperature)

        if not signals.grounded:
            p = min(p, w.ungrounded_cap)

        return AnswerConfidence(score=float(max(0.0, min(1.0, p))), signals=signals)

    def fit_temperature(
        self,
        predictions: list[float],
        outcomes: list[bool],
        grid: tuple[float, float, int] = (0.25, 4.0, 32),
    ) -> float:
        """Fit the scaling temperature on labelled (confidence, correct) pairs.

        Minimises negative log-likelihood over a temperature grid — a robust,
        dependency-free stand-in for full Platt scaling that directly improves
        calibration (and hence the competition's over/under-confidence penalty).

        Args:
            predictions: Pre-temperature calibrated probabilities from a dev run.
            outcomes: Ground-truth correctness for each prediction.
            grid: ``(low, high, steps)`` temperature search grid.

        Returns:
            The best temperature found (also stored on ``self.weights``).
        """
        if not predictions or len(predictions) != len(outcomes):
            return self.weights.temperature

        low, high, steps = grid
        best_t, best_nll = self.weights.temperature, float("inf")
        for s in range(steps):
            t = low + (high - low) * s / max(1, steps - 1)
            nll = 0.0
            for p, y in zip(predictions, outcomes, strict=True):
                scaled = _sigmoid(_logit(p) / t)
                scaled = min(1 - 1e-6, max(1e-6, scaled))
                nll -= math.log(scaled) if y else math.log(1.0 - scaled)
            if nll < best_nll:
                best_nll, best_t = nll, t
        self.weights.temperature = best_t
        return best_t


def expected_calibration_error(
    predictions: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for monitoring.

    Args:
        predictions: Predicted confidences in ``[0, 1]``.
        outcomes: Ground-truth correctness.
        n_bins: Number of equal-width probability bins.

    Returns:
        ECE in ``[0, 1]`` (lower is better).
    """
    if not predictions:
        return 0.0
    total = len(predictions)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(predictions) if (p > lo or b == 0) and p <= hi]
        if not idx:
            continue
        avg_conf = sum(predictions[i] for i in idx) / len(idx)
        avg_acc = sum(1.0 for i in idx if outcomes[i]) / len(idx)
        ece += (len(idx) / total) * abs(avg_conf - avg_acc)
    return ece
