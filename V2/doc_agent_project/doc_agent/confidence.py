"""Confidence calibration — converting evidence into a trustworthy probability.

Calibration is the heart of this challenge: a calibrated agent's stated confidence
should match its actual accuracy (an answer given at 0.7 should be right ~70% of
the time). Both overconfidence and underconfidence are penalized, so we do **not**
trust the model's self-reported number alone — models are systematically
over-confident. Instead we blend three complementary signals:

1. **Verbalized confidence** — the model's own estimate. Useful but biased.
2. **Retrieval support** — did we actually find evidence that grounds the claim?
   Strong, on-topic retrieval raises confidence; thin or off-topic retrieval lowers it.
3. **Execution signal** — for questions flagged ``requires_code_execution``, an
   answer *verified by running code* is far more trustworthy than one inferred from
   reading. Conversely, if execution was required but impossible/failed, confidence
   is pulled down hard and the answer is flagged information-incomplete.

The blended score is then passed through an optional monotonic calibration map
(Platt scaling) that can be *fit* on labeled dev data to correct residual bias.
Until fit, an identity map is used, so the system is calibrated-by-construction and
improves further if you give it ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Settings
from .models import ConfidenceSignals


@dataclass
class PlattScaler:
    """A 1-parameter logistic calibration map ``sigmoid(a * logit(p) + b)``.

    Defaults (``a=1, b=0``) make it the identity on probabilities, so it is safe to
    apply unconditionally. Call :meth:`fit` with held-out predictions and binary
    correctness labels to learn ``a``/``b`` and remove residual mis-calibration.

    Attributes:
        a: Slope in logit space (``>1`` sharpens, ``<1`` softens confidence).
        b: Bias in logit space (shifts the whole curve up/down).
    """

    a: float = 1.0
    b: float = 0.0

    @staticmethod
    def _logit(p: float, eps: float = 1e-6) -> float:
        """Numerically safe logit of a probability clamped away from 0/1."""
        p = min(1 - eps, max(eps, p))
        return math.log(p / (1 - p))

    def transform(self, p: float) -> float:
        """Apply the calibration map to a single probability.

        Args:
            p: Raw blended confidence in ``[0, 1]``.

        Returns:
            The calibrated probability in ``[0, 1]``.
        """
        z = self.a * self._logit(p) + self.b
        return 1.0 / (1.0 + math.exp(-z))

    def fit(
        self,
        probs: list[float],
        labels: list[int],
        lr: float = 0.1,
        epochs: int = 500,
    ) -> "PlattScaler":
        """Fit ``a``/``b`` by gradient descent on binary cross-entropy.

        This is the optional "learn from dev data" path: given the agent's raw
        blended confidences and whether each answer was actually correct, it tunes
        the map so stated confidence tracks empirical accuracy.

        Args:
            probs: Raw blended confidences for dev examples.
            labels: Matching correctness labels (1 = correct, 0 = wrong).
            lr: Learning rate.
            epochs: Gradient steps.

        Returns:
            This scaler, with updated ``a``/``b``.
        """
        if not probs:
            return self
        logits = [self._logit(p) for p in probs]
        for _ in range(epochs):
            ga = gb = 0.0
            for z, y in zip(logits, labels):
                pred = 1.0 / (1.0 + math.exp(-(self.a * z + self.b)))
                err = pred - y
                ga += err * z
                gb += err
            n = len(logits)
            self.a -= lr * ga / n
            self.b -= lr * gb / n
        return self


class ConfidenceCalibrator:
    """Blends :class:`ConfidenceSignals` into a single calibrated confidence.

    The weighting is configured via :class:`~doc_agent.config.Settings`; the
    blending logic encodes the domain rules that keep the score honest (execution
    gating, missing-information flooring, retrieval grounding).
    """

    def __init__(self, settings: Settings, scaler: PlattScaler | None = None) -> None:
        """Initialize with weights from settings and an optional fitted scaler.

        Args:
            settings: Source of the blend weights (``w_verbalized`` etc.).
            scaler: Optional pre-fit :class:`PlattScaler`; identity if ``None``.
        """
        self.settings = settings
        self.scaler = scaler or PlattScaler()

    def calibrate(self, signals: ConfidenceSignals) -> float:
        """Compute the final calibrated confidence for one answer.

        The procedure:

        1. If the agent reports information is incomplete, hard-cap confidence low —
           we should never be confident about something we couldn't ground.
        2. Compute an execution sub-score that depends on whether execution was
           *required*, *attempted*, and *succeeded*.
        3. Take a weighted average of verbalized, retrieval-support, and execution
           sub-scores (weights normalized so they need not sum to 1).
        4. Nudge toward the self-consistency agreement if available.
        5. Pass the blend through the (optionally fitted) Platt map.

        Args:
            signals: The gathered signals for this answer.

        Returns:
            A calibrated confidence in ``[0, 1]``.
        """
        s = self.settings

        # (1) Missing information dominates everything else.
        if not signals.information_complete:
            base = 0.15
            return round(self.scaler.transform(base), 4)

        # (2) Execution sub-score.
        if signals.execution_required:
            if not signals.execution_attempted:
                # Needed to run code but didn't: strong penalty.
                exec_score = 0.25
            elif signals.execution_succeeded:
                exec_score = 0.95  # verified by a clean run — very trustworthy
            else:
                exec_score = 0.30  # tried to run and it failed
        else:
            # Execution not required: neutral-to-positive, slightly rewarded if a
            # run happened anyway and succeeded.
            if signals.execution_attempted and signals.execution_succeeded:
                exec_score = 0.85
            else:
                exec_score = 0.60

        # (3) Weighted blend.
        w_sum = s.w_verbalized + s.w_retrieval + s.w_execution
        blend = (
            s.w_verbalized * signals.verbalized_confidence
            + s.w_retrieval * signals.retrieval_support
            + s.w_execution * exec_score
        ) / max(w_sum, 1e-9)

        # (4) Self-consistency nudge: pull blend toward agreement if measured.
        if signals.self_consistency is not None:
            blend = 0.7 * blend + 0.3 * signals.self_consistency

        # Keep within a sane band. The cap encodes the core calibration rule:
        # we only allow near-certainty when an execution-required answer was
        # actually verified by a clean run. Execution-required questions that were
        # never run (or failed) are capped low, because reading code is not a
        # substitute for observing runtime behaviour the question demanded.
        if signals.execution_required:
            if signals.execution_succeeded:
                blend = min(blend, 0.97)
            else:
                blend = min(blend, 0.50)
        else:
            blend = min(blend, 0.92)

        # (5) Final calibration map.
        return round(self.scaler.transform(min(1.0, max(0.0, blend))), 4)

    @staticmethod
    def retrieval_support_from_scores(scores: list[float], expected: int = 3) -> float:
        """Summarize a list of retrieval scores into one support value in ``[0, 1]``.

        Rewards both *quality* (mean of the top scores) and *coverage* (did we find
        as many strong chunks as we'd expect). Returns 0 when nothing was retrieved,
        which correctly drags confidence down for ungrounded answers.

        Args:
            scores: Relevance scores of retrieved evidence used for the answer.
            expected: How many solid pieces of evidence we'd hope to find.

        Returns:
            A support score in ``[0, 1]``.
        """
        if not scores:
            return 0.0
        top = sorted(scores, reverse=True)[:expected]
        quality = sum(top) / len(top)
        coverage = min(1.0, len(top) / expected)
        return round(0.7 * quality + 0.3 * coverage, 4)
