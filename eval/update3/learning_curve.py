"""
LearningCurveAnalyzer
=====================
Data-sufficiency diagnostics with two modes plus a resilience stress test. Kept
deliberately simple: each method computes a curve, returns it as a table and a
single Plotly figure, and adds a one-line factual summary. No curve-fitting and
no regime/drift classification — read the curve and decide.

random_curve   : score vs training-set SIZE (train on growing random subsets).
                 Shows whether more data of the same kind still helps and how big
                 the train/validation gap is.

temporal_curve : score vs LENGTH OF HISTORY, counted in whole calendar periods
                 (months by default). The training window grows backward one
                 period at a time (last 1 month, last 2 months, ...) and each
                 look-back is scored on the SAME explicit test set
                 (ds.X_test/ds.y_test), exactly like random_curve. Under drift
                 the curve peaks at a limited look-back and then declines — a
                 sign that old data is stale. The training time column may be a
                 datetime column or ordered integer period ids (month numbers
                 1..T).

stress_curve   : refit-free resilience to worst-case drift (delegates to
                 modeva_ext.resilience_stress_curve).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone

from .data import Dataset, EvalConfig
from .metrics import headline_name, score
from .modeva_ext import resilience_stress_curve
from . import viz


@dataclass
class LearningCurveReport:
    """Container returned by every mode.

    mode    : "random" | "temporal" | "stress".
    table   : the curve as a tidy DataFrame (also the source of `fig`).
    fig     : a single Plotly figure.
    verdict : a one-line factual summary (no regime/drift labels).
    details : raw objects for programmatic use (always includes `curve`).
    """
    mode: str
    table: pd.DataFrame
    fig: go.Figure
    verdict: str
    details: dict = field(default_factory=dict)

    def show(self):
        self.fig.show()


class LearningCurveAnalyzer:
    def __init__(self, config: Optional[EvalConfig] = None):
        self.cfg = config or EvalConfig()
        self.rng = np.random.default_rng(self.cfg.random_state)

    # ===================================================================== #
    # A) RANDOM MODE — score vs training-set size                           #
    # ===================================================================== #
    def random_curve(self, model, ds: Dataset,
                     fractions=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
                     repeats: int = 3) -> LearningCurveReport:
        """Train the model on growing random subsets of the training set and
        score train (in-sample) and validation (the held-out test set).

        Parameters
        ----------
        fractions : subset sizes as fractions of the full training set.
        repeats   : random subsets averaged per fraction (reduces noise).

        Returns a `LearningCurveReport` (mode "random") whose `table` has one row
        per fraction: n, train score, validation score, validation std.
        """
        Xtr, ytr = ds.X_train.to_numpy(float), ds.y_train.to_numpy()
        Xte, yte = ds.X_test.to_numpy(float), ds.y_test.to_numpy()
        n_full = len(ytr)
        name = headline_name(ds.task)

        rows = []
        for frac in fractions:
            n = max(20, int(frac * n_full))
            tr_scores, va_scores = [], []
            for _ in range(repeats):
                idx = self.rng.choice(n_full, size=min(n, n_full), replace=False)
                m = clone(model).fit(Xtr[idx], ytr[idx])
                tr_scores.append(score(ytr[idx], m, Xtr[idx], ds.task))
                va_scores.append(score(yte, m, Xte, ds.task))
            rows.append(dict(n=n,
                             train=round(float(np.mean(tr_scores)), 4),
                             val=round(float(np.mean(va_scores)), 4),
                             val_std=round(float(np.std(va_scores)), 4)))
        curve = pd.DataFrame(rows)

        # single figure: train and validation curves (validation with error band)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["n"], y=curve["train"], mode="lines+markers",
                                 name=f"train {name}", line=dict(color=viz.MUTED)))
        fig.add_trace(go.Scatter(
            x=curve["n"], y=curve["val"], mode="lines+markers",
            name=f"validation {name}", line=dict(color=viz.ACCENT),
            error_y=dict(type="data", array=curve["val_std"], visible=True)))
        fig.update_xaxes(title="training samples (n)")
        fig.update_yaxes(title=name)
        fig = viz._base(fig, "Learning curve — score vs training-set size", 440)

        gap = curve["train"].iloc[-1] - curve["val"].iloc[-1]
        table = curve.rename(columns={"train": f"train {name}",
                                      "val": f"val {name}", "val_std": "val ±"})
        verdict = (f"At n={n_full}: train {name}={curve['train'].iloc[-1]:.3f}, "
                   f"val {name}={curve['val'].iloc[-1]:.3f} (gap {gap:+.3f}); "
                   f"val moves {curve['val'].iloc[0]:.3f}→{curve['val'].iloc[-1]:.3f} "
                   "as n grows.")
        return LearningCurveReport("random", table, fig, verdict,
                                   details=dict(curve=curve))

    # ===================================================================== #
    # B) TEMPORAL MODE — score vs length of history (in calendar periods)    #
    # ===================================================================== #
    def temporal_curve(self, model, ds: Dataset, *, freq: str = "M",
                       min_period_samples: int = 30) -> LearningCurveReport:
        """Grow the training window backward one calendar period at a time and
        score each look-back on the **explicit test set** `ds.X_test/ds.y_test`.

        Like `random_curve`, the evaluation target is fixed (the held-out test
        set), so every point is comparable; only the training window changes —
        here it is the most recent L periods of the training set rather than a
        random subset. This matches the out-of-time framing: train = history,
        test = the future period you want to predict. Build the Dataset so the
        test set is that target (e.g. the most recent block).

        The training time column (Dataset built with `time_col=...`) may be:
          * a datetime column, bucketed into calendar periods at `freq`
            ("M" month [default], "W" week, "Q" quarter, "Y" year); or
          * ordered integer period ids, e.g. month numbers 1..T — each distinct
            value is one period, in ascending order.

        Parameters
        ----------
        freq               : calendar period size (used only for datetime input).
        min_period_samples : skip a look-back whose training window has fewer rows.

        Returns a `LearningCurveReport` (mode "temporal") whose `table` has one
        row per look-back: length in periods, the covered period range, training
        sample count, and the test score.
        """
        if ds.time_train is None:
            raise ValueError(
                "temporal_curve needs a time column on the TRAINING data: build "
                "the Dataset with time_col=... (a datetime column or integer "
                "period ids 1..T).")

        pid, labels, unit = self._periods(ds.time_train, freq)
        P = len(labels)
        if P < 2:
            raise ValueError(f"Need at least 2 training periods; found {P} {unit}(s).")

        Xtr, ytr = ds.X_train.to_numpy(float), ds.y_train.to_numpy()
        Xte, yte = ds.X_test.to_numpy(float), ds.y_test.to_numpy()   # fixed target
        name = headline_name(ds.task)
        last_tp = P - 1                              # newest training period

        rows = []
        for L in range(1, P + 1):                    # look-back length in periods
            lo = last_tp - L + 1                      # oldest period included
            m = pid >= lo                             # most recent L periods of train
            if m.sum() < min_period_samples:
                continue
            mdl = clone(model).fit(Xtr[m], ytr[m])
            rows.append(dict(lookback=L,
                             from_period=labels[lo], to_period=labels[last_tp],
                             n_samples=int(m.sum()),
                             test=round(float(score(yte, mdl, Xte, ds.task)), 4)))
        curve = pd.DataFrame(rows)
        if curve.empty:
            raise ValueError("No look-back window met min_period_samples; lower it "
                             "or widen the period (e.g. freq='Q').")

        best = curve.loc[curve["test"].idxmax()]

        # single figure: test score vs history length, best point starred
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve["lookback"], y=curve["test"], mode="lines+markers",
            name=f"test {name}", line=dict(color=viz.ACCENT),
            customdata=np.stack([curve["from_period"], curve["to_period"],
                                 curve["n_samples"]], axis=-1),
            hovertemplate=("look-back %{x} " + unit + "s "
                           "(%{customdata[0]}→%{customdata[1]}), n=%{customdata[2]}"
                           "<br>" + name + "=%{y:.3f}<extra></extra>")))
        fig.add_trace(go.Scatter(
            x=[best["lookback"]], y=[best["test"]], mode="markers", name="best",
            marker=dict(color=viz.GOOD, size=13, symbol="star"), hoverinfo="skip"))
        fig.update_xaxes(title=f"training window = most recent N {unit}s "
                               "(grows backward)", dtick=1)
        fig.update_yaxes(title=f"test {name}")
        fig = viz._base(fig, "Temporal learning curve — history length vs performance",
                        440)

        table = curve.rename(columns={"lookback": f"look-back ({unit}s)",
                                      "from_period": "from", "to_period": "to",
                                      "n_samples": "n", "test": name})
        verdict = (f"Test {name} ranges {curve['test'].min():.3f}–"
                   f"{curve['test'].max():.3f} across look-backs of 1–{P} "
                   f"{unit}s; highest at a {int(best['lookback'])}-{unit} window.")
        return LearningCurveReport("temporal", table, fig, verdict,
                                   details=dict(curve=curve, unit=unit,
                                                periods=labels,
                                                best_lookback=int(best["lookback"])))

    @staticmethod
    def _periods(time_values, freq: str):
        """Map each row to an ordered integer period id (0..P-1).

        Datetime input is bucketed into calendar periods at `freq`. Numeric input
        is treated as discrete ordered period ids (e.g. month numbers 1..T), so
        each distinct value becomes one period. Continuous numeric timestamps are
        rejected with a clear message (bucket them first or pass a datetime).

        Returns
        -------
        pid    : int array, period index of each row.
        labels : ordered list of period names.
        unit   : word used on the axis ("month", "week", ...).
        """
        s = pd.Series(np.asarray(time_values))
        if pd.api.types.is_datetime64_any_dtype(s):
            row_key = pd.to_datetime(s).dt.to_period(freq)
            unit = {"M": "month", "W": "week", "Q": "quarter",
                    "Y": "year"}.get(freq, "period")
        else:
            row_key = s                                   # discrete ordered ids
            unit = "month" if freq == "M" else "period"
            if s.nunique() > 240:
                raise ValueError(
                    f"time_col has {s.nunique()} distinct numeric values and looks "
                    "continuous. Pass integer period ids (e.g. month numbers 1..T) "
                    "or a datetime column, or bucket your timestamps first.")
        keys = sorted(pd.unique(row_key.dropna()))
        index = {k: i for i, k in enumerate(keys)}
        pid = row_key.map(index).to_numpy()
        labels = [str(k) for k in keys]
        return pid, labels, unit

    # ===================================================================== #
    # C) STRESS MODE — refit-free resilience to worst-case drift             #
    # ===================================================================== #
    def stress_curve(self, model, ds: Dataset, scenario: str = "worst-sample",
                     levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)) -> LearningCurveReport:
        """Refit-free resilience stress test (see modeva_ext). Progressively
        oversamples a hard subpopulation of the test set and tracks the metric.
        `scenario` in {worst-sample, worst-cluster, edge, hard-sample}."""
        curve, fig, verdict = resilience_stress_curve(
            model, ds, scenario=scenario, levels=levels,
            n_clusters=self.cfg.n_windows, random_state=self.cfg.random_state)
        return LearningCurveReport("stress", curve, fig, verdict,
                                   details=dict(curve=curve, scenario=scenario))
