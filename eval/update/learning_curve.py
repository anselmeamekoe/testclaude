"""
LearningCurveAnalyzer
=====================
Two things sklearn.learning_curve does NOT give you:

A) SATURATION / SUFFICIENCY (random mode)
   We fit an inverse-power law  err(n) = c + a * n^(-b)  to the *validation*
   error and extrapolate. That yields actionable numbers:
     - asymptotic error c (the floor this model+data can reach),
     - how much error is still on the table,
     - how many more samples to capture 90% of the remaining gain,
     - a regime label: high-bias / high-variance / saturated.
   So instead of "here's a curve", you get "you are data-limited, ~2x the data
   buys you ~30% of the remaining error" or "adding data won't help, raise
   capacity".

B) TEMPORAL / DRIFT (temporal mode)  -- the fraud-detection case
   Random subsampling silently assumes i.i.d. data. Under drift, *older* rows
   can be worse than useless. We fix the validation block to the MOST RECENT
   data and grow the training window BACKWARD in time. If the validation score
   peaks at a limited look-back and then decays as more history is added, older
   data is detrimental — we report the optimal look-back window W*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.base import clone

from .data import Dataset, EvalConfig, Task
from .metrics import headline_name, score
from .modeva_ext import drift_drivers, resilience_stress_curve
from . import viz


def _err(headline: float, task: Task) -> float:
    """Turn a 'higher is better' score into an error in [0,1]."""
    return float(np.clip(1 - headline, 0, 1))


def _power_law(n, c, a, b):
    return c + a * np.power(n, -b)


# --------------------------------------------------------------------------- #
@dataclass
class LearningCurveReport:
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
    # A) RANDOM MODE with saturation extrapolation                          #
    # ===================================================================== #
    def random_curve(self, model, ds: Dataset,
                     fractions=(0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0),
                     repeats: int = 3) -> LearningCurveReport:
        Xtr, ytr = ds.X_train.to_numpy(float), ds.y_train.to_numpy()
        Xte, yte = ds.X_test.to_numpy(float), ds.y_test.to_numpy()
        n_full = len(ytr)
        rows = []
        for frac in fractions:
            n = max(20, int(frac * n_full))
            tr_s, va_s = [], []
            for _ in range(repeats):
                idx = self.rng.choice(n_full, size=min(n, n_full), replace=False)
                m = clone(model).fit(Xtr[idx], ytr[idx])
                tr_s.append(score(ytr[idx], m, Xtr[idx], ds.task))
                va_s.append(score(yte, m, Xte, ds.task))
            rows.append(dict(n=n, train=np.mean(tr_s), val=np.mean(va_s),
                             val_std=np.std(va_s)))
        curve = pd.DataFrame(rows)

        # Fit saturation law on VALIDATION error.
        ns = curve["n"].to_numpy(float)
        val_err = 1 - curve["val"].to_numpy(float)
        c_hat, extra = self._fit_saturation(ns, val_err, n_full)

        fig = self._random_fig(curve, ds.task, extra)
        table, verdict = self._random_table_verdict(curve, ds.task, c_hat, extra, n_full)
        return LearningCurveReport("random", table, fig, verdict,
                                   details=dict(curve=curve, **extra))

    def _fit_saturation(self, ns, val_err, n_full):
        try:
            p0 = [max(val_err.min(), 1e-3), max(val_err.max() - val_err.min(), 1e-3), 0.5]
            bounds = ([0, 0, 0.05], [1, 5, 3.0])
            popt, _ = curve_fit(_power_law, ns, val_err, p0=p0, bounds=bounds, maxfev=8000)
            c, a, b = popt
            cur_err = _power_law(n_full, *popt)
            floor = c
            remaining = max(cur_err - floor, 0)
            # n to capture 90% of the remaining gain from n_full
            target = floor + 0.1 * remaining
            n_needed = (a / max(target - c, 1e-6)) ** (1 / b) if target > c else np.inf
            details = dict(fit_ok=True, c=float(c), a=float(a), b=float(b),
                           cur_err=float(cur_err), floor=float(floor),
                           remaining=float(remaining),
                           n_for_90pct=float(min(n_needed, 1e12)))
            return float(c), details
        except Exception as e:  # pragma: no cover
            return float(val_err.min()), dict(fit_ok=False, reason=str(e))

    def _random_fig(self, curve, task, extra):
        name = headline_name(task)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["n"], y=curve["train"], name=f"train {name}",
                                 mode="lines+markers", line=dict(color=viz.MUTED)))
        fig.add_trace(go.Scatter(
            x=curve["n"], y=curve["val"], name=f"validation {name}",
            mode="lines+markers", line=dict(color=viz.ACCENT),
            error_y=dict(type="data", array=curve["val_std"], visible=True)))
        if extra.get("fit_ok"):
            xs = np.linspace(curve["n"].min(), curve["n"].max() * 3, 100)
            ys = 1 - _power_law(xs, extra["c"], extra["a"], extra["b"])
            fig.add_trace(go.Scatter(x=xs, y=ys, name="extrapolation",
                                     mode="lines", line=dict(color=viz.WARN, dash="dash")))
            fig.add_hline(y=1 - extra["floor"], line=dict(color=viz.GOOD, dash="dot"),
                          annotation_text="estimated ceiling")
        fig.update_xaxes(title="training samples (n)")
        fig.update_yaxes(title=name)
        return viz._base(fig, "Learning curve with saturation extrapolation", height=460)

    def _random_table_verdict(self, curve, task, c_hat, extra, n_full):
        name = headline_name(task)
        last = curve.iloc[-1]
        gap = last["train"] - last["val"]
        rows = [
            ("current n", f"{int(last['n'])}", ""),
            (f"train {name}", f"{last['train']:.3f}", ""),
            (f"val {name}", f"{last['val']:.3f}", ""),
            ("train-val gap", f"{gap:.3f}", "variance signal"),
        ]
        verdict_bits = []
        if extra.get("fit_ok"):
            ceil = 1 - extra["floor"]
            rows += [
                (f"estimated ceiling {name}", f"{ceil:.3f}", "best this model can do"),
                ("remaining error to floor", f"{extra['remaining']:.3f}", ""),
                ("n for 90% of remaining gain", f"{extra['n_for_90pct']:.0f}",
                 f"~{extra['n_for_90pct']/max(n_full,1):.1f}x current data"),
            ]
            if gap > 0.08 and extra["remaining"] > 0.01:
                verdict_bits.append(
                    "HIGH-VARIANCE / data-limited: a visible train-val gap and the "
                    "curve is still descending. More data should help — roughly "
                    f"{extra['n_for_90pct']/max(n_full,1):.1f}x buys most of the "
                    "remaining error.")
            elif extra["remaining"] <= 0.01:
                verdict_bits.append(
                    "SATURATED: validation error has essentially reached its floor. "
                    "More rows of the same kind won't help — raise model capacity or "
                    "add informative features instead.")
            else:
                verdict_bits.append(
                    "MILD returns: the curve is flattening; extra data yields small "
                    "gains. Weigh collection cost against the modest expected lift.")
            if last["train"] - last["val"] < 0.03 and last["val"] < 0.6 \
                    and task == Task.CLASSIFICATION:
                verdict_bits.append(
                    "Train and val are both low and close → HIGH-BIAS: the model "
                    "underfits; more data alone won't fix it.")
        else:
            verdict_bits.append("Saturation fit failed; read the curve directly.")
        return pd.DataFrame(rows, columns=["quantity", "value", "note"]), " ".join(verdict_bits)

    # ===================================================================== #
    # B) TEMPORAL MODE — is older history useful or detrimental?            #
    # ===================================================================== #
    def temporal_curve(self, model, ds: Dataset,
                       val_frac: float = 0.2) -> LearningCurveReport:
        if ds.time_train is None:
            raise ValueError("temporal_curve needs a time column (set config.time_col "
                             "and build the Dataset with time_col=...).")
        # Order the *training* pool by time; hold out the most recent val_frac as
        # the validation block (the 'present' we care about predicting).
        order = np.argsort(ds.time_train.to_numpy())
        X = ds.X_train.to_numpy(float)[order]
        y = ds.y_train.to_numpy()[order]
        t = ds.time_train.to_numpy()[order]
        n = len(y)
        n_val = max(30, int(val_frac * n))
        X_tr_pool, y_tr_pool = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]
        pool_n = len(y_tr_pool)

        # Grow the training window BACKWARD from the boundary: newest-first.
        windows = np.unique(np.linspace(max(30, pool_n // self.cfg.n_windows),
                                        pool_n, self.cfg.n_windows).astype(int))
        rows = []
        for w in windows:
            Xw, yw = X_tr_pool[-w:], y_tr_pool[-w:]   # most recent w samples
            m = clone(model).fit(Xw, yw)
            s = score(y_val, m, X_val, ds.task)
            oldest_ts, newest_ts = t[pool_n - w], t[pool_n - 1]
            rows.append(dict(window=int(w), val=s,
                             oldest=oldest_ts, newest=newest_ts))
        tc = pd.DataFrame(rows)

        # Contrast: newest-N vs oldest-N of equal size on the same val block.
        k = min(windows)
        m_new = clone(model).fit(X_tr_pool[-k:], y_tr_pool[-k:])
        m_old = clone(model).fit(X_tr_pool[:k], y_tr_pool[:k])
        s_new = score(y_val, m_new, X_val, ds.task)
        s_old = score(y_val, m_old, X_val, ds.task)

        best_idx = int(tc["val"].idxmax())
        w_star = int(tc.loc[best_idx, "window"])
        s_star = float(tc.loc[best_idx, "val"])
        s_full = float(tc.iloc[-1]["val"])
        decay = s_star - s_full  # >0 => adding oldest history HURTS

        # WHICH features drift? Compare oldest block vs recent validation block.
        feats = ds.feature_names
        drivers = drift_drivers(pd.DataFrame(X_tr_pool[:k], columns=feats),
                                pd.DataFrame(X_val, columns=feats), feats)

        fig = self._temporal_fig(tc, ds.task, w_star, pool_n)
        table, verdict = self._temporal_table_verdict(
            ds.task, k, s_new, s_old, w_star, s_star, s_full, decay, pool_n)
        # append covariate-drift drivers (PSI catches feature drift, not concept drift)
        top = drivers[drivers["psi"] > 0.1]
        if not top.empty:
            names = ", ".join(f"{r.feature} (PSI={r.psi})" for r in top.head(3).itertuples())
            verdict += f" Feature drift (covariate) is led by: {names}."
        else:
            verdict += (" No strong covariate (feature-distribution) drift — any drift "
                        "above is concept drift in P(y|x), which the temporal refit "
                        "curve captures but PSI/KS cannot.")
        return LearningCurveReport("temporal", table, fig, verdict,
                                   details=dict(curve=tc, w_star=w_star,
                                                s_star=s_star, s_full=s_full,
                                                decay=decay, s_new=s_new, s_old=s_old,
                                                drift_drivers=drivers))

    # ===================================================================== #
    # C) STRESS MODE — resilience to worst-case drift (refit-free, no time)  #
    # ===================================================================== #
    def stress_curve(self, model, ds: Dataset, scenario: str = "worst-sample",
                     levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)) -> LearningCurveReport:
        """Refit-free resilience stress test (Modeva-inspired). Progressively
        oversamples a hard subpopulation of the test set and tracks the metric.
        `scenario` ∈ {worst-sample, worst-cluster, edge, hard-sample}."""
        curve, fig, verdict = resilience_stress_curve(
            model, ds, scenario=scenario, levels=levels,
            n_clusters=self.cfg.n_windows, random_state=self.cfg.random_state)
        return LearningCurveReport("stress", curve, fig, verdict,
                                   details=dict(curve=curve, scenario=scenario))

    def _temporal_fig(self, tc, task, w_star, pool_n):
        name = headline_name(task)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tc["window"], y=tc["val"], mode="lines+markers",
            name=f"val {name} on most-recent block", line=dict(color=viz.ACCENT)))
        fig.add_vline(x=w_star, line=dict(color=viz.GOOD, dash="dash"),
                      annotation_text=f"optimal look-back W*={w_star}")
        fig.add_annotation(x=pool_n, y=tc["val"].iloc[-1],
                           text="all history", showarrow=True, arrowhead=2)
        fig.update_xaxes(title="training window = most recent N samples (grows backward)")
        fig.update_yaxes(title=f"validation {name}")
        return viz._base(fig, "Temporal learning curve — does older history help?",
                         height=460)

    def _temporal_table_verdict(self, task, k, s_new, s_old, w_star, s_star,
                                s_full, decay, pool_n):
        name = headline_name(task)
        rows = [
            (f"newest {k} rows → val {name}", f"{s_new:.3f}", "recency value"),
            (f"oldest {k} rows → val {name}", f"{s_old:.3f}", "antiquity value"),
            ("recency advantage", f"{s_new - s_old:+.3f}", "new minus old"),
            ("optimal look-back W*", f"{w_star}", f"of {pool_n} available"),
            (f"val {name} at W*", f"{s_star:.3f}", ""),
            (f"val {name} using ALL history", f"{s_full:.3f}", ""),
            ("cost of using all history", f"{decay:+.3f}", "W* minus all"),
        ]
        if decay > 0.01 and w_star < 0.8 * pool_n:
            verdict = (f"DRIFT DETECTED. Validation peaks at ~{w_star} recent samples "
                       f"and DEGRADES by {decay:.3f} {name} when the full history is "
                       "added. Older data is stale/detrimental — cap the look-back "
                       f"around W*={w_star} (or add recency weighting / drift features). "
                       f"Recency advantage of newest-vs-oldest equal blocks is "
                       f"{s_new - s_old:+.3f}.")
        elif s_new - s_old > 0.03:
            verdict = ("MILD DRIFT: recent data is clearly more predictive than old "
                       "data, but adding history still doesn't hurt the aggregate. "
                       "Keep all data but consider time-decay weighting.")
        else:
            verdict = ("STABLE: more history keeps helping (or is neutral) and old "
                       "data is about as useful as new. Use all available history; "
                       "the process looks stationary over this range.")
        return pd.DataFrame(rows, columns=["quantity", "value", "note"]), verdict
