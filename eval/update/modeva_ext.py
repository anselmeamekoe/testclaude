"""
modeva_ext.py — refit-free diagnostics inspired by Modeva's testing suite.

Design principle borrowed from Modeva: you can surface most overfitting / drift
signals by SCORING A FIXED, ALREADY-TRAINED MODEL on cleverly chosen
perturbations, slices or subsets — no refitting. That makes every function here
O(predict), i.e. orders of magnitude cheaper than our refit-based indicators
(edf, memorization, bootstrap variance, learning curves).

Cost legend (n rows, d features, T = one predict over n rows, F = one fit):
  input_sensitivity      ~ k*T           (k≈8 predicts, NO fit)
  drift_drivers          ~ O(d*n log n)  (sorting/hist, NO model call)
  adaptive_weak_threshold~ O(#slices)    (free)
  resilience_stress_curve~ (#levels)*T   (+1 KMeans / +1 cov-inverse, NO fit)
Compare: our effective_dof ≈ 12F, prediction_variance ≈ 20F.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ks_2samp, wasserstein_distance

from .data import Dataset, EvalConfig, Task
from .metrics import _proba, headline_name, per_sample_loss, score
from . import viz


def _scalar_output(model, X, task: Task) -> np.ndarray:
    if task == Task.CLASSIFICATION:
        p = _proba(model, X)
        return p[:, 1] if p.shape[1] == 2 else p.max(axis=1)
    return np.asarray(model.predict(X), dtype=float)


# --------------------------------------------------------------------------- #
# 1) INPUT SENSITIVITY  (local Lipschitz)  — a refit-free complexity signal    #
#    Modeva: overfit models have larger ||grad f|| / higher local Lipschitz.   #
#    Cheaper substitute/complement for our bootstrap prediction_variance.      #
# --------------------------------------------------------------------------- #
def input_sensitivity(model, X: np.ndarray, task: Task, *, n_dirs: int = 8,
                      h: float = 0.05, random_state: int = 0) -> dict:
    """Estimate mean local Lipschitz constant E[ |f(x+δ)-f(x)| / ||δ|| ] using
    random directions (SPSA-style). No refit: n_dirs forward passes only.

    Returns {'lipschitz_mean', 'lipschitz_p95'} on the model's scalar output.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0) + 1e-9
    f0 = _scalar_output(model, X, task)
    out_scale = float(np.std(f0)) + 1e-9
    ratios = np.zeros((n_dirs, len(X)))
    abs_moves = np.zeros(n_dirs)
    for k in range(n_dirs):
        delta = rng.normal(0, 1, size=X.shape) * (h * sd)
        fh = _scalar_output(model, X + delta, task)
        norm = np.linalg.norm(delta, axis=1) + 1e-12
        ratios[k] = np.abs(fh - f0) / norm
        abs_moves[k] = np.mean(np.abs(fh - f0))
    per_sample = ratios.mean(axis=0)
    # unit-free: how much of the output's own spread an h-perturbation moves
    relative_jaggedness = float(abs_moves.mean() / out_scale)
    return dict(lipschitz_mean=float(per_sample.mean()),
                lipschitz_p95=float(np.percentile(per_sample, 95)),
                relative_jaggedness=relative_jaggedness,
                per_sample=per_sample)


# --------------------------------------------------------------------------- #
# 2) DRIFT DRIVERS  — which FEATURES shift between two windows                  #
#    Modeva 'resilience': rank features by PSI / KS / Wasserstein.             #
#    For our temporal_curve: explain WHAT drifts once drift is detected.       #
# --------------------------------------------------------------------------- #
def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-6
    e = np.histogram(ref, edges)[0] / (len(ref) + eps) + eps
    a = np.histogram(cur, edges)[0] / (len(cur) + eps) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def drift_drivers(X_ref: pd.DataFrame, X_cur: pd.DataFrame,
                  feature_names: list[str], top: Optional[int] = None) -> pd.DataFrame:
    """Rank features by distribution shift between a reference and current window.
    Cheap: histograms + sort, no model calls."""
    rows = []
    for f in feature_names:
        r = X_ref[f].to_numpy(float)
        c = X_cur[f].to_numpy(float)
        rows.append(dict(
            feature=f,
            psi=round(_psi(r, c), 3),
            ks=round(float(ks_2samp(r, c).statistic), 3),
            wasserstein=round(float(wasserstein_distance(r, c)), 3),
            mean_shift=round(float(c.mean() - r.mean()), 3),
        ))
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    df["severity"] = pd.cut(df["psi"], [-1, 0.1, 0.25, np.inf],
                            labels=["low", "moderate", "high"])
    return df.head(top) if top else df


# --------------------------------------------------------------------------- #
# 3) ADAPTIVE WEAK THRESHOLD  — Modeva: flag regions where gap > mu + beta*std #
#    Replaces our fixed loss_ratio>1.25 gate with a data-driven one. Free.     #
# --------------------------------------------------------------------------- #
def adaptive_weak_threshold(values: np.ndarray, beta: float = 1.5) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values) + beta * np.std(values))


# --------------------------------------------------------------------------- #
# 4) RESILIENCE STRESS CURVE  — Modeva 'resilience' worst-case drift scenarios  #
#    Refit-free stress test that needs NO time column (complements temporal).  #
# --------------------------------------------------------------------------- #
Scenario = Literal["worst-sample", "worst-cluster", "edge", "hard-sample"]


def resilience_stress_curve(model, ds: Dataset, *, scenario: Scenario = "worst-sample",
                            levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                            n_clusters: int = 5, random_state: int = 0):
    """Progressively oversample a 'hard' subpopulation of the TEST set and track
    the metric — a degradation curve. No refit; only predictions + one cheap
    selection step (KMeans or covariance inverse).

    Returns (DataFrame, plotly.Figure, verdict_str).
    """
    Xte = ds.X_test.to_numpy(float)
    yte = ds.y_test.to_numpy()
    n = len(yte)
    rng = np.random.default_rng(random_state)

    # rank samples by "hardness" for the chosen scenario (higher = harder)
    if scenario == "worst-sample":
        hard = per_sample_loss(yte, model, Xte, ds.task)
    elif scenario == "hard-sample":
        if ds.task == Task.CLASSIFICATION:
            p = _proba(model, Xte)
            hard = 1.0 - np.sort(p, axis=1)[:, -1]      # low confidence = hard
        else:
            hard = per_sample_loss(yte, model, Xte, ds.task)
    elif scenario == "edge":
        mu = Xte.mean(axis=0)
        cov = np.cov(Xte, rowvar=False) + 1e-6 * np.eye(Xte.shape[1])
        inv = np.linalg.pinv(cov)
        d = Xte - mu
        hard = np.einsum("ij,jk,ik->i", d, inv, d)      # Mahalanobis^2
    elif scenario == "worst-cluster":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=random_state)
        lab = km.fit_predict(StandardScaler().fit_transform(Xte))
        loss = per_sample_loss(yte, model, Xte, ds.task)
        worst = max(range(n_clusters), key=lambda c: loss[lab == c].mean()
                    if (lab == c).any() else -np.inf)
        hard = (lab == worst).astype(float) + rng.uniform(0, 1e-6, n)
    else:
        raise ValueError(scenario)

    hard_order = np.argsort(-hard)
    base_metric = score(yte, model, Xte, ds.task)
    rows = []
    for alpha in levels:
        n_hard = int(alpha * n)
        idx = np.concatenate([hard_order[:n_hard],
                              rng.choice(n, size=n - n_hard, replace=True)])
        rows.append(dict(drift_level=alpha,
                         metric=round(score(yte[idx], model, Xte[idx], ds.task), 4)))
    curve = pd.DataFrame(rows)

    name = headline_name(ds.task)
    worst_metric = curve["metric"].iloc[-1]
    degradation = base_metric - worst_metric
    fig = go.Figure(go.Scatter(x=curve["drift_level"], y=curve["metric"],
                               mode="lines+markers", line=dict(color=viz.ACCENT)))
    fig.add_hline(y=base_metric, line=dict(color=viz.MUTED, dash="dot"),
                  annotation_text="baseline")
    fig.update_xaxes(title=f"fraction drifted toward '{scenario}'")
    fig.update_yaxes(title=name)
    fig = viz._base(fig, f"Resilience stress curve — {scenario}", height=420)

    rel = degradation / (abs(base_metric) + 1e-9)
    verdict = (f"Under '{scenario}' drift, {name} falls {degradation:.3f} "
               f"({rel:.0%}) from baseline {base_metric:.3f} to {worst_metric:.3f} "
               f"at 50% contamination. "
               + ("LOW resilience — the model degrades sharply toward this "
                  "subpopulation." if rel > 0.15 else
                  "Reasonable resilience to this scenario."))
    return curve, fig, verdict
