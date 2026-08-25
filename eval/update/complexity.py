"""
ComplexityAnalyzer
==================
"Model complexity" is not one number. We report a *panel* of complementary
indicators, most of them model-agnostic (they only need fit/predict), so the
same tool works for trees, linear models, GBMs or nets.

Indicators
----------
1. effective_dof        Effective degrees of freedom via Stein/SURE randomized
                        divergence. How many "free parameters" the fit actually
                        spends on the data, estimated by finite-difference
                        sensitivity of predictions to perturbations of y.
                        Model-agnostic. edf/n near 1 => the model can nearly
                        interpolate => high overfitting risk.
2. memorization         Rademacher-style capacity: refit on RANDOM labels and
                        see how far above chance the *training* fit gets. 0 = can't
                        fit noise, 1 = memorizes noise perfectly (dangerous).
3. pred_variance        Bootstrap variance of predictions on held-out points —
                        the "variance" term of bias/variance. High = unstable fit.
4. generalization_gap   train_score - test_score. The direct overfitting readout.
5. structural_capacity  Best-effort introspection (leaves, params, nnz coefs).

Everything is normalized to a 0..1 "risk" scale for a radar summary, but the raw
values are always kept in the table.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone

from .data import Dataset, EvalConfig, Estimator, Task
from .metrics import _proba, headline_name, score
from .modeva_ext import input_sensitivity
from . import viz


def _clone(model):
    try:
        return clone(model)
    except Exception:
        return copy.deepcopy(model)


def _predict_num(model, X, task: Task) -> np.ndarray:
    """Scalar output per row: P(class 1) for clf, raw prediction for reg."""
    if task == Task.CLASSIFICATION:
        p = _proba(model, X)
        return p[:, 1] if p.shape[1] == 2 else p.max(axis=1)
    return np.asarray(model.predict(X), dtype=float)


# --------------------------------------------------------------------------- #
@dataclass
class ComplexityReport:
    table: pd.DataFrame
    radar_scores: dict
    verdict: str
    details: dict = field(default_factory=dict)

    def figures(self) -> dict[str, go.Figure]:
        t = viz.table(self.table, "Complexity indicators")
        r = viz.radar(list(self.radar_scores.keys()),
                      list(self.radar_scores.values()),
                      "Overfitting-risk profile (0 = safe, 1 = risky)")
        return {"table": t, "radar": r}

    def show(self):
        for f in self.figures().values():
            f.show()


class ComplexityAnalyzer:
    def __init__(self, config: Optional[EvalConfig] = None):
        self.cfg = config or EvalConfig()
        self.rng = np.random.default_rng(self.cfg.random_state)

    # ---- individual indicators ------------------------------------------ #
    def effective_dof(self, model, X: np.ndarray, y: np.ndarray, task: Task) -> float:
        """Effective degrees of freedom via Efron's covariance penalty, estimated
        by a PARAMETRIC BOOTSTRAP so it works for both regression and (binary)
        classification without ever feeding continuous labels to a classifier.

        edf = sum_i Cov(y_i, mu_hat_i) / sigma_i^2,  estimated by resampling
        y* from the fitted model, refitting, and averaging (y*_i - mu_i)(mu*_i - mu_i).

        Returns NaN for multiclass (not defined here) — the caller degrades gracefully.
        """
        y = np.asarray(y)
        base = _clone(model).fit(X, y)

        if task == Task.CLASSIFICATION:
            classes = np.unique(y)
            if len(classes) != 2:
                return float("nan")  # edf defined here for binary only
            p = _proba(base, X)[:, 1].astype(float)
            mu = np.clip(p, 1e-4, 1 - 1e-4)
            sigma2 = mu * (1 - mu)            # per-sample Bernoulli variance
            def draw():
                return (self.rng.uniform(size=len(mu)) < mu).astype(int)
            def refit_mu(ystar):
                return _proba(_clone(model).fit(X, ystar), X)[:, 1].astype(float)
        else:
            mu = np.asarray(base.predict(X), dtype=float)
            sigma2 = np.full(len(mu), max(np.var(np.asarray(y, float) - mu), 1e-6))
            sd = np.sqrt(sigma2[0])
            def draw():
                return mu + self.rng.normal(0, sd, size=len(mu))
            def refit_mu(ystar):
                return np.asarray(_clone(model).fit(X, ystar).predict(X), dtype=float)

        cov = np.zeros(len(mu))
        used = 0
        for _ in range(self.cfg.n_repeats):
            ystar = draw()
            if task == Task.CLASSIFICATION and len(np.unique(ystar)) < 2:
                continue  # degenerate resample; skip
            mustar = refit_mu(ystar)
            cov += (ystar - mu) * (mustar - mu)
            used += 1
        if used == 0:
            return float("nan")
        cov /= used
        edf = float(np.clip(np.sum(cov / sigma2), 0, len(mu)))
        return edf

    def memorization(self, model, X: np.ndarray, y: np.ndarray, task: Task) -> float:
        """Rademacher-style: fit random labels, measure train fit above chance."""
        scores = []
        for _ in range(max(3, self.cfg.n_repeats // 3)):
            yp = self.rng.permutation(y)
            m = _clone(model).fit(X, yp)
            if task == Task.CLASSIFICATION:
                acc = float((np.asarray(m.predict(X)) == yp).mean())
                _, counts = np.unique(yp, return_counts=True)
                chance = counts.max() / len(yp)
                scores.append((acc - chance) / (1 - chance + 1e-9))
            else:
                pred = np.asarray(m.predict(X), float)
                ss = 1 - np.sum((yp - pred) ** 2) / (np.sum((yp - yp.mean()) ** 2) + 1e-9)
                scores.append(ss)
        return float(np.clip(np.mean(scores), 0, 1))

    def prediction_variance(self, model, ds: Dataset) -> float:
        """Bootstrap the training set, refit, measure prediction spread on test."""
        Xtr, ytr = ds.subsample_train(self.cfg.max_rows_for_refit, self.rng)
        Xte = ds.X_test.to_numpy(float)
        preds = []
        n = len(Xtr)
        for _ in range(self.cfg.n_bootstrap):
            idx = self.rng.integers(0, n, size=n)
            m = _clone(model).fit(Xtr[idx], ytr[idx])
            preds.append(_predict_num(m, Xte, ds.task))
        preds = np.vstack(preds)
        return float(np.mean(np.std(preds, axis=0)))

    def structural_capacity(self, model) -> dict:
        """Best-effort introspection; silent about what it can't find."""
        out = {}
        ests = getattr(model, "estimators_", None)
        if ests is not None:
            leaves, depths = [], []
            flat = np.ravel(ests)
            for e in flat:
                tr = getattr(e, "tree_", None)
                if tr is not None:
                    leaves.append(int((tr.children_left == -1).sum()))
                    depths.append(int(tr.max_depth))
            if leaves:
                out["n_estimators"] = len(flat)
                out["total_leaves"] = int(np.sum(leaves))
                out["avg_depth"] = float(np.mean(depths))
        tr = getattr(model, "tree_", None)
        if tr is not None:
            out["n_leaves"] = int((tr.children_left == -1).sum())
            out["depth"] = int(tr.max_depth)
        coef = getattr(model, "coef_", None)
        if coef is not None:
            coef = np.ravel(coef)
            out["n_coef"] = int(coef.size)
            out["nonzero_coef"] = int(np.sum(np.abs(coef) > 1e-8))
            out["coef_l2"] = float(np.linalg.norm(coef))
        return out

    # ---- orchestration --------------------------------------------------- #
    def analyze(self, model, ds: Dataset) -> ComplexityReport:
        Xtr, ytr = ds.subsample_train(self.cfg.max_rows_for_refit, self.rng)
        n = len(ytr)

        edf = self.effective_dof(model, Xtr, ytr, ds.task)
        mem = self.memorization(model, Xtr, ytr, ds.task)
        pvar = self.prediction_variance(model, ds)
        struct = self.structural_capacity(model)

        fitted = _clone(model).fit(ds.X_train.to_numpy(float), ds.y_train.to_numpy())
        s_tr = score(ds.y_train, fitted, ds.X_train.to_numpy(float), ds.task)
        s_te = score(ds.y_test, fitted, ds.X_test.to_numpy(float), ds.task)
        gap = s_tr - s_te

        # refit-free local-Lipschitz sensitivity (Modeva-inspired)
        sens = input_sensitivity(fitted, ds.X_test.to_numpy(float), ds.task,
                                 random_state=self.cfg.random_state)

        edf_ratio = (edf / n) if edf == edf else float("nan")
        # normalize a spread of prediction std to 0..1 by its own dispersion scale
        y_scale = (np.std(ds.y_train.to_numpy(float)) + 1e-9)
        pvar_norm = float(np.clip(pvar / y_scale, 0, 1)) if ds.task == Task.REGRESSION \
            else float(np.clip(pvar / 0.25, 0, 1))  # 0.25 = max Bernoulli std
        gap_norm = float(np.clip(gap / 0.3, 0, 1))
        # 0.5 = an h-perturbation moving half the output's own spread is very jaggy
        sens_norm = float(np.clip(sens["relative_jaggedness"] / 0.5, 0, 1))

        edf_str = f"{edf:.1f}" if edf == edf else "n/a (multiclass)"
        edf_ratio_str = f"{edf_ratio:.2f}" if edf_ratio == edf_ratio else "n/a"
        rows = [
            ("effective_dof", edf_str, f"of n={n}",
             "spent free parameters (Efron cov-penalty)"),
            ("edf / n", edf_ratio_str, "", "→1 means it can interpolate"),
            ("memorization", f"{mem:.2f}", "0..1", "fit of random labels (capacity)"),
            ("pred_variance", f"{pvar:.4f}", "std on test", "bootstrap instability"),
            ("input_sensitivity", f"{sens['lipschitz_mean']:.3f}", "mean |Δf|/‖δ‖",
             "local Lipschitz (refit-free)"),
            ("  sensitivity p95", f"{sens['lipschitz_p95']:.3f}", "", "worst-case local slope"),
            (f"train {headline_name(ds.task)}", f"{s_tr:.3f}", "", ""),
            (f"test {headline_name(ds.task)}", f"{s_te:.3f}", "", ""),
            ("generalization_gap", f"{gap:.3f}", "train - test", "direct overfit signal"),
        ]
        for k, v in struct.items():
            rows.append((f"struct: {k}", f"{v:.3g}" if isinstance(v, float) else str(v),
                         "", "structural capacity"))
        table = pd.DataFrame(rows, columns=["indicator", "value", "unit", "reads as"])

        radar_scores = {
            "memorization": mem,
            "pred variance": pvar_norm,
            "input sensitivity": sens_norm,
            "generalization gap": gap_norm,
        }
        if edf_ratio == edf_ratio:
            radar_scores = {"edf / n": float(np.clip(edf_ratio, 0, 1)), **radar_scores}
        risk = float(np.mean(list(radar_scores.values())))
        verdict = self._verdict(risk, radar_scores, gap, edf_ratio)

        return ComplexityReport(
            table=table, radar_scores=radar_scores, verdict=verdict,
            details=dict(edf=edf, edf_ratio=edf_ratio, memorization=mem,
                         pred_variance=pvar, gap=gap, train_score=s_tr,
                         test_score=s_te, structural=struct, risk=risk,
                         lipschitz_mean=sens["lipschitz_mean"],
                         lipschitz_p95=sens["lipschitz_p95"],
                         relative_jaggedness=sens["relative_jaggedness"]),
        )

    @staticmethod
    def _verdict(risk, scores, gap, edf_ratio) -> str:
        if risk < 0.25:
            head = "LOW overfitting risk — complexity looks well matched to the data."
        elif risk < 0.5:
            head = "MODERATE overfitting risk — watch the flagged signals."
        else:
            head = "HIGH overfitting risk — the model is over-parameterized for this data."
        drivers = [k for k, v in scores.items() if v > 0.5]
        why = ""
        if drivers:
            why = " Main drivers: " + ", ".join(drivers) + "."
        if edf_ratio > 0.5:
            why += (" Effective d.o.f. is a large fraction of n: the fit is close to "
                    "interpolation, so it will react strongly to label noise.")
        if gap > 0.1 and "generalization gap" not in drivers:
            why += f" Note a train/test gap of {gap:.2f}."
        return head + why
