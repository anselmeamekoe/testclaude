"""Metric primitives shared across analyzers (kept tiny and dependency-light)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .data import Task


def per_sample_loss(y_true, model, X, task: Task) -> np.ndarray:
    """Per-row loss: log-loss (clf) or squared error (reg). Lower = better."""
    if task == Task.CLASSIFICATION:
        proba = _proba(model, X)
        eps = 1e-12
        yt = np.asarray(y_true).astype(int)
        p = np.clip(proba[np.arange(len(yt)), yt], eps, 1 - eps)
        return -np.log(p)
    pred = np.asarray(model.predict(X), dtype=float)
    return (np.asarray(y_true, dtype=float) - pred) ** 2


def score(y_true, model, X, task: Task) -> float:
    """A single 'higher is better' headline score: AUC (clf) or R2 (reg)."""
    if task == Task.CLASSIFICATION:
        proba = _proba(model, X)[:, 1] if _proba(model, X).shape[1] == 2 else None
        try:
            if proba is not None:
                return float(roc_auc_score(y_true, proba))
            return float(roc_auc_score(y_true, _proba(model, X), multi_class="ovr"))
        except Exception:
            return float((np.asarray(model.predict(X)) == np.asarray(y_true)).mean())
    pred = np.asarray(model.predict(X), dtype=float)
    yt = np.asarray(y_true, dtype=float)
    ss_res = np.sum((yt - pred) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def _proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)
    if hasattr(model, "decision_function"):
        d = np.asarray(model.decision_function(X), dtype=float)
        if d.ndim == 1:
            p = 1 / (1 + np.exp(-d))
            return np.column_stack([1 - p, p])
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    pred = np.asarray(model.predict(X)).astype(int)
    n_cls = int(pred.max()) + 1
    oh = np.zeros((len(pred), max(n_cls, 2)))
    oh[np.arange(len(pred)), pred] = 1.0
    return oh


def headline_name(task: Task) -> str:
    return "AUC" if task == Task.CLASSIFICATION else "R2"
