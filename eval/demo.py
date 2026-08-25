"""End-to-end demo of model_eval_toolkit.

Builds:
  * a classification set with a PLANTED weak region (label noise in a corner),
  * a DELIBERATELY overfit model (deep tree) to light up complexity signals,
  * a time-ordered, DRIFTING fraud-like stream for the temporal curve.

Also shows the 'pydantic dataloader' entry point via a tiny loader class.
"""
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from model_eval_toolkit import (
    Dataset, EvalConfig, Task,
    ComplexityAnalyzer, LearningCurveAnalyzer, WeakpointAnalyzer, OverfittingReport,
)

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# 1. A "pydantic dataloader" — any object that can yield a DataFrame works.    #
# --------------------------------------------------------------------------- #
class FrameLoader(BaseModel):
    """Stand-in for a real pydantic dataloader; exposes .load() -> DataFrame."""
    model_config = {"arbitrary_types_allowed": True}
    frame: pd.DataFrame

    def load(self) -> pd.DataFrame:
        return self.frame


def make_classification_with_weak_region(n):
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 1.5 * x0 - 1.0 * x1 + 0.5 * x0 * x1
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    # PLANT 1 — irreducible noise: a top strip (x1>1.0) gets ~random labels.
    noisy = x1 > 1.0
    y[noisy] = (rng.uniform(size=noisy.sum()) < 0.5).astype(int)
    return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "y": y})


def drop_training_support(train_df, mask_col="x0", thresh=1.3, keep=0.08):
    """PLANT 2 — data sparsity: keep only a few TRAIN rows where x0>thresh."""
    hi = train_df[mask_col] > thresh
    keep_idx = train_df[hi].sample(frac=keep, random_state=1).index
    return pd.concat([train_df[~hi], train_df.loc[keep_idx]]).sort_index()


def make_drifting_fraud(n):
    """Fraud signal whose direction ROTATES over time -> old data is stale."""
    ts = np.sort(rng.uniform(0, 1, n))          # time in [0,1]
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    theta = 2.5 * ts                            # signal rotates with time
    signal = np.cos(theta) * a + np.sin(theta) * b
    p = 1 / (1 + np.exp(-(2.2 * signal - 1.0)))  # base rate < 0.5
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"amount": a, "velocity": b, "ts": ts, "is_fraud": y})


# --------------------------------------------------------------------------- #
print("=" * 70)
print("PART A — classification, overfit model, planted weak region")
print("=" * 70)
df = make_classification_with_weak_region(9000)
train_df, test_df = df.iloc[:6000].copy(), df.iloc[6000:].copy()
train_df = drop_training_support(train_df)     # thin out x0>1.3 in TRAIN only
train_loader = FrameLoader(frame=train_df)   # <- pydantic dataloader
test_loader = FrameLoader(frame=test_df)

ds = Dataset.from_loaders(train_loader, test_loader, target="y",
                          task="classification")
cfg = EvalConfig(task=Task.CLASSIFICATION, n_repeats=10, n_bootstrap=15,
                 max_slice_depth=3, min_slice_frac=0.04)

overfit_model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1,
                                       random_state=0).fit(
    ds.X_train.to_numpy(float), ds.y_train.to_numpy())

print("\n[Complexity]")
crep = ComplexityAnalyzer(cfg).analyze(overfit_model, ds)
print(crep.table.to_string(index=False))
print("VERDICT:", crep.verdict)

print("\n[Learning curve — random/saturation]")
lca = LearningCurveAnalyzer(cfg)
lrep = lca.random_curve(overfit_model, ds)
print(lrep.table.to_string(index=False))
print("VERDICT:", lrep.verdict)

# A regularized model: global error is controlled, so REGIONAL problems stand out.
good_model = RandomForestClassifier(n_estimators=200, min_samples_leaf=25,
                                     random_state=0).fit(
    ds.X_train.to_numpy(float), ds.y_train.to_numpy())

print("\n[Weak points — on the regularized model]")
wrep = WeakpointAnalyzer(cfg).analyze(good_model, ds)
print(wrep.slices[["rule", "n_test", "loss_ratio", "overfit_gap", "coverage",
                   "label_noise", "likely_cause"]].to_string(index=False))
print("VERDICT:", wrep.verdict)

print("\n" + "=" * 70)
print("PART B — drifting fraud stream, temporal learning curve")
print("=" * 70)
fdf = make_drifting_fraud(7000)
ftr, fte = fdf.iloc[:5000], fdf.iloc[5000:]
fds = Dataset.from_frames(ftr, fte, target="is_fraud", task="classification",
                          time_col="ts")
fcfg = EvalConfig(task=Task.CLASSIFICATION, time_col="ts", n_windows=9)
fmodel = RandomForestClassifier(n_estimators=120, min_samples_leaf=15,
                                random_state=0).fit(
    fds.X_train.to_numpy(float), fds.y_train.to_numpy())
trep = LearningCurveAnalyzer(fcfg).temporal_curve(fmodel, fds)
print(trep.table.to_string(index=False))
print("VERDICT:", trep.verdict)

print("\n" + "=" * 70)
print("PART C — one-call HTML report")
print("=" * 70)
rep = OverfittingReport(cfg).run(overfit_model, ds)
path = rep.to_html("/home/claude/overfitting_report.html")
print("Wrote", path)

# also a temporal report for the fraud model
frep = OverfittingReport(fcfg).run(fmodel, fds)
fpath = frep.to_html("/home/claude/fraud_report.html")
print("Wrote", fpath)
print("\nDONE.")
