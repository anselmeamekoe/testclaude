# `model_eval_toolkit` — Method Documentation

Reference documentation for the three overfitting-diagnosis analyzers, their
public methods, output schemas, configuration, and the literature that justifies
each method. Drop this into your docs site (MkDocs / Sphinx-md / Docusaurus).

- [Data contract](#data-contract)
- [`EvalConfig`](#evalconfig)
- [`ComplexityAnalyzer`](#complexityanalyzer)
- [`LearningCurveAnalyzer`](#learningcurveanalyzer)
- [`WeakpointAnalyzer`](#weakpointanalyzer)
- [`OverfittingReport`](#overfittingreport)
- [Shared metric definitions](#shared-metric-definitions)
- [References by method](#references-by-method)
- [Bibliography](#bibliography)

Conventions used below: `n` = number of training rows; a *headline score* is
**AUC** for classification and **R²** for regression (higher is better); a
*per-sample loss* is **log-loss** for classification and **squared error** for
regression (lower is better).

---

## Data contract

Every analyzer consumes a single `Dataset` and (optionally) an `EvalConfig`. The
`Dataset` is built from your pydantic dataloader through `Dataset.from_loaders`,
which accepts, for each of train and test, any of: a `pandas.DataFrame`; a
callable returning one; an object exposing `.load() / .to_frame() / .dataframe /
.df / .data`; or a list of pydantic rows.

### `Dataset`

| Constructor | Signature |
|---|---|
| `Dataset.from_frames` | `(train, test, target, task="classification", time_col=None, features=None)` |
| `Dataset.from_loaders` | `(train_loader, test_loader, target, task="classification", time_col=None, features=None)` |

| Attribute / method | Type | Meaning |
|---|---|---|
| `X_train`, `X_test` | `DataFrame` | feature matrices |
| `y_train`, `y_test` | `Series` | targets |
| `feature_names` | `list[str]` | feature column order |
| `task` | `Task` | `classification` or `regression` |
| `time_train`, `time_test` | `Series` \| `None` | timestamps (required for the temporal curve) |
| `n_train`, `n_test` | `int` | row counts |
| `numpy()` | `(Xtr, ytr, Xte, yte)` | numpy views |
| `subsample_train(n, rng)` | `(X, y)` | random train subsample (used to cap refit cost) |

---

## `EvalConfig`

A single validated (pydantic) object holding every tunable. Passed to any
analyzer's constructor; sensible defaults run in seconds on a few thousand rows.

| Field | Default | Used by | Effect |
|---|---|---|---|
| `task` | `classification` | all | task type |
| `random_state` | `42` | all | reproducibility |
| `n_repeats` | `12` | complexity | Monte-Carlo repeats for `effective_dof`, `memorization` |
| `n_bootstrap` | `20` | complexity | resamples for `prediction_variance` |
| `max_rows_for_refit` | `4000` | complexity | subsample cap for refit-heavy stats |
| `cv_folds` | `4` | reserved | folds for cross-validated variants |
| `max_slice_depth` | `3` | weakpoint | depth of the error tree ⇒ slice granularity |
| `min_slice_frac` | `0.03` | weakpoint | minimum slice size as a share of the test set |
| `time_col` | `None` | temporal | name of the timestamp column |
| `n_windows` | `8` | temporal | number of look-back windows evaluated |

---

## `ComplexityAnalyzer`

**Purpose.** Report a *panel* of complementary complexity indicators — most of
them model-agnostic (only `fit`/`predict` required) — because a single scalar
cannot capture "how complex is this fit relative to the data". Higher complexity
relative to signal and sample size is what drives overfitting.

```python
ComplexityAnalyzer(config: EvalConfig | None = None)
```

### `analyze(model, ds) -> ComplexityReport`

Fits `model` (via a clone) and assembles all indicators below.

**`ComplexityReport`**

| Attribute | Type | Contents |
|---|---|---|
| `table` | `DataFrame` | `indicator, value, unit, reads as` |
| `radar_scores` | `dict[str,float]` | each indicator normalized to a 0–1 *risk* scale |
| `verdict` | `str` | plain-language LOW/MODERATE/HIGH risk summary + drivers |
| `details` | `dict` | raw values: `edf, edf_ratio, memorization, pred_variance, gap, train_score, test_score, structural, risk` |
| `figures()` | `dict` | `{"table", "radar"}` plotly figures |
| `show()` | — | render figures |

### `effective_dof(model, X, y, task) -> float`

Effective **degrees of freedom** — how many free parameters the fit *actually
spends* on the data — estimated by **Efron's covariance penalty** via a
parametric bootstrap, so the same estimator works for regression and binary
classification without ever feeding continuous targets to a classifier.

```
edf = Σ_i Cov(ŷ_i , y_i) / σ_i²
```

Estimated by resampling `y*` from the fitted model (Gaussian around `ŷ` for
regression; Bernoulli(`p̂`) for classification), refitting, and averaging
`(y*_i − μ_i)(μ*_i − μ_i)`, then dividing by the per-sample variance
`σ_i²` (residual variance for regression; `p̂_i(1−p̂_i)` for classification).

**Reading.** `edf/n → 1` means the model can nearly interpolate the training set,
so it will react strongly to label noise — a primary overfitting signal.
Returns `NaN` for multiclass (not defined here); the radar degrades gracefully.
Cost: `n_repeats` refits.

### `memorization(model, X, y, task) -> float`  *(0–1)*

**Rademacher-style capacity probe.** Refit the model on **randomly permuted
labels** and measure how far above chance the *training* fit gets (classification:
`(train_acc − chance)/(1 − chance)`; regression: in-sample R² on shuffled targets).

**Reading.** `≈0` the model cannot fit noise; `≈1` it memorizes noise perfectly —
i.e. it has capacity to fit patterns that do not generalize. Cost: `~n_repeats/3`
refits.

### `prediction_variance(model, ds) -> float`

**Bootstrap prediction variance** — the *variance* term of the bias–variance
decomposition. Bootstrap-resample the training set `n_bootstrap` times, refit,
and take the mean over test points of the standard deviation of predictions
(P(class 1) for classification, raw prediction for regression).

**Reading.** High = unstable fit that swings with the training sample; a hallmark
of over-flexible models. Cost: `n_bootstrap` refits.

### `structural_capacity(model) -> dict`

Best-effort introspection of raw capacity; silent about fields it cannot find:
tree ensembles → `n_estimators, total_leaves, avg_depth`; single tree →
`n_leaves, depth`; linear models → `n_coef, nonzero_coef, coef_l2`.

**Reading.** Direct capacity counts; `nonzero_coef` is the effective size of an
L1-regularized model.

### `input_sensitivity(model, X, task, n_dirs=8, h=0.05)` → dict  *(refit-free)*

**Local Lipschitz / input-perturbation sensitivity.** An overfit model has a
"jagged" response surface — a large input–output Jacobian norm — so it moves a
lot under tiny input nudges. We estimate this with random directions (SPSA-style),
using only `n_dirs` forward passes and **no refit**:

```
δ_{i,j} ~ N(0, (h·σ_j)²)          (per feature j, σ_j = feature std)
Lipschitz  = mean_i mean_k |f(x_i+δ) − f(x_i)| / ‖δ‖
rel_jagged = mean_i mean_k |f(x_i+δ) − f(x_i)| / scale_f
```

where `f` is the model's scalar output (P(class 1) for classification, prediction
for regression) and `scale_f` is its spread. `Lipschitz` is reported raw
(`lipschitz_mean`, `lipschitz_p95`); the scale-free `rel_jagged` (clipped by 0.5)
feeds the risk radar.

**Reading.** Higher = jaggier decision surface = more overfit-prone. This is a
much cheaper stand-in for `prediction_variance` (≈8 predicts vs `n_bootstrap`
refits); the two are complementary — one probes *input* sensitivity, the other
*training-set* sensitivity. Cost: `n_dirs` predicts, no fit.

> The complexity radar now carries five axes: `edf/n`, `memorization`,
> `pred variance`, **`input sensitivity`**, `generalization gap`.

---

## `LearningCurveAnalyzer`

**Purpose.** Answer "is the data sufficient for this model?" — as size grows
(`random_curve`) and as history lengthens (`temporal_curve`) — plus a refit-free
resilience stress test. Each mode returns a table, **one** Plotly figure, and a
one-line factual summary; there is no curve-fitting and no regime classification —
read the curve and decide.

```python
LearningCurveAnalyzer(config: EvalConfig | None = None)
```

### `random_curve(model, ds, fractions=(0.1,0.25,0.4,0.55,0.7,0.85,1.0), repeats=3) -> LearningCurveReport`

Trains the model on increasing **random subsets** of the training set and scores
train (in-sample) and validation (the held-out test set) at each size, averaging
`repeats` draws per size to reduce noise. Reading the curve: a persistent train–
validation gap that is still closing means more data of the same kind should
help; a flat validation curve means it won't.

**`LearningCurveReport` (mode `"random"`)**

| Attribute | Contents |
|---|---|
| `table` | one row per size: `n`, `train <score>`, `val <score>`, `val ±` (std over `repeats`) |
| `fig` | single plotly figure: train and validation curves, validation with an error band |
| `verdict` | one line: final train/val scores, the gap, and the validation span |
| `details` | `curve` (DataFrame) |

### `temporal_curve(model, ds, freq="M", min_period_samples=30) -> LearningCurveReport`

**Sufficiency measured in calendar periods** — the fraud-detection case, where
older data can be *detrimental*. Requires `time_col` on the training data. The
training window grows **backward one period at a time** — last 1 month, last 2
months, … — and each look-back is scored on the **explicit test set**
(`ds.X_test / ds.y_test`), exactly like `random_curve`. Because the evaluation
target is fixed, every point is comparable; only the training window changes. This
matches the out-of-time framing (train = history, test = the future period you
want to predict), so build the Dataset with the test set as that target.

Counting history in **whole months** (rather than sample counts) is both more
interpretable ("train on the last 3 months") and robust to volume changes between
periods, which would otherwise distort row-count windows.

| Parameter | Meaning |
|---|---|
| `freq` | calendar period size when `time_col` is datetime: `"M"` month (default), `"W"` week, `"Q"` quarter, `"Y"` year |
| `min_period_samples` | skip a look-back whose training window has fewer rows than this |

**Time-column handling** (applies to the training data's `time_col`): may be
either a **datetime** column → bucketed into calendar periods at `freq`; or
**ordered integer period ids** (e.g. month numbers `1..T`) → each distinct value
is one period, in ascending order — no dates required. Needs ≥ 2 training periods.
A continuous numeric column (many distinct values) is rejected with a message
asking for month ids or a datetime; bucket such timestamps first.

**Reading.** If the test score peaks at a limited look-back and **declines** as
older months are added, old data is stale → cap the training window near the peak
(or add recency weighting / drift features). The one-line `verdict` reports the
score range and where it peaks — no automatic labelling; you read the curve.

**`LearningCurveReport` (mode `"temporal"`)**

| Attribute | Contents |
|---|---|
| `table` | one row per look-back: `look-back (<unit>s)`, `from`, `to` (training-period range covered), `n` (training rows), `<score>` (on the test set) |
| `fig` | single plotly figure: test score vs **length of history** (x-axis in periods), best point starred; hover shows the training-period range and sample count |
| `details` | `curve` (DataFrame: `lookback, from_period, to_period, n_samples, test`), `unit`, `periods`, `best_lookback` |

> Feature-level drift attribution (which variables shift) is available separately
> via the refit-free `drift_drivers()` utility in `modeva_ext` (PSI / KS /
> Wasserstein) — it is no longer run inside `temporal_curve`, keeping this method
> to a single figure. See the drift-drivers note under *References by method*.

### `stress_curve(model, ds, scenario="worst-sample", levels=…, n_clusters=5) -> LearningCurveReport`  *(refit-free)*

**Resilience under simulated worst-case drift** — a stress test that needs **no
time column** (complements `temporal_curve`). Rank the test rows by a
scenario-specific *hardness*, then build evaluation sets that progressively
replace random rows with the hardest ones, and track the metric — a degradation
curve. No refit; only predictions plus one cheap selection step.

| `scenario` | hardness score for row `x` |
|---|---|
| `worst-sample` | per-sample loss |
| `hard-sample` | `1 − max_c p_c(x)` (low confidence) |
| `edge` | Mahalanobis distance² `(x−μ)ᵀ Σ⁻¹ (x−μ)` |
| `worst-cluster` | membership of the worst-mean-loss KMeans cluster |

The verdict reports the metric drop at 50% contamination and flags LOW resilience
when the relative drop exceeds ~15%. Cost: `(#levels)·T` predicts (+1 KMeans or
+1 covariance inverse), no fit.

**`LearningCurveReport` (mode `"stress"`)**: `table` (metric vs drift level),
`fig` (degradation curve vs baseline), `details` (`curve`, `scenario`,
`baseline`, `degradation`).

### `drift_drivers(X_ref, X_cur, feature_names, top=None)` → DataFrame  *(utility, refit-free)*

Standalone helper in `modeva_ext` that ranks features by how much their *marginal*
distribution shifts between two windows (e.g. an old month vs the most recent):

```
PSI          = Σ_i (p_i − q_i) · ln(p_i / q_i)      (K quantile bins)
KS           = sup_x |F_cur(x) − F_ref(x)|
Wasserstein  = ∫ |F_cur(x) − F_ref(x)| dx
```

**Scope.** These detect **covariate** drift (feature marginals moving); they are
*blind to concept drift* where only P(y│x) changes while the marginals stay put —
that is what `temporal_curve`'s refit sweep captures. Use the two together. Cost:
`O(d·n log n)`, no model calls.

---

## `WeakpointAnalyzer`

**Purpose.** Find *where* the model is weak (regions worse than its global metric)
and attribute a *cause*, distinguishing failure modes that need different fixes.

```python
WeakpointAnalyzer(config: EvalConfig | None = None)
```

### `analyze(model, ds) -> WeakpointReport`

**Region discovery.** A shallow decision tree (the *error tree*, depth
`max_slice_depth`, leaf size ≥ `min_slice_frac·n_test`) is fit to predict the
**per-sample test loss** from the features. Each leaf is an interpretable,
possibly *multivariate* rule (e.g. `amount>500 & hour≤6`) defining a slice — which
is how it surfaces **regional** problems a per-feature scan misses.

**`WeakpointReport`**

| Attribute | Type | Contents |
|---|---|---|
| `slices` | `DataFrame` | one row per region, ranked by `loss_ratio` (see columns below) |
| `figures()` | `dict` | `{"table", "map", "landscape", "drivers", "profile"}` |
| `verdict` | `str` | count of weak regions, worst region, cause breakdown |
| `details` | `dict` | `global_score, global_test_loss, global_train_loss, weak_threshold, error_drivers` |

Figures: **`table`** (cause-colored slice table), **`map`** (2-D PCA of the test
set colored by loss, dropdown isolates each weak region), **`landscape`** (the
cheap-AMIF density × signal map — see below), **`drivers`** (error-tree feature
importances), **`profile`** (per-feature mean-loss vs quantile-bin curves).

**`slices` columns**

| Column | Meaning |
|---|---|
| `rule` | human-readable region definition |
| `n_test`, `test_share`, `train_share` | region size in test; share of test / train falling in it |
| `local_score`, `global_score` | headline score inside the region vs overall |
| `local_test_loss`, `local_train_loss` | mean loss in the region on test / train |
| `loss_ratio` | `local_test_loss / global_test_loss` (>1 = worse than average) |
| `overfit_gap` | `local_test_loss − local_train_loss` (local generalization gap) |
| `coverage` | `train_share / test_share` (<1 = region under-represented in training) |
| `density_ratio` | region mean kNN-distance ÷ global median — cheap-AMIF geometry axis (>1 = sparser / more OOD) |
| `local_pred_var` | `Var[f̂(X)│X∈R]` — Modeva *Local Complexity(R)* |
| `loss_std` | `Std[loss│X∈R]` — Modeva *Uncertainty(R)* |
| `label_noise` | mean **k-Disagreeing-Neighbors** disagreement (Bayes-error / irreducible proxy) |
| `uncertainty` | mean model uncertainty (closeness to a coin-flip / prediction spread) |
| `likely_cause` | attributed cause (see rules) |

**Adaptive weak threshold** (Modeva). Instead of a fixed cutoff, the healthy/weak
boundary is data-driven across the discovered regions:

```
τ = clip( mean(loss_ratio) + β · std(loss_ratio),  1.1,  1.5 )     (β = 1.0)
```

A region is weak when `loss_ratio > τ`; `τ` is returned in `details["weak_threshold"]`.
The clip band guarantees a region <10% worse is never flagged and one ≥50% worse
always is. Cost: `O(#slices)`, free.

**Error drivers** (`details["error_drivers"]`). The error tree's own
`feature_importances_`, ranked — the features that most drive prediction error.
Free, since the tree is already fit.

**Cheap-AMIF landscape.** Inspired by Modeva's AMIF (density × mutual-information
grid) but refit-free: the `landscape` figure plots every test point on a
**geometry axis** (kNN-distance density; right = sparser) versus a **signal axis**
(local label agreement `1 − kDN`; up = more predictable), colored by loss. Low
density → sparsity; low signal → irreducible noise — the two axes read directly
onto the cause taxonomy. Cost: reuses the `NearestNeighbors` already fit for
`label_noise`; no extra model training. (The full ARF + cross-validated-RF AMIF
is deliberately *not* reproduced — it is the most expensive method in Modeva's
suite; this captures the interpretive value at `O(n log n)`.)

### Cause attribution (transparent rule set)

Evaluated per region, first match wins; `global_gap = global_test_loss −
global_train_loss`. Thresholds are relative to global behavior and exposed in
code for tuning.

| Cause | Condition | Fix it implies |
|---|---|---|
| `(healthy)` | `loss_ratio < τ` (adaptive threshold above) | none |
| **Local overfitting** | `overfit_gap > max(2·global_gap, 0.15)` **and** `local_train_loss < 0.8·global_test_loss` | regularize / prune locally |
| **Data sparsity** | `train_share < 0.4·test_share` **or** `coverage < 0.5` **or** `density_ratio > 1.5` | collect / upsample the region |
| **Distribution shift** | `test_share > 1.8·train_share` | reweight / retrain on recent data |
| **Irreducible noise** | `label_noise > 0.35` | improve labels/features; not a model fix |
| **Hard / underfit** | `local_train_loss > 1.3·global_train_loss` | add capacity or features |
| `Elevated error` | otherwise | inspect manually |

> Division of labour: **global** overfitting is caught by `ComplexityAnalyzer`;
> `WeakpointAnalyzer` stays quiet unless a problem is **regional**. A uniformly
> overfit model shows a large train/test gap everywhere but similar per-region
> loss ratios, so no single slice is flagged — that is intended behavior.

---

## `OverfittingReport`

One-call orchestrator that runs all analyzers and writes a single self-contained
interactive HTML report (plotly embedded).

```python
OverfittingReport(config).run(model, ds).to_html("report.html")
```

| Method | Returns | Notes |
|---|---|---|
| `run(model, ds)` | `self` | runs complexity, random curve, temporal curve (if `time_col`), weak points |
| `to_html(path="overfitting_report.html")` | `str` (path) | embeds every table and figure |

After `run`, individual reports are available as `.complexity`, `.learning`,
`.temporal` (or `None`), `.weak`.

---

## Shared metric definitions

- **Headline score** — classification: ROC-AUC (falls back to accuracy if AUC is
  undefined); regression: R².
- **Per-sample loss** — classification: log-loss `−log p(y_true)`; regression:
  squared error `(y − ŷ)²`.
- **Probabilities** — taken from `predict_proba`; else a softmax/sigmoid of
  `decision_function`; else one-hot of `predict`.
- **Cloning** — refit-based methods clone the estimator with `sklearn.base.clone`
  (falling back to `copy.deepcopy`), so any sklearn-compatible estimator works.

---

## Computational cost

Notation: `n` rows, `d` features, `F` = one model fit, `T` = one predict over the
set. Refit-based methods dominate; the Modeva-inspired additions are all
refit-free and cost less, combined, than a single `effective_dof` call.

| Method | Cost | Refits? |
|---|---|---|
| `effective_dof` | ≈ `n_repeats · F` (≈12 F) | yes |
| `memorization` | ≈ `(n_repeats/3) · F` (≈4 F) | yes |
| `prediction_variance` | ≈ `n_bootstrap · F` (≈20 F) | yes |
| `random_curve` | ≈ `Σ(fractions)·repeats · F` (≈20 F) | yes |
| `temporal_curve` | ≈ `(#periods) · F` (one fit per month of look-back) | yes |
| `WeakpointAnalyzer.analyze` | ≈ `1 F` (error tree) + `O(n log n)` (kNN) + `O(T)` | one small tree |
| **`input_sensitivity`** | ≈ `n_dirs · T` (≈8 T) | **no** |
| **`stress_curve`** | ≈ `#levels · T` (+1 KMeans / +1 cov⁻¹) | **no** |
| **drift drivers** | `O(d · n log n)` | **no** |
| **adaptive threshold** | `O(#slices)` | **no** |
| **cheap-AMIF landscape / error drivers** | free (reuse fitted kNN & error tree) | **no** |

To bound the refit-heavy methods on large data, lower `n_repeats` / `n_bootstrap`
or `max_rows_for_refit` (subsample cap). The refit-free methods scale with predict
cost only and need no such caps.

---

## References by method

| Analyzer / method | Justifying references |
|---|---|
| `effective_dof` (covariance-penalty EDF) | Efron 1986; Efron 2004; Ye 1998; Stein 1981; Hastie–Tibshirani–Friedman 2009 (ch. 7); Tibshirani & Taylor 2012 |
| `memorization` (random-label capacity) | Zhang et al. 2017; Bartlett & Mendelson 2002; Arpit et al. 2017 |
| `prediction_variance` (bootstrap variance) | Efron & Tibshirani 1993; Breiman 1996; Geman et al. 1992; Domingos 2000 |
| `generalization_gap` / optimism | Efron 1986; Hastie–Tibshirani–Friedman 2009 (ch. 7) |
| `structural_capacity` (nnz coefs, leaves) | Tibshirani & Taylor 2012; Hastie–Tibshirani–Friedman 2009 |
| `random_curve` (score vs training-set size) | Cortes et al. 1993; Provost et al. 1999; Perlich et al. 2003 |
| `temporal_curve` (history length in periods / look-back) | Gama et al. 2014; Widmer & Kubat 1996; Klinkenberg 2004; Dal Pozzolo et al. 2018; Tashman 2000; Bergmeir & Benítez 2012 |
| `input_sensitivity` (local Lipschitz / Jacobian) | Novak et al. 2018; Sokolić et al. 2017; Bartlett et al. 2017; Modeva docs (2025) |
| `stress_curve` (worst-case subpopulation drift) | Duchi & Namkoong 2021; Sagawa et al. 2020; Mahalanobis 1936; Modeva docs (2025) |
| `drift_drivers` (PSI / KS / Wasserstein; standalone) | Lin 1991 (JSD/PSI); Massey 1951 (KS); Ramdas et al. 2017 & Villani 2009 (Wasserstein); Modeva docs (2025) |
| Adaptive weak threshold (`μ+β·σ`) | Modeva docs (2025) |
| Cheap-AMIF landscape (density × signal) | Modeva docs (2025, AMIF); Smith et al. 2014 (kDN signal axis); Liu et al. 2008 (density, optional) |
| `local_pred_var`, `loss_std` (Local Complexity / Uncertainty) | Modeva docs (2025); Geman et al. 1992 |
| Weak-region discovery (error tree → interpretable slices) | Chung et al. 2019; Sagadeeva & Boehm 2021; Pastor et al. 2021; d'Eon et al. 2022 |
| Cause: local overfitting (`overfit_gap`) | Efron 1986; Hastie–Tibshirani–Friedman 2009 |
| Cause: data sparsity / `coverage` | Perlich et al. 2003; Cortes et al. 1993 |
| Cause: distribution shift (`test_share` vs `train_share`) | Shimodaira 2000; Sugiyama et al. 2007; Quiñonero-Candela et al. 2009 |
| Cause: irreducible noise (`label_noise` = kDN) | Smith et al. 2014; Cover & Hart 1967; Frénay & Verleysen 2014 |
| `uncertainty` (margin/spread) | Settles 2009 |
| Weak-region map (2-D embedding) | Jolliffe 2002 (PCA); McInnes et al. 2018 (UMAP, optional) |

---

## Bibliography

- Arpit, D., et al. (2017). *A Closer Look at Memorization in Deep Networks.* ICML.
- Bartlett, P. L., Foster, D. J., & Telgarsky, M. (2017). *Spectrally-normalized margin bounds for neural networks.* NeurIPS. *(links Lipschitz/spectral norms to generalization.)*
- Bartlett, P. L., & Mendelson, S. (2002). *Rademacher and Gaussian Complexities: Risk Bounds and Structural Results.* JMLR, 3, 463–482.
- Bergmeir, C., & Benítez, J. M. (2012). *On the use of cross-validation for time series predictor evaluation.* Information Sciences, 191, 192–213.
- Breiman, L. (1996). *Bagging Predictors.* Machine Learning, 24(2), 123–140.
- Chung, Y., Kraska, T., Polyzotis, N., Tae, K. H., & Whang, S. E. (2019). *Slice Finder: Automated Data Slicing for Model Validation.* ICDE, 1550–1553. (Extended: *Automated Data Slicing for Model Validation*, IEEE TKDE.)
- Cortes, C., Jackel, L. D., Solla, S. A., Vapnik, V., & Denker, J. S. (1993). *Learning Curves: Asymptotic Values and Rate of Convergence.* NIPS.
- Cover, T. M., & Hart, P. E. (1967). *Nearest neighbor pattern classification.* IEEE Trans. Information Theory, 13(1), 21–27.
- Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2018). *Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy.* IEEE TNNLS, 29(8), 3784–3797.
- d'Eon, G., d'Eon, J., Wright, J. R., & Leyton-Brown, K. (2022). *The Spotlight: A General Method for Discovering Systematic Errors in Deep Learning Models.* ACM FAccT.
- Domingos, P. (2000). *A Unified Bias-Variance Decomposition and its Applications.* ICML.
- Duchi, J., & Namkoong, H. (2021). *Learning Models with Uniform Performance via Distributionally Robust Optimization.* Annals of Statistics, 49(3), 1378–1406.
- Efron, B. (1986). *How Biased Is the Apparent Error Rate of a Prediction Rule?* JASA, 81(394), 461–470.
- Efron, B. (2004). *The Estimation of Prediction Error: Covariance Penalties and Cross-Validation.* JASA, 99(467), 619–632.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
- Frénay, B., & Verleysen, M. (2014). *Classification in the Presence of Label Noise: A Survey.* IEEE TNNLS, 25(5), 845–869.
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation.* ACM Computing Surveys, 46(4), Article 44.
- Geman, S., Bienenstock, E., & Doursat, R. (1992). *Neural Networks and the Bias/Variance Dilemma.* Neural Computation, 4(1), 1–58.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), ch. 7. Springer.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
- Klinkenberg, R. (2004). *Learning drifting concepts: Example selection vs. example weighting.* Intelligent Data Analysis, 8(3), 281–300.
- Lin, J. (1991). *Divergence Measures Based on the Shannon Entropy (Jensen–Shannon divergence).* IEEE Transactions on Information Theory, 37(1), 145–151. *(theoretical basis of the Population Stability Index.)*
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest.* IEEE ICDM. *(optional cheap density/OOD estimator for the geometry axis.)*
- Mahalanobis, P. C. (1936). *On the Generalised Distance in Statistics.* Proceedings of the National Institute of Sciences of India, 2(1), 49–55.
- Massey, F. J. (1951). *The Kolmogorov–Smirnov Test for Goodness of Fit.* JASA, 46(253), 68–78.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv:1802.03426.
- Pastor, E., de Alfaro, L., & Baralis, E. (2021). *Looking for Trouble: Analyzing Classifier Behavior via Pattern Divergence (DivExplorer).* ACM SIGMOD.
- Modeva Dev Team (2025). *Modeva User Guide — Diagnostic Suite (Weakness Detection, AMIF, Overfitting, Robustness, Resilience).* https://modeva.ai/docs/2-user-guide/
- Novak, R., Bahri, Y., Abolafia, D. A., Pennington, J., & Sohl-Dickstein, J. (2018). *Sensitivity and Generalization in Neural Networks: an Empirical Study.* ICLR. arXiv:1802.08760. *(input–output Jacobian norm correlates with generalization; random labels → higher sensitivity.)*
- Perlich, C., Provost, F., & Simonoff, J. S. (2003). *Tree Induction vs. Logistic Regression: A Learning-Curve Analysis.* JMLR, 4, 211–255.
- Provost, F., Jensen, D., & Oates, T. (1999). *Efficient Progressive Sampling.* ACM SIGKDD.
- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (2009). *Dataset Shift in Machine Learning.* MIT Press.
- Ramdas, A., García Trillos, N., & Cuturi, M. (2017). *On Wasserstein Two-Sample Testing and Related Families of Nonparametric Tests.* Entropy, 19(2), 47.
- Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). *Distributionally Robust Neural Networks for Group Shifts (Group DRO).* ICLR.
- Sokolić, J., Giryes, R., Sapiro, G., & Rodrigues, M. R. D. (2017). *Robust Large Margin Deep Neural Networks.* IEEE Transactions on Signal Processing, 65(16), 4265–4280. *(Jacobian-norm bounds on the generalization gap.)*
- Sagadeeva, S., & Boehm, M. (2021). *SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging.* ACM SIGMOD.
- Settles, B. (2009). *Active Learning Literature Survey.* Univ. Wisconsin–Madison, CS Tech. Report 1648.
- Shimodaira, H. (2000). *Improving predictive inference under covariate shift by weighting the log-likelihood function.* Journal of Statistical Planning and Inference, 90(2), 227–244.
- Smith, M. R., Martinez, T., & Giraud-Carrier, C. (2014). *An instance level analysis of data complexity (k-Disagreeing Neighbors).* Machine Learning, 95(2), 225–256.
- Stein, C. M. (1981). *Estimation of the Mean of a Multivariate Normal Distribution.* Annals of Statistics, 9(6), 1135–1151.
- Sugiyama, M., Krauledat, M., & Müller, K.-R. (2007). *Covariate Shift Adaptation by Importance Weighted Cross Validation.* JMLR, 8, 985–1005.
- Tashman, L. J. (2000). *Out-of-sample tests of forecasting accuracy: an analysis and review.* International Journal of Forecasting, 16(4), 437–450.
- Tibshirani, R. J., & Taylor, J. (2012). *Degrees of Freedom in Lasso Problems.* Annals of Statistics, 40(2), 1198–1232.
- Villani, C. (2009). *Optimal Transport: Old and New.* Springer. *(Wasserstein distance.)*
- Widmer, G., & Kubat, M. (1996). *Learning in the presence of concept drift and hidden contexts.* Machine Learning, 23(1), 69–101.
- Ye, J. (1998). *On Measuring and Correcting the Effects of Data Mining and Model Selection.* JASA, 93(441), 120–131.
- Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). *Understanding deep learning requires rethinking generalization.* ICLR. arXiv:1611.03530.
