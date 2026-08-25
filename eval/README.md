# model_eval_toolkit

A low-code toolkit to **prevent and detect overfitting**. Three analyzers share
one data contract (`Dataset`), one config (`EvalConfig`), and one plotly theme.
Everything returns tables + interactive plotly figures; the whole thing can be
poured into a single self-contained HTML report.

```
pip install scikit-learn plotly pydantic scipy pandas numpy
```

## The three analyzers

### `ComplexityAnalyzer` — how complex is this fit, really?
Not one number but a **panel**, mostly model-agnostic (needs only fit/predict):

| indicator | what it means | overfitting reading |
|---|---|---|
| `effective_dof` (Efron covariance penalty, parametric bootstrap) | free parameters the fit actually spends | `edf/n → 1` ⇒ near-interpolation |
| `memorization` (Rademacher-style) | how well it fits **random** labels | `→1` ⇒ capacity to memorize noise |
| `pred_variance` (bootstrap) | prediction instability across resamples | high ⇒ variance-dominated |
| `generalization_gap` | `train − test` score | the direct readout |
| `structural_capacity` | leaves / params / non-zero coefs | raw capacity |

Returns a table, a 0–1 "risk radar", and a plain-language verdict.

### `LearningCurveAnalyzer` — is the data sufficient? (two modes)
* **`random_curve`** — fits an inverse power law `err(n)=c+a·n^-b` to the
  validation error and **extrapolates**: estimated ceiling, remaining error, and
  *how many more samples* buy 90% of the remaining gain. Labels the regime
  (high-bias / high-variance / saturated). This is the "do I need more data, and
  how much?" answer sklearn's curve doesn't give.
* **`temporal_curve`** — for **drift** (e.g. fraud). Fixes validation on the most
  recent block and grows the training window *backward in time*. If performance
  peaks at a limited look-back and **decays** with more history, older data is
  detrimental → reports the optimal window `W*` and the cost of using all history.

### `WeakpointAnalyzer` — where is it weak, and why?
Discovers interpretable regions with an **error tree** (a shallow tree predicting
per-sample loss → human-readable rules), then attributes a **cause** per region
through a transparent rule set:

`Local overfitting` · `Data sparsity` · `Distribution shift` · `Irreducible noise` · `Hard / underfit`

Outputs a ranked, cause-colored slice table, an **interactive PCA map** (dropdown
isolates each weak region), and univariate error profiles.

## Quick start

```python
from model_eval_toolkit import Dataset, EvalConfig, Task, OverfittingReport

# 1) your data — a pydantic dataloader is any object yielding a DataFrame
ds  = Dataset.from_loaders(train_loader, test_loader,
                           target="is_fraud", task="classification", time_col="ts")
cfg = EvalConfig(task=Task.CLASSIFICATION, time_col="ts")

# 2) one call → interactive HTML
OverfittingReport(cfg).run(model, ds).to_html("report.html")
```

Or drive analyzers individually:

```python
from model_eval_toolkit import ComplexityAnalyzer, LearningCurveAnalyzer, WeakpointAnalyzer

crep = ComplexityAnalyzer(cfg).analyze(model, ds);      crep.show()
lrep = LearningCurveAnalyzer(cfg).random_curve(model, ds); lrep.show()
trep = LearningCurveAnalyzer(cfg).temporal_curve(model, ds)  # needs time_col
wrep = WeakpointAnalyzer(cfg).analyze(model, ds);       wrep.show()
print(crep.verdict); print(wrep.slices)
```

## Plugging in your pydantic dataloader
`Dataset.from_loaders` accepts, for train and test, any of:
a `pandas.DataFrame`; a **callable** returning one; or an object exposing
`.load()/.to_frame()/.dataframe/.df/.data`; or a list of pydantic rows. If your
loader looks different, expose one of those (a one-line adapter) — the analyzers
never touch the loader directly, only the resulting `Dataset`.

## Cost knobs (in `EvalConfig`)
`n_repeats`, `n_bootstrap` (refit-heavy stats), `max_rows_for_refit` (subsample
cap), `cv_folds`, `max_slice_depth` / `min_slice_frac` (slice granularity),
`n_windows` (temporal resolution). Defaults are tuned to run in seconds on a few
thousand rows; raise them for more stable estimates.

## Notes / limitations
* `effective_dof` is defined for **regression and binary classification**;
  multiclass returns `n/a` and the radar degrades gracefully.
* Refit-based indicators clone the estimator (`sklearn.base.clone`, else
  `deepcopy`) — any sklearn-compatible estimator works.
* The weak-point map uses PCA for layout; swap in UMAP if installed for nicer
  geometry (only the 2-D embedding call needs changing).
