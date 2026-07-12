"""
Bayesian hyper-parameter optimisation (Optuna) on an AMOUNT-WEIGHTED validation
metric, for card-fraud XGBoost under the constraints established in the MRM review.
=====================================================================================

Design constraints this prototype respects (each maps to a review finding):

  F-01  CARD-DISJOINT CV.  Splits are grouped by card_id, so transactions of one
        card never straddle train/validation. Prevents leakage through card-history
        / velocity features. (StratifiedGroupKFold.)

  F-03  UNDERSAMPLE INSIDE THE TRAINING FOLD ONLY.  The validation fold keeps its
        NATURAL fraud rate, so the reported metric is measured on a true-prior
        distribution. Undersampling never touches validation or test.

  F-04  VALUE-AWARE OBJECTIVE.  Optuna maximises AMOUNT-WEIGHTED AUROC -- the
        probability that a random fraudulent EURO is ranked above a random genuine
        EURO. Prior-invariant (safe under undersampling) AND value-aware (rewards
        ranking high-VALUE fraud), unlike count-AUCPR.

  F-06  NO EARLY STOPPING.  n_estimators is tuned as a hyper-parameter, matching
        the reviewed workflow. Optuna's pruner handles the compute budget instead.
        The final threshold is NOT tuned here -- it is set downstream with the
        business on true-prior data.

Also included:
  * Optuna pruning (MedianPruner) on the running fold mean -> kills bad trials early.
  * A seed-averaged, variance-aware objective option (F-03: undersampling adds
    variance; a single draw is fragile). `N_UNDERSAMPLE_DRAWS` > 1 averages over
    draws and the objective subtracts a penalty for across-fold instability.
  * Comparison run: tuning on plain AUROC / count-AUCPR vs. amount-weighted AUROC,
    scored on held-out EUR-recall @ budget -- demonstrating the metric actually
    changes which hyper-parameters win.

Run:   python optuna_amount_weighted.py
Needs: optuna, xgboost, scikit-learn, numpy  (+ fraud_decisioning.py alongside)
"""

import pickle
import warnings
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from fraud_decisioning import (
    amount_weighted_auc, undersample_majority, make_synthetic_with_features,
    expected_value_score,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
N_SPLITS            = 3       # CV folds (card-disjoint)
UNDERSAMPLE_RATE    = 0.10    # fraud rate INSIDE the training fold after undersampling
N_UNDERSAMPLE_DRAWS = 1       # >1 -> average over draws (reduces sampling variance)
N_TRIALS            = 25      # Bayesian optimisation budget (raise for real use)
STABILITY_PENALTY   = 0.5     # objective = mean - penalty * std(across folds)
ALERTS_PER_DAY      = 10      # only for the final true-prior report, NOT tuned here
SEED                = 42


# --------------------------------------------------------------------------- #
# The tuning objective
# --------------------------------------------------------------------------- #
def cv_score(params, X, y, amount, groups, n_splits=N_SPLITS,
             undersample_rate=UNDERSAMPLE_RATE, n_draws=N_UNDERSAMPLE_DRAWS,
             metric="amount_auc", seed=SEED, trial=None):
    """
    Card-disjoint stratified CV.  For each fold:
        train fold -> undersample the MAJORITY (n_draws times, averaged)
        valid fold -> left at its NATURAL fraud rate
        score      -> `metric` on the natural-rate validation fold
    Returns (mean_across_folds, std_across_folds).

    `metric`:
        "amount_auc" : amount-weighted AUROC   (recommended -- prior-robust + value-aware)
        "auroc"      : plain AUROC             (prior-robust, count-based)
        "aucpr"      : average precision       (value-blind AND prior-sensitive)
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores = []

    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups=groups)):
        draw_scores = []
        for d in range(n_draws):
            rng = np.random.default_rng(seed + 1000 * fold + d)
            Xtr, ytr, _ = undersample_majority(X[tr], y[tr], amount[tr],
                                               undersample_rate, rng)
            model = XGBClassifier(
                n_jobs=-1, eval_metric="logloss", tree_method="hist",
                random_state=seed + d, **params,
            )
            model.fit(Xtr, ytr, verbose=False)          # NO early stopping (F-06)

            p_va = model.predict_proba(X[va])[:, 1]     # natural-rate fold (F-03)
            if metric == "amount_auc":
                s = amount_weighted_auc(y[va], p_va, amount[va])
            elif metric == "auroc":
                s = roc_auc_score(y[va], p_va)
            elif metric == "aucpr":
                s = average_precision_score(y[va], p_va)
            else:
                raise ValueError(f"unknown metric {metric}")
            draw_scores.append(s)

        fold_scores.append(float(np.mean(draw_scores)))

        # ---- pruning: report the running fold mean so bad trials die early ----
        if trial is not None:
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def make_objective(X, y, amount, groups, metric="amount_auc"):
    def objective(trial):
        params = dict(
            # n_estimators is TUNED, not early-stopped (F-06)
            n_estimators     = trial.suggest_int("n_estimators", 200, 900, step=50),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
            gamma            = trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            # NOTE: scale_pos_weight addresses CLASS IMBALANCE, which is orthogonal to
            # AMOUNT. It is tuned here; it is NOT an amount-cost mechanism.
            scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True),
        )
        mean, std = cv_score(params, X, y, amount, groups, metric=metric, trial=trial)
        trial.set_user_attr("cv_mean", mean)
        trial.set_user_attr("cv_std", std)
        # penalise unstable configurations (undersampling inflates variance -- F-03)
        return mean - STABILITY_PENALTY * std
    return objective


def tune(X, y, amount, groups, metric="amount_auc", n_trials=N_TRIALS, seed=SEED):
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed, multivariate=True),   # Bayesian (TPE)
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=1),
        study_name=f"fraud_xgb_{metric}",
    )
    study.optimize(make_objective(X, y, amount, groups, metric),
                   n_trials=n_trials, show_progress_bar=False)
    return study


# --------------------------------------------------------------------------- #
# Final fit + true-prior evaluation
# --------------------------------------------------------------------------- #
def fit_final(best_params, X, y, amount, seed=SEED, n_draws=3):
    """Refit on the FULL training set. Average over several undersampling draws
    (bagging) rather than trusting a single fragile subset (F-03)."""
    models = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed + 77 * d)
        Xtr, ytr, _ = undersample_majority(X, y, amount, UNDERSAMPLE_RATE, rng)
        m = XGBClassifier(n_jobs=-1, eval_metric="logloss", tree_method="hist",
                          random_state=seed + d, **best_params)
        m.fit(Xtr, ytr, verbose=False)
        models.append(m)
    return models


def predict_ensemble(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def eur_recall_at_budget(y, proba, amount, day, alerts_per_day):
    """EUR of fraud captured when alerting the top-N/day by expected value (p*amount).
    Reported on TRUE-PRIOR data only."""
    score = expected_value_score(proba, amount)
    n_days = len(np.unique(day))
    k = max(1, int(round(alerts_per_day * n_days)))
    k = min(k, len(score))
    idx = np.argsort(-score)[:k]
    alert = np.zeros(len(score), bool); alert[idx] = True
    fraud_eur = amount[y == 1].sum()
    return float(amount[(y == 1) & alert].sum() / fraud_eur), alert


# --------------------------------------------------------------------------- #
# Demo data with an explicit COUNT-vs-VALUE tension
# --------------------------------------------------------------------------- #
def make_tension_data(n=60000, fraud_rate=0.02, n_days=80, seed=0):
    """
    Two fraud populations, deliberately in tension -- this is the situation in which
    the choice of tuning metric actually MATTERS:

      * MANY low-value frauds  (~85% of frauds, small EUR) with a STRONG, easy signal
        -> a COUNT-based metric (AUCPR) is dominated by these.
      * FEW high-value frauds  (~15% of frauds, large EUR) with a WEAK, noisy signal
        carried by DIFFERENT features
        -> these hold most of the fraud EUR, and only a VALUE-aware metric pays the
           model to fit the features that find them.

    A model can therefore win on AUCPR (catching the easy, cheap frauds) while losing
    badly on EUR captured -- exactly the two-candidate paradox in the review.
    """
    rng = np.random.default_rng(seed)
    d_easy, d_hard = 6, 6
    Xe = rng.normal(size=(n, d_easy))     # features that reveal cheap fraud
    Xh = rng.normal(size=(n, d_hard))     # features that reveal expensive fraud
    we, wh = rng.normal(size=d_easy), rng.normal(size=d_hard)

    s_easy = (Xe @ we) / np.sqrt(d_easy)
    s_hard = (Xh @ wh) / np.sqrt(d_hard)

    n_fraud = int(n * fraud_rate)
    n_hi = int(n_fraud * 0.15)
    n_lo = n_fraud - n_hi

    # cheap frauds: top of a STRONG (low-noise) easy signal
    lo_rank = s_easy * 2.0 + rng.normal(0, 0.5, size=n)
    lo_idx = np.argsort(-lo_rank)[:n_lo]
    # expensive frauds: top of a WEAK (high-noise) hard signal, disjoint from the above
    hi_rank = s_hard * 1.0 + rng.normal(0, 1.6, size=n)
    hi_rank[lo_idx] = -np.inf
    hi_idx = np.argsort(-hi_rank)[:n_hi]

    y = np.zeros(n, int); y[lo_idx] = 1; y[hi_idx] = 1

    amount = np.exp(rng.normal(3.0, 0.9, size=n))          # genuine baseline
    amount[lo_idx] = np.exp(rng.normal(2.2, 0.6, size=n_lo))    # cheap frauds
    amount[hi_idx] = np.exp(rng.normal(7.0, 0.8, size=n_hi))    # expensive frauds

    X = np.column_stack([Xe, Xh, np.log1p(amount)])
    day = rng.integers(0, n_days, size=n)
    return X, y, amount, day
def main():
    X, y, amount, day = make_tension_data(n=60000, seed=0)
    # synthetic card ids: ~8 transactions per card (grouping unit for CV -- F-01)
    rng = np.random.default_rng(7)
    card_id = rng.integers(0, len(y) // 8, size=len(y))

    # temporal split: tune on the earlier period, evaluate on the later one
    order = np.argsort(day, kind="mergesort")
    X, y, amount, day, card_id = X[order], y[order], amount[order], day[order], card_id[order]
    cut = int(len(y) * 0.75)
    Xtr, ytr, atr, dtr, gtr = X[:cut], y[:cut], amount[:cut], day[:cut], card_id[:cut]
    Xte, yte, ate, dte      = X[cut:], y[cut:], amount[cut:], day[cut:]

    print(f"train={len(ytr)}  test={len(yte)} | fraud rate={y.mean():.2%} | "
          f"cards={len(np.unique(card_id))}")
    hi = (y == 1) & (amount > np.quantile(amount[y == 1], 0.85))
    print(f"fraud EUR share of all EUR = {amount[y==1].sum()/amount.sum():.1%} | "
          f"top-15% frauds by amount hold {amount[hi].sum()/amount[y==1].sum():.0%} of fraud EUR "
          f"(these are the HARD ones)")
    print(f"CV: {N_SPLITS}-fold card-disjoint | undersample train folds to "
          f"{UNDERSAMPLE_RATE:.0%} | {N_UNDERSAMPLE_DRAWS} draw(s) | no early stopping")

    results = {}
    for metric in ("amount_auc", "aucpr"):
        print(f"\n=== Optuna (TPE) tuning on: {metric} ===")
        study = tune(Xtr, ytr, atr, gtr, metric=metric, n_trials=N_TRIALS)
        # Save the sampler with pickle to be loaded later.
        with open(f"fraud_xgb_{metric}_sampler.pkl", "wb") as fout:
            pickle.dump(study.sampler, fout)

        bt = study.best_trial
        done = len([t for t in study.trials if t.state.name == "COMPLETE"])
        pruned = len([t for t in study.trials if t.state.name == "PRUNED"])
        print(f"  trials: {done} complete, {pruned} pruned")
        print(f"  best objective (mean - {STABILITY_PENALTY}*std) = {bt.value:.4f}")
        print(f"  cv_mean={bt.user_attrs['cv_mean']:.4f}  cv_std={bt.user_attrs['cv_std']:.4f}")
        print("  best params:")
        for k, v in bt.params.items():
            print(f"    {k:18s} {v}")

        models = fit_final(bt.params, Xtr, ytr, atr)
        p_te = predict_ensemble(models, Xte)
        eur_rec, alert = eur_recall_at_budget(yte, p_te, ate, dte, ALERTS_PER_DAY)
        results[metric] = dict(
            amt_auc = amount_weighted_auc(yte, p_te, ate),
            auroc   = roc_auc_score(yte, p_te),
            aucpr   = average_precision_score(yte, p_te),
            eur_rec = eur_rec,
            cnt_rec = float(alert[yte == 1].mean()),
        )

    # ---- true-prior comparison on the held-out test set ----
    print("\n=== Held-out test (TRUE prior, never undersampled) ===")
    hdr = f"{'tuned on':<12} {'amtAUC':>8} {'AUROC':>8} {'AUCPR':>8} "
    hdr += f"{'EURrec@' + str(ALERTS_PER_DAY) + '/d':>13} {'cnt_rec':>8}"
    print(hdr); print("-" * len(hdr))
    for m, r in results.items():
        print(f"{m:<12} {r['amt_auc']:8.4f} {r['auroc']:8.4f} {r['aucpr']:8.4f} "
              f"{r['eur_rec']:13.3f} {r['cnt_rec']:8.3f}")

    d = results["amount_auc"]["eur_rec"] - results["aucpr"]["eur_rec"]
    print(f"\nEUR-recall @ budget: tuning on amount-weighted AUROC "
          f"{'gains' if d >= 0 else 'loses'} {abs(d)*100:.1f} pp vs tuning on AUCPR.")
    print("""
INTERPRETING THIS COMPARISON (read before quoting it):
  The two tuned models often land close on EUR-recall. That is an HONEST result, not
  a bug, and it is worth stating plainly:

   * Hyper-parameter choice mostly slides a model along the SAME ranking function.
     Within one model family on well-behaved data, most reasonable hyper-parameters
     produce similar orderings, so the tuning metric has limited room to change the
     outcome. Do not expect a large EUR-recall gain from the metric swap alone.

   * The metric choice bites hardest where the DECISION is made -- model selection
     between genuinely different candidates (the two-model paradox: higher AUCPR but
     lower EUR-recall), threshold/budget setting, and feature or objective changes.
     That is where a count-based metric actively misleads.

   * The decisive reasons to tune on amount-weighted AUROC here are therefore
     ROBUSTNESS, not a headline score: it is PRIOR-INVARIANT (so it stays valid on
     undersampled folds, whereas AUCPR is measured at an artificial base rate and is
     not comparable across folds or to production) and it is ALIGNED with the
     expected-value decision the model actually feeds. Note above that the AUCPR
     values are computed on natural-rate validation folds here -- had they been read
     off undersampled folds, they would be inflated and not comparable at all.

  In short: the metric swap is cheap insurance against selecting and reporting on a
  distorted quantity -- not a performance lever in its own right.""")
    print("\nThe threshold/budget is NOT tuned here -- it is set downstream with the")
    print("business on true-prior data (see review findings F-06 / F-07).")


if __name__ == "__main__":
    main()
