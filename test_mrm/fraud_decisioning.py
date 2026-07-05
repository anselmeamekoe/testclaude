"""
Fraud decisioning under undersampled training + business-owned threshold.
========================================================================

Extends `cost_sensitive_xgb.py` with the pieces that fit the reviewed setup:

  (a) amount_weighted_auc  : a THRESHOLD-FREE, VALUE-AWARE, PRIOR-ROBUST ranking
                             metric for hyper-parameter tuning. It keeps AUROC's
                             invariance to the class ratio (so it is safe on
                             undersampled folds) but weights each transaction by
                             its amount, so the objective rewards ranking
                             HIGH-VALUE fraud correctly -- the thing plain AUROC
                             and plain AUPRC both miss.
      cv_amount_weighted_auc : manual CV that mirrors the auditees' workflow --
                             undersample INSIDE each training fold, fit with a
                             FIXED n_estimators (no early stopping), and score on
                             the untouched natural-rate validation fold.

  (b) budget policy        : rank by expected value p*amount and alert down the
                             ranking until the daily budget is spent. This gives a
                             near-constant number of alerts/day AND the
                             expected-value prioritisation that a `p>t AND amount>c`
                             rectangle would lose. Needs only a MONOTONE score, so
                             it is unaffected by the undersampling mis-calibration.

  (c) rule layer           : hard business rules that an expected-value rule cannot
                             express (allow-list, card-testing velocity rule,
                             high-risk CNP combo, amount-floor triage), shown as a
                             precedence cascade over the ML budget queue.

Run:  python fraud_decisioning.py       (expects cost_sensitive_xgb.py alongside)
Requires: xgboost, scikit-learn, numpy.
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

from cost_sensitive_xgb import savings_score, amount_recall, total_cost


# =========================================================================== #
# (a) Amount-weighted AUC  (prior-robust + value-aware tuning objective)
# =========================================================================== #
def amount_weighted_auc(y_true, score, amount):
    """
    Amount-weighted ROC-AUC: the probability that a random fraudulent EURO is
    scored above a random genuine EURO. Weight w_i = amount_i.
      * amount = ones  -> ordinary AUROC.
      * PRIOR-INVARIANT: unaffected by the fraud/genuine ratio, so it is safe to
        compute on undersampled folds (unlike AUPRC / precision).
    O(n log n) via a weighted Mann-Whitney statistic with tie handling.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    w = np.asarray(amount, dtype=float)

    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]

    Wp = w[y == 1].sum()
    Wn = w[y == 0].sum()
    if Wp == 0 or Wn == 0:
        return 0.5

    auc = 0.0
    cum_neg = 0.0          # genuine weight strictly below current score group
    i, n = 0, len(s)
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        blk = slice(i, j)
        yb, wb = y[blk], w[blk]
        neg_blk = wb[yb == 0].sum()
        pos_blk = wb[yb == 1].sum()
        # every positive in a tied block beats `cum_neg` fully + half of its own block's negatives
        auc += pos_blk * (cum_neg + 0.5 * neg_blk)
        cum_neg += neg_blk
        i = j
    return float(auc / (Wp * Wn))


def undersample_majority(X, y, amount, target_fraud_rate, rng):
    """Drop random genuine rows so the fraud rate reaches `target_fraud_rate`.
    Undersamples the MAJORITY only; keeps every fraud."""
    y = np.asarray(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos = len(pos)
    n_neg_keep = int(round(n_pos * (1 - target_fraud_rate) / target_fraud_rate))
    n_neg_keep = min(n_neg_keep, len(neg))
    neg_keep = rng.choice(neg, size=n_neg_keep, replace=False)
    idx = np.concatenate([pos, neg_keep])
    rng.shuffle(idx)
    return X[idx], y[idx], amount[idx]


def cv_amount_weighted_auc(params, X, y, amount, n_splits=3,
                           undersample_rate=0.10, n_estimators=400, seed=0):
    """
    Manual stratified CV that reproduces the auditees' setup:
      - undersample the MAJORITY inside each TRAIN fold only,
      - fit XGB with a FIXED number of trees (no early stopping),
      - evaluate on the UNTOUCHED natural-rate validation fold,
      - score with amount-weighted AUC (prior-robust, value-aware).
    Returns (mean, std) across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    scores = []
    for tr, va in skf.split(X, y):
        Xtr, ytr, atr = undersample_majority(X[tr], y[tr], amount[tr],
                                              undersample_rate, rng)
        model = XGBClassifier(n_estimators=n_estimators, n_jobs=-1,
                              eval_metric="logloss", **params)
        model.fit(Xtr, ytr, verbose=False)                 # no early stopping
        p_va = model.predict_proba(X[va])[:, 1]            # natural-rate fold
        scores.append(amount_weighted_auc(y[va], p_va, amount[va]))
    return float(np.mean(scores)), float(np.std(scores))


# =========================================================================== #
# (b) Budget-constrained decisioning:  rank by p*amount, cut at N alerts/day
# =========================================================================== #
def expected_value_score(proba, amount):
    """Ranking score = expected monetary loss if allowed. Monotone in both."""
    return np.asarray(proba, float) * np.asarray(amount, float)


def budget_cutoff(score_ref, day_ref, alerts_per_day):
    """
    Score cutoff that yields ~`alerts_per_day` on a reference set, generalisable
    to future data. Calibrated as the k-th largest score where
    k = alerts_per_day * (#distinct days in the reference set).
    """
    score_ref = np.asarray(score_ref, float)
    n_days = len(np.unique(day_ref))
    k = int(round(alerts_per_day * n_days))
    k = max(1, min(k, len(score_ref)))
    return float(np.partition(score_ref, len(score_ref) - k)[len(score_ref) - k])


def decide_budget(score, cutoff):
    """Alert iff the expected-value score clears the budget cutoff."""
    return (np.asarray(score, float) >= cutoff).astype(int)


# =========================================================================== #
# (c) Rule layer  (business rules an expected-value rule cannot express)
# =========================================================================== #
# Feature dict keys used by the rules below:
#   is_cnp, country_mismatch, new_device, velocity_1h, mcc_risk,
#   hour, trusted_recurring, amount

FLOOR_EUR = 5.0        # review economics: not worth an analyst's time below this
VELOCITY_LIMIT = 6     # >= this many txns in the last hour -> card-testing burst
COMBO_AMOUNT = 150.0   # CNP high-risk combo only fires above this amount


def rule_allow_list(F):
    """Trusted recurring merchants (e.g. subscriptions) -> never alert."""
    return F["trusted_recurring"] == 1


def rule_hard_block(F):
    """
    Must-alert rules, independent of the ML score:
      R1  velocity: >= VELOCITY_LIMIT txns in 1h  (catches small-amount card-testing
          probes that p*amount ranking would deprioritise),
      R2  classic high-risk CNP combo: card-not-present AND country mismatch AND
          new device AND amount above COMBO_AMOUNT.
    """
    r1_velocity = F["velocity_1h"] >= VELOCITY_LIMIT
    r2_combo = ((F["is_cnp"] == 1) & (F["country_mismatch"] == 1) &
                (F["new_device"] == 1) & (F["amount"] > COMBO_AMOUNT))
    return r1_velocity | r2_combo, r1_velocity, r2_combo


def rule_amount_floor(F, floor=FLOOR_EUR):
    """Triage: transactions below the floor are not worth reviewing."""
    return F["amount"] < floor


def decide_cascade(F, proba, day, alerts_per_day,
                   F_ref=None, proba_ref=None, day_ref=None):
    """
    Precedence cascade (auditable order):
        1. hard-block rules   -> ALERT   (safety first)
        2. allow-list         -> pass
        3. amount-floor triage-> pass
        4. remaining: rank by p*amount, alert down to the daily ML budget.
    The ML budget cutoff is calibrated on the ELIGIBLE population of a reference
    set (`*_ref`); if none is given it is calibrated in-sample (demo convenience).
    Returns (final_alert, parts_dict).
    """
    amount = F["amount"]
    hard, r_vel, r_combo = rule_hard_block(F)
    allow = rule_allow_list(F)
    floor = rule_amount_floor(F)
    eligible = ~hard & ~allow & ~floor

    score = expected_value_score(proba, amount)

    # calibrate the ML budget cutoff on the reference eligible population, and
    # hold the TOTAL daily alerts near `alerts_per_day` by subtracting the alerts
    # the hard rules are expected to spend per day.
    if F_ref is None:
        n_days = len(np.unique(day))
        ml_budget = max(0.0, alerts_per_day - hard.sum() / n_days)
        cutoff = budget_cutoff(score[eligible], day[eligible], ml_budget)
    else:
        hard_r, _, _ = rule_hard_block(F_ref)
        elig_ref = ~hard_r & ~rule_allow_list(F_ref) & ~rule_amount_floor(F_ref)
        score_ref = expected_value_score(proba_ref, F_ref["amount"])
        n_days_ref = len(np.unique(day_ref))
        ml_budget = max(0.0, alerts_per_day - hard_r.sum() / n_days_ref)
        cutoff = budget_cutoff(score_ref[elig_ref], day_ref[elig_ref], ml_budget)

    ml_alert = eligible & (score >= cutoff)
    final_alert = hard | ml_alert
    parts = dict(hard=hard, r_velocity=r_vel, r_combo=r_combo, allow=allow,
                 floor=floor, ml_alert=ml_alert, cutoff=cutoff)
    return final_alert.astype(int), parts


# =========================================================================== #
# Enriched synthetic data with interpretable features
# =========================================================================== #
def make_synthetic_with_features(n=60000, fraud_rate=0.012, n_days=60, seed=0):
    rng = np.random.default_rng(seed)

    is_cnp = (rng.uniform(size=n) < 0.55).astype(int)
    country_mismatch = (rng.uniform(size=n) < 0.08).astype(int)
    new_device = (rng.uniform(size=n) < 0.12).astype(int)
    mcc_risk = (rng.uniform(size=n) < 0.15).astype(int)
    hour = rng.integers(0, 24, size=n)
    trusted_recurring = (rng.uniform(size=n) < 0.10).astype(int)

    # velocity: mostly quiet, with occasional card-testing bursts
    velocity_1h = rng.poisson(0.4, size=n)
    burst = rng.uniform(size=n) < 0.02
    velocity_1h[burst] += rng.integers(6, 16, size=burst.sum())

    amount = np.exp(rng.normal(3.2, 1.1, size=n))          # lognormal EUR
    day = rng.integers(0, n_days, size=n)

    latent = rng.normal(size=n)                            # extra learnable signal

    # fraud propensity driven by the interpretable features
    logit = (-3.0
             + 1.6 * is_cnp
             + 1.3 * country_mismatch
             + 1.4 * new_device
             + 0.9 * mcc_risk
             + 0.12 * velocity_1h          # modest per-txn signal from velocity
             + 2.2 * (is_cnp & country_mismatch & new_device)
             - 4.0 * trusted_recurring
             + 0.8 * latent
             + rng.normal(0, 0.8, size=n))
    thr = np.quantile(logit, 1.0 - fraud_rate)             # exact base rate
    y = (logit > thr).astype(int)

    # CARD-TESTING frauds: bursts of MANY TINY-amount txns with only modest
    # per-transaction signal. Expected value p*amount is small, so a budget policy
    # deprioritises them -- the VELOCITY RULE is what catches the pattern.
    card_testing = burst & (rng.uniform(size=n) < 0.6)
    y[card_testing] = 1
    amount[card_testing] = rng.uniform(1.0, 8.0, size=card_testing.sum())
    trusted_recurring[card_testing] = 0

    # make some HIGH-AMOUNT frauds harder: inflate amount and blur their signal
    big = (rng.uniform(size=n) < 0.18) & (y == 1) & (~card_testing)
    amount[big] *= 9.0
    latent[big] += rng.normal(0, 1.5, size=big.sum())
    hide = big & (rng.uniform(size=n) < 0.5)
    new_device[hide] = 0

    F = dict(is_cnp=is_cnp, country_mismatch=country_mismatch, new_device=new_device,
             mcc_risk=mcc_risk, hour=hour, trusted_recurring=trusted_recurring,
             velocity_1h=velocity_1h, amount=amount)

    model_cols = ["is_cnp", "country_mismatch", "new_device", "mcc_risk", "hour",
                  "trusted_recurring", "velocity_1h"]
    X = np.column_stack([F[c] for c in model_cols] +
                        [np.log1p(amount), latent]).astype(float)
    return F, X, y, amount, day


# =========================================================================== #
# Demo
# =========================================================================== #
def _report(name, y, yhat, amount, day, extra=""):
    n_days = len(np.unique(day))
    print(f"\n{name}")
    print(f"  alerts/day (avg) : {yhat.sum()/n_days:.1f}")
    print(f"  count recall     : {np.mean(yhat[y==1]==1):.3f}")
    print(f"  amount recall    : {amount_recall(y, yhat, amount):.3f}   <-- EUR detected")
    prec = (yhat[y==1]==1).sum() / max(yhat.sum(), 1)
    print(f"  precision        : {prec:.3f}")
    print(f"  savings          : {savings_score(y, yhat, amount, 3.0):.3f}")
    if extra:
        print("  " + extra)


def main():
    F, X, y, amount, day = make_synthetic_with_features()
    n = len(y)
    i1, i2 = int(n * 0.6), int(n * 0.8)

    def sub(a, s, e):  # slice helper for the feature dict
        return {k: v[s:e] for k, v in a.items()}

    X_tr, X_val, X_te = X[:i1], X[i1:i2], X[i2:]
    y_tr, y_val, y_te = y[:i1], y[i1:i2], y[i2:]
    a_tr, a_val, a_te = amount[:i1], amount[i1:i2], amount[i2:]
    d_val, d_te = day[i1:i2], day[i2:]
    F_val, F_te = sub(F, i1, i2), sub(F, i2, n)

    print(f"n={n} | fraud rate={y.mean():.2%} | fraud EUR share={amount[y==1].sum()/amount.sum():.1%}")

    # ---- (a) amount-weighted-AUC CV tuning (undersample inside train folds) ----
    print("\n=== (a) CV tuning on amount-weighted AUC (prior-robust, value-aware) ===")
    grid = [dict(max_depth=d, learning_rate=lr)
            for d in (4, 6) for lr in (0.05, 0.10)]
    best, best_params = -1, None
    for p in grid:
        m, s = cv_amount_weighted_auc(p, X_tr, y_tr, a_tr, n_splits=3,
                                      undersample_rate=0.10, n_estimators=400)
        print(f"  {p}: amtAUC = {m:.4f} +/- {s:.4f}")
        if m > best:
            best, best_params = m, p
    print(f"  -> selected {best_params}  (amtAUC={best:.4f})")

    # ---- final model on undersampled train; monotone score is all we need ----
    rng = np.random.default_rng(1)
    Xtr_u, ytr_u, _ = undersample_majority(X_tr, y_tr, a_tr, 0.10, rng)
    model = XGBClassifier(n_estimators=400, n_jobs=-1, eval_metric="logloss",
                          **best_params)
    model.fit(Xtr_u, ytr_u, verbose=False)
    p_val = model.predict_proba(X_val)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    ALERTS_PER_DAY = 10          # ~200 txns/day in the test window -> ~5% alert rate

    # ---- (b) budget policy: rank by p*amount, cut at N/day (NO rules) ----
    print(f"\n=== (b) Budget policy: rank by p*amount, ~{ALERTS_PER_DAY} alerts/day ===")
    score_val = expected_value_score(p_val, a_val)
    score_te = expected_value_score(p_te, a_te)
    cutoff = budget_cutoff(score_val, d_val, ALERTS_PER_DAY)      # calibrate on val
    yhat_budget = decide_budget(score_te, cutoff)
    _report("Rank by p*amount -> budget cut", y_te, yhat_budget, a_te, d_te)

    # contrast: p>t AND amount>c rectangle, VOLUME-MATCHED to the same alerts/day.
    # Fix a plausible amount gate, then solve the p gate on val to hit the budget.
    c_rect = 50.0
    n_days_val = len(np.unique(d_val))
    target = ALERTS_PER_DAY * n_days_val
    cand = p_val[a_val > c_rect]
    t_rect = np.quantile(cand, 1 - min(1.0, target / max(len(cand), 1))) if len(cand) else 1.0
    yhat_rect = ((p_te > t_rect) & (a_te > c_rect)).astype(int)
    _report(f"Rectangle p>{t_rect:.3f} AND amount>{c_rect:.0f}  (volume-matched)",
            y_te, yhat_rect, a_te, d_te)
    missed_hi = (y_te == 1) & (yhat_rect == 0) & (yhat_budget == 1)
    print(f"  -> ranking catches {int(missed_hi.sum())} frauds the rectangle misses "
          f"(their fraud EUR: {a_te[missed_hi].sum():,.0f}); these are high-amount /"
          f" moderate-p cases below the rectangle's p gate.")

    # ---- (c) full cascade: rules + allow-list + floor + ML budget ----
    print(f"\n=== (c) Rule cascade + ML budget (~{ALERTS_PER_DAY} alerts/day) ===")
    yhat_casc, parts = decide_cascade(F_te, p_te, d_te, ALERTS_PER_DAY,
                                      F_ref=F_val, proba_ref=p_val, day_ref=d_val)
    hard, r_vel, r_combo = parts["hard"], parts["r_velocity"], parts["r_combo"]
    allow, floor, ml = parts["allow"], parts["floor"], parts["ml_alert"]
    extra = (f"rule-alerts: velocity={r_vel.sum()} (frauds {int(y_te[r_vel].sum())}), "
             f"combo={r_combo.sum()} (frauds {int(y_te[r_combo].sum())}) | "
             f"ML-alerts={ml.sum()} | allow-listed={allow.sum()} "
             f"(frauds among them {int(y_te[allow].sum())}) | floor-skipped={floor.sum()}")
    _report("Cascade (rules override ML)", y_te, yhat_casc, a_te, d_te, extra=extra)

    # how many small-amount card-testing frauds did the VELOCITY rule save that the
    # pure budget policy (which ranks by p*amount) would have dropped?
    saved = r_vel & (y_te == 1) & (yhat_budget == 0)
    print(f"\n  Velocity rule caught {int(saved.sum())} fraudulent card-testing txns "
          f"that the p*amount budget policy missed (median amount "
          f"EUR {np.median(a_te[saved]) if saved.any() else 0:.0f}).")

    # ---- three example transactions, one per rule ----
    print("\n=== Rule-firing examples (test set) ===")
    def show(idx, tag):
        if idx is None:
            print(f"  [{tag}] none in sample"); return
        print(f"  [{tag}] amount=EUR {F_te['amount'][idx]:.0f}  cnp={F_te['is_cnp'][idx]}  "
              f"ctry_mismatch={F_te['country_mismatch'][idx]}  new_device={F_te['new_device'][idx]}  "
              f"velocity_1h={F_te['velocity_1h'][idx]}  trusted={F_te['trusted_recurring'][idx]}  "
              f"| actual_fraud={y_te[idx]}")
    first = lambda mask: int(np.where(mask)[0][0]) if mask.any() else None
    show(first(r_vel),  "VELOCITY  hard-block")
    show(first(r_combo), "CNP-COMBO hard-block")
    show(first(allow),  "ALLOW-LIST pass     ")
    show(first(floor & (y_te == 0)), "AMOUNT-FLOOR triage ")


if __name__ == "__main__":
    main()
