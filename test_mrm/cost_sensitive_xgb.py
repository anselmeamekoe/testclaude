"""
Cost-sensitive XGBoost for card fraud detection (scikit-learn API).
=================================================================

Instance-dependent ("example-dependent") cost-sensitive learning where the cost
of MISSING a fraud is the transaction amount, following the cost matrix of
Bahnsen, Stojanovic, Aouada & Ottersten (2013) / Hand et al. (2007):

                          Predicted genuine (0)     Predicted fraud (1)
    Actual genuine (0)        C_TN = 0                  C_FP = c_admin
    Actual fraud  (1)         C_FN = amount_i           C_TP = c_admin

Amount enters the pipeline in THREE coordinated places, which is what makes the
model prioritise high-value fraud:

  1. Training objective   : minimises the Average Expected Cost (csboost-style,
                            Hoppner et al. 2020). A missed fraud costs `amount`,
                            so the gradient for a fraud scales with its amount.
  2. Early-stopping metric : mean expected cost (lower is better) on a validation
                            set -> tuning/early-stopping optimise euros, not counts.
  3. Decision threshold    : Bayes-minimum-risk. Block (predict fraud) iff
                                 p * amount > c_admin
                            i.e. an instance-dependent threshold t_i = c_admin/amount_i.
                            A EUR 5,000 txn is blocked at a far lower probability
                            than a EUR 5 one.

Requires:  xgboost >= 1.6  (developed/tested on 3.3.0), scikit-learn, numpy.
    pip install xgboost scikit-learn numpy

Author's notes for model review:
  * A custom objective distorts probability calibration; recalibrate (Platt /
    isotonic on a TRUE-distribution set) before using raw probabilities anywhere
    other than the cost-based decision rule below.
  * Keep the C_FP term (review/attrition cost) non-trivial: prioritising
    high-amount fraud pushes the model to block large GENUINE transactions, which
    are exactly where false-decline costs (attrition, lost interchange) are worst.
  * Evaluate on the ORIGINAL imbalanced distribution with a temporal split.
"""

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def cost_matrix(y_true, amount, c_admin):
    """Per-instance costs for the four confusion-matrix cells."""
    amount = np.asarray(amount, dtype=float)
    c_tp = np.full_like(amount, c_admin)   # catch a fraud -> still pay admin cost
    c_fp = np.full_like(amount, c_admin)   # review a good txn -> admin cost
    c_fn = amount                          # miss a fraud -> lose the amount
    c_tn = np.zeros_like(amount)           # correctly pass a good txn -> free
    return c_tp, c_fp, c_fn, c_tn


# --------------------------------------------------------------------------- #
# 1. Training objectives (custom, sklearn signature: obj(y_true, y_pred))
#    NOTE: under a custom objective, `y_pred` is the RAW MARGIN, so p = sigmoid(y_pred).
# --------------------------------------------------------------------------- #
def make_expected_cost_objective(amount_train, c_admin, hess_floor=1e-6):
    """
    Average-Expected-Cost objective (directly minimises money).

    Expected cost of instance i with p = sigmoid(margin):
        y=1 :  p*C_TP + (1-p)*C_FN  =  C_FN + p*(C_TP - C_FN)
        y=0 :  p*C_FP + (1-p)*C_TN  =  C_TN + p*(C_FP - C_TN)
    AEC is linear in p, so with slope_i = (cost of predicting fraud - predicting genuine):
        d(AEC)/d(margin)  = slope_i * p(1-p)
        d2(AEC)/d(margin) = slope_i * p(1-p)(1-2p)      <-- can be negative
    The 2nd derivative is indefinite (cost is linear in p), so we use |hess|
    floored to a small positive value -- a PSD/Gauss-Newton approximation that
    keeps XGBoost's Newton step well-defined. This is standard for such objectives.
    """
    amount_train = np.asarray(amount_train, dtype=float)

    def objective(y_true, y_pred):
        y = np.asarray(y_true, dtype=float)
        p = sigmoid(y_pred)
        # slope_i = cost(predict fraud) - cost(predict genuine) at instance i
        slope = np.where(y == 1.0,
                         c_admin - amount_train,   # C_TP - C_FN  (usually negative)
                         c_admin - 0.0)            # C_FP - C_TN  (positive)
        pq = p * (1.0 - p)
        grad = slope * pq
        hess = np.maximum(np.abs(slope * pq * (1.0 - 2.0 * p)), hess_floor)
        return grad, hess

    return objective


def make_weighted_logloss_objective(amount_train, c_admin):
    """
    Alternative: cost-WEIGHTED cross-entropy. Always-positive hessian, very stable.
    Weight w_i = amount_i for frauds, c_admin for genuine transactions.

    This is mathematically equivalent to the one-liner:
        w = np.where(y == 1, amount, c_admin)
        XGBClassifier(...).fit(X, y, sample_weight=w)
    but is written out so the cost logic is explicit and extensible.
    """
    amount_train = np.asarray(amount_train, dtype=float)

    def objective(y_true, y_pred):
        y = np.asarray(y_true, dtype=float)
        p = sigmoid(y_pred)
        w = np.where(y == 1.0, amount_train, c_admin)
        grad = w * (p - y)
        hess = w * p * (1.0 - p)
        return grad, hess

    return objective


# --------------------------------------------------------------------------- #
# 2. Early-stopping / tuning metric (custom: metric(y_true, y_pred) -> (name, val))
# --------------------------------------------------------------------------- #
def make_expected_cost_metric(amount_eval, c_admin):
    """
    Mean expected cost on an eval set (LOWER is better -> XGBoost minimises it).

    The sklearn API (xgboost >= 2.0) expects a custom `eval_metric` callable to
    return a single float. (The low-level `xgb.train(custom_metric=...)` API instead
    expects a `(name, value)` tuple -- use `metric_named` below for that path.)
    """
    amount_eval = np.asarray(amount_eval, dtype=float)

    def metric(y_true, y_pred):
        y = np.asarray(y_true, dtype=float)
        p = sigmoid(y_pred)
        exp_cost = np.where(y == 1.0,
                            p * c_admin + (1.0 - p) * amount_eval,
                            p * c_admin)
        return float(np.mean(exp_cost))

    return metric


def make_expected_cost_metric_named(amount_eval, c_admin):
    """Same metric returning ('mean_exp_cost', value) for the low-level xgb.train API."""
    base = make_expected_cost_metric(amount_eval, c_admin)
    return lambda y_true, y_pred: ("mean_exp_cost", base(y_true, y_pred))


# --------------------------------------------------------------------------- #
# 3. Decision rule + evaluation metrics
# --------------------------------------------------------------------------- #
def predict_proba_cs(model, X):
    """Probabilities under a custom objective = sigmoid of the raw margin."""
    margin = model.predict(X, output_margin=True)
    return sigmoid(margin)


def cost_based_decision(proba, amount, c_admin):
    """Bayes-minimum-risk: block (predict fraud) iff  p * amount > c_admin.

    IMPORTANT: this rule already puts the transaction amount into the DECISION.
    If you also trained with an amount-weighted objective, you are counting the
    amount twice and will over-block. Pick ONE amount-aware mechanism:
      * calibrated probabilities + this BMR rule            (amount in the decision), or
      * amount-weighted objective + `tune_threshold_min_cost` (amount in training).
    """
    proba = np.asarray(proba, dtype=float)
    amount = np.asarray(amount, dtype=float)
    return (proba * amount > c_admin).astype(int)


def tune_threshold_min_cost(proba_val, y_val, amount_val, c_admin, n_grid=200):
    """Pick the single global probability threshold that minimises total cost on
    a validation set. Pairs with an amount-weighted training objective so that the
    amount is not double-counted at the decision step."""
    grid = np.unique(np.quantile(proba_val, np.linspace(0.0, 1.0, n_grid)))
    best_t, best_c = 0.5, np.inf
    for t in grid:
        c = total_cost(y_val, (proba_val > t).astype(int), amount_val, c_admin)
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t


def total_cost(y_true, y_pred, amount, c_admin):
    """Total realised monetary cost of a set of decisions."""
    y = np.asarray(y_true); yhat = np.asarray(y_pred)
    c_tp, c_fp, c_fn, c_tn = cost_matrix(y, amount, c_admin)
    cost = np.where((y == 1) & (yhat == 1), c_tp,
           np.where((y == 0) & (yhat == 1), c_fp,
           np.where((y == 1) & (yhat == 0), c_fn, c_tn)))
    return float(cost.sum())


def savings_score(y_true, y_pred, amount, c_admin):
    """
    Bahnsen 'savings': fraction of the trivial-baseline cost that the model saves.
        savings = (cost_baseline - cost_model) / cost_baseline
    where cost_baseline is the cheaper of 'predict everything genuine' and
    'predict everything fraud'. 1.0 = perfect, 0.0 = no better than trivial, <0 = worse.
    """
    y = np.asarray(y_true)
    cost_model    = total_cost(y, y_pred,               amount, c_admin)
    cost_all_good = total_cost(y, np.zeros_like(y),     amount, c_admin)  # sum of fraud amounts
    cost_all_bad  = total_cost(y, np.ones_like(y),      amount, c_admin)  # N * c_admin
    cost_baseline = min(cost_all_good, cost_all_bad)
    return 0.0 if cost_baseline == 0 else (cost_baseline - cost_model) / cost_baseline


def amount_recall(y_true, y_pred, amount):
    """Value-weighted recall: fraction of fraudulent EUROS detected."""
    y = np.asarray(y_true); yhat = np.asarray(y_pred); amt = np.asarray(amount, float)
    fraud_amt = amt[y == 1].sum()
    detected  = amt[(y == 1) & (yhat == 1)].sum()
    return float(detected / fraud_amt) if fraud_amt > 0 else 0.0


# --------------------------------------------------------------------------- #
# Training wrapper
# --------------------------------------------------------------------------- #
def train_cost_sensitive_xgb(X_tr, y_tr, amt_tr, X_val, y_val, amt_val, c_admin,
                             objective="expected_cost", **xgb_kwargs):
    """
    Fit a cost-sensitive XGBClassifier with early stopping on expected cost.
    `objective`: "expected_cost" (AEC / csboost) or "weighted_logloss".
    """
    if objective == "expected_cost":
        obj = make_expected_cost_objective(amt_tr, c_admin)
    elif objective == "weighted_logloss":
        obj = make_weighted_logloss_objective(amt_tr, c_admin)
    else:
        raise ValueError("objective must be 'expected_cost' or 'weighted_logloss'")

    params = dict(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0,
        reg_lambda=1.0, n_jobs=-1,
    )
    params.update(xgb_kwargs)

    model = XGBClassifier(
        objective=obj,
        eval_metric=make_expected_cost_metric(amt_val, c_admin),
        early_stopping_rounds=40,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _make_synthetic(n=60000, fraud_rate=0.012, seed=0):
    """Synthetic data with a count-vs-value tension: some high-amount frauds
    carry a WEAKER predictive signal, so a count-optimal model tends to miss them."""
    rng = np.random.default_rng(seed)
    d = 12
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    signal = (X @ w) / np.sqrt(d)                          # learnable part
    score = signal + rng.normal(0.0, 1.0, size=n)          # + irreducible noise
    thr = np.quantile(score, 1.0 - fraud_rate)             # exact base rate
    y = (score > thr).astype(int)

    amount = np.exp(rng.normal(3.2, 1.1, size=n))          # lognormal EUR
    big = (rng.uniform(size=n) < 0.18) & (y == 1)          # some large frauds ...
    amount[big] *= 9.0
    # ... whose feature signal we then corrupt, so a COUNT-optimal model tends to
    # miss exactly the high-value frauds (the count-vs-value tension we care about).
    X[big] += rng.normal(0.0, 1.5, size=(big.sum(), d))
    return X, y, amount


def _report(name, y_true, yhat, amt, c_admin):
    print(f"\n{name}")
    print(f"  count recall    : {recall_score(y_true, yhat):.3f}")
    print(f"  amount recall   : {amount_recall(y_true, yhat, amt):.3f}   <-- EUR detected")
    print(f"  precision       : {precision_score(y_true, yhat, zero_division=0):.3f}")
    print(f"  savings         : {savings_score(y_true, yhat, amt, c_admin):.3f}")
    print(f"  total cost (EUR): {total_cost(y_true, yhat, amt, c_admin):,.0f}")


def main():
    c_admin = 3.0  # EUR cost to review / contact for one alert
    X, y, amount = _make_synthetic()

    n = len(y)
    i1, i2 = int(n * 0.6), int(n * 0.8)                    # temporal-style split
    X_tr, y_tr, a_tr = X[:i1], y[:i1], amount[:i1]
    X_val, y_val, a_val = X[i1:i2], y[i1:i2], amount[i1:i2]
    X_te, y_te, a_te = X[i2:], y[i2:], amount[i2:]

    print(f"train={len(y_tr)}  val={len(y_val)}  test={len(y_te)} | "
          f"fraud rate={y.mean():.3%} | fraud EUR share={amount[y==1].sum()/amount.sum():.1%}")

    # === (0) The problem: count-oriented model (AUCPR) + fixed 0.5 threshold =====
    base = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         eval_metric="aucpr", early_stopping_rounds=40, n_jobs=-1)
    base.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    p_base_val = base.predict_proba(X_val)[:, 1]
    p_base = base.predict_proba(X_te)[:, 1]
    _report("(0) Baseline XGB  |  AUCPR obj, fixed 0.5 threshold",
            y_te, (p_base > 0.5).astype(int), a_te, c_admin)

    # === (1) Amount in the DECISION: same model + Bayes-minimum-risk threshold ===
    #     Highest-leverage, most auditable change. (Recalibrate probs in production.)
    _report("(1) Baseline XGB  +  BMR threshold  [amount in decision]",
            y_te, cost_based_decision(p_base, a_te, c_admin), a_te, c_admin)

    # === (2) Amount in TRAINING: cost-weighted objective + cost-tuned threshold ==
    #     Note: use a plain cost-tuned threshold here, NOT BMR, to avoid double-counting.
    csw = train_cost_sensitive_xgb(X_tr, y_tr, a_tr, X_val, y_val, a_val, c_admin,
                                   objective="weighted_logloss")
    p_csw_val = predict_proba_cs(csw, X_val)
    p_csw = predict_proba_cs(csw, X_te)
    t = tune_threshold_min_cost(p_csw_val, y_val, a_val, c_admin)
    _report(f"(2) Cost-weighted XGB  +  cost-tuned threshold={t:.3f}  [amount in training]",
            y_te, (p_csw > t).astype(int), a_te, c_admin)

    print("\nTakeaways:")
    print("  * (1) recovers most of the value with NO retraining -- put the amount in")
    print("    the decision layer first; it is transparent and easy to audit.")
    print("  * (2) an amount-aware objective is a refinement; pair it with a plain")
    print("    cost-tuned threshold so the amount is not counted twice.")
    print("  * Watch precision: prioritising high-amount fraud raises false declines")
    print("    on large GENUINE transactions -- keep the C_FP term realistic.")


if __name__ == "__main__":
    main()
