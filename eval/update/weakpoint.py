"""
WeakpointAnalyzer
=================
Find WHERE the model is weak (regions worse than its global metric) and, more
importantly, WHY — separating causes that call for different fixes:

    Local overfitting   train loss low but test loss high in the SAME region
                        => the model memorized local training points. Fix: reduce
                        local capacity / regularize / prune.
    Data sparsity       few training rows support the region => high variance.
                        Fix: collect/upsample there.
    Distribution shift  region is over-represented at test time vs train
                        (covariate shift). Fix: reweight / retrain on recent data.
    Irreducible noise   near-duplicate inputs carry conflicting labels
                        => Bayes error is high here, not an overfit. Fix: nothing
                        model-side; improve labels/features.
    Hard / underfit     both train AND test loss high => model too weak locally.
                        Fix: add capacity or features.

How regions are found
---------------------
We fit a shallow decision tree ("error tree") that predicts per-sample TEST loss
from the features. Each leaf is an interpretable rule (e.g. amount>500 & hour<6)
and defines a slice. Because a tree, not a per-feature scan, the slices can be
multivariate — that is what surfaces *regional* problems a 1-D scan misses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .data import Dataset, EvalConfig, Task
from .metrics import _proba, headline_name, per_sample_loss, score
from .modeva_ext import adaptive_weak_threshold, _scalar_output
from . import viz

CAUSE_COLOR = {
    "Local overfitting": viz.BAD,
    "Data sparsity": viz.WARN,
    "Distribution shift": "#8a4fff",
    "Irreducible noise": viz.MUTED,
    "Hard / underfit": "#c86b00",
    "Elevated error": "#5b6b82",
    "(healthy)": viz.GOOD,
}


# --------------------------------------------------------------------------- #
@dataclass
class WeakpointReport:
    slices: pd.DataFrame
    figures_: dict = field(default_factory=dict)
    verdict: str = ""
    details: dict = field(default_factory=dict)

    def figures(self) -> dict[str, go.Figure]:
        return self.figures_

    def show(self):
        for f in self.figures_.values():
            f.show()


class WeakpointAnalyzer:
    def __init__(self, config: Optional[EvalConfig] = None):
        self.cfg = config or EvalConfig()

    # ---- public ---------------------------------------------------------- #
    def analyze(self, model, ds: Dataset) -> WeakpointReport:
        feats = ds.feature_names
        Xtr, Xte = ds.X_train.to_numpy(float), ds.X_test.to_numpy(float)
        ytr, yte = ds.y_train.to_numpy(), ds.y_test.to_numpy()

        loss_te = per_sample_loss(yte, model, Xte, ds.task)
        loss_tr = per_sample_loss(ytr, model, Xtr, ds.task)
        g_loss_te = float(loss_te.mean())
        g_loss_tr = float(loss_tr.mean())
        g_score = score(yte, model, Xte, ds.task)

        # ---- discover slices via an error tree --------------------------- #
        etree = DecisionTreeRegressor(
            max_depth=self.cfg.max_slice_depth,
            min_samples_leaf=max(20, int(self.cfg.min_slice_frac * len(yte))),
            random_state=self.cfg.random_state,
        ).fit(Xte, loss_te)
        leaf_te = etree.apply(Xte)
        leaf_tr = etree.apply(Xtr)
        rules = _leaf_rules(etree, feats)

        # neighbor structure for label-noise proxy AND density axis (fit once)
        scaler = StandardScaler().fit(Xte)
        nn = NearestNeighbors(n_neighbors=min(11, len(yte))).fit(scaler.transform(Xte))
        dist, nbr_idx = nn.kneighbors(scaler.transform(Xte))
        noise_all = self._neighbor_disagreement(yte, nbr_idx, ds.task)
        # AMIF-style density axis (cheap): mean distance to neighbours; larger = sparser
        density_all = dist[:, 1:].mean(axis=1)
        g_density = float(np.median(density_all) + 1e-12)

        rows = []
        for lid in np.unique(leaf_te):
            m_te = leaf_te == lid
            if m_te.sum() < max(10, int(0.5 * self.cfg.min_slice_frac * len(yte))):
                continue
            m_tr = leaf_tr == lid
            row = self._diagnose_slice(
                ds, model, m_te, m_tr, loss_te, loss_tr, yte, Xte,
                g_loss_te, g_loss_tr, g_score, noise_all, density_all, g_density,
                rules.get(lid, "all"))
            rows.append(row)

        slices = pd.DataFrame(rows).sort_values("loss_ratio", ascending=False)
        slices = slices.reset_index(drop=True)

        # ADAPTIVE threshold (Modeva: mu + beta*sigma over regions), kept in a
        # sensible band: never flag a region <10% worse, always flag one >=50% worse.
        if len(slices) > 2:
            raw = adaptive_weak_threshold(slices["loss_ratio"].to_numpy(), beta=1.0)
            thr = float(min(1.5, max(1.1, raw)))
        else:
            thr = 1.25
        slices["likely_cause"] = [
            self._attribute(r, g_loss_te, g_loss_tr, thr) for _, r in slices.iterrows()]

        # error drivers = feature importances of the error tree (free — already fit)
        drivers = pd.DataFrame({"feature": feats,
                                "importance": np.round(etree.feature_importances_, 3)}) \
            .sort_values("importance", ascending=False).reset_index(drop=True)

        figs = {
            "table": self._table_fig(slices),
            "map": self._embedding_fig(Xte, loss_te, leaf_te, slices, feats, Xte),
            "landscape": self._landscape_fig(density_all, noise_all, loss_te,
                                             g_density),
            "drivers": self._drivers_fig(drivers),
            "profile": self._feature_profile_fig(ds, loss_te),
        }
        verdict = self._verdict(slices, g_score, ds.task, thr)
        return WeakpointReport(slices=slices, figures_=figs, verdict=verdict,
                               details=dict(global_score=g_score,
                                            global_test_loss=g_loss_te,
                                            global_train_loss=g_loss_tr,
                                            weak_threshold=round(thr, 3),
                                            error_drivers=drivers))

    # ---- per-slice diagnostics ------------------------------------------ #
    def _diagnose_slice(self, ds, model, m_te, m_tr, loss_te, loss_tr, yte, Xte,
                        g_loss_te, g_loss_tr, g_score, noise_all, density_all,
                        g_density, rule) -> dict:
        n_te = int(m_te.sum())
        n_tr = int(m_tr.sum())
        test_frac = n_te / len(loss_te)
        train_frac = n_tr / len(loss_tr)

        local_loss_te = float(loss_te[m_te].mean())
        local_loss_tr = float(loss_tr[m_tr].mean()) if n_tr > 0 else np.nan
        loss_ratio = local_loss_te / (g_loss_te + 1e-12)
        overfit_gap = (local_loss_te - local_loss_tr) if n_tr > 0 else np.nan
        coverage = (train_frac / (test_frac + 1e-12))
        noise = float(noise_all[m_te].mean())
        density_ratio = float(density_all[m_te].mean() / g_density)   # >1 = sparser
        loss_std = float(loss_te[m_te].std())                         # Modeva Uncertainty(R)
        pred_var = float(np.var(_scalar_output(model, Xte[m_te], ds.task))) \
            if n_te > 1 else np.nan                                   # Modeva LocalComplexity(R)
        can_score = n_te > 5 and (ds.task == Task.REGRESSION or
                                   len(np.unique(yte[m_te])) > 1)
        local_score = score(yte[m_te], model, Xte[m_te], ds.task) if can_score else np.nan
        margin = self._uncertainty(model, Xte[m_te], ds.task)

        return dict(
            rule=rule, n_test=n_te, test_share=round(test_frac, 3),
            train_share=round(train_frac, 3),
            local_score=round(float(local_score), 3) if local_score == local_score else np.nan,
            global_score=round(g_score, 3),
            local_test_loss=round(local_loss_te, 3),
            local_train_loss=round(local_loss_tr, 3) if n_tr else np.nan,
            loss_ratio=round(loss_ratio, 2),
            overfit_gap=round(overfit_gap, 3) if n_tr else np.nan,
            coverage=round(coverage, 2), density_ratio=round(density_ratio, 2),
            label_noise=round(noise, 3), local_pred_var=round(pred_var, 4),
            loss_std=round(loss_std, 3), uncertainty=round(margin, 3),
        )

    def _attribute(self, r, g_te, g_tr, thr) -> str:
        """Transparent rule set over a slice row. `thr` is the adaptive
        (mu+beta*sigma) loss-ratio threshold computed across regions."""
        if r["loss_ratio"] < thr:
            return "(healthy)"
        l_te, l_tr = r["local_test_loss"], r["local_train_loss"]
        overfit_gap = r["overfit_gap"]
        global_gap = g_te - g_tr
        # 1) local overfitting: test loss >> train loss HERE, beyond the global gap
        if overfit_gap == overfit_gap and overfit_gap > max(2 * global_gap, 0.15) \
                and (l_tr < 0.8 * g_te):
            return "Local overfitting"
        # 2) sparsity: few train points OR AMIF density axis says region is sparse
        if r["train_share"] < 0.4 * r["test_share"] or r["coverage"] < 0.5 \
                or r["density_ratio"] > 1.5:
            return "Data sparsity"
        # 3) covariate shift: region much bigger at test time than train time
        if r["test_share"] > 1.8 * r["train_share"]:
            return "Distribution shift"
        # 4) irreducible: neighbours disagree a lot => high Bayes error
        if r["label_noise"] > 0.35:
            return "Irreducible noise"
        # 5) hard/underfit: train ALSO bad here
        if l_tr == l_tr and l_tr > 1.3 * g_tr:
            return "Hard / underfit"
        return "Elevated error"

    @staticmethod
    def _neighbor_disagreement(y, nbr_idx, task: Task) -> np.ndarray:
        y = np.asarray(y)
        neigh = y[nbr_idx[:, 1:]]  # drop self
        if task == Task.CLASSIFICATION:
            return (neigh != y[:, None]).mean(axis=1)
        # regression: coefficient of variation of neighbor targets, clipped
        mu = neigh.mean(axis=1)
        sd = neigh.std(axis=1)
        return np.clip(sd / (np.abs(mu) + np.std(y) + 1e-9), 0, 1)

    @staticmethod
    def _uncertainty(model, X, task: Task) -> float:
        if len(X) == 0:
            return np.nan
        if task == Task.CLASSIFICATION:
            p = _proba(model, X)
            top = np.sort(p, axis=1)[:, -1]
            return float(1 - np.mean(top))  # closeness to a coin flip
        return float(np.std(np.asarray(model.predict(X), float)))

    # ---- figures --------------------------------------------------------- #
    def _table_fig(self, slices: pd.DataFrame) -> go.Figure:
        show = slices[["rule", "n_test", "test_share", "train_share", "local_score",
                       "loss_ratio", "overfit_gap", "coverage", "density_ratio",
                       "label_noise", "likely_cause"]].copy()
        colors = [CAUSE_COLOR.get(c, "white") for c in show["likely_cause"]]
        # tint only the cause column
        fill = []
        for col in show.columns:
            if col == "likely_cause":
                fill.append([_tint(c) for c in colors])
            else:
                fill.append(["white"] * len(show))
        fig = go.Figure(go.Table(
            header=dict(values=[f"<b>{c}</b>" for c in show.columns],
                        fill_color="#f2f5fa", align="left",
                        font=dict(color=viz.INK, size=11), height=30),
            cells=dict(values=[show[c].tolist() for c in show.columns],
                       fill_color=fill, align="left",
                       font=dict(color=viz.INK, size=10.5), height=26),
        ))
        fig.update_layout(title=dict(text="Weak regions ranked by loss ratio (vs global)",
                                     font=dict(size=16, color=viz.INK)),
                          margin=dict(l=10, r=10, t=50, b=10),
                          height=70 + 28 * (len(show) + 1))
        return fig

    def _embedding_fig(self, X, loss, leaf, slices, feats, X_raw) -> go.Figure:
        """2D PCA of test points colored by loss; dropdown isolates each weak slice."""
        emb = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
        hover = ["<br>".join(f"{f}={v:.3g}" for f, v in zip(feats, row))
                 for row in X_raw]
        base = go.Scatter(
            x=emb[:, 0], y=emb[:, 1], mode="markers", name="all test points",
            marker=dict(size=5, color=loss, colorscale="OrRd", showscale=True,
                        colorbar=dict(title="loss")),
            text=hover, hovertemplate="%{text}<br>loss=%{marker.color:.3f}<extra></extra>")
        fig = go.Figure(base)

        # one (hidden) highlight trace per weak slice, toggled by a dropdown
        buttons = [dict(label="All points", method="update",
                        args=[{"visible": [True] + [False] * len(slices)}])]
        for i, (_, srow) in enumerate(slices.iterrows()):
            mask = np.array([_row_matches(srow["rule"], dict(zip(feats, r))) for r in X_raw])
            fig.add_trace(go.Scatter(
                x=emb[mask, 0], y=emb[mask, 1], mode="markers",
                name=f"slice {i}", visible=False,
                marker=dict(size=8, color=CAUSE_COLOR.get(srow["likely_cause"], viz.BAD),
                            line=dict(color="black", width=0.5)),
                hovertext=[srow["rule"]] * int(mask.sum()),
                hovertemplate="%{hovertext}<extra></extra>"))
            visible = [True] + [False] * len(slices)
            visible[i + 1] = True
            buttons.append(dict(
                label=f"{i}: {srow['likely_cause']} (×{srow['loss_ratio']})",
                method="update", args=[{"visible": visible}]))

        fig.update_layout(
            updatemenus=[dict(buttons=buttons, x=1.02, y=1, xanchor="left")],
            title=dict(text="Interactive weak-region map (PCA of test set)",
                       font=dict(size=16, color=viz.INK)))
        fig.update_xaxes(title="PC1", gridcolor=viz.GRID)
        fig.update_yaxes(title="PC2", gridcolor=viz.GRID)
        fig.update_layout(template="plotly_white", height=520,
                          margin=dict(l=50, r=180, t=60, b=50))
        return fig

    def _feature_profile_fig(self, ds: Dataset, loss_te) -> go.Figure:
        """Mean test loss across quantile bins of each feature (univariate view)."""
        X = ds.X_test
        feats = ds.feature_names
        pick = feats[: min(6, len(feats))]
        fig = go.Figure()
        for f in pick:
            col = X[f].to_numpy(float)
            try:
                q = pd.qcut(col, q=min(6, len(np.unique(col))), duplicates="drop")
            except Exception:
                continue
            grp = pd.DataFrame({"loss": loss_te, "bin": q}).groupby("bin", observed=True)["loss"].mean()
            centers = [iv.mid for iv in grp.index]
            fig.add_trace(go.Scatter(x=centers, y=grp.values, mode="lines+markers", name=f))
        fig.add_hline(y=float(np.mean(loss_te)), line=dict(color=viz.MUTED, dash="dot"),
                      annotation_text="global mean loss")
        fig.update_xaxes(title="feature value (quantile bin centers)")
        fig.update_yaxes(title="mean test loss")
        return viz._base(fig, "Univariate error profiles", height=440)

    def _landscape_fig(self, density_all, noise_all, loss_te, g_density) -> go.Figure:
        """AMIF-style two-axis view (cheap): x = sparsity (neighbour distance),
        y = label noise (irreducibility), colour = test loss. Bottom-left = dense
        + clean; top-right = sparse + noisy."""
        fig = go.Figure(go.Scatter(
            x=density_all, y=noise_all, mode="markers",
            marker=dict(size=5, color=loss_te, colorscale="OrRd", showscale=True,
                        colorbar=dict(title="loss")),
            hovertemplate="sparsity=%{x:.2f}<br>noise=%{y:.2f}"
                          "<br>loss=%{marker.color:.3f}<extra></extra>"))
        fig.add_vline(x=1.5 * g_density, line=dict(color=viz.WARN, dash="dot"),
                      annotation_text="sparse")
        fig.add_hline(y=0.35, line=dict(color=viz.MUTED, dash="dot"),
                      annotation_text="noisy")
        fig.update_xaxes(title="data sparsity  (mean kNN distance →)")
        fig.update_yaxes(title="label noise  (kDN disagreement →)")
        return viz._base(fig, "Weakness landscape (density × signal, AMIF-style)",
                         height=460)

    def _drivers_fig(self, drivers: pd.DataFrame) -> go.Figure:
        d = drivers.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h",
                               marker_color=viz.ACCENT))
        fig.update_xaxes(title="error-tree importance")
        return viz._base(fig, "Error drivers (features that predict where loss is high)",
                         height=360)

    def _verdict(self, slices, g_score, task, thr) -> str:
        name = headline_name(task)
        weak = slices[slices["likely_cause"] != "(healthy)"]
        if weak.empty:
            return (f"No region beyond the adaptive threshold (loss ratio > {thr:.2f}) "
                    f"vs global {name}={g_score:.3f}. Error looks evenly spread — no "
                    "obvious local/regional overfitting.")
        counts = weak["likely_cause"].value_counts().to_dict()
        worst = weak.iloc[0]
        parts = [f"Global {name}={g_score:.3f}. Adaptive weak threshold = loss ratio "
                 f"> {thr:.2f}. Found {len(weak)} weak region(s). "
                 f"Worst: `{worst['rule']}` — {worst['n_test']} test rows, "
                 f"loss ×{worst['loss_ratio']} vs global, attributed to "
                 f"{worst['likely_cause']}."]
        by = ", ".join(f"{k}: {v}" for k, v in counts.items())
        parts.append("Cause breakdown — " + by + ".")
        if "Local overfitting" in counts:
            parts.append("Local overfitting present → regularize/prune where train "
                         "loss is low but test loss spikes.")
        if "Distribution shift" in counts:
            parts.append("Shift present → reweight or retrain on recent data.")
        if "Data sparsity" in counts:
            parts.append("Sparsity present → collect/upsample the flagged regions.")
        return " ".join(parts)


# --------------------------------------------------------------------------- #
# rule extraction / matching helpers                                          #
# --------------------------------------------------------------------------- #
def _leaf_rules(tree, feature_names) -> dict:
    t = tree.tree_
    rules: dict[int, list] = {}

    def recurse(node, conds):
        if t.children_left[node] == -1:
            rules[node] = dict(conds)  # copy
            return
        f = feature_names[t.feature[node]]
        thr = t.threshold[node]
        left = dict(conds)
        left[f] = (left.get(f, (-np.inf, np.inf))[0], min(left.get(f, (-np.inf, np.inf))[1], thr))
        recurse(t.children_left[node], left)
        right = dict(conds)
        right[f] = (max(right.get(f, (-np.inf, np.inf))[0], thr), right.get(f, (-np.inf, np.inf))[1])
        recurse(t.children_right[node], right)

    recurse(0, {})
    return {lid: _fmt_rule(bounds) for lid, bounds in rules.items()}


def _fmt_rule(bounds: dict) -> str:
    parts = []
    for f, (lo, hi) in bounds.items():
        if lo == -np.inf:
            parts.append(f"{f}≤{hi:.3g}")
        elif hi == np.inf:
            parts.append(f"{f}>{lo:.3g}")
        else:
            parts.append(f"{lo:.3g}<{f}≤{hi:.3g}")
    return " & ".join(parts) if parts else "all"


def _parse_rule(rule: str) -> list:
    """Parse formatted rule back into (feat, lo, hi) constraints for matching."""
    out = []
    if rule == "all":
        return out
    for clause in rule.split(" & "):
        if "<" in clause and "≤" in clause and clause.count("≤") == 1 and "<" in clause.split("≤")[0]:
            lo_s, rest = clause.split("<", 1)
            feat, hi_s = rest.split("≤")
            out.append((feat, float(lo_s), float(hi_s)))
        elif "≤" in clause:
            feat, hi_s = clause.split("≤")
            out.append((feat, -np.inf, float(hi_s)))
        elif ">" in clause:
            feat, lo_s = clause.split(">")
            out.append((feat, float(lo_s), np.inf))
    return out


def _row_matches(rule: str, row: dict) -> bool:
    for feat, lo, hi in _parse_rule(rule):
        v = row.get(feat, np.nan)
        if not (lo < v <= hi):
            return False
    return True


def _tint(hex_color: str) -> str:
    """Light background tint of a cause color."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r, g, b = [int(c + (255 - c) * 0.72) for c in (r, g, b)]
    return f"rgb({r},{g},{b})"
