"""One-call orchestrator that runs all three analyzers and writes a single
self-contained interactive HTML report (plotly embedded)."""
from __future__ import annotations

from typing import Optional

import plotly.io as pio

from .data import Dataset, EvalConfig
from .complexity import ComplexityAnalyzer
from .learning_curve import LearningCurveAnalyzer
from .weakpoint import WeakpointAnalyzer


class OverfittingReport:
    def __init__(self, config: Optional[EvalConfig] = None):
        self.cfg = config or EvalConfig()

    def run(self, model, ds: Dataset) -> "OverfittingReport":
        self.complexity = ComplexityAnalyzer(self.cfg).analyze(model, ds)
        lca = LearningCurveAnalyzer(self.cfg)
        self.learning = lca.random_curve(model, ds)
        self.temporal = None
        if ds.time_train is not None:
            try:
                self.temporal = lca.temporal_curve(model, ds)
            except Exception as e:  # pragma: no cover
                self.temporal = None
                self._temporal_err = str(e)
        self.weak = WeakpointAnalyzer(self.cfg).analyze(model, ds)
        self.model_ds = (model, ds)
        return self

    # ------------------------------------------------------------------ #
    def to_html(self, path: str = "overfitting_report.html") -> str:
        blocks = []

        def section(title, verdict, figs):
            html = [f'<h2>{title}</h2>', f'<div class="verdict">{verdict}</div>']
            for f in figs:
                html.append(pio.to_html(f, include_plotlyjs=False, full_html=False))
            return "\n".join(html)

        cfigs = self.complexity.figures()
        blocks.append(section("1 · Complexity", self.complexity.verdict,
                              [cfigs["radar"], cfigs["table"]]))
        blocks.append(section("2 · Learning curve — saturation", self.learning.verdict,
                              [self.learning.fig, self.learning.table_fig()
                               if hasattr(self.learning, "table_fig") else _tbl(self.learning)]))
        if self.temporal is not None:
            blocks.append(section("2b · Learning curve — temporal / drift",
                                  self.temporal.verdict,
                                  [self.temporal.fig, _tbl(self.temporal)]))
        wfigs = self.weak.figures()
        blocks.append(section("3 · Weak points", self.weak.verdict,
                              [wfigs["table"], wfigs["map"], wfigs["profile"]]))

        html = _TEMPLATE.format(
            plotlyjs=pio.to_html(cfigs["radar"], include_plotlyjs="cdn",
                                 full_html=False).split("<div")[0],
            body="\n<hr/>\n".join(blocks))
        with open(path, "w") as fh:
            fh.write(html)
        return path


def _tbl(report):
    from . import viz
    return viz.table(report.table, "Details")


_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Overfitting report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{{font-family:Inter,Segoe UI,sans-serif;color:#1f2733;max-width:1050px;
      margin:24px auto;padding:0 18px;background:#fbfcfe}}
 h1{{font-size:24px}} h2{{margin-top:8px;font-size:19px}}
 .verdict{{background:#f2f5fa;border-left:4px solid #2f6df6;padding:10px 14px;
          border-radius:6px;margin:8px 0 14px;font-size:14px;line-height:1.5}}
 hr{{border:none;border-top:1px solid #e9edf3;margin:26px 0}}
</style></head><body>
<h1>Model evaluation — overfitting report</h1>
{body}
</body></html>"""
