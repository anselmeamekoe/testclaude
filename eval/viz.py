"""Shared plotly helpers + a consistent theme. Everything the toolkit draws
comes through here, so the look is uniform and easy to restyle."""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

# One place to control the palette.
INK = "#1f2733"
MUTED = "#8a94a6"
GRID = "#e9edf3"
ACCENT = "#2f6df6"
GOOD = "#2fa36b"
WARN = "#e8a13a"
BAD = "#d1495b"
SEQ = ["#2f6df6", "#8a4fff", "#2fa36b", "#e8a13a", "#d1495b", "#00b3b3"]


def _base(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK)),
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=12),
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.6)", borderwidth=0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def table(df: pd.DataFrame, title: str, color_col: Optional[str] = None,
          height: Optional[int] = None) -> go.Figure:
    """Render a DataFrame as an interactive plotly table."""
    fill = "white"
    if color_col and color_col in df.columns:
        vals = df[color_col]
        fill = [[c] * len(df) if col != color_col else list(vals) for col in df.columns]
    header_vals = [f"<b>{c}</b>" for c in df.columns]
    cells = [df[c].tolist() for c in df.columns]
    fig = go.Figure(go.Table(
        header=dict(values=header_vals, fill_color="#f2f5fa",
                    align="left", font=dict(color=INK, size=12), height=30),
        cells=dict(values=cells, align="left",
                   fill_color=[fill] if isinstance(fill, str) else fill,
                   font=dict(color=INK, size=11), height=26),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=INK)),
                      margin=dict(l=10, r=10, t=50, b=10),
                      height=height or (60 + 28 * (len(df) + 1)))
    return fig


def bars(labels: Sequence[str], values: Sequence[float], title: str,
         colors: Optional[Sequence[str]] = None, ytitle: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values),
        marker_color=colors or ACCENT,
        text=[f"{v:.3g}" for v in values], textposition="outside",
    ))
    fig.update_yaxes(title=ytitle)
    return _base(fig, title, height=380)


def radar(labels: Sequence[str], values: Sequence[float], title: str) -> go.Figure:
    labels = list(labels) + [labels[0]]
    values = list(values) + [values[0]]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself",
                                    line_color=ACCENT))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK)),
        polar=dict(radialaxis=dict(range=[0, 1], gridcolor=GRID)),
        template="plotly_white", height=440,
        margin=dict(l=60, r=60, t=60, b=40),
    )
    return fig
