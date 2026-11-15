"""Reusable Plotly figure builders for the dashboard.

Every builder takes plain Python sequences (lists/dicts) - never a pandas
DataFrame - and returns a ``plotly.graph_objects.Figure``.  The figures carry no
Streamlit dependency so they can be unit-tested and reused (e.g. exported to
HTML).  If a builder is handed no data it returns a figure with an explicit
"no data" annotation rather than an empty canvas.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import plotly.graph_objects as go

from dashboard.components.data import THEME

# A calm, colour-blind-friendly palette reused across sections.
PALETTE: List[str] = [
    "#2E5AAC",
    "#E4572E",
    "#17BEBB",
    "#FFC914",
    "#76B041",
    "#9B5DE5",
    "#F15BB5",
    "#00BBF9",
]
QUARTER_COLORS: Dict[str, str] = {"2026_Q1": "#2E5AAC", "2026_Q2": "#E4572E"}
SEQUENTIAL = "Blues"

_NPREFIX = 0.0


def _empty(message: str = "No data available for this view.") -> go.Figure:
    """A figure that explains the absence of data instead of showing nothing."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=14, color="#666666"),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=THEME, height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _finish(
    fig: go.Figure,
    *,
    title: str,
    height: int = 420,
    legend: bool = True,
    x_title: str = "",
    y_title: str = "",
    margin_left: int = 10,
) -> go.Figure:
    """Apply the shared theme/layout to a finished figure."""
    fig.update_layout(
        template=THEME,
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15)),
        height=height,
        showlegend=legend,
        margin=dict(l=margin_left, r=20, t=52, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------
def horizontal_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    x_title: str = "Reports",
    color: Optional[str] = None,
    colorscale: bool = False,
    text: Optional[Sequence[str]] = None,
    height: Optional[int] = None,
    hover_suffix: str = "reports",
) -> go.Figure:
    """Horizontal bar chart; largest value on top."""
    if not labels or not values:
        return _empty()
    pairs = list(zip(labels, values))
    pairs.reverse()  # Plotly draws bottom-up
    rev_labels = [p[0] for p in pairs]
    rev_values = [p[1] for p in pairs]
    marker = dict(
        color=rev_values if colorscale else (color or PALETTE[0]),
        colorscale=SEQUENTIAL if colorscale else None,
        showscale=False,
    )
    rev_text = list(reversed(list(text))) if text else None
    fig = go.Figure(
        go.Bar(
            x=rev_values,
            y=rev_labels,
            orientation="h",
            marker=marker,
            text=rev_text,
            textposition="outside" if rev_text else "none",
            hovertemplate="%{y}<br>%{x:,} " + hover_suffix + "<extra></extra>",
        )
    )
    height = height or max(320, 22 * len(labels) + 120)
    fig = _finish(
        fig,
        title=title,
        height=height,
        legend=False,
        x_title=x_title,
        margin_left=10,
    )
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    fig.update_xaxes(automargin=True)
    return fig


def vertical_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    y_title: str = "Reports",
    x_title: str = "",
    color: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    text: Optional[Sequence[str]] = None,
    height: int = 420,
) -> go.Figure:
    """Vertical bar chart."""
    if not labels or not values:
        return _empty()
    marker = dict(color=list(colors) if colors else (color or PALETTE[0]))
    fig = go.Figure(
        go.Bar(
            x=list(labels),
            y=list(values),
            marker=marker,
            text=list(text) if text else None,
            textposition="outside" if text else "none",
            hovertemplate="%{x}<br>%{y:,}<extra></extra>",
        )
    )
    return _finish(fig, title=title, height=height, legend=False, x_title=x_title, y_title=y_title)


def grouped_bar(
    categories: Sequence[str],
    series: Dict[str, Sequence[float]],
    *,
    title: str,
    y_title: str = "Reports",
    colors: Optional[Dict[str, str]] = None,
    barmode: str = "group",
    height: int = 440,
    y_log: bool = False,
) -> go.Figure:
    """Grouped (or stacked) bar chart with one trace per series."""
    if not categories or not series:
        return _empty()
    fig = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        color = (colors or {}).get(name) or PALETTE[index % len(PALETTE)]
        fig.add_trace(
            go.Bar(
                x=list(categories),
                y=list(values),
                name=str(name),
                marker_color=color,
                hovertemplate="%{x}<br>" + str(name) + ": %{y:,}<extra></extra>",
            )
        )
    fig.update_layout(barmode=barmode)
    if y_log:
        fig.update_yaxes(type="log")
    return _finish(fig, title=title, height=height, y_title=y_title)


# ---------------------------------------------------------------------------
# Line / trend charts
# ---------------------------------------------------------------------------
def line_chart(
    x: Sequence,
    y: Sequence[float],
    *,
    title: str,
    x_title: str = "",
    y_title: str = "Reports",
    name: str = "Reports",
    color: Optional[str] = None,
    markers: bool = True,
    height: int = 420,
) -> go.Figure:
    """Single-series line chart (used for monthly trends)."""
    if not x or not y:
        return _empty()
    fig = go.Figure(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines+markers" if markers else "lines",
            name=name,
            line=dict(color=color or PALETTE[0], width=2),
            marker=dict(size=5),
            hovertemplate="%{x}<br>%{y:,}<extra></extra>",
        )
    )
    return _finish(fig, title=title, height=height, legend=False, x_title=x_title, y_title=y_title)


def multi_line(
    x: Sequence,
    series: Dict[str, Sequence[float]],
    *,
    title: str,
    x_title: str = "",
    y_title: str = "Reports",
    height: int = 440,
) -> go.Figure:
    """Multi-series line chart with one trace per series."""
    if not x or not series:
        return _empty()
    fig = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=list(x),
                y=list(values),
                mode="lines+markers",
                name=str(name),
                line=dict(color=PALETTE[index % len(PALETTE)], width=2),
                marker=dict(size=5),
                hovertemplate="%{x}<br>" + str(name) + ": %{y:,}<extra></extra>",
            )
        )
    return _finish(fig, title=title, height=height, x_title=x_title, y_title=y_title)


def diverging_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    x_title: str = "",
    height: Optional[int] = None,
    hover_suffix: str = "",
) -> go.Figure:
    """Horizontal bar chart coloured by sign (positive/negative)."""
    if not labels or not values:
        return _empty()
    pairs = list(zip(labels, values))
    pairs.reverse()
    rev_labels = [p[0] for p in pairs]
    rev_values = [p[1] for p in pairs]
    colors = [PALETTE[4] if v >= 0 else PALETTE[1] for v in rev_values]
    fig = go.Figure(
        go.Bar(
            x=rev_values,
            y=rev_labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>%{x:,}" + hover_suffix + "<extra></extra>",
        )
    )
    height = height or max(320, 24 * len(labels) + 120)
    fig = _finish(fig, title=title, height=height, legend=False, x_title=x_title)
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    fig.add_vline(x=0, line_width=1, line_color="#888888")
    return fig


# ---------------------------------------------------------------------------
# Distribution / part-to-whole
# ---------------------------------------------------------------------------
def histogram(
    values: Sequence[float],
    *,
    title: str,
    x_title: str = "",
    y_title: str = "Count",
    bins: int = 50,
    color: Optional[str] = None,
    height: int = 420,
) -> go.Figure:
    """Histogram of a numeric column."""
    if not values:
        return _empty()
    fig = go.Figure(
        go.Histogram(
            x=list(values),
            nbinsx=bins,
            marker_color=color or PALETTE[0],
            hovertemplate="bin %{x}<br>count %{y:,}<extra></extra>",
        )
    )
    return _finish(fig, title=title, height=height, legend=False, x_title=x_title, y_title=y_title)


def pie_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    height: int = 420,
    hole: float = 0.0,
    colors: Optional[Sequence[str]] = None,
) -> go.Figure:
    """Pie / donut chart."""
    if not labels or not values:
        return _empty()
    fig = go.Figure(
        go.Pie(
            labels=list(labels),
            values=list(values),
            hole=hole,
            marker=dict(colors=list(colors) if colors else PALETTE),
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,} reports (%{percent})<extra></extra>",
        )
    )
    return _finish(fig, title=title, height=height)


# ---------------------------------------------------------------------------
# Scatter / interval
# ---------------------------------------------------------------------------
def scatter(
    x: Sequence[float],
    y: Sequence[float],
    *,
    title: str,
    x_title: str = "",
    y_title: str = "",
    color: Optional[Sequence[float]] = None,
    colorbar_title: str = "",
    hover_text: Optional[Sequence[str]] = None,
    log_x: bool = False,
    log_y: bool = False,
    height: int = 460,
    marker_size: int = 7,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Scatter plot with an optional colour dimension and hover text."""
    if not x or not y:
        return _empty()
    marker = dict(size=marker_size, line=dict(width=0.4, color="#555555"))
    if color is not None:
        marker["color"] = list(color)
        marker["colorscale"] = colorscale
        marker["showscale"] = True
        marker["colorbar"] = dict(title=colorbar_title or None)
    else:
        marker["color"] = PALETTE[0]
    fig = go.Figure(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="markers",
            marker=marker,
            text=list(hover_text) if hover_text else None,
            hovertemplate="%{text}<br>(%{x}, %{y})<extra></extra>"
            if hover_text
            else "x=%{x}<br>y=%{y}<extra></extra>",
        )
    )
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    return _finish(fig, title=title, height=height, legend=False, x_title=x_title, y_title=y_title)


def error_bar_scatter(
    x: Sequence[str],
    y: Sequence[float],
    err_low: Sequence[float],
    err_high: Sequence[float],
    *,
    title: str,
    x_title: str = "Signal rank",
    y_title: str = "PRR (95% CI)",
    log_y: bool = True,
    height: int = 480,
) -> go.Figure:
    """Point estimates with asymmetric error bars (confidence intervals)."""
    if not x or not y:
        return _empty()
    lower = [max(0.0, y[i] - err_low[i]) for i in range(len(y))]
    upper = [max(0.0, err_high[i] - y[i]) for i in range(len(y))]
    fig = go.Figure(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="markers",
            marker=dict(color=PALETTE[0], size=6),
            error_y=dict(
                type="data",
                symmetric=False,
                array=upper,
                arrayminus=lower,
                color="#888888",
                thickness=1,
            ),
            hovertemplate="%{x}<br>value %{y:.3g}<extra></extra>",
        )
    )
    if log_y:
        fig.update_yaxes(type="log")
    return _finish(fig, title=title, height=height, legend=False, x_title=x_title, y_title=y_title)


def network_graph(
    nodes: Sequence[Dict[str, object]],
    edges: Sequence[Dict[str, object]],
    *,
    title: str,
    height: int = 640,
) -> go.Figure:
    """Drug relationship network.

    ``nodes`` items need ``x, y, label, size, color`` and optional ``hover``.
    ``edges`` items need ``x0, y0, x1, y1`` and optional ``hover``/``width``.
    Edge segments are drawn with ``None`` separators so a whole edge list is one
    trace (fast and light).
    """
    if not nodes:
        return _empty("Network layout unavailable.")
    fig = go.Figure()
    if edges:
        ex: List[Optional[float]] = []
        ey: List[Optional[float]] = []
        for e in edges:
            ex += [e.get("x0"), e.get("x1"), None]
            ey += [e.get("y0"), e.get("y1"), None]
        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(width=0.7, color="#B8B8B8"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[n.get("x") for n in nodes],
            y=[n.get("y") for n in nodes],
            mode="markers+text",
            text=[n.get("label") for n in nodes],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=[n.get("size", 12) for n in nodes],
                color=[n.get("color", PALETTE[0]) for n in nodes],
                line=dict(width=0.6, color="#444444"),
            ),
            hovertext=[n.get("hover", n.get("label")) for n in nodes],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _finish(fig, title=title, height=height, legend=False)


def heatmap(
    z: Sequence[Sequence[float]],
    x: Sequence[str],
    y: Sequence[str],
    *,
    title: str,
    x_title: str = "",
    y_title: str = "",
    colorscale: str = "Blues",
    height: int = 460,
    text: Optional[Sequence[Sequence[str]]] = None,
) -> go.Figure:
    """Heatmap (used for the age x gender reporting matrix)."""
    if not z:
        return _empty()
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(x),
            y=list(y),
            colorscale=colorscale,
            text=text,
            texttemplate="%{text}" if text else None,
            hovertemplate="%{y} / %{x}<br>%{z:,}<extra></extra>",
            colorbar=dict(title="Reports"),
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title=title, height=height, legend=False, x_title=x_title, y_title=y_title)
