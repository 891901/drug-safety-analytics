"""Reusable Streamlit UI building blocks for the dashboard.

Everything here is presentation-only: headers, metric cards, disclaimers,
tables, download buttons and "output missing" notices.  No data is invented -
when a source is absent the helpers render a clear message.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, Iterable, List, Optional, Sequence

import streamlit as st

from dashboard.components import data as ds

DISCLAIMER = (
    "FAERS reports describe spontaneous reporting patterns. They do not "
    "establish causation, incidence, clinical risk or confirmed drug interactions."
)

# ---------------------------------------------------------------------------
# Headers / framing
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "") -> None:
    """Render the section title and optional subtitle."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def section_intro(text: str) -> None:
    """Render a short explanatory paragraph."""
    st.markdown(text)


def disclaimer(expanded: bool = False) -> None:
    """Render the platform-wide spontaneous-reporting disclaimer."""
    st.warning(DISCLAIMER, icon="\u26a0\ufe0f")


def missing_output(label: str, key: str) -> None:
    """Clear notice shown when a required analytical output is absent."""
    path = ds.source_path(key)
    st.info(
        f"**{label}** is not available yet. Expected file:\n\n`{path}`\n\n"
        "Run the corresponding analysis pipeline to generate it. "
        "No placeholder or synthetic data is shown.",
        icon="\u2139\ufe0f",
    )


def empty_notice(message: str) -> None:
    """Informational notice for an empty (but present) result."""
    st.info(message, icon="\u2139\ufe0f")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metric_row(items: Sequence[Dict[str, Any]]) -> None:
    """Render a row of metric cards.

    Each item is ``{"label": str, "value": str, "help": str (optional),
    "delta": str (optional)}``.
    """
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            st.metric(
                label=item.get("label", ""),
                value=item.get("value", "n/a"),
                delta=item.get("delta"),
                help=item.get("help"),
            )


def stat_caption(text: str) -> None:
    """Small caption under a metric group (provenance / caveats)."""
    st.caption(text)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _html_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    """Minimal, escaped HTML table used as a fallback renderer."""
    head = "".join(f"<th>{_escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_escape(row.get(col, ''))}</td>" for col in columns)
        body.append(f"<tr>{cells}</tr>")
    return (
        "<div style='max-height:420px;overflow:auto'>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.85rem'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_table(
    rows: Sequence[Dict[str, Any]],
    *,
    columns: Optional[Sequence[str]] = None,
    height: int = 380,
    caption: str = "",
) -> None:
    """Render rows as an interactive table, falling back to plain HTML.

    Uses ``st.dataframe`` when the environment supports it (it needs the
    dataframe engine) and degrades to a static HTML table otherwise.
    """
    if not rows:
        empty_notice("No rows to display.")
        return
    cols = list(columns) if columns else list(rows[0].keys())
    if caption:
        st.caption(caption)
    try:
        st.dataframe(
            rows,
            column_order=cols,
            width="stretch",
            height=height,
            hide_index=True,
        )
    except Exception:  # noqa: BLE001 - fall back to a dependency-free renderer
        st.markdown(_html_table(rows, cols), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def rows_to_csv_bytes(rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> bytes:
    """Serialise rows to CSV bytes with the standard library (no pandas)."""
    if not rows:
        return b""
    cols = list(columns) if columns else list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in cols})
    return buffer.getvalue().encode("utf-8")


def download_rows(
    rows: Sequence[Dict[str, Any]],
    filename: str,
    *,
    label: str = "Download CSV",
    columns: Optional[Sequence[str]] = None,
    key: Optional[str] = None,
) -> None:
    """Render a CSV download button for ``rows``."""
    payload = rows_to_csv_bytes(rows, columns)
    if not payload:
        st.button(label, disabled=True, key=key, help="Nothing to download.")
        return
    st.download_button(
        label=label,
        data=payload,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def download_file(path_key: str, *, label: str = "Download file") -> None:
    """Offer a registered file (CSV/JSON/HTML) as a download."""
    path = ds.source_path(path_key)
    if not path or not path.is_file():
        st.button(label, disabled=True, key=f"dl_missing_{path_key}")
        return
    data = path.read_bytes()
    mime = "text/html" if path.suffix == ".html" else "application/octet-stream"
    st.download_button(
        label=label,
        data=data,
        file_name=path.name,
        mime=mime,
        key=f"dl_{path_key}",
    )


# ---------------------------------------------------------------------------
# Sidebar / global widgets
# ---------------------------------------------------------------------------
def sidebar_availability() -> None:
    """Render the output-availability panel in the sidebar."""
    status = ds.output_status()
    present = sum(1 for item in status if item["exists"])
    with st.sidebar.expander(f"Data outputs available ({present}/{len(status)})", expanded=False):
        for item in status:
            icon = "\u2705" if item["exists"] else "\u274c"
            st.markdown(f"{icon} {item['label']}")
        st.caption(f"Analytics directory: `{ds.ANALYTICS_DIR}`")


def footer() -> None:
    """Render the dashboard footer."""
    st.divider()
    st.caption(
        "Big Data-Driven Drug Safety Intelligence Platform \u00b7 "
        "all figures read from the project's real analytical outputs \u00b7 "
        "no synthetic data."
    )
