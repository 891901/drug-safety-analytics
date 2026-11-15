"""Big Data-Driven Drug Safety Intelligence Platform - Streamlit dashboard.

Run with::

    python -m streamlit run dashboard/app.py

The dashboard is a thin navigation shell: each sidebar entry renders one of the
ten sections in :mod:`dashboard.sections`, and every figure/table is read from
the project's *real* analytical outputs through :mod:`dashboard.components.data`.
Nothing is recomputed or synthesised here.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable regardless of where Streamlit is launched
# from (``streamlit run`` puts the script's own directory on ``sys.path``).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Callable, Dict

import streamlit as st

from dashboard.components import data as ds
from dashboard.components import ui
from dashboard.sections import SECTION_MODULES

APP_TITLE = "Big Data-Driven Drug Safety Intelligence Platform"
APP_SUBTITLE = (
    "Interactive analytics over the FDA FAERS 2026 Q1/Q2 spontaneous reporting "
    "extracts - adverse reactions, disproportionality signals, machine-learning "
    "anomalies, drug co-occurrence and network structure."
)


def _render_map() -> Dict[str, Callable[[], None]]:
    return {label: render for label, render in SECTION_MODULES}


def _sidebar() -> str:
    """Draw the sidebar and return the selected section label."""
    st.sidebar.title(APP_TITLE)
    st.sidebar.caption(APP_SUBTITLE)

    labels = [label for label, _ in SECTION_MODULES]
    choice = st.sidebar.radio("Section", options=labels, index=0, key="nav_section")

    st.sidebar.divider()
    present = sum(1 for item in ds.output_status() if item["exists"])
    total = len(ds.output_status())
    st.sidebar.caption(f"Data outputs available: {present}/{total}")
    ui.sidebar_availability()

    st.sidebar.divider()
    with st.sidebar.expander("About / disclaimer"):
        st.write(ui.DISCLAIMER)
    return choice


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="\U0001f48a",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    choice = _sidebar()
    section = _render_map().get(choice)

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    ui.disclaimer()
    st.divider()

    if section is None:
        st.error(f"Unknown section: {choice}")
        return

    try:
        section()
    except Exception as exc:  # noqa: BLE001 - keep the app alive, show the error
        st.error(
            f"This section failed to render: {type(exc).__name__}: {exc}. "
            "Other sections remain available from the sidebar."
        )
        with st.expander("Error details"):
            st.exception(exc)

    ui.footer()


if __name__ == "__main__":
    main()
