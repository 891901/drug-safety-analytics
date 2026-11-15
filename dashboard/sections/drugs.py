"""Section 2 - Drug Analytics.

Top reported drugs and their reporting frequency, from
``adr_drug_report_counts.csv`` and the ADR summary.  An interactive drug
selector drills into one drug's reported reactions.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui

_COLUMNS = ["drug_name", "reports", "pct_of_reports"]


def render() -> None:
    ui.page_header(
        "Drug Analytics",
        "Most frequently reported drugs in the FAERS 2026 Q1/Q2 extracts, with an "
        "interactive drill-down.",
    )
    ui.disclaimer()

    if not ds.is_available("adr_drug_report_counts"):
        ui.missing_output("Drug report counts", "adr_drug_report_counts")
        return

    rows = ds.get_rows("adr_drug_report_counts", columns=_COLUMNS)
    if not rows:
        ui.empty_notice("The drug report-count output is present but empty.")
        return

    total_rows = len(rows)
    top_n = st.slider(
        "Number of drugs to display",
        min_value=5,
        max_value=max(5, min(total_rows, 100)),
        value=min(ds.DEFAULT_TOP_N, total_rows),
        step=5,
    )
    selected = rows[:top_n]

    labels = [r.get("drug_name", "") for r in selected]
    values = [ds.as_int(r.get("reports")) for r in selected]

    st.plotly_chart(
        ch.horizontal_bar(
            labels,
            values,
            title=f"Top {top_n} reported drugs",
            x_title="Reports",
            colorscale=True,
        ),
        width="stretch",
    )

    left, right = st.columns([2, 3])
    with left:
        ui.stat_caption(
            f"{total_rows} drugs ranked in the output. Percentages are shares of "
            "all reports in the extract."
        )
        ui.download_rows(
            selected,
            "drug_report_counts.csv",
            label="Download drug report counts",
            key="dl_drug_counts",
        )
    with right:
        ui.render_table(
            selected,
            columns=["drug_name", "reports", "pct_of_reports"],
            height=360,
            caption="Reporting frequency",
        )

    st.divider()

    # --------------------------------------------------------- drug drill-down
    st.markdown("#### Interactive drug selection")
    drugs = [r.get("drug_name", "") for r in rows if r.get("drug_name")]
    if not drugs:
        ui.empty_notice("No drug names available for selection.")
        return

    chosen = st.selectbox("Select a drug", options=drugs, index=0)

    chosen_row = next((r for r in rows if r.get("drug_name") == chosen), {})
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Reports", ds.fmt_int(chosen_row.get("reports")))
    with col_b:
        st.metric("Share of reports", f"{ds.fmt_num(chosen_row.get('pct_of_reports'), 3)} %")
    with col_c:
        rank = next((i + 1 for i, r in enumerate(rows) if r.get("drug_name") == chosen), None)
        st.metric("Rank", f"#{rank}" if rank else "n/a")

    if ds.is_available("adr_drug_reaction_pairs"):
        pairs = ds.filter_rows(
            "adr_drug_reaction_pairs", ["drug_name"], chosen, limit=500
        )
        if pairs:
            pairs_sorted = sorted(
                pairs, key=lambda r: ds.as_int(r.get("reports")), reverse=True
            )[:20]
            st.plotly_chart(
                ch.horizontal_bar(
                    [r.get("reaction_pt", "") for r in pairs_sorted],
                    [ds.as_int(r.get("reports")) for r in pairs_sorted],
                    title=f"Reported reactions for {chosen}",
                    x_title="Reports",
                    color=ch.PALETTE[1],
                ),
                width="stretch",
            )
            st.caption(
                "Part of the top-100 drug-reaction pair ranking only. For the full "
                "signal view and confidence intervals see 'Statistical Signal Detection'."
            )
            ui.download_rows(
                pairs_sorted,
                f"drug_reactions_{chosen}.csv",
                label="Download these reaction pairs",
                key="dl_drug_pairs",
            )
        else:
            ui.empty_notice(
                f"{chosen} does not appear in the top-100 drug-reaction pair output."
            )
    else:
        ui.missing_output("Drug-reaction pairs", "adr_drug_reaction_pairs")
