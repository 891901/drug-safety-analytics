"""Section 8 - Temporal Analysis.

Monthly reporting trend, quarterly volume comparison and drug reporting changes
between 2026 Q1 and Q2.  Months come straight from the report dates in the
data; nothing is interpolated.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui


def render() -> None:
    ui.page_header(
        "Temporal Analysis",
        "Reporting volume over time. These are counts of received reports (which "
        "can span several years of event dates), not incidence rates.",
    )
    ui.disclaimer()

    summary = ds.get_json("temporal_summary") if ds.is_available("temporal_summary") else None
    temporal_block = (summary or {}).get("temporal", {}) if summary else {}

    # -------------------------------------------------------------- monthly
    st.markdown("#### Monthly reporting trend")
    if not ds.is_available("temporal_monthly_trend"):
        ui.missing_output("Monthly trend", "temporal_monthly_trend")
    else:
        trend = ds.get_rows("temporal_monthly_trend", columns=["report_month", "reports"])
        if not trend:
            ui.empty_notice("No monthly trend rows available.")
        else:
            trend = sorted(trend, key=lambda r: r.get("report_month", ""))
            months = [r.get("report_month", "") for r in trend]
            counts = [ds.as_int(r.get("reports")) for r in trend]
            st.plotly_chart(
                ch.line_chart(
                    months,
                    counts,
                    title="Reports by month (from report dates)",
                    x_title="Report month",
                    y_title="Reports",
                    height=420,
                ),
                width="stretch",
            )
            if temporal_block:
                st.caption(
                    f"{len(months)} month(s) present, range "
                    f"{temporal_block.get('date_range', ['?', '?'])[0]} to "
                    f"{temporal_block.get('date_range', ['?', '?'])[-1]} · "
                    f"{ds.fmt_int(temporal_block.get('reports_with_report_date'))} reports "
                    f"dated, {ds.fmt_int(temporal_block.get('reports_missing_date'))} undated."
                )
            ui.download_rows(
                trend, "monthly_trend.csv", label="Download monthly trend", key="dl_monthly"
            )

    st.divider()
    st.markdown("#### Quarterly comparison")
    if not ds.is_available("temporal_reports_by_quarter"):
        ui.missing_output("Reports by quarter", "temporal_reports_by_quarter")
    else:
        q_rows = ds.get_rows(
            "temporal_reports_by_quarter",
            columns=["quarter", "reports", "reaction_pairs", "reports_with_reaction", "serious_reports"],
        )
        if not q_rows:
            ui.empty_notice("No quarterly rows available.")
        else:
            q_rows = sorted(q_rows, key=lambda r: r.get("quarter", ""))
            quarters = [r.get("quarter", "") for r in q_rows]
            st.plotly_chart(
                ch.grouped_bar(
                    quarters,
                    {
                        "Reports": [ds.as_int(r.get("reports")) for r in q_rows],
                        "Report-reaction pairs": [
                            ds.as_int(r.get("reaction_pairs")) for r in q_rows
                        ],
                        "Serious reports": [
                            ds.as_int(r.get("serious_reports")) for r in q_rows
                        ],
                    },
                    title="Reports, reaction pairs and serious reports by quarter",
                    y_title="Count",
                    colors=ch.QUARTER_COLORS,
                    height=440,
                ),
                width="stretch",
            )
            ui.render_table(q_rows, height=180)
            ui.download_rows(
                q_rows, "reports_by_quarter.csv", label="Download quarterly summary", key="dl_q"
            )

    st.divider()
    st.markdown("#### Drug reporting changes (Q1 to Q2)")
    if not ds.is_available("temporal_drug_frequency_change"):
        ui.missing_output("Drug frequency change", "temporal_drug_frequency_change")
    else:
        ch_rows = ds.get_rows(
            "temporal_drug_frequency_change",
            columns=["drug_name", "2026_Q1_reports", "2026_Q2_reports", "delta", "pct_change"],
        )
        if not ch_rows:
            ui.empty_notice("No drug frequency-change rows available.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    ch.grouped_bar(
                        [r.get("drug_name", "") for r in ch_rows],
                        {
                            "2026_Q1": [ds.as_int(r.get("2026_Q1_reports")) for r in ch_rows],
                            "2026_Q2": [ds.as_int(r.get("2026_Q2_reports")) for r in ch_rows],
                        },
                        title="Reports per drug, Q1 vs Q2",
                        y_title="Reports",
                        colors=ch.QUARTER_COLORS,
                        height=460,
                    ),
                    width="stretch",
                )
            with col2:
                by_delta = sorted(
                    ch_rows, key=lambda r: ds.as_float(r.get("delta"), 0.0), reverse=True
                )
                st.plotly_chart(
                    ch.diverging_bar(
                        [r.get("drug_name", "") for r in by_delta],
                        [ds.as_float(r.get("delta"), 0.0) for r in by_delta],
                        title="Change in reports (Q2 - Q1)",
                        x_title="Change in reports",
                        height=460,
                        hover_suffix=" reports",
                    ),
                    width="stretch",
                )
            ui.render_table(
                ch_rows,
                height=320,
                caption="Drugs whose report counts moved the most between the quarters",
            )
            ui.download_rows(
                ch_rows, "drug_frequency_change.csv", label="Download drug changes", key="dl_change"
            )

    st.divider()
    st.markdown("#### Top reactions by quarter")
    if not ds.is_available("temporal_top_reactions_by_quarter"):
        ui.missing_output("Reactions by quarter", "temporal_top_reactions_by_quarter")
    else:
        tr = ds.get_rows("temporal_top_reactions_by_quarter", columns=["quarter", "reaction_pt", "reports"])
        if not tr:
            ui.empty_notice("No per-quarter reaction rows available.")
        else:
            top_n = st.slider("Reactions per quarter", 5, 20, 10, 1, key="temp_reac_n")
            quarters = [q for q in ds.QUARTERS if any(r.get("quarter") == q for r in tr)]
            for q in quarters:
                q_view = sorted(
                    [r for r in tr if r.get("quarter") == q],
                    key=lambda r: ds.as_int(r.get("reports")),
                    reverse=True,
                )[:top_n]
                st.plotly_chart(
                    ch.horizontal_bar(
                        [r.get("reaction_pt", "") for r in q_view],
                        [ds.as_int(r.get("reports")) for r in q_view],
                        title=f"Top reactions in {q}",
                        x_title="Reports",
                        color=ch.QUARTER_COLORS.get(q, ch.PALETTE[0]),
                        height=max(320, 24 * len(q_view) + 120),
                    ),
                    width="stretch",
                )
            ui.download_rows(
                tr, "reactions_by_quarter.csv", label="Download reactions by quarter", key="dl_tr"
            )
