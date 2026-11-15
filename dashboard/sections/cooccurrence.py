"""Section 4 - Drug Co-Occurrence.

Frequently co-reported drug pairs with their support, confidence and lift, plus
an interactive pair explorer and the mined co-reported groups.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui

_PAIR_COLUMNS = [
    "drug_a",
    "drug_b",
    "reports",
    "support",
    "confidence_a_to_b",
    "confidence_b_to_a",
    "lift",
    "reports_a",
    "reports_b",
    "passes_thresholds",
]


def _pair_label(row: dict) -> str:
    return f"{row.get('drug_a', '')} + {row.get('drug_b', '')}"


def render() -> None:
    ui.page_header(
        "Drug Co-Occurrence",
        "Drugs reported together in the same FAERS report, with association "
        "metrics. Co-reporting is not evidence of an interaction.",
    )
    ui.disclaimer()

    if not ds.is_available("cooccurrence_top_pairs"):
        ui.missing_output("Co-occurrence pairs", "cooccurrence_top_pairs")
        return

    pairs = ds.get_rows("cooccurrence_top_pairs", columns=_PAIR_COLUMNS)
    if not pairs:
        ui.empty_notice("The co-occurrence pair output is present but empty.")
        return

    only_qualifying = st.checkbox(
        "Show only pairs passing the configured thresholds", value=False
    )
    view = (
        [p for p in pairs if ds.as_bool(p.get("passes_thresholds"))]
        if only_qualifying
        else pairs
    )
    if not view:
        ui.empty_notice("No pairs match the current filter.")
        return

    top_n = st.slider(
        "Pairs to display",
        min_value=5,
        max_value=max(5, min(len(view), 50)),
        value=min(20, len(view)),
        step=5,
    )
    top = sorted(view, key=lambda r: ds.as_int(r.get("reports")), reverse=True)[:top_n]

    st.plotly_chart(
        ch.horizontal_bar(
            [_pair_label(r) for r in top],
            [ds.as_int(r.get("reports")) for r in top],
            title=f"Top {len(top)} most frequently co-reported drug pairs",
            x_title="Reports containing both drugs",
            color=ch.PALETTE[0],
            hover_suffix="co-reports",
        ),
        width="stretch",
    )

    st.divider()
    st.markdown("#### Support, confidence and lift")
    left, right = st.columns([3, 2])
    with left:
        ui.render_table(
            top,
            columns=_PAIR_COLUMNS,
            height=380,
            caption="Pairs ranked by co-report count",
        )
    with right:
        scatter_rows = sorted(view, key=lambda r: ds.as_int(r.get("reports")), reverse=True)[:300]
        st.plotly_chart(
            ch.scatter(
                [ds.as_float(r.get("support"), 0.0) for r in scatter_rows],
                [ds.as_float(r.get("lift"), 0.0) for r in scatter_rows],
                title="Support vs lift",
                x_title="Support (share of all reports)",
                y_title="Lift",
                color=[ds.as_int(r.get("reports")) for r in scatter_rows],
                colorbar_title="Reports",
                hover_text=[_pair_label(r) for r in scatter_rows],
                log_y=True,
                height=400,
                colorscale="Blues",
            ),
            width="stretch",
        )
    ui.download_rows(
        view, "cooccurrence_pairs.csv", label="Download co-occurrence pairs", key="dl_cooc"
    )

    st.divider()
    st.markdown("#### Interactive pair exploration")
    drugs = sorted({p.get("drug_a", "") for p in pairs} | {p.get("drug_b", "") for p in pairs})
    drugs = [d for d in drugs if d]
    if not drugs:
        return
    chosen = st.selectbox("Select a drug", options=drugs, key="cooc_drug")
    related = ds.filter_rows(
        "cooccurrence_top_pairs", ["drug_a", "drug_b"], chosen, limit=200
    )
    if not related:
        ui.empty_notice(f"{chosen} has no pair in the top co-occurrence output.")
        return

    related = sorted(related, key=lambda r: ds.as_int(r.get("reports")), reverse=True)
    st.plotly_chart(
        ch.horizontal_bar(
            [
                (r.get("drug_b") if r.get("drug_a") == chosen else r.get("drug_a"))
                for r in related[:20]
            ],
            [ds.as_int(r.get("reports")) for r in related[:20]],
            title=f"Drugs most often co-reported with {chosen}",
            x_title="Reports containing both drugs",
            color=ch.PALETTE[3],
            hover_suffix="co-reports",
        ),
        width="stretch",
    )
    ui.render_table(related[:20], columns=_PAIR_COLUMNS, height=320)
    ui.download_rows(
        related, f"cooccurrence_{chosen}.csv", label="Download these pairs", key="dl_cooc_drug"
    )

    st.divider()
    st.markdown("#### Co-reported drug groups")
    if not ds.is_available("cooccurrence_top_groups"):
        ui.missing_output("Co-occurrence groups", "cooccurrence_top_groups")
    else:
        groups = ds.get_rows(
            "cooccurrence_top_groups",
            columns=["drug_a", "drug_b", "drug_c", "reports", "support"],
        )
        if not groups:
            ui.empty_notice("No co-reported groups available.")
        else:
            groups = sorted(groups, key=lambda r: ds.as_int(r.get("reports")), reverse=True)[:20]
            labels = [
                " + ".join(x for x in (g.get("drug_a"), g.get("drug_b"), g.get("drug_c")) if x)
                for g in groups
            ]
            st.plotly_chart(
                ch.horizontal_bar(
                    labels,
                    [ds.as_int(r.get("reports")) for r in groups],
                    title="Most frequent triples",
                    x_title="Reports containing all three drugs",
                    color=ch.PALETTE[4],
                    hover_suffix="co-reports",
                ),
                width="stretch",
            )
            ui.download_rows(groups, "cooccurrence_groups.csv", key="dl_cooc_groups")

    st.divider()
    st.markdown("#### Quarterly co-occurrence")
    if not ds.is_available("cooccurrence_pairs_by_quarter"):
        ui.missing_output("Pairs by quarter", "cooccurrence_pairs_by_quarter")
        return
    q_rows = ds.get_rows("cooccurrence_pairs_by_quarter", columns=_PAIR_COLUMNS + ["quarter"])
    if not q_rows:
        ui.empty_notice("No per-quarter co-occurrence rows available.")
        return
    quarter = st.radio(
        "Quarter", options=ds.QUARTERS, horizontal=True, key="cooc_q"
    )
    q_view = [r for r in q_rows if r.get("quarter") == quarter]
    if not q_view:
        ui.empty_notice(f"No co-occurrence rows for {quarter}.")
        return
    q_view = sorted(q_view, key=lambda r: ds.as_int(r.get("reports")), reverse=True)[:20]
    st.plotly_chart(
        ch.horizontal_bar(
            [_pair_label(r) for r in q_view],
            [ds.as_int(r.get("reports")) for r in q_view],
            title=f"Top co-reported pairs in {quarter}",
            x_title="Reports containing both drugs",
            color=ch.QUARTER_COLORS.get(quarter, ch.PALETTE[0]),
            hover_suffix="co-reports",
        ),
        width="stretch",
    )
    ui.render_table(q_view, columns=_PAIR_COLUMNS, height=320)
    ui.download_rows(q_view, f"cooccurrence_{quarter}.csv", key="dl_cooc_q")
