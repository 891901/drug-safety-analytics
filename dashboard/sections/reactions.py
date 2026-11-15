"""Section 3 - Adverse Reaction Analysis.

Top reported reactions, their quarterly movement, the drugs most often
co-reported with a chosen reaction, and the overall outcome-code mix.
"""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui


def render() -> None:
    ui.page_header(
        "Adverse Reaction Analysis",
        "Most frequently reported MedDRA preferred terms (reactions) in the "
        "2026 Q1/Q2 extracts.",
    )
    ui.disclaimer()

    if not ds.is_available("adr_top_reactions"):
        ui.missing_output("Top reactions", "adr_top_reactions")
        return

    rows = ds.get_rows(
        "adr_top_reactions", columns=["reaction_pt", "reports", "pct_of_reports"]
    )
    if not rows:
        ui.empty_notice("The top-reactions output is present but empty.")
        return

    top_n = st.slider(
        "Number of reactions to display",
        min_value=5,
        max_value=max(5, min(len(rows), 100)),
        value=min(20, len(rows)),
        step=5,
    )
    selected = rows[:top_n]

    st.plotly_chart(
        ch.horizontal_bar(
            [r.get("reaction_pt", "") for r in selected],
            [ds.as_int(r.get("reports")) for r in selected],
            title=f"Top {top_n} reported reactions",
            x_title="Reports",
            color=ch.PALETTE[2],
        ),
        width="stretch",
    )

    left, right = st.columns([3, 2])
    with left:
        ui.render_table(
            selected,
            columns=["reaction_pt", "reports", "pct_of_reports"],
            height=340,
            caption="Reaction frequency",
        )
    with right:
        ui.download_rows(
            selected,
            "top_reactions.csv",
            label="Download top reactions",
            key="dl_top_reactions",
        )
        ui.stat_caption(
            f"{len(rows)} reactions are ranked in the output. Counts here are "
            "report-level: one report contributes at most once per reaction."
        )

    st.divider()

    reactions = [r.get("reaction_pt", "") for r in rows if r.get("reaction_pt")]
    if not reactions:
        return

    tab_q, tab_dr, tab_out = st.tabs(
        ["Reaction by quarter", "Drug-reaction comparison", "Outcome mix"]
    )

    # ------------------------------------------------- quarter comparison
    with tab_q:
        if not ds.is_available("adr_reaction_by_quarter"):
            ui.missing_output("Reactions by quarter", "adr_reaction_by_quarter")
        else:
            chosen = st.selectbox("Reaction", options=reactions, key="reac_q")
            q_rows = ds.filter_rows(
                "adr_reaction_by_quarter", ["reaction_pt"], chosen, limit=10
            )
            if not q_rows:
                ui.empty_notice(f"No per-quarter rows for {chosen}.")
            else:
                by_q: Dict[str, int] = {
                    r.get("quarter", ""): ds.as_int(r.get("reports")) for r in q_rows
                }
                quarters = [q for q in ds.QUARTERS if q in by_q] or list(by_q)
                st.plotly_chart(
                    ch.vertical_bar(
                        quarters,
                        [by_q.get(q, 0) for q in quarters],
                        title=f"{chosen} - reports by quarter",
                        y_title="Reports",
                        colors=[ch.QUARTER_COLORS.get(q, ch.PALETTE[0]) for q in quarters],
                        text=[f"{by_q.get(q, 0):,}" for q in quarters],
                        height=380,
                    ),
                    width="stretch",
                )
                ui.download_rows(
                    q_rows, f"reaction_quarters_{chosen}.csv", key="dl_reac_q"
                )

    # ------------------------------------------------- drug-reaction compare
    with tab_dr:
        if not ds.is_available("adr_drug_reaction_pairs"):
            ui.missing_output("Drug-reaction pairs", "adr_drug_reaction_pairs")
        else:
            chosen = st.selectbox("Reaction", options=reactions, key="reac_dr")
            dr = ds.filter_rows(
                "adr_drug_reaction_pairs", ["reaction_pt"], chosen, limit=100
            )
            if not dr:
                ui.empty_notice(
                    f"{chosen} is not in the top-100 drug-reaction pair ranking."
                )
            else:
                dr = sorted(dr, key=lambda r: ds.as_int(r.get("reports")), reverse=True)
                st.plotly_chart(
                    ch.horizontal_bar(
                        [r.get("drug_name", "") for r in dr[:20]],
                        [ds.as_int(r.get("reports")) for r in dr[:20]],
                        title=f"Top drugs co-reported with {chosen}",
                        x_title="Reports",
                        color=ch.PALETTE[5],
                    ),
                    width="stretch",
                )
                ui.download_rows(dr, f"drug_reaction_{chosen}.csv", key="dl_dr_reac")
                st.caption(
                    "Only pairs from the top-100 ranked drug-reaction pairs. This is "
                    "a co-reporting count, not an interaction or causal link."
                )

    # ------------------------------------------------- outcome mix
    with tab_out:
        if not ds.is_available("adr_outcome_counts"):
            ui.missing_output("Outcome counts", "adr_outcome_counts")
        else:
            outc = ds.get_rows(
                "adr_outcome_counts",
                columns=["outc_cod", "outcome_label", "reports", "pct_of_reports"],
            )
            if not outc:
                ui.empty_notice("Outcome output is empty.")
            else:
                outc = sorted(
                    outc, key=lambda r: ds.as_int(r.get("reports")), reverse=True
                )
                labels = [
                    r.get("outcome_label") or r.get("outc_cod", "") for r in outc
                ]
                st.plotly_chart(
                    ch.horizontal_bar(
                        labels,
                        [ds.as_int(r.get("reports")) for r in outc],
                        title="Reported serious outcome codes",
                        x_title="Reports",
                        color=ch.PALETTE[6],
                    ),
                    width="stretch",
                )
                ui.render_table(
                    outc,
                    columns=["outc_cod", "outcome_label", "reports", "pct_of_reports"],
                    height=300,
                    caption="Outcome distribution (a report may carry several outcome codes)",
                )
                ui.download_rows(
                    outc, "outcome_counts.csv", label="Download outcome counts", key="dl_outc"
                )
