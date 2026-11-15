"""Section 10 - Drug Investigation.

Pick a single drug and see everything the project knows about it, assembled
from the real outputs: its reported reactions, the drugs it is most often
co-reported with, its statistical signals (PRR/ROR with confidence intervals),
its Isolation Forest anomaly records and its position in the co-reporting
network.

All lookups stream the relevant CSV with an early stop, so choosing a drug
never loads or recomputes the full FAERS dataset.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui

PAIR_COLUMNS = [
    "drug_a",
    "drug_b",
    "reports",
    "support",
    "confidence_a_to_b",
    "confidence_b_to_a",
    "lift",
    "passes_thresholds",
]

SIGNAL_COLUMNS = [
    "drug",
    "reaction",
    "a",
    "n_drug",
    "n_reac",
    "prr",
    "prr_ci_low",
    "prr_ci_high",
    "ror",
    "ror_ci_low",
    "ror_ci_high",
    "chi_square",
    "corrected",
]

ANOM_COLUMNS = [
    "anomaly_rank",
    "drug",
    "reaction",
    "a",
    "n_drug",
    "n_reac",
    "prr",
    "ror",
    "anomaly_score",
    "is_anomaly",
]

NODE_COLUMNS = [
    "drug",
    "degree",
    "weighted_degree",
    "degree_centrality",
    "betweenness_centrality",
    "community",
    "in_displayed_network",
]


def _drug_options() -> List[str]:
    """Build the selectable drug list from the real drug report-count output.

    Falls back to the drugs present in the ranked signal output if the
    report-count file is unavailable.  Nothing is invented: if neither source
    exists the list is empty and the caller shows a notice.
    """
    options: List[str] = []
    if ds.is_available("adr_drug_report_counts"):
        options = [
            r.get("drug_name", "").strip()
            for r in ds.get_rows("adr_drug_report_counts", columns=["drug_name"])
        ]
        options = [d for d in options if d]
    if not options and ds.is_available("signal_signals"):
        options = ds.distinct_values("signal_signals", "drug", limit=5000)
    return options


def _report_rank(drug: str) -> Optional[int]:
    rows = ds.get_rows("adr_drug_report_counts", columns=["drug_name", "reports", "pct_of_reports"])
    for index, row in enumerate(rows):
        if row.get("drug_name", "").strip() == drug:
            return index + 1
    return None


def _report_row(drug: str) -> Dict[str, str]:
    rows = ds.get_rows("adr_drug_report_counts", columns=["drug_name", "reports", "pct_of_reports"])
    for row in rows:
        if row.get("drug_name", "").strip() == drug:
            return row
    return {}


def _network_row(drug: str) -> Dict[str, str]:
    if not ds.is_available("network_nodes"):
        return {}
    for row in ds.get_rows("network_nodes", columns=NODE_COLUMNS):
        if row.get("drug", "").strip() == drug:
            return row
    return {}


def render() -> None:
    ui.page_header(
        "Drug Investigation",
        "Select one drug to see its reported reactions, co-reported drugs, "
        "statistical signals, anomaly records and network position - all read "
        "from the project's real outputs.",
    )
    ui.disclaimer()

    options = _drug_options()
    if not options:
        ui.empty_notice(
            "No drug list is available. Run the ADR analysis (drug report counts) "
            "or the signal detection pipeline first."
        )
        return

    drug = st.selectbox("Select a drug", options=options, index=0, key="inv_drug")

    # ------------------------------------------------------------ headline card
    row = _report_row(drug)
    rank = _report_rank(drug)
    net = _network_row(drug)
    ui.metric_row(
        [
            {
                "label": "Reports involving this drug",
                "value": ds.fmt_int(row.get("reports")) if row else "n/a",
                "help": "Distinct FAERS reports mentioning the drug.",
            },
            {
                "label": "Share of all reports",
                "value": f"{ds.fmt_num(row.get('pct_of_reports'), 3)} %"
                if row
                else "n/a",
            },
            {
                "label": "Reporting rank",
                "value": f"#{rank}" if rank else "n/a",
                "help": "Position in the ranked drug report-count output.",
            },
            {
                "label": "Network degree",
                "value": ds.fmt_int(net.get("degree")) if net else "n/a",
                "help": "Distinct co-reported partners in the network output.",
            },
        ]
    )
    if net:
        ui.stat_caption(
            f"Network position: degree centrality "
            f"{ds.fmt_num(net.get('degree_centrality'), 4)}, betweenness "
            f"{ds.fmt_num(net.get('betweenness_centrality'), 4)}, Louvain "
            f"community {net.get('community')}."
        )

    st.divider()

    # ------------------------------------------------------- reported reactions
    st.markdown("#### Associated reactions")
    if ds.is_available("adr_drug_reaction_pairs"):
        pairs = ds.filter_rows("adr_drug_reaction_pairs", ["drug_name"], drug, limit=500)
        if pairs:
            ordered = sorted(pairs, key=lambda r: ds.as_int(r.get("reports")), reverse=True)
            top = ordered[:20]
            st.plotly_chart(
                ch.horizontal_bar(
                    [r.get("reaction_pt", "") for r in top],
                    [ds.as_int(r.get("reports")) for r in top],
                    title=f"Most reported reactions for {drug}",
                    x_title="Reports",
                    color=ch.PALETTE[1],
                ),
                width="stretch",
            )
            ui.render_table(
                ordered,
                columns=["drug_name", "reaction_pt", "reports"],
                height=320,
            )
            st.caption(
                "From the top-ranked drug-reaction pair output. Full PRR/ROR "
                "estimates with confidence intervals are in the signal section below."
            )
            ui.download_rows(
                ordered,
                f"investigation_{drug}_reactions.csv",
                label="Download reactions",
                key="inv_dl_reactions",
            )
        else:
            ui.empty_notice(f"{drug} has no row in the drug-reaction pair output.")
    else:
        ui.missing_output("Drug-reaction pairs", "adr_drug_reaction_pairs")

    st.divider()

    # -------------------------------------------------------- co-reported drugs
    st.markdown("#### Co-reported drugs")
    if ds.is_available("cooccurrence_top_pairs"):
        related = ds.filter_rows(
            "cooccurrence_top_pairs", ["drug_a", "drug_b"], drug, limit=200
        )
        if related:
            related = sorted(related, key=lambda r: ds.as_int(r.get("reports")), reverse=True)
            partners = [
                (r.get("drug_b") if r.get("drug_a") == drug else r.get("drug_a"))
                for r in related
            ]
            st.plotly_chart(
                ch.horizontal_bar(
                    partners[:20],
                    [ds.as_int(r.get("reports")) for r in related[:20]],
                    title=f"Drugs most often co-reported with {drug}",
                    x_title="Reports containing both drugs",
                    color=ch.PALETTE[3],
                    hover_suffix="co-reports",
                ),
                width="stretch",
            )
            ui.render_table(related[:20], columns=PAIR_COLUMNS, height=300)
            st.caption(
                "Support, confidence and lift describe co-reporting association, "
                "not a confirmed interaction."
            )
            ui.download_rows(
                related,
                f"investigation_{drug}_cooccurrence.csv",
                label="Download co-reported pairs",
                key="inv_dl_cooc",
            )
        else:
            ui.empty_notice(f"{drug} has no co-occurrence pair in the top-ranked output.")
    else:
        ui.missing_output("Co-occurrence pairs", "cooccurrence_top_pairs")

    st.divider()

    # -------------------------------------------------------- statistical signals
    st.markdown("#### Statistical signals (PRR / ROR)")
    if ds.is_available("signal_signals"):
        signals = ds.filter_rows("signal_signals", ["drug"], drug, limit=200)
        if signals:
            signals = sorted(signals, key=lambda r: ds.as_float(r.get("prr"), 0.0), reverse=True)
            top = signals[:20]
            left, right = st.columns([3, 2])
            with left:
                st.plotly_chart(
                    ch.error_bar_scatter(
                        [r.get("reaction", "")[:40] for r in top],
                        [ds.as_float(r.get("prr"), 0.0) for r in top],
                        [ds.as_float(r.get("prr_ci_low"), 0.0) for r in top],
                        [ds.as_float(r.get("prr_ci_high"), 0.0) for r in top],
                        title=f"Top signals for {drug} by PRR (95% CI)",
                        x_title="Reaction",
                        y_title="PRR (95% CI, log scale)",
                        log_y=True,
                        height=460,
                    ),
                    width="stretch",
                )
            with right:
                ui.render_table(top, columns=SIGNAL_COLUMNS, height=460)
            st.caption(
                f"{len(signals)} signal rows for {drug} in the ranked signal output. "
                "Wide intervals indicate uncertain estimates."
            )
            ui.download_rows(
                signals,
                f"investigation_{drug}_signals.csv",
                label="Download signals",
                key="inv_dl_signals",
            )
        else:
            ui.empty_notice(
                f"{drug} has no signal in the ranked signal output "
                "(it is a capped, ranked subset of all qualifying pairs)."
            )
    else:
        ui.missing_output("Signal table", "signal_signals")

    st.divider()

    # --------------------------------------------------------- anomaly records
    st.markdown("#### Anomaly information (Isolation Forest)")
    if ds.is_available("anomaly_results"):
        anomalies = ds.filter_rows("anomaly_results", ["drug"], drug, limit=200)
        if anomalies:
            anomalies = sorted(
                anomalies, key=lambda r: ds.as_float(r.get("anomaly_score"), 0.0), reverse=True
            )
            top = anomalies[:20]
            st.plotly_chart(
                ch.horizontal_bar(
                    [r.get("reaction", "")[:50] for r in top],
                    [ds.as_float(r.get("anomaly_score"), 0.0) for r in top],
                    title=f"Anomaly scores for {drug} (top {len(top)})",
                    x_title="Anomaly score",
                    color=ch.PALETTE[6],
                    hover_suffix="score",
                ),
                width="stretch",
            )
            ui.render_table(top, columns=ANOM_COLUMNS, height=320)
            flagged = sum(1 for r in anomalies if ds.as_bool(r.get("is_anomaly")))
            st.caption(
                f"{flagged} of {len(anomalies)} records for {drug} are flagged "
                "anomalies. A flag is a statistical outlier, not a safety signal."
            )
            ui.download_rows(
                anomalies,
                f"investigation_{drug}_anomalies.csv",
                label="Download anomaly records",
                key="inv_dl_anomalies",
            )
        else:
            ui.empty_notice(
                f"{drug} has no record in the scored anomaly output "
                "(the anomaly table is a capped, ranked subset)."
            )
    else:
        ui.missing_output("Anomaly results", "anomaly_results")

    st.divider()
    st.caption(
        "Investigation reads only the head of each ranked output (bounded "
        "memory, no full-dataset recompute). Figures for a drug absent from a "
        "ranked subset are reported as unavailable rather than fabricated."
    )
