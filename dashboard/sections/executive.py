"""Section 1 - Executive Overview.

Headline totals pulled straight from the analysis summaries, plus a Q1 vs Q2
comparison.  Every value is read from a real JSON/CSV output; if one is missing
the metric shows ``n/a`` and a note explains which pipeline to run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui


def _get(key: str) -> Optional[Dict[str, Any]]:
    return ds.get_json(key) if ds.is_available(key) else None


def _counts(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not summary:
        return {}
    counts = summary.get("counts")
    return counts if isinstance(counts, dict) else {}


def render() -> None:
    ui.page_header(
        "Executive Overview",
        "Headline totals and the 2026 Q1 vs Q2 comparison, read directly from "
        "the project's analytical outputs.",
    )
    ui.disclaimer()

    adr = _get("adr_summary")
    signals = _get("signal_summary")
    anomalies = _get("anomaly_summary")
    cooc = _get("cooccurrence_summary")
    temporal = _get("temporal_summary")
    apriori = _get("apriori_summary")

    adr_counts = _counts(adr)
    sig_counts = _counts(signals)
    anom_scoring = (anomalies or {}).get("scoring", {}) if anomalies else {}
    cooc_counts = _counts(cooc)
    rules = _counts(apriori)

    st.markdown("#### Headline totals")

    metrics: List[Dict[str, Any]] = []
    if adr_counts:
        metrics.append(
            {
                "label": "Total reports analysed",
                "value": ds.fmt_int(adr_counts.get("total_reports")),
                "help": "Distinct (quarter, PRIMARYID) FAERS reports.",
            }
        )
        metrics.append(
            {
                "label": "Report-reaction pairs",
                "value": ds.fmt_int(adr_counts.get("distinct_report_reaction_pairs")),
                "help": "Distinct (quarter, PRIMARYID, PT) triples.",
            }
        )
    if sig_counts:
        metrics.append(
            {
                "label": "Drug-reaction pairs screened",
                "value": ds.fmt_int(sig_counts.get("screened_pairs")),
                "help": "Distinct (drug, reaction) contingency pairs evaluated.",
            }
        )
    if cooc_counts:
        metrics.append(
            {
                "label": "Co-occurrence pairs",
                "value": ds.fmt_int(cooc_counts.get("distinct_pairs_overall")),
                "help": "Distinct unordered drug pairs reported together.",
            }
        )
    if sig_counts:
        metrics.append(
            {
                "label": "Statistical signals",
                "value": ds.fmt_int(sig_counts.get("signals_total")),
                "help": "Pairs meeting the PRR / ROR / chi-square thresholds.",
            }
        )
    if anom_scoring:
        metrics.append(
            {
                "label": "Anomalies detected",
                "value": ds.fmt_int(anom_scoring.get("anomaly_count")),
                "help": "Signal pairs flagged by the Isolation Forest.",
            }
        )
    if rules:
        metrics.append(
            {
                "label": "Association rules",
                "value": ds.fmt_int(rules.get("rules_overall")),
                "help": "Apriori rules meeting support/confidence/lift thresholds.",
            }
        )
    if not metrics:
        ui.empty_notice(
            "No summary outputs were found. Run the analytics pipelines to "
            "generate the JSON summaries this section reads."
        )
        return

    # Split the cards across two rows of at most four for readability.
    ui.metric_row(metrics[:4])
    if len(metrics) > 4:
        ui.metric_row(metrics[4:8])

    st.divider()

    # ------------------------------------------------------------------ Q1/Q2
    st.markdown("#### 2026 Q1 vs Q2")
    temporal_block = (temporal or {}).get("temporal", {}) if temporal else {}
    reports_by_q = temporal_block.get("reports_by_quarter", {})
    reactions_by_q = temporal_block.get("reaction_pairs_by_quarter", {})
    serious_by_q = temporal_block.get("serious_reports_by_quarter", {})
    signals_by_q = (signals or {}).get("signals_by_quarter", {}) if signals else {}

    quarters = [q for q in ds.QUARTERS if q in reports_by_q or q in signals_by_q]

    if not quarters:
        ui.missing_output("Quarterly comparison", "temporal_reports_by_quarter")
        return

    q1, q2 = ds.QUARTERS[0], ds.QUARTERS[1]
    r1 = ds.as_int(reports_by_q.get(q1))
    r2 = ds.as_int(reports_by_q.get(q2))
    delta = r2 - r1
    pct = (delta / r1 * 100.0) if r1 else None

    cmp_metrics = [
        {
            "label": f"Reports {q1}",
            "value": ds.fmt_int(r1),
        },
        {
            "label": f"Reports {q2}",
            "value": ds.fmt_int(r2),
            "delta": f"{delta:+,}" if r1 or r2 else None,
            "help": f"{pct:+.2f}% vs {q1}" if pct is not None else None,
        },
        {
            "label": "Report-reaction pairs (Q1 / Q2)",
            "value": f"{ds.fmt_int(reactions_by_q.get(q1))} / {ds.fmt_int(reactions_by_q.get(q2))}",
        },
        {
            "label": "Serious reports (Q1 / Q2)",
            "value": f"{ds.fmt_int(serious_by_q.get(q1))} / {ds.fmt_int(serious_by_q.get(q2))}",
        },
    ]
    if signals_by_q:
        cmp_metrics.append(
            {
                "label": "Signals (Q1 / Q2)",
                "value": f"{ds.fmt_int(signals_by_q.get(q1))} / {ds.fmt_int(signals_by_q.get(q2))}",
            }
        )
    ui.metric_row(cmp_metrics[:4])
    if len(cmp_metrics) > 4:
        ui.metric_row(cmp_metrics[4:])

    series: Dict[str, List[float]] = {}
    if reports_by_q:
        series["Reports"] = [ds.as_int(reports_by_q.get(q)) for q in quarters]
    if reactions_by_q:
        series["Report-reaction pairs"] = [
            ds.as_int(reactions_by_q.get(q)) for q in quarters
        ]
    if serious_by_q:
        series["Serious reports"] = [ds.as_int(serious_by_q.get(q)) for q in quarters]

    left, right = st.columns(2)
    with left:
        if series:
            st.plotly_chart(
                ch.grouped_bar(
                    quarters,
                    series,
                    title="Q1 vs Q2 volumes",
                    y_title="Reports",
                    colors=ch.QUARTER_COLORS,
                    height=420,
                ),
                width="stretch",
            )
        else:
            ui.empty_notice("Quarterly volume data unavailable.")
    with right:
        if len(quarters) == 2:
            growth = {
                "Report-reaction pairs": [
                    ds.as_int(reactions_by_q.get(q)) for q in quarters
                ],
                "Serious reports": [ds.as_int(serious_by_q.get(q)) for q in quarters],
            }
            growth = {k: v for k, v in growth.items() if any(v)}
            if growth:
                st.plotly_chart(
                    ch.grouped_bar(
                        quarters,
                        growth,
                        title="Q2 vs Q1 secondary volumes",
                        y_title="Count",
                        height=420,
                    ),
                    width="stretch",
                )
            else:
                ui.empty_notice("Secondary quarterly volumes unavailable.")
        else:
            ui.empty_notice("Two quarters are required for a Q1/Q2 comparison.")

    if signals_by_q:
        st.plotly_chart(
            ch.grouped_bar(
                quarters,
                {"Signals": [ds.as_int(signals_by_q.get(q)) for q in quarters]},
                title="Statistical signals by quarter",
                y_title="Signals",
                colors=ch.QUARTER_COLORS,
                height=360,
            ),
            width="stretch",
        )

    notes = []
    if sig_counts:
        notes.append(
            f"Signal table is capped at {ds.fmt_int(sig_counts.get('signals_written'))} "
            f"rows out of {ds.fmt_int(sig_counts.get('signals_total'))} qualifying signals."
        )
    if anom_scoring:
        notes.append(
            f"Isolation Forest contamination = {ds.fmt_num(anom_scoring.get('anomaly_rate'), 4)}."
        )
    if notes:
        ui.stat_caption("Note: " + "  ".join(notes))
