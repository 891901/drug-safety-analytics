"""Section 6 - Statistical Signal Detection.

PRR / ROR disproportionality signals with 95% confidence intervals, the
top-ranked signals and a Q1 vs Q2 comparison.  A signal only means the pair was
reported together more often than the background rate predicts.
"""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui

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


def _signal_table(rows: List[dict]) -> None:
    ui.render_table(rows, columns=SIGNAL_COLUMNS, height=400)


def _ci_chart(rows: List[dict], title: str) -> None:
    subset = rows[:30]
    st.plotly_chart(
        ch.error_bar_scatter(
            [f"#{i + 1}" for i in range(len(subset))],
            [ds.as_float(r.get("prr"), 0.0) for r in subset],
            [ds.as_float(r.get("prr_ci_low"), 0.0) for r in subset],
            [ds.as_float(r.get("prr_ci_high"), 0.0) for r in subset],
            title=title,
            x_title="Signal rank",
            y_title="PRR (95% CI, log scale)",
            log_y=True,
            height=460,
        ),
        width="stretch",
    )


def render() -> None:
    ui.page_header(
        "Statistical Signal Detection",
        "Disproportionality analysis (PRR and ROR) with 95% confidence intervals. "
        "A signal is not proof of causation or clinical risk.",
    )
    ui.disclaimer()

    summary = ds.get_json("signal_summary") if ds.is_available("signal_summary") else None
    if summary:
        counts = summary.get("counts", {})
        thresholds = summary.get("thresholds", {})
        ui.metric_row(
            [
                {"label": "Pairs screened", "value": ds.fmt_int(counts.get("screened_pairs"))},
                {
                    "label": "Signals (meeting thresholds)",
                    "value": ds.fmt_int(counts.get("signals_total")),
                },
                {
                    "label": "Signals in output",
                    "value": ds.fmt_int(counts.get("signals_written")),
                    "help": "Ranked output is capped; it is a subset of all signals.",
                },
                {
                    "label": "Haldane-Anscombe corrected",
                    "value": ds.fmt_int(counts.get("corrected_rows")),
                },
            ]
        )
        if thresholds:
            ui.stat_caption(
                "Thresholds: a >= {a}, PRR >= {prr}, ROR CI low > {ror}, "
                "chi-square >= {chi}. 95% CI z = {z}.".format(
                    a=thresholds.get("min_reports"),
                    prr=thresholds.get("prr_threshold"),
                    ror=thresholds.get("ror_lower_ci_threshold"),
                    chi=thresholds.get("chi_square_threshold"),
                    z=thresholds.get("z_95"),
                )
            )
    else:
        ui.missing_output("Signal summary", "signal_summary")

    if not ds.is_available("signal_signals"):
        ui.missing_output("Signal table", "signal_signals")
        return

    st.divider()
    tab_overall, tab_quarter, tab_explore = st.tabs(
        ["Top signals", "Q1 vs Q2", "Explorer"]
    )

    # ------------------------------------------------------------- top signals
    with tab_overall:
        top_n = st.slider(
            "Number of signals", min_value=10, max_value=200, value=25, step=5,
            key="sig_top_n",
        )
        rows = ds.get_rows("signal_signals", limit=top_n, columns=SIGNAL_COLUMNS)
        if not rows:
            ui.empty_notice("No signal rows available.")
        else:
            _ci_chart(rows, "Top signals by PRR with 95% confidence intervals")
            _signal_table(rows)
            st.caption(
                "Very large PRR/ROR values arise when the background cell is zero "
                "(Haldane-Anscombe corrected). The wide intervals show how "
                "uncertain those estimates are."
            )
            st.plotly_chart(
                ch.scatter(
                    [ds.as_float(r.get("prr"), 0.0) for r in rows],
                    [ds.as_float(r.get("ror"), 0.0) for r in rows],
                    title="PRR vs ROR (log scales)",
                    x_title="PRR",
                    y_title="ROR",
                    color=[ds.as_float(r.get("chi_square"), 0.0) for r in rows],
                    colorbar_title="chi-square",
                    hover_text=[
                        f"{r.get('drug')} / {r.get('reaction')}" for r in rows
                    ],
                    log_x=True,
                    log_y=True,
                    colorscale="Blues",
                ),
                width="stretch",
            )
            ui.download_rows(rows, "signals_top.csv", label="Download these signals", key="dl_sig_top")

    # ------------------------------------------------------------- Q1 vs Q2
    with tab_quarter:
        if summary:
            by_q = summary.get("signals_by_quarter", {})
            screened_q = summary.get("screened_pairs_by_quarter", {})
            quarters = [q for q in ds.QUARTERS if q in by_q]
            if quarters:
                st.plotly_chart(
                    ch.grouped_bar(
                        quarters,
                        {
                            "Screened pairs": [ds.as_int(screened_q.get(q)) for q in quarters],
                            "Signals": [ds.as_int(by_q.get(q)) for q in quarters],
                        },
                        title="Screened pairs and signals by quarter",
                        y_title="Count",
                        colors=ch.QUARTER_COLORS,
                        height=400,
                    ),
                    width="stretch",
                )
        if not ds.is_available("signal_signals_by_quarter"):
            ui.missing_output("Signals by quarter", "signal_signals_by_quarter")
        else:
            quarter = st.radio("Quarter", options=ds.QUARTERS, horizontal=True, key="sig_q")
            n = st.slider("Signals per quarter", 10, 200, 25, 5, key="sig_q_n")
            q_rows = ds.filter_rows(
                "signal_signals_by_quarter", ["quarter"], quarter, limit=n
            )
            if not q_rows:
                ui.empty_notice(f"No signals for {quarter}.")
            else:
                st.plotly_chart(
                    ch.horizontal_bar(
                        [f"{r.get('drug')} / {r.get('reaction')}"[:60] for r in q_rows],
                        [ds.as_float(r.get("prr"), 0.0) for r in q_rows],
                        title=f"Top signals in {quarter} by PRR",
                        x_title="PRR",
                        color=ch.QUARTER_COLORS.get(quarter, ch.PALETTE[0]),
                        hover_suffix="PRR",
                    ),
                    width="stretch",
                )
                _signal_table(q_rows)
                ui.download_rows(
                    q_rows, f"signals_{quarter}.csv", label="Download signals", key="dl_sig_q"
                )

    # ------------------------------------------------------------- explorer
    with tab_explore:
        seed_rows = ds.get_rows("signal_signals", limit=2000, columns=SIGNAL_COLUMNS)
        if not seed_rows:
            ui.empty_notice("No signal rows available.")
            return
        drugs = sorted({r.get("drug", "") for r in seed_rows if r.get("drug")})
        reactions = sorted({r.get("reaction", "") for r in seed_rows if r.get("reaction")})
        min_prr = st.number_input(
            "Minimum PRR", min_value=0.0, value=0.0, step=1.0, key="sig_min_prr"
        )
        drug_filter = st.selectbox("Drug filter", options=["(any)"] + drugs, key="sig_drug")
        reac_filter = st.selectbox(
            "Reaction filter", options=["(any)"] + reactions, key="sig_reac"
        )
        filtered = [
            r
            for r in seed_rows
            if ds.as_float(r.get("prr"), 0.0) >= float(min_prr)
            and (drug_filter == "(any)" or r.get("drug") == drug_filter)
            and (reac_filter == "(any)" or r.get("reaction") == reac_filter)
        ]
        st.caption(
            f"{len(filtered)} of the first {len(seed_rows)} ranked signals match. "
            "The explorer searches the head of the ranked output, not the full file."
        )
        if filtered:
            _signal_table(filtered[:200])
            ui.download_rows(
                filtered, "signals_filtered.csv", label="Download filtered signals", key="dl_sig_f"
            )
        else:
            ui.empty_notice("No signals match the current filters.")
