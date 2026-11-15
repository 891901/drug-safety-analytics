"""Section 7 - Machine Learning (Isolation Forest anomalies).

Anomaly scores, the score distribution, the top-ranked anomalous pairs and an
interactive filter.  An anomaly flag only means the pair sits apart from the
rest of the screened population across the model features.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui

ANOM_COLUMNS = [
    "anomaly_rank",
    "drug",
    "reaction",
    "a",
    "n_drug",
    "n_reac",
    "prr",
    "ror",
    "chi_square",
    "quarters_observed",
    "quarter_report_balance",
    "anomaly_score",
    "is_anomaly",
]


def render() -> None:
    ui.page_header(
        "Machine Learning - Isolation Forest",
        "Unsupervised anomaly detection over the engineered signal features. An "
        "anomaly is a statistical outlier, not a new safety signal.",
    )
    ui.disclaimer()

    summary = ds.get_json("anomaly_summary") if ds.is_available("anomaly_summary") else None
    if summary:
        scoring = summary.get("scoring", {})
        model = summary.get("model", {})
        ui.metric_row(
            [
                {"label": "Rows scored", "value": ds.fmt_int(scoring.get("rows_scored"))},
                {"label": "Anomalies flagged", "value": ds.fmt_int(scoring.get("anomaly_count"))},
                {
                    "label": "Anomaly rate",
                    "value": ds.fmt_num(scoring.get("anomaly_rate"), 4),
                    "help": "Matches the configured contamination.",
                },
                {
                    "label": "Score range",
                    "value": f"{ds.fmt_num(scoring.get('score_min'), 4)} .. "
                    f"{ds.fmt_num(scoring.get('score_max'), 4)}",
                },
            ]
        )
        if model:
            ui.stat_caption(
                f"IsolationForest: n_estimators={model.get('n_estimators')}, "
                f"contamination={model.get('contamination')}, "
                f"max_samples={model.get('max_samples')}, "
                f"random_state={model.get('random_state')}. "
                f"{len(summary.get('features', []))} engineered features."
            )

    if not ds.is_available("anomaly_results"):
        ui.missing_output("Anomaly results", "anomaly_results")
        return

    st.divider()
    st.markdown("#### Anomaly score distribution")
    scores = ds.numeric_column("anomaly_results", "anomaly_score")
    if not scores:
        ui.empty_notice("No anomaly scores available.")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(
                ch.histogram(
                    scores,
                    title="Distribution of Isolation Forest anomaly scores",
                    x_title="Anomaly score",
                    y_title="Signal pairs",
                    bins=60,
                    color=ch.PALETTE[0],
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                ch.histogram(
                    [abs(s) for s in scores],
                    title="|Anomaly score| (tail = most anomalous)",
                    x_title="|Anomaly score|",
                    y_title="Signal pairs",
                    bins=60,
                    color=ch.PALETTE[1],
                ),
                width="stretch",
            )
        st.caption(
            f"{len(scores):,} scored pairs. Higher scores sit further from the "
            "bulk of the population; the tail drives the anomaly flag."
        )

    st.divider()
    st.markdown("#### Top anomalous records")
    top_n = st.slider("Records to display", 10, 500, 25, 5, key="anom_top_n")
    top = ds.get_rows("anomaly_results", limit=top_n, columns=ANOM_COLUMNS)
    if not top:
        ui.empty_notice("No anomalous records available.")
        return

    st.plotly_chart(
        ch.horizontal_bar(
            [f"{r.get('drug')} / {r.get('reaction')}"[:60] for r in top],
            [ds.as_float(r.get("anomaly_score"), 0.0) for r in top],
            title=f"Top {len(top)} anomalous drug-reaction pairs",
            x_title="Anomaly score",
            color=ch.PALETTE[6],
            hover_suffix="score",
        ),
        width="stretch",
    )
    ui.render_table(top, columns=ANOM_COLUMNS, height=420)
    ui.download_rows(
        top, "anomaly_top_records.csv", label="Download top anomalies", key="dl_anom_top"
    )

    st.divider()
    st.markdown("#### Anomaly explorer")
    pool = ds.get_rows("anomaly_results", limit=3000, columns=ANOM_COLUMNS)
    if not pool:
        return
    min_score = st.slider(
        "Minimum anomaly score",
        min_value=float(min(ds.as_float(r.get("anomaly_score"), 0.0) for r in pool)),
        max_value=float(max(ds.as_float(r.get("anomaly_score"), 0.0) for r in pool)),
        value=float(sorted(ds.as_float(r.get("anomaly_score"), 0.0) for r in pool)[int(len(pool) * 0.5)]),
        key="anom_min_score",
    )
    drugs = sorted({r.get("drug", "") for r in pool if r.get("drug")})
    drug = st.selectbox("Drug filter", options=["(any)"] + drugs, key="anom_drug")
    only_flagged = st.checkbox("Only flagged anomalies", value=True, key="anom_flag")
    filtered = [
        r
        for r in pool
        if ds.as_float(r.get("anomaly_score"), 0.0) >= float(min_score)
        and (drug == "(any)" or r.get("drug") == drug)
        and (not only_flagged or ds.as_bool(r.get("is_anomaly")))
    ]
    st.caption(
        f"{len(filtered)} of the top {len(pool):,} ranked records match. The "
        "explorer searches the head of the ranked output."
    )
    if filtered:
        ui.render_table(filtered[:200], columns=ANOM_COLUMNS, height=360)
        st.plotly_chart(
            ch.scatter(
                [ds.as_float(r.get("prr"), 0.0) for r in filtered],
                [ds.as_float(r.get("rough_ror", r.get("ror")), 0.0) for r in filtered],
                title="PRR vs ROR for the filtered anomalies",
                x_title="PRR",
                y_title="ROR",
                color=[ds.as_float(r.get("anomaly_score"), 0.0) for r in filtered],
                colorbar_title="Anomaly score",
                hover_text=[f"{r.get('drug')} / {r.get('reaction')}" for r in filtered],
                log_x=True,
                log_y=True,
                colorscale="Hot",
            ),
            width="stretch",
        )
        ui.download_rows(
            filtered, "anomalies_filtered.csv", label="Download filtered anomalies", key="dl_anom_f"
        )
    else:
        ui.empty_notice("No anomalies match the current filter.")
