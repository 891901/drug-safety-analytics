"""Section 9 - Demographic Analysis.

Age and gender distributions and serious-outcome shares.  These describe who was
REPORTED, not who was exposed, so they are not incidence or risk estimates.
"""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from dashboard.components import charts as ch
from dashboard.components import data as ds
from dashboard.components import ui

AGE_ORDER = ["0-17", "18-44", "45-64", "65-74", "75+", "Unknown"]
SEX_ORDER = ["F", "M", "Unknown"]


def _order(items: List[str], order: List[str]) -> List[str]:
    known = [x for x in order if x in items]
    extra = [x for x in items if x not in known]
    return known + extra


def render() -> None:
    ui.page_header(
        "Demographic Analysis",
        "Age and gender breakdowns of the reported population and serious-outcome "
        "shares. Missing values are shown as an explicit 'Unknown' group.",
    )
    ui.disclaimer()

    age = ds.get_rows("demographic_age_distribution", columns=["age_bucket", "reports", "pct_of_reports"])
    sex = ds.get_rows("demographic_gender_distribution", columns=["sex", "reports", "pct_of_reports"])

    st.markdown("#### Age and gender distribution")
    col1, col2 = st.columns(2)
    with col1:
        if not age:
            ui.missing_output("Age distribution", "demographic_age_distribution")
        else:
            ordered = sorted(age, key=lambda r: _index(r.get("age_bucket"), AGE_ORDER))
            st.plotly_chart(
                ch.vertical_bar(
                    [r.get("age_bucket", "") for r in ordered],
                    [ds.as_int(r.get("reports")) for r in ordered],
                    title="Reports by age group",
                    y_title="Reports",
                    color=ch.PALETTE[0],
                    text=[f"{ds.as_int(r.get('reports')):,}" for r in ordered],
                    height=400,
                ),
                width="stretch",
            )
    with col2:
        if not sex:
            ui.missing_output("Gender distribution", "demographic_gender_distribution")
        else:
            ordered = sorted(sex, key=lambda r: _index(r.get("sex"), SEX_ORDER))
            st.plotly_chart(
                ch.pie_chart(
                    [r.get("sex", "") for r in ordered],
                    [ds.as_int(r.get("reports")) for r in ordered],
                    title="Reports by sex",
                    hole=0.45,
                    height=400,
                ),
                width="stretch",
            )

    if age or sex:
        combined = []
        for r in age:
            combined.append(
                {"dimension": "age", "group": r.get("age_bucket"), "reports": r.get("reports"),
                 "pct_of_reports": r.get("pct_of_reports")}
            )
        for r in sex:
            combined.append(
                {"dimension": "sex", "group": r.get("sex"), "reports": r.get("reports"),
                 "pct_of_reports": r.get("pct_of_reports")}
            )
        ui.download_rows(combined, "demographics.csv", label="Download demographics", key="dl_demog")

    st.divider()
    st.markdown("#### Age x gender reporting matrix")
    if not ds.is_available("demographic_age_by_gender"):
        ui.missing_output("Age by gender", "demographic_age_by_gender")
    else:
        grid = ds.get_rows("demographic_age_by_gender", columns=["sex", "age_bucket", "reports"])
        if not grid:
            ui.empty_notice("No age-by-gender rows available.")
        else:
            sexes = _order(sorted({r.get("sex", "") for r in grid}), SEX_ORDER)
            buckets = _order(sorted({r.get("age_bucket", "") for r in grid}), AGE_ORDER)
            lookup = {(r.get("sex"), r.get("age_bucket")): ds.as_int(r.get("reports")) for r in grid}
            z = [[lookup.get((s, b), 0) for b in buckets] for s in sexes]
            text = [[f"{v:,}" for v in row] for row in z]
            st.plotly_chart(
                ch.heatmap(
                    z,
                    buckets,
                    sexes,
                    title="Reports by sex and age group",
                    x_title="Age group",
                    y_title="Sex",
                    text=text,
                    height=380,
                ),
                width="stretch",
            )
            ui.download_rows(grid, "age_by_gender.csv", key="dl_age_gender")

    st.divider()
    st.markdown("#### Reactions by demographic group")
    tab_age, tab_sex = st.tabs(["By age group", "By sex"])
    with tab_age:
        rows = ds.get_rows("demographic_reactions_by_age", columns=["age_bucket", "reaction_pt", "reports"])
        if not rows:
            ui.missing_output("Reactions by age", "demographic_reactions_by_age")
        else:
            buckets = _order(sorted({r.get("age_bucket", "") for r in rows}), AGE_ORDER)
            chosen = st.selectbox("Age group", options=buckets, key="demog_age")
            view = sorted(
                [r for r in rows if r.get("age_bucket") == chosen],
                key=lambda r: ds.as_int(r.get("reports")),
                reverse=True,
            )[:15]
            st.plotly_chart(
                ch.horizontal_bar(
                    [r.get("reaction_pt", "") for r in view],
                    [ds.as_int(r.get("reports")) for r in view],
                    title=f"Top reactions reported in age group {chosen}",
                    x_title="Reports",
                    color=ch.PALETTE[2],
                ),
                width="stretch",
            )
            ui.download_rows(rows, "reactions_by_age.csv", key="dl_rxn_age")
    with tab_sex:
        rows = ds.get_rows("demographic_reactions_by_gender", columns=["sex", "reaction_pt", "reports"])
        if not rows:
            ui.missing_output("Reactions by gender", "demographic_reactions_by_gender")
        else:
            sexes = _order(sorted({r.get("sex", "") for r in rows}), SEX_ORDER)
            chosen = st.selectbox("Sex", options=sexes, key="demog_sex")
            view = sorted(
                [r for r in rows if r.get("sex") == chosen],
                key=lambda r: ds.as_int(r.get("reports")),
                reverse=True,
            )[:15]
            st.plotly_chart(
                ch.horizontal_bar(
                    [r.get("reaction_pt", "") for r in view],
                    [ds.as_int(r.get("reports")) for r in view],
                    title=f"Top reactions reported for sex {chosen}",
                    x_title="Reports",
                    color=ch.PALETTE[5],
                ),
                width="stretch",
            )
            ui.download_rows(rows, "reactions_by_gender.csv", key="dl_rxn_sex")

    st.divider()
    st.markdown("#### Serious outcome comparisons")
    if not ds.is_available("demographic_serious_outcomes"):
        ui.missing_output("Serious outcomes by demographic", "demographic_serious_outcomes")
        return
    so = ds.get_rows(
        "demographic_serious_outcomes",
        columns=["group_type", "group", "serious_reports", "total_reports", "share_pct"],
    )
    if not so:
        ui.empty_notice("No serious-outcome rows available.")
        return

    for group_type, title in (("age", "Share of reports with a serious outcome, by age group"),
                              ("gender", "Share of reports with a serious outcome, by sex")):
        subset = [r for r in so if r.get("group_type") == group_type]
        if not subset:
            continue
        st.plotly_chart(
            ch.vertical_bar(
                [r.get("group", "") for r in subset],
                [ds.as_float(r.get("share_pct"), 0.0) for r in subset],
                title=title,
                y_title="Serious reports as % of group reports",
                color=ch.PALETTE[1] if group_type == "age" else ch.PALETTE[3],
                text=[f"{ds.fmt_num(r.get('share_pct'), 1)}%" for r in subset],
                height=380,
            ),
            width="stretch",
        )
    ui.render_table(
        so,
        height=300,
        caption="Serious-outcome share by group (a report can carry several outcome codes)",
    )
    ui.download_rows(
        so, "serious_outcomes_demographic.csv", label="Download serious outcomes", key="dl_serious"
    )
    st.caption(
        "Shares describe the reported population only. A higher share in a group "
        "does not mean that group is at higher clinical risk."
    )


def _index(value, order: List[str]) -> int:
    try:
        return order.index(value)
    except ValueError:
        return len(order)
