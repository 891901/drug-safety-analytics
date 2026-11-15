"""Temporal and demographic analysis of FAERS adverse-event reporting.

This module answers two questions about the **real** FDA FAERS 2026 Q1/Q2
extracts:

*Part A - temporal*  how reporting volume varies over time
  (Q1 vs Q2 volumes, monthly reporting trend from the report dates, the most
  frequent reactions per quarter, and how drug reporting frequency moves from
  one quarter to the next);

*Part B - demographic*  how reporting is distributed across patients
  (age-group and gender distribution, reactions across age groups, reporting
  patterns by gender, and serious outcomes by demographic group).

It reuses the project's own ingestion + preprocessing modules
(:mod:`src.ingestion.load_faers`, :mod:`src.preprocessing.clean_faers`) and the
demographic constants already published by :mod:`src.analytics.adr_analysis`, so
the age buckets, gender labels and OUTC outcome labels stay identical across the
platform.

Design decisions
----------------
* **Report grain = one distinct ``(quarter, PRIMARYID)``.** ``PRIMARYID`` is the
  FAERS report identifier, so it is the only key used to connect DEMO, REAC,
  DRUG and OUTC. Every aggregation is built on the report grain or on a
  *distinct* ``(quarter, PRIMARYID, <item>)`` projection, so duplicate source
  rows never multiply a count.
* **No UDFs, no pandas, no PyArrow, no Parquet.** All aggregation is done with
  the PySpark DataFrame API using built-in functions only. The (small) result
  sets are collected to the driver and persisted with the Python standard
  library (``csv`` / ``json``).
* **Missing / invalid demographics are surfaced, never invented.** A missing age
  or gender becomes an explicit ``Unknown`` category that is shown in every
  table and figure, and the report lists the null counts.
* **The monthly trend is only drawn from a real report date.** The primary date
  is ``REPT_DT_DATE`` (the date the report was received); when it is missing the
  ``FDA_DT_DATE`` fallback is used, and reports with **no** usable date are
  counted separately and never placed into a month. If *no* report carries a
  date the module says so instead of fabricating a trend.
* **Raw data is untouched.** The ``.txt`` files are opened read-only and no
  dataset is written.
* **Counts are counts of reported events.** FAERS is a spontaneous reporting
  system with no denominator and heavy reporting bias: nothing here can be read
  as incidence, prevalence, clinical risk or drug causation. The demographic
  distributions reflect who was *reported*, not who was exposed.

Environment constraints honoured: PySpark DataFrames only (no pandas / PyArrow),
Plotly only for rendering (no Streamlit dependency - every figure builder returns
a ``plotly.graph_objects.Figure`` so the dashboard can import it directly).

Usage::

    python -m src.visualization.temporal_demographic
    python -m src.visualization.temporal_demographic --quarters 2026_Q1 2026_Q2
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.analytics.adr_analysis import (
    AGE_BUCKET_ORDER,
    CAUSATION_DISCLAIMER,
    OUTC_CODE_LABELS,
    SEX_ORDER,
)
from src.ingestion.load_faers import (
    QUARTER_COLUMN,
    IngestionError,
    load_settings,
)
from src.ingestion.spark_session import (
    SparkEnvironmentError,
    create_spark_session,
    stop_spark_session,
)
from src.preprocessing.clean_faers import CleaningError, TableCleaning, clean_table
from src.utils.config import get_path, get_quarter_dirs, get_visualization_config
from src.utils.logging_setup import get_logger

__all__ = [
    "TemporalDemographicError",
    "TEMPORAL_DEMOGRAPHIC_DISCLAIMER",
    "ANALYSIS_TABLES",
    "REPORT_DATE_COLUMN",
    "FALLBACK_DATE_COLUMN",
    "TemporalDemographicResults",
    "analyze",
    "quarterly_comparison_figure",
    "monthly_trend_figure",
    "age_distribution_figure",
    "gender_distribution_figure",
    "top_reactions_figure",
    "serious_outcome_figure",
    "drug_frequency_change_figure",
    "write_outputs",
    "render_report",
    "run",
    "main",
]

#: Tables this module needs (a subset of the seven FAERS tables).
ANALYSIS_TABLES: Tuple[str, ...] = ("DEMO", "DRUG", "REAC", "OUTC")

#: Primary report date column produced by preprocessing (YYYYMMDD/YYYYMM -> DATE).
REPORT_DATE_COLUMN = "REPT_DT_DATE"

#: Fallback report date used when REPT_DT is empty (FDA received date).
FALLBACK_DATE_COLUMN = "FDA_DT_DATE"

#: Month label format used on the trend chart and in the CSV.
MONTH_LABEL_FORMAT = "yyyy-MM"

#: Mandatory caution for the temporal/demographic view.
TEMPORAL_DEMOGRAPHIC_DISCLAIMER = (
    CAUSATION_DISCLAIMER
    + " The demographic breakdowns describe who was REPORTED, not who was "
    "exposed or affected, so they must not be read as disease incidence or as "
    "evidence that any group is at higher clinical risk."
)

#: Output sizing.
TOP_REACTIONS = 20
TOP_DRUGS = 20
TOP_REACTIONS_PER_GROUP = 10
TOP_MONTHS_REPORT = 24
ANALYSIS_GRAIN = (
    "Report = one distinct (quarter, PRIMARYID) pair (FAERS report). "
    "Reaction pair = one distinct (quarter, PRIMARYID, PT) triple. "
    "Serious report = one distinct (quarter, PRIMARYID) with an OUTC row."
)


class TemporalDemographicError(RuntimeError):
    """Raised when a temporal/demographic aggregation cannot be produced."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TemporalDemographicResults:
    """Every artefact produced by one temporal/demographic run."""

    quarters: List[str] = field(default_factory=list)
    cleanings: Dict[str, TableCleaning] = field(default_factory=dict)

    # --- Part A: temporal ---------------------------------------------------
    total_reports: int = 0
    reports_by_quarter: Dict[str, int] = field(default_factory=dict)
    reaction_pairs_by_quarter: Dict[str, int] = field(default_factory=dict)
    reports_with_reaction_by_quarter: Dict[str, int] = field(default_factory=dict)
    serious_reports_by_quarter: Dict[str, int] = field(default_factory=dict)
    reports_with_reaction: int = 0
    reports_with_serious_outcome: int = 0

    report_date_column: str = REPORT_DATE_COLUMN
    fallback_date_column: str = FALLBACK_DATE_COLUMN
    reports_with_report_date: int = 0
    reports_using_fallback_date: int = 0
    reports_missing_date: int = 0
    monthly_reports: List[Tuple[str, int]] = field(default_factory=list)
    date_range: Tuple[str, str] = ("", "")

    reaction_top_by_quarter: List[Tuple[str, str, int]] = field(default_factory=list)
    #: (drug, first_quarter_reports, last_quarter_reports, delta, pct_change)
    drug_frequency: List[Tuple[str, int, int, int, Optional[float]]] = field(
        default_factory=list
    )
    drug_total_top: List[Tuple[str, int]] = field(default_factory=list)

    # --- Part B: demographic ------------------------------------------------
    age_counts: List[Tuple[str, int]] = field(default_factory=list)
    sex_counts: List[Tuple[str, int]] = field(default_factory=list)
    age_by_quarter: List[Tuple[str, str, int]] = field(default_factory=list)
    sex_by_quarter: List[Tuple[str, str, int]] = field(default_factory=list)
    age_missing: int = 0
    sex_missing: int = 0
    age_unknown_reports: int = 0
    sex_unknown_reports: int = 0

    reactions_by_age: List[Tuple[str, str, int]] = field(default_factory=list)
    reactions_by_sex: List[Tuple[str, str, int]] = field(default_factory=list)
    age_by_sex: List[Tuple[str, str, int]] = field(default_factory=list)

    outcome_by_quarter: List[Tuple[str, str, str, int]] = field(default_factory=list)
    #: (bucket, serious_reports, total_reports, share_of_reports)
    serious_by_age: List[Tuple[str, int, int, float]] = field(default_factory=list)
    serious_by_sex: List[Tuple[str, int, int, float]] = field(default_factory=list)

    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # -- convenience ---------------------------------------------------------
    @property
    def quarter_labels(self) -> List[str]:
        return list(self.quarters)

    def as_summary(self, generated_at: str) -> Dict[str, Any]:
        """Bounded JSON payload (never the whole monthly series when truncated)."""
        return {
            "generated_at": generated_at,
            "quarters": self.quarters,
            "disclaimer": TEMPORAL_DEMOGRAPHIC_DISCLAIMER,
            "grain": ANALYSIS_GRAIN,
            "temporal": {
                "total_reports": self.total_reports,
                "reports_by_quarter": self.reports_by_quarter,
                "reaction_pairs_by_quarter": self.reaction_pairs_by_quarter,
                "serious_reports_by_quarter": self.serious_reports_by_quarter,
                "reports_with_reaction": self.reports_with_reaction,
                "reports_with_serious_outcome": self.reports_with_serious_outcome,
                "report_date_column": self.report_date_column,
                "fallback_date_column": self.fallback_date_column,
                "reports_with_report_date": self.reports_with_report_date,
                "reports_using_fallback_date": self.reports_using_fallback_date,
                "reports_missing_date": self.reports_missing_date,
                "months_with_data": len(self.monthly_reports),
                "date_range": list(self.date_range),
                "top_reactions": [
                    {"quarter": q, "reaction": pt, "reports": c}
                    for q, pt, c in self.reaction_top_by_quarter[:TOP_REACTIONS * len(self.quarters)]
                ],
                "top_drugs_by_reports": [
                    {"drug": name, "reports": c} for name, c in self.drug_total_top
                ],
            },
            "demographic": {
                "age": [
                    {"bucket": bucket, "reports": c} for bucket, c in self.age_counts
                ],
                "sex": [{"sex": sex, "reports": c} for sex, c in self.sex_counts],
                "age_missing": self.age_missing,
                "sex_missing": self.sex_missing,
                "serious_by_age": [
                    {"bucket": b, "serious_reports": s, "total_reports": t, "share": sh}
                    for b, s, t, sh in self.serious_by_age
                ],
                "serious_by_sex": [
                    {"sex": sex, "serious_reports": s, "total_reports": t, "share": sh}
                    for sex, s, t, sh in self.serious_by_sex
                ],
            },
            "checks": [
                {"name": n, "passed": ok, "detail": d} for n, ok, d in self.checks
            ],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Transformation helpers (built-in Spark functions only - no UDFs)
# ---------------------------------------------------------------------------
def _age_bucket() -> "F.Column":
    """Bin ``AGE_YEARS`` into the shared AGE_BUCKET_ORDER buckets."""
    age = F.col("AGE_YEARS")
    return (
        F.when(age.isNull(), F.lit("Unknown"))
        .when(age < F.lit(18), F.lit("0-17"))
        .when(age < F.lit(45), F.lit("18-44"))
        .when(age < F.lit(65), F.lit("45-64"))
        .when(age < F.lit(75), F.lit("65-74"))
        .otherwise(F.lit("75+"))
    )


def _sex_category() -> "F.Column":
    """Map ``SEX_NORM`` to a presentation category (NULL -> 'Unknown')."""
    return F.coalesce(F.col("SEX_NORM"), F.lit("Unknown"))


def _count_rows() -> "F.Column":
    """A portable row counter."""
    return F.count(F.lit(1))


def _ordered(items: Sequence[Tuple[str, int]], order: Sequence[str]) -> List[Tuple[str, int]]:
    """Sort ``(label, count)`` pairs by a fixed label order, unknown labels last."""
    position = {label: index for index, label in enumerate(order)}
    return sorted(items, key=lambda pair: (position.get(pair[0], len(order)), pair[0]))


def _clip(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` characters, adding an ellipsis when cut."""
    return text if len(text) <= width else text[: width - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def _clean_inputs(
    spark: Any,
    *,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> Tuple[Dict[str, DataFrame], Dict[str, TableCleaning], List[str]]:
    """Read + clean the four required tables via the preprocessing module."""
    log = get_logger(__name__)
    frames: Dict[str, DataFrame] = {}
    cleanings: Dict[str, TableCleaning] = {}
    errors: List[str] = []

    for table in ANALYSIS_TABLES:
        try:
            frame, cleaning = clean_table(spark, table, quarters, settings, quarter_dirs)
        except (CleaningError, IngestionError) as exc:
            log.error("%s: cleaning failed: %s", table, exc)
            errors.append(f"[{table}] {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - one bad table must not abort the run
            log.error("%s: unexpected failure: %s", table, exc)
            errors.append(f"[{table}] {type(exc).__name__}: {exc}")
            continue
        frames[table] = frame
        cleanings[table] = cleaning
        log.info(
            "%s: %s rows cleaned (%s -> %s columns)",
            table,
            f"{cleaning.rows_after:,}",
            cleaning.columns_before,
            cleaning.columns_after,
        )
    return frames, cleanings, errors


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(
    spark: Any,
    *,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> TemporalDemographicResults:
    """Run every temporal + demographic aggregation and collect the results."""
    log = get_logger(__name__)
    results = TemporalDemographicResults(quarters=list(quarters))

    frames, cleanings, errors = _clean_inputs(
        spark, quarters=quarters, settings=settings, quarter_dirs=quarter_dirs
    )
    results.cleanings = cleanings
    results.errors.extend(errors)

    missing = [table for table in ANALYSIS_TABLES if table not in frames]
    if missing:
        raise TemporalDemographicError(
            "required table(s) unavailable: " + ", ".join(missing) + "; see errors"
        )

    # --- report grain: one row per (quarter, PRIMARYID) with demographics ------
    reports = (
        frames["DEMO"]
        .select(
            QUARTER_COLUMN,
            "PRIMARYID",
            "AGE_YEARS",
            "SEX_NORM",
            REPORT_DATE_COLUMN,
            FALLBACK_DATE_COLUMN,
        )
        .filter(F.col("PRIMARYID").isNotNull())
        .dropDuplicates([QUARTER_COLUMN, "PRIMARYID"])
        .withColumn("age_bucket", _age_bucket())
        .withColumn("sex_category", _sex_category())
        .persist()
    )
    report_demog = reports.select(
        QUARTER_COLUMN, "PRIMARYID", "age_bucket", "sex_category"
    ).persist()
    pairs = (
        frames["REAC"]
        .select(QUARTER_COLUMN, "PRIMARYID", "PT")
        .filter(F.col("PT").isNotNull())
        .distinct()
        .persist()
    )
    outcomes = (
        frames["OUTC"]
        .select(QUARTER_COLUMN, "PRIMARYID", "OUTC_COD")
        .filter(F.col("OUTC_COD").isNotNull())
        .distinct()
        .persist()
    )
    drug_pairs = (
        frames["DRUG"]
        .select(QUARTER_COLUMN, "PRIMARYID", "DRUGNAME_STD")
        .filter(F.col("DRUGNAME_STD").isNotNull())
        .distinct()
        .persist()
    )

    try:
        _temporal(reports, pairs, outcomes, drug_pairs, results)
        _demographic(reports, report_demog, pairs, outcomes, results)
    finally:
        for frame in (reports, report_demog, pairs, outcomes, drug_pairs):
            frame.unpersist()

    log.info(
        "reports=%s  monthly points=%s  missing date=%s",
        f"{results.total_reports:,}",
        len(results.monthly_reports),
        f"{results.reports_missing_date:,}",
    )

    results.checks = _validate(results)
    return results


def _temporal(
    reports: DataFrame,
    pairs: DataFrame,
    outcomes: DataFrame,
    drug_pairs: DataFrame,
    results: TemporalDemographicResults,
) -> None:
    """Part A - temporal aggregations."""
    # ---- A1. reporting volumes + serious volume by quarter -----------------
    results.total_reports = reports.count()
    results.reports_by_quarter = {
        row[QUARTER_COLUMN]: int(row["reports"])
        for row in reports.groupBy(QUARTER_COLUMN).agg(_count_rows().alias("reports")).collect()
    }
    results.reaction_pairs_by_quarter = {
        row[QUARTER_COLUMN]: int(row["pairs"])
        for row in pairs.groupBy(QUARTER_COLUMN).agg(_count_rows().alias("pairs")).collect()
    }
    results.reports_with_reaction_by_quarter = {
        row[QUARTER_COLUMN]: int(row["reports"])
        for row in pairs.select(QUARTER_COLUMN, "PRIMARYID")
        .distinct()
        .groupBy(QUARTER_COLUMN)
        .agg(_count_rows().alias("reports"))
        .collect()
    }
    serious_reports = outcomes.select(QUARTER_COLUMN, "PRIMARYID").distinct().persist()
    try:
        results.reports_with_serious_outcome = serious_reports.count()
        results.serious_reports_by_quarter = {
            row[QUARTER_COLUMN]: int(row["reports"])
            for row in serious_reports.groupBy(QUARTER_COLUMN)
            .agg(_count_rows().alias("reports"))
            .collect()
        }
    finally:
        serious_reports.unpersist()
    results.reports_with_reaction = sum(results.reports_with_reaction_by_quarter.values())

    # ---- A2. monthly reporting trend from the report date ------------------
    dated = reports.withColumn(
        "report_date", F.coalesce(F.col(REPORT_DATE_COLUMN), F.col(FALLBACK_DATE_COLUMN))
    ).withColumn("report_month", F.date_format(F.col("report_date"), MONTH_LABEL_FORMAT))

    results.reports_with_report_date = reports.filter(
        F.col(REPORT_DATE_COLUMN).isNotNull()
    ).count()
    results.reports_missing_date = dated.filter(F.col("report_date").isNull()).count()
    results.reports_using_fallback_date = (
        results.total_reports
        - results.reports_with_report_date
        - results.reports_missing_date
    )
    monthly = (
        dated.filter(F.col("report_month").isNotNull())
        .groupBy("report_month")
        .agg(_count_rows().alias("reports"))
        .collect()
    )
    results.monthly_reports = sorted(
        ((row["report_month"], int(row["reports"])) for row in monthly),
        key=lambda pair: pair[0],
    )
    if results.monthly_reports:
        results.date_range = (
            results.monthly_reports[0][0],
            results.monthly_reports[-1][0],
        )

    # ---- A3. most frequent reactions by quarter ----------------------------
    per_quarter = pairs.groupBy("PT", QUARTER_COLUMN).agg(_count_rows().alias("reports"))
    rank = Window.partitionBy(QUARTER_COLUMN).orderBy(F.desc("reports"), F.asc("PT"))
    ranked = (
        per_quarter.withColumn("rank", F.row_number().over(rank))
        .filter(F.col("rank") <= F.lit(TOP_REACTIONS))
        .select(QUARTER_COLUMN, "PT", "reports")
    )
    results.reaction_top_by_quarter = sorted(
        (
            (row[QUARTER_COLUMN], row["PT"], int(row["reports"]))
            for row in ranked.collect()
        ),
        key=lambda item: (item[0], -item[2], item[1]),
    )

    # ---- A4. drug reporting frequency change --------------------------------
    drug_quarter = drug_pairs.groupBy("DRUGNAME_STD", QUARTER_COLUMN).agg(
        _count_rows().alias("reports")
    )
    totals = drug_pairs.groupBy("DRUGNAME_STD").agg(_count_rows().alias("reports"))
    top_rows = totals.orderBy(F.desc("reports"), F.asc("DRUGNAME_STD")).limit(TOP_DRUGS).collect()
    results.drug_total_top = [(row["DRUGNAME_STD"], int(row["reports"])) for row in top_rows]
    top_names = [name for name, _count in results.drug_total_top]

    counts: Dict[str, Dict[str, int]] = {}
    for row in drug_quarter.filter(F.col("DRUGNAME_STD").isin(top_names)).collect():
        counts.setdefault(row["DRUGNAME_STD"], {})[row[QUARTER_COLUMN]] = int(row["reports"])

    first_q, last_q = results.quarters[0], results.quarters[-1]
    for name in top_names:
        by_quarter = counts.get(name, {})
        first_count = by_quarter.get(first_q, 0)
        last_count = by_quarter.get(last_q, 0)
        delta = last_count - first_count
        pct = (100.0 * delta / first_count) if first_count else None
        results.drug_frequency.append((name, first_count, last_count, delta, pct))


def _demographic(
    reports: DataFrame,
    report_demog: DataFrame,
    pairs: DataFrame,
    outcomes: DataFrame,
    results: TemporalDemographicResults,
) -> None:
    """Part B - demographic aggregations."""
    # ---- B1/B2. age + gender distribution ----------------------------------
    age_rows = reports.groupBy("age_bucket").agg(_count_rows().alias("reports")).collect()
    results.age_counts = _ordered(
        [(row["age_bucket"], int(row["reports"])) for row in age_rows], AGE_BUCKET_ORDER
    )
    sex_rows = reports.groupBy("sex_category").agg(_count_rows().alias("reports")).collect()
    results.sex_counts = _ordered(
        [(row["sex_category"], int(row["reports"])) for row in sex_rows], SEX_ORDER
    )
    results.age_missing = reports.filter(F.col("AGE_YEARS").isNull()).count()
    results.sex_missing = reports.filter(F.col("SEX_NORM").isNull()).count()
    results.age_unknown_reports = sum(
        count for bucket, count in results.age_counts if bucket == "Unknown"
    )
    results.sex_unknown_reports = sum(
        count for sex, count in results.sex_counts if sex == "Unknown"
    )

    results.age_by_quarter = sorted(
        (
            (row[QUARTER_COLUMN], row["age_bucket"], int(row["reports"]))
            for row in reports.groupBy(QUARTER_COLUMN, "age_bucket")
            .agg(_count_rows().alias("reports"))
            .collect()
        ),
        key=lambda item: (
            item[0],
            AGE_BUCKET_ORDER.index(item[1]) if item[1] in AGE_BUCKET_ORDER else 99,
        ),
    )
    results.sex_by_quarter = sorted(
        (
            (row[QUARTER_COLUMN], row["sex_category"], int(row["reports"]))
            for row in reports.groupBy(QUARTER_COLUMN, "sex_category")
            .agg(_count_rows().alias("reports"))
            .collect()
        ),
        key=lambda item: (
            item[0],
            SEX_ORDER.index(item[1]) if item[1] in SEX_ORDER else 99,
        ),
    )

    # ---- B3/B4. reactions across age groups / genders ----------------------
    by_age = pairs.join(report_demog, on=[QUARTER_COLUMN, "PRIMARYID"], how="inner")
    per_age = by_age.groupBy("age_bucket", "PT").agg(_count_rows().alias("reports"))
    age_rank = Window.partitionBy("age_bucket").orderBy(F.desc("reports"), F.asc("PT"))
    ranked_age = (
        per_age.withColumn("rank", F.row_number().over(age_rank))
        .filter(F.col("rank") <= F.lit(TOP_REACTIONS_PER_GROUP))
        .select("age_bucket", "PT", "reports")
    )
    results.reactions_by_age = sorted(
        ((row["age_bucket"], row["PT"], int(row["reports"])) for row in ranked_age.collect()),
        key=lambda item: (
            AGE_BUCKET_ORDER.index(item[0]) if item[0] in AGE_BUCKET_ORDER else 99,
            -item[2],
            item[1],
        ),
    )

    per_sex = by_age.groupBy("sex_category", "PT").agg(_count_rows().alias("reports"))
    sex_rank = Window.partitionBy("sex_category").orderBy(F.desc("reports"), F.asc("PT"))
    ranked_sex = (
        per_sex.withColumn("rank", F.row_number().over(sex_rank))
        .filter(F.col("rank") <= F.lit(TOP_REACTIONS_PER_GROUP))
        .select("sex_category", "PT", "reports")
    )
    results.reactions_by_sex = sorted(
        ((row["sex_category"], row["PT"], int(row["reports"])) for row in ranked_sex.collect()),
        key=lambda item: (
            SEX_ORDER.index(item[0]) if item[0] in SEX_ORDER else 99,
            -item[2],
            item[1],
        ),
    )

    results.age_by_sex = sorted(
        (
            (row["sex_category"], row["age_bucket"], int(row["reports"]))
            for row in reports.groupBy("sex_category", "age_bucket")
            .agg(_count_rows().alias("reports"))
            .collect()
        ),
        key=lambda item: (
            SEX_ORDER.index(item[0]) if item[0] in SEX_ORDER else 99,
            AGE_BUCKET_ORDER.index(item[1]) if item[1] in AGE_BUCKET_ORDER else 99,
        ),
    )

    # ---- B5. serious outcomes by demographic group -------------------------
    results.outcome_by_quarter = sorted(
        (
            (
                row[QUARTER_COLUMN],
                row["OUTC_COD"],
                OUTC_CODE_LABELS.get(row["OUTC_COD"], "Unmapped code"),
                int(row["reports"]),
            )
            for row in outcomes.groupBy(QUARTER_COLUMN, "OUTC_COD")
            .agg(_count_rows().alias("reports"))
            .collect()
        ),
        key=lambda item: (item[0], -item[3], item[1]),
    )

    serious_reports = outcomes.select(QUARTER_COLUMN, "PRIMARYID").distinct()
    serious_joined = serious_reports.join(
        report_demog, on=[QUARTER_COLUMN, "PRIMARYID"], how="inner"
    )
    serious_age = {
        row["age_bucket"]: int(row["reports"])
        for row in serious_joined.groupBy("age_bucket").agg(_count_rows().alias("reports")).collect()
    }
    serious_sex = {
        row["sex_category"]: int(row["reports"])
        for row in serious_joined.groupBy("sex_category").agg(_count_rows().alias("reports")).collect()
    }
    results.serious_by_age = [
        (bucket, serious_age.get(bucket, 0), total, _share(serious_age.get(bucket, 0), total))
        for bucket, total in results.age_counts
    ]
    results.serious_by_sex = [
        (sex, serious_sex.get(sex, 0), total, _share(serious_sex.get(sex, 0), total))
        for sex, total in results.sex_counts
    ]


def _share(part: int, whole: int) -> float:
    """Percentage share of ``part`` in ``whole`` (0.0 when the whole is 0)."""
    return round(100.0 * part / whole, 4) if whole else 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(results: TemporalDemographicResults) -> List[Tuple[str, bool, str]]:
    """Cross-check the aggregates used by the tables and every figure."""
    checks: List[Tuple[str, bool, str]] = []

    cleaning_ok = all(
        item.rows_before == item.rows_after and not item.errors
        for item in results.cleanings.values()
    )
    checks.append(
        (
            "cleaning removed no rows (DEMO/DRUG/REAC/OUTC)",
            cleaning_ok,
            "; ".join(
                f"{table}={item.rows_before:,}->{item.rows_after:,}"
                for table, item in results.cleanings.items()
            ),
        )
    )

    quarter_sum = sum(results.reports_by_quarter.values())
    checks.append(
        (
            "quarterly report counts sum to the total distinct reports",
            quarter_sum == results.total_reports and results.total_reports > 0,
            f"{quarter_sum:,} vs {results.total_reports:,}",
        )
    )
    checks.append(
        (
            "Q1 and Q2 are both present with positive report counts",
            set(results.reports_by_quarter) == set(results.quarters)
            and all(value > 0 for value in results.reports_by_quarter.values()),
            ", ".join(f"{q}={n:,}" for q, n in sorted(results.reports_by_quarter.items())),
        )
    )
    checks.append(
        (
            "reports carrying a reaction match the report grain in every quarter",
            results.reports_with_reaction_by_quarter == results.reports_by_quarter,
            "; ".join(
                f"{q}={results.reports_with_reaction_by_quarter.get(q, 0):,}"
                for q in results.quarters
            ),
        )
    )

    monthly_sum = sum(count for _month, count in results.monthly_reports)
    checks.append(
        (
            "monthly counts plus undated reports equal the total reports",
            monthly_sum + results.reports_missing_date == results.total_reports,
            f"dated {monthly_sum:,} + undated {results.reports_missing_date:,} "
            f"vs total {results.total_reports:,}",
        )
    )
    checks.append(
        (
            "monthly report months are strictly increasing and unique",
            all(
                results.monthly_reports[i][0] < results.monthly_reports[i + 1][0]
                for i in range(len(results.monthly_reports) - 1)
            ),
            f"{len(results.monthly_reports)} month(s), range "
            f"{results.date_range[0] or 'n/a'}..{results.date_range[1] or 'n/a'}",
        )
    )
    checks.append(
        (
            "report-date coverage is sufficient for a monthly trend",
            results.reports_with_report_date > 0 and len(results.monthly_reports) > 0,
            f"{results.reports_with_report_date:,} with REPT_DT, "
            f"{results.reports_using_fallback_date:,} using FDA_DT fallback, "
            f"{results.reports_missing_date:,} with no date",
        )
    )

    age_sum = sum(count for _bucket, count in results.age_counts)
    checks.append(
        (
            "age buckets cover every report",
            age_sum == results.total_reports,
            f"{age_sum:,} vs {results.total_reports:,}",
        )
    )
    sex_sum = sum(count for _sex, count in results.sex_counts)
    checks.append(
        (
            "gender categories cover every report",
            sex_sum == results.total_reports,
            f"{sex_sum:,} vs {results.total_reports:,}",
        )
    )
    checks.append(
        (
            "missing age/gender are reported as an explicit Unknown category",
            results.age_unknown_reports == results.age_missing
            and results.sex_unknown_reports == results.sex_missing,
            f"age Unknown={results.age_unknown_reports:,} vs null AGE_YEARS="
            f"{results.age_missing:,}; gender Unknown={results.sex_unknown_reports:,} "
            f"vs null SEX_NORM={results.sex_missing:,}",
        )
    )

    serious_age_sum = sum(serious for _b, serious, _t, _sh in results.serious_by_age)
    serious_sex_sum = sum(serious for _s, serious, _t, _sh in results.serious_by_sex)
    checks.append(
        (
            "serious reports per age group sum to the total serious reports",
            serious_age_sum == results.reports_with_serious_outcome,
            f"{serious_age_sum:,} vs {results.reports_with_serious_outcome:,}",
        )
    )
    checks.append(
        (
            "serious reports per gender sum to the total serious reports",
            serious_sex_sum == results.reports_with_serious_outcome,
            f"{serious_sex_sum:,} vs {results.reports_with_serious_outcome:,}",
        )
    )
    checks.append(
        (
            "serious reporting share never exceeds 100% in any group",
            all(0.0 <= share <= 100.0 for _b, _s, _t, share in results.serious_by_age)
            and all(0.0 <= share <= 100.0 for _s, _r, _t, share in results.serious_by_sex),
            "all age and gender group shares within [0, 100]",
        )
    )

    # --- chart-data checks: every figure's series matches the source lists ----
    charts_ok = (
        len(results.reports_by_quarter) == len(results.quarters)
        and len([q for q, _pt, _c in results.reaction_top_by_quarter]) > 0
        and len(results.age_counts) > 0
        and len(results.sex_counts) > 0
        and len(results.drug_frequency) == len(results.drug_total_top)
    )
    checks.append(
        (
            "chart inputs are populated and consistent with the source aggregations",
            charts_ok,
            f"quarters={len(results.reports_by_quarter)}, "
            f"reaction rows={len(results.reaction_top_by_quarter)}, "
            f"age buckets={len(results.age_counts)}, gender={len(results.sex_counts)}, "
            f"drug-change rows={len(results.drug_frequency)}",
        )
    )
    checks.append(
        (
            "top-reaction-per-quarter counts never exceed the quarter pair total",
            all(
                count <= results.reaction_pairs_by_quarter.get(quarter, 0)
                for quarter, _pt, count in results.reaction_top_by_quarter
            ),
            f"{len(results.reaction_top_by_quarter)} ranked (quarter, reaction) row(s)",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Plotly figures (UI-agnostic: each returns a go.Figure, no Streamlit)
# ---------------------------------------------------------------------------
def quarterly_comparison_figure(
    results: TemporalDemographicResults,
    *,
    theme: str = "plotly_white",
    title: Optional[str] = None,
) -> go.Figure:
    """Q1 vs Q2 comparison: reports, reaction pairs and serious reports."""
    quarters = results.quarters
    figure = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Reports (PRIMARYID)",
            "Distinct report-reaction pairs",
            "Reports with a serious outcome",
        ),
    )
    series = (
        ("reports", [results.reports_by_quarter.get(q, 0) for q in quarters], "#1f77b4", 1),
        (
            "reaction pairs",
            [results.reaction_pairs_by_quarter.get(q, 0) for q in quarters],
            "#ff7f0e",
            2,
        ),
        (
            "serious reports",
            [results.serious_reports_by_quarter.get(q, 0) for q in quarters],
            "#d62728",
            3,
        ),
    )
    for name, values, color, col in series:
        figure.add_trace(
            go.Bar(
                x=quarters,
                y=values,
                name=name,
                marker=dict(color=color),
                text=[f"{value:,}" for value in values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>" + name + ": %{y:,}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    figure.update_layout(
        template=theme,
        title=title or "Reporting volume by quarter (2026 Q1 vs Q2)",
        showlegend=False,
        margin=dict(l=10, r=30, t=70, b=10),
        height=460,
    )
    figure.update_yaxes(title_text="count")
    return figure


def monthly_trend_figure(
    results: TemporalDemographicResults,
    *,
    theme: str = "plotly_white",
    title: Optional[str] = None,
) -> go.Figure:
    """Monthly reporting trend from the report dates (no trend when undated)."""
    figure = go.Figure()
    if not results.monthly_reports:
        figure.update_layout(
            template=theme,
            title=title or "Monthly reporting trend - no report date available",
            annotations=[
                dict(
                    text="No report carried a usable report date, so no monthly trend is drawn.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14),
                )
            ],
        )
        return figure

    months = [month for month, _count in results.monthly_reports]
    counts = [count for _month, count in results.monthly_reports]
    figure.add_trace(
        go.Scatter(
            x=months,
            y=counts,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
            name="reports",
            hovertemplate="<b>%{x}</b><br>reports: %{y:,}<extra></extra>",
        )
    )
    figure.update_layout(
        template=theme,
        title=title
        or (
            "Monthly reporting trend by report date "
            f"({results.report_date_column} / {results.fallback_date_column})"
        ),
        xaxis_title="report month (yyyy-MM)",
        yaxis_title="reports",
        hovermode="x unified",
        margin=dict(l=10, r=20, t=70, b=40),
        height=480,
    )
    return figure


def age_distribution_figure(
    results: TemporalDemographicResults,
    *,
    theme: str = "plotly_white",
    title: Optional[str] = None,
) -> go.Figure:
    """Reports per age group (Unknown included)."""
    buckets = [bucket for bucket, _count in results.age_counts]
    counts = [count for _bucket, count in results.age_counts]
    total = results.total_reports or 1
    colors = ["#9aa4b2" if bucket == "Unknown" else "#2ca02c" for bucket in buckets]
    figure = go.Figure(
        go.Bar(
            x=buckets,
            y=counts,
            marker=dict(color=colors),
            text=[f"{100.0 * c / total:.1f}%" for c in counts],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>reports: %{y:,}<extra></extra>",
        )
    )
    figure.update_layout(
        template=theme,
        title=title or "Report volume by age group",
        xaxis_title="age group (years)",
        yaxis_title="reports",
        showlegend=False,
        margin=dict(l=10, r=20, t=70, b=40),
        height=460,
    )
    return figure


def gender_distribution_figure(
    results: TemporalDemographicResults,
    *,
    theme: str = "plotly_white",
    title: Optional[str] = None,
) -> go.Figure:
    """Reports per gender category (Unknown included)."""
    sexes = [sex for sex, _count in results.sex_counts]
    counts = [count for _sex, count in results.sex_counts]
    total = results.total_reports or 1
    colors = ["#9aa4b2" if sex == "Unknown" else "#1f77b4" for sex in sexes]
    figure = go.Figure(
        go.Bar(
            x=sexes,
            y=counts,
            marker=dict(color=colors),
            text=[f"{100.0 * c / total:.1f}%" for c in counts],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>reports: %{y:,}<extra></extra>",
        )
    )
    figure.update_layout(
        template=theme,
        title=title or "Report volume by gender category",
        xaxis_title="gender (M/F/U = reported but neither; Unknown = not reported)",
        yaxis_title="reports",
        showlegend=False,
        margin=dict(l=10, r=20, t=70, b=40),
        height=460,
    )
    return figure


def top_reactions_figure(
    results: TemporalDemographicResults,
    *,
    theme: str = "plotly_white",
    top_n: int = TOP_REACTIONS,
    title: Optional[str] = None,
) -> go.Figure:
    """Top reactions overall, each split into its Q1 and Q2 counts."""
    overall: Dict[str, int] = {}
    per_quarter: Dict[Tuple[str, str], int] = {}
    for quarter, pt, count in results.reaction_top_by_quarter:
        overall[pt] = overall.get(pt, 0) + count
        per_quarter[(quarter, pt)] = count
    ranked = sorted(overall.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    reactions = [pt for pt, _count in ranked][::-1]

    figure = go.Figure()
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    for index, quarter in enumerate(results.quarters):
        figure.add_trace(
            go.Bar(
                y=[_clip(pt, 40) for pt in reactions],
                x=[per_quarter.get((quarter, pt), 0) for pt in reactions],
                orientation="h",
                name=quarter,
                marker=dict(color=palette[index % len(palette)]),
                hovertemplate="<b>%{y}</b><br>" + quarter + ": %{x:,}<extra></extra>",
            )
        )
    figure.update_layout(
        template=theme,
        title=title or f"Top {len(reactions)} reactions by quarter",
        barmode="group",
        xaxis_title="reports",
        yaxis_title="reaction (MedDRA PT)",
        legend=dict(title="quarter"),
        height=max(460, 22 * max(len(reactions), 1) + 140),
        margin=dict(l=10, r=20, t=70, b=20),
    )
    return figure


def serious_outcome_figure(
    results: TemporalDemographicResults,
    *,
    theme: str = "plotly_white",
    title: Optional[str] = None,
) -> go.Figure:
    """Serious-outcome report counts by age group and by gender."""
    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.16,
        subplot_titles=(
            "Reports with a serious outcome by age group",
            "Reports with a serious outcome by gender",
        ),
    )
    age_buckets = [bucket for bucket, _s, _t, _sh in results.serious_by_age]
    age_serious = [serious for _b, serious, _t, _sh in results.serious_by_age]
    age_share = [share for _b, _s, _t, share in results.serious_by_age]
    figure.add_trace(
        go.Bar(
            x=age_buckets,
            y=age_serious,
            marker=dict(color="#d62728"),
            customdata=[[share] for share in age_share],
            text=[f"{value:,}" for value in age_serious],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>serious reports: %{y:,}<br>"
            "share of group: %{customdata[0]:.2f}%<extra></extra>",
            name="serious reports",
        ),
        row=1,
        col=1,
    )
    sexes = [sex for sex, _s, _t, _sh in results.serious_by_sex]
    sex_serious = [serious for _s, serious, _t, _sh in results.serious_by_sex]
    sex_share = [share for _s, _r, _t, share in results.serious_by_sex]
    figure.add_trace(
        go.Bar(
            x=sexes,
            y=sex_serious,
            marker=dict(color="#9467bd"),
            customdata=[[share] for share in sex_share],
            text=[f"{value:,}" for value in sex_serious],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>serious reports: %{y:,}<br>"
            "share of group: %{customdata[0]:.2f}%<extra></extra>",
            name="serious reports",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        template=theme,
        title=title or "Serious outcomes by demographic group (a report may carry several)",
        showlegend=False,
        margin=dict(l=10, r=20, t=70, b=40),
        height=470,
    )
    figure.update_yaxes(title_text="reports", row=1, col=1)
    return figure


def drug_frequency_change_figure(
    results: TemporalDemographicResults,
    *,
    theme: str = "plotly_white",
    top_n: int = TOP_DRUGS,
    title: Optional[str] = None,
) -> go.Figure:
    """Drug reporting frequency: quarter volumes and the Q-to-Q change."""
    rows = results.drug_frequency[:top_n]
    drugs = [name for name, _f, _l, _d, _p in rows][::-1]
    first_q = results.quarters[0] if results.quarters else "Q1"
    last_q = results.quarters[-1] if results.quarters else "Q2"
    first_counts = [first for _n, first, _l, _d, _p in rows][::-1]
    last_counts = [last for _n, _f, last, _d, _p in rows][::-1]
    deltas = [delta for _n, _f, _l, delta, _p in rows][::-1]

    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.16,
        subplot_titles=(
            f"Reports per quarter ({first_q} vs {last_q})",
            f"Change in reports ({last_q} - {first_q})",
        ),
    )
    figure.add_trace(
        go.Bar(
            y=[_clip(name, 24) for name in drugs],
            x=first_counts,
            orientation="h",
            name=first_q,
            marker=dict(color="#1f77b4"),
            hovertemplate="<b>%{y}</b><br>" + first_q + ": %{x:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            y=[_clip(name, 24) for name in drugs],
            x=last_counts,
            orientation="h",
            name=last_q,
            marker=dict(color="#ff7f0e"),
            hovertemplate="<b>%{y}</b><br>" + last_q + ": %{x:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            y=[_clip(name, 24) for name in drugs],
            x=deltas,
            orientation="h",
            name="change",
            marker=dict(
                color=["#2ca02c" if delta >= 0 else "#d62728" for delta in deltas]
            ),
            hovertemplate="<b>%{y}</b><br>change: %{x:,}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        template=theme,
        title=title or f"Drug reporting frequency change (top {len(rows)} drugs)",
        barmode="group",
        legend=dict(title="quarter"),
        height=max(500, 22 * max(len(drugs), 1) + 160),
        margin=dict(l=10, r=20, t=70, b=20),
    )
    figure.update_xaxes(title_text="reports", row=1, col=1)
    figure.update_xaxes(title_text="change in reports", row=1, col=2)
    return figure


# ---------------------------------------------------------------------------
# Output writers (Python standard library + Plotly only)
# ---------------------------------------------------------------------------
def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_html(figure: go.Figure, path: Path) -> None:
    """Write a self-contained interactive HTML page (Plotly only, no pandas)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs=True, full_html=True)


#: Figure name -> ("file name", builder) used by write_outputs and the tests.
FIGURE_BUILDERS: Mapping[str, Any] = {
    "quarterly_comparison": quarterly_comparison_figure,
    "monthly_trend": monthly_trend_figure,
    "age_distribution": age_distribution_figure,
    "gender_distribution": gender_distribution_figure,
    "top_reactions": top_reactions_figure,
    "serious_outcomes": serious_outcome_figure,
    "drug_frequency_change": drug_frequency_change_figure,
}


def write_outputs(
    results: TemporalDemographicResults,
    *,
    visualization_dir: Path,
    analytics_dir: Path,
    theme: str = "plotly_white",
    top_n: int = TOP_REACTIONS,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Path]:
    """Persist every figure (HTML) and table (CSV/JSON); return the paths."""
    timestamp = (generated_at or datetime.now()).isoformat(timespec="seconds")
    visualization_dir.mkdir(parents=True, exist_ok=True)
    analytics_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    # --- figures ------------------------------------------------------------
    for name, builder in FIGURE_BUILDERS.items():
        path = visualization_dir / f"{name}.html"
        figure = builder(results, theme=theme, top_n=top_n) if name in (
            "top_reactions",
            "drug_frequency_change",
        ) else builder(results, theme=theme)
        _write_html(figure, path)
        written[path.name] = path

    # --- tables -------------------------------------------------------------
    total = results.total_reports or 1

    def emit(name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        target = analytics_dir / name
        _write_csv(target, header, rows)
        written[name] = target

    emit(
        "temporal_reports_by_quarter.csv",
        ["quarter", "reports", "reaction_pairs", "reports_with_reaction", "serious_reports"],
        [
            [
                quarter,
                results.reports_by_quarter.get(quarter, 0),
                results.reaction_pairs_by_quarter.get(quarter, 0),
                results.reports_with_reaction_by_quarter.get(quarter, 0),
                results.serious_reports_by_quarter.get(quarter, 0),
            ]
            for quarter in results.quarters
        ],
    )
    emit(
        "temporal_monthly_trend.csv",
        ["report_month", "reports"],
        [[month, count] for month, count in results.monthly_reports],
    )
    emit(
        "temporal_top_reactions_by_quarter.csv",
        ["quarter", "reaction_pt", "reports"],
        [[quarter, pt, count] for quarter, pt, count in results.reaction_top_by_quarter],
    )
    emit(
        "temporal_drug_frequency_change.csv",
        ["drug_name", f"{results.quarters[0]}_reports", f"{results.quarters[-1]}_reports",
         "delta", "pct_change"],
        [
            [name, first, last, delta, "" if pct is None else round(pct, 4)]
            for name, first, last, delta, pct in results.drug_frequency
        ],
    )
    emit(
        "demographic_age_distribution.csv",
        ["age_bucket", "reports", "pct_of_reports"],
        [[bucket, count, round(100.0 * count / total, 4)] for bucket, count in results.age_counts],
    )
    emit(
        "demographic_gender_distribution.csv",
        ["sex", "reports", "pct_of_reports"],
        [[sex, count, round(100.0 * count / total, 4)] for sex, count in results.sex_counts],
    )
    emit(
        "demographic_age_by_quarter.csv",
        ["quarter", "age_bucket", "reports"],
        [[quarter, bucket, count] for quarter, bucket, count in results.age_by_quarter],
    )
    emit(
        "demographic_gender_by_quarter.csv",
        ["quarter", "sex", "reports"],
        [[quarter, sex, count] for quarter, sex, count in results.sex_by_quarter],
    )
    emit(
        "demographic_reactions_by_age.csv",
        ["age_bucket", "reaction_pt", "reports"],
        [[bucket, pt, count] for bucket, pt, count in results.reactions_by_age],
    )
    emit(
        "demographic_reactions_by_gender.csv",
        ["sex", "reaction_pt", "reports"],
        [[sex, pt, count] for sex, pt, count in results.reactions_by_sex],
    )
    emit(
        "demographic_age_by_gender.csv",
        ["sex", "age_bucket", "reports"],
        [[sex, bucket, count] for sex, bucket, count in results.age_by_sex],
    )
    emit(
        "demographic_serious_outcomes.csv",
        ["group_type", "group", "serious_reports", "total_reports", "share_pct"],
        [
            ["age", bucket, serious, tot, share]
            for bucket, serious, tot, share in results.serious_by_age
        ]
        + [
            ["gender", sex, serious, tot, share]
            for sex, serious, tot, share in results.serious_by_sex
        ],
    )
    emit(
        "temporal_serious_outcomes_by_quarter.csv",
        ["quarter", "outc_cod", "outcome_label", "reports"],
        [
            [quarter, code, label, count]
            for quarter, code, label, count in results.outcome_by_quarter
        ],
    )

    summary_path = analytics_dir / "temporal_demographic_summary.json"
    summary_path.write_text(
        json.dumps(results.as_summary(timestamp), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    written[summary_path.name] = summary_path
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(
    results: TemporalDemographicResults,
    *,
    generated_at: Optional[datetime] = None,
) -> str:
    """Render the temporal + demographic report as plain text."""
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    total = results.total_reports or 1
    pct = lambda count: f"{100.0 * count / total:5.2f}%"  # noqa: E731 - local formatting helper
    first_q, last_q = results.quarters[0], results.quarters[-1]

    lines: List[str] = [
        "=" * 78,
        "FAERS TEMPORAL AND DEMOGRAPHIC ANALYSIS REPORT",
        f"Generated : {stamp}",
        f"Quarters  : {', '.join(results.quarters)}",
        "Engine    : PySpark DataFrame API only (no pandas, no PyArrow, no UDFs)",
        "Rendering : Plotly figures (UI-agnostic; no Streamlit dependency)",
        "Source    : real FAERS 2026 Q1/Q2 ASCII extracts, via ingestion + preprocessing",
        "=" * 78,
        "",
        "0. SCOPE AND METHOD",
        "-" * 78,
        f"  {ANALYSIS_GRAIN}",
        "  Reports are connected by PRIMARYID only. Duplicate source rows are",
        "  removed only inside aggregations (distinct projections), never from the",
        "  data. The two quarters are grouped separately and never merged.",
        "",
        "  !! " + TEMPORAL_DEMOGRAPHIC_DISCLAIMER.replace(". ", ".\n  !! "),
        "",
        "PART A - TEMPORAL ANALYSIS",
        "-" * 78,
        f"A1. REPORTING VOLUME BY QUARTER (total reports {results.total_reports:,})",
        f"  {'QUARTER':<10}{'REPORTS':>12}{'REACTION PAIRS':>16}{'SERIOUS REPORTS':>18}",
    ]
    for quarter in results.quarters:
        lines.append(
            f"  {quarter:<10}{results.reports_by_quarter.get(quarter, 0):>12,}"
            f"{results.reaction_pairs_by_quarter.get(quarter, 0):>16,}"
            f"{results.serious_reports_by_quarter.get(quarter, 0):>18,}"
        )
    delta = results.reports_by_quarter.get(last_q, 0) - results.reports_by_quarter.get(first_q, 0)
    lines.append(
        f"  Change {first_q} -> {last_q}: {delta:+,} reports "
        f"({results.reports_by_quarter.get(first_q, 0):,} -> "
        f"{results.reports_by_quarter.get(last_q, 0):,})"
    )
    lines.append(
        f"  Reports with at least one reaction: {results.reports_with_reaction:,} "
        f"({pct(results.reports_with_reaction)});  with a serious outcome: "
        f"{results.reports_with_serious_outcome:,} ({pct(results.reports_with_serious_outcome)})"
    )

    lines.append("")
    lines.append("A2. MONTHLY REPORTING TREND (by report date)")
    lines.append("-" * 78)
    lines.append(
        f"  primary date column : {results.report_date_column}; "
        f"fallback: {results.fallback_date_column}"
    )
    lines.append(
        f"  reports with a report date : {results.reports_with_report_date:,} "
        f"({pct(results.reports_with_report_date)})"
    )
    lines.append(
        f"  reports using the fallback : {results.reports_using_fallback_date:,} "
        f"({pct(results.reports_using_fallback_date)})"
    )
    lines.append(
        f"  reports with no usable date: {results.reports_missing_date:,} "
        f"({pct(results.reports_missing_date)}) - excluded from the trend, not guessed"
    )
    if results.monthly_reports:
        lines.append(
            f"  months with data: {len(results.monthly_reports)}  "
            f"(range {results.date_range[0]} .. {results.date_range[1]})"
        )
        shown = results.monthly_reports[-TOP_MONTHS_REPORT:]
        lines.append(f"  last {len(shown)} month(s):")
        for month, count in shown:
            lines.append(f"    {month}  {count:>10,}")
    else:
        lines.append("  NO report carried a usable date - no monthly trend was produced.")

    lines.append("")
    lines.append(f"A3. MOST FREQUENT REACTIONS BY QUARTER (top {TOP_REACTIONS} per quarter)")
    lines.append("-" * 78)
    for quarter in results.quarters:
        subset = [item for item in results.reaction_top_by_quarter if item[0] == quarter][:TOP_REACTIONS]
        q_pairs = results.reaction_pairs_by_quarter.get(quarter, 0) or 1
        lines.append(
            f"  {quarter}: {results.reaction_pairs_by_quarter.get(quarter, 0):,} distinct report-reaction pairs"
        )
        for index, (_q, pt, count) in enumerate(subset, start=1):
            lines.append(
                f"    {index:>3}  {_clip(pt, 46):<48}{count:>12,}{100.0 * count / q_pairs:>13.2f}%"
            )

    lines.append("")
    lines.append(f"A4. DRUG REPORTING FREQUENCY CHANGE (top {TOP_DRUGS} drugs by total reports)")
    lines.append("-" * 78)
    lines.append(
        f"  {'#':>3}  {'DRUG (DRUGNAME_STD)':<40}{first_q:>11}{last_q:>11}{'DELTA':>10}{'% CHANGE':>11}"
    )
    for index, (name, first, last, dlt, pct_change) in enumerate(results.drug_frequency, start=1):
        change = "n/a" if pct_change is None else f"{pct_change:+.1f}%"
        lines.append(
            f"  {index:>3}  {_clip(name, 38):<40}{first:>11,}{last:>11,}{dlt:>+10,}{change:>11}"
        )

    lines.append("")
    lines.append("PART B - DEMOGRAPHIC ANALYSIS")
    lines.append("-" * 78)
    lines.append("B1. AGE-GROUP DISTRIBUTION (from AGE_YEARS)")
    lines.append(f"  {'AGE GROUP':<16}{'REPORTS':>12}{'% OF REPORTS':>14}")
    for bucket, count in results.age_counts:
        lines.append(f"  {bucket:<16}{count:>12,}{pct(count):>14}")
    lines.append(
        f"  reports with no usable age: {results.age_missing:,} "
        f"(shown as 'Unknown' above)"
    )

    lines.append("")
    lines.append("B2. GENDER DISTRIBUTION (from SEX_NORM; U = reported, neither M nor F)")
    lines.append(f"  {'GENDER':<16}{'REPORTS':>12}{'% OF REPORTS':>14}")
    for sex, count in results.sex_counts:
        lines.append(f"  {sex:<16}{count:>12,}{pct(count):>14}")
    lines.append(
        f"  reports with no usable gender: {results.sex_missing:,} "
        f"(shown as 'Unknown' above)"
    )

    lines.append("")
    lines.append(f"B3. TOP REACTIONS BY AGE GROUP (top {TOP_REACTIONS_PER_GROUP} per group)")
    lines.append("-" * 78)
    for bucket, _count in results.age_counts:
        subset = [item for item in results.reactions_by_age if item[0] == bucket][:TOP_REACTIONS_PER_GROUP]
        detail = ", ".join(f"{_clip(pt, 30)}={c:,}" for _b, pt, c in subset)
        lines.append(f"  {bucket:<10}{detail}")

    lines.append("")
    lines.append(f"B4. TOP REACTIONS BY GENDER (top {TOP_REACTIONS_PER_GROUP} per gender)")
    lines.append("-" * 78)
    for sex, _count in results.sex_counts:
        subset = [item for item in results.reactions_by_sex if item[0] == sex][:TOP_REACTIONS_PER_GROUP]
        detail = ", ".join(f"{_clip(pt, 30)}={c:,}" for _s, pt, c in subset)
        lines.append(f"  {sex:<10}{detail}")

    lines.append("")
    lines.append("B5. SERIOUS OUTCOMES BY DEMOGRAPHIC GROUP")
    lines.append("-" * 78)
    lines.append("  BY AGE GROUP")
    lines.append(f"  {'AGE GROUP':<16}{'SERIOUS REPORTS':>18}{'GROUP REPORTS':>16}{'SHARE':>10}")
    for bucket, serious, tot, share in results.serious_by_age:
        lines.append(f"  {bucket:<16}{serious:>18,}{tot:>16,}{share:>9.2f}%")
    lines.append("  BY GENDER")
    lines.append(f"  {'GENDER':<16}{'SERIOUS REPORTS':>18}{'GROUP REPORTS':>16}{'SHARE':>10}")
    for sex, serious, tot, share in results.serious_by_sex:
        lines.append(f"  {sex:<16}{serious:>18,}{tot:>16,}{share:>9.2f}%")
    lines.append("  Serious outcome codes by quarter (a report may carry several):")
    for quarter in results.quarters:
        subset = [item for item in results.outcome_by_quarter if item[0] == quarter]
        detail = ", ".join(f"{code}={count:,}" for _q, code, _label, count in subset)
        lines.append(f"    {quarter}: {detail}")

    lines.append("")
    lines.append("C. VALIDATION CHECKS")
    lines.append("-" * 78)
    for name, ok, detail in results.checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        lines.append(f"         {detail}")

    lines.append("")
    lines.append("D. ISSUES AND ERRORS")
    lines.append("-" * 78)
    if results.errors:
        for error in results.errors:
            lines.append(f"  {error}")
    else:
        lines.append("  none")

    failed = [name for name, ok, _detail in results.checks if not ok]
    lines.append("")
    lines.append("E. SUMMARY")
    lines.append("-" * 78)
    lines.append(
        f"  {len(results.quarters)} quarter(s), {len(results.cleanings)} table(s); "
        f"{results.total_reports:,} reports, {len(results.monthly_reports)} month(s) "
        f"in the trend, {len(results.reaction_top_by_quarter)} ranked reaction row(s), "
        f"{len(results.drug_frequency)} drug change row(s)."
    )
    if results.errors or failed:
        lines.append(
            f"  Status: INCOMPLETE - {len(results.errors)} error(s), "
            f"{len(failed)} failed check(s)."
        )
    else:
        lines.append("  Status: COMPLETE - all aggregations produced and all checks passed.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(
    spark: Any,
    *,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
    visualization_dir: Optional[Path] = None,
    analytics_dir: Optional[Path] = None,
    theme: str = "plotly_white",
    top_n: int = TOP_REACTIONS,
) -> TemporalDemographicResults:
    """Analyse, persist the outputs (when directories are given) and return results."""
    log = get_logger(__name__)
    results = analyze(
        spark, quarters=quarters, settings=settings, quarter_dirs=quarter_dirs
    )
    if visualization_dir is not None and analytics_dir is not None:
        written = write_outputs(
            results,
            visualization_dir=visualization_dir,
            analytics_dir=analytics_dir,
            theme=theme,
            top_n=top_n,
        )
        for name, path in written.items():
            log.info("wrote %s -> %s", name, path)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Temporal + demographic FAERS analysis with Plotly figures (read-only)."
    )
    parser.add_argument("--quarters", nargs="*", default=None, help="e.g. 2026_Q1 2026_Q2")
    parser.add_argument(
        "--report",
        default=None,
        help="Report path (default reports/temporal_demographic_report.txt; '' disables it).",
    )
    parser.add_argument(
        "--visualizations",
        default=None,
        help="HTML figure directory (default data/analytics/visualizations; '' disables it).",
    )
    parser.add_argument(
        "--outputs",
        default=None,
        help="CSV/JSON directory (default data/analytics/temporal_demographic; '' disables it).",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Top-N sizing for the charts.")
    args = parser.parse_args(argv)

    log = get_logger(__name__)
    settings = load_settings()

    all_quarters = [quarter for quarter, _dir in get_quarter_dirs()]
    quarters = list(args.quarters or all_quarters)
    unknown = [q for q in quarters if q not in all_quarters]
    if unknown:
        log.error("Unknown quarter(s): %s", ", ".join(unknown))
        return 1

    quarter_dirs = dict(get_quarter_dirs())
    try:
        vis_config = get_visualization_config()
    except Exception:  # noqa: BLE001 - config already validated by load_settings
        vis_config = {}
    theme = str(vis_config.get("theme", "plotly_white"))
    top_n = int(args.top_n or vis_config.get("top_n", TOP_REACTIONS) or TOP_REACTIONS)

    if args.visualizations is None:
        visualization_dir: Optional[Path] = get_path("analytics") / "visualizations"
    else:
        visualization_dir = Path(args.visualizations) if args.visualizations else None

    if args.outputs is None:
        analytics_dir: Optional[Path] = get_path("analytics") / "temporal_demographic"
    else:
        analytics_dir = Path(args.outputs) if args.outputs else None

    try:
        spark = create_spark_session(app_name="DrugSafetyTemporalDemographic")
    except SparkEnvironmentError as exc:
        log.error("%s", exc)
        return 1

    try:
        results = run(
            spark,
            quarters=quarters,
            settings=settings,
            quarter_dirs=quarter_dirs,
            visualization_dir=visualization_dir if analytics_dir is not None else None,
            analytics_dir=analytics_dir,
            theme=theme,
            top_n=top_n,
        )
    except (TemporalDemographicError, CleaningError, IngestionError) as exc:
        log.error("temporal/demographic analysis failed: %s", exc)
        return 2
    finally:
        stop_spark_session(spark)

    report = render_report(results)
    print(report)

    report_path = args.report
    if report_path is None:
        report_path = str(get_path("reports") / "temporal_demographic_report.txt")
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report + "\n", encoding="utf-8")
        log.info("Report written to: %s", target)

    if results.errors or any(not ok for _n, ok, _d in results.checks):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
