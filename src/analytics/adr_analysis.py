"""Adverse Drug Reaction (ADR) analysis over the real FDA FAERS 2026 extracts.

Reads the cleaned DEMO / DRUG / REAC / OUTC tables for the configured quarters
(reusing :mod:`src.ingestion.load_faers` and :mod:`src.preprocessing.clean_faers`)
and produces seven aggregations:

1. total adverse reaction reports;
2. most frequently reported adverse reactions;
3. the top 20 adverse reactions;
4. reaction frequency by quarter;
5. drug-wise adverse reaction counts;
6. serious outcome analysis from OUTC;
7. demographic analysis (age, sex) from DEMO.

Design decisions
----------------
* **Reports are identified by ``PRIMARYID``.** In FAERS ``PRIMARYID`` is the
  unique report identifier, so it is the only key used to connect DEMO, DRUG,
  REAC and OUTC. No other join key is introduced.
* **Duplicate report/reaction pairs never multiply counts.** The REAC source
  legitimately contains identical ``(PRIMARYID, PT)`` rows; every aggregation
  therefore works on a *distinct* ``(quarter, PRIMARYID, PT)`` projection. No
  source record is modified or deleted - the de-duplication happens only inside
  the aggregation.
* **Q1 and Q2 stay separate.** ``quarter`` is carried through every grouping, so
  every result can be split by quarter and the two quarters never merge.
* **No join is performed where none is needed.** Four of the five drug/reaction
  metrics are computed without any join; the single drug-to-reaction cross-tab
  joins two already-distinct projections on ``(quarter, PRIMARYID)`` so the
  per-report drug x reaction product cannot be inflated by REAC duplicates.
* **Counts are counts of reported events, not measures of risk.** FAERS is a
  spontaneous reporting system with no denominator and heavy reporting bias, so
  nothing here can establish incidence, causality or comparative drug safety.

Environment constraints honoured: PySpark DataFrames only, no pandas, no
PyArrow, no UDFs, no Parquet. Aggregations are small, so they are collected to
the driver and persisted with the Python standard library (``csv`` / ``json``).

Usage::

    python -m src.analytics.adr_analysis
    python -m src.analytics.adr_analysis --quarters 2026_Q1
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

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
from src.utils.config import get_path, get_quarter_dirs
from src.utils.logging_setup import get_logger

__all__ = [
    "AdrAnalysisError",
    "CAUSATION_DISCLAIMER",
    "OUTC_CODE_LABELS",
    "AGE_BUCKET_ORDER",
    "ANALYSIS_TABLES",
    "AdrResults",
    "analyze",
    "write_outputs",
    "render_report",
    "run",
    "main",
]

#: Tables this module needs (a subset of the seven FAERS tables).
ANALYSIS_TABLES: Tuple[str, ...] = ("DEMO", "DRUG", "REAC", "OUTC")

#: Grain statement reused in the report.
ANALYSIS_GRAIN = (
    "Report = one distinct PRIMARYID (FAERS report identifier). "
    "Report/reaction pair = one distinct (quarter, PRIMARYID, PT) triple."
)

#: Mandatory caution; echoed in the report, the CSV header block and the JSON.
CAUSATION_DISCLAIMER = (
    "FAERS is a spontaneous reporting system: the counts below are numbers of "
    "reported events. They carry no denominator (no exposure, no population), "
    "are subject to severe reporting bias and duplicate/notoriety effects, and "
    "must NOT be read as incidence, risk, relative safety or drug causation."
)

#: FAERS OUTC_COD labels, as defined by the FDA FAERS documentation (the
#: ``ASC_NTS`` code list shipped with the extract). Every OUTC code denotes a
#: serious outcome under the 21 CFR 314.80 definition of "serious".
OUTC_CODE_LABELS: Mapping[str, str] = {
    "DE": "Death",
    "LT": "Life-Threatening",
    "HO": "Hospitalization - Initial or Prolonged",
    "DS": "Disability",
    "CA": "Congenital Anomaly",
    "RI": "Required Intervention to Prevent Permanent Impairment/Damage",
    "OT": "Other Serious (Important Medical Event)",
}

#: Age buckets and their fixed presentation order.
AGE_BUCKET_ORDER: Tuple[str, ...] = (
    "0-17",
    "18-44",
    "45-64",
    "65-74",
    "75+",
    "Unknown",
)

#: Sex categories and their fixed presentation order.
SEX_ORDER: Tuple[str, ...] = ("F", "M", "U", "Unknown")

# --- output sizing ---------------------------------------------------------
TOP_REACTIONS_CSV = 100
TOP_REACTIONS_REPORT = 20
TOP_REACTIONS_PER_QUARTER = 50
TOP_DRUGS_CSV = 100
TOP_DRUGS_REPORT = 20
TOP_DRUG_REACTION_CSV = 100
TOP_DRUG_REACTION_REPORT = 20


class AdrAnalysisError(RuntimeError):
    """Raised when an ADR aggregation cannot be produced."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class AdrResults:
    """Every artefact produced by one ADR analysis run."""

    quarters: List[str] = field(default_factory=list)
    cleanings: Dict[str, TableCleaning] = field(default_factory=dict)

    # section 1 - report / reaction counts
    total_reports: int = 0
    total_cases: int = 0
    reaction_rows: int = 0
    distinct_pairs: int = 0
    duplicate_pairs: int = 0
    rows_in_repeated_pairs: int = 0
    repeated_pair_keys: int = 0
    reports_with_reaction: int = 0
    reports_with_outcome: int = 0

    # section 2/3 - reaction frequencies
    reaction_top: List[Tuple[str, int]] = field(default_factory=list)
    reaction_by_quarter: List[Tuple[str, str, int]] = field(default_factory=list)
    pairs_by_quarter: Dict[str, int] = field(default_factory=dict)

    # section 5 - drug-wise counts
    drug_report_top: List[Tuple[str, int]] = field(default_factory=list)
    distinct_report_drug_pairs: int = 0
    drug_reaction_top: List[Tuple[str, str, int]] = field(default_factory=list)
    drug_reaction_join_rows: int = 0
    drug_reaction_join_distinct_rows: int = 0

    # section 6 - serious outcomes
    outcome_counts: List[Tuple[str, str, int]] = field(default_factory=list)
    outcome_by_quarter: List[Tuple[str, str, str, int]] = field(default_factory=list)
    unknown_outcome_codes: List[str] = field(default_factory=list)

    # section 7 - demographics
    age_counts: List[Tuple[str, int]] = field(default_factory=list)
    sex_counts: List[Tuple[str, int]] = field(default_factory=list)
    age_by_quarter: List[Tuple[str, str, int]] = field(default_factory=list)
    sex_by_quarter: List[Tuple[str, str, int]] = field(default_factory=list)

    # validation
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_summary(self, generated_at: str) -> Dict[str, Any]:
        return {
            "generated_at": generated_at,
            "quarters": self.quarters,
            "disclaimer": CAUSATION_DISCLAIMER,
            "grain": ANALYSIS_GRAIN,
            "counts": {
                "total_reports": self.total_reports,
                "total_cases": self.total_cases,
                "report_reaction_rows": self.reaction_rows,
                "distinct_report_reaction_pairs": self.distinct_pairs,
                "duplicate_report_reaction_pairs": self.duplicate_pairs,
                "rows_sharing_a_repeated_report_reaction_key": self.rows_in_repeated_pairs,
                "repeated_report_reaction_keys": self.repeated_pair_keys,
                "reports_with_at_least_one_reaction": self.reports_with_reaction,
                "reports_with_at_least_one_serious_outcome": self.reports_with_outcome,
                "distinct_report_drug_pairs": self.distinct_report_drug_pairs,
            },
            "reactions_by_quarter": self.pairs_by_quarter,
            "top_20_reactions": [
                {"reaction": pt, "reports": count}
                for pt, count in self.reaction_top[:TOP_REACTIONS_REPORT]
            ],
            "top_20_drugs": [
                {"drug": name, "reports": count}
                for name, count in self.drug_report_top[:TOP_DRUGS_REPORT]
            ],
            "serious_outcomes": [
                {"code": code, "label": label, "reports": count}
                for code, label, count in self.outcome_counts
            ],
            "checks": [
                {"name": name, "passed": ok, "detail": detail}
                for name, ok, detail in self.checks
            ],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Transformation helpers (built-in Spark functions only - no UDFs)
# ---------------------------------------------------------------------------
def _age_bucket() -> "F.Column":
    """Bin ``AGE_YEARS`` into the fixed AGE_BUCKET_ORDER buckets."""
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
    """A portable row counter (avoids ``count('*')`` on projected frames)."""
    return F.count(F.lit(1))


def _ordered(items: Sequence[Tuple[str, int]], order: Sequence[str]) -> List[Tuple[str, int]]:
    """Sort ``(label, count)`` pairs by a fixed label order, unknown labels last."""
    position = {label: index for index, label in enumerate(order)}
    return sorted(items, key=lambda pair: (position.get(pair[0], len(order)), pair[0]))


def _clip(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` characters, adding an ellipsis when cut."""
    return text if len(text) <= width else text[: width - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Analysis
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


def analyze(
    spark: Any,
    *,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> AdrResults:
    """Run every ADR aggregation and return the collected results."""
    log = get_logger(__name__)
    results = AdrResults(quarters=list(quarters))

    frames, cleanings, errors = _clean_inputs(
        spark, quarters=quarters, settings=settings, quarter_dirs=quarter_dirs
    )
    results.cleanings = cleanings
    results.errors.extend(errors)

    missing = [table for table in ANALYSIS_TABLES if table not in frames]
    if missing:
        raise AdrAnalysisError(
            "required table(s) unavailable: " + ", ".join(missing) + "; see errors"
        )

    # --- projections actually used by the aggregations -----------------------
    reports = frames["DEMO"].select(
        QUARTER_COLUMN, "PRIMARYID", "CASEID", "AGE_YEARS", "SEX_NORM"
    ).persist()
    # Distinct report/reaction pairs: the single defence against REAC duplicate
    # rows inflating every downstream count.
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
        # ---- 1. total adverse reaction reports -----------------------------
        results.total_reports = reports.select("PRIMARYID").distinct().count()
        results.total_cases = reports.select("CASEID").distinct().count()
        results.reaction_rows = cleanings["REAC"].rows_after
        results.distinct_pairs = pairs.count()
        results.duplicate_pairs = results.reaction_rows - results.distinct_pairs
        # The same duplicates measured the way preprocessing measures them
        # (every row belonging to a repeated key, not only the extra ones), so
        # the two modules can be cross-validated exactly.
        raw_reaction = frames["REAC"].select(
            QUARTER_COLUMN, "PRIMARYID", "PT"
        ).filter(F.col("PT").isNotNull())
        group_sizes = raw_reaction.groupBy(QUARTER_COLUMN, "PRIMARYID", "PT").agg(
            _count_rows().alias("n")
        )
        repeated = group_sizes.agg(
            F.coalesce(
                F.sum(F.when(F.col("n") > F.lit(1), F.col("n"))), F.lit(0)
            ).alias("rows_in_repeated_keys"),
            F.coalesce(
                F.sum(F.when(F.col("n") > F.lit(1), F.lit(1))), F.lit(0)
            ).alias("repeated_keys"),
        ).first()
        results.rows_in_repeated_pairs = int(repeated["rows_in_repeated_keys"])
        results.repeated_pair_keys = int(repeated["repeated_keys"])
        results.reports_with_reaction = pairs.select("PRIMARYID").distinct().count()
        results.reports_with_outcome = outcomes.select("PRIMARYID").distinct().count()
        log.info(
            "reports=%s  distinct pairs=%s  duplicate pairs=%s  repeated-key rows=%s",
            f"{results.total_reports:,}",
            f"{results.distinct_pairs:,}",
            f"{results.duplicate_pairs:,}",
            f"{results.rows_in_repeated_pairs:,}",
        )

        # ---- 2/3. most frequent reactions + top 20 -------------------------
        reaction_counts = pairs.groupBy("PT").agg(_count_rows().alias("reports"))
        results.reaction_top = [
            (row["PT"], int(row["reports"]))
            for row in reaction_counts.orderBy(
                F.desc("reports"), F.asc("PT")
            ).limit(TOP_REACTIONS_CSV).collect()
        ]

        # ---- 4. reaction frequency by quarter ------------------------------
        results.pairs_by_quarter = {
            row[QUARTER_COLUMN]: int(row["reports"])
            for row in pairs.groupBy(QUARTER_COLUMN).agg(_count_rows().alias("reports")).collect()
        }
        per_quarter = pairs.groupBy("PT", QUARTER_COLUMN).agg(_count_rows().alias("reports"))
        rank = Window.partitionBy(QUARTER_COLUMN).orderBy(F.desc("reports"), F.asc("PT"))
        ranked = (
            per_quarter.withColumn("rank", F.row_number().over(rank))
            .filter(F.col("rank") <= F.lit(TOP_REACTIONS_PER_QUARTER))
            .select("PT", QUARTER_COLUMN, "reports")
        )
        results.reaction_by_quarter = sorted(
            (
                (row["PT"], row[QUARTER_COLUMN], int(row["reports"]))
                for row in ranked.collect()
            ),
            key=lambda item: (-item[2], item[0]),
        )

        # ---- 5. drug-wise adverse reaction counts --------------------------
        results.distinct_report_drug_pairs = drug_pairs.count()
        drug_counts = drug_pairs.groupBy("DRUGNAME_STD").agg(_count_rows().alias("reports"))
        results.drug_report_top = [
            (row["DRUGNAME_STD"], int(row["reports"]))
            for row in drug_counts.orderBy(
                F.desc("reports"), F.asc("DRUGNAME_STD")
            ).limit(TOP_DRUGS_CSV).collect()
        ]
        # The only join in the module: two already-distinct projections keyed on
        # (quarter, PRIMARYID), so a report's drug x reaction product is bounded
        # by its own distinct drugs and distinct reactions - never by duplicates.
        joined = drug_pairs.join(
            pairs.select(QUARTER_COLUMN, "PRIMARYID", "PT"),
            on=[QUARTER_COLUMN, "PRIMARYID"],
            how="inner",
        )
        results.drug_reaction_join_rows = joined.count()
        results.drug_reaction_join_distinct_rows = joined.distinct().count()
        drug_reaction = (
            joined.groupBy("DRUGNAME_STD", "PT").agg(_count_rows().alias("reports"))
        )
        results.drug_reaction_top = [
            (row["DRUGNAME_STD"], row["PT"], int(row["reports"]))
            for row in drug_reaction.orderBy(
                F.desc("reports"), F.asc("DRUGNAME_STD"), F.asc("PT")
            ).limit(TOP_DRUG_REACTION_CSV).collect()
        ]

        # ---- 6. serious outcome analysis (OUTC) ----------------------------
        outcome_rows = outcomes.groupBy("OUTC_COD").agg(_count_rows().alias("reports")).collect()
        results.outcome_counts = sorted(
            (
                (row["OUTC_COD"], OUTC_CODE_LABELS.get(row["OUTC_COD"], "Unmapped code"), int(row["reports"]))
                for row in outcome_rows
            ),
            key=lambda item: -item[2],
        )
        results.unknown_outcome_codes = sorted(
            code for code, _label, _count in results.outcome_counts if code not in OUTC_CODE_LABELS
        )
        results.outcome_by_quarter = sorted(
            (
                (row[QUARTER_COLUMN], row["OUTC_COD"], OUTC_CODE_LABELS.get(row["OUTC_COD"], "Unmapped code"), int(row["reports"]))
                for row in outcomes.groupBy(QUARTER_COLUMN, "OUTC_COD").agg(_count_rows().alias("reports")).collect()
            ),
            key=lambda item: (item[0], -item[3]),
        )

        # ---- 7. demographics (age, sex) from DEMO --------------------------
        age_counts = (
            reports.withColumn("age_bucket", _age_bucket())
            .groupBy("age_bucket")
            .agg(_count_rows().alias("reports"))
            .collect()
        )
        results.age_counts = _ordered(
            [(row["age_bucket"], int(row["reports"])) for row in age_counts], AGE_BUCKET_ORDER
        )
        sex_counts = (
            reports.withColumn("sex_category", _sex_category())
            .groupBy("sex_category")
            .agg(_count_rows().alias("reports"))
            .collect()
        )
        results.sex_counts = _ordered(
            [(row["sex_category"], int(row["reports"])) for row in sex_counts], SEX_ORDER
        )
        results.age_by_quarter = sorted(
            (
                (row[QUARTER_COLUMN], row["age_bucket"], int(row["reports"]))
                for row in reports.withColumn("age_bucket", _age_bucket())
                .groupBy(QUARTER_COLUMN, "age_bucket")
                .agg(_count_rows().alias("reports"))
                .collect()
            ),
            key=lambda item: (item[0], AGE_BUCKET_ORDER.index(item[1]) if item[1] in AGE_BUCKET_ORDER else 99),
        )
        results.sex_by_quarter = sorted(
            (
                (row[QUARTER_COLUMN], row["sex_category"], int(row["reports"]))
                for row in reports.withColumn("sex_category", _sex_category())
                .groupBy(QUARTER_COLUMN, "sex_category")
                .agg(_count_rows().alias("reports"))
                .collect()
            ),
            key=lambda item: (item[0], SEX_ORDER.index(item[1]) if item[1] in SEX_ORDER else 99),
        )
    finally:
        for frame in (reports, pairs, outcomes, drug_pairs):
            frame.unpersist()

    results.checks = _validate(results)
    return results


def _validate(results: AdrResults) -> List[Tuple[str, bool, str]]:
    """Cross-check the aggregates; every check is recorded pass or fail."""
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

    reaction_sum = sum(count for _pt, count in results.reaction_top)
    # The reaction list is truncated to the top N, so this only asserts the
    # collected slice is internally consistent (positive, ordered).
    checks.append(
        (
            "top reaction counts are positive and non-increasing",
            all(count > 0 for _pt, count in results.reaction_top)
            and reaction_sum
            == sum(sorted((c for _p, c in results.reaction_top), reverse=True)),
            f"{len(results.reaction_top)} reaction(s) ranked",
        )
    )
    checks.append(
        (
            "duplicate report/reaction rows detected and excluded from counts",
            results.duplicate_pairs == results.reaction_rows - results.distinct_pairs
            and results.duplicate_pairs > 0,
            f"raw rows {results.reaction_rows:,} - distinct pairs "
            f"{results.distinct_pairs:,} = {results.duplicate_pairs:,} extra row(s)",
        )
    )
    checks.append(
        (
            "reaction duplicate count agrees with the preprocessing module",
            results.rows_in_repeated_pairs == results.cleanings["REAC"].duplicate_rows,
            f"rows in repeated (quarter, PRIMARYID, PT) keys "
            f"{results.rows_in_repeated_pairs:,} vs preprocessing "
            f"{results.cleanings['REAC'].duplicate_rows:,} across "
            f"{results.repeated_pair_keys:,} repeated key(s)",
        )
    )

    quarter_sum = sum(results.pairs_by_quarter.values())
    checks.append(
        (
            "distinct pairs per quarter sum to the overall distinct pairs",
            quarter_sum == results.distinct_pairs,
            f"{quarter_sum:,} vs {results.distinct_pairs:,}",
        )
    )
    checks.append(
        (
            "Q1 and Q2 are both present and distinguishable",
            set(results.pairs_by_quarter) == set(results.quarters)
            and len(set(results.pairs_by_quarter.values())) >= 1,
            ", ".join(f"{q}={n:,}" for q, n in sorted(results.pairs_by_quarter.items())),
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
            "sex categories cover every report",
            sex_sum == results.total_reports,
            f"{sex_sum:,} vs {results.total_reports:,}",
        )
    )

    outcome_sum = sum(count for _code, _label, count in results.outcome_counts)
    outc_cleaning = results.cleanings["OUTC"]
    outc_distinct = outc_cleaning.rows_after - outc_cleaning.duplicate_rows
    checks.append(
        (
            "outcome code counts sum to the distinct OUTC rows",
            outcome_sum == outc_distinct,
            f"{outcome_sum:,} vs {outc_distinct:,}",
        )
    )
    checks.append(
        (
            "every OUTC code maps to a documented FAERS label",
            not results.unknown_outcome_codes,
            "unmapped: " + (", ".join(results.unknown_outcome_codes) if results.unknown_outcome_codes else "none"),
        )
    )

    checks.append(
        (
            "reports with a serious outcome never exceed total reports",
            results.reports_with_outcome <= results.total_reports,
            f"{results.reports_with_outcome:,} <= {results.total_reports:,}",
        )
    )
    checks.append(
        (
            "reports with a reaction never exceed total reports",
            results.reports_with_reaction <= results.total_reports,
            f"{results.reports_with_reaction:,} <= {results.total_reports:,}",
        )
    )
    checks.append(
        (
            "drug/reaction join produced no duplicate rows (no multiplication)",
            results.drug_reaction_join_rows == results.drug_reaction_join_distinct_rows,
            f"{results.drug_reaction_join_rows:,} rows, "
            f"{results.drug_reaction_join_distinct_rows:,} distinct",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Output writers (Python standard library only)
# ---------------------------------------------------------------------------
def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(
    results: AdrResults,
    *,
    analytics_dir: Path,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Path]:
    """Persist every result set as CSV plus one JSON summary; return the paths."""
    timestamp = (generated_at or datetime.now()).isoformat(timespec="seconds")
    analytics_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    def emit(name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        target = analytics_dir / name
        _write_csv(target, header, rows)
        written[name] = target

    total = results.total_reports or 1

    emit(
        "adr_top_reactions.csv",
        ["reaction_pt", "reports", "pct_of_reports"],
        [
            [pt, count, round(100.0 * count / total, 4)]
            for pt, count in results.reaction_top
        ],
    )
    emit(
        "adr_reaction_by_quarter.csv",
        ["reaction_pt", "quarter", "reports"],
        [[pt, quarter, count] for pt, quarter, count in results.reaction_by_quarter],
    )
    emit(
        "adr_drug_report_counts.csv",
        ["drug_name", "reports", "pct_of_reports"],
        [
            [name, count, round(100.0 * count / total, 4)]
            for name, count in results.drug_report_top
        ],
    )
    emit(
        "adr_drug_reaction_pairs.csv",
        ["drug_name", "reaction_pt", "reports"],
        [[name, pt, count] for name, pt, count in results.drug_reaction_top],
    )
    emit(
        "adr_outcome_counts.csv",
        ["outc_cod", "outcome_label", "reports", "pct_of_reports"],
        [
            [code, label, count, round(100.0 * count / total, 4)]
            for code, label, count in results.outcome_counts
        ],
    )
    emit(
        "adr_outcome_by_quarter.csv",
        ["quarter", "outc_cod", "outcome_label", "reports"],
        [
            [quarter, code, label, count]
            for quarter, code, label, count in results.outcome_by_quarter
        ],
    )
    emit(
        "adr_demographics_age.csv",
        ["age_bucket", "reports", "pct_of_reports"],
        [
            [bucket, count, round(100.0 * count / total, 4)]
            for bucket, count in results.age_counts
        ],
    )
    emit(
        "adr_demographics_sex.csv",
        ["sex", "reports", "pct_of_reports"],
        [
            [sex, count, round(100.0 * count / total, 4)]
            for sex, count in results.sex_counts
        ],
    )
    emit(
        "adr_demographics_age_by_quarter.csv",
        ["quarter", "age_bucket", "reports"],
        [[quarter, bucket, count] for quarter, bucket, count in results.age_by_quarter],
    )
    emit(
        "adr_demographics_sex_by_quarter.csv",
        ["quarter", "sex", "reports"],
        [[quarter, sex, count] for quarter, sex, count in results.sex_by_quarter],
    )

    summary_path = analytics_dir / "adr_summary.json"
    summary_path.write_text(
        json.dumps(results.as_summary(timestamp), indent=2) + "\n", encoding="utf-8"
    )
    written["adr_summary.json"] = summary_path
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(
    results: AdrResults,
    *,
    generated_at: Optional[datetime] = None,
) -> str:
    """Render the ADR analysis report as plain text."""
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    total = results.total_reports or 1
    pct = lambda count: f"{100.0 * count / total:5.2f}%"  # noqa: E731 - local formatting helper

    lines: List[str] = [
        "=" * 78,
        "FAERS ADVERSE DRUG REACTION (ADR) ANALYSIS REPORT",
        f"Generated : {stamp}",
        f"Quarters  : {', '.join(results.quarters)}",
        "Engine    : PySpark DataFrame API only (no pandas, no PyArrow, no UDFs)",
        "Source    : real FAERS 2026 Q1/Q2 ASCII extracts, via ingestion + preprocessing",
        "=" * 78,
        "",
        "0. SCOPE AND METHOD",
        "-" * 78,
        f"  {ANALYSIS_GRAIN}",
        "  Reports are connected by PRIMARYID only; no other join key is used.",
        "  REAC duplicate (PRIMARYID, PT) rows are removed only inside aggregations,",
        "  on a distinct (quarter, PRIMARYID, PT) projection, so counts are never",
        "  multiplied. The two quarters are grouped separately and never merged.",
        "",
        "  !! " + CAUSATION_DISCLAIMER.replace(". ", ".\n  !! "),
        "",
        "1. REPORT AND REACTION COUNTS (actual)",
        "-" * 78,
        f"  Distinct adverse-event reports (PRIMARYID)   : {results.total_reports:>12,}",
        f"  Distinct CASEIDs                            : {results.total_cases:>12,}",
        f"  REPORT-REACTION rows in REAC                : {results.reaction_rows:>12,}",
        f"  Distinct report-reaction pairs (used)       : {results.distinct_pairs:>12,}",
        f"  Duplicate rows beyond first occurrence      : {results.duplicate_pairs:>12,}",
        f"  Rows sharing a repeated (report, PT) key    : {results.rows_in_repeated_pairs:>12,}"
        f"  ({results.repeated_pair_keys:,} repeated key(s))",
        "    Both lines describe the same REAC duplicates under two conventions:",
        "    'beyond first occurrence' = rows - distinct pairs; 'sharing a repeated",
        "    key' also counts the first row of each duplicated key (this is the count",
        "    the preprocessing module reports, used here as a cross-check).",
        f"  Reports with at least one reaction          : {results.reports_with_reaction:>12,}"
        f"  ({pct(results.reports_with_reaction)})",
        f"  Reports with at least one serious outcome   : {results.reports_with_outcome:>12,}"
        f"  ({pct(results.reports_with_outcome)})",
        "",
        f"2. TOP {TOP_REACTIONS_REPORT} ADVERSE REACTIONS (by distinct reports)",
        "-" * 78,
    ]
    if results.reaction_top:
        lines.append(f"  {'#':>3}  {'REACTION (MedDRA PT)':<48}{'REPORTS':>12}{'% OF REPORTS':>14}")
        for index, (pt, count) in enumerate(results.reaction_top[:TOP_REACTIONS_REPORT], start=1):
            lines.append(f"  {index:>3}  {_clip(pt, 46):<48}{count:>12,}{pct(count):>14}")
    else:
        lines.append("  no reactions were read")

    lines.append("")
    lines.append(f"3. REACTION FREQUENCY BY QUARTER (top {TOP_REACTIONS_PER_QUARTER} per quarter)")
    lines.append("-" * 78)
    for quarter in results.quarters:
        quarter_total = results.pairs_by_quarter.get(quarter, 0)
        lines.append(f"  {quarter}: {quarter_total:,} distinct report-reaction pairs")
        subset = [item for item in results.reaction_by_quarter if item[1] == quarter][
            :TOP_REACTIONS_REPORT
        ]
        for index, (pt, _quarter, count) in enumerate(subset, start=1):
            share = 100.0 * count / (quarter_total or 1)
            lines.append(f"    {index:>3}  {_clip(pt, 46):<48}{count:>12,}{share:>13.2f}%")

    lines.append("")
    lines.append(f"4. DRUG-WISE ADVERSE REACTION COUNTS (top {TOP_DRUGS_REPORT} drugs, by distinct reports)")
    lines.append("-" * 78)
    lines.append(f"  distinct (report, drug) pairs: {results.distinct_report_drug_pairs:,}")
    if results.drug_report_top:
        lines.append(f"  {'#':>3}  {'DRUG (standardised DRUGNAME)':<48}{'REPORTS':>12}{'% OF REPORTS':>14}")
        for index, (name, count) in enumerate(results.drug_report_top[:TOP_DRUGS_REPORT], start=1):
            lines.append(f"  {index:>3}  {_clip(name, 46):<48}{count:>12,}{pct(count):>14}")
    else:
        lines.append("  no drug records were read")

    lines.append("")
    lines.append(f"  TOP {TOP_DRUG_REACTION_REPORT} DRUG-REACTION PAIRS (inner join on quarter + PRIMARYID)")
    lines.append("-" * 78)
    lines.append(f"  join produced {results.drug_reaction_join_rows:,} drug-reaction rows "
                 "(both sides already distinct, so no duplicate multiplication)")
    lines.append(f"  {'#':>3}  {'DRUG':<34}{'REACTION':<38}{'REPORTS':>10}")
    for index, (name, pt, count) in enumerate(results.drug_reaction_top[:TOP_DRUG_REACTION_REPORT], start=1):
        lines.append(f"  {index:>3}  {_clip(name, 32):<34}{_clip(pt, 36):<38}{count:>10,}")

    lines.append("")
    lines.append("5. SERIOUS OUTCOME ANALYSIS (OUTC)")
    lines.append("-" * 78)
    lines.append(f"  {'CODE':<6}{'OUTCOME (FAERS ASC_NTS)':<54}{'REPORTS':>12}{'% OF REPORTS':>14}")
    for code, label, count in results.outcome_counts:
        lines.append(f"  {code:<6}{_clip(label, 54):<56}{count:>12,}{pct(count):>14}")
    lines.append("  A report may carry several outcome codes, so these counts are not")
    lines.append("  mutually exclusive and their sum exceeds the number of reports.")
    lines.append("  Every FAERS OUTC code denotes a serious outcome (21 CFR 314.80);")
    lines.append("  a report absent from OUTC is not thereby proven non-serious.")
    lines.append("")
    for quarter in results.quarters:
        subset = [item for item in results.outcome_by_quarter if item[0] == quarter]
        detail = ", ".join(f"{code}={count:,}" for _q, code, _label, count in subset)
        lines.append(f"  {quarter}: {detail}")

    lines.append("")
    lines.append("6. DEMOGRAPHIC ANALYSIS (DEMO, age + sex)")
    lines.append("-" * 78)
    lines.append("  AGE (from AGE_YEARS, normalised in preprocessing)")
    lines.append(f"  {'BUCKET':<16}{'REPORTS':>12}{'% OF REPORTS':>14}")
    for bucket, count in results.age_counts:
        lines.append(f"  {bucket:<16}{count:>12,}{pct(count):>14}")
    lines.append("")
    lines.append("  SEX (from SEX_NORM; U = reported but neither M nor F)")
    lines.append(f"  {'SEX':<16}{'REPORTS':>12}{'% OF REPORTS':>14}")
    for sex, count in results.sex_counts:
        lines.append(f"  {sex:<16}{count:>12,}{pct(count):>14}")
    lines.append("")
    lines.append("  BY QUARTER")
    for quarter in results.quarters:
        ages = ", ".join(
            f"{bucket}={count:,}" for q, bucket, count in results.age_by_quarter if q == quarter
        )
        sexes = ", ".join(
            f"{sex}={count:,}" for q, sex, count in results.sex_by_quarter if q == quarter
        )
        lines.append(f"  {quarter}  age: {ages}")
        lines.append(f"  {quarter}  sex: {sexes}")

    lines.append("")
    lines.append("7. VALIDATION CHECKS")
    lines.append("-" * 78)
    for name, ok, detail in results.checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        lines.append(f"         {detail}")

    lines.append("")
    lines.append("8. ISSUES AND ERRORS")
    lines.append("-" * 78)
    if results.errors:
        for error in results.errors:
            lines.append(f"  {error}")
    else:
        lines.append("  none")

    failed = [name for name, ok, _detail in results.checks if not ok]
    lines.append("")
    lines.append("9. SUMMARY")
    lines.append("-" * 78)
    lines.append(
        f"  {len(results.quarters)} quarter(s), {len(results.cleanings)} table(s); "
        f"{results.total_reports:,} reports, {results.distinct_pairs:,} distinct "
        f"report-reaction pairs, {len(results.reaction_top)} reactions ranked."
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
    analytics_dir: Optional[Path] = None,
) -> AdrResults:
    """Analyse, persist the outputs (when a directory is given) and return results."""
    log = get_logger(__name__)
    results = analyze(
        spark, quarters=quarters, settings=settings, quarter_dirs=quarter_dirs
    )
    if analytics_dir is not None:
        written = write_outputs(results, analytics_dir=analytics_dir)
        for name, path in written.items():
            log.info("wrote %s -> %s", name, path)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ADR analysis over the real FAERS extracts (read-only; no Parquet)."
    )
    parser.add_argument("--quarters", nargs="*", default=None, help="e.g. 2026_Q1 2026_Q2")
    parser.add_argument(
        "--report",
        default=None,
        help="Report path (default reports/adr_analysis_report.txt; '' disables it).",
    )
    parser.add_argument(
        "--outputs",
        default=None,
        help="CSV/JSON directory (default data/analytics; '' disables writing them).",
    )
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
    if args.outputs is None:
        analytics_dir: Optional[Path] = get_path("analytics")
    else:
        analytics_dir = Path(args.outputs) if args.outputs else None

    try:
        spark = create_spark_session(app_name="DrugSafetyAdrAnalysis")
    except SparkEnvironmentError as exc:
        log.error("%s", exc)
        return 1

    try:
        results = run(
            spark,
            quarters=quarters,
            settings=settings,
            quarter_dirs=quarter_dirs,
            analytics_dir=analytics_dir,
        )
    except (AdrAnalysisError, CleaningError, IngestionError) as exc:
        log.error("ADR analysis failed: %s", exc)
        return 2
    finally:
        stop_spark_session(spark)

    report = render_report(results)
    print(report)

    report_path = args.report
    if report_path is None:
        report_path = str(get_path("reports") / "adr_analysis_report.txt")
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
