"""PRR / ROR disproportionality signal detection over the real FAERS 2026 extracts.

Reuses the existing ingestion and preprocessing modules to read the cleaned
DEMO, DRUG and REAC tables for the configured quarters, builds a report-level
2x2 contingency table for every drug/reaction pair and computes the standard
disproportionality statistics:

* **PRR** - Proportional Reporting Ratio
* **ROR** - Reporting Odds Ratio
* **95% confidence intervals** for both, on the log scale
* **Pearson chi-square** for the 2x2 table

Contingency table (report-level, ``(quarter, PRIMARYID)`` = one report)
---------------------------------------------------------------------
=====  ==========================================
``a``  reports containing the drug **and** the reaction
``b``  reports containing the drug **without** the reaction
``c``  reports containing the reaction **without** the drug
``d``  reports containing **neither**
=====  ==========================================

Every cell is a count of *distinct reports*, never of raw DRUG or REAC rows, so
a report that lists a drug twice or a reaction three times still counts once.
``b``, ``c`` and ``d`` are derived from the report-level marginals rather than
re-counted with extra joins::

    n_drug  = a + b   (reports with the drug)
    n_reac  = a + c   (reports with the reaction)
    n_total = a + b + c + d   (all reports)

so ``b = n_drug - a``, ``c = n_reac - a`` and ``d = n_total - n_drug - n_reac + a``.

Measures
--------
::

    PRR = [a / (a + b)] / [c / (c + d)]
    ROR = (a / b) / (c / d) = (a * d) / (b * c)

    se(ln PRR) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    se(ln ROR) = sqrt(1/a + 1/b + 1/c + 1/d)
    CI         = exp(ln(measure) +/- 1.96 * se)

    chi-square = n * (a*d - b*c)^2
                 / ((a+b) * (c+d) * (a+c) * (b+d))

Zero-count handling
-------------------
When any of the four cells is zero, ``ln`` / division would be undefined. The
**Haldane-Anscombe** correction is applied: 0.5 is added to *all four* cells
before the measures and their variances are computed. Every row records whether
the correction was applied (``corrected``), and the raw integer cells are always
written out unchanged so the correction can be undone. This is the standard,
documented fix for sparse 2x2 tables in pharmacovigilance signal detection.

Correctness notes
-----------------
* The drug x reaction join is between two already-distinct projections keyed on
  ``(quarter, PRIMARYID)``, so it produces exactly one row per
  (report, drug, reaction) combination - no duplicate multiplication.
* Q1 and Q2 are analysed separately; the overall table is the exact sum of the
  two per-quarter tables (each report belongs to exactly one quarter).
* Metrics are computed in Spark so the whole pair space is screened without
  collecting it; only the pairs that pass the thresholds are brought to the
  driver, and they are capped.

A signal is **not** evidence of causation, of a real increased risk, or of a
drug-drug interaction. FAERS is a spontaneous reporting system with no
denominator: these numbers describe reporting patterns only, and are subject to
reporting bias, notoriety bias and duplicate reports.

Environment constraints honoured: PySpark DataFrames only, no pandas, no
PyArrow, no UDFs, no Parquet. Raw datasets are never modified.

Usage::

    python -m src.analytics.signal_detection
    python -m src.analytics.signal_detection --quarters 2026_Q1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pyspark.sql import DataFrame
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
from src.utils.config import get_analytics_config, get_path, get_quarter_dirs
from src.utils.logging_setup import get_logger

__all__ = [
    "SignalDetectionError",
    "SIGNAL_DISCLAIMER",
    "SIGNAL_TABLES",
    "SignalPair",
    "SignalResults",
    "analyze",
    "write_outputs",
    "render_report",
    "run",
    "main",
]

#: Tables this module needs (a subset of the seven FAERS tables).
SIGNAL_TABLES: Tuple[str, ...] = ("DEMO", "DRUG", "REAC")

#: Standardised drug-name column produced by the preprocessing module.
DRUG_NAME_COLUMN = "DRUGNAME_STD"

#: MedDRA Preferred Term column produced by the preprocessing module.
REACTION_COLUMN = "PT"

#: z for a two-sided 95% confidence interval.
Z_95 = 1.96

#: Haldane-Anscombe correction added to every cell of a table containing a zero.
HALDANE_ANSCOMBE = 0.5

#: Grain statement reused in the report and the JSON summary.
SIGNAL_GRAIN = (
    "One report = one distinct (quarter, PRIMARYID). A drug/reaction contingency "
    "table is then built from distinct report membership, so multiple drugs or "
    "multiple reactions in one report never multiply a count."
)

#: Mandatory caution; echoed in the report and the JSON summary.
SIGNAL_DISCLAIMER = (
    "A disproportionality signal (PRR/ROR) only means the drug/reaction pair was "
    "reported together more often than the background reporting rate predicts. It "
    "is NOT proof of causation, of a drug-drug interaction, of an increased "
    "clinical risk, or of incidence. FAERS has no exposure denominator and is "
    "affected by reporting, notoriety and duplicate bias."
)

# --- output sizing ---------------------------------------------------------
TOP_SIGNALS_REPORT = 20
TOP_SIGNALS_PER_QUARTER_REPORT = 10
TOP_PAIRS_REPORT = 10
#: Safety cap on the number of signal rows written to CSV (already bounded by the
#: thresholds; the cap only guards against an unexpectedly large signal set).
MAX_SIGNALS_WRITTEN = 200000
#: How many of the strongest signals the JSON summary embeds. The complete
#: tables live in the CSVs, so the JSON only carries a bounded ranked sample.
SUMMARY_TOP_SIGNALS = 100
#: How many signals are re-verified with an independent Spark formulation.
RECOUNT_SAMPLE = 5


class SignalDetectionError(RuntimeError):
    """Raised when the disproportionality analysis cannot be produced."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class SignalPair:
    """One drug/reaction pair with its contingency table and measures.

    ``a``/``b``/``c``/``d`` are the raw integer report counts; ``prr``/``ror``
    and their intervals are derived from the (possibly corrected) cells.
    """

    drug: str
    reaction: str
    a: int
    b: int
    c: int
    d: int
    n_drug: int
    n_reac: int
    n_total: int
    corrected: bool
    prr: float
    prr_ci_low: float
    prr_ci_high: float
    ror: float
    ror_ci_low: float
    ror_ci_high: float
    chi_square: float
    quarter: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.drug} -> {self.reaction}"


@dataclass
class SignalResults:
    """Every artefact produced by one signal-detection run."""

    quarters: List[str] = field(default_factory=list)
    cleanings: Dict[str, TableCleaning] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)

    # section 1 - report / pair counts
    total_reports: int = 0
    total_reports_by_quarter: Dict[str, int] = field(default_factory=dict)
    drug_rows: int = 0
    reaction_rows: int = 0
    distinct_report_drug_rows: int = 0
    distinct_report_reaction_rows: int = 0
    drug_reaction_join_rows: int = 0
    drug_reaction_join_distinct_rows: int = 0
    screened_pairs: int = 0
    screened_pairs_by_quarter: Dict[str, int] = field(default_factory=dict)

    # section 3/4 - signals (capped lists plus true counts)
    signals: List[SignalPair] = field(default_factory=list)
    signals_total: int = 0
    signals_by_quarter: List[SignalPair] = field(default_factory=list)
    signals_by_quarter_total: Dict[str, int] = field(default_factory=dict)
    signals_cap: int = MAX_SIGNALS_WRITTEN
    top_pairs_by_reports: List[Tuple[str, str, int]] = field(default_factory=list)

    # validation
    corrected_rows: int = 0
    duplicate_signals: int = 0
    duplicate_signals_by_quarter: int = 0
    #: largest RELATIVE difference between Spark's measures and the pure-Python
    #: recomputation (a relative bound is used because the CI bounds can be
    #: 1e5-1e6, where float64/JVM rounding is ~1e-13 relative).
    measure_max_rel_diff: float = 0.0
    ci_max_rel_diff: float = 0.0
    contingency_recount_size: int = 0
    contingency_max_abs_diff: int = 0
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # -- helpers ------------------------------------------------------------
    def as_summary(self, generated_at: str) -> Dict[str, Any]:
        return {
            "generated_at": generated_at,
            "quarters": self.quarters,
            "disclaimer": SIGNAL_DISCLAIMER,
            "grain": SIGNAL_GRAIN,
            "thresholds": self.thresholds,
            "counts": {
                "total_reports": self.total_reports,
                "distinct_report_drug_rows": self.distinct_report_drug_rows,
                "distinct_report_reaction_rows": self.distinct_report_reaction_rows,
                "drug_reaction_join_rows": self.drug_reaction_join_rows,
                "drug_reaction_join_distinct_rows": self.drug_reaction_join_distinct_rows,
                "screened_pairs": self.screened_pairs,
                "signals_total": self.signals_total,
                "signals_written": len(self.signals),
                "signals_cap": self.signals_cap,
                "corrected_rows": self.corrected_rows,
            },
            "total_reports_by_quarter": self.total_reports_by_quarter,
            "screened_pairs_by_quarter": self.screened_pairs_by_quarter,
            "signals_by_quarter": self.signals_by_quarter_total,
            "top_signals": [
                self._signal_dict(signal)
                for signal in self.signals[:SUMMARY_TOP_SIGNALS]
            ],
            "checks": [
                {"name": name, "passed": ok, "detail": detail}
                for name, ok, detail in self.checks
            ],
            "errors": self.errors,
        }

    @staticmethod
    def _signal_dict(signal: SignalPair) -> Dict[str, Any]:
        return {
            "drug": signal.drug,
            "reaction": signal.reaction,
            "quarter": signal.quarter,
            "a": signal.a,
            "b": signal.b,
            "c": signal.c,
            "d": signal.d,
            "n_drug": signal.n_drug,
            "n_reac": signal.n_reac,
            "n_total": signal.n_total,
            "corrected": signal.corrected,
            "prr": signal.prr,
            "prr_ci_low": signal.prr_ci_low,
            "prr_ci_high": signal.prr_ci_high,
            "ror": signal.ror,
            "ror_ci_low": signal.ror_ci_low,
            "ror_ci_high": signal.ror_ci_high,
            "chi_square": signal.chi_square,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _count_rows() -> "F.Column":
    """A portable row counter."""
    return F.count(F.lit(1))


def _clip(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` characters, adding an ellipsis when cut."""
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def _clean_inputs(
    spark: Any,
    *,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> Tuple[Dict[str, DataFrame], Dict[str, TableCleaning], List[str]]:
    """Read + clean the DEMO, DRUG and REAC tables via the preprocessing module."""
    log = get_logger(__name__)
    frames: Dict[str, DataFrame] = {}
    cleanings: Dict[str, TableCleaning] = {}
    errors: List[str] = []

    for table in SIGNAL_TABLES:
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
        log.info("%s: %s rows cleaned", table, f"{cleaning.rows_after:,}")
    return frames, cleanings, errors


# ---------------------------------------------------------------------------
# Contingency tables and measures
# ---------------------------------------------------------------------------
def _add_metrics(frame: DataFrame) -> DataFrame:
    """Add the integer cells and the PRR/ROR/CI/chi-square columns.

    ``frame`` must carry ``a``, ``n_drug``, ``n_reac`` and ``n_total``. The
    Haldane-Anscombe (+0.5) correction is applied to all four cells of any table
    that contains a zero, so the log-scale measures stay finite.
    """
    a_i = F.col("a")
    b_i = F.col("n_drug") - F.col("a")
    c_i = F.col("n_reac") - F.col("a")
    d_i = F.col("n_total") - F.col("n_drug") - F.col("n_reac") + F.col("a")

    a = a_i.cast("double")
    b = b_i.cast("double")
    c = c_i.cast("double")
    d = d_i.cast("double")

    zero = (a <= F.lit(0.0)) | (b <= F.lit(0.0)) | (c <= F.lit(0.0)) | (d <= F.lit(0.0))
    half = F.lit(HALDANE_ANSCOMBE)
    a2 = F.when(zero, a + half).otherwise(a)
    b2 = F.when(zero, b + half).otherwise(b)
    c2 = F.when(zero, c + half).otherwise(c)
    d2 = F.when(zero, d + half).otherwise(d)
    n2 = a2 + b2 + c2 + d2

    prr = (a2 / (a2 + b2)) / (c2 / (c2 + d2))
    ror = (a2 * d2) / (b2 * c2)

    one = F.lit(1.0)
    prr_se = F.sqrt(one / a2 - one / (a2 + b2) + one / c2 - one / (c2 + d2))
    ror_se = F.sqrt(one / a2 + one / b2 + one / c2 + one / d2)
    z = F.lit(Z_95)

    cross = a2 * d2 - b2 * c2
    chi_square = (
        n2 * (cross * cross) / ((a2 + b2) * (c2 + d2) * (a2 + c2) * (b2 + d2))
    )

    return (
        frame.withColumn("b", b_i)
        .withColumn("c", c_i)
        .withColumn("d", d_i)
        .withColumn("corrected", zero)
        .withColumn("prr", prr)
        .withColumn("prr_ci_low", prr * F.exp(F.lit(-1.0) * z * prr_se))
        .withColumn("prr_ci_high", prr * F.exp(z * prr_se))
        .withColumn("ror", ror)
        .withColumn("ror_ci_low", ror * F.exp(F.lit(-1.0) * z * ror_se))
        .withColumn("ror_ci_high", ror * F.exp(z * ror_se))
        .withColumn("chi_square", chi_square)
    )


def _signal_condition(thresholds: Mapping[str, float]) -> "F.Column":
    """The configurable signal definition applied in Spark."""
    return (
        (F.col("a") >= F.lit(int(thresholds["min_reports"])))
        & (F.col("prr") >= F.lit(float(thresholds["prr_threshold"])))
        & (F.col("ror_ci_low") > F.lit(float(thresholds["ror_lower_ci_threshold"])))
        & (F.col("chi_square") >= F.lit(float(thresholds["chi_square_threshold"])))
    )


#: Deterministic ranking of signals: strongest PRR first, then the more
#: conservative ROR lower bound, then chi-square, then the pair label.
#: Kept as plain names + an ascending flags list so no Spark expression is
#: built at import time (a SparkContext may not exist yet).
_SIGNAL_ORDER: Tuple[str, ...] = ("prr", "ror_ci_low", "chi_square", "drug", "reaction")
_SIGNAL_ORDER_ASCENDING: Tuple[bool, ...] = (False, False, False, True, True)

#: Column order shared by the two signal CSVs.
_SIGNAL_HEADER = [
    "drug",
    "reaction",
    "a",
    "b",
    "c",
    "d",
    "n_drug",
    "n_reac",
    "n_total",
    "corrected",
    "prr",
    "prr_ci_low",
    "prr_ci_high",
    "ror",
    "ror_ci_low",
    "ror_ci_high",
    "chi_square",
]


def _signal_projection(frame: DataFrame, *, with_quarter: bool) -> DataFrame:
    """Project a metrics frame into the signal CSV/collect shape."""
    quarter = (
        F.col(QUARTER_COLUMN).alias(QUARTER_COLUMN)
        if with_quarter
        else F.lit(None).cast("string").alias(QUARTER_COLUMN)
    )
    return frame.select(
        quarter,
        F.col(DRUG_NAME_COLUMN).alias("drug"),
        F.col(REACTION_COLUMN).alias("reaction"),
        F.col("a"),
        F.col("b"),
        F.col("c"),
        F.col("d"),
        F.col("n_drug"),
        F.col("n_reac"),
        F.col("n_total"),
        F.col("corrected"),
        F.col("prr"),
        F.col("prr_ci_low"),
        F.col("prr_ci_high"),
        F.col("ror"),
        F.col("ror_ci_low"),
        F.col("ror_ci_high"),
        F.col("chi_square"),
    )


def _collect_signals(frame: DataFrame, *, limit: int) -> List[SignalPair]:
    """Collect at most ``limit`` signals (already projected) into dataclasses."""
    rows = (
        frame.orderBy(*_SIGNAL_ORDER, ascending=list(_SIGNAL_ORDER_ASCENDING))
        .limit(limit)
        .collect()
    )
    return [
        SignalPair(
            drug=row["drug"],
            reaction=row["reaction"],
            a=int(row["a"]),
            b=int(row["b"]),
            c=int(row["c"]),
            d=int(row["d"]),
            n_drug=int(row["n_drug"]),
            n_reac=int(row["n_reac"]),
            n_total=int(row["n_total"]),
            corrected=bool(row["corrected"]),
            prr=float(row["prr"]),
            prr_ci_low=float(row["prr_ci_low"]),
            prr_ci_high=float(row["prr_ci_high"]),
            ror=float(row["ror"]),
            ror_ci_low=float(row["ror_ci_low"]),
            ror_ci_high=float(row["ror_ci_high"]),
            chi_square=float(row["chi_square"]),
            quarter=row[QUARTER_COLUMN],
        )
        for row in rows
    ]


def _independent_contingency(
    drug_pairs: DataFrame,
    reaction_pairs: DataFrame,
    sample: Sequence[Tuple[str, str]],
    *,
    n_total: int,
) -> Dict[Tuple[str, str], Tuple[int, int, int, int]]:
    """Recompute sampled contingency tables with a *different* Spark plan.

    Instead of the grouped drug x reaction join used for mining, this counts the
    intersection of the two report sets with ``intersect`` (a distinct-based set
    intersection), and re-derives the marginals with their own ``distinct`` pass.
    Returns ``{(drug, reaction): (a, n_drug, n_reac, N)}``.
    """
    recount: Dict[Tuple[str, str], Tuple[int, int, int, int]] = {}
    for drug, reaction in sample:
        drug_reports = (
            drug_pairs.filter(F.col(DRUG_NAME_COLUMN) == drug)
            .select(QUARTER_COLUMN, "PRIMARYID")
            .distinct()
        )
        reaction_reports = (
            reaction_pairs.filter(F.col(REACTION_COLUMN) == reaction)
            .select(QUARTER_COLUMN, "PRIMARYID")
            .distinct()
        )
        both = drug_reports.intersect(reaction_reports).count()
        recount[(drug, reaction)] = (
            int(both),
            int(drug_reports.count()),
            int(reaction_reports.count()),
            int(n_total),
        )
    return recount


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(
    spark: Any,
    *,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> SignalResults:
    """Build contingency tables and score every drug/reaction pair in FAERS."""
    log = get_logger(__name__)
    results = SignalResults(quarters=list(quarters))

    thresholds = get_analytics_config("signal_detection")
    min_reports = int(thresholds.get("min_reports", 0) or 0)
    prr_threshold = float(thresholds.get("prr_threshold", 0.0) or 0.0)
    ror_threshold = float(thresholds.get("ror_lower_ci_threshold", 0.0) or 0.0)
    chi_square_threshold = float(thresholds.get("chi_square_threshold", 0.0) or 0.0)
    results.thresholds = {
        "min_reports": min_reports,
        "prr_threshold": prr_threshold,
        "ror_lower_ci_threshold": ror_threshold,
        "chi_square_threshold": chi_square_threshold,
        "z_95": Z_95,
        "haldane_anscombe": HALDANE_ANSCOMBE,
    }

    frames, cleanings, errors = _clean_inputs(
        spark, quarters=quarters, settings=settings, quarter_dirs=quarter_dirs
    )
    results.cleanings = cleanings
    results.errors.extend(errors)

    missing = [table for table in SIGNAL_TABLES if table not in frames]
    if missing:
        raise SignalDetectionError(
            "required table(s) unavailable: " + ", ".join(missing) + "; see errors"
        )

    quarter = QUARTER_COLUMN
    reports = frames["DEMO"].select(quarter, "PRIMARYID").distinct().persist()
    drug_pairs = (
        frames["DRUG"]
        .select(quarter, "PRIMARYID", DRUG_NAME_COLUMN)
        .filter(F.col(DRUG_NAME_COLUMN).isNotNull())
        .distinct()
        .persist()
    )
    reaction_pairs = (
        frames["REAC"]
        .select(quarter, "PRIMARYID", REACTION_COLUMN)
        .filter(F.col(REACTION_COLUMN).isNotNull())
        .distinct()
        .persist()
    )

    # The only join in the module: two already-distinct projections keyed on
    # (quarter, PRIMARYID), so the product is one row per (report, drug,
    # reaction) - a report's own distinct drugs x its own distinct reactions,
    # never inflated by duplicate rows.
    joined = (
        drug_pairs.join(
            reaction_pairs.select(quarter, "PRIMARYID", REACTION_COLUMN),
            on=[quarter, "PRIMARYID"],
            how="inner",
        )
        .persist()
    )

    try:
        # ---- 1. report and pair counts -------------------------------------
        results.drug_rows = cleanings["DRUG"].rows_after
        results.reaction_rows = cleanings["REAC"].rows_after
        results.total_reports = reports.select("PRIMARYID").distinct().count()
        results.total_reports_by_quarter = {
            row[quarter]: int(row["reports"])
            for row in reports.groupBy(quarter)
            .agg(F.countDistinct("PRIMARYID").alias("reports"))
            .collect()
        }
        results.distinct_report_drug_rows = drug_pairs.count()
        results.distinct_report_reaction_rows = reaction_pairs.count()
        results.drug_reaction_join_rows = joined.count()
        # A strong no-multiplication check: both join sides are already distinct
        # on (report, item), so the join must contain no duplicate rows.
        results.drug_reaction_join_distinct_rows = joined.distinct().count()
        log.info(
            "reports=%s  report-drug rows=%s  report-reaction rows=%s  join rows=%s",
            f"{results.total_reports:,}",
            f"{results.distinct_report_drug_rows:,}",
            f"{results.distinct_report_reaction_rows:,}",
            f"{results.drug_reaction_join_rows:,}",
        )

        # ---- 2. contingency cells per (quarter, drug, reaction) ------------
        # Deliberately NOT filtered here: a pair may fall below min_reports in
        # each single quarter yet reach it once both quarters are combined, so
        # the per-quarter filter is applied only to the per-quarter table.
        a_by_quarter = (
            joined.groupBy(quarter, DRUG_NAME_COLUMN, REACTION_COLUMN)
            .agg(_count_rows().alias("a"))
            .persist()
        )
        drug_by_quarter = drug_pairs.groupBy(quarter, DRUG_NAME_COLUMN).agg(
            F.countDistinct("PRIMARYID").alias("n_drug")
        )
        reaction_by_quarter = reaction_pairs.groupBy(quarter, REACTION_COLUMN).agg(
            F.countDistinct("PRIMARYID").alias("n_reac")
        )
        n_by_quarter = reports.groupBy(quarter).agg(
            F.countDistinct("PRIMARYID").alias("n_total")
        )

        quarter_frame = (
            a_by_quarter.filter(F.col("a") >= F.lit(min_reports))
            .join(
                F.broadcast(drug_by_quarter),
                on=[quarter, DRUG_NAME_COLUMN],
                how="inner",
            )
            .join(
                F.broadcast(reaction_by_quarter),
                on=[quarter, REACTION_COLUMN],
                how="inner",
            )
            .join(F.broadcast(n_by_quarter), on=[quarter], how="inner")
        )
        quarter_metrics = _add_metrics(quarter_frame).persist()

        # ---- 3. overall contingency cells (exact sum over quarters) --------
        a_overall = (
            a_by_quarter.groupBy(DRUG_NAME_COLUMN, REACTION_COLUMN)
            .agg(F.sum("a").alias("a"))
            .filter(F.col("a") >= F.lit(min_reports))
        )
        drug_overall = drug_by_quarter.groupBy(DRUG_NAME_COLUMN).agg(
            F.sum("n_drug").alias("n_drug")
        )
        reaction_overall = reaction_by_quarter.groupBy(REACTION_COLUMN).agg(
            F.sum("n_reac").alias("n_reac")
        )
        n_overall = n_by_quarter.agg(F.sum("n_total").alias("n_total"))
        overall_frame = (
            a_overall.join(F.broadcast(drug_overall), on=DRUG_NAME_COLUMN, how="inner")
            .join(F.broadcast(reaction_overall), on=REACTION_COLUMN, how="inner")
            .crossJoin(n_overall)
        )
        overall_metrics = _add_metrics(overall_frame).persist()

        # ---- screened pair counts ------------------------------------------
        results.screened_pairs = overall_metrics.count()
        results.screened_pairs_by_quarter = {
            row[quarter]: int(row["pairs"])
            for row in quarter_metrics.groupBy(quarter).agg(_count_rows().alias("pairs")).collect()
        }
        log.info("screened pairs (a >= %s): %s", min_reports, f"{results.screened_pairs:,}")

        # ---- 4. signals meeting the configurable thresholds -----------------
        condition = _signal_condition(results.thresholds)
        signals_overall = _signal_projection(overall_metrics.filter(condition), with_quarter=False)
        results.signals_total = signals_overall.count()
        results.signals = _collect_signals(signals_overall, limit=MAX_SIGNALS_WRITTEN)

        signals_quarter = _signal_projection(
            quarter_metrics.filter(condition), with_quarter=True
        )
        results.signals_by_quarter_total = {
            row[quarter]: int(row["signals"])
            for row in signals_quarter.groupBy(quarter)
            .agg(_count_rows().alias("signals"))
            .collect()
        }
        results.signals_by_quarter = _collect_signals(
            signals_quarter, limit=MAX_SIGNALS_WRITTEN
        )
        log.info(
            "signals: %s overall (%s written), per quarter %s",
            f"{results.signals_total:,}",
            f"{len(results.signals):,}",
            ", ".join(
                f"{q}={results.signals_by_quarter_total.get(q, 0):,}"
                for q in results.quarters
            ),
        )

        # Top pairs by observed report count (frequency, not signal strength).
        totals = (
            a_by_quarter.groupBy(DRUG_NAME_COLUMN, REACTION_COLUMN)
            .agg(F.sum("a").alias("reports"))
            .orderBy(F.desc("reports"), F.asc(DRUG_NAME_COLUMN), F.asc(REACTION_COLUMN))
            .limit(TOP_PAIRS_REPORT)
            .collect()
        )
        results.top_pairs_by_reports = [
            (row[DRUG_NAME_COLUMN], row[REACTION_COLUMN], int(row["reports"]))
            for row in totals
        ]

        # ---- validation support --------------------------------------------
        results.corrected_rows = sum(1 for signal in results.signals if signal.corrected)

        results.duplicate_signals = _duplicate_count(
            (signal.drug, signal.reaction) for signal in results.signals
        )
        results.duplicate_signals_by_quarter = _duplicate_count(
            (signal.quarter, signal.drug, signal.reaction)
            for signal in results.signals_by_quarter
        )

        # Independent recomputation of the measures from the integer cells.
        measure_diff, ci_diff = _recompute_measures(results.signals)
        results.measure_max_rel_diff = measure_diff
        results.ci_max_rel_diff = ci_diff

        # Independent Spark recount of sampled contingency tables.
        sample = [(s.drug, s.reaction) for s in results.signals[:RECOUNT_SAMPLE]]
        recount = _independent_contingency(
            drug_pairs, reaction_pairs, sample, n_total=results.total_reports
        )
        results.contingency_recount_size = len(sample)
        max_diff = 0
        for signal in results.signals[:RECOUNT_SAMPLE]:
            expected = recount.get((signal.drug, signal.reaction))
            if expected is None:
                continue
            max_diff = max(
                max_diff,
                abs(expected[0] - signal.a),
                abs(expected[1] - signal.n_drug),
                abs(expected[2] - signal.n_reac),
            )
        results.contingency_max_abs_diff = max_diff

        results.checks = _validate(results)
    finally:
        for frame in (reports, drug_pairs, reaction_pairs, joined):
            frame.unpersist()
        try:
            a_by_quarter.unpersist()
            quarter_metrics.unpersist()
            overall_metrics.unpersist()
        except Exception:  # noqa: BLE001 - unpersist is best-effort
            pass

    return results


def _duplicate_count(keys: Any) -> int:
    """Number of keys seen more than once."""
    seen: set = set()
    duplicates = 0
    for key in keys:
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _recompute_measures(signals: Sequence[SignalPair]) -> Tuple[float, float]:
    """Recompute PRR/ROR and both confidence intervals from the integer cells.

    Uses the Haldane-Anscombe correction exactly as the Spark path does, then
    compares against the values Spark wrote. Returns the largest *relative*
    difference for the point measures and for the four interval bounds. The
    comparison is relative because the interval bounds can reach magnitudes of
    1e5-1e6, where a small float64 rounding difference between the JVM and
    CPython ``exp``/``sqrt`` is ~1e-13 relative but ~1e-7 absolute; an absolute
    tolerance would flag that pure rounding noise as a failure.
    """
    measure_diff = 0.0
    ci_diff = 0.0

    def _rel(computed: float, actual: float) -> float:
        return abs(computed - actual) / max(abs(actual), 1.0)

    for signal in signals:
        a, b, c, d = float(signal.a), float(signal.b), float(signal.c), float(signal.d)
        if signal.corrected:
            a, b, c, d = a + HALDANE_ANSCOMBE, b + HALDANE_ANSCOMBE, c + HALDANE_ANSCOMBE, d + HALDANE_ANSCOMBE
        prr = (a / (a + b)) / (c / (c + d))
        ror = (a * d) / (b * c)
        prr_se = math.sqrt(1.0 / a - 1.0 / (a + b) + 1.0 / c - 1.0 / (c + d))
        ror_se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
        prr_low = prr * math.exp(-Z_95 * prr_se)
        prr_high = prr * math.exp(Z_95 * prr_se)
        ror_low = ror * math.exp(-Z_95 * ror_se)
        ror_high = ror * math.exp(Z_95 * ror_se)
        measure_diff = max(
            measure_diff,
            _rel(prr, signal.prr),
            _rel(ror, signal.ror),
        )
        ci_diff = max(
            ci_diff,
            _rel(prr_low, signal.prr_ci_low),
            _rel(prr_high, signal.prr_ci_high),
            _rel(ror_low, signal.ror_ci_low),
            _rel(ror_high, signal.ror_ci_high),
        )
    return measure_diff, ci_diff


def _validate(results: SignalResults) -> List[Tuple[str, bool, str]]:
    """Cross-check the pipeline; every check is recorded pass or fail."""
    checks: List[Tuple[str, bool, str]] = []
    thresholds = results.thresholds
    min_reports = int(thresholds.get("min_reports", 0))

    cleaning_ok = all(
        item.rows_before == item.rows_after and not item.errors
        for item in results.cleanings.values()
    )
    checks.append(
        (
            "cleaning removed no rows (DEMO/DRUG/REAC)",
            cleaning_ok,
            "; ".join(
                f"{table}={item.rows_before:,}->{item.rows_after:,}"
                for table, item in results.cleanings.items()
            ),
        )
    )

    checks.append(
        (
            "report-level projections de-duplicate (report, item) rows",
            results.distinct_report_drug_rows > 0
            and results.distinct_report_reaction_rows > 0
            and results.distinct_report_drug_rows <= results.drug_rows
            and results.distinct_report_reaction_rows <= results.reaction_rows,
            f"DRUG {results.drug_rows:,} rows -> {results.distinct_report_drug_rows:,} distinct "
            f"(report, drug) rows; REAC {results.reaction_rows:,} rows -> "
            f"{results.distinct_report_reaction_rows:,} distinct (report, reaction) rows",
        )
    )

    checks.append(
        (
            "drug x reaction join produced no duplicate rows (no multiplication)",
            results.drug_reaction_join_rows > 0
            and results.drug_reaction_join_rows == results.drug_reaction_join_distinct_rows,
            f"{results.drug_reaction_join_rows:,} join row(s), "
            f"{results.drug_reaction_join_distinct_rows:,} distinct row(s), from "
            f"{results.distinct_report_drug_rows:,} distinct report-drug rows and "
            f"{results.distinct_report_reaction_rows:,} distinct report-reaction rows",
        )
    )

    # contingency identities on every collected signal
    identity_ok = all(
        signal.a + signal.b == signal.n_drug
        and signal.a + signal.c == signal.n_reac
        and signal.a + signal.b + signal.c + signal.d == signal.n_total
        and signal.b >= 0
        and signal.c >= 0
        and signal.d >= 0
        and signal.a <= signal.n_drug
        and signal.a <= signal.n_reac
        for signal in results.signals
    )
    checks.append(
        (
            "contingency identities hold (a+b=n_drug, a+c=n_reac, a+b+c+d=N)",
            identity_ok and bool(results.signals),
            f"checked {len(results.signals):,} signal(s); "
            f"{results.corrected_rows:,} needed the Haldane-Anscombe correction and kept raw cells",
        )
    )

    checks.append(
        (
            "every signal meets the observed-count and metric thresholds",
            bool(results.signals)
            and all(
                signal.a >= min_reports
                and signal.prr >= thresholds.get("prr_threshold", 0.0)
                and signal.ror_ci_low > thresholds.get("ror_lower_ci_threshold", 0.0)
                and signal.chi_square >= thresholds.get("chi_square_threshold", 0.0)
                for signal in results.signals
            ),
            f"a >= {min_reports}, PRR >= {thresholds.get('prr_threshold')}, "
            f"ROR CI low > {thresholds.get('ror_lower_ci_threshold')}, "
            f"chi-square >= {thresholds.get('chi_square_threshold')}",
        )
    )

    bounded = all(
        signal.prr > 0.0
        and signal.ror_ci_low >= 0.0
        and math.isfinite(signal.prr)
        and math.isfinite(signal.ror)
        and math.isfinite(signal.chi_square)
        and signal.chi_square >= 0.0
        and signal.ror_ci_low <= signal.ror <= signal.ror_ci_high
        and signal.prr_ci_low <= signal.prr <= signal.prr_ci_high
        for signal in results.signals
    )
    checks.append(
        (
            "measures are finite, positive and inside their confidence intervals",
            bounded and bool(results.signals),
            f"checked {len(results.signals):,} signal(s); "
            f"{results.corrected_rows:,} zero-cell table(s) corrected",
        )
    )

    checks.append(
        (
            "PRR/ROR and 95% CIs recomputed independently from the integer cells",
            results.measure_max_rel_diff <= 1e-9 and results.ci_max_rel_diff <= 1e-9,
            f"max relative |measure| diff = {results.measure_max_rel_diff:.3e}, "
            f"max relative |CI| diff = {results.ci_max_rel_diff:.3e}",
        )
    )

    checks.append(
        (
            "sampled contingency tables recounted independently in Spark",
            results.contingency_max_abs_diff == 0 and results.contingency_recount_size > 0,
            f"{results.contingency_recount_size} table(s) recounted with intersect(), "
            f"max |a/n_drug/n_reac| difference = {results.contingency_max_abs_diff}",
        )
    )

    checks.append(
        (
            "no duplicate drug/reaction signal keys",
            results.duplicate_signals == 0 and results.duplicate_signals_by_quarter == 0,
            f"{len(results.signals):,} overall signal(s) ({results.duplicate_signals} dup), "
            f"{len(results.signals_by_quarter):,} per-quarter signal(s) "
            f"({results.duplicate_signals_by_quarter} dup)",
        )
    )

    checks.append(
        (
            "Q1 and Q2 are scored separately and are distinguishable",
            set(results.signals_by_quarter_total) == set(results.quarters)
            and all(
                results.signals_by_quarter_total.get(q, 0) >= 0 for q in results.quarters
            ),
            ", ".join(
                f"{q}: {results.signals_by_quarter_total.get(q, 0):,} signal(s) from "
                f"{results.screened_pairs_by_quarter.get(q, 0):,} screened pair(s)"
                for q in results.quarters
            ),
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


def _signal_row(signal: SignalPair) -> List[Any]:
    return [
        signal.drug,
        signal.reaction,
        signal.a,
        signal.b,
        signal.c,
        signal.d,
        signal.n_drug,
        signal.n_reac,
        signal.n_total,
        signal.corrected,
        round(signal.prr, 10),
        round(signal.prr_ci_low, 10),
        round(signal.prr_ci_high, 10),
        round(signal.ror, 10),
        round(signal.ror_ci_low, 10),
        round(signal.ror_ci_high, 10),
        round(signal.chi_square, 10),
    ]


def write_outputs(
    results: SignalResults,
    *,
    output_dir: Path,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Path]:
    """Persist the signal tables and the summary; return the written paths."""
    timestamp = (generated_at or datetime.now()).isoformat(timespec="seconds")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    def emit(name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        target = output_dir / name
        _write_csv(target, header, rows)
        written[name] = target

    # signals are already ranked and capped
    emit(
        "signal_detection_signals.csv",
        _SIGNAL_HEADER,
        [_signal_row(signal) for signal in results.signals],
    )
    emit(
        "signal_detection_signals_by_quarter.csv",
        [QUARTER_COLUMN, *_SIGNAL_HEADER],
        [[signal.quarter, *_signal_row(signal)] for signal in results.signals_by_quarter],
    )

    summary_path = output_dir / "signal_detection_summary.json"
    summary_path.write_text(
        json.dumps(results.as_summary(timestamp), indent=2) + "\n", encoding="utf-8"
    )
    written["signal_detection_summary.json"] = summary_path
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(
    results: SignalResults,
    *,
    generated_at: Optional[datetime] = None,
) -> str:
    """Render the signal-detection report as plain text."""
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    thresholds = results.thresholds

    lines: List[str] = [
        "=" * 96,
        "FAERS PRR / ROR DISPROPORTIONALITY SIGNAL DETECTION REPORT",
        f"Generated : {stamp}",
        f"Quarters  : {', '.join(results.quarters)}",
        "Engine    : PySpark DataFrame API only (no pandas, no PyArrow, no UDFs)",
        "Measures  : PRR, ROR, 95% confidence intervals, Pearson chi-square",
        "Source    : real FAERS 2026 Q1/Q2 ASCII extracts, via ingestion + preprocessing",
        "=" * 96,
        "",
        "0. SCOPE AND METHOD",
        "-" * 96,
        f"  {SIGNAL_GRAIN}",
        "  Every cell of the 2x2 table is a count of distinct reports, so a report",
        "  listing a drug twice or a reaction three times still counts once.",
        "  b, c and d come from the report-level marginals:",
        "    b = n_drug - a,  c = n_reac - a,  d = N - n_drug - n_reac + a",
        "  Q1 and Q2 are scored separately; the overall table is the exact sum of the",
        "  two per-quarter tables (each report belongs to exactly one quarter).",
        "",
        "  !! " + SIGNAL_DISCLAIMER,
        "",
        "1. REPORTS AND PAIRS",
        "-" * 96,
        f"  Distinct reports (DEMO, quarter + PRIMARYID) : {results.total_reports:>14,}",
        f"  DRUG rows / distinct (report, drug) rows     : {results.drug_rows:>14,} / {results.distinct_report_drug_rows:,}",
        f"  REAC rows / distinct (report, reaction) rows : {results.reaction_rows:>14,} / {results.distinct_report_reaction_rows:,}",
        f"  Drug x reaction join rows (all distinct)     : {results.drug_reaction_join_rows:>14,}",
        f"  Pairs screened (observed count >= {thresholds.get('min_reports')})      : {results.screened_pairs:>14,}",
        "",
        "  Reports by quarter:",
    ]
    for label in results.quarters:
        lines.append(
            f"    {label}: {results.total_reports_by_quarter.get(label, 0):>12,} report(s), "
            f"{results.screened_pairs_by_quarter.get(label, 0):,} pair(s) screened"
        )

    lines.extend(
        [
            "",
            "2. MEASURES AND CORRECTION",
            "-" * 96,
            "    a = reports with drug and reaction      PRR = [a/(a+b)] / [c/(c+d)]",
            "    b = reports with drug, not reaction     ROR = (a*d) / (b*c)",
            "    c = reports with reaction, not drug     se(ln PRR) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))",
            "    d = reports with neither                se(ln ROR) = sqrt(1/a + 1/b + 1/c + 1/d)",
            f"    95% CI = exp(ln(measure) +/- {Z_95} * se)",
            "    chi-square = N*(a*d-b*c)^2 / [(a+b)(c+d)(a+c)(b+d)]",
            f"  Zero cells: the Haldane-Anscombe correction ({HALDANE_ANSCOMBE} added to all four",
            "  cells) is applied whenever any cell is zero. Raw integer cells are always kept,",
            "  and the 'corrected' column flags the rows it touched.",
            f"  Corrected rows among the reported signals: {results.corrected_rows:,}",
            "",
            f"3. TOP {TOP_SIGNALS_REPORT} SIGNALS (overall, by PRR)",
            "-" * 96,
            f"  {'#':>3}  {'DRUG':<26}{'REACTION':<28}{'a':>7}{'PRR':>9}{'ROR':>10}{'ROR CI LOW':>12}{'CHI2':>11}",
        ]
    )
    for index, signal in enumerate(results.signals[:TOP_SIGNALS_REPORT], start=1):
        lines.append(
            f"  {index:>3}  {_clip(signal.drug, 24):<26}{_clip(signal.reaction, 26):<28}"
            f"{signal.a:>7,}{signal.prr:>9.3f}{signal.ror:>10.3f}"
            f"{signal.ror_ci_low:>12.3f}{signal.chi_square:>11.1f}"
        )
    lines.append("  PRR > 1 and ROR CI low > 1 mean over-reporting relative to the background;")
    lines.append("  a reporting pattern only - not incidence, risk or causation.")

    lines.append("")
    lines.append(f"4. MOST FREQUENTLY REPORTED PAIRS (by observed report count a)")
    lines.append("-" * 96)
    for index, (drug, reaction, reports) in enumerate(results.top_pairs_by_reports, start=1):
        lines.append(
            f"  {index:>3}  {_clip(drug, 34):<36}{_clip(reaction, 34):<36}{reports:>9,}"
        )

    lines.append("")
    lines.append("5. Q1 vs Q2 COMPARISON (scored separately)")
    lines.append("-" * 96)
    for label in results.quarters:
        subset = [s for s in results.signals_by_quarter if s.quarter == label][
            :TOP_SIGNALS_PER_QUARTER_REPORT
        ]
        lines.append(
            f"  {label}: {results.total_reports_by_quarter.get(label, 0):,} report(s), "
            f"{results.screened_pairs_by_quarter.get(label, 0):,} pair(s) screened, "
            f"{results.signals_by_quarter_total.get(label, 0):,} signal(s)"
        )
        for index, signal in enumerate(subset, start=1):
            lines.append(
                f"    {index:>3}  {_clip(signal.drug, 24):<26}{_clip(signal.reaction, 26):<28}"
                f"{signal.a:>7,}{signal.prr:>9.3f}{signal.ror:>10.3f}"
                f"{signal.ror_ci_low:>12.3f}{signal.chi_square:>11.1f}"
            )

    lines.append("")
    lines.append("6. THRESHOLD SCREENING (config.yaml -> analytics.signal_detection)")
    lines.append("-" * 96)
    lines.append(
        f"  observed count a >= {thresholds.get('min_reports')}  "
        f"PRR >= {thresholds.get('prr_threshold')}  "
        f"ROR CI low > {thresholds.get('ror_lower_ci_threshold')}  "
        f"chi-square >= {thresholds.get('chi_square_threshold')}"
    )
    lines.append(
        f"  {results.screened_pairs:,} pair(s) passed the observed-count filter; "
        f"{results.signals_total:,} also passed"
    )
    lines.append(
        f"  every metric threshold. {len(results.signals):,} signal row(s) written "
        f"(cap {results.signals_cap:,})."
    )
    lines.append(
        "  Passing the thresholds only marks pairs worth review: they are screening"
    )
    lines.append("  rules, not a test of causation or clinical risk.")

    lines.append("")
    lines.append("7. VALIDATION CHECKS")
    lines.append("-" * 96)
    for name, ok, detail in results.checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        lines.append(f"         {detail}")

    lines.append("")
    lines.append("8. ISSUES AND ERRORS")
    lines.append("-" * 96)
    if results.errors:
        for error in results.errors:
            lines.append(f"  {error}")
    else:
        lines.append("  none")

    failed = [name for name, ok, _detail in results.checks if not ok]
    lines.append("")
    lines.append("9. SUMMARY")
    lines.append("-" * 96)
    lines.append(
        f"  {len(results.quarters)} quarter(s), {len(results.cleanings)} table(s); "
        f"{results.total_reports:,} reports, {results.screened_pairs:,} pairs screened, "
        f"{results.signals_total:,} signal(s)."
    )
    if results.errors or failed:
        lines.append(
            f"  Status: INCOMPLETE - {len(results.errors)} error(s), "
            f"{len(failed)} failed check(s)."
        )
    else:
        lines.append("  Status: COMPLETE - signal detection finished and all checks passed.")
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
    output_dir: Optional[Path] = None,
) -> SignalResults:
    """Score, persist the outputs (when a directory is given) and return results."""
    log = get_logger(__name__)
    results = analyze(
        spark, quarters=quarters, settings=settings, quarter_dirs=quarter_dirs
    )
    if output_dir is not None:
        written = write_outputs(results, output_dir=output_dir)
        for name, path in written.items():
            log.info("wrote %s -> %s", name, path)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PRR/ROR disproportionality signal detection over the real FAERS "
        "extracts (read-only; no Parquet)."
    )
    parser.add_argument("--quarters", nargs="*", default=None, help="e.g. 2026_Q1 2026_Q2")
    parser.add_argument(
        "--report",
        default=None,
        help="Report path (default reports/signal_detection_report.txt; '' disables it).",
    )
    parser.add_argument(
        "--outputs",
        default=None,
        help="Result directory (default data/analytics/signal_detection; '' disables it).",
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
        output_dir: Optional[Path] = get_path("analytics") / "signal_detection"
    else:
        output_dir = Path(args.outputs) if args.outputs else None

    try:
        spark = create_spark_session(app_name="DrugSafetySignalDetection")
    except SparkEnvironmentError as exc:
        log.error("%s", exc)
        return 1

    try:
        results = run(
            spark,
            quarters=quarters,
            settings=settings,
            quarter_dirs=quarter_dirs,
            output_dir=output_dir,
        )
    except (SignalDetectionError, CleaningError, IngestionError) as exc:
        log.error("Signal detection failed: %s", exc)
        return 2
    finally:
        stop_spark_session(spark)

    report = render_report(results)
    print(report)

    report_path = args.report
    if report_path is None:
        report_path = str(get_path("reports") / "signal_detection_report.txt")
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
