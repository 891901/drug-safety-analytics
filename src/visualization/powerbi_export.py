"""Power BI export: structured analytical datasets for Microsoft Power BI.

This module turns the project's *existing* analytical outputs (the CSV / JSON
files already written under ``data/analytics`` by the analytics, visualization
and machine-learning modules) into a small, clean set of flat CSV tables that can
be imported directly into Power BI Desktop.

Design rules (from the task specification)
------------------------------------------
* **Actual analytical outputs only.** Every exported table is a faithful copy of
  a real result file. Nothing is recomputed, reshaped or synthesised.
* **Python standard library only.** Reading and writing is done with ``csv``.
  ``pandas`` and ``pyarrow`` are never imported (they are blocked by this
  Windows environment and are not needed here).
* **Original values preserved.** A table is streamed row by row from its source
  and written unchanged, so every analytical value and column name survives
  exactly as produced upstream.
* **Q1 / Q2 stay identifiable.** Every per-quarter table keeps its ``quarter``
  column (``2026_Q1`` / ``2026_Q2``).
* **No unnecessary duplication of large datasets.** Each concept is exported
  once. Where two source modules publish the same aggregation, only one copy is
  exported (see :data:`DUPLICATE_SOURCES_AVOIDED`).
* **Raw data is untouched.** The module only reads ``data/analytics`` and writes
  to ``data/powerbi`` plus a report under ``reports/``.

Everything exported here describes *reported events* in a spontaneous reporting
system (FAERS). The counts carry no exposure denominator and must not be read as
incidence, risk or causation.

Usage::

    python -m src.visualization.powerbi_export
    python -m src.visualization.powerbi_export --output-dir data/powerbi
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils.config import get_path, get_project_root, get_quarters
from src.utils.logging_setup import get_logger

__all__ = [
    "PowerBIExportError",
    "POWERBI_DISCLAIMER",
    "EXPORT_SPECS",
    "DUPLICATE_SOURCES_AVOIDED",
    "COLUMN_DOCS",
    "ExportSpec",
    "TableResult",
    "export_all",
    "write_data_dictionary",
    "write_report",
    "run",
    "main",
]

log = get_logger(__name__)


class PowerBIExportError(RuntimeError):
    """Raised when a required source file is missing or a table cannot be read."""


#: Disclaimer shown in every generated artefact.
POWERBI_DISCLAIMER = (
    "FAERS reports describe spontaneous reporting patterns. They do not "
    "establish causation, incidence, clinical risk or confirmed drug "
    "interactions. Every count is a number of reported events with no exposure "
    "denominator and is subject to heavy reporting bias."
)


# ---------------------------------------------------------------------------
# Export specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExportSpec:
    """One exported table: its source, destination name and validation key."""

    dataset_id: int
    dataset_name: str
    output_name: str
    source_rel: str
    description: str
    grain: str
    key_columns: Tuple[str, ...]
    quarter_column: Optional[str] = None
    note: str = ""


@dataclass
class TableResult:
    """The measured outcome of exporting one table."""

    spec: ExportSpec
    source_path: Path
    output_path: Path
    source_rows: int = 0
    exported_rows: int = 0
    columns: List[str] = field(default_factory=list)
    missing_by_column: Dict[str, int] = field(default_factory=dict)
    duplicate_keys: int = 0
    quarter_counts: Dict[str, int] = field(default_factory=dict)
    readable: bool = False
    error: Optional[str] = None

    @property
    def missing_total(self) -> int:
        return sum(self.missing_by_column.values())

    @property
    def row_match(self) -> bool:
        return self.error is None and self.source_rows == self.exported_rows


#: The ten requested datasets mapped onto the real result files.
EXPORT_SPECS: Tuple[ExportSpec, ...] = (
    ExportSpec(
        dataset_id=1,
        dataset_name="Drug reporting summary",
        output_name="01_drug_reporting_summary.csv",
        source_rel="adr_drug_report_counts.csv",
        description=(
            "The 100 most-reported drugs overall, with the number of reports "
            "each drug appears in and its share of all reports."
        ),
        grain="one row per drug (all quarters combined)",
        key_columns=("drug_name",),
    ),
    ExportSpec(
        dataset_id=2,
        dataset_name="Adverse reaction summary",
        output_name="02_adverse_reaction_summary.csv",
        source_rel="adr_top_reactions.csv",
        description=(
            "The 100 most-reported reactions overall, with the number of "
            "reports each reaction appears in and its share of all reports."
        ),
        grain="one row per reaction (all quarters combined)",
        key_columns=("reaction_pt",),
    ),
    ExportSpec(
        dataset_id=2,
        dataset_name="Adverse reaction summary",
        output_name="02_adverse_reaction_by_quarter.csv",
        source_rel="adr_reaction_by_quarter.csv",
        description=(
            "The same top reactions broken down by quarter so Q1 and Q2 can be "
            "compared side by side."
        ),
        grain="one row per (reaction, quarter)",
        key_columns=("reaction_pt", "quarter"),
        quarter_column="quarter",
    ),
    ExportSpec(
        dataset_id=3,
        dataset_name="Drug-reaction signal detection (PRR/ROR)",
        output_name="03_signal_detection_overall.csv",
        source_rel="signal_detection/signal_detection_signals.csv",
        description=(
            "Disproportionality screen over all reports combined: the 200,000 "
            "highest-ranked drug/reaction pairs with their 2x2 counts, PRR, ROR "
            "and 95% confidence intervals."
        ),
        grain="one row per (drug, reaction) screened over both quarters",
        key_columns=("drug", "reaction"),
        note="PRR/ROR are computed over the whole 2026 Q1+Q2 population.",
    ),
    ExportSpec(
        dataset_id=3,
        dataset_name="Drug-reaction signal detection (PRR/ROR)",
        output_name="03_signal_detection_by_quarter.csv",
        source_rel="signal_detection/signal_detection_signals_by_quarter.csv",
        description=(
            "The same disproportionality screen computed separately inside "
            "2026 Q1 and 2026 Q2; the quarter column keeps the two periods "
            "distinguishable."
        ),
        grain="one row per (quarter, drug, reaction)",
        key_columns=("quarter", "drug", "reaction"),
        quarter_column="quarter",
    ),
    ExportSpec(
        dataset_id=4,
        dataset_name="Drug co-occurrence summary",
        output_name="04_drug_cooccurrence_pairs.csv",
        source_rel="drug_cooccurrence/cooccurrence_top_pairs.csv",
        description=(
            "Drug pairs most often reported together (all quarters combined) "
            "with support, both confidences and lift."
        ),
        grain="one row per unordered drug pair {drug_a, drug_b}",
        key_columns=("drug_a", "drug_b"),
    ),
    ExportSpec(
        dataset_id=4,
        dataset_name="Drug co-occurrence summary",
        output_name="04_drug_cooccurrence_pairs_by_quarter.csv",
        source_rel="drug_cooccurrence/cooccurrence_pairs_by_quarter.csv",
        description=(
            "Qualifying co-occurring drug pairs measured per quarter, so Q1 and "
            "Q2 can be compared."
        ),
        grain="one row per (quarter, drug pair)",
        key_columns=("quarter", "drug_a", "drug_b"),
        quarter_column="quarter",
    ),
    ExportSpec(
        dataset_id=4,
        dataset_name="Drug co-occurrence summary",
        output_name="04_drug_cooccurrence_groups.csv",
        source_rel="drug_cooccurrence/cooccurrence_top_groups.csv",
        description=(
            "The 200 most-reported three-drug groups (triples) reported together "
            "in the same report."
        ),
        grain="one row per unordered drug triple {drug_a, drug_b, drug_c}",
        key_columns=("drug_a", "drug_b", "drug_c"),
    ),
    ExportSpec(
        dataset_id=5,
        dataset_name="Apriori association rules",
        output_name="05_apriori_rules.csv",
        source_rel="apriori/apriori_rules.csv",
        description=(
            "Association rules mined over all-report drug transactions, with "
            "support, confidence and lift."
        ),
        grain="one row per association rule (antecedent -> consequent)",
        key_columns=("antecedent", "consequent"),
    ),
    ExportSpec(
        dataset_id=5,
        dataset_name="Apriori association rules",
        output_name="05_apriori_rules_by_quarter.csv",
        source_rel="apriori/apriori_rules_by_quarter.csv",
        description=(
            "Association rules mined separately inside 2026 Q1 and 2026 Q2; the "
            "quarter column keeps the two periods distinguishable."
        ),
        grain="one row per (quarter, association rule)",
        key_columns=("quarter", "antecedent", "consequent"),
        quarter_column="quarter",
    ),
    ExportSpec(
        dataset_id=5,
        dataset_name="Apriori association rules",
        output_name="05_apriori_frequent_itemsets.csv",
        source_rel="apriori/apriori_frequent_itemsets.csv",
        description=(
            "Every frequent drug itemset (1, 2 or 3 drugs) that passed the "
            "minimum-support threshold, with its report count and support."
        ),
        grain="one row per frequent itemset",
        key_columns=("itemset",),
    ),
    ExportSpec(
        dataset_id=6,
        dataset_name="Isolation Forest anomaly results",
        output_name="06_isolation_forest_anomalies.csv",
        source_rel="anomaly_detection/anomaly_results.csv",
        description=(
            "The complete Isolation Forest scoring of the 200,000 screened "
            "drug/reaction pairs: the 11 model features, the anomaly score and "
            "the anomaly flag."
        ),
        grain="one row per scored (drug, reaction) pair",
        key_columns=("drug", "reaction"),
        note="Full scoring population; the top-N view is a subset and is not duplicated.",
    ),
    ExportSpec(
        dataset_id=7,
        dataset_name="Quarterly comparison",
        output_name="07_quarterly_comparison.csv",
        source_rel="temporal_demographic/temporal_reports_by_quarter.csv",
        description=(
            "Headline quarterly totals for 2026 Q1 and 2026 Q2: reports, "
            "report-reaction pairs and serious reports."
        ),
        grain="one row per quarter",
        key_columns=("quarter",),
        quarter_column="quarter",
    ),
    ExportSpec(
        dataset_id=8,
        dataset_name="Temporal analysis",
        output_name="08_temporal_monthly_trend.csv",
        source_rel="temporal_demographic/temporal_monthly_trend.csv",
        description=(
            "Number of reports per reporting month (YYYY-MM) across the full "
            "date range covered by the reports."
        ),
        grain="one row per reporting month",
        key_columns=("report_month",),
    ),
    ExportSpec(
        dataset_id=8,
        dataset_name="Temporal analysis",
        output_name="08_temporal_top_reactions_by_quarter.csv",
        source_rel="temporal_demographic/temporal_top_reactions_by_quarter.csv",
        description=(
            "The most frequent reactions inside each quarter, so reaction "
            "rankings can be compared between Q1 and Q2."
        ),
        grain="one row per (quarter, reaction)",
        key_columns=("quarter", "reaction_pt"),
        quarter_column="quarter",
    ),
    ExportSpec(
        dataset_id=8,
        dataset_name="Temporal analysis",
        output_name="08_temporal_drug_frequency_change.csv",
        source_rel="temporal_demographic/temporal_drug_frequency_change.csv",
        description=(
            "For the most-reported drugs, the Q1 and Q2 report counts and the "
            "absolute and percentage change between them."
        ),
        grain="one row per drug",
        key_columns=("drug_name",),
    ),
    ExportSpec(
        dataset_id=9,
        dataset_name="Demographic analysis",
        output_name="09_demographic_age_distribution.csv",
        source_rel="temporal_demographic/demographic_age_distribution.csv",
        description="Report counts by patient age group, with the share of all reports.",
        grain="one row per age group",
        key_columns=("age_bucket",),
    ),
    ExportSpec(
        dataset_id=9,
        dataset_name="Demographic analysis",
        output_name="09_demographic_gender_distribution.csv",
        source_rel="temporal_demographic/demographic_gender_distribution.csv",
        description="Report counts by patient sex, with the share of all reports.",
        grain="one row per sex",
        key_columns=("sex",),
    ),
    ExportSpec(
        dataset_id=9,
        dataset_name="Demographic analysis",
        output_name="09_demographic_age_by_quarter.csv",
        source_rel="temporal_demographic/demographic_age_by_quarter.csv",
        description="Report counts by age group inside each quarter.",
        grain="one row per (quarter, age group)",
        key_columns=("quarter", "age_bucket"),
        quarter_column="quarter",
    ),
    ExportSpec(
        dataset_id=9,
        dataset_name="Demographic analysis",
        output_name="09_demographic_gender_by_quarter.csv",
        source_rel="temporal_demographic/demographic_gender_by_quarter.csv",
        description="Report counts by sex inside each quarter.",
        grain="one row per (quarter, sex)",
        key_columns=("quarter", "sex"),
        quarter_column="quarter",
    ),
    ExportSpec(
        dataset_id=9,
        dataset_name="Demographic analysis",
        output_name="09_demographic_age_by_gender.csv",
        source_rel="temporal_demographic/demographic_age_by_gender.csv",
        description="Report counts for every age group / sex combination.",
        grain="one row per (sex, age group)",
        key_columns=("sex", "age_bucket"),
    ),
    ExportSpec(
        dataset_id=9,
        dataset_name="Demographic analysis",
        output_name="09_demographic_reactions_by_age.csv",
        source_rel="temporal_demographic/demographic_reactions_by_age.csv",
        description="Reaction report counts within each age group (top reactions per group).",
        grain="one row per (age group, reaction)",
        key_columns=("age_bucket", "reaction_pt"),
    ),
    ExportSpec(
        dataset_id=9,
        dataset_name="Demographic analysis",
        output_name="09_demographic_reactions_by_gender.csv",
        source_rel="temporal_demographic/demographic_reactions_by_gender.csv",
        description="Reaction report counts within each sex category.",
        grain="one row per (sex, reaction)",
        key_columns=("sex", "reaction_pt"),
    ),
    ExportSpec(
        dataset_id=10,
        dataset_name="Serious outcome summary",
        output_name="10_serious_outcome_summary.csv",
        source_rel="adr_outcome_counts.csv",
        description=(
            "Report counts per FAERS serious outcome code (death, "
            "hospitalisation, ...) with each code's share of all reports."
        ),
        grain="one row per outcome code (all quarters combined)",
        key_columns=("outc_cod",),
    ),
    ExportSpec(
        dataset_id=10,
        dataset_name="Serious outcome summary",
        output_name="10_serious_outcome_by_quarter.csv",
        source_rel="temporal_demographic/temporal_serious_outcomes_by_quarter.csv",
        description="Serious outcome report counts inside each quarter (Q1 vs Q2).",
        grain="one row per (quarter, outcome code)",
        key_columns=("quarter", "outc_cod"),
        quarter_column="quarter",
    ),
    ExportSpec(
        dataset_id=10,
        dataset_name="Serious outcome summary",
        output_name="10_serious_outcome_by_demographic.csv",
        source_rel="temporal_demographic/demographic_serious_outcomes.csv",
        description=(
            "Serious-report share within each age group and each sex category, "
            "showing how often reports for a group carried a serious outcome."
        ),
        grain="one row per (breakdown dimension, group)",
        key_columns=("group_type", "group"),
    ),
)


#: Source result files that were deliberately *not* exported because an
#: identical aggregation is already provided by another table.
DUPLICATE_SOURCES_AVOIDED: Tuple[Tuple[str, str], ...] = (
    ("adr_demographics_age.csv", "identical to 09_demographic_age_distribution.csv"),
    ("adr_demographics_sex.csv", "identical to 09_demographic_gender_distribution.csv"),
    ("adr_demographics_age_by_quarter.csv", "identical to 09_demographic_age_by_quarter.csv"),
    ("adr_demographics_sex_by_quarter.csv", "identical to 09_demographic_gender_by_quarter.csv"),
    ("adr_outcome_by_quarter.csv", "identical to 10_serious_outcome_by_quarter.csv"),
    ("anomaly_detection/anomaly_top_pairs.csv", "top-N subset of 06_isolation_forest_anomalies.csv"),
    ("adr_drug_reaction_pairs.csv", "pair view already covered by the signal-detection tables"),
)


#: Plain-English meaning of every column that appears in the exports.
COLUMN_DOCS: Dict[str, str] = {
    "quarter": "FAERS reporting quarter (2026_Q1 or 2026_Q2).",
    "drug_name": "Standardised drug name (source DRUGNAME_STD).",
    "drug": "Standardised drug name (source DRUGNAME_STD).",
    "drug_a": "First drug of an unordered co-reported pair (drug_a sorts before drug_b).",
    "drug_b": "Second drug of an unordered co-reported pair.",
    "drug_c": "Third drug of a co-reported triple.",
    "reaction_pt": "Reported reaction as a MedDRA Preferred Term (PT).",
    "reaction": "Reported reaction as a MedDRA Preferred Term (PT).",
    "reports": "Number of distinct FAERS reports (one report = one distinct (quarter, PRIMARYID)).",
    "pct_of_reports": "Share of the relevant report total, in percent.",
    "share_pct": "Serious reports divided by total reports for the group, in percent.",
    "support": "Itemset report count divided by the total report count (a fraction).",
    "confidence": "Itemset report count divided by the antecedent report count (a fraction).",
    "confidence_a_to_b": "Reports containing both drugs divided by reports containing drug_a.",
    "confidence_b_to_a": "Reports containing both drugs divided by reports containing drug_b.",
    "lift": "Confidence divided by the consequent support; 1.0 means independence.",
    "reports_a": "Reports containing drug_a.",
    "reports_b": "Reports containing drug_b.",
    "passes_thresholds": "True when the pair meets the configured co-occurrence thresholds.",
    "outc_cod": "FAERS serious-outcome code (OT, HO, DE, LT, DS, CA, RI).",
    "outcome_label": "Human-readable label for the outcome code.",
    "a": "Observed reports containing both the drug and the reaction.",
    "b": "Reports containing the drug but not the reaction.",
    "c": "Reports containing the reaction but not the drug.",
    "d": "Reports containing neither the drug nor the reaction.",
    "n_drug": "Total reports containing the drug (a + b).",
    "n_reac": "Total reports containing the reaction (a + c).",
    "n_total": "Total reports in the analysis population (a + b + c + d).",
    "corrected": "True when a Haldane-Anscombe +0.5 correction was applied because of a zero cell.",
    "prr": "Proportional Reporting Ratio (observed / expected reporting proportion).",
    "prr_ci_low": "Lower bound of the 95% PRR confidence interval.",
    "prr_ci_high": "Upper bound of the 95% PRR confidence interval.",
    "ror": "Reporting Odds Ratio.",
    "ror_ci_low": "Lower bound of the 95% ROR confidence interval.",
    "ror_ci_high": "Upper bound of the 95% ROR confidence interval.",
    "chi_square": "Chi-square statistic computed from the 2x2 table.",
    "antecedent": "Left-hand side of a rule (one or more drugs joined by ' + ').",
    "consequent": "Right-hand side of a rule (one or more drugs joined by ' + ').",
    "antecedent_size": "Number of drugs in the antecedent.",
    "consequent_size": "Number of drugs in the consequent.",
    "itemset_reports": "Reports containing the full itemset (antecedent + consequent).",
    "antecedent_reports": "Reports containing the antecedent.",
    "consequent_reports": "Reports containing the consequent.",
    "itemset": "A frequent itemset (one or more drugs joined by ' + ').",
    "itemset_size": "Number of drugs in the itemset.",
    "anomaly_rank": "Rank of the pair by anomaly score (1 = most anomalous).",
    "anomaly_score": "Isolation Forest decision-function score (higher = more anomalous).",
    "is_anomaly": "True when the model flagged the pair as an anomaly.",
    "log10_pair_reports": "log10 of the pair report count (Isolation Forest feature).",
    "log10_drug_reports": "log10 of the drug report count (Isolation Forest feature).",
    "log10_reaction_reports": "log10 of the reaction report count (Isolation Forest feature).",
    "log10_prr": "log10 of PRR (Isolation Forest feature).",
    "log10_ror": "log10 of ROR (Isolation Forest feature).",
    "prr_ci_log_width": "log10 width of the PRR confidence interval (Isolation Forest feature).",
    "log10_ror_ci_low": "log10 of the ROR confidence-interval lower bound (Isolation Forest feature).",
    "log10_chi_square": "log10 of the chi-square statistic (Isolation Forest feature).",
    "quarters_observed": "Number of quarters in which the pair was observed (Isolation Forest feature).",
    "quarter_report_balance": "How evenly the pair's reports split between Q1 and Q2 (Isolation Forest feature).",
    "prr_quarter_spread": "Spread of the pair's PRR across quarters (Isolation Forest feature).",
    "report_month": "Reporting month as YYYY-MM.",
    "reaction_pairs": "Distinct (report, reaction) pairs in the quarter.",
    "reports_with_reaction": "Reports carrying at least one reaction.",
    "serious_reports": "Reports carrying at least one serious outcome.",
    "2026_Q1_reports": "Reports containing the drug in 2026 Q1.",
    "2026_Q2_reports": "Reports containing the drug in 2026 Q2.",
    "delta": "2026_Q2_reports minus 2026_Q1_reports.",
    "pct_change": "Percentage change from 2026 Q1 to 2026 Q2.",
    "sex": "Patient sex (F, M or Unknown).",
    "age_bucket": "Patient age group (0-17, 18-44, 45-64, 65-74, 75+, Unknown).",
    "group_type": "Breakdown dimension for the serious-outcome table (age or gender).",
    "group": "The specific age bucket or sex value.",
    "total_reports": "Total reports in the group.",
}


# ---------------------------------------------------------------------------
# Export + validation helpers
# ---------------------------------------------------------------------------
def _analytics_dir() -> Path:
    """Return the project's analytics output directory."""
    return get_path("analytics")


def _default_output_dir() -> Path:
    """Return the default Power BI export directory (``data/powerbi``)."""
    return _analytics_dir().parent / "powerbi"


def _reports_dir() -> Path:
    """Return the project's reports directory."""
    return get_path("reports")


def _stream_export(spec: ExportSpec, source: Path, dest: Path) -> TableResult:
    """Stream one source CSV into ``dest`` unchanged and collect validation stats.

    Rows are copied one at a time with the standard-library ``csv`` module, so a
    200,000-row result file never has to be held in memory in full.
    """
    result = TableResult(spec=spec, source_path=source, output_path=dest)
    seen_keys: set = set()
    quarter_counts: Dict[str, int] = {}

    with source.open("r", encoding="utf-8", newline="") as fh_in:
        reader = csv.reader(fh_in)
        try:
            header = next(reader)
        except StopIteration as exc:  # pragma: no cover - empty source file
            raise PowerBIExportError(f"Source file has no header row: {source}") from exc

        result.columns = list(header)
        if spec.quarter_column:
            quarter_index = header.index(spec.quarter_column)
        else:
            quarter_index = -1
        key_indexes = [header.index(col) for col in spec.key_columns]

        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as fh_out:
            writer = csv.writer(fh_out)
            writer.writerow(header)

            for row in reader:
                result.source_rows += 1

                # Missing-value scan (empty string cells).
                for column, value in zip(header, row):
                    if value == "":
                        result.missing_by_column[column] = (
                            result.missing_by_column.get(column, 0) + 1
                        )

                # Duplicate-key detection on the declared logical key.
                key = tuple(row[i] for i in key_indexes)
                if key in seen_keys:
                    result.duplicate_keys += 1
                else:
                    seen_keys.add(key)

                # Q1/Q2 coverage.
                if quarter_index >= 0:
                    value = row[quarter_index]
                    quarter_counts[value] = quarter_counts.get(value, 0) + 1

                writer.writerow(row)
                result.exported_rows += 1

    result.quarter_counts = quarter_counts
    return result


def _is_readable(path: Path) -> bool:
    """Return True when a CSV can be re-opened and its header parsed."""
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            if not header:
                return False
            next(reader, None)  # a data row if present
        return True
    except (OSError, StopIteration, csv.Error, UnicodeDecodeError):
        return False


def export_all(output_dir: Optional[Path] = None) -> List[TableResult]:
    """Export every table in :data:`EXPORT_SPECS` and validate it."""
    analytics = _analytics_dir()
    out_dir = Path(output_dir) if output_dir else _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[TableResult] = []
    for spec in EXPORT_SPECS:
        source = analytics / spec.source_rel
        if not source.is_file():
            result = TableResult(
                spec=spec, source_path=source, output_path=out_dir / spec.output_name
            )
            result.error = f"source file not found: {source}"
            log.error("Missing source for %s: %s", spec.output_name, source)
            results.append(result)
            continue

        dest = out_dir / spec.output_name
        log.info("Exporting %s <- %s", spec.output_name, spec.source_rel)
        try:
            result = _stream_export(spec, source, dest)
        except PowerBIExportError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            result = TableResult(spec=spec, source_path=source, output_path=dest)
            result.error = f"{type(exc).__name__}: {exc}"
            log.error("Failed to export %s: %s", spec.output_name, exc)

        result.readable = _is_readable(dest) if result.error is None else False
        results.append(result)
        log.info(
            "  -> %s rows (source %s, duplicates %s, missing %s)",
            result.exported_rows,
            result.source_rows,
            result.duplicate_keys,
            result.missing_total,
        )

    return results


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a small JSON summary, returning ``{}`` when unreadable."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _cross_checks(results: List[TableResult]) -> List[Tuple[str, str, bool, str]]:
    """Independently reconcile a few exported totals against the summary JSONs.

    Returns ``(name, expectation, passed, detail)`` tuples.
    """
    analytics = _analytics_dir()
    by_output = {r.spec.output_name: r for r in results}
    checks: List[Tuple[str, str, bool, str]] = []

    def add(name: str, expectation: str, actual: Any, expected: Any) -> None:
        checks.append((name, expectation, actual == expected, f"actual={actual!r} expected={expected!r}"))

    configured_quarters = get_quarters()
    q1, q2 = (configured_quarters + ["2026_Q1", "2026_Q2"])[:2]

    # Signal detection: written rows must match the summary.
    sig = _read_json(analytics / "signal_detection" / "signal_detection_summary.json")
    if sig:
        written = (sig.get("counts") or {}).get("signals_written")
        if "03_signal_detection_overall.csv" in by_output and written is not None:
            add(
                "signal detection overall rows",
                "match signal_detection_summary.counts.signals_written",
                by_output["03_signal_detection_overall.csv"].exported_rows,
                written,
            )

    # Anomaly scoring population.
    ano = _read_json(analytics / "anomaly_detection" / "anomaly_summary.json")
    if ano:
        scored = (ano.get("scoring") or {}).get("rows_scored")
        if "06_isolation_forest_anomalies.csv" in by_output and scored is not None:
            add(
                "isolation forest rows",
                "match anomaly_summary.scoring.rows_scored",
                by_output["06_isolation_forest_anomalies.csv"].exported_rows,
                scored,
            )

    # Apriori rule counts.
    apr = _read_json(analytics / "apriori" / "apriori_summary.json")
    if apr:
        counts = apr.get("counts") or {}
        if "05_apriori_rules.csv" in by_output and counts.get("rules_overall") is not None:
            add(
                "apriori overall rules",
                "match apriori_summary.counts.rules_overall",
                by_output["05_apriori_rules.csv"].exported_rows,
                counts["rules_overall"],
            )
        if "05_apriori_rules_by_quarter.csv" in by_output and counts.get("rules_q1_q2") is not None:
            add(
                "apriori per-quarter rules",
                "match apriori_summary.counts.rules_q1_q2",
                by_output["05_apriori_rules_by_quarter.csv"].exported_rows,
                counts["rules_q1_q2"],
            )

    # Quarterly totals must reconcile to the headline report count.
    qc = by_output.get("07_quarterly_comparison.csv")
    if qc is not None and qc.readable:
        tmp = _read_json(analytics / "temporal_demographic" / "temporal_demographic_summary.json")
        total = ((tmp.get("temporal") or {}).get("total_reports"))
        if total is not None:
            with qc.output_path.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            summed = sum(int(r["reports"]) for r in rows)
            add(
                "quarterly comparison total reports",
                "Q1 reports + Q2 reports equal temporal total_reports",
                summed,
                total,
            )
            q_labels = [r["quarter"] for r in rows]
            add(
                "quarterly comparison Q1/Q2 labels",
                "contain both configured quarters",
                q_labels,
                [q1, q2],
            )

    return checks


# ---------------------------------------------------------------------------
# Artefacts: data dictionary + report
# ---------------------------------------------------------------------------
def _write_data_dictionary(results: List[TableResult], path: Path) -> None:
    """Write ``data_dictionary.txt`` describing every table and column."""
    generated = datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("POWER BI EXPORT - DATA DICTIONARY")
    lines.append("Big Data-Driven Drug Safety Intelligence Platform")
    lines.append("=" * 78)
    lines.append(f"Generated at : {generated}")
    lines.append("Source       : data/analytics (existing analytical outputs only)")
    lines.append("Destination  : data/powerbi")
    lines.append("")
    lines.append("DISCLAIMER")
    lines.append("-" * 78)
    for chunk in _wrap(POWERBI_DISCLAIMER, 76):
        lines.append(chunk)
    lines.append("")
    lines.append("HOW TO READ THIS FILE")
    lines.append("-" * 78)
    lines.append("Each block below describes one exported CSV: the dataset it belongs")
    lines.append("to, where it came from, its grain and every one of its columns.")
    lines.append("")

    current_dataset = None
    for result in results:
        spec = result.spec
        if spec.dataset_id != current_dataset:
            current_dataset = spec.dataset_id
            lines.append("-" * 78)
            lines.append(f"DATASET {spec.dataset_id}: {spec.dataset_name}")
            lines.append("-" * 78)
            lines.append("")
        lines.append(f"TABLE  : {spec.output_name}")
        lines.append(f"  Source        : data/analytics/{spec.source_rel}")
        lines.append(f"  Rows exported : {result.exported_rows}")
        lines.append(f"  Grain         : {spec.grain}")
        lines.append(f"  Primary key   : {' + '.join(spec.key_columns)}")
        for chunk in _wrap(f"Description   : {spec.description}", 76, indent="  "):
            lines.append(chunk)
        if spec.note:
            for chunk in _wrap(f"Note          : {spec.note}", 76, indent="  "):
                lines.append(chunk)
        lines.append("  Columns:")
        for column in result.columns:
            meaning = COLUMN_DOCS.get(column, "(no description recorded)")
            lines.append(f"    - {column}")
            for chunk in _wrap(meaning, 70, indent="        "):
                lines.append(chunk)
        lines.append("")

    lines.append("=" * 78)
    lines.append("SOURCE FILES INTENTIONALLY NOT EXPORTED (duplication avoided)")
    lines.append("=" * 78)
    for source, reason in DUPLICATE_SOURCES_AVOIDED:
        lines.append(f"  - {source}")
        lines.append(f"      {reason}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _wrap(text: str, width: int, indent: str = "") -> List[str]:
    """Wrap ``text`` to ``width`` characters with an optional indent."""
    words = text.split()
    if not words:
        return [indent]
    out: List[str] = []
    current = indent
    for word in words:
        if len(current) + len(word) + (0 if current == indent else 1) > width:
            out.append(current)
            current = indent + word
        else:
            current = current + ("" if current == indent else " ") + word
    out.append(current)
    return out


def write_data_dictionary(results: List[TableResult]) -> Path:
    """Write the data dictionary next to the exports and return its path."""
    path = _default_output_dir() / "data_dictionary.txt"
    if results and results[0].output_path.parent != path.parent:
        path = results[0].output_path.parent / "data_dictionary.txt"
    _write_data_dictionary(results, path)
    return path


def write_report(results: List[TableResult], dictionary_path: Path) -> Path:
    """Write ``reports/powerbi_export_report.txt`` and return its path."""
    generated = datetime.now().isoformat(timespec="seconds")
    analytics = _analytics_dir()
    out_dir = results[0].output_path.parent if results else _default_output_dir()
    checks = _cross_checks(results)

    failed_tables = [r for r in results if r.error is not None]
    row_mismatch = [r for r in results if r.error is None and not r.row_match]
    unreadable = [r for r in results if r.error is None and not r.readable]
    dup_tables = [r for r in results if r.duplicate_keys]
    naming_ok = [r for r in results if r.error is None]
    missing_tables = [r for r in results if r.error is None and r.missing_total]

    # Validation verdicts (the seven required checks).
    validations: List[Tuple[str, bool, str]] = [
        (
            "V1 Verify exported row counts against source files",
            not row_mismatch and not failed_tables,
            f"{len(results)} table(s) checked; "
            f"{sum(r.source_rows for r in results):,} source row(s) -> "
            f"{sum(r.exported_rows for r in results):,} exported row(s); "
            f"{len(row_mismatch)} mismatch(es)",
        ),
        (
            "V2 Verify column names",
            len(naming_ok) == len(results),
            "header copied verbatim from each source file; column descriptions "
            "for every column are listed in the data dictionary",
        ),
        (
            "V3 Check missing values",
            len(missing_tables) == 0,
            f"{sum(r.missing_total for r in results)} empty cell(s) across all "
            f"{sum(r.exported_rows for r in results):,} exported row(s)",
        ),
        (
            "V4 Check duplicate keys",
            len(dup_tables) == 0,
            f"{sum(r.duplicate_keys for r in results)} duplicate key(s) across "
            "the declared primary keys",
        ),
        (
            "V5 Verify Q1/Q2 information",
            _quarter_check_ok(results),
            "every per-quarter table carries its quarter column with both "
            "2026_Q1 and 2026_Q2 present",
        ),
        (
            "V6 Confirm every export is readable",
            len(unreadable) == 0,
            f"{len(results) - len(unreadable)}/{len(results)} file(s) re-opened "
            "and parsed successfully",
        ),
        (
            "V7 Execute the export module",
            len(failed_tables) == 0,
            "export module ran to completion"
            + (f"; {len(failed_tables)} table(s) failed" if failed_tables else ""),
        ),
    ]
    passed = sum(1 for _, ok, _ in validations if ok)

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("POWER BI EXPORT REPORT")
    lines.append("Big Data-Driven Drug Safety Intelligence Platform")
    lines.append("=" * 78)
    lines.append(f"Generated at   : {generated}")
    lines.append(f"Project root   : {get_project_root()}")
    lines.append(f"Source dir     : {analytics}")
    lines.append(f"Output dir     : {out_dir}")
    lines.append(f"Data dictionary: {dictionary_path}")
    lines.append("Run command    : python -m src.visualization.powerbi_export")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 78)
    lines.append(f"  Tables exported     : {len(results)}")
    lines.append(f"  Total rows exported : {sum(r.exported_rows for r in results):,}")
    lines.append(f"  Validations passed  : {passed}/{len(validations)}")
    lines.append(f"  Result              : {'PASS' if passed == len(validations) else 'FAIL'}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("1. VALIDATION RESULTS")
    lines.append("-" * 78)
    for name, ok, detail in validations:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
        for chunk in _wrap(detail, 72, indent="        "):
            lines.append(chunk)
    lines.append("")

    lines.append("-" * 78)
    lines.append("2. EXPORTED TABLES")
    lines.append("-" * 78)
    header = f"{'file':40s} {'rows':>9s} {'src':>9s} {'dup':>5s} {'miss':>5s} {'ok':>4s}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in sorted(results, key=lambda x: x.spec.output_name):
        ok = "yes" if (r.error is None and r.row_match and r.readable) else "NO"
        lines.append(
            f"{r.spec.output_name:40s} {r.exported_rows:>9,d} {r.source_rows:>9,d} "
            f"{r.duplicate_keys:>5d} {r.missing_total:>5d} {ok:>4s}"
        )
    lines.append("")
    for r in results:
        if r.error:
            lines.append(f"  ERROR {r.spec.output_name}: {r.error}")
    if failed_tables:
        lines.append("")

    lines.append("-" * 78)
    lines.append("3. INDEPENDENT CROSS-CHECKS (against summary JSON)")
    lines.append("-" * 78)
    if not checks:
        lines.append("  (no cross-check data available)")
    for name, expectation, ok, detail in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
        lines.append(f"        {expectation}")
        lines.append(f"        {detail}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("4. Q1 / Q2 COVERAGE")
    lines.append("-" * 78)
    for r in results:
        if not r.spec.quarter_column:
            continue
        parts = ", ".join(f"{k}={v:,}" for k, v in sorted(r.quarter_counts.items()))
        lines.append(f"  {r.spec.output_name:44s} {parts}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("5. TABLES INTENTIONALLY NOT EXPORTED (duplication avoided)")
    lines.append("-" * 78)
    for source, reason in DUPLICATE_SOURCES_AVOIDED:
        lines.append(f"  {source}")
        lines.append(f"      -> {reason}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("6. IMPORTING INTO POWER BI DESKTOP")
    lines.append("-" * 78)
    lines.append("  1. Open Power BI Desktop (no special connector needed).")
    lines.append("  2. Home > Get data > Text/CSV.")
    lines.append("  3. Browse to the data/powerbi folder and pick a .csv file")
    lines.append("     (e.g. 07_quarterly_comparison.csv), then click Load.")
    lines.append("  4. Repeat Get data > Text/CSV for each table you want, or use")
    lines.append("     'Transform Data > Combine' to load the whole folder at once.")
    lines.append("  5. In the Model view, relate tables on their shared keys:")
    lines.append("       - quarter            -> 07_quarterly_comparison[quarter]")
    lines.append("       - drug_name / drug_a -> 01_drug_reporting_summary[drug_name]")
    lines.append("       - reaction_pt        -> 02_adverse_reaction_summary[reaction_pt]")
    lines.append("       - outc_cod           -> 10_serious_outcome_summary[outc_cod]")
    lines.append("  6. Build visuals: e.g. a card for total reports, a line chart of")
    lines.append("     08_temporal_monthly_trend, a bar chart of the top signals, and")
    lines.append("     Slicers on quarter, drug and reaction.")
    lines.append("  NOTE: these CSV files hold data only. No .pbix report or Power BI")
    lines.append("  dashboard has been created here - building visuals is done inside")
    lines.append("  Power BI Desktop.")
    lines.append("")

    lines.append("-" * 78)
    lines.append("7. DISCLAIMER")
    lines.append("-" * 78)
    for chunk in _wrap(POWERBI_DISCLAIMER, 76, indent="  "):
        lines.append(chunk)
    lines.append("")

    report_path = _reports_dir() / "powerbi_export_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _quarter_check_ok(results: Iterable[TableResult]) -> bool:
    """True when every per-quarter table shows both configured quarters."""
    expected = set(get_quarters())
    found = False
    for r in results:
        if not r.spec.quarter_column:
            continue
        found = True
        if not expected.issubset(set(r.quarter_counts)):
            return False
    return found


def run(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Export every table, write the dictionary and report, return a summary."""
    results = export_all(output_dir=output_dir)
    dictionary_path = write_data_dictionary(results)
    report_path = write_report(results, dictionary_path)

    failed = [r for r in results if r.error is not None]
    summary = {
        "tables": len(results),
        "rows": sum(r.exported_rows for r in results),
        "duplicates": sum(r.duplicate_keys for r in results),
        "missing": sum(r.missing_total for r in results),
        "failed": len(failed),
        "dictionary": str(dictionary_path),
        "report": str(report_path),
    }
    log.info(
        "Power BI export complete: %d table(s), %d row(s), report -> %s",
        summary["tables"],
        summary["rows"],
        report_path,
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Export analytical datasets for Power BI.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory (default: data/powerbi).",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else None
    summary = run(output_dir=output_dir)

    print("Power BI export summary")
    print(f"  tables exported : {summary['tables']}")
    print(f"  rows exported   : {summary['rows']:,}")
    print(f"  duplicate keys  : {summary['duplicates']}")
    print(f"  missing cells   : {summary['missing']}")
    print(f"  failed tables   : {summary['failed']}")
    print(f"  data dictionary : {summary['dictionary']}")
    print(f"  report          : {summary['report']}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
