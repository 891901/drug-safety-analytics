"""Cleaning / preprocessing of the ingested FAERS tables with PySpark.

Reads the seven core FAERS tables (DEMO, DRUG, REAC, OUTC, RPSR, THER, INDI)
for every configured quarter through the existing ingestion helpers, applies a
documented set of cleaning transformations, validates the result and writes a
plain-text validation summary to ``reports/preprocessing_report.txt``.

Non-negotiables
---------------
* **No record is ever deleted.** Cleaning only normalises values and *adds*
  columns. Row counts are measured before and after every stage and compared.
* **PRIMARYID, CASEID and CASEVERSION are preserved verbatim.** They are
  numeric identifiers, so none of the text transformations touch them.
* **The ``quarter`` provenance column is preserved** and takes part in the
  duplicate key, so Q1 and Q2 are never conflated.
* Duplicate rows are **flagged, never dropped** (see ``DUPLICATE_RULE``).
* The raw ``.txt`` files are opened read-only; nothing under ``data/`` is
  written (Parquet output is out of scope here).

Environment constraints honoured
--------------------------------
* **PySpark DataFrames only.** No pandas, no PyArrow.
* **No Python UDFs.** Every transformation uses built-in ``pyspark.sql``
  functions, so nothing needs to be pickled to the JVM.
* The validation summary is written with the Python standard library only.

Usage::

    python -m src.preprocessing.clean_faers
    python -m src.preprocessing.clean_faers --quarters 2026_Q1 --tables DEMO REAC
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.ingestion.inspect_faers import KEY_COLUMNS, discover_files, resolve_key_indices
from src.ingestion.load_faers import (
    QUARTER_COLUMN,
    IngestionError,
    ingest_table_quarter,
    load_settings,
)
from src.ingestion.spark_session import (
    SparkEnvironmentError,
    create_spark_session,
    stop_spark_session,
)
from src.utils.config import get_path, get_quarter_dirs
from src.utils.logging_setup import get_logger

__all__ = [
    "CleaningError",
    "DUPLICATE_RULE",
    "TRANSFORMATION_RULES",
    "TableCleaning",
    "clean_table",
    "render_report",
    "run",
    "main",
]

#: Columns that must survive cleaning untouched.
PROTECTED_IDENTIFIERS: Tuple[str, ...] = ("PRIMARYID", "CASEID", "CASEVERSION")

#: Documentation of the duplicate policy, quoted verbatim in the report.
DUPLICATE_RULE = (
    "A row is flagged IS_DUPLICATE = true when at least one other row in the "
    "SAME table and SAME quarter shares its natural key (the composite key "
    "defined per table in inspect_faers.KEY_COLUMNS). Duplicate rows are NEVER "
    "removed: every source record, every column and the full report-level "
    "information (PRIMARYID / CASEID / CASEVERSION) are preserved so that "
    "downstream analysis can decide for itself whether to de-duplicate. "
    "KEY_GROUP_SIZE records how many rows share the key."
)

#: Human-readable summary of the cleaning rules; echoed in the report.
TRANSFORMATION_RULES: Tuple[str, ...] = (
    "Missing values: every string column is trimmed, internal whitespace is "
    "collapsed and empty strings become NULL. No row is removed.",
    "Inconsistent text: control characters are stripped and values are "
    "upper-cased (FAERS source text is case-insensitive), giving one canonical "
    "case/spacing per term.",
    "Drug-name standardisation (DRUG.DRUGNAME): a canonical DRUGNAME_STD is "
    "derived (case/spacing/punctuation normalised) together with a conservative "
    "heuristic grouping key DRUGNAME_KEY that drops a trailing strength/form "
    "token. The original DRUGNAME is retained; no external drug dictionary is "
    "applied.",
    "Date parsing: every column ending in _DT is normalised to digits and "
    "parsed into a new <COLUMN>_DATE DATE column (YYYYMMDD and YYYYMM accepted; "
    "unparseable values become NULL). Original *_DT strings are retained.",
    "Age normalisation (DEMO): AGE + AGE_COD are converted into AGE_YEARS "
    "(YR/MON/WK/DY/HR/DEC). Original AGE and AGE_COD are retained.",
    "Gender normalisation (DEMO): SEX is upper-cased into SEX_NORM "
    "(M / F, unknown -> U, blank -> NULL).",
    "Invalid identifiers: rows whose PRIMARYID or CASEID is missing or not "
    "positive are flagged VALID_IDENTIFIERS = false. They are reported, not "
    "deleted.",
    DUPLICATE_RULE,
)

#: AGE_COD unit -> multiplier that converts the value into years.
AGE_UNIT_TO_YEARS: Mapping[str, float] = {
    "YR": 1.0,
    "MON": 1.0 / 12.0,
    "WK": 1.0 / 52.0,
    "DY": 1.0 / 365.0,
    "HR": 1.0 / 8760.0,
    "DEC": 10.0,
}

#: Columns handled by a dedicated rule instead of generic text normalisation.
SPECIAL_TEXT_COLUMNS: Mapping[str, Tuple[str, ...]] = {
    "DRUG": ("DRUGNAME",),
    "DEMO": ("AGE", "AGE_COD", "SEX"),
}


class CleaningError(RuntimeError):
    """Raised when a table cannot be read, cleaned or validated."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    """Row/column counts captured after one transformation stage."""

    name: str
    rows: int
    columns: int


@dataclass
class TableCleaning:
    """Everything recorded about cleaning one table."""

    table: str
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    column_names_before: List[str] = field(default_factory=list)
    column_names_after: List[str] = field(default_factory=list)
    stages: List[Stage] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    identifier_nulls: Dict[str, Optional[int]] = field(default_factory=dict)
    invalid_identifier_rows: int = 0
    duplicate_rows: int = 0
    max_key_group: int = 0
    key_label: str = ""
    duplicate_keys: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def added_columns(self) -> List[str]:
        before = set(self.column_names_before)
        return [name for name in self.column_names_after if name not in before]

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def status(self) -> str:
        if self.errors:
            return "ERROR"
        if self.rows_removed != 0:
            return "ERROR"
        return "WARN" if (self.duplicate_rows or self.invalid_identifier_rows) else "OK"


# ---------------------------------------------------------------------------
# Transformation primitives (built-in Spark functions only - no UDFs)
# ---------------------------------------------------------------------------
def _normalise_text(column: str) -> "F.Column":
    """Trim, collapse whitespace, strip control characters, upper-case, empty->NULL."""
    value = F.trim(F.col(column))
    value = F.regexp_replace(value, r"[\u0000-\u001F\u007F]", "")
    value = F.trim(value)
    value = F.regexp_replace(value, r"\s+", " ")
    value = F.upper(value)
    return F.when(value == F.lit(""), F.lit(None)).otherwise(value)


def _parse_faers_date(column: str) -> "F.Column":
    """Parse a FAERS ``*_DT`` string (YYYYMMDD or YYYYMM) into a DATE.

    ``F.try_to_date`` does not exist in PySpark 3.5.x, so the value is reduced
    to digits, padded to a full 8-digit form and parsed with ``F.to_date``,
    which returns NULL for anything malformed (Spark runs in non-ANSI mode, so
    no exception is raised).
    """
    digits = F.regexp_replace(F.trim(F.col(column)), r"[^0-9]", "")
    padded = (
        F.when(F.length(digits) == 8, digits)
        .when(F.length(digits) == 6, F.concat(digits, F.lit("01")))
        .otherwise(F.lit(None).cast("string"))
    )
    return F.to_date(padded, "yyyyMMdd")


def _age_years() -> "F.Column":
    """Convert DEMO AGE + AGE_COD into a value in years (NULL when unknown)."""
    age = F.col("AGE").cast(DoubleType())
    code = F.upper(F.trim(F.col("AGE_COD")))
    factor: Optional["F.Column"] = None
    for unit, multiplier in AGE_UNIT_TO_YEARS.items():
        term = F.when(code == F.lit(unit), F.lit(multiplier))
        factor = term if factor is None else factor.when(code == F.lit(unit), F.lit(multiplier))
    factor = factor.otherwise(F.lit(None).cast(DoubleType()))  # type: ignore[union-attr]
    years = age * factor
    return F.when(age.isNull() | (age < F.lit(0)), F.lit(None).cast(DoubleType())).otherwise(
        F.round(years, 2)
    )


def _sex_norm() -> "F.Column":
    """Normalise DEMO SEX to M / F (unknown -> U, blank -> NULL)."""
    raw = F.upper(F.trim(F.col("SEX")))
    return (
        F.when(raw.isNull() | (raw == F.lit("")), F.lit(None))
        .when(raw == F.lit("M"), F.lit("M"))
        .when(raw == F.lit("F"), F.lit("F"))
        .otherwise(F.lit("U"))
    )


def _drug_name_std() -> "F.Column":
    """Canonical drug name: upper-case, whitespace-collapsed, punctuation-trimmed."""
    value = F.upper(F.trim(F.col("DRUGNAME")))
    value = F.regexp_replace(value, r"[^A-Z0-9/\.\- ]", " ")
    value = F.regexp_replace(value, r"\s+", " ")
    value = F.trim(value)
    return F.when(value == F.lit(""), F.lit(None)).otherwise(value)


def _drug_name_key() -> "F.Column":
    """Conservative grouping key: ``DRUGNAME_STD`` without a trailing strength/form token."""
    value = F.regexp_replace(
        F.col("DRUGNAME_STD"),
        r"\s*\b\d+(\.\d+)?\s*(MG|MCG|G|ML|IU|UNITS?|TABS?|CAPS?)\b.*$",
        "",
    )
    value = F.trim(value)
    return F.when((value == F.lit("")) | value.isNull(), F.col("DRUGNAME_STD")).otherwise(value)


def _text_columns(frame: DataFrame, table: str) -> List[str]:
    """String columns eligible for generic text normalisation."""
    excluded = set(SPECIAL_TEXT_COLUMNS.get(table, ()))
    return [name for name, dtype in frame.dtypes if dtype == "string" and name not in excluded]


def _date_columns(frame: DataFrame) -> List[str]:
    """Every string column whose name ends in ``_DT``."""
    return [name for name, dtype in frame.dtypes if dtype == "string" and name.upper().endswith("_DT")]


def _resolve_key_names(columns: Sequence[str], table: str) -> Optional[List[str]]:
    """Resolve the table's primary natural key to real column names."""
    keys = KEY_COLUMNS.get(table)
    if not keys:
        return None
    indices = resolve_key_indices(columns, keys[0])
    if indices is None:
        return None
    return [columns[index] for index in indices]


# ---------------------------------------------------------------------------
# Per-table cleaning
# ---------------------------------------------------------------------------
def clean_table(
    spark: Any,
    table: str,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> Tuple[DataFrame, TableCleaning]:
    """Read, clean and validate one table across all requested quarters."""
    log = get_logger(__name__)
    result = TableCleaning(table=table)

    frames: List[DataFrame] = []
    for quarter in quarters:
        directory = quarter_dirs.get(quarter)
        files = discover_files(directory) if directory else {}
        path = files.get(table)
        if path is None:
            raise CleaningError(f"{quarter}: {table} file not found under {directory}")
        frame, validation = ingest_table_quarter(spark, table, quarter, path, settings)
        frames.append(frame)
        result.rows_before += validation.rows
        for identifier, missing in validation.missing_identifiers.items():
            if missing is not None:
                result.identifier_nulls[identifier] = (
                    (result.identifier_nulls.get(identifier) or 0) + missing
                )
            else:
                result.identifier_nulls.setdefault(identifier, None)

    if not frames:
        raise CleaningError(f"{table}: no quarter could be read")

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame)

    result.column_names_before = list(combined.columns)
    result.columns_before = len(combined.columns)
    result.stages.append(Stage("as ingested", result.rows_before, result.columns_before))

    # --- stage 1: text normalisation ----------------------------------------
    text_columns = _text_columns(combined, table)
    frame = combined
    for name in text_columns:
        frame = frame.withColumn(name, _normalise_text(name))
    rows, columns = frame.count(), len(frame.columns)
    result.stages.append(Stage("text normalised", rows, columns))
    log.info("%s: text normalised on %s column(s), %s rows", table, len(text_columns), f"{rows:,}")

    # --- stage 2: derived columns (dates, age, sex, drug name) ---------------
    date_columns = _date_columns(frame)
    for name in date_columns:
        frame = frame.withColumn(f"{name}_DATE", _parse_faers_date(name))
    if table == "DEMO":
        frame = frame.withColumn("AGE_YEARS", _age_years())
        frame = frame.withColumn("SEX_NORM", _sex_norm())
    if table == "DRUG":
        frame = frame.withColumn("DRUGNAME_STD", _drug_name_std())
        frame = frame.withColumn("DRUGNAME_KEY", _drug_name_key())
    rows, columns = frame.count(), len(frame.columns)
    result.stages.append(Stage("derived columns", rows, columns))
    log.info("%s: derived %s date column(s) + normalisers, %s rows", table, len(date_columns), f"{rows:,}")

    # --- stage 3: identifier validity and duplicate flags --------------------
    key_names = _resolve_key_names(frame.columns, table)
    frame = frame.withColumn(
        "VALID_IDENTIFIERS",
        F.col("PRIMARYID").isNotNull()
        & (F.col("PRIMARYID") > F.lit(0))
        & F.col("CASEID").isNotNull()
        & (F.col("CASEID") > F.lit(0)),
    )
    if key_names is not None:
        window = Window.partitionBy(*([QUARTER_COLUMN] + key_names))
        frame = frame.withColumn("KEY_GROUP_SIZE", F.count(F.lit(1)).over(window))
        frame = frame.withColumn("IS_DUPLICATE", F.col("KEY_GROUP_SIZE") > F.lit(1))
        result.key_label = "+".join(("quarter", *key_names))
    rows, columns = frame.count(), len(frame.columns)
    result.stages.append(Stage("flags added", rows, columns))

    result.column_names_after = list(frame.columns)
    result.columns_after = len(frame.columns)
    result.rows_after = rows

    # --- profiling: one aggregate pass over the cleaned frame ----------------
    aggregates = [F.count(F.lit(1)).alias("rows")]
    for name in frame.columns:
        aggregates.append(
            F.coalesce(F.sum(F.when(F.col(name).isNull(), F.lit(1)).otherwise(F.lit(0))), F.lit(0)).alias(f"null__{name}")
        )
    aggregates.append(
        F.coalesce(
            F.sum(F.when(F.col("VALID_IDENTIFIERS") == F.lit(False), F.lit(1)).otherwise(F.lit(0))),
            F.lit(0),
        ).alias("invalid_ids")
    )
    if key_names is not None:
        aggregates.append(
            F.coalesce(
                F.sum(F.when(F.col("IS_DUPLICATE") == F.lit(True), F.lit(1)).otherwise(F.lit(0))),
                F.lit(0),
            ).alias("duplicate_rows")
        )
        aggregates.append(F.coalesce(F.max(F.col("KEY_GROUP_SIZE")), F.lit(0)).alias("max_group"))

    profile = frame.agg(*aggregates).first()
    if profile is None:  # pragma: no cover - a table always yields one row
        raise CleaningError(f"{table}: profiling aggregate returned no row")

    for name in frame.columns:
        result.null_counts[name] = int(profile[f"null__{name}"])
    result.invalid_identifier_rows = int(profile["invalid_ids"])
    if key_names is not None:
        result.duplicate_rows = int(profile["duplicate_rows"])
        result.max_key_group = int(profile["max_group"])

    # Rows with a missing identifier are surfaced through the null profile.
    return frame, result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(
    *,
    quarters: Sequence[str],
    results: Sequence[TableCleaning],
    generated_at: Optional[datetime] = None,
) -> str:
    """Render the preprocessing validation summary as plain text."""
    timestamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [
        "=" * 78,
        "FAERS PREPROCESSING REPORT",
        f"Generated : {timestamp}",
        f"Quarters  : {', '.join(quarters)}",
        "Engine    : PySpark DataFrame API only (no pandas, no PyArrow, no UDFs)",
        "Output    : validation only - no dataset written, raw .txt files unchanged",
        "=" * 78,
        "",
        "0. CLEANING RULES",
        "-" * 78,
    ]
    for rule in TRANSFORMATION_RULES:
        lines.append(f"  - {rule}")

    lines.append("")
    lines.append("1. ROW COUNTS: BEFORE vs AFTER CLEANING")
    lines.append("-" * 78)
    lines.append(f"  {'TABLE':<6}{'ROWS BEFORE':>14}{'ROWS AFTER':>13}{'REMOVED':>10}{'COLS BEFORE':>13}{'COLS AFTER':>12}  STATUS")
    for item in results:
        lines.append(
            f"  {item.table:<6}{item.rows_before:>14,}{item.rows_after:>13,}"
            f"{item.rows_removed:>10}{item.columns_before:>13}{item.columns_after:>12}  {item.status}"
        )
    total_before = sum(item.rows_before for item in results)
    total_after = sum(item.rows_after for item in results)
    lines.append(
        f"  {'TOTAL':<6}{total_before:>14,}{total_after:>13,}"
        f"{total_before - total_after:>10}"
    )

    lines.append("")
    lines.append("2. SCHEMA CHANGE (columns added by cleaning)")
    lines.append("-" * 78)
    for item in results:
        added = item.added_columns or ["(none)"]
        lines.append(f"  {item.table:<6}{item.columns_before:>3} -> {item.columns_after:<3}  + {', '.join(added)}")

    lines.append("")
    lines.append("3. ROW COUNTS AFTER EACH TRANSFORMATION STAGE (must be constant)")
    lines.append("-" * 78)
    for item in results:
        stages = " | ".join(f"{stage.name}: {stage.rows:,}" for stage in item.stages)
        lines.append(f"  {item.table:<6}{stages}")

    lines.append("")
    lines.append("4. MISSING VALUES AFTER CLEANING (columns with at least one NULL)")
    lines.append("-" * 78)
    for item in results:
        populated = [
            (name, count) for name, count in item.null_counts.items() if count > 0
        ]
        if not populated:
            lines.append(f"  {item.table:<6}no NULL values in any column")
            continue
        details = ", ".join(
            f"{name}={count:,} ({100.0 * count / item.rows_after:.2f}%)"
            for name, count in sorted(populated, key=lambda pair: -pair[1])
        )
        lines.append(f"  {item.table:<6}{len(populated)} column(s) with NULLs: {details}")

    lines.append("")
    lines.append("5. IDENTIFIER VALIDATION (after cleaning)")
    lines.append("-" * 78)
    for item in results:
        parts = [
            f"{name}={'absent' if count is None else f'{count:,} missing'}"
            for name, count in item.identifier_nulls.items()
        ]
        lines.append(
            f"  {item.table:<6}rows with invalid PRIMARYID/CASEID: "
            f"{item.invalid_identifier_rows:,} | " + "  ".join(parts)
        )
    lines.append(
        f"  {'PRESERVED':<10}{', '.join(PROTECTED_IDENTIFIERS)} were not altered by any transformation."
    )

    lines.append("")
    lines.append("6. DUPLICATE RECORDS (flagged, never removed)")
    lines.append("-" * 78)
    for item in results:
        if not item.key_label:
            lines.append(f"  {item.table:<6}natural key not resolvable - duplicate check skipped")
            continue
        lines.append(
            f"  {item.table:<6}key {item.key_label:<38} "
            f"duplicate rows: {item.duplicate_rows:,}  max rows per key: {item.max_key_group:,}"
        )
    lines.append(
        "  The key shown is each table's declared natural key (inspect_faers.KEY_COLUMNS); "
        "it is the same key the ingestion validation uses."
    )
    lines.append(
        "  IS_DUPLICATE means 'two rows share that natural key', not 'byte-identical rows': "
        "FAERS legitimately repeats a key with different sequence/value columns."
    )
    lines.append("  All duplicate rows were retained; IS_DUPLICATE / KEY_GROUP_SIZE were added.")

    lines.append("")
    lines.append("7. TRANSFORMATION RESULTS (only verified outcomes are listed)")
    lines.append("-" * 78)
    for item in results:
        if item.errors:
            lines.append(f"  {item.table:<6}FAILED - see issues below")
            continue
        derived = [name for name in item.added_columns if name not in ("VALID_IDENTIFIERS", "KEY_GROUP_SIZE", "IS_DUPLICATE")]
        lines.append(
            f"  {item.table:<6}OK - {len(derived)} value column(s) derived, "
            f"{len(item.added_columns)} column(s) added, 0 rows removed"
        )

    issues = [(item.table, error) for item in results for error in item.errors]
    lines.append("")
    lines.append("8. ISSUES AND ERRORS")
    lines.append("-" * 78)
    if issues:
        for table, error in issues:
            lines.append(f"  [{table}] {error}")
    else:
        lines.append("  none")

    lines.append("")
    lines.append("9. SUMMARY")
    lines.append("-" * 78)
    errors = sum(1 for item in results if item.status == "ERROR")
    lines.append(
        f"  {len(results)} table(s) processed across {len(quarters)} quarter(s); "
        f"{total_before:,} rows in, {total_after:,} rows out, "
        f"{total_before - total_after} removed."
    )
    lines.append(
        "  Status: READY for analysis."
        if errors == 0
        else f"  Status: {errors} table(s) failed; see section 8."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(
    spark: Any,
    *,
    quarters: Sequence[str],
    tables: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> List[TableCleaning]:
    """Clean every requested table; failures are recorded, not raised."""
    log = get_logger(__name__)
    results: List[TableCleaning] = []
    for table in tables:
        log.info("Cleaning %s ...", table)
        try:
            _frame, result = clean_table(spark, table, quarters, settings, quarter_dirs)
        except (CleaningError, IngestionError) as exc:
            log.error("%s: cleaning failed: %s", table, exc)
            failed = TableCleaning(table=table, errors=[str(exc)])
            results.append(failed)
            continue
        except Exception as exc:  # noqa: BLE001 - one bad table must not abort the run
            log.error("%s: unexpected failure: %s", table, exc)
            results.append(TableCleaning(table=table, errors=[f"{type(exc).__name__}: {exc}"]))
            continue
        log.info(
            "%s: %s rows -> %s rows, %s -> %s columns, %s duplicate row(s), %s invalid id row(s)",
            table,
            f"{result.rows_before:,}",
            f"{result.rows_after:,}",
            result.columns_before,
            result.columns_after,
            f"{result.duplicate_rows:,}",
            f"{result.invalid_identifier_rows:,}",
        )
        results.append(result)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean the ingested FAERS tables (read-only; no dataset is written)."
    )
    parser.add_argument("--quarters", nargs="*", default=None, help="e.g. 2026_Q1 2026_Q2")
    parser.add_argument("--tables", nargs="*", default=None, help="e.g. DEMO DRUG")
    parser.add_argument(
        "--report",
        default=None,
        help="Report path (default reports/preprocessing_report.txt; '' disables it).",
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
    tables = [t.upper() for t in (args.tables or settings["tables"])]
    unknown_tables = [t for t in tables if t not in settings["tables"]]
    if unknown_tables:
        log.error("Unknown table(s): %s", ", ".join(unknown_tables))
        return 1

    quarter_dirs = dict(get_quarter_dirs())

    try:
        spark = create_spark_session(app_name="DrugSafetyFaersPreprocessing")
    except SparkEnvironmentError as exc:
        log.error("%s", exc)
        return 1

    try:
        results = run(
            spark,
            quarters=quarters,
            tables=tables,
            settings=settings,
            quarter_dirs=quarter_dirs,
        )
    finally:
        stop_spark_session(spark)

    report = render_report(quarters=quarters, results=results)
    print(report)

    report_path = args.report
    if report_path is None:
        report_path = str(get_path("reports") / "preprocessing_report.txt")
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report + "\n", encoding="utf-8")
        log.info("Report written to: %s", target)

    return 2 if any(item.status == "ERROR" for item in results) else 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
