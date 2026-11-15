"""PySpark ingestion of the FDA FAERS ASCII extracts.

Reads the seven core FAERS tables (DEMO, DRUG, REAC, OUTC, RPSR, THER, INDI)
for every configured quarter, validates them, tags each row with its source
quarter and (optionally) writes one combined Parquet dataset per table under
``data/processed/``.

Guarantees:
  * the raw ``.txt`` files are only ever opened read-only;
  * no record is dropped, rewritten or synthesised - columns are only renamed,
    re-typed to 64-bit integers where the source is numeric, and extended with
    a ``quarter`` provenance column;
  * no joins and no whole-table in-memory Pandas conversion.

Reading
-------
Spark reads the ASCII files itself (CSV, ``$``-delimited, UTF-8). Every value is
read as a string; only the numeric identifier / sequence columns listed under
``faers.numeric_columns`` are cast to 64-bit integers, so nothing in the source
is reinterpreted or discarded.

Parquet output
--------------
The write step lives in :mod:`src.ingestion.parquet_output` and is gated behind
``faers.write_parquet`` in ``config.yaml``. It is **disabled** here because this
Windows host cannot write Parquet with Spark (the Hadoop committer needs
``winutils.exe``) and the pure-Python serialisers (PyArrow/pandas) have their
native DLLs blocked by an OS Application Control policy. When the switch is
``false`` the pipeline runs validation-only and no dataset is written; set it to
``true`` once the environment provides ``winutils.exe`` / ``HADOOP_HOME``.

Usage::

    python -m src.ingestion.load_faers
    python -m src.ingestion.load_faers --quarters 2026_Q1 --tables DEMO DRUG
    python -m src.ingestion.load_faers --overwrite
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType

from src.ingestion.inspect_faers import (
    KEY_COLUMNS,
    discover_files,
    read_header,
    resolve_key_indices,
    spark_charset,
)
from src.ingestion.parquet_output import (
    ParquetWriteError,
    parquet_enabled,
    write_table_parquet,
)
from src.ingestion.spark_session import (
    SparkEnvironmentError,
    create_spark_session,
    stop_spark_session,
)
from src.utils.config import (
    get_faers_config,
    get_path,
    get_quarter_dirs,
    get_spark_config,
)
from src.utils.logging_setup import get_logger

__all__ = [
    "IngestionError",
    "QUARTER_COLUMN",
    "TableValidation",
    "load_settings",
    "canonical_columns",
    "read_table_file",
    "ingest_table_quarter",
    "render_report",
    "write_summary_json",
    "main",
]

#: Provenance column added to every ingested table (never overwrites a source
#: column: FAERS has no column with this name).
QUARTER_COLUMN = "quarter"

#: Fallback read-side partition cap (bytes) used to bound driver memory.
DEFAULT_MAX_PARTITION_BYTES = 64 * 1024 * 1024


class IngestionError(RuntimeError):
    """Raised when a table cannot be read or validated."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class TableValidation:
    """Validation outcome for one table in one quarter."""

    table: str
    quarter: str
    path: Path
    rows: int
    column_count: int
    expected_columns: Optional[int]
    column_names: List[str]
    numeric_columns: List[str]
    missing_identifiers: Dict[str, Optional[int]] = field(default_factory=dict)
    duplicate_records: Dict[str, Optional[int]] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(issue.startswith("ERROR") for issue in self.issues):
            return "ERROR"
        return "WARN" if self.issues else "OK"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "quarter": self.quarter,
            "source_file": self.path.name,
            "rows": self.rows,
            "column_count": self.column_count,
            "expected_columns": self.expected_columns,
            "column_names": list(self.column_names),
            "numeric_columns": list(self.numeric_columns),
            "missing_identifiers": dict(self.missing_identifiers),
            "duplicate_records": dict(self.duplicate_records),
            "status": self.status,
            "issues": list(self.issues),
        }


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def load_settings() -> Dict[str, Any]:
    """Return the FAERS settings needed by the ingestion pipeline."""
    faers = get_faers_config()
    expected = faers.get("expected_columns") or {}
    return {
        "delimiter": str(faers["delimiter"]),
        "encoding": str(faers["encoding"]),
        "has_header": bool(faers.get("header", False)),
        "tables": [str(name) for name in faers["tables"]],
        "expected_columns": {str(k).upper(): int(v) for k, v in expected.items()},
        "id_columns": [str(name).upper() for name in (faers.get("id_columns") or [])],
        "numeric_columns": {str(name).upper() for name in (faers.get("numeric_columns") or [])},
        "write_parquet": bool(faers.get("write_parquet", False)),
        "max_partition_bytes": int(
            get_spark_config().get("max_partition_bytes", DEFAULT_MAX_PARTITION_BYTES)
        ),
    }


def canonical_columns(path: Path, delimiter: str, encoding: str) -> List[str]:
    """Return the column names taken from the file's header row, upper-cased.

    The 2026 extracts carry a header, so the names come from the data itself.
    """
    names, _notes = read_header(path, encoding, delimiter)
    if not names:
        raise IngestionError(
            f"{path.name}: no header row found, but faers.header is true - "
            "cannot resolve column names."
        )
    return [name.strip().upper() for name in names]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_table_file(
    spark: Any,
    path: Path,
    *,
    columns: Sequence[str],
    delimiter: str,
    encoding: str,
    has_header: bool,
    numeric_columns: Sequence[str],
) -> DataFrame:
    """Read one FAERS ASCII file into a Spark DataFrame with canonical columns.

    All source values are read as strings first; only the numeric identifier /
    sequence columns are cast, so nothing is reinterpreted or discarded.
    """
    raw = (
        spark.read.option("encoding", spark_charset(encoding))
        .option("sep", delimiter)
        .option("header", has_header)
        .option("inferSchema", False)
        .option("quote", '"')
        .option("mode", "PERMISSIVE")
        .csv(str(path))
    )
    if len(raw.columns) != len(columns):
        raise IngestionError(
            f"{path.name}: the reader exposed {len(raw.columns)} fields but the "
            f"header has {len(columns)} columns."
        )

    frame = raw.toDF(*columns)
    numeric = {name.upper() for name in numeric_columns}
    for name in columns:
        if name.upper() in numeric:
            frame = frame.withColumn(name, F.col(name).cast(LongType()))
    return frame


def _resolve_key_names(
    columns: Sequence[str], key: Sequence[str]
) -> Optional[List[str]]:
    """Map a logical composite key to the actual header column names."""
    indices = resolve_key_indices(columns, key)
    if indices is None:
        return None
    return [columns[index] for index in indices]


def _validate(
    frame: DataFrame,
    *,
    table: str,
    quarter: str,
    path: Path,
    columns: Sequence[str],
    settings: Mapping[str, Any],
) -> TableValidation:
    """Validate one table/quarter: counts, types, identifiers and duplicates."""
    expected = settings["expected_columns"].get(table)
    numeric_present = sorted(name for name, dtype in frame.dtypes if dtype == "bigint")

    issues: List[str] = []
    if expected is not None and len(columns) != expected:
        issues.append(f"ERROR column count {len(columns)} != expected {expected}")

    validation = TableValidation(
        table=table,
        quarter=quarter,
        path=path,
        rows=frame.count(),
        column_count=len(columns),
        expected_columns=expected,
        column_names=list(columns),
        numeric_columns=numeric_present,
    )

    # --- identifier presence -------------------------------------------------
    id_columns = settings["id_columns"]
    by_name = {name.upper(): name for name in columns}
    for identifier in id_columns:
        column = by_name.get(identifier)
        if column is None:
            validation.missing_identifiers[identifier] = None
            if identifier in ("PRIMARYID", "CASEID"):
                issues.append(f"ERROR required identifier {identifier} is absent")
            continue
        missing = frame.filter(
            F.col(column).isNull() | (F.trim(F.col(column)) == "")
        ).count()
        validation.missing_identifiers[identifier] = missing
        if missing:
            issues.append(f"WARN {missing:,} row(s) with a missing {identifier}")

    # --- duplicate records on the table's natural key ------------------------
    for key in KEY_COLUMNS.get(table, ()):
        label = "+".join(key)
        names = _resolve_key_names(columns, key)
        if not names:
            validation.duplicate_records[label] = None
            continue
        distinct = frame.select(*names).distinct().count()
        duplicates = validation.rows - distinct
        validation.duplicate_records[label] = duplicates
        if duplicates:
            issues.append(f"WARN {duplicates:,} duplicate row(s) on key {label}")

    validation.issues = issues
    return validation


def ingest_table_quarter(
    spark: Any,
    table: str,
    quarter: str,
    path: Path,
    settings: Mapping[str, Any],
) -> Tuple[DataFrame, TableValidation]:
    """Read and validate one table for one quarter; add the ``quarter`` column."""
    columns = canonical_columns(path, settings["delimiter"], settings["encoding"])
    frame = read_table_file(
        spark,
        path,
        columns=columns,
        delimiter=settings["delimiter"],
        encoding=settings["encoding"],
        has_header=settings["has_header"],
        numeric_columns=settings["numeric_columns"],
    )
    validation = _validate(
        frame, table=table, quarter=quarter, path=path, columns=columns, settings=settings
    )
    frame = frame.withColumn(QUARTER_COLUMN, F.lit(quarter))
    return frame, validation


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(
    *,
    quarters: Sequence[str],
    tables: Sequence[str],
    validations: Mapping[str, Mapping[str, TableValidation]],
    parquet_info: Mapping[str, Mapping[str, Any]],
    processed_dir: Path,
    generated_at: Optional[datetime] = None,
) -> str:
    """Render the ingestion report as plain text."""
    timestamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [
        "=" * 78,
        "FAERS INGESTION REPORT",
        f"Generated    : {timestamp}",
        f"Quarters     : {', '.join(quarters)}",
        f"Output       : {processed_dir}",
        "Raw .txt files were read-only; no source row was dropped or altered.",
        "=" * 78,
        "",
        "1. ROW COUNTS (actual)",
        "-" * 78,
        f"  {'TABLE':<6}{'COLUMNS':>8}{'EXPECTED':>9}   "
        + "".join(f"{q:>12}" for q in quarters)
        + f"{'COMBINED':>12}  STATUS",
    ]

    grand_total = 0
    for table in tables:
        per_quarter = validations.get(table, {})
        counts = [per_quarter[q].rows if q in per_quarter else None for q in quarters]
        combined = sum(count for count in counts if count is not None)
        grand_total += combined
        first = next((per_quarter[q] for q in quarters if q in per_quarter), None)
        column_count = first.column_count if first else 0
        expected = first.expected_columns if first else None
        statuses = {per_quarter[q].status for q in quarters if q in per_quarter}
        status = "ERROR" if "ERROR" in statuses else ("WARN" if "WARN" in statuses else "OK")
        lines.append(
            f"  {table:<6}{column_count:>8}{str(expected or '-'):>9}   "
            + "".join(
                f"{(f'{count:,}' if count is not None else '-'):>12}" for count in counts
            )
            + f"{combined:>12,}  {status}"
        )
    lines.append(
        f"  {'TOTAL':<6}{'':>8}{'':>9}   "
        + "".join(" " * 12 for _ in quarters)
        + f"{grand_total:>12,}"
    )
    lines.append("")
    lines.append("2. SCHEMA AND TYPES")
    lines.append("-" * 78)
    for table in tables:
        first = next(
            (validations[table][q] for q in quarters if q in validations.get(table, {})),
            None,
        )
        if first is None:
            continue
        lines.append(
            f"  {table:<6}{first.column_count:>3} columns | "
            f"bigint: {', '.join(first.numeric_columns) or 'none'} | "
            "remaining columns: string"
        )

    lines.append("")
    lines.append("3. IDENTIFIER VALIDATION (missing / empty values)")
    lines.append("-" * 78)
    any_identifier = False
    for table in tables:
        for quarter in quarters:
            item = validations.get(table, {}).get(quarter)
            if item is None:
                continue
            any_identifier = True
            parts = [
                f"{identifier}={'absent' if missing is None else f'{missing:,}'}"
                for identifier, missing in item.missing_identifiers.items()
            ]
            lines.append(f"  {quarter}  {table:<6} " + "  ".join(parts))
    if not any_identifier:
        lines.append("  no table was read")

    lines.append("")
    lines.append("4. DUPLICATE RECORD CHECK (rows beyond the first occurrence of a key)")
    lines.append("-" * 78)
    any_duplicate = False
    for table in tables:
        for quarter in quarters:
            item = validations.get(table, {}).get(quarter)
            if item is None:
                continue
            any_duplicate = True
            parts = [
                f"{label}={'n/a' if count is None else f'{count:,}'}"
                for label, count in item.duplicate_records.items()
            ]
            lines.append(f"  {quarter}  {table:<6} " + "  ".join(parts))
    if not any_duplicate:
        lines.append("  no table was read")

    lines.append("")
    lines.append("5. PARQUET OUTPUT")
    lines.append("-" * 78)
    if parquet_info:
        lines.append(f"  {'TABLE':<6}{'ROWS BEFORE':>13}{'ROWS AFTER':>12}{'SIZE':>12}  RESULT")
        for table in tables:
            info = parquet_info.get(table)
            if info is None:
                lines.append(f"  {table:<6}{'-':>13}{'-':>12}{'-':>12}  NOT WRITTEN")
                continue
            rows_before = int(info["rows_before"])
            rows_after = int(info["rows_after"])
            size_mb = info["size_bytes"] / (1024 * 1024)
            if not info["schema_ok"]:
                result = "SCHEMA MISMATCH"
            elif rows_before == rows_after:
                result = "OK"
            else:
                result = "ROW MISMATCH"
            lines.append(
                f"  {table:<6}{rows_before:>13,}{rows_after:>12,}{size_mb:>10.1f} MB  {result}"
            )
    else:
        lines.append("  Disabled (faers.write_parquet=false). Validation-only run;")
        lines.append("  no Parquet dataset was written under data/processed/.")
        lines.append(
            "  Reason: this Windows host has no winutils.exe/HADOOP_HOME for Spark's"
        )
        lines.append(
            "  Hadoop writer, and the PyArrow/pandas native libraries are blocked by"
        )
        lines.append("  an OS Application Control policy. See src/ingestion/parquet_output.py.")

    issues = [
        (f"{quarter}/{table}", issue)
        for table in tables
        for quarter in quarters
        if (item := validations.get(table, {}).get(quarter))
        for issue in item.issues
    ]
    lines.append("")
    lines.append("6. ISSUES")
    lines.append("-" * 78)
    if issues:
        for where, issue in issues:
            lines.append(f"  [{where}] {issue}")
    else:
        lines.append("  none")

    lines.append("")
    lines.append("7. READINESS")
    lines.append("-" * 78)
    if parquet_info:
        written_total = sum(int(info["rows_after"]) for info in parquet_info.values())
        lines.append(
            f"  {len(parquet_info)} of {len(tables)} table(s) written across "
            f"{len(quarters)} quarter(s); {written_total:,} rows read back from Parquet."
        )
    else:
        read_total = sum(
            item.rows
            for table_validations in validations.values()
            for item in table_validations.values()
        )
        lines.append(
            f"  {len(tables)} table(s) read and validated across {len(quarters)} "
            f"quarter(s) ({read_total:,} source rows); Parquet output disabled."
        )
    errors = sum(1 for _where, issue in issues if issue.startswith("ERROR"))
    lines.append(
        "  Status: READY for preprocessing."
        if errors == 0
        else f"  Status: {errors} error(s) must be fixed before preprocessing."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metadata / validation summary (Python standard library only)
# ---------------------------------------------------------------------------
def write_summary_json(
    path: Path,
    *,
    quarters: Sequence[str],
    tables: Sequence[str],
    validations: Mapping[str, Mapping[str, TableValidation]],
    parquet_info: Mapping[str, Mapping[str, Any]],
    processed_dir: Path,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Persist a compact machine-readable validation summary as JSON.

    Uses only the standard library (``json``) - no pandas, no PyArrow - so it
    works regardless of the blocked native libraries.
    """
    timestamp = (generated_at or datetime.now()).isoformat(timespec="seconds")
    payload: Dict[str, Any] = {
        "generated_at": timestamp,
        "processed_dir": str(processed_dir),
        "quarters": list(quarters),
        "write_parquet": bool(parquet_info),
        "tables": {},
        "row_counts": {},
    }

    for table in tables:
        per_quarter = validations.get(table, {})
        first = next((per_quarter[q] for q in quarters if q in per_quarter), None)
        entry: Dict[str, Any] = {
            "expected_columns": first.expected_columns if first else None,
            "column_count": first.column_count if first else None,
            "column_names": list(first.column_names) if first else [],
            "numeric_columns": list(first.numeric_columns) if first else [],
            "combined_rows": 0,
            "quarters": {},
        }
        for quarter in quarters:
            item = per_quarter.get(quarter)
            if item is None:
                continue
            entry["quarters"][quarter] = item.as_dict()
            entry["combined_rows"] += item.rows
        payload["tables"][table] = entry
        payload["row_counts"][table] = {
            quarter: per_quarter[quarter].rows
            for quarter in quarters
            if quarter in per_quarter
        }

    payload["parquet"] = {table: dict(info) for table, info in parquet_info.items()}

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def ingest(
    spark: Any,
    *,
    quarters: Sequence[str],
    tables: Sequence[str],
    processed_dir: Path,
    overwrite: bool,
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> Tuple[Dict[str, Dict[str, TableValidation]], Dict[str, Dict[str, Any]]]:
    """Ingest every requested table/quarter and return validations + write info."""
    log = get_logger(__name__)

    # Read-side partition cap: bounds the bytes pulled per Spark task.
    spark.conf.set(
        "spark.sql.files.maxPartitionBytes", str(settings["max_partition_bytes"])
    )

    do_write = parquet_enabled(settings)
    if do_write:
        log.info("Parquet output enabled; writing under %s", processed_dir)
    else:
        log.info(
            "Parquet output disabled (faers.write_parquet=false); validation-only run."
        )

    validations: Dict[str, Dict[str, TableValidation]] = {table: {} for table in tables}
    parquet_info: Dict[str, Dict[str, Any]] = {}

    for table in tables:
        frames: List[DataFrame] = []
        for quarter in quarters:
            directory = quarter_dirs.get(quarter)
            files = discover_files(directory) if directory else {}
            path = files.get(table)
            if path is None:
                log.error("%s: %s file not found under %s", quarter, table, directory)
                continue
            frame, validation = ingest_table_quarter(spark, table, quarter, path, settings)
            validations[table][quarter] = validation
            frames.append(frame)
            log.info(
                "%s %s: %s rows, %s columns, status=%s",
                quarter,
                table,
                f"{validation.rows:,}",
                validation.column_count,
                validation.status,
            )

        if not frames:
            log.error("%s: no quarter could be read; skipping", table)
            continue

        # Combine the quarterly frames, preserving the ``quarter`` column.
        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.unionByName(frame)

        if not do_write:
            continue

        target = processed_dir / table
        try:
            result = write_table_parquet(
                combined,
                target,
                mode="overwrite" if overwrite else "errorifexists",
                log=log,
            )
        except ParquetWriteError as exc:
            log.error("%s: Parquet write failed: %s", table, exc)
            continue
        parquet_info[table] = result.as_dict()

    return validations, parquet_info


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest the FDA FAERS ASCII extracts (read-only on the raw files)."
    )
    parser.add_argument("--quarters", nargs="*", default=None, help="e.g. 2026_Q1 2026_Q2")
    parser.add_argument("--tables", nargs="*", default=None, help="e.g. DEMO DRUG")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Parquet datasets under data/processed/ (when enabled).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Report path (default reports/ingestion_report.txt; '' disables it).",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="JSON summary path (default reports/ingestion_summary.json; '' disables it).",
    )
    args = parser.parse_args(argv)

    log = get_logger(__name__)

    settings = load_settings()
    all_quarters = [quarter for quarter, _dir in get_quarter_dirs()]
    quarters = [q for q in (args.quarters or all_quarters)]
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
    processed_dir = get_path("processed")

    try:
        spark = create_spark_session(app_name="DrugSafetyFaersIngestion")
    except SparkEnvironmentError as exc:
        log.error("%s", exc)
        return 1

    try:
        validations, parquet_info = ingest(
            spark,
            quarters=quarters,
            tables=tables,
            processed_dir=processed_dir,
            overwrite=args.overwrite,
            settings=settings,
            quarter_dirs=quarter_dirs,
        )
    except IngestionError as exc:
        log.error("Ingestion failed: %s", exc)
        return 2
    finally:
        stop_spark_session(spark)

    report = render_report(
        quarters=quarters,
        tables=tables,
        validations=validations,
        parquet_info=parquet_info,
        processed_dir=processed_dir,
    )
    print(report)

    report_path = args.report
    if report_path is None:
        report_path = str(get_path("reports") / "ingestion_report.txt")
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report + "\n", encoding="utf-8")
        log.info("Report written to: %s", target)

    summary_path = args.summary
    if summary_path is None:
        summary_path = str(get_path("reports") / "ingestion_summary.json")
    if summary_path:
        write_summary_json(
            Path(summary_path),
            quarters=quarters,
            tables=tables,
            validations=validations,
            parquet_info=parquet_info,
            processed_dir=processed_dir,
        )
        log.info("Summary written to: %s", summary_path)

    has_error = any(
        item.status == "ERROR"
        for table_validations in validations.values()
        for item in table_validations.values()
    )
    return 2 if has_error else 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
