"""Reusable inspection of the FDA FAERS quarterly ASCII extracts.

Design constraints:
  * **Read-only.** Nothing in this module writes to ``data/``. The only file it
    produces is the report under ``reports/``.
  * **Streaming.** The pure-Python backend reads files line by line with the
    stdlib ``csv`` reader; it never loads a whole table into pandas.
  * **Data-driven schema.** The ASCII extracts shipped for 2026 Q1/Q2 carry a
    header line with the column names, so the column names reported here come
    from the files themselves - never from a hard-coded assumption. When a file
    has no header the columns are reported as generic ``column_N`` placeholders.
  * **Two backends.** PySpark is used when the environment provides it
    (parallel counting, exact aggregates). Otherwise a bounded-memory stdlib
    streaming backend is used and the limitation is reported instead of
    silently producing weaker numbers.

Usage::

    python -m src.ingestion.inspect_faers
    python -m src.ingestion.inspect_faers --backend python --sample-rows 2000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # allow direct execution as a script
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.spark_session import (  # noqa: E402
    SparkEnvironmentError,
    check_spark_environment,
)
from src.utils.config import (  # noqa: E402
    get_faers_config,
    get_path,
    get_project_root,
    get_quarter_dirs,
)

__all__ = [
    "KEY_COLUMNS",
    "ProfileResult",
    "TableInspection",
    "discover_files",
    "discover_other_files",
    "detect_encoding",
    "detect_delimiter",
    "read_header",
    "inspect_table",
    "inspect_quarter",
    "compare_quarters",
    "render_report",
    "write_report",
    "main",
]

BACKEND_AUTO, BACKEND_SPARK, BACKEND_PYTHON = "auto", "spark", "python"

#: Delimiter candidates probed during detection.
DELIMITER_CANDIDATES: Tuple[str, ...] = ("$", "\t", "|", ",")

#: Row ceiling for the in-memory duplicate-key sets used by the Python backend.
#: Above this the duplicate check is reported as "not evaluated" rather than
#: risking an out-of-memory failure; the Spark backend has no such limit.
MAX_ROWS_FOR_KEY_SETS = 3_000_000

#: Natural key(s) per table, used to count duplicate records. Names are matched
#: case-insensitively against the header, with COLUMN_ALIASES applied.
KEY_COLUMNS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "DEMO": (("PRIMARYID",), ("CASEID", "CASEVERSION")),
    "DRUG": (("PRIMARYID", "DRUG_SEQ"),),
    "REAC": (("PRIMARYID", "PT"),),
    "OUTC": (("PRIMARYID", "OUTC_COD"),),
    "RPSR": (("PRIMARYID", "RPSR_COD"),),
    "THER": (("PRIMARYID", "DRUG_SEQ"),),
    "INDI": (("PRIMARYID", "INDI_PT"),),
}

#: Alternative names for a logical column, e.g. THER uses ``dsg_drug_seq`` in the
#: 2026 extracts while earlier releases used ``drug_seq``.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "DRUG_SEQ": ("drug_seq", "dsg_drug_seq"),
    "OUTC_COD": ("outc_cod", "outc"),
    "RPSR_COD": ("rpsr_cod", "rpsr"),
}

#: A header line consists purely of identifier-like tokens.
_HEADER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_DATE_RE = re.compile(r"^\d{8}$")
_INT_RE = re.compile(r"^\d+$")
_QUARTER_RE = re.compile(r"^(\d{4})_Q([1-4])$")

#: Python codec name -> JVM charset name. Spark reads via the JVM and does not
#: accept every Python alias (``latin-1`` is not a Java charset name).
_SPARK_CHARSETS: Dict[str, str] = {
    "utf-8": "UTF-8",
    "utf-8-sig": "UTF-8",
    "latin-1": "ISO-8859-1",
    "iso-8859-1": "ISO-8859-1",
    "cp1252": "windows-1252",
}


def spark_charset(encoding: str) -> str:
    """Translate a Python codec name into a charset the JVM accepts."""
    return _SPARK_CHARSETS.get(encoding.lower(), encoding)


def faers_tables() -> Tuple[str, ...]:
    """Return the seven FAERS table codes declared in ``config.yaml``."""
    return tuple(get_faers_config()["tables"])


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ColumnProfile:
    """Per-column observations for one table."""

    name: str
    index: int
    empty_count: int = 0
    max_length: int = 0
    type_guess: str = "unknown"
    examples: List[str] = field(default_factory=list)

    def missing_pct(self, rows: int) -> float:
        return (100.0 * self.empty_count / rows) if rows else 0.0


@dataclass
class ProfileResult:
    """Raw counters produced by a backend for a single file."""

    field_counts: Counter = field(default_factory=Counter)
    inconsistent_rows: int = 0
    first_bad_line: Optional[int] = None
    columns: List[ColumnProfile] = field(default_factory=list)
    duplicate_counts: Dict[str, Optional[int]] = field(default_factory=dict)
    distinct_primaryid: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class TableInspection:
    """Everything recorded about one FAERS table in one quarter."""

    table: str
    quarter: str
    path: Path
    size_bytes: int = 0
    delimiter: str = ""
    delimiter_evidence: List[str] = field(default_factory=list)
    encoding: str = ""
    encoding_notes: List[str] = field(default_factory=list)
    has_header: bool = False
    header_names: List[str] = field(default_factory=list)
    header_notes: List[str] = field(default_factory=list)
    rows: int = 0
    field_counts: Counter = field(default_factory=Counter)
    inconsistent_rows: int = 0
    first_bad_line: Optional[int] = None
    columns: List[ColumnProfile] = field(default_factory=list)
    duplicate_counts: Dict[str, Optional[int]] = field(default_factory=dict)
    distinct_primaryid: Optional[int] = None
    id_notes: List[str] = field(default_factory=list)
    backend: str = BACKEND_PYTHON
    warnings: List[str] = field(default_factory=list)

    @property
    def column_count(self) -> int:
        return len(self.columns)

    @property
    def observed_field_count(self) -> int:
        return max(self.field_counts) if self.field_counts else 0

    @property
    def schema_match(self) -> bool:
        """True when every row has the same number of fields as the columns."""
        return len(self.field_counts) <= 1 and self.inconsistent_rows == 0

    @property
    def column_names(self) -> List[str]:
        return [column.name for column in self.columns]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def expected_filenames(quarter: str) -> List[str]:
    """Return the expected file names for a quarter, from ``config.yaml``."""
    faers = get_faers_config()
    match = _QUARTER_RE.match(quarter)
    if not match:
        return []
    year, quarter_number = match.group(1), match.group(2)
    return [
        faers["filename_pattern"].format(table=table, yy=year[-2:], q=quarter_number)
        for table in faers_tables()
    ]


def discover_files(quarter_dir: Path) -> Dict[str, Path]:
    """Map table code -> file path for one quarter directory.

    Matching is case-insensitive on the file stem prefix (``ASCII/DEMO26Q1.txt``
    -> ``DEMO``), which tolerates the folder nesting and naming variations seen
    across FAERS releases.
    """
    found: Dict[str, Path] = {}
    if not quarter_dir.is_dir():
        return found
    for path in sorted(quarter_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in (".txt", ".csv"):
            continue
        stem = path.stem.upper()
        for table in faers_tables():
            if stem.startswith(table) and table not in found:
                found[table] = path
    return found


def discover_other_files(quarter_dir: Path, table_files: Mapping[str, Path]) -> List[Path]:
    """Return delimited files that are not one of the seven core tables.

    FAERS ships a ``DELETE`` file listing removed cases; it is reported for
    awareness (it will matter for de-duplication) but is not a core table.
    """
    known = {path.resolve() for path in table_files.values()}
    others: List[Path] = []
    if not quarter_dir.is_dir():
        return others
    for path in sorted(quarter_dir.rglob("*")):
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in (".txt", ".csv")
            and path.resolve() not in known
        ):
            others.append(path)
    return others


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
def detect_encoding(path: Path, sample_bytes: int = 1 << 16) -> Tuple[str, List[str]]:
    """Return ``(encoding, notes)`` for a FAERS file, based on a byte sample."""
    notes: List[str] = []
    with path.open("rb") as handle:
        raw = handle.read(sample_bytes)

    if raw.startswith(b"\xef\xbb\xbf"):
        notes.append("UTF-8 BOM detected")
        return "utf-8-sig", notes

    encoding = "latin-1"
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        notes.append(
            f"not valid UTF-8 (first bad byte at offset {exc.start}); "
            "latin-1/cp1252 decoding required"
        )
    else:
        encoding = "utf-8"
        notes.append("sample decodes cleanly as UTF-8")

    non_ascii = sum(1 for byte in raw if byte > 127)
    notes.append(f"{non_ascii} non-ASCII byte(s) in the first {len(raw)} bytes")
    return encoding, notes


def detect_delimiter(
    path: Path,
    encoding: str,
    candidates: Sequence[str] = DELIMITER_CANDIDATES,
    sample_lines: int = 500,
) -> Tuple[str, List[str]]:
    """Return ``(delimiter, evidence)`` by scoring candidates on sample lines."""
    evidence: List[str] = []
    best_delimiter, best_score = "", -1.0

    lines: List[str] = []
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        for index, line in enumerate(handle):
            if index >= sample_lines:
                break
            if line.strip():
                lines.append(line.rstrip("\r\n"))

    if not lines:
        return "", ["file contains no non-empty lines"]

    for candidate in candidates:
        counts = Counter(line.count(candidate) for line in lines)
        mode, mode_rows = counts.most_common(1)[0]
        if mode == 0:
            evidence.append(f"{candidate!r}: absent")
            continue
        consistency = mode_rows / len(lines)
        score = mode * consistency
        evidence.append(
            f"{candidate!r}: {mode} per line on {mode_rows}/{len(lines)} sampled rows "
            f"(consistency {consistency:.2f})"
        )
        if score > best_score:
            best_delimiter, best_score = candidate, score

    return best_delimiter, evidence


def read_header(
    path: Path,
    encoding: str,
    delimiter: str,
) -> Tuple[List[str], List[str]]:
    """Read the first non-empty line and decide whether it is a header row.

    Returns:
        ``(header_names, notes)``. ``header_names`` is empty when the first line
        does not look like a header (e.g. it starts with a numeric PRIMARYID).
    """
    notes: List[str] = []
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        for line in handle:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            fields = stripped.split(delimiter)
            if len(fields) > 1 and all(_HEADER_NAME_RE.match(part) for part in fields):
                notes.append(
                    f"first non-empty line contains {len(fields)} identifier-like "
                    "names -> treated as a header row"
                )
                return fields, notes
            notes.append(
                "first non-empty line does not look like a header "
                "(no header row found; column names are placeholders)"
            )
            return [], notes

    notes.append("file is empty")
    return [], notes


def _probe_field_count(
    path: Path, encoding: str, delimiter: str, lines: int = 200
) -> int:
    """Most common field count over the first ``lines`` non-empty lines."""
    counts: Counter = Counter()
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        for index, line in enumerate(handle):
            if index >= lines:
                break
            stripped = line.rstrip("\r\n")
            if stripped.strip():
                counts[stripped.count(delimiter) + 1] += 1
    return counts.most_common(1)[0][0] if counts else 0


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------
def infer_type(values: Sequence[str]) -> str:
    """Classify sampled values as date / integer / float / string."""
    if not values:
        return "unknown"
    if all(_DATE_RE.match(value) for value in values):
        return "date(YYYYMMDD)"
    if all(_INT_RE.match(value) for value in values):
        return "integer"
    try:
        for value in values:
            float(value)
    except ValueError:
        return "string"
    return "float"


def _distinct_examples(values: Sequence[str], limit: int = 3) -> List[str]:
    """Up to ``limit`` distinct example values, preserving first-seen order."""
    examples: List[str] = []
    for value in values:
        if value not in examples:
            examples.append(value)
            if len(examples) >= limit:
                break
    return examples


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------
def resolve_key_indices(
    columns: Sequence[str], key_names: Sequence[str]
) -> Optional[List[int]]:
    """Map logical key column names to positions in the observed header.

    Matching is case-insensitive and honours :data:`COLUMN_ALIASES`. Returns
    ``None`` when any part of the key cannot be resolved.
    """
    lowered = {column.lower(): index for index, column in enumerate(columns)}
    indices: List[int] = []
    for name in key_names:
        candidates = (name.lower(),) + COLUMN_ALIASES.get(name.upper(), ())
        position = next((lowered[c] for c in candidates if c in lowered), None)
        if position is None:
            return None
        indices.append(position)
    return indices


def _key_digest(values: Sequence[str]) -> int:
    """Stable 64-bit digest of a composite key (keeps the key sets small)."""
    joined = "\x1f".join(values).encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(joined, digest_size=8).digest(), "big")


# ---------------------------------------------------------------------------
# Python (stdlib, streaming) backend
# ---------------------------------------------------------------------------
def _profile_python(
    path: Path,
    *,
    encoding: str,
    delimiter: str,
    columns: Sequence[str],
    has_header: bool,
    key_indices: Mapping[str, Optional[List[int]]],
    sample_rows: int,
) -> ProfileResult:
    """Stream a file once, accumulating counts and a bounded value sample."""
    result = ProfileResult()
    width = len(columns)
    empty_counts = [0] * width
    max_lengths = [0] * width
    samples: List[List[str]] = [[] for _ in range(width)]

    seen: Dict[str, set] = {key: set() for key in key_indices}
    overflow: Dict[str, bool] = {key: False for key in key_indices}
    duplicates: Dict[str, int] = {key: 0 for key in key_indices}
    seen_rows: Dict[str, int] = {key: 0 for key in key_indices}

    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for line_number, row in enumerate(reader, start=1):
            if has_header and line_number == 1:
                continue

            result.field_counts[len(row)] += 1
            if len(row) != width:
                result.inconsistent_rows += 1
                if result.first_bad_line is None:
                    result.first_bad_line = line_number

            for index in range(min(len(row), width)):
                value = row[index]
                if value == "":
                    empty_counts[index] += 1
                    continue
                if len(value) > max_lengths[index]:
                    max_lengths[index] = len(value)
                if len(samples[index]) < sample_rows:
                    samples[index].append(value)

            for key, indices in key_indices.items():
                if indices is None or overflow[key]:
                    continue
                values = tuple(row[i] for i in indices if i < len(row))
                if len(values) != len(indices):
                    continue
                seen_rows[key] += 1
                digest = _key_digest(values)
                if digest in seen[key]:
                    duplicates[key] += 1
                else:
                    seen[key].add(digest)
                    if len(seen[key]) > MAX_ROWS_FOR_KEY_SETS:
                        overflow[key] = True
                        seen[key].clear()

    result.columns = [
        ColumnProfile(
            name=name,
            index=index,
            empty_count=empty_counts[index],
            max_length=max_lengths[index],
            type_guess=infer_type(samples[index]),
            examples=_distinct_examples(samples[index]),
        )
        for index, name in enumerate(columns)
    ]

    for key, indices in key_indices.items():
        if indices is None:
            result.duplicate_counts[key] = None
            result.warnings.append(
                f"key {key} not evaluated: the columns it needs are not present in the header"
            )
        elif overflow[key]:
            result.duplicate_counts[key] = None
            result.warnings.append(
                f"key {key} duplicate check skipped: more than "
                f"{MAX_ROWS_FOR_KEY_SETS:,} distinct keys (needs the Spark backend)"
            )
        else:
            result.duplicate_counts[key] = duplicates[key]

    if "PRIMARYID" in key_indices and not overflow.get("PRIMARYID", True):
        result.distinct_primaryid = seen_rows["PRIMARYID"] - duplicates["PRIMARYID"]
    return result


# ---------------------------------------------------------------------------
# Spark backend
# ---------------------------------------------------------------------------
def _profile_spark(
    spark: Any,
    path: Path,
    *,
    encoding: str,
    delimiter: str,
    columns: Sequence[str],
    has_header: bool,
    key_indices: Mapping[str, Optional[List[int]]],
    sample_rows: int,
) -> ProfileResult:
    """Profile a file with Spark: exact counts, exact null counts, bounded sample."""
    from pyspark.sql import functions as F  # noqa: PLC0415

    result = ProfileResult()
    charset = spark_charset(encoding)
    raw = spark.read.option("encoding", charset).csv(
        str(path), sep=delimiter, header=has_header, inferSchema=False, quote='"'
    )
    width = len(columns)
    if len(raw.columns) == width:
        frame = raw.toDF(*columns)
    else:
        result.warnings.append(
            f"the reader exposed {len(raw.columns)} fields but {width} column names "
            "were resolved; names were aligned positionally"
        )
        frame = raw.toDF(*[f"col_{i + 1}" for i in range(len(raw.columns))])

    rows = frame.count()

    # Field-count distribution from the raw text lines. The 'encoding' option is
    # only honoured by newer Spark releases; field counting tolerates the
    # replacement characters older versions emit for non-UTF-8 bytes.
    pattern = re.escape(delimiter)
    text = spark.read.option("encoding", charset).text(str(path))
    counted = (
        text.withColumn("n", F.size(F.split(F.col("value"), pattern)))
        .groupBy("n")
        .count()
        .collect()
    )
    for row in counted:
        result.field_counts[int(row["n"])] = int(row["count"])
    if has_header and columns:
        header_width = len(columns)
        remaining = result.field_counts.get(header_width, 0) - 1
        if remaining > 0:
            result.field_counts[header_width] = remaining
        else:
            result.field_counts.pop(header_width, None)

    result.inconsistent_rows = rows - result.field_counts.get(width, 0)

    # Exact empty/null counts per column in a single aggregation.
    empty_exprs = [
        F.sum(
            F.when(F.col(column).isNull() | (F.trim(F.col(column)) == ""), 1).otherwise(0)
        ).alias(column)
        for column in frame.columns
    ]
    totals = frame.agg(*empty_exprs).collect()[0]
    empty_counts = {column: int(totals[column] or 0) for column in frame.columns}

    # Bounded sample for type inference and example values.
    samples: Dict[str, List[str]] = {column: [] for column in frame.columns}
    max_lengths = {column: 0 for column in frame.columns}
    for row in frame.limit(sample_rows).collect():
        for column in frame.columns:
            value = row[column]
            if value is None or value == "":
                continue
            text_value = str(value)
            if len(text_value) > max_lengths[column]:
                max_lengths[column] = len(text_value)
            if len(samples[column]) < sample_rows:
                samples[column].append(text_value)

    result.columns = [
        ColumnProfile(
            name=column,
            index=index,
            empty_count=empty_counts.get(column, 0),
            max_length=max_lengths.get(column, 0),
            type_guess=infer_type(samples.get(column, [])),
            examples=_distinct_examples(samples.get(column, [])),
        )
        for index, column in enumerate(frame.columns)
    ]

    for key, indices in key_indices.items():
        if indices is None:
            result.duplicate_counts[key] = None
            result.warnings.append(
                f"key {key} not evaluated: the columns it needs are not present in the header"
            )
            continue
        key_columns = [frame.columns[i] for i in indices]
        distinct = frame.select(*key_columns).distinct().count()
        result.duplicate_counts[key] = rows - distinct

    if "PRIMARYID" in key_indices and key_indices["PRIMARYID"] is not None:
        primary = frame.columns[key_indices["PRIMARYID"][0]]
        result.distinct_primaryid = frame.select(primary).distinct().count()

    return result


# ---------------------------------------------------------------------------
# Inspection driver
# ---------------------------------------------------------------------------
def _id_notes(profiles: Sequence[ColumnProfile], rows: int) -> List[str]:
    """Describe the identifier columns requested by the inspection spec."""
    notes: List[str] = []
    by_name = {column.name.lower(): column for column in profiles}
    for identifier in ("PRIMARYID", "CASEID", "CASEVERSION"):
        column = by_name.get(identifier.lower())
        if column is None:
            notes.append(f"{identifier}: not present in this table")
            continue
        notes.append(
            f"{identifier} (header '{column.name}'): position {column.index + 1}, "
            f"type={column.type_guess}, empty={column.empty_count} "
            f"({column.missing_pct(rows):.3f}%), examples={column.examples}"
        )
    return notes


def inspect_table(
    table: str,
    quarter: str,
    path: Path,
    *,
    backend: str,
    spark: Any = None,
    sample_rows: int = 5000,
) -> TableInspection:
    """Inspect a single FAERS table file."""
    encoding, encoding_notes = detect_encoding(path)
    delimiter, delimiter_evidence = detect_delimiter(path, encoding)
    header_names, header_notes = read_header(path, encoding, delimiter)

    if header_names:
        columns: List[str] = header_names
    else:
        width = _probe_field_count(path, encoding, delimiter)
        columns = [f"column_{index + 1}" for index in range(width)]

    key_indices: Dict[str, Optional[List[int]]] = {}
    for key in KEY_COLUMNS.get(table, ()):
        key_indices["+".join(key)] = resolve_key_indices(columns, key)

    inspection = TableInspection(
        table=table,
        quarter=quarter,
        path=path,
        size_bytes=path.stat().st_size,
        delimiter=delimiter,
        delimiter_evidence=delimiter_evidence,
        encoding=encoding,
        encoding_notes=encoding_notes,
        has_header=bool(header_names),
        header_names=list(header_names),
        header_notes=header_notes,
        backend=backend,
    )

    common = {
        "encoding": encoding,
        "delimiter": delimiter,
        "columns": columns,
        "has_header": bool(header_names),
        "key_indices": key_indices,
        "sample_rows": sample_rows,
    }
    if backend == BACKEND_SPARK and spark is not None:
        result = _profile_spark(spark, path, **common)
    else:
        result = _profile_python(path, **common)

    inspection.field_counts = result.field_counts
    inspection.inconsistent_rows = result.inconsistent_rows
    inspection.first_bad_line = result.first_bad_line
    inspection.columns = result.columns
    inspection.duplicate_counts = result.duplicate_counts
    inspection.distinct_primaryid = result.distinct_primaryid
    inspection.warnings = list(result.warnings)
    inspection.rows = sum(result.field_counts.values())
    inspection.id_notes = _id_notes(result.columns, inspection.rows)

    if not inspection.schema_match:
        inspection.warnings.append(
            f"ragged rows: {inspection.inconsistent_rows:,} row(s) do not have "
            f"{inspection.column_count} fields "
            f"(field-count distribution {dict(sorted(inspection.field_counts.items()))})"
        )
    return inspection


def inspect_quarter(
    quarter: str,
    *,
    backend: str,
    spark: Any = None,
    sample_rows: int = 5000,
    quarter_dir: Optional[Path] = None,
) -> Dict[str, TableInspection]:
    """Inspect every discovered table in one quarter directory."""
    directory = quarter_dir or (get_path("raw") / quarter)
    inspections: Dict[str, TableInspection] = {}
    for table, path in discover_files(directory).items():
        inspections[table] = inspect_table(
            table, quarter, path, backend=backend, spark=spark, sample_rows=sample_rows
        )
    return inspections


# ---------------------------------------------------------------------------
# Q1 vs Q2 comparison
# ---------------------------------------------------------------------------
def compare_quarters(
    first: Mapping[str, TableInspection],
    second: Mapping[str, TableInspection],
    first_label: str,
    second_label: str,
) -> List[str]:
    """Return human-readable schema/volume differences between two quarters."""
    lines: List[str] = []
    for table in faers_tables():
        left, right = first.get(table), second.get(table)
        if left is None and right is None:
            lines.append(f"{table}: absent in both {first_label} and {second_label}")
            continue
        if left is None:
            lines.append(f"{table}: only present in {second_label}")
            continue
        if right is None:
            lines.append(f"{table}: only present in {first_label}")
            continue

        if left.column_names == right.column_names:
            lines.append(
                f"{table}: schemas identical - {left.column_count} columns, same names "
                "in the same order"
            )
        else:
            lines.append(
                f"{table}: SCHEMA DIFFERENCE - {left.column_count} vs "
                f"{right.column_count} columns"
            )
            added = [c for c in right.column_names if c not in left.column_names]
            removed = [c for c in left.column_names if c not in right.column_names]
            if added:
                lines.append(f"    only in {second_label}: {', '.join(added)}")
            if removed:
                lines.append(f"    only in {first_label}: {', '.join(removed)}")
            if not added and not removed:
                lines.append("    same column names but a different order")

        if left.has_header != right.has_header:
            lines.append(
                f"{table}: header presence differs "
                f"({first_label}={left.has_header}, {second_label}={right.has_header})"
            )
        if left.schema_match != right.schema_match:
            lines.append(
                f"{table}: ragged-row status differs "
                f"({first_label} consistent={left.schema_match}, "
                f"{second_label} consistent={right.schema_match})"
            )
        if left.rows != right.rows:
            delta = right.rows - left.rows
            pct = (100.0 * delta / left.rows) if left.rows else 0.0
            lines.append(
                f"{table}: rows {first_label}={left.rows:,} -> {second_label}="
                f"{right.rows:,} ({delta:+,} / {pct:+.2f}%)"
            )
    return lines


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _format_bytes(size: int) -> str:
    return f"{size:,} bytes ({size / 1024 / 1024:.2f} MB)"


def _render_table_block(inspection: TableInspection) -> List[str]:
    header_state = (
        f"present ({inspection.column_count} names, e.g. "
        f"'{inspection.header_names[0]}', '{inspection.header_names[-1]}')"
        if inspection.has_header
        else "absent (placeholder column names)"
    )
    lines = [
        f"{'-' * 78}",
        f"[{inspection.quarter}] {inspection.table}  -  {inspection.path.name}",
        f"{'-' * 78}",
        f"  File            : {inspection.path}",
        f"  Size            : {_format_bytes(inspection.size_bytes)}",
        f"  Rows (data)     : {inspection.rows:,}",
        f"  Columns         : {inspection.column_count}",
        f"  Header row      : {header_state}",
        f"  Delimiter       : {inspection.delimiter!r}",
        f"  Encoding        : {inspection.encoding}",
        f"  Observed fields : {inspection.observed_field_count} "
        f"(distribution over data rows: {dict(sorted(inspection.field_counts.items()))})",
        f"  Ragged rows     : {inspection.inconsistent_rows:,}"
        + (
            f" (first at file line {inspection.first_bad_line})"
            if inspection.first_bad_line
            else ""
        ),
        f"  Backend         : {inspection.backend}",
    ]

    lines.append("")
    lines.append("  Columns (names taken from the file's header row):")
    lines.append(
        f"    {'#':>3}  {'COLUMN':<20} {'TYPE':<15} {'MISSING':>10} {'MISS%':>7} "
        f"{'MAXLEN':>6}  EXAMPLES"
    )
    for column in inspection.columns:
        lines.append(
            f"    {column.index + 1:>3}  {column.name:<20} {column.type_guess:<15} "
            f"{column.empty_count:>10,} {column.missing_pct(inspection.rows):>7.3f} "
            f"{column.max_length:>6}  {', '.join(column.examples)[:60]}"
        )

    lines.append("")
    lines.append("  Identifiers:")
    lines.extend(f"    - {note}" for note in inspection.id_notes)

    lines.append("")
    lines.append("  Duplicate records (natural key -> duplicate row count):")
    for key, count in inspection.duplicate_counts.items():
        rendered = "not evaluated (needs Spark)" if count is None else f"{count:,}"
        lines.append(f"    - {key}: {rendered}")
    distinct = (
        "not evaluated"
        if inspection.distinct_primaryid is None
        else f"{inspection.distinct_primaryid:,}"
    )
    lines.append(f"    distinct PRIMARYID values: {distinct}")

    lines.append("")
    lines.append("  Delimiter evidence:")
    lines.extend(f"    - {item}" for item in inspection.delimiter_evidence)
    lines.append("  Encoding evidence:")
    lines.extend(f"    - {item}" for item in inspection.encoding_notes)
    lines.append("  Header evidence:")
    lines.extend(f"    - {item}" for item in inspection.header_notes)

    if inspection.warnings:
        lines.append("")
        lines.append("  Warnings:")
        lines.extend(f"    ! {item}" for item in inspection.warnings)
    lines.append("")
    return lines


def render_report(
    *,
    backend: str,
    backend_reason: str,
    quarter_files: Mapping[str, Mapping[str, Path]],
    inspections: Mapping[str, Mapping[str, TableInspection]],
    other_files: Optional[Mapping[str, Sequence[Path]]] = None,
    generated_at: Optional[datetime] = None,
    comparison: Optional[Sequence[str]] = None,
) -> str:
    """Render the full inspection report as plain text."""
    timestamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    faers = get_faers_config()
    raw_dir = get_path("raw")
    tables = faers_tables()

    lines: List[str] = [
        "=" * 78,
        "FDA FAERS DATASET INSPECTION REPORT",
        f"Generated      : {timestamp}",
        f"Project root   : {get_project_root()}",
        f"Backend        : {backend}" + (f" ({backend_reason})" if backend_reason else ""),
        "Read-only inspection - no source file was modified.",
        "=" * 78,
        "",
        "1. DATASET AVAILABILITY",
        "-" * 78,
    ]

    available_quarters = [q for q, files in quarter_files.items() if files]
    for quarter, files in quarter_files.items():
        state = (
            f"{len(files)} of {len(tables)} tables found"
            + (f" (missing: {', '.join(t for t in tables if t not in files)})"
               if len(files) < len(tables) else "")
            if files
            else "MISSING"
        )
        lines.append(f"  {quarter}: {raw_dir / quarter}")
        lines.append(f"      -> {state}")

    if other_files:
        lines.append("")
        lines.append("  Additional delimited files (not one of the seven core tables):")
        for quarter, paths in other_files.items():
            for path in paths:
                lines.append(f"    - [{quarter}] {path.name} ({_format_bytes(path.stat().st_size)})")
            lines.append(
                "      Note: DELETE files list cases withdrawn from the FAERS database;"
            )
            lines.append(
                "      they are relevant to de-duplication but are not core analysis tables."
            )

    lines.append("")
    lines.append("2. ENVIRONMENT")
    lines.append("-" * 78)
    report = check_spark_environment()
    lines.append(f"  Python         : {report['python_version']}")
    lines.append(f"  PySpark        : {report['pyspark_version'] or 'MISSING'}")
    lines.append(f"  JAVA_HOME      : {report['java_home'] or 'not set'}")
    lines.append(f"  java           : {report['java_version'] or 'MISSING'}")
    for issue in report["issues"]:
        lines.append(f"    ! {issue}")

    if not available_quarters:
        lines.append("")
        lines.append("3. STATUS: INSPECTION NOT POSSIBLE - DATASETS MISSING")
        lines.append("-" * 78)
        lines.append("  No FAERS ASCII files were found in the raw data directories.")
        lines.append("  No synthetic or sample data was generated and no numbers are")
        lines.append("  reported, because any figure here would be fabricated.")
        lines.append("")
        lines.append("  ACTION REQUIRED - place the extracted files here:")
        for quarter in quarter_files:
            lines.append(f"    {raw_dir / quarter}\\")
            for filename in expected_filenames(quarter):
                lines.append(f"        {filename}")
        lines.append("")
        lines.append("  Files must be the untouched FDA FAERS ASCII extracts ('$' delimited).")
        lines.append("  Once present, re-run: python -m src.ingestion.inspect_faers")
        lines.append("")
        return "\n".join(lines)

    lines.append("")
    lines.append("3. FILE INVENTORY")
    lines.append("-" * 78)
    for quarter, files in quarter_files.items():
        for table in tables:
            if table not in files:
                continue
            inspection = inspections.get(quarter, {}).get(table)
            rows = f"{inspection.rows:,} rows" if inspection else "not inspected"
            lines.append(
                f"  {quarter}  {table:<5} {files[table].name:<16} "
                f"{_format_bytes(files[table].stat().st_size):<32} {rows}"
            )

    lines.append("")
    lines.append("4. PER-TABLE INSPECTION")
    lines.append("-" * 78)
    for quarter, tables_inspected in inspections.items():
        for table in tables:
            if table in tables_inspected:
                lines.extend(_render_table_block(tables_inspected[table]))

    lines.append("")
    lines.append("5. QUARTER COMPARISON")
    lines.append("-" * 78)
    if comparison:
        lines.extend(f"  {item}" for item in comparison)
    else:
        lines.append("  Only one quarter was inspected; nothing to compare.")

    lines.append("")
    lines.append("6. READINESS SUMMARY")
    lines.append("-" * 78)
    for quarter, files in quarter_files.items():
        tables_inspected = inspections.get(quarter, {})
        total_rows = sum(inspection.rows for inspection in tables_inspected.values())
        consistent = sum(1 for inspection in tables_inspected.values() if inspection.schema_match)
        lines.append(
            f"  {quarter}: {len(files)}/{len(tables)} tables, {total_rows:,} data rows, "
            f"{consistent}/{len(tables_inspected)} tables with a consistent field count"
        )

    detected_headers = {
        inspection.has_header
        for tables_inspected in inspections.values()
        for inspection in tables_inspected.values()
    }
    configured_header = bool(faers.get("header", False))
    if detected_headers and detected_headers != {configured_header}:
        lines.append("")
        lines.append(
            f"  CONFIG NOTE: faers.header is '{configured_header}' in config.yaml but the "
            f"files were read with header={sorted(detected_headers)[0]}. "
            "Update faers.header before writing the ingestion layer."
        )

    lines.append("")
    lines.append(f"  Delimiter   : '{faers['delimiter']}' (as configured)")
    lines.append(f"  Encoding    : {faers['encoding']} (as configured)")
    lines.append("")
    return "\n".join(lines)


def write_report(text: str, output_path: Optional[Path] = None) -> Path:
    """Write the report to ``reports/dataset_inspection.txt`` (UTF-8)."""
    target = output_path or (get_path("reports") / "dataset_inspection.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve_backend(backend: str) -> Tuple[str, str]:
    """Resolve ``auto`` to the best available backend, with a reason string."""
    if backend == BACKEND_PYTHON:
        return BACKEND_PYTHON, "forced on the command line"

    report = check_spark_environment()
    if report["pyspark_available"] and report["java_available"]:
        return BACKEND_SPARK, f"PySpark {report['pyspark_version']}"

    if backend == BACKEND_SPARK:
        raise SparkEnvironmentError(
            "The Spark backend was requested but cannot be used:\n  - "
            + "\n  - ".join(report["issues"])
        )
    return BACKEND_PYTHON, "; ".join(report["issues"]) or "PySpark or Java unavailable"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the FDA FAERS ASCII extracts (read-only)."
    )
    parser.add_argument(
        "--backend", choices=(BACKEND_AUTO, BACKEND_SPARK, BACKEND_PYTHON), default=BACKEND_AUTO
    )
    parser.add_argument(
        "--quarters", nargs="*", default=None, help="Quarter labels, e.g. 2026_Q1 2026_Q2"
    )
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument(
        "--output", default=None, help="Report path (default reports/dataset_inspection.txt)"
    )
    args = parser.parse_args(argv)

    try:
        backend, backend_reason = resolve_backend(args.backend)
    except SparkEnvironmentError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    quarter_dirs = (
        [(q, get_path("raw") / q) for q in args.quarters]
        if args.quarters
        else get_quarter_dirs()
    )

    quarter_files = {quarter: discover_files(directory) for quarter, directory in quarter_dirs}
    other_files = {
        quarter: discover_other_files(directory, quarter_files[quarter])
        for quarter, directory in quarter_dirs
    }

    spark = None
    inspections: Dict[str, Dict[str, TableInspection]] = {}
    if backend == BACKEND_SPARK and any(quarter_files.values()):
        from src.ingestion.spark_session import create_spark_session  # noqa: PLC0415

        spark = create_spark_session(app_name="DrugSafetyDatasetInspection")

    try:
        for quarter, directory in quarter_dirs:
            if not quarter_files.get(quarter):
                inspections[quarter] = {}
                continue
            print(f"Inspecting {quarter} ...", file=sys.stderr)
            inspections[quarter] = inspect_quarter(
                quarter,
                backend=backend,
                spark=spark,
                sample_rows=args.sample_rows,
                quarter_dir=directory,
            )
    finally:
        if spark is not None:
            from src.ingestion.spark_session import stop_spark_session  # noqa: PLC0415

            stop_spark_session(spark)

    comparison: List[str] = []
    quarters_with_data = [q for q in quarter_files if quarter_files[q]]
    if len(quarters_with_data) >= 2:
        first, second = quarters_with_data[0], quarters_with_data[1]
        comparison = compare_quarters(inspections[first], inspections[second], first, second)

    text = render_report(
        backend=backend,
        backend_reason=backend_reason,
        quarter_files=quarter_files,
        inspections=inspections,
        other_files=other_files,
        comparison=comparison,
    )
    target = write_report(text, Path(args.output) if args.output else None)
    print(f"\nReport written to: {target}")

    if not quarters_with_data:
        print("\nSTATUS: DATASETS MISSING - inspection could not be performed.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
