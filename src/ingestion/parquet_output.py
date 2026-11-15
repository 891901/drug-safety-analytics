"""Spark-native Parquet output for the ingested FAERS datasets.

This module is deliberately separate from the ingestion/validation logic so the
write step can be switched on without touching the rest of the pipeline. It uses
Spark's own Parquet writer (``DataFrameWriter.parquet``) - there is no
third-party serialiser involved.

Why it is disabled here
-----------------------
Spark commits Parquet output through Hadoop's ``FileOutputCommitter``, which
calls ``RawLocalFileSystem.setPermission`` for every directory it creates. On
Windows that call needs ``winutils.exe`` and fails with::

    java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset.
        at org.apache.hadoop.util.Shell.getWinUtilsPath(Shell.java:735)
        at org.apache.hadoop.util.Shell.getSetPermissionCommand(Shell.java:270)
        at org.apache.hadoop.fs.RawLocalFileSystem.setPermission(RawLocalFileSystem.java:978)
        at org.apache.hadoop.fs.RawLocalFileSystem.mkOneDirWithMode(RawLocalFileSystem.java:660)
        at org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter.setupJob(FileOutputCommitter.java:356)

Selecting ``org.apache.hadoop.fs.RawLocalFileSystem`` does not avoid it - that
class performs the same permission call - and ``pyarrow``/``pandas`` (the usual
pure-Python alternatives) have their native DLLs blocked by an OS Application
Control policy on this machine.

Enable the step by setting ``faers.write_parquet: true`` in ``config.yaml`` once
``winutils.exe`` + ``HADOOP_HOME`` are available.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from pyspark.sql import DataFrame

__all__ = [
    "ParquetWriteError",
    "ParquetResult",
    "parquet_enabled",
    "write_table_parquet",
]


class ParquetWriteError(RuntimeError):
    """Raised when Spark cannot write or verify a Parquet dataset."""


@dataclass
class ParquetResult:
    """Outcome of writing and reading back one Parquet dataset."""

    path: Path
    rows_before: int
    rows_after: int
    size_bytes: int
    schema_ok: bool

    @property
    def ok(self) -> bool:
        return self.rows_before == self.rows_after and self.schema_ok

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "size_bytes": self.size_bytes,
            "schema_ok": self.schema_ok,
            "ok": self.ok,
        }


def parquet_enabled(settings: Mapping[str, Any]) -> bool:
    """True when ``faers.write_parquet`` is switched on."""
    return bool(settings.get("write_parquet", False))


def _directory_size(target: Path) -> int:
    """Total size of the files under ``target`` (Spark writes a directory)."""
    if target.is_file():
        return target.stat().st_size
    return sum(path.stat().st_size for path in target.rglob("*") if path.is_file())


def write_table_parquet(
    frame: DataFrame,
    target: Path,
    *,
    mode: str = "overwrite",
    log: Any = None,
) -> ParquetResult:
    """Write ``frame`` to Parquet with Spark and verify it by reading it back.

    Spark never collects the table into the driver - it streams partition by
    partition to disk. The dataset is then read back through Spark so the caller
    can compare the row count and schema before trusting it.

    Raises:
        ParquetWriteError: If the write or the read-back fails.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "overwrite" and target.exists():
        # Spark's overwrite can leave stale part-files behind; a clean directory
        # keeps the dataset unambiguous.
        shutil.rmtree(target, ignore_errors=True)

    expected_dtypes = frame.dtypes
    rows_before = frame.count()

    try:
        frame.write.mode(mode).parquet(str(target))
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ParquetWriteError(
            f"Spark could not write Parquet to {target}: {exc}\n"
            "On Windows this normally means winutils.exe is missing; see the "
            "module docstring in src/ingestion/parquet_output.py."
        ) from exc

    try:
        verified = frame.sparkSession.read.parquet(str(target))
        rows_after = verified.count()
        schema_ok = verified.dtypes == expected_dtypes
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ParquetWriteError(
            f"the Parquet dataset at {target} could not be read back: {exc}"
        ) from exc

    result = ParquetResult(
        path=target,
        rows_before=rows_before,
        rows_after=rows_after,
        size_bytes=_directory_size(target),
        schema_ok=schema_ok,
    )
    if log is not None:
        log.info(
            "%s: %s rows before write, %s rows read back, schema %s",
            target.name,
            f"{rows_before:,}",
            f"{rows_after:,}",
            "ok" if schema_ok else "MISMATCH",
        )
    return result
