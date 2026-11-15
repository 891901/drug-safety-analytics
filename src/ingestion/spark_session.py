"""Reusable PySpark session management.

Responsibilities:
  * detect whether PySpark and a usable JDK are available;
  * build a correctly configured local ``SparkSession`` from ``config.yaml``;
  * fail with an actionable message (never a bare ImportError) when the
    environment is incomplete.

This module does **not** read, transform or write any dataset.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.utils.config import get_spark_config

__all__ = [
    "SparkEnvironmentError",
    "check_spark_environment",
    "create_spark_session",
    "get_spark_session",
    "stop_spark_session",
    "format_environment_report",
]

#: JDK versions officially supported by the PySpark 3.5.x line.
SUPPORTED_JAVA_MAJORS = (8, 11, 17)

#: First Python version the PySpark 3.5.x line does not officially support.
MAX_SUPPORTED_PYTHON = (3, 12)

_JAVA_VERSION_PATTERN = re.compile(r'version "([^"]+)"')


class SparkEnvironmentError(RuntimeError):
    """Raised when PySpark or the JDK needed by Spark is unavailable."""


# ---------------------------------------------------------------------------
# Environment discovery
# ---------------------------------------------------------------------------
def _detect_java_home() -> Optional[str]:
    """Find a candidate JAVA_HOME if not already in os.environ."""
    existing = os.environ.get("JAVA_HOME")
    if existing and Path(existing).is_dir():
        return existing
    if os.name == "nt":
        java_dir = Path(r"C:\Program Files\Java")
        if java_dir.is_dir():
            for child in sorted(java_dir.iterdir(), reverse=True):
                if child.is_dir() and (child / "bin" / "java.exe").is_file():
                    os.environ["JAVA_HOME"] = str(child)
                    return str(child)
    return None


def _java_executable() -> Optional[Path]:
    """Locate the ``java`` binary: JAVA_HOME first, then PATH."""
    java_home = _detect_java_home()
    if java_home:
        candidate = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if candidate.is_file():
            return candidate

    found = shutil.which("java")
    return Path(found) if found else None


def _java_version(java_exe: Path) -> Optional[str]:
    """Return the raw ``java -version`` first line, or None if it cannot run."""
    try:
        completed = subprocess.run(
            [str(java_exe), "-version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    # `java -version` writes to stderr on every known JDK.
    output = (completed.stderr or completed.stdout or "").strip()
    if not output:
        return None
    return output.splitlines()[0].strip()


def _parse_java_major(version_line: Optional[str]) -> Optional[int]:
    """Extract the major version from a ``java -version`` line."""
    if not version_line:
        return None
    match = _JAVA_VERSION_PATTERN.search(version_line)
    if not match:
        return None
    raw = match.group(1)
    parts = raw.split(".")
    if parts[0] == "1" and len(parts) > 1:  # legacy "1.8.0_491" style
        return int(parts[1])
    try:
        return int(parts[0])
    except ValueError:
        return None


def _pyspark_version() -> Optional[str]:
    """Return the installed PySpark version, or None when it is missing."""
    try:
        from pyspark import __version__ as pyspark_version  # noqa: PLC0415
    except ImportError:
        return None
    return pyspark_version


def check_spark_environment() -> Dict[str, Any]:
    """Inspect the local environment without starting Spark.

    Returns:
        A dict describing availability plus a list of human-readable ``issues``.
        This function never raises - it is safe to call from diagnostics and
        tests.
    """
    java_exe = _java_executable()
    java_version = _java_version(java_exe) if java_exe else None
    java_major = _parse_java_major(java_version)
    pyspark_version = _pyspark_version()
    python_version = ".".join(str(part) for part in sys.version_info[:3])

    issues: List[str] = []

    if pyspark_version is None:
        issues.append(
            "PySpark is not installed. Install it later with: "
            "python -m pip install \"pyspark>=3.5,<4\""
        )

    if java_exe is None:
        issues.append(
            "No Java runtime found. Install a JDK 11 or 17 and set JAVA_HOME "
            "(e.g. JAVA_HOME=C:\\Program Files\\Java\\jdk-17)."
        )
    elif java_major is not None and java_major not in SUPPORTED_JAVA_MAJORS:
        issues.append(
            f"Java {java_major} detected ({java_version}). PySpark 3.5.x is only "
            f"tested against Java {', '.join(str(v) for v in SUPPORTED_JAVA_MAJORS)}."
        )

    if not (3, 10) <= sys.version_info[:2] <= MAX_SUPPORTED_PYTHON:
        issues.append(
            f"Python {python_version} detected. PySpark 3.5.x officially supports "
            f"Python 3.10-{'.'.join(str(p) for p in MAX_SUPPORTED_PYTHON)}; "
            "use a matching interpreter inside a virtual environment."
        )

    winutils = None
    if os.name == "nt":
        hadoop_home = os.environ.get("HADOOP_HOME")
        if hadoop_home:
            candidate = Path(hadoop_home) / "bin" / "winutils.exe"
            winutils = str(candidate) if candidate.is_file() else None

    return {
        "pyspark_available": pyspark_version is not None,
        "pyspark_version": pyspark_version,
        "java_available": java_exe is not None,
        "java_executable": str(java_exe) if java_exe else None,
        "java_version": java_version,
        "java_major": java_major,
        "java_home": os.environ.get("JAVA_HOME"),
        "python_version": python_version,
        "winutils": winutils,
        "issues": issues,
    }


def format_environment_report(report: Optional[Mapping[str, Any]] = None) -> str:
    """Render :func:`check_spark_environment` as a readable multi-line string."""
    report = report or check_spark_environment()
    lines = [
        "PySpark environment report",
        "--------------------------",
        f"  Python         : {report['python_version']}",
        f"  PySpark        : {report['pyspark_version'] or 'MISSING'}",
        f"  JAVA_HOME      : {report['java_home'] or 'not set'}",
        f"  java           : {report['java_version'] or 'MISSING'}"
        + (f"  [{report['java_executable']}]" if report["java_executable"] else ""),
        f"  winutils.exe   : {report['winutils'] or 'not found (only needed for Hadoop FS writes)'}",
    ]
    if report["issues"]:
        lines.append("  Issues:")
        lines.extend(f"    - {issue}" for issue in report["issues"])
    else:
        lines.append("  Issues         : none")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------
def _require_pyspark():
    """Import and return ``SparkSession``, or raise a clear error."""
    report = check_spark_environment()
    if not report["java_available"]:
        raise SparkEnvironmentError(
            "Cannot create a Spark session: no Java runtime was found.\n"
            "PySpark needs a JDK on the machine.\n"
            "  1. Install a JDK 11 or 17 (Java 8 is the bare minimum).\n"
            "  2. Set JAVA_HOME, e.g. JAVA_HOME=C:\\Program Files\\Java\\jdk-17.\n"
            "  3. Restart the terminal so the variable is picked up.\n"
            f"Detected python: {report['python_version']}"
        )
    if not report["pyspark_available"]:
        raise SparkEnvironmentError(
            "Cannot create a Spark session: PySpark is not installed.\n"
            'Install it with: python -m pip install "pyspark>=3.5,<4"\n'
            f"Note: PySpark 3.5.x supports Python 3.10-3.12, "
            f"but this interpreter is Python {report['python_version']}."
        )

    try:
        from pyspark.sql import SparkSession  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - defensive
        raise SparkEnvironmentError(
            f"PySpark is present but could not be imported: {exc}"
        ) from exc
    return SparkSession


def _build_builder_options(
    spark_config: Mapping[str, Any],
    app_name: Optional[str],
    master: Optional[str],
    overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Translate the ``spark`` config section into builder settings."""
    resolved_master = master or str(spark_config["master"])
    is_local = resolved_master.startswith("local")

    options: Dict[str, Any] = {
        "app_name": app_name or str(spark_config["app_name"]),
        "master": resolved_master,
        "driver_memory": str(spark_config["driver_memory"]),
    }

    # In local mode the driver *is* the executor, so executor.memory is ignored
    # by Spark; only set it for non-local masters to avoid a misleading UI value.
    executor_memory = spark_config.get("executor_memory")
    if executor_memory and not is_local:
        options["executor_memory"] = str(executor_memory)

    if "shuffle_partitions" in spark_config:
        options["shuffle_partitions"] = int(spark_config["shuffle_partitions"])

    # Add extra JVM flags only when the user explicitly asks for them.
    extra_options = {k: v for k, v in (overrides or {}).items() if v is not None}
    options.update(extra_options)
    return options


def create_spark_session(
    app_name: Optional[str] = None,
    master: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
):
    """Create (or return) a configured :class:`SparkSession`.

    Args:
        app_name: Overrides ``spark.app_name`` from the config file.
        master: Overrides ``spark.master`` (default ``local[*]``).
        overrides: Extra ``SparkConf`` style key/value pairs,
            e.g. ``{"spark.sql.shuffle.partitions": 8, "spark.ui.enabled": False}``.

    Raises:
        SparkEnvironmentError: If PySpark or Java is unavailable.
    """
    SparkSession = _require_pyspark()
    spark_config = get_spark_config()

    # Make sure the driver and workers use the same interpreter that is
    # currently running this code (important on Windows with several Pythons).
    py_exe = sys.executable
    if os.name == "nt" and " " in py_exe:
        short_cand = Path(r"C:\PROGRA~1\Python314\python.exe")
        if short_cand.is_file():
            py_exe = str(short_cand)
    os.environ.setdefault("PYSPARK_PYTHON", py_exe)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", py_exe)

    options = _build_builder_options(spark_config, app_name, master, overrides)

    builder = SparkSession.builder.appName(options.pop("app_name"))
    builder = builder.master(options.pop("master"))
    builder = builder.config("spark.driver.memory", options.pop("driver_memory"))

    if "executor_memory" in options:
        builder = builder.config("spark.executor.memory", options.pop("executor_memory"))
    if "shuffle_partitions" in options:
        builder = builder.config(
            "spark.sql.shuffle.partitions", options.pop("shuffle_partitions")
        )

    # Anything left over is passed through verbatim as Spark properties.
    for key, value in options.items():
        builder = builder.config(str(key), str(value))

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(str(spark_config.get("log_level", "WARN")))
    return session


def get_spark_session(
    app_name: Optional[str] = None,
    master: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
):
    """Return the active session if one exists, otherwise create a new one."""
    SparkSession = _require_pyspark()
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return create_spark_session(app_name=app_name, master=master, overrides=overrides)


def stop_spark_session(session: Any = None) -> None:
    """Stop the given session (or the active one) if it is running."""
    if session is not None:
        session.stop()
        return
    try:
        from pyspark.sql import SparkSession  # noqa: PLC0415
    except ImportError:
        return
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    print(format_environment_report())
