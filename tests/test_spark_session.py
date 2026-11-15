"""Verification of the Spark session layer.

Runs under pytest (``pytest tests/test_spark_session.py``) and also standalone
(``python tests/test_spark_session.py``).

When PySpark or Java is missing the session test is *skipped with an explicit
reason* instead of failing, and this module reports exactly what is missing
rather than installing anything.
"""

from __future__ import annotations

import importlib.metadata as metadata
import sys
from pathlib import Path

# Make the project root importable for both invocation styles.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.spark_session import (  # noqa: E402  (import after sys.path setup)
    check_spark_environment,
    create_spark_session,
    format_environment_report,
    stop_spark_session,
)

#: Packages this project depends on, keyed by distribution name.
PROJECT_PACKAGES = (
    "pyspark",
    "pandas",
    "numpy",
    "scikit-learn",
    "PyYAML",
    "pyarrow",
    "plotly",
    "streamlit",
    "networkx",
    "mlxtend",
    "pytest",
)


def installed_packages() -> dict:
    """Return ``{distribution: version_or_None}`` for the project's packages."""
    status = {}
    for package in PROJECT_PACKAGES:
        try:
            status[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            status[package] = None
    return status


def test_environment_report_shape() -> None:
    """The environment probe must return the documented keys and never raise."""
    report = check_spark_environment()
    for key in (
        "pyspark_available",
        "pyspark_version",
        "java_available",
        "java_version",
        "java_home",
        "python_version",
        "issues",
    ):
        assert key in report, f"environment report is missing '{key}'"
    assert isinstance(report["issues"], list)


def test_environment_report_renders() -> None:
    """The report must render to a non-empty string."""
    text = format_environment_report()
    assert "PySpark environment report" in text


def test_spark_session_initializes() -> None:
    """Create a session, run a trivial no-data query, then shut it down.

    Skipped (with a reason printed) when PySpark or Java is unavailable.
    This touches no dataset from ``data/``.
    """
    report = check_spark_environment()
    missing = []
    if not report["pyspark_available"]:
        missing.append("PySpark")
    if not report["java_available"]:
        missing.append("Java runtime")
    if missing:
        print(f"      SKIPPED - missing: {', '.join(missing)}")
        return

    session = create_spark_session(app_name="DrugSafetyEnvironmentCheck")
    try:
        assert session is not None
        assert session.version, "Spark reported an empty version"
        # Sanity check that the session can actually execute a job.
        assert session.range(3).count() == 3
    finally:
        stop_spark_session(session)


def main() -> int:
    """Print the dependency report, run the checks, then print the env report."""
    print("Dependency status")
    print("-----------------")
    status = installed_packages()
    for package, version in status.items():
        print(f"  {package:14s} {'INSTALLED ' + version if version else 'MISSING'}")

    missing = [name for name, version in status.items() if version is None]
    if missing:
        print(f"\n  Missing packages (not installed): {', '.join(missing)}")
    print()

    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        print(f"RUN   {test.__name__}")
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report any failure
            failures += 1
            print(f"FAIL  {test.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {test.__name__}")

    print()
    print(format_environment_report())
    print(f"\n{len(tests) - failures}/{len(tests)} Spark checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
