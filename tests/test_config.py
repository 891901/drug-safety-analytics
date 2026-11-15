"""Verification of the configuration layer.

Runs under pytest (``pytest tests/test_config.py``) and also standalone
(``python tests/test_config.py``) so it can be used before pytest is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable for both invocation styles.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (  # noqa: E402  (import after sys.path setup)
    DEFAULT_CONFIG_PATH,
    REQUIRED_KEYS,
    ConfigError,
    get_analytics_config,
    get_config,
    get_faers_config,
    get_path,
    get_paths,
    get_quarter_dirs,
    get_setting,
    get_spark_config,
    load_config,
)


def test_config_file_exists() -> None:
    """config.yaml must exist at the project root."""
    assert DEFAULT_CONFIG_PATH.is_file(), f"config.yaml not found at {DEFAULT_CONFIG_PATH}"


def test_config_loads() -> None:
    """config.yaml must load into a non-empty mapping."""
    config = get_config()
    assert isinstance(config, dict), "config.yaml did not parse into a mapping"
    assert config, "config.yaml is empty"


def test_required_sections_and_keys() -> None:
    """Every section/key declared in REQUIRED_KEYS must be present."""
    config = get_config()
    missing = []
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            missing.append(section)
            continue
        missing.extend(f"{section}.{key}" for key in keys if key not in config[section])
    assert not missing, f"missing configuration entries: {missing}"


def test_paths_resolve_under_project_root() -> None:
    """All configured paths must be absolute and inside the project."""
    for name, path in get_paths().items():
        assert path.is_absolute(), f"path '{name}' is not absolute: {path}"
        assert str(path).startswith(str(PROJECT_ROOT)), f"path '{name}' escapes root: {path}"


def test_faers_settings() -> None:
    """FAERS settings must describe the real ASCII layout."""
    faers = get_faers_config()
    assert faers["delimiter"] == "$", "FAERS ASCII files are '$'-delimited"
    assert len(faers["tables"]) == 7, "expected the 7 FAERS ASCII tables"
    assert set(faers["tables"]) == {"DEMO", "DRUG", "REAC", "OUTC", "RPSR", "THER", "INDI"}
    assert faers["join_key"] == "PRIMARYID"

    quarter_dirs = get_quarter_dirs()
    assert len(quarter_dirs) == len(faers["quarters"])
    assert quarter_dirs[0][1] == get_path("raw") / quarter_dirs[0][0]


def test_spark_settings() -> None:
    """Spark settings must be complete enough to build a local session."""
    spark = get_spark_config()
    assert spark["master"].startswith("local"), "this project runs Spark locally"
    assert spark["driver_memory"], "spark.driver_memory must be set"
    assert int(spark["shuffle_partitions"]) > 0


def test_analytics_settings() -> None:
    """Analytics thresholds must be present for signal detection and ML."""
    signals = get_analytics_config("signal_detection")
    assert float(signals["prr_threshold"]) > 0
    anomalies = get_analytics_config("anomalies")
    assert 0 < float(anomalies["contamination"]) < 1


def test_nested_setting_access() -> None:
    """Dotted access works, and a default is honoured for unknown keys."""
    assert float(get_setting("analytics.signal_detection.prr_threshold")) == 2.0
    assert get_setting("does.not.exist", default=None) is None


def test_missing_file_raises_clear_error() -> None:
    """A missing config file must fail with an actionable message."""
    missing_path = PROJECT_ROOT / "definitely_not_a_real_config.yaml"
    try:
        load_config(missing_path)
    except ConfigError as exc:
        assert "not found" in str(exc).lower(), f"unhelpful message: {exc}"
    else:
        raise AssertionError("expected ConfigError when the config file is missing")


def main() -> int:
    """Run every ``test_*`` function in this module and print a summary."""
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report any failure
            failures += 1
            print(f"FAIL  {test.__name__}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {test.__name__}")

    print(f"\n{len(tests) - failures}/{len(tests)} configuration checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
