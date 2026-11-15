"""Configuration loading for the Drug Safety Intelligence Platform.

All tunable settings live in the project-root ``config.yaml``.  This module is
the single place that knows where that file is and how to read it, so no other
module in the project needs to hard-code paths or thresholds.

Typical use::

    from src.utils.config import get_path, get_faers_config

    raw_dir = get_path("raw")            # -> <project root>/data/raw
    delimiter = get_faers_config()["delimiter"]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

__all__ = [
    "ConfigError",
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_PATH",
    "get_project_root",
    "load_config",
    "get_config",
    "validate_config",
    "reset_cache",
    "get_section",
    "get_setting",
    "resolve_path",
    "get_path",
    "get_paths",
    "get_faers_config",
    "get_quarters",
    "get_quarter_dirs",
    "get_spark_config",
    "get_analytics_config",
    "get_visualization_config",
    "get_logging_config",
]

#: Environment variable that can point the loader at an alternative config file.
CONFIG_ENV_VAR = "DRUG_SAFETY_CONFIG"

#: Project root = two levels above this file (``src/utils/config.py``).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Default configuration location.
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

#: Keys that must be present for the rest of the platform to work.
REQUIRED_KEYS: Dict[str, Tuple[str, ...]] = {
    "project": ("name", "version"),
    "paths": ("raw", "processed", "analytics", "models", "reports", "logs"),
    "faers": (
        "quarters",
        "filename_pattern",
        "delimiter",
        "encoding",
        "header",
        "tables",
        "join_key",
    ),
    "spark": ("app_name", "master", "driver_memory", "shuffle_partitions", "log_level"),
    "analytics": ("signal_detection", "cooccurrence", "anomalies"),
    "visualization": ("theme", "top_n"),
    "logging": ("level", "file"),
}

_MISSING = object()

# Cache keyed by resolved config path, so ``force_reload`` and explicit
# alternative config files both behave predictably.
_cache: Dict[str, Dict[str, Any]] = {}


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing, unreadable or invalid."""


# ---------------------------------------------------------------------------
# Locating and reading config.yaml
# ---------------------------------------------------------------------------
def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return PROJECT_ROOT


def _default_config_path() -> Path:
    """Return the config path, honouring the ``DRUG_SAFETY_CONFIG`` override."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return DEFAULT_CONFIG_PATH


def load_config(
    config_path: Optional[Path | str] = None,
    *,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """Load and validate ``config.yaml``.

    Args:
        config_path: Explicit config file. Defaults to ``DRUG_SAFETY_CONFIG``
            if set, otherwise ``<project root>/config.yaml``.
        force_reload: Bypass the in-memory cache.

    Returns:
        The parsed configuration as a plain dict.

    Raises:
        ConfigError: If the file is missing, unreadable, not valid YAML, not a
            mapping, or missing required keys.
    """
    path = Path(config_path) if config_path is not None else _default_config_path()
    try:
        resolved = path.expanduser().resolve()
    except OSError:  # pragma: no cover - only on exotic/ill-formed paths
        resolved = path
    cache_key = str(resolved)

    if not force_reload and cache_key in _cache:
        return _cache[cache_key]

    if not resolved.exists():
        raise ConfigError(
            f"Configuration file not found: {resolved}\n"
            f"Expected the project configuration at: {DEFAULT_CONFIG_PATH}\n"
            f"Either create it, or set the {CONFIG_ENV_VAR} environment variable "
            f"to an explicit path."
        )
    if not resolved.is_file():
        raise ConfigError(f"Configuration path is not a file: {resolved}")

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {resolved}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read {resolved}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"{resolved} must contain a YAML mapping at the top level, "
            f"got {type(data).__name__}."
        )

    validate_config(data, source=resolved)
    _cache[cache_key] = data
    return data


def get_config() -> Dict[str, Any]:
    """Return the cached default configuration, loading it on first use."""
    return load_config()


def validate_config(config: Mapping[str, Any], source: Optional[Path] = None) -> None:
    """Check that every required section and key is present.

    Raises:
        ConfigError: Listing all missing entries at once.
    """
    where = f" in {source}" if source else ""
    problems: List[str] = []

    for section, required_keys in REQUIRED_KEYS.items():
        value = config.get(section, _MISSING)
        if value is _MISSING:
            problems.append(f"missing section '{section}'")
            continue
        if not isinstance(value, Mapping):
            problems.append(f"section '{section}' must be a mapping")
            continue
        for key in required_keys:
            if key not in value:
                problems.append(f"missing key '{section}.{key}'")

    if problems:
        raise ConfigError(
            f"Invalid configuration{where}:\n  - " + "\n  - ".join(problems)
        )


def reset_cache() -> None:
    """Clear the configuration cache (mainly useful in tests)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Generic access helpers
# ---------------------------------------------------------------------------
def get_section(name: str) -> Dict[str, Any]:
    """Return a top-level config section as a dict.

    Raises:
        ConfigError: If the section is absent or is not a mapping.
    """
    config = get_config()
    value = config.get(name, _MISSING)
    if value is _MISSING:
        available = ", ".join(sorted(config))
        raise ConfigError(
            f"No '{name}' section in the configuration. Available sections: {available}"
        )
    if not isinstance(value, Mapping):
        raise ConfigError(f"Section '{name}' is not a mapping.")
    return dict(value)


def get_setting(key_path: str, default: Any = _MISSING) -> Any:
    """Read a nested setting using a dotted path, e.g. ``"spark.app_name"``.

    Args:
        key_path: Dotted path into the configuration.
        default: Returned when the key is absent. If omitted, a missing key
            raises :class:`ConfigError`.
    """
    node: Any = get_config()
    for part in key_path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            if default is not _MISSING:
                return default
            raise ConfigError(f"Missing configuration key: '{key_path}'")
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def resolve_path(value: Path | str, base: Optional[Path] = None) -> Path:
    """Resolve a config path value against the project root using pathlib.

    Absolute paths are returned unchanged; relative paths are joined to
    ``base`` (project root by default). No existence check is performed.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or PROJECT_ROOT) / path
    return Path(os.path.normpath(str(path)))


def get_paths() -> Dict[str, Path]:
    """Return every entry of the ``paths`` section as absolute Paths."""
    return {key: resolve_path(value) for key, value in get_section("paths").items()}


def get_path(key: str) -> Path:
    """Return one directory from the ``paths`` section as an absolute Path."""
    paths = get_section("paths")
    if key not in paths:
        available = ", ".join(sorted(paths))
        raise ConfigError(f"Unknown path key '{key}'. Available: {available}")
    return resolve_path(paths[key])


# ---------------------------------------------------------------------------
# Domain-specific accessors
# ---------------------------------------------------------------------------
def get_faers_config() -> Dict[str, Any]:
    """Return the ``faers`` section (delimiter, encoding, tables, quarters...)."""
    section = get_section("faers")
    quarters = section.get("quarters")
    if not isinstance(quarters, (list, tuple)) or not quarters:
        raise ConfigError("'faers.quarters' must be a non-empty list.")
    tables = section.get("tables")
    if not isinstance(tables, Mapping) or not tables:
        raise ConfigError("'faers.tables' must be a non-empty mapping.")
    return section


def get_quarters() -> List[str]:
    """Return the configured quarter labels, e.g. ``["2026_Q1", "2026_Q2"]``."""
    return [str(q) for q in get_faers_config()["quarters"]]


def get_quarter_dirs() -> List[Tuple[str, Path]]:
    """Return ``(quarter_label, raw_directory)`` pairs for every quarter."""
    raw_dir = get_path("raw")
    return [(quarter, raw_dir / quarter) for quarter in get_quarters()]


def get_spark_config() -> Dict[str, Any]:
    """Return the ``spark`` section."""
    return get_section("spark")


def get_analytics_config(name: Optional[str] = None) -> Dict[str, Any]:
    """Return the ``analytics`` section, or one of its sub-sections."""
    section = get_section("analytics")
    if name is None:
        return section
    if name not in section:
        available = ", ".join(sorted(k for k, v in section.items() if isinstance(v, Mapping)))
        raise ConfigError(f"Unknown analytics section '{name}'. Available: {available}")
    return dict(section[name])


def get_visualization_config() -> Dict[str, Any]:
    """Return the ``visualization`` section."""
    return get_section("visualization")


def get_logging_config() -> Dict[str, Any]:
    """Return the ``logging`` section."""
    return get_section("logging")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    config = get_config()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config file  : {DEFAULT_CONFIG_PATH}")
    print(f"Sections     : {', '.join(sorted(config))}")
    print("Paths:")
    for name, path in get_paths().items():
        print(f"  {name:12s} -> {path}")
    print("Quarters:")
    for quarter, directory in get_quarter_dirs():
        print(f"  {quarter} -> {directory} (exists={directory.exists()})")
