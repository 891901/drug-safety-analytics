"""Data-access layer for the dashboard.

Responsibilities
----------------
* Know **where every real analytical output lives** (the registry in
  :data:`SOURCES`).  Nothing here invents data: if a file is missing the loader
  returns an empty result and :func:`output_status` flags it so the UI can show
  a clear "output missing" message instead of a fake chart.
* Read CSV with the **standard :mod:`csv` module** and JSON with :mod:`json`
  (never pandas / PyArrow), so it honours the Windows Application Control
  restriction.  Large files are read either with a hard **row cap** or by
  **streaming** with an early stop, keeping memory bounded.
* Cache every read with ``st.cache_data`` keyed by the file's mtime, so a page
  interaction never re-reads or recomputes anything.

All paths resolve through :mod:`src.utils.config` (the same ``config.yaml`` the
rest of the platform uses), with a self-contained fallback if that import fails.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap + project configuration
# ---------------------------------------------------------------------------
_DASHBOARD_DIR = Path(__file__).resolve().parents[1]  # .../dashboard
PROJECT_ROOT = _DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - exercised implicitly
    from src.utils.config import (  # type: ignore
        get_path,
        get_quarters,
        get_visualization_config,
    )

    ANALYTICS_DIR: Path = get_path("analytics")
    REPORTS_DIR: Path = get_path("reports")
    MODELS_DIR: Path = get_path("models")
    _viz = get_visualization_config()
    THEME: str = str(_viz.get("theme", "plotly_white"))
    DEFAULT_TOP_N: int = int(_viz.get("top_n", 20))
    QUARTERS: List[str] = get_quarters()
except Exception:  # noqa: BLE001 - dashboard must still boot if config is absent
    ANALYTICS_DIR = PROJECT_ROOT / "data" / "analytics"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    MODELS_DIR = PROJECT_ROOT / "models"
    THEME = "plotly_white"
    DEFAULT_TOP_N = 20
    QUARTERS = ["2026_Q1", "2026_Q2"]

VISUALIZATIONS_DIR: Path = ANALYTICS_DIR / "visualizations"

# ---------------------------------------------------------------------------
# Output registry - logical key -> real file on disk
# ---------------------------------------------------------------------------
SOURCES: Dict[str, Path] = {
    # --- ADR (adverse drug reaction) core ---------------------------------
    "adr_summary": ANALYTICS_DIR / "adr_summary.json",
    "adr_top_reactions": ANALYTICS_DIR / "adr_top_reactions.csv",
    "adr_drug_report_counts": ANALYTICS_DIR / "adr_drug_report_counts.csv",
    "adr_drug_reaction_pairs": ANALYTICS_DIR / "adr_drug_reaction_pairs.csv",
    "adr_reaction_by_quarter": ANALYTICS_DIR / "adr_reaction_by_quarter.csv",
    "adr_outcome_counts": ANALYTICS_DIR / "adr_outcome_counts.csv",
    "adr_outcome_by_quarter": ANALYTICS_DIR / "adr_outcome_by_quarter.csv",
    "adr_demographics_age": ANALYTICS_DIR / "adr_demographics_age.csv",
    "adr_demographics_age_by_quarter": ANALYTICS_DIR / "adr_demographics_age_by_quarter.csv",
    "adr_demographics_sex": ANALYTICS_DIR / "adr_demographics_sex.csv",
    "adr_demographics_sex_by_quarter": ANALYTICS_DIR / "adr_demographics_sex_by_quarter.csv",
    # --- drug co-occurrence -----------------------------------------------
    "cooccurrence_summary": ANALYTICS_DIR / "drug_cooccurrence" / "cooccurrence_summary.json",
    "cooccurrence_top_pairs": ANALYTICS_DIR / "drug_cooccurrence" / "cooccurrence_top_pairs.csv",
    "cooccurrence_top_groups": ANALYTICS_DIR / "drug_cooccurrence" / "cooccurrence_top_groups.csv",
    "cooccurrence_pairs_by_quarter": ANALYTICS_DIR
    / "drug_cooccurrence"
    / "cooccurrence_pairs_by_quarter.csv",
    # --- apriori association rules ----------------------------------------
    "apriori_summary": ANALYTICS_DIR / "apriori" / "apriori_summary.json",
    "apriori_rules": ANALYTICS_DIR / "apriori" / "apriori_rules.csv",
    "apriori_frequent_itemsets": ANALYTICS_DIR / "apriori" / "apriori_frequent_itemsets.csv",
    "apriori_rules_by_quarter": ANALYTICS_DIR / "apriori" / "apriori_rules_by_quarter.csv",
    # --- statistical signal detection (PRR / ROR) -------------------------
    "signal_summary": ANALYTICS_DIR / "signal_detection" / "signal_detection_summary.json",
    "signal_signals": ANALYTICS_DIR / "signal_detection" / "signal_detection_signals.csv",
    "signal_signals_by_quarter": ANALYTICS_DIR
    / "signal_detection"
    / "signal_detection_signals_by_quarter.csv",
    # --- machine learning (Isolation Forest) ------------------------------
    "anomaly_summary": ANALYTICS_DIR / "anomaly_detection" / "anomaly_summary.json",
    "anomaly_results": ANALYTICS_DIR / "anomaly_detection" / "anomaly_results.csv",
    "anomaly_top_pairs": ANALYTICS_DIR / "anomaly_detection" / "anomaly_top_pairs.csv",
    # --- drug network ------------------------------------------------------
    "network_summary": ANALYTICS_DIR / "network" / "drug_network_summary.json",
    "network_nodes": ANALYTICS_DIR / "network" / "drug_network_nodes.csv",
    "network_edges": ANALYTICS_DIR / "network" / "drug_network_edges.csv",
    "network_communities": ANALYTICS_DIR / "network" / "drug_network_communities.csv",
    "network_html": ANALYTICS_DIR / "network" / "drug_network.html",
    "network_top_connected_html": ANALYTICS_DIR / "network" / "top_connected_drugs.html",
    "network_relationships_html": ANALYTICS_DIR / "network" / "drug_relationships.html",
    # --- temporal + demographic -------------------------------------------
    "temporal_summary": ANALYTICS_DIR / "temporal_demographic" / "temporal_demographic_summary.json",
    "temporal_reports_by_quarter": ANALYTICS_DIR
    / "temporal_demographic"
    / "temporal_reports_by_quarter.csv",
    "temporal_monthly_trend": ANALYTICS_DIR
    / "temporal_demographic"
    / "temporal_monthly_trend.csv",
    "temporal_top_reactions_by_quarter": ANALYTICS_DIR
    / "temporal_demographic"
    / "temporal_top_reactions_by_quarter.csv",
    "temporal_drug_frequency_change": ANALYTICS_DIR
    / "temporal_demographic"
    / "temporal_drug_frequency_change.csv",
    "temporal_serious_by_quarter": ANALYTICS_DIR
    / "temporal_demographic"
    / "temporal_serious_outcomes_by_quarter.csv",
    "demographic_age_distribution": ANALYTICS_DIR
    / "temporal_demographic"
    / "demographic_age_distribution.csv",
    "demographic_gender_distribution": ANALYTICS_DIR
    / "temporal_demographic"
    / "demographic_gender_distribution.csv",
    "demographic_reactions_by_age": ANALYTICS_DIR
    / "temporal_demographic"
    / "demographic_reactions_by_age.csv",
    "demographic_reactions_by_gender": ANALYTICS_DIR
    / "temporal_demographic"
    / "demographic_reactions_by_gender.csv",
    "demographic_age_by_quarter": ANALYTICS_DIR
    / "temporal_demographic"
    / "demographic_age_by_quarter.csv",
    "demographic_gender_by_quarter": ANALYTICS_DIR
    / "temporal_demographic"
    / "demographic_gender_by_quarter.csv",
    "demographic_serious_outcomes": ANALYTICS_DIR
    / "temporal_demographic"
    / "demographic_serious_outcomes.csv",
    # --- pre-rendered HTML figures (reused verbatim) ----------------------
    "viz_quarterly_comparison": VISUALIZATIONS_DIR / "quarterly_comparison.html",
    "viz_monthly_trend": VISUALIZATIONS_DIR / "monthly_trend.html",
    "viz_age_distribution": VISUALIZATIONS_DIR / "age_distribution.html",
    "viz_gender_distribution": VISUALIZATIONS_DIR / "gender_distribution.html",
    "viz_top_reactions": VISUALIZATIONS_DIR / "top_reactions.html",
    "viz_serious_outcomes": VISUALIZATIONS_DIR / "serious_outcomes.html",
    "viz_drug_frequency_change": VISUALIZATIONS_DIR / "drug_frequency_change.html",
}

#: Human-readable labels used by the availability panel.
SOURCE_LABELS: Dict[str, str] = {
    "adr_summary": "ADR summary",
    "adr_top_reactions": "ADR top reactions",
    "adr_drug_report_counts": "ADR drug report counts",
    "adr_drug_reaction_pairs": "ADR drug-reaction pairs",
    "adr_reaction_by_quarter": "ADR reactions by quarter",
    "adr_outcome_counts": "ADR outcome counts",
    "adr_outcome_by_quarter": "ADR outcomes by quarter",
    "adr_demographics_age": "ADR age distribution",
    "adr_demographics_sex": "ADR gender distribution",
    "cooccurrence_summary": "Co-occurrence summary",
    "cooccurrence_top_pairs": "Co-occurrence top pairs",
    "cooccurrence_top_groups": "Co-occurrence top groups",
    "cooccurrence_pairs_by_quarter": "Co-occurrence pairs by quarter",
    "apriori_summary": "Apriori summary",
    "apriori_rules": "Apriori rules",
    "apriori_frequent_itemsets": "Apriori frequent itemsets",
    "signal_summary": "Signal detection summary",
    "signal_signals": "Signal detection signals",
    "signal_signals_by_quarter": "Signals by quarter",
    "anomaly_summary": "Anomaly summary",
    "anomaly_results": "Anomaly results",
    "anomaly_top_pairs": "Anomaly top pairs",
    "network_summary": "Network summary",
    "network_nodes": "Network nodes",
    "network_edges": "Network edges",
    "network_communities": "Network communities",
    "temporal_summary": "Temporal/demographic summary",
    "temporal_reports_by_quarter": "Reports by quarter",
    "temporal_monthly_trend": "Monthly reporting trend",
    "temporal_drug_frequency_change": "Drug frequency change",
    "temporal_serious_by_quarter": "Serious outcomes by quarter",
    "demographic_age_distribution": "Age distribution",
    "demographic_gender_distribution": "Gender distribution",
    "demographic_reactions_by_age": "Reactions by age",
    "demographic_reactions_by_gender": "Reactions by gender",
    "demographic_serious_outcomes": "Serious outcomes by demographic",
}


# ---------------------------------------------------------------------------
# Small conversion helpers
# ---------------------------------------------------------------------------
def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort float conversion; returns ``default`` on failure."""
    if value is None or value == "":
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    """Best-effort int conversion; returns ``default`` on failure."""
    if value is None or value == "":
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    """Interpret the ``True``/``False`` strings written by the pipelines."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fmt_int(value: Any) -> str:
    """Thousands-separated integer string, or ``n/a``."""
    number = as_int(value, default=-1)
    return "n/a" if number < 0 else f"{number:,}"


def fmt_num(value: Any, digits: int = 3) -> str:
    """Fixed-precision float string, or ``n/a``."""
    number = as_float(value)
    return "n/a" if number is None else f"{number:,.{digits}f}"


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def source_path(key: str) -> Optional[Path]:
    """Return the on-disk path for a registry key, or ``None``."""
    return SOURCES.get(key)


def is_available(key: str) -> bool:
    """Whether the output behind ``key`` exists on disk."""
    path = SOURCES.get(key)
    return bool(path and path.is_file())


def output_status() -> List[Dict[str, Any]]:
    """Return ``[{key, label, path, exists}]`` for every registered output."""
    rows = []
    for key in SOURCES:
        path = SOURCES[key]
        rows.append(
            {
                "key": key,
                "label": SOURCE_LABELS.get(key, key),
                "path": str(path),
                "exists": path.is_file(),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Cached readers
# ---------------------------------------------------------------------------
def _mtime(path: Optional[Path]) -> float:
    """Modification time used as part of every cache key (0.0 if unavailable)."""
    try:
        return path.stat().st_mtime if path else 0.0
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_json(key: str, _mtime_token: float) -> Optional[Dict[str, Any]]:
    path = SOURCES.get(key)
    if not path or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def _load_rows(
    key: str,
    _mtime_token: float,
    limit: Optional[int],
    columns: Optional[tuple],
) -> List[Dict[str, str]]:
    path = SOURCES.get(key)
    if not path or not path.is_file():
        return []
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                if limit is not None and index >= limit:
                    break
                if columns is not None:
                    rows.append({name: row.get(name, "") for name in columns})
                else:
                    rows.append(dict(row))
    except OSError:
        return []
    return rows


@st.cache_data(show_spinner=False)
def _count_rows(key: str, _mtime_token: float) -> int:
    path = SOURCES.get(key)
    if not path or not path.is_file():
        return 0
    total = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            total = sum(1 for _ in reader)
    except OSError:
        return 0
    return total


@st.cache_data(show_spinner=False)
def _stream_filter(
    key: str,
    _mtime_token: float,
    columns: tuple,
    value: str,
    limit: int,
) -> List[Dict[str, str]]:
    """Stream a CSV, keeping rows where any of ``columns`` equals ``value``."""
    path = SOURCES.get(key)
    if not path or not path.is_file():
        return []
    target = str(value).strip()
    out: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if any(str(row.get(name, "")).strip() == target for name in columns):
                    out.append(dict(row))
                    if len(out) >= limit:
                        break
    except OSError:
        return []
    return out


@st.cache_data(show_spinner=False)
def _column_values(
    key: str,
    _mtime_token: float,
    column: str,
    limit: Optional[int],
) -> List[str]:
    """Stream a single CSV column into a list of raw strings (bounded)."""
    path = SOURCES.get(key)
    if not path or not path.is_file():
        return []
    out: List[str] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                out.append(row.get(column, ""))
                if limit is not None and len(out) >= limit:
                    break
    except OSError:
        return []
    return out


# --- public API ------------------------------------------------------------
def get_json(key: str) -> Optional[Dict[str, Any]]:
    """Load a JSON output (cached). Returns ``None`` when missing/invalid."""
    return _load_json(key, _mtime(SOURCES.get(key)))


def column_values(
    key: str,
    column: str,
    *,
    limit: Optional[int] = None,
) -> List[str]:
    """Return the raw values of one CSV column (cached, bounded by ``limit``)."""
    return _column_values(key, _mtime(SOURCES.get(key)), column, limit)


def numeric_column(
    key: str,
    column: str,
    *,
    limit: Optional[int] = None,
) -> List[float]:
    """Return one CSV column coerced to floats, dropping unparseable cells."""
    values: List[float] = []
    for raw in column_values(key, column, limit=limit):
        number = as_float(raw)
        if number is not None:
            values.append(number)
    return values



def get_rows(
    key: str,
    *,
    limit: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    """Load CSV rows (cached), optionally capped to ``limit`` rows / columns."""
    cols = tuple(columns) if columns else None
    return _load_rows(key, _mtime(SOURCES.get(key)), limit, cols)


def count_rows(key: str) -> int:
    """Count the data rows of a CSV without holding it in memory (cached)."""
    return _count_rows(key, _mtime(SOURCES.get(key)))


def filter_rows(
    key: str,
    columns: Sequence[str],
    value: Any,
    *,
    limit: int = 500,
) -> List[Dict[str, str]]:
    """Stream a CSV and return rows whose ``columns`` match ``value`` (cached).

    This is the bounded-memory path used by the drug investigation page: the file
    is scanned once and only matching rows are retained.
    """
    return _stream_filter(
        key, _mtime(SOURCES.get(key)), tuple(columns), str(value), int(limit)
    )


def distinct_values(
    key: str,
    column: str,
    *,
    limit: Optional[int] = None,
) -> List[str]:
    """Return the sorted distinct values of ``column`` (bounded by ``limit``).

    Ordering by descending frequency would need a full scan; to stay bounded and
    honest we scan at most ``limit`` rows and sort the result alphabetically.
    """
    rows = get_rows(key, limit=limit, columns=[column])
    seen = {row.get(column, "").strip() for row in rows}
    seen.discard("")
    return sorted(seen)


def dataset_links() -> List[Dict[str, Any]]:
    """Convenience list of registry entries used by the download/index widgets."""
    return output_status()
