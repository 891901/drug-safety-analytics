"""Isolation Forest anomaly detection over the project's own PRR/ROR results.

Reads the drug/reaction signal tables produced by
:mod:`src.analytics.signal_detection` - real aggregated FAERS 2026 Q1/Q2
disproportionality results - engineers a compact numerical feature matrix and
fits a scikit-learn :class:`~sklearn.ensemble.IsolationForest` to flag unusual
*reporting patterns* among those pairs.

Why the inputs are the project's own aggregate tables
-----------------------------------------------------
No synthetic data is generated anywhere in this module. The only inputs are the
CSV artefacts the project already produced from the real FAERS ASCII extracts:

* ``data/analytics/signal_detection/signal_detection_signals.csv``
  - one row per screened drug/reaction pair (overall), with the 2x2 contingency
    counts (``a``, ``b``, ``c``, ``d``), the report-level marginals
    (``n_drug``, ``n_reac``, ``n_total``), ``PRR``, ``ROR``, both 95%
    confidence intervals and the Pearson chi-square statistic.
* ``data/analytics/signal_detection/signal_detection_signals_by_quarter.csv``
  - the same measures scored separately for Q1 and Q2, used for the
    quarter-wise features.

Documented population limitation
--------------------------------
That upstream signal table is **capped at 200,000 rows** (the strongest pairs by
PRR), so the model is trained on the screened signal population rather than on
all 811,327 screened pairs. This is a deliberate, documented scope - the
anomaly detector therefore searches for unusual patterns *within* the pairs
that already pass the pharmacovigilance screening thresholds. It is not a
population-wide detector, and it cannot see pairs that the upstream screen
discarded.

Feature engineering
-------------------
Eleven features are derived per drug/reaction pair:

1. ``log10_pair_reports``     - drug/reaction pair frequency (``a``)
2. ``log10_drug_reports``     - drug report frequency (``n_drug``)
3. ``log10_reaction_reports`` - reaction report frequency (``n_reac``)
4. ``log10_prr``              - proportional reporting ratio
5. ``log10_ror``              - reporting odds ratio
6. ``prr_ci_log_width``       - log PRR CI width, ``log10(high) - log10(low)``
7. ``log10_ror_ci_low``       - lower 95% ROR confidence bound
8. ``log10_chi_square``       - Pearson chi-square statistic
9. ``quarters_observed``      - in how many quarters the pair is a signal
10. ``quarter_report_balance``- ``(max - min) / sum`` of the per-quarter counts
11. ``prr_quarter_spread``    - ``log10`` span of the per-quarter PRR values

Counts and statistics are strongly right-skewed and span many orders of
magnitude, so they are **log-transformed** (a variance-stabilising monotone
transform). They are deliberately **not standardised**: an Isolation Forest
partitions each feature independently at random thresholds, so it is invariant
to any per-feature monotone rescaling - adding a scaler would change nothing
except the extra dependency. That is what "scale only where appropriate" means
here, and it is reported explicitly.

Missing / infinite values
-------------------------
Every input number is parsed defensively; anything that is not finite (or is
non-positive where a logarithm needs it) becomes ``NaN``, is counted per
feature, and is then **median-imputed** from the finite values of that feature.
The number of imputed cells is recorded in the report and the summary. The
quarter features are never ``NaN``: when fewer than two quarters are observed
they are defined as ``0.0`` (no measurable change) and the
``quarters_observed`` feature disambiguates that from a genuine zero.

Anomalies are not adverse reactions
-----------------------------------
An anomaly here means "this drug/reaction pair looks unlike the rest of the
screened signal population across the chosen features". It is **not** proof of a
causal drug reaction, of clinical risk, of incidence and not a new safety
finding. FAERS is a spontaneous reporting system with no exposure denominator
and strong reporting bias.

Environment notes
-----------------
Pure ``numpy`` + ``scikit-learn`` + the Python standard library. No pandas, no
PyArrow, no Spark, no UDFs. The input tables are streamed row-by-row and only
the aggregated feature matrix (200,000 x 11 float64, ~18 MB) is held in memory.

Usage::

    python -m src.machine_learning.anomaly_detection
    python -m src.machine_learning.anomaly_detection --contamination 0.02
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.utils.config import ConfigError, get_analytics_config, get_path, get_quarters
from src.utils.logging_setup import get_logger

__all__ = [
    "AnomalyDetectionError",
    "ANOMALY_DISCLAIMER",
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "AnomalyResults",
    "read_signal_table",
    "read_quarter_table",
    "build_feature_matrix",
    "assemble_matrix",
    "detect_anomalies",
    "write_outputs",
    "render_report",
    "run",
    "main",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: The feature vector, in the exact order the model consumes it.
FEATURE_NAMES: Tuple[str, ...] = (
    "log10_pair_reports",
    "log10_drug_reports",
    "log10_reaction_reports",
    "log10_prr",
    "log10_ror",
    "prr_ci_log_width",
    "log10_ror_ci_low",
    "log10_chi_square",
    "quarters_observed",
    "quarter_report_balance",
    "prr_quarter_spread",
)

#: One-line description per feature, echoed in the report.
FEATURE_DESCRIPTIONS: Mapping[str, str] = {
    "log10_pair_reports": "log10(drug-reaction pair reports + 1)",
    "log10_drug_reports": "log10(drug report frequency + 1)",
    "log10_reaction_reports": "log10(reaction report frequency + 1)",
    "log10_prr": "log10(PRR)",
    "log10_ror": "log10(ROR)",
    "prr_ci_log_width": "log10(PRR CI high) - log10(PRR CI low)",
    "log10_ror_ci_low": "log10(ROR 95% CI lower bound)",
    "log10_chi_square": "log10(chi-square + 1)",
    "quarters_observed": "number of quarters in which the pair is a signal",
    "quarter_report_balance": "(max - min) / sum of per-quarter report counts",
    "prr_quarter_spread": "log10 span of the per-quarter PRR values",
}

#: Raw upstream columns this module consumes from the overall signal table.
_RAW_COLUMNS: Tuple[str, ...] = (
    "a",
    "n_drug",
    "n_reac",
    "prr",
    "ror",
    "prr_ci_low",
    "prr_ci_high",
    "ror_ci_low",
    "chi_square",
)

#: How many anomalous pairs are listed in the report.
TOP_ANOMALIES_REPORT = 20
#: How many anomalous pairs are written to the dedicated shortlist CSV.
TOP_ANOMALIES_CSV = 500
#: How many anomalous pairs are embedded in the JSON summary (kept small).
TOP_ANOMALIES_SUMMARY = 25

#: Mandatory caution; repeated in the report and the summary.
ANOMALY_DISCLAIMER = (
    "An anomaly flag means only that the drug/reaction pair sits apart from the "
    "rest of the screened signal population across the chosen features. It is "
    "NOT evidence of a causal adverse drug reaction, NOT a measure of clinical "
    "risk or incidence, and NOT a new safety signal on its own. FAERS is a "
    "spontaneous reporting system with no exposure denominator and heavy "
    "reporting bias."
)


class AnomalyDetectionError(RuntimeError):
    """Raised when the input artefacts or the model cannot be used."""


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def _log10_shifted(values: np.ndarray) -> np.ndarray:
    """``log10(x + 1)`` for finite ``x >= -1``; ``NaN`` elsewhere."""
    out = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values) & (values >= -1.0)
    out[mask] = np.log10(values[mask] + 1.0)
    return out


def _log10_positive(values: np.ndarray) -> np.ndarray:
    """``log10(x)`` for finite ``x > 0``; ``NaN`` elsewhere."""
    out = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values) & (values > 0.0)
    out[mask] = np.log10(values[mask])
    return out


def _to_float(text: Any) -> float:
    """Parse a CSV cell defensively; anything unparseable becomes ``NaN``."""
    if text is None or text == "":
        return float("nan")
    try:
        value = float(text)
    except (TypeError, ValueError):
        return float("nan")
    return value


def _to_int(value: float) -> int:
    """Integer for a parsed count; a non-finite value degrades to ``0``."""
    return int(value) if math.isfinite(value) else 0


def _fmt(value: Any, digits: int = 10) -> Any:
    """Round floats for CSV output; non-finite values are written as empty."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return round(float(value), digits)
    return value


# ---------------------------------------------------------------------------
# Input loading (streamed, standard library only)
# ---------------------------------------------------------------------------
def read_signal_table(path: Path) -> Tuple[List[Tuple[str, str]], Dict[str, np.ndarray]]:
    """Stream the overall signal CSV into labels plus raw numeric columns.

    Returns:
        ``(labels, columns)`` where ``labels[i]`` is ``(drug, reaction)`` and
        ``columns[name][i]`` is the raw upstream value.

    Raises:
        AnomalyDetectionError: If the file is missing, malformed or empty.
    """
    if not path.exists():
        raise AnomalyDetectionError(
            f"Signal table not found: {path}\n"
            "Run 'python -m src.analytics.signal_detection' first - this module "
            "consumes its real FAERS-derived output and never fabricates data."
        )

    labels: List[Tuple[str, str]] = []
    columns: Dict[str, List[float]] = {name: [] for name in _RAW_COLUMNS}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [c for c in ("drug", "reaction", *_RAW_COLUMNS) if c not in fields]
        if missing:
            raise AnomalyDetectionError(
                f"{path} is missing expected column(s): {', '.join(missing)}"
            )
        for row in reader:
            labels.append((row["drug"], row["reaction"]))
            for name in _RAW_COLUMNS:
                columns[name].append(_to_float(row[name]))

    if not labels:
        raise AnomalyDetectionError(f"Signal table has no data rows: {path}")

    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    return labels, arrays


def read_quarter_table(path: Path) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]:
    """Stream the per-quarter signal CSV into ``{pair: {quarter: (a, prr)}}``.

    A missing or malformed file yields an empty mapping; the quarter features
    then degrade to ``0.0`` rather than failing the whole run.
    """
    quarterly: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    if not path.exists():
        return quarterly
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"quarter", "drug", "reaction", "a", "prr"} <= fields:
            return quarterly
        for row in reader:
            key = (row["drug"], row["reaction"])
            quarterly.setdefault(key, {})[row["quarter"]] = (
                _to_float(row["a"]),
                _to_float(row["prr"]),
            )
    return quarterly


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_feature_matrix(
    labels: Sequence[Tuple[str, str]],
    raw: Mapping[str, np.ndarray],
    quarterly: Mapping[Tuple[str, str], Mapping[str, Tuple[float, float]]],
) -> Dict[str, np.ndarray]:
    """Derive the eleven model features from the raw aggregate columns."""
    size = len(labels)
    features: Dict[str, np.ndarray] = {}

    features["log10_pair_reports"] = _log10_shifted(raw["a"])
    features["log10_drug_reports"] = _log10_shifted(raw["n_drug"])
    features["log10_reaction_reports"] = _log10_shifted(raw["n_reac"])
    features["log10_prr"] = _log10_positive(raw["prr"])
    features["log10_ror"] = _log10_positive(raw["ror"])
    features["prr_ci_log_width"] = (
        _log10_positive(np.asarray(raw["prr_ci_high"], dtype=float))
        - _log10_positive(np.asarray(raw["prr_ci_low"], dtype=float))
    )
    features["log10_ror_ci_low"] = _log10_positive(raw["ror_ci_low"])
    features["log10_chi_square"] = _log10_shifted(raw["chi_square"])

    quarters_observed = np.zeros(size, dtype=float)
    balance = np.zeros(size, dtype=float)
    spread = np.zeros(size, dtype=float)
    for index, key in enumerate(labels):
        seen = quarterly.get(key)
        if not seen:
            continue
        quarters_observed[index] = float(len(seen))
        counts = np.asarray([value[0] for value in seen.values()], dtype=float)
        counts = counts[np.isfinite(counts) & (counts >= 0.0)]
        total = float(counts.sum()) if counts.size else 0.0
        if total > 0.0:
            balance[index] = (float(counts.max()) - float(counts.min())) / total
        if len(seen) >= 2:
            values = np.asarray([value[1] for value in seen.values()], dtype=float)
            values = values[np.isfinite(values) & (values > 0.0)]
            if values.size >= 2:
                spread[index] = math.log10(float(values.max())) - math.log10(float(values.min()))

    features["quarters_observed"] = quarters_observed
    features["quarter_report_balance"] = balance
    features["prr_quarter_spread"] = spread
    return features


def assemble_matrix(
    features: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, float], Dict[str, int]]:
    """Stack the features into one matrix and median-impute non-finite cells.

    Returns:
        ``(matrix, non_finite_by_feature, medians, imputed_by_feature)``. Every
        cell of the returned matrix is finite.
    """
    size = len(next(iter(features.values())))
    matrix = np.empty((size, len(FEATURE_NAMES)), dtype=float)
    non_finite_by_feature: Dict[str, int] = {}
    imputed_by_feature: Dict[str, int] = {}
    medians: Dict[str, float] = {}

    for position, name in enumerate(FEATURE_NAMES):
        column = matrix[:, position] = np.asarray(features[name], dtype=float)
        bad = ~np.isfinite(column)
        count = int(bad.sum())
        finite = column[~bad]
        median = float(np.median(finite)) if finite.size else 0.0
        if count:
            column[bad] = median
        non_finite_by_feature[name] = count
        imputed_by_feature[name] = count
        medians[name] = median

    if not np.isfinite(matrix).all():  # pragma: no cover - defensive
        raise AnomalyDetectionError(
            "Feature matrix still contains non-finite values after imputation."
        )
    return matrix, non_finite_by_feature, medians, imputed_by_feature


def _feature_stats(matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per-feature summary statistics used by the report."""
    stats: Dict[str, Dict[str, float]] = {}
    for position, name in enumerate(FEATURE_NAMES):
        column = matrix[:, position]
        stats[name] = {
            "min": float(column.min()),
            "median": float(np.median(column)),
            "mean": float(column.mean()),
            "max": float(column.max()),
        }
    return stats


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------
@dataclass
class AnomalyResults:
    """Everything the report, the CSV writers and the validation need."""

    generated_at: str = ""
    quarters: List[str] = field(default_factory=list)
    input_signals: str = ""
    input_quarter_signals: str = ""
    population_rows: int = 0
    quarter_rows: int = 0
    pairs_with_quarter_data: int = 0

    feature_names: Tuple[str, ...] = FEATURE_NAMES
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    non_finite_by_feature: Dict[str, int] = field(default_factory=dict)
    imputed_by_feature: Dict[str, int] = field(default_factory=dict)
    medians: Dict[str, float] = field(default_factory=dict)

    model_params: Dict[str, Any] = field(default_factory=dict)
    model_path: str = ""
    rows_scored: int = 0
    anomaly_count: int = 0
    anomaly_rate: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    score_mean: float = 0.0
    flagged_without_quarter_change: int = 0

    top_anomalies: List[Dict[str, Any]] = field(default_factory=list)

    reproducibility_max_diff: float = 0.0
    reload_max_diff: float = 0.0
    duplicate_keys: int = 0

    output_paths: Dict[str, str] = field(default_factory=dict)
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_summary(self) -> Dict[str, Any]:
        """Compact JSON summary - bounded, never the full scored table."""
        return {
            "generated_at": self.generated_at,
            "disclaimer": ANOMALY_DISCLAIMER,
            "quarters": self.quarters,
            "inputs": {
                "signals": self.input_signals,
                "signals_by_quarter": self.input_quarter_signals,
                "population_rows": self.population_rows,
                "quarter_rows": self.quarter_rows,
                "pairs_with_quarter_data": self.pairs_with_quarter_data,
            },
            "features": list(self.feature_names),
            "feature_stats": self.feature_stats,
            "non_finite_by_feature": self.non_finite_by_feature,
            "imputed_by_feature": self.imputed_by_feature,
            "model": self.model_params,
            "model_path": self.model_path,
            "scoring": {
                "rows_scored": self.rows_scored,
                "anomaly_count": self.anomaly_count,
                "anomaly_rate": self.anomaly_rate,
                "score_min": self.score_min,
                "score_max": self.score_max,
                "score_mean": self.score_mean,
                "duplicate_pair_keys": self.duplicate_keys,
            },
            "reproducibility": {
                "refit_max_abs_score_diff": self.reproducibility_max_diff,
                "reload_max_abs_score_diff": self.reload_max_diff,
            },
            "top_anomalies": self.top_anomalies[:TOP_ANOMALIES_SUMMARY],
            "outputs": self.output_paths,
            "checks": [
                {"name": name, "passed": ok, "detail": detail}
                for name, ok, detail in self.checks
            ],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Model fitting / scoring
# ---------------------------------------------------------------------------
def _make_model(params: Mapping[str, Any]) -> IsolationForest:
    return IsolationForest(
        n_estimators=int(params["n_estimators"]),
        max_samples=params["max_samples"],
        contamination=float(params["contamination"]),
        random_state=int(params["random_state"]),
        n_jobs=-1,
    )


def detect_anomalies(
    matrix: np.ndarray,
    *,
    params: Mapping[str, Any],
) -> Tuple[IsolationForest, np.ndarray, np.ndarray]:
    """Fit the Isolation Forest and return ``(model, anomaly_score, is_anomaly)``.

    ``anomaly_score`` is ``-decision_function`` so that **higher means more
    anomalous** (``decision_function`` is higher for inliers, which reads
    backwards on a chart). ``is_anomaly`` is true where ``predict`` returns -1.
    """
    model = _make_model(params)
    model.fit(matrix)
    scores = -model.decision_function(matrix)
    predicted = model.predict(matrix)
    return model, np.asarray(scores, dtype=float), np.asarray(predicted, dtype=int)


def _top_anomalies(
    labels: Sequence[Tuple[str, str]],
    raw: Mapping[str, np.ndarray],
    matrix: np.ndarray,
    scores: np.ndarray,
    predicted: np.ndarray,
    limit: int,
) -> List[Dict[str, Any]]:
    """Rank the flagged pairs by anomaly score, strongest first.

    Feature values come from the assembled matrix (the exact values the model
    consumed), so the JSON summary can never contain a non-finite number.
    """
    flagged = np.flatnonzero(predicted < 0)
    if flagged.size == 0:
        return []
    order = flagged[np.argsort(-scores[flagged], kind="stable")][:limit]
    records: List[Dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        i = int(index)
        drug, reaction = labels[i]
        record: Dict[str, Any] = {
            "rank": rank,
            "drug": drug,
            "reaction": reaction,
            "anomaly_score": float(scores[i]),
            "a": _to_int(raw["a"][i]),
            "n_drug": _to_int(raw["n_drug"][i]),
            "n_reac": _to_int(raw["n_reac"][i]),
            "prr": float(raw["prr"][i]),
            "ror": float(raw["ror"][i]),
            "ror_ci_low": float(raw["ror_ci_low"][i]),
            "chi_square": float(raw["chi_square"][i]),
        }
        for position, name in enumerate(FEATURE_NAMES):
            record[name] = float(matrix[i, position])
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Independent verification helper
# ---------------------------------------------------------------------------
def _recompute_feature_sample(
    labels: Sequence[Tuple[str, str]],
    raw: Mapping[str, np.ndarray],
    quarterly: Mapping[Tuple[str, str], Mapping[str, Tuple[float, float]]],
    matrix: np.ndarray,
    sample: Sequence[int],
) -> float:
    """Re-derive the features for sampled rows straight from the source values.

    This is an independent path: it re-reads the raw columns and the quarterly
    dictionary and recomputes every feature in plain Python instead of calling
    :func:`build_feature_matrix`, so a mistake in the vectorised path shows up
    as a non-zero difference. Cells the assembly step imputed with the median
    are skipped (a genuine ``NaN`` has no single expected value).
    """
    max_diff = 0.0
    for index in sample:
        a = float(raw["a"][index])
        n_drug = float(raw["n_drug"][index])
        n_reac = float(raw["n_reac"][index])
        prr = float(raw["prr"][index])
        ror = float(raw["ror"][index])
        chi = float(raw["chi_square"][index])
        ci_low = float(raw["prr_ci_low"][index])
        ci_high = float(raw["prr_ci_high"][index])
        ror_low = float(raw["ror_ci_low"][index])

        expected: List[float] = [
            math.log10(a + 1.0) if a >= -1 else float("nan"),
            math.log10(n_drug + 1.0) if n_drug >= -1 else float("nan"),
            math.log10(n_reac + 1.0) if n_reac >= -1 else float("nan"),
            math.log10(prr) if prr > 0 else float("nan"),
            math.log10(ror) if ror > 0 else float("nan"),
            (math.log10(ci_high) - math.log10(ci_low))
            if (ci_high > 0 and ci_low > 0)
            else float("nan"),
            math.log10(ror_low) if ror_low > 0 else float("nan"),
            math.log10(chi + 1.0) if chi >= -1 else float("nan"),
        ]

        seen = quarterly.get(labels[index], {})
        expected.append(float(len(seen)))
        counts = [v[0] for v in seen.values() if math.isfinite(v[0]) and v[0] >= 0.0]
        total = sum(counts)
        expected.append((max(counts) - min(counts)) / total if total > 0 else 0.0)
        values = [v[1] for v in seen.values() if math.isfinite(v[1]) and v[1] > 0.0]
        spread = 0.0
        if len(seen) >= 2 and len(values) >= 2:
            spread = math.log10(max(values)) - math.log10(min(values))
        expected.append(spread)

        for position, want in enumerate(expected):
            if math.isnan(want):
                continue
            max_diff = max(max_diff, abs(float(matrix[index, position]) - want))
    return max_diff


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class _Pipeline:
    """Internal carrier so the CSV writers and validators never re-read inputs."""

    results: AnomalyResults
    labels: List[Tuple[str, str]]
    raw: Dict[str, np.ndarray]
    quarterly: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]
    features: Dict[str, np.ndarray]
    matrix: np.ndarray
    scores: np.ndarray
    predicted: np.ndarray
    model: IsolationForest


def analyze(
    *,
    signals_path: Path,
    quarter_signals_path: Path,
    params: Mapping[str, Any],
) -> _Pipeline:
    """Load the real aggregate tables, engineer features, fit and score."""
    log = get_logger(__name__)
    results = AnomalyResults(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        quarters=[str(q) for q in get_quarters()],
        input_signals=str(signals_path),
        input_quarter_signals=str(quarter_signals_path),
    )

    labels, raw = read_signal_table(signals_path)
    quarterly = read_quarter_table(quarter_signals_path)
    results.population_rows = len(labels)
    results.quarter_rows = sum(len(v) for v in quarterly.values())
    results.pairs_with_quarter_data = sum(1 for key in labels if key in quarterly)
    results.duplicate_keys = len(labels) - len(set(labels))
    log.info(
        "population=%s pair(s), per-quarter rows=%s, pairs with quarterly data=%s",
        f"{results.population_rows:,}",
        f"{results.quarter_rows:,}",
        f"{results.pairs_with_quarter_data:,}",
    )

    features = build_feature_matrix(labels, raw, quarterly)
    matrix, non_finite, medians, imputed = assemble_matrix(features)
    results.non_finite_by_feature = non_finite
    results.imputed_by_feature = imputed
    results.medians = medians
    results.feature_stats = _feature_stats(matrix)
    log.info(
        "features=%s, non-finite cells=%s",
        len(FEATURE_NAMES),
        f"{sum(non_finite.values()):,}",
    )

    model, scores, predicted = detect_anomalies(matrix, params=params)
    results.model_params = dict(params)
    results.rows_scored = int(scores.size)
    results.anomaly_count = int((predicted < 0).sum())
    results.anomaly_rate = (
        results.anomaly_count / results.rows_scored if results.rows_scored else 0.0
    )
    results.score_min = float(scores.min())
    results.score_max = float(scores.max())
    results.score_mean = float(scores.mean())
    observed = np.asarray(features["quarters_observed"], dtype=float)
    results.flagged_without_quarter_change = int(
        np.sum((predicted < 0) & (observed < 2))
    )
    results.top_anomalies = _top_anomalies(
        labels, raw, matrix, scores, predicted, TOP_ANOMALIES_CSV
    )
    log.info(
        "scored %s pair(s); %s flagged as anomalies (%.4f)",
        f"{results.rows_scored:,}",
        f"{results.anomaly_count:,}",
        results.anomaly_rate,
    )

    return _Pipeline(
        results=results,
        labels=labels,
        raw=raw,
        quarterly=quarterly,
        features=features,
        matrix=matrix,
        scores=scores,
        predicted=predicted,
        model=model,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(pipeline: _Pipeline, *, params: Mapping[str, Any]) -> List[Tuple[str, bool, str]]:
    """Cross-check the whole pipeline; every check is recorded pass or fail."""
    results = pipeline.results
    matrix = pipeline.matrix
    labels = pipeline.labels
    raw = pipeline.raw
    quarterly = pipeline.quarterly
    features = pipeline.features
    scores = pipeline.scores
    predicted = pipeline.predicted
    model = pipeline.model
    log = get_logger(__name__)
    checks: List[Tuple[str, bool, str]] = []

    # 1. input features
    expected_shape = (results.population_rows, len(FEATURE_NAMES))
    checks.append(
        (
            "input features assembled with the expected shape and order",
            matrix.shape == expected_shape and tuple(features) == FEATURE_NAMES,
            f"matrix={matrix.shape} (expected {expected_shape}); "
            f"{len(FEATURE_NAMES)} feature(s) in the documented order",
        )
    )

    stride = max(matrix.shape[0] // 9, 1)
    sample_indices = list(range(0, matrix.shape[0], stride))[:10]
    sample_diff = _recompute_feature_sample(labels, raw, quarterly, matrix, sample_indices)
    checks.append(
        (
            "sampled features recomputed independently from the source values",
            sample_diff <= 1e-9,
            f"{len(sample_indices)} row(s) re-derived in plain Python; "
            f"max abs difference = {sample_diff:.3e}",
        )
    )

    # 2. missing / infinite values
    total_non_finite = int(sum(results.non_finite_by_feature.values()))
    finite_after = bool(np.isfinite(matrix).all())
    checks.append(
        (
            "missing and infinite values detected and median-imputed",
            finite_after
            and total_non_finite == sum(results.imputed_by_feature.values())
            and sum(results.imputed_by_feature.values()) == total_non_finite,
            f"{total_non_finite} non-finite cell(s) found across "
            f"{sum(1 for n in results.non_finite_by_feature.values() if n)} feature(s); "
            f"matrix fully finite afterwards = {finite_after}",
        )
    )

    # 3. model configuration
    ok_params = (
        model.n_estimators == int(params["n_estimators"])
        and float(model.contamination) == float(params["contamination"])
        and model.random_state == int(params["random_state"])
        and getattr(model, "n_features_in_", None) == len(FEATURE_NAMES)
    )
    checks.append(
        (
            "model trained with the configured, seeded hyper-parameters",
            ok_params,
            f"n_estimators={model.n_estimators}, contamination={model.contamination}, "
            f"max_samples={params['max_samples']}, random_state={model.random_state}, "
            f"n_features={getattr(model, 'n_features_in_', 'n/a')}",
        )
    )

    # 4. anomaly scores
    scores_ok = bool(np.isfinite(scores).all()) and scores.size == results.population_rows
    checks.append(
        (
            "anomaly scores generated for every pair and all finite",
            scores_ok,
            f"{scores.size:,} score(s); range [{scores.min():.4f}, {scores.max():.4f}], "
            f"mean {scores.mean():.4f}",
        )
    )

    # 5. anomalous records
    target = float(params["contamination"])
    rate_ok = abs(results.anomaly_rate - target) <= max(0.01, 0.1 * target)
    checks.append(
        (
            "anomalous records identified at the configured contamination rate",
            rate_ok and results.anomaly_count > 0,
            f"{results.anomaly_count:,} flagged ({results.anomaly_rate:.4%}) "
            f"vs contamination {target:.4%}",
        )
    )

    # 6. reproducibility
    try:
        _, scores_again, _ = detect_anomalies(matrix, params=params)
        refit_diff = float(np.max(np.abs(scores - scores_again))) if scores.size else 0.0
    except Exception as exc:  # pragma: no cover - defensive
        log.error("refit verification failed: %s", exc)
        results.errors.append(f"refit verification failed: {exc}")
        refit_diff = float("inf")
    results.reproducibility_max_diff = refit_diff
    checks.append(
        (
            "model is reproducible with a fixed random seed",
            refit_diff == 0.0,
            f"refit with random_state={params['random_state']}; "
            f"max abs score difference = {refit_diff:.3e}",
        )
    )

    # 7. save / reload
    try:
        target_path = Path(results.model_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target_path)
        reloaded = joblib.load(target_path)
        reload_scores = -reloaded.decision_function(matrix)
        reload_diff = float(np.max(np.abs(scores - reload_scores))) if scores.size else 0.0
        same_predictions = bool(np.array_equal(reloaded.predict(matrix), predicted))
        saved = target_path.exists() and target_path.stat().st_size > 0
    except Exception as exc:  # pragma: no cover - defensive
        log.error("model save/reload failed: %s", exc)
        results.errors.append(f"model save/reload failed: {exc}")
        reload_diff = float("inf")
        same_predictions = False
        saved = False
    results.reload_max_diff = reload_diff
    checks.append(
        (
            "model saved to disk and reloaded with identical results",
            saved and reload_diff == 0.0 and same_predictions,
            f"{results.model_path}; max abs score difference after reload = "
            f"{reload_diff:.3e}; identical predictions = {same_predictions}",
        )
    )

    # 8. key hygiene
    checks.append(
        (
            "no duplicate drug/reaction pair keys in the population",
            results.duplicate_keys == 0,
            f"{results.population_rows:,} pair(s), {results.duplicate_keys} duplicate key(s)",
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Output writers (Python standard library only)
# ---------------------------------------------------------------------------
_RESULT_COLUMNS: Tuple[str, ...] = (
    "anomaly_rank",
    "drug",
    "reaction",
    "a",
    "n_drug",
    "n_reac",
    "prr",
    "ror",
    "prr_ci_low",
    "prr_ci_high",
    "ror_ci_low",
    "chi_square",
    *FEATURE_NAMES,
    "anomaly_score",
    "is_anomaly",
)


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(pipeline: _Pipeline, *, output_dir: Path) -> Dict[str, Path]:
    """Write the scored table, the anomaly shortlist and the JSON summary."""
    results = pipeline.results
    labels = pipeline.labels
    raw = pipeline.raw
    matrix = pipeline.matrix
    scores = pipeline.scores
    predicted = pipeline.predicted
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    ranks = np.empty(scores.size, dtype=int)
    order = np.argsort(-scores, kind="stable")
    ranks[order] = np.arange(1, scores.size + 1)

    def row(index: int, rank: int) -> List[Any]:
        drug, reaction = labels[index]
        return [
            rank,
            drug,
            reaction,
            _to_int(raw["a"][index]),
            _to_int(raw["n_drug"][index]),
            _to_int(raw["n_reac"][index]),
            _fmt(raw["prr"][index]),
            _fmt(raw["ror"][index]),
            _fmt(raw["prr_ci_low"][index]),
            _fmt(raw["prr_ci_high"][index]),
            _fmt(raw["ror_ci_low"][index]),
            _fmt(raw["chi_square"][index]),
            *[_fmt(matrix[index, position], 8) for position in range(len(FEATURE_NAMES))],
            _fmt(scores[index], 8),
            bool(predicted[index] < 0),
        ]

    # Full scored table, strongest anomaly first.
    full_path = output_dir / "anomaly_results.csv"
    _write_csv(full_path, _RESULT_COLUMNS, (row(int(i), int(ranks[i])) for i in order))
    written["anomaly_results.csv"] = full_path

    # Short, human-reviewable shortlist of the strongest anomalies.
    flagged = np.flatnonzero(predicted < 0)
    top_order = flagged[np.argsort(-scores[flagged], kind="stable")][:TOP_ANOMALIES_CSV]
    shortlist_path = output_dir / "anomaly_top_pairs.csv"
    _write_csv(
        shortlist_path,
        _RESULT_COLUMNS,
        (row(int(i), rank) for rank, i in enumerate(top_order, start=1)),
    )
    written["anomaly_top_pairs.csv"] = shortlist_path

    summary_path = output_dir / "anomaly_summary.json"
    summary_path.write_text(
        json.dumps(results.as_summary(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    written["anomaly_summary.json"] = summary_path
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def render_report(results: AnomalyResults) -> str:
    """Render the anomaly-detection report as plain text."""
    params = results.model_params
    lines: List[str] = [
        "=" * 96,
        "FAERS ISOLATION FOREST ANOMALY DETECTION REPORT",
        f"Generated : {results.generated_at}",
        f"Quarters  : {', '.join(results.quarters)}",
        "Engine    : scikit-learn IsolationForest (numpy only; no pandas, no PyArrow, no Spark)",
        "Input     : the project's own PRR/ROR signal tables (real FAERS 2026 Q1/Q2 data)",
        "=" * 96,
        "",
        "0. SCOPE AND METHOD",
        "-" * 96,
        "  Each row of the input signal table is one screened drug/reaction pair with",
        "  its report-level 2x2 counts, PRR, ROR, 95% confidence intervals and",
        "  chi-square. Eleven numeric features are derived per pair, missing and",
        "  infinite values are median-imputed, and an Isolation Forest scores how",
        "  unlike the rest of the population each pair is.",
        "  Counts and statistics are log-transformed (variance stabilisation). No",
        "  standardisation is applied: an Isolation Forest splits each feature",
        "  independently, so it is invariant to per-feature monotone rescaling.",
        "",
        "  !! " + ANOMALY_DISCLAIMER,
        "",
        "1. INPUT POPULATION",
        "-" * 96,
        f"  Overall signal table      : {results.input_signals}",
        f"  Per-quarter signal table  : {results.input_quarter_signals}",
        f"  Pairs (rows) scored       : {results.population_rows:>14,}",
        f"  Per-quarter rows read     : {results.quarter_rows:>14,}",
        f"  Pairs with quarterly data : {results.pairs_with_quarter_data:>14,}",
        f"  Duplicate pair keys       : {results.duplicate_keys:>14,}",
        "",
        "  NOTE: the upstream signal table is capped at 200,000 rows (strongest pairs",
        "  by PRR), so this is anomaly detection WITHIN the screened signal population,",
        "  not across all screened pairs. Pairs the upstream screen discarded are not",
        "  visible here.",
        "",
        "2. FEATURES",
        "-" * 96,
        f"  {'FEATURE':<26}{'MIN':>12}{'MEDIAN':>12}{'MEAN':>12}{'MAX':>12}{'IMPUTED':>9}",
    ]
    for name in FEATURE_NAMES:
        stats = results.feature_stats.get(name, {})
        lines.append(
            f"  {name:<26}{stats.get('min', float('nan')):>12.4f}"
            f"{stats.get('median', float('nan')):>12.4f}"
            f"{stats.get('mean', float('nan')):>12.4f}"
            f"{stats.get('max', float('nan')):>12.4f}"
            f"{results.imputed_by_feature.get(name, 0):>9,}"
        )
    lines.append("")
    for name in FEATURE_NAMES:
        lines.append(f"    {name:<26} {FEATURE_DESCRIPTIONS[name]}")

    lines.extend(
        [
            "",
            "3. MISSING AND INFINITE VALUES",
            "-" * 96,
            "  Every cell is parsed defensively; a value that is not finite (or not",
            "  positive where a logarithm needs it) becomes NaN, is counted, and is",
            "  median-imputed from the finite values of the same feature.",
        ]
    )
    total_non_finite = sum(results.non_finite_by_feature.values())
    if total_non_finite:
        for name in FEATURE_NAMES:
            count = results.non_finite_by_feature.get(name, 0)
            if count:
                lines.append(
                    f"    {name:<26} {count:>8,} imputed with median "
                    f"{results.medians.get(name, 0.0):.6f}"
                )
    else:
        lines.append("    No non-finite value was found; no imputation was required.")

    lines.extend(
        [
            "",
            "4. MODEL CONFIGURATION (config.yaml -> analytics.anomalies)",
            "-" * 96,
            "  estimator              : IsolationForest (scikit-learn)",
            f"  n_estimators           : {params.get('n_estimators')}",
            f"  max_samples            : {params.get('max_samples')}",
            f"  contamination          : {params.get('contamination')}",
            f"  random_state           : {params.get('random_state')} (fixed for reproducibility)",
            f"  features               : {len(FEATURE_NAMES)}",
            "",
            "5. ANOMALY RESULTS",
            "-" * 96,
            f"  Pairs scored            : {results.rows_scored:>13,}",
            f"  Pairs flagged           : {results.anomaly_count:>13,}  ({results.anomaly_rate:.4%})",
            f"  Flagged, single quarter : {results.flagged_without_quarter_change:>13,}",
            f"  Anomaly score range     : {results.score_min:>13.5f} .. {results.score_max:.5f}",
            f"  Anomaly score mean      : {results.score_mean:>13.5f}",
            "",
            f"  TOP {TOP_ANOMALIES_REPORT} ANOMALOUS PAIRS (by anomaly score)",
            "-" * 96,
            f"  {'#':>3}  {'DRUG':<24}{'REACTION':<26}{'a':>7}{'PRR':>11}{'ROR':>12}{'CHI2':>11}{'SCORE':>9}",
        ]
    )
    for record in results.top_anomalies[:TOP_ANOMALIES_REPORT]:
        lines.append(
            f"  {record['rank']:>3}  {_clip(record['drug'], 22):<24}{_clip(record['reaction'], 24):<26}"
            f"{record['a']:>7,}{record['prr']:>11.3f}{record['ror']:>12.3f}"
            f"{record['chi_square']:>11.1f}{record['anomaly_score']:>9.4f}"
        )

    lines.extend(["", "6. VALIDATION CHECKS", "-" * 96])
    for name, ok, detail in results.checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        lines.append(f"         {detail}")

    lines.extend(["", "7. ISSUES AND ERRORS", "-" * 96])
    if results.errors:
        for error in results.errors:
            lines.append(f"  {error}")
    else:
        lines.append("  none")

    lines.extend(["", "8. OUTPUTS", "-" * 96])
    for name, path in results.output_paths.items():
        lines.append(f"  {name:<26} {path}")
    lines.append(f"  {'isolation_forest_model.pkl':<26} {results.model_path}")

    failed = [name for name, ok, _detail in results.checks if not ok]
    lines.extend(["", "9. SUMMARY", "-" * 96])
    lines.append(
        f"  {results.population_rows:,} pair(s), {len(FEATURE_NAMES)} feature(s), "
        f"{results.anomaly_count:,} flagged as anomalous ({results.anomaly_rate:.4%})."
    )
    if results.errors or failed:
        lines.append(
            f"  Status: INCOMPLETE - {len(results.errors)} error(s), {len(failed)} failed check(s)."
        )
    else:
        lines.append("  Status: COMPLETE - anomaly detection finished and all checks passed.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
_MAX_SAMPLES_KEYS = {"auto", "none"}


def _coerce_max_samples(value: Any) -> Any:
    """Accept ``'auto'``, an int, or a float in (0, 1] from the CLI."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _MAX_SAMPLES_KEYS:
        return "auto" if text.lower() == "auto" else None
    try:
        return int(text)
    except ValueError:
        return float(text)


def resolve_params(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge config.yaml ``analytics.anomalies`` with any non-null overrides."""
    params = get_analytics_config("anomalies")
    required = ("contamination", "n_estimators", "max_samples", "random_state")
    missing = [key for key in required if key not in params]
    if missing:
        raise AnomalyDetectionError(
            f"Missing anomaly parameter(s) in config.yaml: {', '.join(missing)}"
        )
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    params["max_samples"] = _coerce_max_samples(params["max_samples"])
    return params


def run(
    *,
    signals_path: Path,
    quarter_signals_path: Path,
    model_path: Path,
    output_dir: Optional[Path] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> AnomalyResults:
    """Load, fit, score, validate, persist model + tables, and return results."""
    log = get_logger(__name__)
    resolved = resolve_params(params or {})

    pipeline = analyze(
        signals_path=signals_path,
        quarter_signals_path=quarter_signals_path,
        params=resolved,
    )
    pipeline.results.model_path = str(model_path)
    pipeline.results.checks = _validate(pipeline, params=resolved)

    if output_dir is not None:
        written = write_outputs(pipeline, output_dir=output_dir)
        pipeline.results.output_paths = {name: str(path) for name, path in written.items()}
        for name, path in written.items():
            log.info("wrote %s -> %s", name, path)

    pipeline.results.generated_at = datetime.now().isoformat(timespec="seconds")
    return pipeline.results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Isolation Forest anomaly detection over the project's real "
        "FAERS-derived PRR/ROR signal tables."
    )
    parser.add_argument("--signals", default=None, help="Overall signal CSV (default: upstream output).")
    parser.add_argument(
        "--quarter-signals", default=None, help="Per-quarter signal CSV (default: upstream output)."
    )
    parser.add_argument(
        "--model", default=None, help="Model path (default models/isolation_forest_model.pkl)."
    )
    parser.add_argument("--outputs", default=None, help="Output directory ('' disables writing).")
    parser.add_argument("--report", default=None, help="Report path ('' disables writing).")
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--max-samples", default=None, help="'auto', an int, or a float in (0, 1].")
    args = parser.parse_args(argv)

    log = get_logger(__name__)

    try:
        signal_dir = get_path("analytics") / "signal_detection"
        signals_path = Path(args.signals) if args.signals else signal_dir / "signal_detection_signals.csv"
        quarter_path = (
            Path(args.quarter_signals)
            if args.quarter_signals
            else signal_dir / "signal_detection_signals_by_quarter.csv"
        )
        model_path = (
            Path(args.model)
            if args.model
            else get_path("models") / "isolation_forest_model.pkl"
        )
        if args.outputs is None:
            output_dir: Optional[Path] = get_path("analytics") / "anomaly_detection"
        else:
            output_dir = Path(args.outputs) if args.outputs else None
        if args.report is None:
            report_path: Optional[Path] = get_path("reports") / "anomaly_detection_report.txt"
        else:
            report_path = Path(args.report) if args.report else None
    except ConfigError as exc:  # pragma: no cover - defensive
        log.error("%s", exc)
        return 1

    overrides = {
        "contamination": args.contamination,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "max_samples": args.max_samples,
    }

    try:
        results = run(
            signals_path=signals_path,
            quarter_signals_path=quarter_path,
            model_path=model_path,
            output_dir=output_dir,
            params=overrides,
        )
    except (AnomalyDetectionError, ConfigError) as exc:
        log.error("Anomaly detection failed: %s", exc)
        return 2
    except MemoryError as exc:  # pragma: no cover - environment guard
        log.error("Anomaly detection ran out of memory: %s", exc)
        return 3

    report = render_report(results)
    print(report)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report + "\n", encoding="utf-8")
        log.info("Report written to: %s", report_path)

    if results.errors or any(not ok for _n, ok, _d in results.checks):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
