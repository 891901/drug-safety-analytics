"""Apriori association-rule mining over the real FDA FAERS 2026 extracts.

Reuses the existing ingestion and preprocessing modules to read the cleaned
DEMO and DRUG tables for the configured quarters, turns every FAERS report into
a *drug transaction* and mines frequent itemsets and association rules with a
Spark-native, level-wise Apriori implementation.

Why not ``mlxtend``
-------------------
``mlxtend.frequent_patterns.apriori`` expects a pandas DataFrame of one-hot
encoded transactions. On this Windows host the pandas / PyArrow native
libraries are blocked by an OS Application Control policy (``WinError 4551``),
and the whole point of the project is to avoid loading the full FAERS dataset
into pandas anyway. A Spark-native level-wise Apriori is therefore implemented
instead, which keeps processing bounded-memory and distributed.

What is produced
----------------
1. the transaction count and the item count (real FAERS reports, no synthesis);
2. frequent 1-, 2- and 3-itemsets (Apriori level-wise, anti-monotone pruning);
3. association rules ``A -> B`` derived from every frequent itemset;
4. support, confidence and lift for every rule, computed from integer report
   counts so they can be recomputed and verified independently;
5. a Q1 vs Q2 comparison (each quarter mined separately), never merged.

Design decisions
----------------
* **Transaction = one distinct ``(quarter, PRIMARYID)`` report with >= 1 drug.**
  Its items are the distinct ``DRUGNAME_STD`` values of that report.
* **Duplicate drug names inside a transaction are removed** by working on the
  distinct ``(quarter, PRIMARYID, DRUGNAME_STD)`` projection, so a report that
  repeats a drug name cannot inflate an itemset's support.
* **Level-wise Apriori with anti-monotone pruning.**
  ``L1`` = items whose report count meets ``min_support``; ``L2`` is counted by
  an equi self-join of the ``L1``-filtered transactions on ``(quarter,
  PRIMARYID)`` filtered to ``drug_a < drug_b``; ``L3`` is counted by a three-way
  join of the transactions restricted to the ``L2`` item universe filtered to
  ``drug_a < drug_b < drug_c``. A frequent itemset's every subset is frequent,
  so restricting the (k)-th pass to the ``L(k-1)`` item universe loses nothing.
* **No Cartesian join** - every join is an equi-join on the report key
  ``(quarter, PRIMARYID)``; the only broadcast look-ups are the small frequent
  item/pair universes.
* **Rules are derived from integer counts.**
  ``support = itemset_reports / total_transactions``,
  ``confidence = itemset_reports / antecedent_reports``,
  ``lift = confidence / support(consequent)``.
* **Q1 and Q2 are mined separately** and are never merged into one count.
* **A rule is a reporting association, not a drug interaction.** Two drugs
  appearing together says nothing about interaction, causality or comparative
  safety.

Environment constraints honoured: PySpark DataFrames only, no pandas, no
PyArrow, no UDFs, no Parquet. Only small aggregated result sets (frequent
itemsets and rules) are collected to the driver and written with the Python
standard library.

Usage::

    python -m src.analytics.apriori_analysis
    python -m src.analytics.apriori_analysis --quarters 2026_Q1
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.ingestion.load_faers import (
    QUARTER_COLUMN,
    IngestionError,
    load_settings,
)
from src.ingestion.spark_session import (
    SparkEnvironmentError,
    create_spark_session,
    stop_spark_session,
)
from src.preprocessing.clean_faers import CleaningError, TableCleaning, clean_table
from src.utils.config import get_analytics_config, get_path, get_quarter_dirs
from src.utils.logging_setup import get_logger

__all__ = [
    "AprioriError",
    "APRIORI_DISCLAIMER",
    "APRIORI_TABLES",
    "FrequentItemset",
    "AssociationRule",
    "AprioriResults",
    "analyze",
    "write_outputs",
    "render_report",
    "run",
    "main",
]

#: Tables this module needs (a subset of the seven FAERS tables).
APRIORI_TABLES: Tuple[str, ...] = ("DEMO", "DRUG")

#: The standardised drug-name column produced by the preprocessing module.
DRUG_NAME_COLUMN = "DRUGNAME_STD"

#: Grain statement reused in the report and the JSON summary.
APRIORI_GRAIN = (
    "Transaction = one distinct (quarter, PRIMARYID) report with at least one "
    "drug; its items are that report's distinct DRUGNAME_STD values. "
    "Frequent itemsets and rules are mined per transaction, never synthesised."
)

#: Mandatory caution; echoed in the report and the JSON summary.
APRIORI_DISCLAIMER = (
    "An association rule only says the drugs were reported together more often "
    "than independence predicts. It is NOT evidence of a drug-drug interaction, "
    "of a shared adverse event, of an underlying indication, or of causation."
)

# --- output sizing ---------------------------------------------------------
TOP_RULES_REPORT = 20
TOP_ITEMS_REPORT = 20
TOP_ITEMS_PER_SIZE_REPORT = 10
#: Safety cap on the number of rules written to CSV (they are already bounded by
#: the thresholds; the cap only guards against an unexpectedly large rule set).
MAX_RULES_WRITTEN = 200000
#: How many frequent itemsets are re-counted independently in Spark as a check.
RECOUNT_SAMPLE = 25


class AprioriError(RuntimeError):
    """Raised when the Apriori mining cannot be produced."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class FrequentItemset:
    """One frequent itemset (a drug combination) and its report count."""

    items: Tuple[str, ...]
    reports: int
    total_transactions: int
    quarter: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def support(self) -> float:
        """Reports containing the itemset / total transactions."""
        return self.reports / self.total_transactions if self.total_transactions else 0.0

    @property
    def label(self) -> str:
        return " + ".join(self.items)


@dataclass
class AssociationRule:
    """One association rule ``antecedent -> consequent``.

    Every measure is derived from integer report counts, so it can be
    recomputed exactly and verified independently of Spark's floating point.
    """

    antecedent: Tuple[str, ...]
    consequent: Tuple[str, ...]
    itemset_reports: int
    antecedent_reports: int
    consequent_reports: int
    total_transactions: int
    quarter: Optional[str] = None

    @property
    def support(self) -> float:
        """Reports containing the whole itemset / total transactions."""
        return self.itemset_reports / self.total_transactions if self.total_transactions else 0.0

    @property
    def confidence(self) -> float:
        """Reports containing the itemset / reports containing the antecedent."""
        return self.itemset_reports / self.antecedent_reports if self.antecedent_reports else 0.0

    @property
    def lift(self) -> float:
        """confidence / support of the consequent (1.0 = independence)."""
        if not (self.antecedent_reports and self.consequent_reports and self.total_transactions):
            return 0.0
        return (self.itemset_reports * self.total_transactions) / (
            self.antecedent_reports * self.consequent_reports
        )

    @property
    def antecedent_label(self) -> str:
        return " + ".join(self.antecedent)

    @property
    def consequent_label(self) -> str:
        return " + ".join(self.consequent)


@dataclass
class AprioriResults:
    """Every artefact produced by one Apriori run."""

    quarters: List[str] = field(default_factory=list)
    cleanings: Dict[str, TableCleaning] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)

    # section 1 - transactions
    total_reports: int = 0
    total_reports_by_quarter: Dict[str, int] = field(default_factory=dict)
    drug_rows: int = 0
    distinct_report_drug_rows: int = 0
    distinct_drug_names: int = 0
    reports_with_drugs: int = 0
    transactions: int = 0
    transactions_by_quarter: Dict[str, int] = field(default_factory=dict)

    # section 2 - frequent itemsets (size -> {items: report count})
    itemsets_overall: Dict[int, Dict[Tuple[str, ...], int]] = field(default_factory=dict)
    itemsets_by_quarter: Dict[str, Dict[int, Dict[Tuple[str, ...], int]]] = field(
        default_factory=dict
    )

    # section 3/4 - association rules
    rules_overall: List[AssociationRule] = field(default_factory=list)
    rules_scanned_overall: int = 0
    rules_by_quarter: List[AssociationRule] = field(default_factory=list)
    rules_scanned_by_quarter: Dict[str, int] = field(default_factory=dict)

    # section 5 - Q1 vs Q2
    shared_rules: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = field(default_factory=list)

    # validation
    duplicate_rules: int = 0
    identity_max_abs_diff: float = 0.0
    recount_sample_size: int = 0
    recount_max_abs_diff: int = 0
    checks: List[Tuple[str, bool, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # -- helpers ------------------------------------------------------------
    def itemset_count(self, size: int) -> int:
        return len(self.itemsets_overall.get(size, {}))

    def quarter_itemset_count(self, quarter: str, size: int) -> int:
        return len(self.itemsets_by_quarter.get(quarter, {}).get(size, {}))

    def rank_itemsets(self, size: int) -> List[FrequentItemset]:
        return [
            FrequentItemset(
                items=items, reports=count, total_transactions=self.transactions
            )
            for items, count in sorted(
                self.itemsets_overall.get(size, {}).items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

    def as_summary(self, generated_at: str) -> Dict[str, Any]:
        return {
            "generated_at": generated_at,
            "quarters": self.quarters,
            "disclaimer": APRIORI_DISCLAIMER,
            "grain": APRIORI_GRAIN,
            "thresholds": self.thresholds,
            "counts": {
                "total_reports": self.total_reports,
                "reports_with_drugs": self.reports_with_drugs,
                "transactions": self.transactions,
                "drug_rows": self.drug_rows,
                "distinct_report_drug_rows": self.distinct_report_drug_rows,
                "distinct_drug_names": self.distinct_drug_names,
                "frequent_1_itemsets": self.itemset_count(1),
                "frequent_2_itemsets": self.itemset_count(2),
                "frequent_3_itemsets": self.itemset_count(3),
                "rules_scanned": self.rules_scanned_overall,
                "rules_overall": len(self.rules_overall),
                "rules_q1_q2": len(self.rules_by_quarter),
                "shared_rules": len(self.shared_rules),
            },
            "total_reports_by_quarter": self.total_reports_by_quarter,
            "transactions_by_quarter": self.transactions_by_quarter,
            "frequent_itemsets_by_quarter": {
                quarter: {
                    str(size): len(mapping)
                    for size, mapping in sorted(sizes.items())
                }
                for quarter, sizes in sorted(self.itemsets_by_quarter.items())
            },
            "top_rules_by_lift": [
                self._rule_dict(rule) for rule in self.rules_overall[:TOP_RULES_REPORT]
            ],
            "checks": [
                {"name": name, "passed": ok, "detail": detail}
                for name, ok, detail in self.checks
            ],
            "errors": self.errors,
        }

    @staticmethod
    def _rule_dict(rule: AssociationRule) -> Dict[str, Any]:
        return {
            "antecedent": list(rule.antecedent),
            "consequent": list(rule.consequent),
            "antecedent_label": rule.antecedent_label,
            "consequent_label": rule.consequent_label,
            "itemset_reports": rule.itemset_reports,
            "antecedent_reports": rule.antecedent_reports,
            "consequent_reports": rule.consequent_reports,
            "support": rule.support,
            "confidence": rule.confidence,
            "lift": rule.lift,
            "quarter": rule.quarter,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _count_rows() -> "F.Column":
    """A portable row counter."""
    return F.count(F.lit(1))


def _clip(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` characters, adding an ellipsis when cut."""
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def _clean_inputs(
    spark: Any,
    *,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> Tuple[Dict[str, DataFrame], Dict[str, TableCleaning], List[str]]:
    """Read + clean the DEMO and DRUG tables via the preprocessing module."""
    log = get_logger(__name__)
    frames: Dict[str, DataFrame] = {}
    cleanings: Dict[str, TableCleaning] = {}
    errors: List[str] = []

    for table in APRIORI_TABLES:
        try:
            frame, cleaning = clean_table(spark, table, quarters, settings, quarter_dirs)
        except (CleaningError, IngestionError) as exc:
            log.error("%s: cleaning failed: %s", table, exc)
            errors.append(f"[{table}] {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - one bad table must not abort the run
            log.error("%s: unexpected failure: %s", table, exc)
            errors.append(f"[{table}] {type(exc).__name__}: {exc}")
            continue
        frames[table] = frame
        cleanings[table] = cleaning
        log.info("%s: %s rows cleaned", table, f"{cleaning.rows_after:,}")
    return frames, cleanings, errors


def _items_frame(pairs: DataFrame) -> DataFrame:
    """The distinct item universe appearing in a pair-count frame.

    Derived in Spark (never via ``createDataFrame`` from a Python list, which
    cannot be pickled on this Python 3.14 + PySpark build).
    """
    return (
        pairs.select(F.col("drug_a").alias(DRUG_NAME_COLUMN))
        .union(pairs.select(F.col("drug_b").alias(DRUG_NAME_COLUMN)))
        .distinct()
    )


# ---------------------------------------------------------------------------
# Apriori mining (Spark-native, level-wise)
# ---------------------------------------------------------------------------
def _mine_frequent_itemsets(
    txns: DataFrame,
    *,
    total_transactions: int,
    min_support: float,
    max_itemset_size: int,
) -> Dict[int, Dict[Tuple[str, ...], int]]:
    """Level-wise Apriori; returns ``{size: {itemset: report count}}``.

    ``txns`` must be the distinct ``(quarter, PRIMARYID, DRUGNAME_STD)``
    projection. Support counting is done by equi-joins on the report key
    ``(quarter, PRIMARYID)``, never by a Cartesian product, and the frequent
    item universes are derived in Spark so nothing large crosses to the driver.
    """
    log = get_logger(__name__)
    mined: Dict[int, Dict[Tuple[str, ...], int]] = {1: {}, 2: {}, 3: {}}
    if total_transactions <= 0:
        return mined

    threshold = float(min_support) * float(total_transactions)
    thr = F.lit(threshold)

    # --- L1: frequent single drugs -----------------------------------------
    item_counts = txns.groupBy(DRUG_NAME_COLUMN).agg(_count_rows().alias("reports")).persist()
    frequent_items = item_counts.filter(F.col("reports") >= thr)
    mined[1] = {
        (row[DRUG_NAME_COLUMN],): int(row["reports"]) for row in frequent_items.collect()
    }
    log.info("Apriori L1: %s frequent item(s) (min count %.2f)", f"{len(mined[1]):,}", threshold)
    if len(mined[1]) < 2:
        item_counts.unpersist()
        return mined

    # --- L2: frequent drug pairs -------------------------------------------
    # Restrict transactions to the frequent item universe (anti-monotone prune):
    # a frequent pair can only be built from two frequent items.
    l1_txns = txns.join(
        F.broadcast(frequent_items.select(DRUG_NAME_COLUMN)), on=DRUG_NAME_COLUMN, how="inner"
    )
    left = l1_txns.select(
        F.col(QUARTER_COLUMN).alias(QUARTER_COLUMN),
        F.col("PRIMARYID").alias("PRIMARYID"),
        F.col(DRUG_NAME_COLUMN).alias("drug_a"),
    )
    right = l1_txns.select(
        F.col(QUARTER_COLUMN).alias(QUARTER_COLUMN),
        F.col("PRIMARYID").alias("PRIMARYID"),
        F.col(DRUG_NAME_COLUMN).alias("drug_b"),
    )
    pair_counts = (
        left.join(right, on=[QUARTER_COLUMN, "PRIMARYID"])
        .filter(F.col("drug_a") < F.col("drug_b"))
        .groupBy("drug_a", "drug_b")
        .agg(_count_rows().alias("reports"))
        .persist()
    )
    frequent_pairs = pair_counts.filter(F.col("reports") >= thr)
    mined[2] = {
        (row["drug_a"], row["drug_b"]): int(row["reports"]) for row in frequent_pairs.collect()
    }
    log.info("Apriori L2: %s frequent pair(s)", f"{len(mined[2]):,}")
    if max_itemset_size < 3 or not mined[2]:
        mined[3] = {}
        pair_counts.unpersist()
        item_counts.unpersist()
        return mined

    # --- L3: frequent drug triples -----------------------------------------
    # Restrict to the L2 item universe (a frequent triple's every item appears
    # in a frequent pair).
    l2_txns = txns.join(
        F.broadcast(_items_frame(frequent_pairs)), on=DRUG_NAME_COLUMN, how="inner"
    )
    first = l2_txns.select(
        F.col(QUARTER_COLUMN), F.col("PRIMARYID"), F.col(DRUG_NAME_COLUMN).alias("drug_a")
    )
    second = l2_txns.select(
        F.col(QUARTER_COLUMN), F.col("PRIMARYID"), F.col(DRUG_NAME_COLUMN).alias("drug_b")
    )
    third = l2_txns.select(
        F.col(QUARTER_COLUMN), F.col("PRIMARYID"), F.col(DRUG_NAME_COLUMN).alias("drug_c")
    )
    triple_counts = (
        first.join(second, on=[QUARTER_COLUMN, "PRIMARYID"])
        .join(third, on=[QUARTER_COLUMN, "PRIMARYID"])
        .filter((F.col("drug_a") < F.col("drug_b")) & (F.col("drug_b") < F.col("drug_c")))
        .groupBy("drug_a", "drug_b", "drug_c")
        .agg(_count_rows().alias("reports"))
    )
    mined[3] = {
        (row["drug_a"], row["drug_b"], row["drug_c"]): int(row["reports"])
        for row in triple_counts.filter(F.col("reports") >= thr).collect()
    }
    log.info("Apriori L3: %s frequent triple(s)", f"{len(mined[3]):,}")
    pair_counts.unpersist()
    item_counts.unpersist()
    return mined


def _generate_rules(
    mined: Mapping[int, Mapping[Tuple[str, ...], int]],
    *,
    total_transactions: int,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    quarter: Optional[str] = None,
) -> Tuple[List[AssociationRule], int]:
    """Turn every frequent itemset (size >= 2) into candidate rules.

    A rule ``A -> C`` is emitted for every proper non-empty subset ``A`` of an
    itemset; ``C`` is the remainder. Support, confidence and lift are computed
    from the integer report counts and screened against the thresholds.
    """
    support_counts: Dict[Tuple[str, ...], int] = {}
    for mapping in mined.values():
        support_counts.update(mapping)

    rules: List[AssociationRule] = []
    scanned = 0
    for size in sorted(mined):
        if size < 2:
            continue
        for items, reports in mined[size].items():
            support = reports / total_transactions if total_transactions else 0.0
            if support < min_support:
                continue
            for cardinality in range(1, size):
                for antecedent in combinations(items, cardinality):
                    scanned += 1
                    consequent = tuple(item for item in items if item not in antecedent)
                    antecedent_reports = support_counts.get(antecedent)
                    consequent_reports = support_counts.get(consequent)
                    if not antecedent_reports or not consequent_reports:
                        continue
                    confidence = reports / antecedent_reports
                    lift = (reports * total_transactions) / (
                        antecedent_reports * consequent_reports
                    )
                    if confidence >= min_confidence and lift >= min_lift:
                        rules.append(
                            AssociationRule(
                                antecedent=tuple(antecedent),
                                consequent=consequent,
                                itemset_reports=reports,
                                antecedent_reports=antecedent_reports,
                                consequent_reports=consequent_reports,
                                total_transactions=total_transactions,
                                quarter=quarter,
                            )
                        )
    return rules, scanned


def _sort_rules(rules: Sequence[AssociationRule]) -> List[AssociationRule]:
    """Deterministic order: strongest lift first, then confidence and support."""
    return sorted(
        rules,
        key=lambda rule: (
            -rule.lift,
            -rule.confidence,
            -rule.support,
            rule.antecedent,
            rule.consequent,
        ),
    )


def _independent_recount(
    txns: DataFrame,
    sample: Sequence[Tuple[Tuple[str, ...], int]],
) -> int:
    """Recount sampled itemsets with a *different* Spark formulation.

    Instead of the k-way self-join used for mining, this filters the
    transactions to the itemset's items, groups by report and counts the
    distinct items seen: a report contains the whole itemset iff that distinct
    count equals the itemset size. Returns the largest absolute difference.
    """
    max_diff = 0
    for items, expected in sample:
        size = len(items)
        counted = (
            txns.filter(F.col(DRUG_NAME_COLUMN).isin(list(items)))
            .groupBy(QUARTER_COLUMN, "PRIMARYID")
            .agg(F.countDistinct(DRUG_NAME_COLUMN).alias("present"))
            .filter(F.col("present") == F.lit(size))
            .count()
        )
        max_diff = max(max_diff, abs(int(counted) - int(expected)))
    return max_diff


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(
    spark: Any,
    *,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
) -> AprioriResults:
    """Mine frequent itemsets and association rules from real FAERS reports."""
    log = get_logger(__name__)
    results = AprioriResults(quarters=list(quarters))

    thresholds = get_analytics_config("apriori")
    min_support = float(thresholds.get("min_support", 0.0) or 0.0)
    min_confidence = float(thresholds.get("min_confidence", 0.0) or 0.0)
    min_lift = float(thresholds.get("min_lift", 0.0) or 0.0)
    max_itemset = int(thresholds.get("max_itemset_size", 2) or 2)
    results.thresholds = {
        "min_support": min_support,
        "min_confidence": min_confidence,
        "min_lift": min_lift,
        "max_itemset_size": max_itemset,
    }

    frames, cleanings, errors = _clean_inputs(
        spark, quarters=quarters, settings=settings, quarter_dirs=quarter_dirs
    )
    results.cleanings = cleanings
    results.errors.extend(errors)

    missing = [table for table in APRIORI_TABLES if table not in frames]
    if missing:
        raise AprioriError(
            "required table(s) unavailable: " + ", ".join(missing) + "; see errors"
        )

    quarter = QUARTER_COLUMN
    reports = frames["DEMO"].select(quarter, "PRIMARYID").persist()
    # Transactions: one distinct drug per report -> duplicate drug names inside
    # a report are removed before any itemset is formed.
    txns = (
        frames["DRUG"]
        .select(quarter, "PRIMARYID", DRUG_NAME_COLUMN)
        .filter(F.col(DRUG_NAME_COLUMN).isNotNull())
        .distinct()
        .persist()
    )

    try:
        # ---- 1. transactions ----------------------------------------------
        results.drug_rows = cleanings["DRUG"].rows_after
        results.total_reports = reports.select("PRIMARYID").distinct().count()
        results.total_reports_by_quarter = {
            row[quarter]: int(row["reports"])
            for row in reports.groupBy(quarter).agg(_count_rows().alias("reports")).collect()
        }
        results.distinct_report_drug_rows = txns.count()
        results.distinct_drug_names = txns.select(DRUG_NAME_COLUMN).distinct().count()
        results.transactions = txns.select(quarter, "PRIMARYID").distinct().count()
        # Independent formulation of the same quantity: distinct reports with a
        # drug, counted straight off the cleaned DRUG table rather than the
        # de-duplicated (report, drug) projection.
        results.reports_with_drugs = (
            frames["DRUG"]
            .select(quarter, "PRIMARYID")
            .filter(F.col(DRUG_NAME_COLUMN).isNotNull())
            .distinct()
            .count()
        )
        results.transactions_by_quarter = {
            row[quarter]: int(row["transactions"])
            for row in txns.groupBy(quarter)
            .agg(F.countDistinct("PRIMARYID").alias("transactions"))
            .collect()
        }
        log.info(
            "transactions=%s  items=%s  distinct (report, drug) rows=%s",
            f"{results.transactions:,}",
            f"{results.distinct_drug_names:,}",
            f"{results.distinct_report_drug_rows:,}",
        )

        # ---- 2/3/4. Apriori: frequent itemsets + rules (overall) -----------
        mined_overall = _mine_frequent_itemsets(
            txns,
            total_transactions=results.transactions,
            min_support=min_support,
            max_itemset_size=max_itemset,
        )
        results.itemsets_overall = mined_overall
        rules_overall, scanned_overall = _generate_rules(
            mined_overall,
            total_transactions=results.transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
        )
        results.rules_overall = _sort_rules(rules_overall)
        results.rules_scanned_overall = scanned_overall
        log.info(
            "overall: %s frequent itemset(s), %s rule(s) after thresholds (%s scanned)",
            f"{sum(len(m) for m in mined_overall.values()):,}",
            f"{len(results.rules_overall):,}",
            f"{scanned_overall:,}",
        )

        # ---- 5. Q1 vs Q2 (mined separately) -------------------------------
        for label in quarters:
            quarter_transactions = results.transactions_by_quarter.get(label, 0)
            quarter_txns = txns.filter(F.col(quarter) == label)
            mined_quarter = _mine_frequent_itemsets(
                quarter_txns,
                total_transactions=quarter_transactions,
                min_support=min_support,
                max_itemset_size=max_itemset,
            )
            results.itemsets_by_quarter[label] = mined_quarter
            quarter_rules, quarter_scanned = _generate_rules(
                mined_quarter,
                total_transactions=quarter_transactions,
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift,
                quarter=label,
            )
            results.rules_by_quarter.extend(_sort_rules(quarter_rules))
            results.rules_scanned_by_quarter[label] = quarter_scanned
            log.info(
                "%s: %s frequent itemset(s), %s rule(s) after thresholds",
                label,
                f"{sum(len(m) for m in mined_quarter.values()):,}",
                f"{len(quarter_rules):,}",
            )

        # rules that survive in BOTH quarters (same antecedent -> consequent)
        per_quarter_rules: Dict[str, set] = {}
        for rule in results.rules_by_quarter:
            per_quarter_rules.setdefault(rule.quarter or "", set()).add(
                (rule.antecedent, rule.consequent)
            )
        if len(per_quarter_rules) >= 2:
            quarter_sets = list(per_quarter_rules.values())
            shared: set = set()
            for key in quarter_sets[0]:
                if all(key in other for other in quarter_sets[1:]):
                    shared.add(key)
            results.shared_rules = sorted(shared)

        # ---- validation support --------------------------------------------
        # Duplicate-rule check: (antecedent, consequent) must be unique.
        seen: set = set()
        duplicates = 0
        for rule in results.rules_overall:
            key = (rule.antecedent, rule.consequent)
            if key in seen:
                duplicates += 1
            seen.add(key)
        results.duplicate_rules = duplicates

        # Algebraic identity: lift == confidence / support(consequent).
        identity_diff = 0.0
        for rule in results.rules_overall:
            consequent_support = (
                rule.consequent_reports / rule.total_transactions
                if rule.total_transactions
                else 0.0
            )
            if consequent_support:
                identity_diff = max(
                    identity_diff, abs(rule.lift - rule.confidence / consequent_support)
                )
        results.identity_max_abs_diff = identity_diff

        # Independent Spark recount of a sample of frequent itemsets.
        sample: List[Tuple[Tuple[str, ...], int]] = []
        for size in (2, 3):
            for items, count in list(mined_overall.get(size, {}).items()):
                if len(sample) >= RECOUNT_SAMPLE:
                    break
                sample.append((items, count))
        results.recount_sample_size = len(sample)
        results.recount_max_abs_diff = (
            _independent_recount(txns, sample) if sample else 0
        )

        results.checks = _validate(results)
    finally:
        for frame in (reports, txns):
            frame.unpersist()

    return results


def _validate(results: AprioriResults) -> List[Tuple[str, bool, str]]:
    """Cross-check the mining; every check is recorded pass or fail."""
    checks: List[Tuple[str, bool, str]] = []
    min_support = results.thresholds.get("min_support", 0.0)
    min_confidence = results.thresholds.get("min_confidence", 0.0)
    min_lift = results.thresholds.get("min_lift", 0.0)
    min_count = min_support * results.transactions

    cleaning_ok = all(
        item.rows_before == item.rows_after and not item.errors
        for item in results.cleanings.values()
    )
    checks.append(
        (
            "cleaning removed no rows (DEMO/DRUG)",
            cleaning_ok,
            "; ".join(
                f"{table}={item.rows_before:,}->{item.rows_after:,}"
                for table, item in results.cleanings.items()
            ),
        )
    )

    checks.append(
        (
            "duplicate drug names within a transaction removed",
            results.distinct_report_drug_rows < results.drug_rows
            and results.distinct_report_drug_rows > 0,
            f"{results.drug_rows:,} DRUG rows reduced to "
            f"{results.distinct_report_drug_rows:,} distinct (report, drug) rows",
        )
    )

    checks.append(
        (
            "transactions equal the distinct (quarter, PRIMARYID) reports with a drug",
            results.transactions == results.reports_with_drugs and results.transactions > 0,
            f"{results.transactions:,} transaction(s) vs "
            f"{results.reports_with_drugs:,} distinct report(s) carrying a drug",
        )
    )

    checks.append(
        (
            "transactions are listed once per quarter and sum to the overall count",
            sum(results.transactions_by_quarter.values()) == results.transactions
            and set(results.transactions_by_quarter) == set(results.quarters),
            "; ".join(
                f"{q}={results.transactions_by_quarter.get(q, 0):,}"
                for q in results.quarters
            )
            + f" (sum {sum(results.transactions_by_quarter.values()):,})",
        )
    )

    # ---- frequent itemsets -------------------------------------------------
    l1 = results.itemsets_overall.get(1, {})
    l2 = results.itemsets_overall.get(2, {})
    l3 = results.itemsets_overall.get(3, {})
    checks.append(
        (
            "every frequent 1-itemset meets the minimum support",
            bool(l1) and all(count >= min_count for count in l1.values()),
            f"{len(l1):,} frequent item(s); minimum report count = {min_count:,.2f}",
        )
    )

    bounded = all(
        min_count <= count <= results.transactions
        for mapping in results.itemsets_overall.values()
        for count in mapping.values()
    )
    checks.append(
        (
            "frequent itemset counts lie within [min count, total transactions]",
            bounded,
            f"checked {sum(len(m) for m in results.itemsets_overall.values()):,} itemset(s)",
        )
    )

    # Apriori anti-monotonicity: every subset of a frequent itemset is frequent.
    anti_monotone = True
    for items in l3:
        for pair in combinations(items, 2):
            if pair not in l2:
                anti_monotone = False
                break
    for items in list(l2) + list(l3):
        for item in items:
            if (item,) not in l1:
                anti_monotone = False
                break
    checks.append(
        (
            "Apriori anti-monotonicity: every subset of a frequent itemset is frequent",
            anti_monotone,
            f"all 2-subsets of {len(l3):,} triple(s) are frequent pairs; all items are frequent",
        )
    )

    # ---- association rules -------------------------------------------------
    checks.append(
        (
            "no duplicate association rules",
            results.duplicate_rules == 0,
            f"{len(results.rules_overall):,} rule(s), {results.duplicate_rules} duplicate(s)",
        )
    )

    within_thresholds = all(
        rule.support >= min_support
        and rule.confidence >= min_confidence
        and rule.lift >= min_lift
        for rule in results.rules_overall
    )
    checks.append(
        (
            "every rule satisfies the configured support, confidence and lift thresholds",
            within_thresholds,
            f"support >= {min_support}, confidence >= {min_confidence}, lift >= {min_lift}",
        )
    )

    not_above_antecedent = all(
        rule.itemset_reports <= rule.antecedent_reports
        and rule.itemset_reports <= rule.consequent_reports
        and rule.itemset_reports <= rule.total_transactions
        for rule in results.rules_overall
    )
    checks.append(
        (
            "rule support never exceeds its antecedent or consequent support",
            not_above_antecedent,
            f"checked {len(results.rules_overall):,} rule(s)",
        )
    )

    checks.append(
        (
            "confidence and lift identities recomputed from integer counts",
            results.identity_max_abs_diff <= 1e-9,
            f"max |lift - confidence / support(consequent)| = "
            f"{results.identity_max_abs_diff:.3e}",
        )
    )

    checks.append(
        (
            "sampled frequent itemsets recounted independently in Spark",
            results.recount_max_abs_diff == 0 and results.recount_sample_size > 0,
            f"{results.recount_sample_size} item(s) recount-sampled, "
            f"max |Spark - mining| difference = {results.recount_max_abs_diff}",
        )
    )

    checks.append(
        (
            "Q1 and Q2 are both mined separately and are distinguishable",
            set(results.itemsets_by_quarter) == set(results.quarters)
            and all(results.quarter_itemset_count(q, 1) >= 0 for q in results.quarters),
            ", ".join(
                f"{q}: {results.quarter_itemset_count(q, 1):,} items, "
                f"{results.quarter_itemset_count(q, 2):,} pairs, "
                f"{results.quarter_itemset_count(q, 3):,} triples"
                for q in results.quarters
            ),
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Output writers (Python standard library only)
# ---------------------------------------------------------------------------
def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(
    results: AprioriResults,
    *,
    output_dir: Path,
    generated_at: Optional[datetime] = None,
) -> Dict[str, Path]:
    """Persist frequent itemsets, rules and the summary; return the paths."""
    timestamp = (generated_at or datetime.now()).isoformat(timespec="seconds")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    def emit(name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        target = output_dir / name
        _write_csv(target, header, rows)
        written[name] = target

    # frequent itemsets (overall), every size, largest support first
    itemset_rows: List[List[Any]] = []
    for size in sorted(results.itemsets_overall):
        for items, count in sorted(
            results.itemsets_overall[size].items(), key=lambda item: (-item[1], item[0])
        ):
            itemset_rows.append(
                [
                    " + ".join(items),
                    size,
                    count,
                    round(count / results.transactions, 10) if results.transactions else 0.0,
                ]
            )
    emit(
        "apriori_frequent_itemsets.csv",
        ["itemset", "itemset_size", "reports", "support"],
        itemset_rows,
    )

    # association rules (overall)
    rule_header = [
        "antecedent",
        "consequent",
        "antecedent_size",
        "consequent_size",
        "itemset_reports",
        "antecedent_reports",
        "consequent_reports",
        "support",
        "confidence",
        "lift",
    ]

    def rule_row(rule: AssociationRule) -> List[Any]:
        return [
            rule.antecedent_label,
            rule.consequent_label,
            len(rule.antecedent),
            len(rule.consequent),
            rule.itemset_reports,
            rule.antecedent_reports,
            rule.consequent_reports,
            round(rule.support, 10),
            round(rule.confidence, 10),
            round(rule.lift, 10),
        ]

    emit(
        "apriori_rules.csv",
        rule_header,
        [rule_row(rule) for rule in results.rules_overall[:MAX_RULES_WRITTEN]],
    )
    emit(
        "apriori_rules_by_quarter.csv",
        ["quarter", *rule_header],
        [
            [rule.quarter, *rule_row(rule)]
            for rule in results.rules_by_quarter[:MAX_RULES_WRITTEN]
        ],
    )

    summary_path = output_dir / "apriori_summary.json"
    summary_path.write_text(
        json.dumps(results.as_summary(timestamp), indent=2) + "\n", encoding="utf-8"
    )
    written["apriori_summary.json"] = summary_path
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(
    results: AprioriResults,
    *,
    generated_at: Optional[datetime] = None,
) -> str:
    """Render the Apriori report as plain text."""
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    thresholds = results.thresholds
    min_count = results.thresholds.get("min_support", 0.0) * results.transactions

    lines: List[str] = [
        "=" * 90,
        "FAERS APRIORI ASSOCIATION-RULE MINING REPORT",
        f"Generated : {stamp}",
        f"Quarters  : {', '.join(results.quarters)}",
        "Engine    : PySpark DataFrame API only (no pandas, no PyArrow, no UDFs)",
        "Algorithm : Spark-native level-wise Apriori (mlxtend needs pandas, blocked)",
        "Source    : real FAERS 2026 Q1/Q2 ASCII extracts, via ingestion + preprocessing",
        "=" * 90,
        "",
        "0. SCOPE AND METHOD",
        "-" * 90,
        f"  {APRIORI_GRAIN}",
        "  Transactions are identified by (quarter, PRIMARYID); each transaction's",
        "  items are de-duplicated on (quarter, PRIMARYID, DRUGNAME_STD) first, so a",
        "  repeated drug name cannot inflate an itemset's support.",
        "  Mining is level-wise Apriori: L1 = frequent items, L2 = frequent pairs,",
        "  L3 = frequent triples, each pass restricted to the previous pass's item",
        "  universe (anti-monotone pruning). Support is counted by equi-joins on the",
        "  report key - no Cartesian product and no synthetic transactions.",
        "  Q1 and Q2 are mined separately and are never merged into one count.",
        "",
        "  !! " + APRIORI_DISCLAIMER,
        "",
        "  Measures (per rule A -> C, always from integer report counts):",
        "    support     = reports containing the whole itemset / total transactions",
        "    confidence  = reports containing the whole itemset / reports containing A",
        "    lift        = confidence / support of C",
        "                  (1.0 = A and C are independent)",
        "",
        "1. TRANSACTIONS",
        "-" * 90,
        f"  Distinct reports (DEMO, quarter + PRIMARYID) : {results.total_reports:>12,}",
        f"  Reports carrying at least one drug          : {results.reports_with_drugs:>12,}",
        f"  DRUG rows after cleaning                    : {results.drug_rows:>12,}",
        f"  Distinct (report, drug) rows                : {results.distinct_report_drug_rows:>12,}",
        f"  Distinct drug names (items)                 : {results.distinct_drug_names:>12,}",
        f"  Transactions used                           : {results.transactions:>12,}",
        "",
        "  Transactions by quarter:",
    ]
    for label in results.quarters:
        lines.append(
            f"    {label}: {results.transactions_by_quarter.get(label, 0):>12,} transaction(s) "
            f"over {results.total_reports_by_quarter.get(label, 0):,} report(s)"
        )

    lines.append("")
    lines.append("2. FREQUENT ITEMSETS (Apriori)")
    lines.append("-" * 90)
    lines.append(
        f"  minimum support = {thresholds.get('min_support', 0.0)} "
        f"-> minimum report count = {min_count:,.2f}"
    )
    lines.append(
        f"  {'SIZE':>5}  {'ITEMSETS':>12}  {'EXAMPLE':<60}"
    )
    for size in sorted(results.itemsets_overall):
        ranked = results.rank_itemsets(size)
        example = ranked[0].label if ranked else "-"
        lines.append(f"  {size:>5}  {len(ranked):>12,}  {_clip(example, 58):<60}")
    lines.append("")
    for size in sorted(results.itemsets_overall):
        ranked = results.rank_itemsets(size)[:TOP_ITEMS_PER_SIZE_REPORT]
        if not ranked:
            continue
        lines.append(f"  Top {len(ranked)} frequent {size}-itemset(s):")
        for index, itemset in enumerate(ranked, start=1):
            lines.append(
                f"    {index:>3}  {_clip(itemset.label, 60):<62}"
                f"{itemset.reports:>9,}{itemset.support * 100:>9.3f}%"
            )

    lines.append("")
    lines.append(f"3. TOP {TOP_RULES_REPORT} ASSOCIATION RULES (overall, by lift)")
    lines.append("-" * 90)
    lines.append(
        f"  {'#':>3}  {'ANTECEDENT':<30}{'CONSEQUENT':<30}"
        f"{'SUPPORT':>9}{'CONF':>8}{'LIFT':>9}"
    )
    for index, rule in enumerate(results.rules_overall[:TOP_RULES_REPORT], start=1):
        lines.append(
            f"  {index:>3}  {_clip(rule.antecedent_label, 28):<30}"
            f"{_clip(rule.consequent_label, 28):<30}"
            f"{rule.support * 100:>8.3f}%{rule.confidence:>8.4f}{rule.lift:>9.3f}"
        )
    lines.append("  support is a percentage of all reports; lift > 1 means the drugs were")
    lines.append("  reported together more often than independence predicts - a reporting")
    lines.append("  pattern, not a pharmacological interaction.")

    lines.append("")
    lines.append("4. Q1 vs Q2 COMPARISON (each quarter mined separately)")
    lines.append("-" * 90)
    for label in results.quarters:
        subset = [rule for rule in results.rules_by_quarter if rule.quarter == label][
            :TOP_RULES_REPORT
        ]
        lines.append(
            f"  {label}: {results.quarter_itemset_count(label, 1):,} item(s), "
            f"{results.quarter_itemset_count(label, 2):,} pair(s), "
            f"{results.quarter_itemset_count(label, 3):,} triple(s), "
            f"{len([r for r in results.rules_by_quarter if r.quarter == label]):,} rule(s)"
        )
        for index, rule in enumerate(subset, start=1):
            lines.append(
                f"    {index:>3}  {_clip(rule.antecedent_label, 28):<30}"
                f"{_clip(rule.consequent_label, 28):<30}"
                f"{rule.support * 100:>8.3f}%{rule.confidence:>8.4f}{rule.lift:>9.3f}"
            )
    if results.shared_rules:
        lines.append("")
        lines.append(
            f"  {len(results.shared_rules)} rule(s) survive the thresholds in BOTH quarters:"
        )
        for antecedent, consequent in results.shared_rules[:TOP_RULES_REPORT]:
            lines.append(f"    {' + '.join(antecedent)} -> {' + '.join(consequent)}")
    else:
        lines.append("")
        lines.append("  no rule survives the thresholds in both quarters.")

    lines.append("")
    lines.append("5. THRESHOLD SCREENING (config.yaml -> analytics.apriori)")
    lines.append("-" * 90)
    lines.append(
        f"  thresholds: support >= {thresholds.get('min_support', 0.0)}  "
        f"confidence >= {thresholds.get('min_confidence', 0.0)}  "
        f"lift >= {thresholds.get('min_lift', 0.0)}  "
        f"max_itemset_size = {thresholds.get('max_itemset_size', 0)}"
    )
    lines.append(
        f"  {results.rules_scanned_overall:,} candidate rule(s) were evaluated from the "
        f"frequent itemsets;"
    )
    lines.append(
        f"  {len(results.rules_overall):,} satisfy all three thresholds and are reported."
    )
    lines.append(
        "  Passing these thresholds only marks rules worth reviewing: they are"
    )
    lines.append(
        "  association-rule settings, not a test of interaction or causation."
    )

    lines.append("")
    lines.append("6. VALIDATION CHECKS")
    lines.append("-" * 90)
    for name, ok, detail in results.checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        lines.append(f"         {detail}")

    lines.append("")
    lines.append("7. ISSUES AND ERRORS")
    lines.append("-" * 90)
    if results.errors:
        for error in results.errors:
            lines.append(f"  {error}")
    else:
        lines.append("  none")

    failed = [name for name, ok, _detail in results.checks if not ok]
    lines.append("")
    lines.append("8. SUMMARY")
    lines.append("-" * 90)
    lines.append(
        f"  {len(results.quarters)} quarter(s), {len(results.cleanings)} table(s); "
        f"{results.transactions:,} transactions, {results.distinct_drug_names:,} items, "
        f"{results.itemset_count(1):,}/{results.itemset_count(2):,}/{results.itemset_count(3):,} "
        f"frequent 1/2/3-itemsets, {len(results.rules_overall):,} association rule(s)."
    )
    if results.errors or failed:
        lines.append(
            f"  Status: INCOMPLETE - {len(results.errors)} error(s), "
            f"{len(failed)} failed check(s)."
        )
    else:
        lines.append("  Status: COMPLETE - mining finished and all checks passed.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(
    spark: Any,
    *,
    quarters: Sequence[str],
    settings: Mapping[str, Any],
    quarter_dirs: Mapping[str, Path],
    output_dir: Optional[Path] = None,
) -> AprioriResults:
    """Mine, persist the outputs (when a directory is given) and return results."""
    log = get_logger(__name__)
    results = analyze(spark, quarters=quarters, settings=settings, quarter_dirs=quarter_dirs)
    if output_dir is not None:
        written = write_outputs(results, output_dir=output_dir)
        for name, path in written.items():
            log.info("wrote %s -> %s", name, path)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apriori association-rule mining over the real FAERS extracts "
        "(read-only; no Parquet)."
    )
    parser.add_argument("--quarters", nargs="*", default=None, help="e.g. 2026_Q1 2026_Q2")
    parser.add_argument(
        "--report",
        default=None,
        help="Report path (default reports/apriori_report.txt; '' disables it).",
    )
    parser.add_argument(
        "--outputs",
        default=None,
        help="Result directory (default data/analytics/apriori; '' disables it).",
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

    quarter_dirs = dict(get_quarter_dirs())
    if args.outputs is None:
        output_dir: Optional[Path] = get_path("analytics") / "apriori"
    else:
        output_dir = Path(args.outputs) if args.outputs else None

    try:
        spark = create_spark_session(app_name="DrugSafetyApriori")
    except SparkEnvironmentError as exc:
        log.error("%s", exc)
        return 1

    try:
        results = run(
            spark,
            quarters=quarters,
            settings=settings,
            quarter_dirs=quarter_dirs,
            output_dir=output_dir,
        )
    except (AprioriError, CleaningError, IngestionError) as exc:
        log.error("Apriori mining failed: %s", exc)
        return 2
    finally:
        stop_spark_session(spark)

    report = render_report(results)
    print(report)

    report_path = args.report
    if report_path is None:
        report_path = str(get_path("reports") / "apriori_report.txt")
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report + "\n", encoding="utf-8")
        log.info("Report written to: %s", target)

    if results.errors or any(not ok for _n, ok, _d in results.checks):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
