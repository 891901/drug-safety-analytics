"""Headless validation for the Streamlit dashboard.

This script never renders a browser UI.  It

1. inventories every registered analytical output (present / missing),
2. reconciles the headline totals *displayed by the dashboard* against the
   real source CSV/JSON files (read independently of the dashboard's own
   readers, so the check is not circular),
3. smoke-tests the bounded data readers and the Plotly chart builders,
4. imports every section module, and
5. runs the real app headlessly with ``streamlit.testing.v1.AppTest`` and
   switches through all ten sections, recording any runtime exception.

The result is written to ``reports/dashboard_validation_report.txt``.

Run with::

    python dashboard/validate_dashboard.py
"""

from __future__ import annotations

import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_PATH = ROOT / "reports" / "dashboard_validation_report.txt"

# ---------------------------------------------------------------------------
# Tiny result collector
# ---------------------------------------------------------------------------
RESULTS: List[Dict[str, str]] = []


def record(category: str, name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"category": category, "name": name, "status": status, "detail": detail})


def check(category: str, name: str, condition: bool, detail: str = "") -> bool:
    record(category, name, "PASS" if condition else "FAIL", detail)
    return bool(condition)


# ---------------------------------------------------------------------------
# Independent (non-dashboard) file readers - used for reconciliation
# ---------------------------------------------------------------------------
def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def iter_rows(path: Path) -> Iterator[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
    except OSError:
        return


def count_rows(path: Path) -> int:
    return sum(1 for _ in iter_rows(path))


def count_where(path: Path, column: str, value: str) -> int:
    target = value.strip().lower()
    return sum(1 for row in iter_rows(path) if str(row.get(column, "")).strip().lower() == target)


# ---------------------------------------------------------------------------
# Part 1 - output inventory
# ---------------------------------------------------------------------------
def validate_inventory() -> None:
    from dashboard.components import data as ds

    status = ds.output_status()
    present = [s for s in status if s["exists"]]
    missing = [s for s in status if not s["exists"]]
    check(
        "1. Output inventory",
        f"Registered outputs present ({len(present)}/{len(status)})",
        not missing,
        "" if not missing else "missing: " + ", ".join(m["key"] for m in missing),
    )
    # The dashboard must be able to name the location of a missing file.
    if missing:
        sample = missing[0]
        check(
            "1. Output inventory",
            "Missing output exposes its expected path",
            bool(sample["path"]),
            sample["path"],
        )


# ---------------------------------------------------------------------------
# Part 2 - totals reconciliation (dashboard values vs source files)
# ---------------------------------------------------------------------------
def validate_totals() -> None:
    from dashboard.components import data as ds

    adr = read_json(ds.SOURCES["adr_summary"]) or {}
    sig = read_json(ds.SOURCES["signal_summary"]) or {}
    anom = read_json(ds.SOURCES["anomaly_summary"]) or {}
    cooc = read_json(ds.SOURCES["cooccurrence_summary"]) or {}
    temporal = read_json(ds.SOURCES["temporal_summary"]) or {}

    adr_counts = adr.get("counts", {})
    sig_counts = sig.get("counts", {})
    anom_scoring = anom.get("scoring", {})
    cooc_counts = cooc.get("counts", {})
    temporal_block = temporal.get("temporal", {})

    # --- total reports -----------------------------------------------------
    total_reports = adr_counts.get("total_reports")
    temporal_q = temporal_block.get("reports_by_quarter", {})
    temporal_total = sum(int(v) for v in temporal_q.values()) if temporal_q else None
    sig_q = sig.get("total_reports_by_quarter", {})
    sig_total = sum(int(v) for v in sig_q.values()) if sig_q else None
    cooc_total = cooc_counts.get("total_reports")

    detail = (
        f"adr_summary={total_reports:,} | temporal sum={temporal_total:,} | "
        f"signal sum={sig_total:,} | cooccurrence={cooc_total:,}"
    ) if total_reports else "adr_summary missing total_reports"
    check(
        "2. Totals reconciliation",
        "Total reports agree across summaries",
        None not in (total_reports, temporal_total, sig_total, cooc_total)
        and len({total_reports, temporal_total, sig_total, cooc_total}) == 1,
        detail,
    )

    # --- Q1 / Q2 split -----------------------------------------------------
    q1, q2 = ds.QUARTERS[0], ds.QUARTERS[1]
    q_split = (
        f"{q1}={temporal_q.get(q1):,}, {q2}={temporal_q.get(q2):,}, "
        f"sum={temporal_total:,}"
    ) if temporal_q else "temporal reports_by_quarter missing"
    check(
        "2. Totals reconciliation",
        "Q1 + Q2 equals the total report count",
        bool(temporal_q) and temporal_total == total_reports,
        q_split,
    )

    # --- report-reaction pairs --------------------------------------------
    adr_pairs = adr_counts.get("distinct_report_reaction_pairs")
    sig_reac_rows = sig_counts.get("distinct_report_reaction_rows")
    check(
        "2. Totals reconciliation",
        "Report-reaction pairs agree (ADR vs signal)",
        adr_pairs == sig_reac_rows,
        f"adr={adr_pairs:,} | signal={sig_reac_rows:,}",
    )

    # --- serious reports ---------------------------------------------------
    adr_serious = adr_counts.get("reports_with_at_least_one_serious_outcome")
    serious_q = temporal_block.get("serious_reports_by_quarter", {})
    serious_total = sum(int(v) for v in serious_q.values()) if serious_q else None
    check(
        "2. Totals reconciliation",
        "Serious-report total agrees (ADR vs temporal)",
        adr_serious == serious_total,
        f"adr={adr_serious:,} | temporal sum={serious_total:,}",
    )

    # --- signals written vs the actual CSV --------------------------------
    signals_written = sig_counts.get("signals_written")
    signals_total = sig_counts.get("signals_total")
    sig_rows = count_rows(ds.SOURCES["signal_signals"])
    check(
        "2. Totals reconciliation",
        "Signal table row count equals signals_written",
        signals_written == sig_rows,
        f"summary signals_written={signals_written:,} | csv rows={sig_rows:,}",
    )
    check(
        "2. Totals reconciliation",
        "Hand-check: signals_total >= signals_written (capped output)",
        isinstance(signals_total, int) and isinstance(signals_written, int)
        and signals_total >= signals_written,
        f"signals_total={signals_total:,} >= signals_written={signals_written:,}",
    )

    # --- anomaly rows scored / flagged ------------------------------------
    rows_scored = anom_scoring.get("rows_scored")
    anomaly_count = anom_scoring.get("anomaly_count")
    anom_rows = count_rows(ds.SOURCES["anomaly_results"])
    flagged = count_where(ds.SOURCES["anomaly_results"], "is_anomaly", "True")
    check(
        "2. Totals reconciliation",
        "Anomaly result row count equals rows_scored",
        rows_scored == anom_rows,
        f"summary rows_scored={rows_scored:,} | csv rows={anom_rows:,}",
    )
    check(
        "2. Totals reconciliation",
        "Flagged-anomaly count equals anomaly_count",
        anomaly_count == flagged,
        f"summary anomaly_count={anomaly_count:,} | flagged csv rows={flagged:,}",
    )

    # --- co-occurrence -----------------------------------------------------
    check(
        "2. Totals reconciliation",
        "Co-occurrence distinct pairs reported and positive",
        isinstance(cooc_counts.get("distinct_pairs_overall"), int)
        and cooc_counts.get("distinct_pairs_overall", 0) > 0,
        f"distinct_pairs_overall={cooc_counts.get('distinct_pairs_overall'):,} | "
        f"qualifying_pairs={cooc_counts.get('qualifying_pairs'):,}",
    )


# ---------------------------------------------------------------------------
# Part 3 - data-reader smoke tests
# ---------------------------------------------------------------------------
def validate_readers() -> None:
    from dashboard.components import data as ds

    summary = ds.get_json("adr_summary")
    check("3. Data readers", "get_json('adr_summary') returns a dict",
          isinstance(summary, dict), f"type={type(summary).__name__}")

    head = ds.get_rows("adr_top_reactions", limit=5, columns=["reaction_pt", "reports"])
    check(
        "3. Data readers",
        "get_rows applies the row limit and column projection",
        len(head) == 5 and all(set(r.keys()) == {"reaction_pt", "reports"} for r in head),
        f"rows={len(head)} cols={sorted(head[0].keys()) if head else []}",
    )

    filtered = ds.filter_rows("adr_drug_reaction_pairs", ["drug_name"], "DUPIXENT", limit=25)
    check(
        "3. Data readers",
        "filter_rows returns only the requested drug (DUPIXENT)",
        bool(filtered) and all(r.get("drug_name") == "DUPIXENT" for r in filtered),
        f"rows={len(filtered)}",
    )

    scores = ds.numeric_column("anomaly_results", "anomaly_score", limit=1000)
    check(
        "3. Data readers",
        "numeric_column streams exactly `limit` floats",
        len(scores) == 1000 and all(isinstance(v, float) for v in scores),
        f"values={len(scores)} first={scores[0] if scores else None}",
    )

    counted = ds.count_rows("signal_signals")
    check("3. Data readers", "count_rows('signal_signals') is positive", counted > 0, f"{counted:,}")

    distinct = ds.distinct_values("network_nodes", "drug", limit=200)
    check("3. Data readers", "distinct_values returns sorted unique strings",
          bool(distinct) and distinct == sorted(set(distinct)), f"{len(distinct)} values")

    cols = ds.column_values("signal_signals", "drug", limit=10)
    check("3. Data readers", "column_values streams one column",
          len(cols) == 10, f"{len(cols)} values")


# ---------------------------------------------------------------------------
# Part 4 - chart builder smoke tests
# ---------------------------------------------------------------------------
def validate_charts() -> None:
    import plotly.graph_objects as go

    from dashboard.components import charts as ch

    builders = {
        "horizontal_bar": lambda: ch.horizontal_bar(["A", "B"], [3, 5], title="t"),
        "vertical_bar": lambda: ch.vertical_bar(["A", "B"], [3, 5], title="t"),
        "grouped_bar": lambda: ch.grouped_bar(["Q1", "Q2"], {"x": [1, 2]}, title="t"),
        "line_chart": lambda: ch.line_chart(["a", "b"], [1, 2], title="t"),
        "multi_line": lambda: ch.multi_line(["a", "b"], {"x": [1, 2]}, title="t"),
        "diverging_bar": lambda: ch.diverging_bar(["A", "B"], [-1, 2], title="t"),
        "histogram": lambda: ch.histogram([1, 2, 2, 3], title="t"),
        "pie_chart": lambda: ch.pie_chart(["A", "B"], [3, 5], title="t"),
        "scatter": lambda: ch.scatter([1, 2], [3, 4], title="t"),
        "error_bar_scatter": lambda: ch.error_bar_scatter(["a", "b"], [2, 3], [1, 1], [3, 4], title="t"),
        "network_graph": lambda: ch.network_graph(
            [{"x": 0, "y": 0, "label": "A", "size": 10, "color": "#000"}],
            [],
            title="t",
        ),
        "heatmap": lambda: ch.heatmap([[1, 2], [3, 4]], ["c1", "c2"], ["r1", "r2"], title="t"),
    }
    for name, build in builders.items():
        try:
            fig = build()
            ok = isinstance(fig, go.Figure) and len(fig.data) > 0
            check("4. Chart builders", f"{name} returns a populated Figure", ok,
                  f"traces={len(fig.data)}")
        except Exception as exc:  # noqa: BLE001
            check("4. Chart builders", f"{name} returns a populated Figure", False,
                  f"{type(exc).__name__}: {exc}")

    # Empty input must produce an explanatory annotation, never a blank canvas.
    empty_fig = ch.horizontal_bar([], [], title="t")
    check(
        "4. Chart builders",
        "Empty input yields an explanatory (not blank) figure",
        len(empty_fig.layout.annotations) >= 1,
        f"annotations={len(empty_fig.layout.annotations)}",
    )


# ---------------------------------------------------------------------------
# Part 5 - section module imports
# ---------------------------------------------------------------------------
def validate_imports() -> None:
    import importlib

    modules = [
        "executive",
        "drugs",
        "reactions",
        "cooccurrence",
        "network",
        "signals",
        "machine_learning",
        "temporal",
        "demographic",
        "investigation",
    ]
    for name in modules:
        try:
            module = importlib.import_module(f"dashboard.sections.{name}")
            ok = callable(getattr(module, "render", None))
            check("5. Section imports", f"dashboard.sections.{name}.render is callable", ok)
        except Exception as exc:  # noqa: BLE001
            check("5. Section imports", f"dashboard.sections.{name} imports", False,
                  f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Part 6 - headless render test
# ---------------------------------------------------------------------------
def validate_render() -> None:
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as exc:  # noqa: BLE001
        record("6. Headless render", "AppTest availability", "SKIP",
               f"{type(exc).__name__}: {exc}")
        return

    from dashboard.sections import SECTION_MODULES

    app_path = ROOT / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=600)

    try:
        at.run()
    except Exception as exc:  # noqa: BLE001
        check("6. Headless render", "app.py starts without raising", False,
              f"{type(exc).__name__}: {exc}")
        return

    ok = not at.exception
    detail = "" if ok else "; ".join(str(e.value) for e in at.exception)
    check("6. Headless render", "app.py starts without raising", ok, detail)

    titles = [t.value for t in at.title]
    check("6. Headless render", "Platform title is rendered", bool(titles),
          f"title={titles[0] if titles else '(none)'}")

    labels = [label for label, _ in SECTION_MODULES]
    if not at.sidebar.radio:
        check("6. Headless render", "Sidebar navigation radio exists", False)
        return
    check("6. Headless render", "Sidebar navigation radio exists", True,
          f"{len(labels)} sections")

    for label in labels:
        try:
            at.sidebar.radio[0].set_value(label)
            at.run()
            ok = not at.exception
            detail = "" if ok else "; ".join(str(e.value) for e in at.exception)
            hint = f"markdown={len(at.markdown)}, metrics={len(at.metric)}"
            check("6. Headless render", f"section renders: {label}", ok,
                  detail or hint)
        except Exception as exc:  # noqa: BLE001
            check("6. Headless render", f"section renders: {label}", False,
                  f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_report(extra_notes: List[str]) -> Dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for item in RESULTS:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    total = len(RESULTS)

    import plotly
    import streamlit

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("DASHBOARD VALIDATION REPORT")
    lines.append("Big Data-Driven Drug Safety Intelligence Platform")
    lines.append("=" * 78)
    lines.append(f"Generated at   : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Project root   : {ROOT}")
    lines.append(f"Python         : {sys.version.split()[0]}")
    lines.append(f"Streamlit      : {streamlit.__version__}")
    lines.append(f"Plotly         : {plotly.__version__}")
    lines.append("Run command    : python -m streamlit run dashboard/app.py")
    lines.append("")
    lines.append("SUMMARY")
    lines.append(f"  Total checks : {total}")
    lines.append(f"  Passed       : {counts['PASS']}")
    lines.append(f"  Failed       : {counts['FAIL']}")
    lines.append(f"  Skipped      : {counts['SKIP']}")
    lines.append(f"  Result       : {'PASS' if counts['FAIL'] == 0 else 'FAIL'}")
    lines.append("")

    # group by category, preserving order
    categories: List[str] = []
    for item in RESULTS:
        if item["category"] not in categories:
            categories.append(item["category"])

    for category in categories:
        lines.append("-" * 78)
        lines.append(category.upper())
        lines.append("-" * 78)
        for item in [r for r in RESULTS if r["category"] == category]:
            lines.append(f"[{item['status']}] {item['name']}")
            if item["detail"]:
                lines.append(f"        {item['detail']}")
        lines.append("")

    lines.append("-" * 78)
    lines.append("UNRESOLVED ERRORS / NOTES")
    lines.append("-" * 78)
    failures = [r for r in RESULTS if r["status"] == "FAIL"]
    if failures:
        for item in failures:
            lines.append(f"* FAIL - {item['category']} / {item['name']}: {item['detail']}")
    else:
        lines.append("* No failing checks.")
    for note in extra_notes:
        lines.append(f"* NOTE - {note}")
    lines.append("")
    lines.append("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return counts


def main() -> int:
    notes: List[str] = []
    steps = [
        ("inventory", validate_inventory),
        ("totals", validate_totals),
        ("readers", validate_readers),
        ("charts", validate_charts),
        ("imports", validate_imports),
        ("render", validate_render),
    ]
    for name, step in steps:
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            record("0. Harness", f"validation step '{name}' crashed", "FAIL",
                   f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            notes.append(f"validation step '{name}' raised {type(exc).__name__}: {exc}")

    counts = write_report(notes)
    print(f"Validation complete: {counts['PASS']} passed, "
          f"{counts['FAIL']} failed, {counts['SKIP']} skipped.")
    print(f"Report written to: {REPORT_PATH}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
