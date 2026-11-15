# Big Data-Driven Drug Safety Intelligence Platform

Adverse Drug Reaction (ADR) analysis, drug co-occurrence mining, disproportionality
signal detection and anomaly detection on **real FDA FAERS ASCII data
(2026 Q1 + Q2)** using PySpark and classical machine learning.

> **Current status: implemented and validated.**
> All twelve analysis modules run against the real FAERS extracts and write
> auditable CSV / JSON / HTML outputs. The ten-section Streamlit dashboard
> renders headlessly with 53/53 validation checks passing, and 26 Power BI-ready
> CSVs are exported. See [reports/final_validation_report.txt](reports/final_validation_report.txt)
> for the component-by-component result and [reports/FINAL_PROJECT_REPORT.md](reports/FINAL_PROJECT_REPORT.md)
> for the full write-up.

---

## 1. Objective

Turn the raw FDA FAERS quarterly extracts into an auditable drug safety
intelligence platform that answers:

| Question | Technique |
| --- | --- |
| Which drugs are reported most, and in what roles? | PySpark aggregation |
| Which adverse events co-occur with which drugs? | Contingency tables + PRR / ROR |
| Which drugs are co-reported together? | Apriori association rules |
| Which drug–reaction pairs are statistically disproportionate? | PRR, ROR with 95% CI, chi-square |
| Which reports look structurally unusual? | Isolation Forest |
| How are reports structured across drugs and reactions? | Drug co-occurrence network (Louvain) |
| How do signals evolve over time and across demographics? | Temporal / demographic analytics |
| How can an analyst explore the results? | 10-section Streamlit dashboard + Power BI CSV exports |

---

## 2. Technology stack

| Layer | Technology |
| --- | --- |
| Processing engine | **PySpark 3.5.9** (DataFrame API only) |
| Language | Python 3.14.7 |
| Machine learning | scikit-learn (`IsolationForest`) |
| Graph analytics | NetworkX (Louvain communities, betweenness) |
| Visualisation | Plotly 7.1.0 (UI-agnostic figures) |
| Dashboard | Streamlit 1.67.0 |
| Configuration | PyYAML (`config.yaml`) |
| Tests | pytest |

The pipeline deliberately uses the **PySpark DataFrame API only** — no pandas,
no PyArrow and no Python UDFs. This is a hard environment constraint: the host
Windows policy blocks the native libraries those components load. It is also a
correctness choice, since keeping all logic in Spark avoids an implicit
collect-to-driver step.

> **Not used / not claimed:** Hadoop HDFS and cluster execution (Spark runs with
> `master: local[*]`), and the Parquet write path (disabled — see §7).

---

## 3. Repository structure

```
Drug Safety Analytics/
├── data/
│   ├── raw/                  # FAERS ASCII source files  (READ ONLY)
│   │   ├── 2026_Q1/          #   DEMO26Q1.txt, DRUG26Q1.txt, ... + DELETE26Q1.txt
│   │   └── 2026_Q2/
│   ├── processed/            # reserved for cleaned/typed output (Parquet disabled)
│   ├── analytics/            # aggregated analysis outputs (CSV, JSON, HTML figures)
│   └── powerbi/              # 26 Power BI-ready CSVs + data_dictionary.txt
├── src/
│   ├── ingestion/            # inspect_faers, load_faers, spark_session, parquet_output
│   ├── preprocessing/        # clean_faers
│   ├── analytics/            # adr_analysis, drug_cooccurrence, apriori_analysis, signal_detection
│   ├── machine_learning/     # anomaly_detection (Isolation Forest)
│   ├── visualization/        # drug_network, temporal_demographic, powerbi_export
│   └── utils/                # config loader, logging setup
├── dashboard/
│   ├── app.py                # Streamlit entry point (navigation shell)
│   ├── components/           # data.py, charts.py, ui.py
│   ├── sections/             # the 10 dashboard sections
│   └── validate_dashboard.py # headless (AppTest) validation harness
├── models/                   # isolation_forest_model.pkl
├── reports/                  # generated reports, logs and the final documentation
├── notebooks/                # exploratory work
├── tests/                    # pytest unit tests
├── config.yaml               # all paths, thresholds and Spark settings
├── requirements.txt
└── .gitignore
```

| Folder | Purpose |
| --- | --- |
| `data/raw/` | The untouched quarterly FAERS ASCII files. **Nothing in this project writes here.** |
| `data/processed/` | Target for cleaned/typed Parquet. Because Parquet writing is disabled here, it holds only the `.gitkeep` and two **empty** leftover probe directories (`DEMO.parquet/`, `_fs_probe/`) from the capability tests — no processed dataset was written. |
| `data/analytics/` | Final tabular results consumed by the dashboard: signal tables, association rules, anomaly scores, network tables, aggregated counts. |
| `data/powerbi/` | Flat CSV exports (with a data dictionary) for import into Power BI. |
| `models/` | Persisted Isolation Forest artefact. |
| `reports/` | Generated `.txt` reports, validation reports and the final documentation set. |

---

## 4. Getting the data

The FAERS ASCII extracts are **not** redistributed here (size and licensing).
Download the 2026 Q1 and Q2 ASCII packages from the FDA FAERS quarterly data
page and extract them so each quarter folder contains the seven `$`-delimited
files (they ship with a header row):

```
data/raw/2026_Q1/ASCII/DEMO26Q1.txt
                         DRUG26Q1.txt
                         REAC26Q1.txt
                         OUTC26Q1.txt
                         RPSR26Q1.txt
                         THER26Q1.txt
                         INDI26Q1.txt
                         (and DELETE26Q1.txt — withdrawn cases)
```

File naming, delimiter, encoding and the table list are declared in
[config.yaml](config.yaml) under `faers:`; no script hard-codes them.

---

## 5. How to run

All modules are run as Python modules from the project root.

```bash
# 1. Inspect the raw files (read-only) and write reports/dataset_inspection.txt
python -m src.ingestion.inspect_faers

# 2. Ingest the seven ASCII tables (optionally write Parquet if enabled)
python -m src.ingestion.load_faers

# 3. Clean, type and flag (validation only — raw files unchanged)
python -m src.preprocessing.clean_faers

# 4. ADR counts, top reactions/drugs and outcomes
python -m src.analytics.adr_analysis

# 5. Drug co-occurrence pairs and groups
python -m src.analytics.drug_cooccurrence

# 6. Apriori frequent itemsets and association rules
python -m src.analytics.apriori_analysis

# 7. PRR / ROR disproportionality signal detection
python -m src.analytics.signal_detection

# 8. Isolation Forest anomaly detection
python -m src.machine_learning.anomaly_detection

# 9. Drug co-occurrence network (communities + interactive HTML)
python -m src.visualization.drug_network

# 10. Temporal and demographic analytics
python -m src.visualization.temporal_demographic

# 11. Export the Power BI CSV tables
python -m src.visualization.powerbi_export

# 12. Validate the dashboard headlessly
python dashboard/validate_dashboard.py
```

### Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

---

## 6. Dashboard sections

The dashboard reads only from `data/analytics/` (never from `data/raw/`) and is a
thin navigation shell over ten sections:

1. Executive Overview
2. Drug Analytics
3. Adverse Reaction Analysis
4. Drug Co-Occurrence
5. Drug Network
6. Statistical Signal Detection
7. Machine Learning (Isolation Forest)
8. Temporal Analysis
9. Demographic Analysis
10. Drug Investigation

Every section offers CSV download of the underlying table via `st.download_button`,
and a failing section is caught and reported inline rather than crashing the app.

---

## 7. Environment notes (Windows)

* Python **3.14.7**, PySpark **3.5.9**, Streamlit **1.67.0**, Plotly **7.1.0**.
* `JAVA_HOME` is **not set**; the JDK on `PATH` is `java 1.8.0_491`.
* PySpark 3.5.x officially targets Python 3.10–3.12, so a version-mismatch warning
  is printed at startup but does not affect results.
* **Parquet output is disabled** (`faers.write_parquet: false`): Spark's Hadoop-backed
  writer needs `winutils.exe` (absent) and the PyArrow/pandas native libraries are
  blocked by the OS Application Control policy. All analysis therefore operates on
  the in-memory Spark frames and writes plain-text CSV/JSON outputs.
* **Hadoop HDFS / cluster mode is not used**: `spark.master: local[*]`.
* The JVM sometimes exits with a non-zero code **during shutdown after a run
  completes** (a JVM `.timestamp` write restriction in the sandbox). This happens
  after all output has been written and does not affect results; it is recorded in
  the validation report rather than hidden.

---

## 8. Data integrity rules

These rules are binding for every script in this project:

1. **`data/raw/` is read-only.** Original datasets are never modified, moved or
   rewritten. All output goes to `data/analytics/`, `data/powerbi/`, `models/` or `reports/`.
2. **Real data only.** No synthetic, simulated or placeholder results are reported as findings.
3. **No fabricated metrics.** If an input table is missing, the pipeline fails loudly
   rather than silently substituting values.
4. **Reproducibility.** Config-driven parameters and fixed random seeds
   (Isolation Forest `random_state=42`, network layout `seed=42`).

---

## 9. Component status

| Component | Status |
| --- | --- |
| FAERS dataset inspection | PASS |
| Data ingestion | PASS |
| Data preprocessing | PASS |
| ADR analysis | PASS |
| Drug co-occurrence analysis | PASS |
| Apriori association mining | PASS |
| PRR/ROR signal detection | PASS |
| Isolation Forest | PASS |
| Drug network visualisation | PASS |
| Temporal & demographic analysis | PASS |
| Streamlit dashboard | PASS |
| Power BI CSV export | PASS |
| Power BI Desktop `.pbix` report | NOT TESTED (no `.pbix` produced — CSV exports only) |
| Hadoop HDFS / cluster execution | NOT TESTED (not implemented) |
| Parquet / PyArrow write path | BLOCKED (environment) |

Full detail, with the exact numbers, is in
[reports/final_validation_report.txt](reports/final_validation_report.txt).
