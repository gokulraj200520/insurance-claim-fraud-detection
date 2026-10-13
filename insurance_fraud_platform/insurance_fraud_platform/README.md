# Insurance Claim Fraud Detection Platform

**Data Warehousing and Data Mining - Capstone Project (A Section) | Team No. 13 | Difficulty: Medium**

An end-to-end analytics platform that loads insurance claims into a **star-schema data warehouse (SQLite)** through an **ETL pipeline**, explores them with **OLAP**, and uses **data mining** (classification, PCA, Isolation Forest, statistical outlier detection) to score every claim for fraud risk. Everything - front end and back end - is written in **Python with Streamlit**, so it can be deployed directly to Streamlit Community Cloud.

| Reg. No. | Name of the Student |
|---|---|
| 611223104046 | Gokul Raj S |
| 611223104013 | Ashwin Prasath B |
| 611223104018 | Bharath J K |
| 611223104054 | Hari Prasath J |

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction and Problem Statement](#2-introduction-and-problem-statement)
3. [Objectives and Scope](#3-objectives-and-scope)
4. [Key Features](#4-key-features)
5. [System Architecture](#5-system-architecture)
6. [Data Warehouse Design (Unit I)](#6-data-warehouse-design-unit-i)
7. [ETL and OLAP (Unit II)](#7-etl-and-olap-unit-ii)
8. [Data Cleaning and PCA (Unit III)](#8-data-cleaning-and-pca-unit-iii)
9. [Classification Models and Evaluation (Unit IV)](#9-classification-models-and-evaluation-unit-iv)
10. [Anomaly Detection (Unit V)](#10-anomaly-detection-unit-v)
11. [Composite Risk Scoring](#11-composite-risk-scoring)
12. [Results](#12-results)
13. [Application Modules](#13-application-modules)
14. [Tools and Technologies Used](#14-tools-and-technologies-used)
15. [Project Structure](#15-project-structure)
16. [Installation and Execution (VS Code)](#16-installation-and-execution-vs-code)
17. [Deployment on Streamlit Cloud](#17-deployment-on-streamlit-cloud)
18. [Using Your Own Data](#18-using-your-own-data)
19. [Limitations and Future Enhancements](#19-limitations-and-future-enhancements)
20. [Conclusion](#20-conclusion)

---

## 1. Abstract

Insurance fraud costs the industry billions every year, inflates premiums for honest customers and overwhelms investigators who must review thousands of claims manually. This project presents an **Insurance Claim Fraud Detection Platform** that applies data warehousing and data mining techniques to identify suspicious claims and support the fraud investigation process.

Raw claim records, containing duplicates, missing values, inconsistent coding and invalid entries, are extracted into a staging area, cleaned and transformed by an ETL pipeline, and loaded into a **SQLite star-schema data warehouse** consisting of one fact table and five dimensions. Multidimensional analysis is provided through **OLAP operations** (roll-up, drill-down, slice, dice and pivot). For mining, features are engineered and analysed with **Principal Component Analysis**; three **classification models** (Logistic Regression, Decision Tree, Random Forest) are trained and compared using accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC; and unsupervised **Isolation Forest** and **Z-score / IQR statistical outlier detection** capture anomalies without relying on labels.

The outputs are combined into a transparent **0-100 composite risk score** with plain-language red flags for each claim. Results are delivered through an interactive **Streamlit** application containing an executive dashboard, a fraud alert queue with a case-management workflow, risk reports, customer profiling and segmentation, an OLAP explorer, a data-mining lab, and a real-time claim scorer.

**Keywords:** Data Warehouse, Star Schema, ETL, OLAP, Data Mining, Fraud Detection, Classification, PCA, Isolation Forest, Anomaly Detection, Streamlit, SQLite.

## 2. Introduction and Problem Statement

Fraudulent claims take many forms: staged accidents, inflated repair bills, claims filed shortly after a policy is bought, falsified documents, or repeat claimants working with complicit providers. Investigators have limited time, so the challenge is to **rank claims by fraud likelihood** and explain *why* a claim is suspicious.

**Problem statement:** *Build an insurance analytics platform that identifies suspicious claims and supports fraud investigation processes.*

Operational claim data is rarely analysis-ready: it is spread across systems, inconsistently coded and incomplete. A data warehouse gives a clean, integrated, subject-oriented history on which mining algorithms can run reliably.

## 3. Objectives and Scope

**Objectives**
- Design and implement a star-schema data warehouse for insurance claims.
- Build a repeatable ETL pipeline that cleans and integrates raw claim data.
- Provide OLAP-style multidimensional analysis of claims, cost and fraud.
- Detect fraud using supervised classification and unsupervised anomaly detection.
- Produce an explainable risk score, fraud alerts, risk reports and a claims analytics dashboard.
- Support investigators with case status tracking, customer profiles and claim-level red flags.

**Scope:** four lines of business (Auto, Health, Home, Life), about 6,000 claims over 2022-2025, 2,000 customers and 160 service providers across 12 Indian cities. The data is **synthetically generated** (see Section 18 for plugging in real data).

## 4. Key Features

| Feature | Description |
|---|---|
| Fraud detection | Random Forest classifier + Isolation Forest + Z-score/IQR tests |
| Risk scoring | Composite 0-100 score with Low / Medium / High bands |
| Claim analytics | Executive dashboard, OLAP explorer, risk reports |
| Customer profiling | Individual customer profiles and K-Means behavioural segments |
| Investigation workflow | Alert queue, red-flag explanations, case status updates saved to the database |
| Real-time scoring | Enter a new claim and obtain an instant fraud probability |
| Exports | Download alerts, OLAP results and monthly risk reports as CSV |

## 5. System Architecture

```
  Raw claims (CSV)                       Streamlit application (frontend + backend)
        |                                  +--------------------------------------------+
        v                                  |  Dashboard | Alerts | Reports | Profiling   |
  +-------------+   +----------------+     |  OLAP Explorer | Warehouse | Mining Lab     |
  |  EXTRACT    |-->|   TRANSFORM    |     |  Claim Risk Scorer | About                  |
  | staging tbl |   | clean/standard |     +---------------------^-----------------------+
  +-------------+   | impute/derive  |                           |
                    +-------+--------+                 SQL / pandas queries
                            |                                    |
                            v                                    |
                    +----------------+      +---------------------+-----------------+
                    |      LOAD      |----->|  SQLite DATA WAREHOUSE (star schema)    |
                    +----------------+      |  fact_claims + 5 dimensions + scores    |
                                            +---------------------^-----------------+
                                                                  |
                    +---------------------------------------------+---------------+
                    | DATA MINING: PCA | Logistic / Tree / Forest | Isolation Forest |
                    | Z-score & IQR  -->  Composite risk score + red-flag reasons    |
                    +----------------------------------------------------------------+
```

## 6. Data Warehouse Design (Unit I)

A **star schema** with grain *one row per claim*.

| Table | Type | Key attributes |
|---|---|---|
| `fact_claims` | Fact | claim_amount, claim_to_coverage_ratio, days_since_policy_start, days_to_report, num_documents, police_report, witness_present, prior_claims, is_fraud |
| `dim_date` | Dimension | date_key, year, quarter, month, month_name, day_name, is_weekend |
| `dim_customer` | Dimension | customer_id, age, age_group, gender, city, occupation, income_band, credit_score |
| `dim_policy` | Dimension | policy_id, policy_type, premium, coverage_amount, start_date |
| `dim_provider` | Dimension | provider_id, provider_name, provider_type |
| `dim_claim_type` | Dimension | claim_type, claim_category |
| `fraud_scores` | Mining output | model_probability, iso_score, z_flag, iqr_flag, risk_score, risk_level, top_reasons, status |
| `stg_raw_claims` | Staging | Unmodified copy of the raw extract |
| `etl_log` | Metadata | ETL run statistics |
| `vw_claims_full` | View | Denormalised join of the star used by all analytics |

The star-schema diagram is drawn in the **Data Warehouse and ETL** page of the app.

## 7. ETL and OLAP (Unit II)

**Extract** - raw records are read from `data/raw_claims.csv` and kept unchanged in `stg_raw_claims`.

**Transform**
1. Remove exact duplicate claim records.
2. Standardise codes and types (gender `M/male/F` -> `Male/Female`, policy type casing and whitespace, `Y/yes/True` -> 1).
3. Parse mixed date formats (`YYYY-MM-DD` and `DD/MM/YYYY`).
4. Treat impossible values (age < 18 or > 90, credit score outside 300-900) as missing.
5. Impute missing values: customer-level value from the same customer's other claims first, then median by policy type / global median or mode.
6. Derive features: claim-to-coverage ratio, days since policy start, age group, income band, claim category.

**Load** - dimensions are loaded first, surrogate keys are resolved by join, then the fact table is populated. Run statistics are written to `etl_log`.

**OLAP operations** (all executed as SQL `GROUP BY` over the star schema; the generated SQL is shown in the app):

| Operation | Meaning in this project |
|---|---|
| Roll-up / Drill-down | Move along the time hierarchy Year -> Quarter -> Month |
| Slice | Fix one dimension (e.g. Policy Type = Auto) |
| Dice | Filter several dimensions at once (e.g. City, Year, Policy Type) |
| Pivot | Rotate the cube - any two dimensions as rows x columns, shown as a heat map |

Measures: total claims, total / average claim amount, confirmed fraud cases, fraud rate, high-risk alerts, average risk score.

## 8. Data Cleaning and PCA (Unit III)

- **Data cleaning:** performed in the ETL transform stage; a before/after missing-value comparison is shown in the app.
- **PCA:** the 20 engineered features (16 numeric + 4 policy-type indicators) are standardised and reduced with PCA. The Data Mining Lab shows the **scree plot**, cumulative explained variance, a **2-D projection** of claims coloured by fraud label, and the strongest feature loadings.

## 9. Classification Models and Evaluation (Unit IV)

| Model | Notes |
|---|---|
| Logistic Regression | Standardised inputs, balanced class weights - interpretable baseline |
| Decision Tree | Depth-limited (6), balanced class weights |
| Random Forest | 200 trees, balanced sub-sample weights - production model |

A stratified 75/25 train/test split is used. Because fraud is rare (about 9%), accuracy alone is misleading, so the platform reports **accuracy, precision, recall, F1-score, ROC-AUC and PR-AUC**, together with ROC curves, confusion matrices and Random Forest feature importance. Final risk scores use **5-fold out-of-fold probabilities**, so no claim is scored by a model that saw its own label.

## 10. Anomaly Detection (Unit V)

- **Isolation Forest** - isolates unusual claims across seven behavioural features without using labels; the contamination rate can be tuned interactively.
- **Z-score** - flags claim values more than 3 standard deviations above the mean of the policy type.
- **IQR fences** - flags values above Q3 + 1.5 x IQR within each policy type.

## 11. Composite Risk Scoring

```
Risk score = 0.60 x Classifier probability + 0.25 x Isolation Forest score + 0.15 x Statistical outlier signal      (scaled 0 - 100)
```

| Risk level | Score |
|---|---|
| High | 60 and above |
| Medium | 35 to < 60 |
| Low | below 35 |

Each claim also receives up to four human-readable **red flags** (for example *"Filed only 12 days after policy inception"*, *"Claim equals 78% of insured coverage"*, *"Only 1 supporting document submitted"*).

## 12. Results

Results obtained on the bundled synthetic dataset (6,000 claims, 547 labelled frauds):

**ETL summary:** 6,120 raw rows extracted, 120 duplicates removed, 37 invalid values corrected, 814 missing values imputed, 6,000 clean rows loaded.

**Classification (hold-out test set)**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.863 | 0.381 | 0.796 | 0.515 | 0.883 | 0.630 |
| Decision Tree | 0.800 | 0.278 | 0.745 | 0.405 | 0.813 | 0.482 |
| Random Forest | 0.928 | 0.592 | 0.679 | 0.633 | 0.879 | 0.605 |

**Anomaly detection (against labelled frauds)**

| Method | Flagged | Frauds caught | Precision | Recall |
|---|---|---|---|---|
| Z-score (abs z > 3) | 340 | 192 | 0.56 | 0.35 |
| IQR fences | 834 | 343 | 0.41 | 0.63 |
| Isolation Forest (9%) | 540 | 264 | 0.49 | 0.48 |

**Composite risk score**

| Risk level | Claims | Confirmed frauds inside |
|---|---|---|
| High | 494 | 303 (61% precision) |
| Medium | 228 | 71 |
| Low | 5,278 | 173 |

The High band alone captures 55% of all frauds while containing only 8% of claims; High + Medium captures 68% with 12% of claims, which lets investigators focus effort where it matters most. *(Exact figures may differ slightly if you rebuild the data on a different library version.)*

## 13. Application Modules

| Page | Purpose |
|---|---|
| Executive Dashboard | KPIs, monthly trend, risk distribution, segment charts, risk heat map, top-10 risky claims |
| Fraud Alerts | Filterable alert queue, case investigation panel with red flags, status updates, CSV export |
| Risk Reports | Auto-generated key findings, segment analysis, provider risk ranking, monthly report |
| Customer Profiling | Individual customer history and K-Means segmentation (4 segments) |
| OLAP Explorer | Roll-up, drill-down, slice, dice, pivot with generated SQL |
| Data Warehouse and ETL | Star schema diagram, table row counts, ETL log, data-quality chart, rebuild button |
| Data Mining Lab | Cleaning summary, PCA, model comparison, ROC, confusion matrix, anomaly detection |
| Claim Risk Scorer | Enter claim details and get an instant fraud probability with red flags |
| About the Project | Abstract, syllabus mapping, technology stack, team |

## 14. Tools and Technologies Used

| Category | Tool | Purpose |
|---|---|---|
| Programming language | **Python 3.10+** | Entire project |
| Web framework | **Streamlit** | Frontend and backend in a single application; Streamlit Cloud deployment |
| Database | **SQLite** (`sqlite3`) | Data warehouse (star schema), staging, scores |
| Data processing | **Pandas, NumPy, SciPy** | ETL transformations, aggregation, numerical computing |
| Machine learning | **Scikit-learn** | PCA, Logistic Regression, Decision Tree, Random Forest, Isolation Forest, K-Means, metrics |
| Visualisation | **Plotly**, **Graphviz** (rendered by Streamlit) | Interactive charts, heat maps, star-schema diagram |
| Styling | Custom CSS | White professional theme in Times New Roman |
| Environment | **venv**, `requirements.txt`, `pip` | Dependency management |
| IDE | **Visual Studio Code** | Development |
| Version control / hosting | Git, GitHub, Streamlit Community Cloud | Source control and deployment |

## 15. Project Structure

```
insurance_fraud_platform/
|-- app.py                  # Streamlit entry point (navigation + first-run bootstrap)
|-- build_database.py       # CLI: regenerate data, run ETL, score claims
|-- requirements.txt        # Python dependencies
|-- README.md
|-- .streamlit/config.toml  # White theme configuration
|-- data/
|   |-- raw_claims.csv      # Raw (dirty) operational extract
|   `-- insurance_dw.db     # SQLite data warehouse (pre-built)
|-- src/                    # Back end (no Streamlit dependency)
|   |-- config.py           # Paths, thresholds, weights
|   |-- data_generator.py   # Synthetic raw claim generator
|   |-- database.py         # Star schema DDL and query helpers
|   |-- etl.py              # Extract - Transform - Load
|   |-- features.py         # Feature engineering
|   |-- preprocessing.py    # Data-quality report, PCA
|   |-- models.py           # Classification and evaluation
|   |-- anomaly.py          # Isolation Forest, Z-score, IQR
|   |-- scoring.py          # Composite risk score and red flags
|   |-- olap.py             # OLAP query engine
|   `-- pipeline.py         # End-to-end pipeline
`-- views/                  # Front end (Streamlit pages)
    |-- ui.py               # Theme, KPI cards, tables, chart styling
    |-- data_access.py      # Cached data loading
    |-- dashboard.py  alerts.py  reports.py  profiling.py  olap_explorer.py
    `-- warehouse.py  mining_lab.py  scorer.py  about.py
```

## 16. Installation and Execution (VS Code)

1. Install **Python 3.10 or newer** and **VS Code**.
2. Extract the zip, then open the folder in VS Code (`File > Open Folder`).
3. Open a terminal (`Terminal > New Terminal`) and create a virtual environment:

   **Windows (PowerShell / CMD)**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   > If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first.

   **macOS / Linux**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```
6. The browser opens at `http://localhost:8501`.

The warehouse (`data/insurance_dw.db`) is **pre-built**, so the app starts immediately. If the file is missing, the app rebuilds it automatically on first run (about 30 seconds). To rebuild manually:
```bash
python build_database.py
```
(In VS Code choose the interpreter from `venv` via `Ctrl+Shift+P > Python: Select Interpreter`.)

## 17. Deployment on Streamlit Cloud

1. Create a GitHub repository and push this folder (the `venv/` folder is excluded by `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Select the repository and branch, and set **Main file path** to `app.py`.
4. Under *Advanced settings* choose Python 3.11 or 3.12, then click **Deploy**.

Streamlit Cloud installs the packages in `requirements.txt` automatically. Note that Streamlit Cloud storage is ephemeral: case-status updates saved from the Fraud Alerts page persist while the app is running but are reset when the app restarts (the database is rebuilt from the repository copy).

## 18. Using Your Own Data

The bundled dataset is synthetic. To use real claims, provide `data/raw_claims.csv` with these columns, then run `python build_database.py` without regenerating (or call `run_etl()` and `score_all_claims()` from `src`):

`claim_id, claim_date, customer_id, customer_name, age, gender, city, occupation, annual_income, credit_score, policy_id, policy_type (Auto/Health/Home/Life), premium, coverage_amount, policy_start_date, claim_type, provider_id, provider_name, provider_type, claim_amount, days_to_report, num_documents, police_report, witness_present, prior_claims, is_fraud`

`is_fraud` (0/1) is the historical investigation outcome used to train the classifiers.

## 19. Limitations and Future Enhancements

- Data is synthetic; real-world performance would depend on data quality and fraud prevalence.
- Possible enhancements: gradient-boosting models, SMOTE/cost-sensitive learning, SHAP explanations, provider-network (graph) analysis, text mining of claim descriptions, scheduled incremental ETL, role-based authentication, and migration to PostgreSQL for multi-user production use.

## 20. Conclusion

The project demonstrates the full data warehousing and data mining life cycle: dimensional modelling, ETL, OLAP, data cleaning and PCA, classification with proper evaluation, and anomaly detection, integrated into a single explainable fraud-scoring platform and delivered as an interactive, deployable Streamlit application.
