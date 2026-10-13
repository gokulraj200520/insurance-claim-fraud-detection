"""Project information page."""
import pandas as pd
import streamlit as st

from . import ui

TEAM = pd.DataFrame({
    "Reg. No.": ["611223104046", "611223104013", "611223104018", "611223104054"],
    "Name of the Student": ["Gokul Raj S", "Ashwin Prasath B", "Bharath J K", "Hari Prasath J"],
})

UNITS = pd.DataFrame({
    "Unit": ["Unit I", "Unit II", "Unit III", "Unit IV", "Unit V"],
    "Concept": ["Data Warehouse Design", "ETL Pipelines, OLAP", "Data Cleaning, PCA",
                "Classification Models, Evaluation Metrics", "Isolation Forest, Statistical Outlier Detection"],
    "Implementation in this project": [
        "Star schema with one fact table and five dimensions in SQLite (Data Warehouse page)",
        "Extract-Transform-Load pipeline with staging and logging; roll-up, drill-down, slice, dice and pivot (OLAP Explorer)",
        "Duplicate removal, imputation, standardisation, invalid-value correction; PCA scree plot and projection (Data Mining Lab)",
        "Logistic Regression, Decision Tree, Random Forest with accuracy, precision, recall, F1, ROC-AUC, PR-AUC",
        "Isolation Forest, Z-score and IQR outlier tests combined into the composite risk score"],
})


def render() -> None:
    ui.page_header("About the Project", "Data Warehousing and Data Mining - Capstone Project (A Section)")
    ui.section("Abstract")
    st.markdown(
        "Insurance fraud inflates premiums and drains claim reserves. This project presents an **Insurance Claim Fraud "
        "Detection Platform** that integrates raw claim data into a SQLite star-schema data warehouse through an ETL "
        "pipeline, explores it with OLAP operations, and applies data mining to identify suspicious claims. A supervised "
        "classifier, an Isolation Forest and statistical outlier tests are blended into a 0-100 composite risk score, with "
        "plain-language red flags to support investigators. Results are delivered through an interactive Streamlit "
        "dashboard containing fraud alerts, risk reports, customer profiling and a claims analytics workbench.")
    ui.section("Composite Risk Score")
    st.markdown("**Risk score = 0.60 x Classifier probability + 0.25 x Isolation Forest score + 0.15 x Statistical outlier signal**, "
                "scaled to 0-100. Claims scoring **60 or more** are *High* risk, **35 to 60** *Medium*, and below 35 *Low*.")
    ui.section("Syllabus Mapping")
    ui.html_table(UNITS, max_height=300)
    ui.section("Technology Stack")
    ui.html_table(pd.DataFrame({
        "Layer": ["Language", "Front end and back end", "Database", "Data processing", "Machine learning", "Visualisation"],
        "Technology": ["Python 3", "Streamlit", "SQLite", "Pandas, NumPy, SciPy", "Scikit-learn", "Plotly, Graphviz (via Streamlit)"]}),
        max_height=280)
    ui.section("Project Team (Team No. 13)")
    ui.html_table(TEAM, max_height=200)
    ui.note("<b>Project:</b> Insurance Claim Fraud Detection Platform &nbsp;|&nbsp; <b>Difficulty:</b> Medium &nbsp;|&nbsp; "
            "<b>Core concepts:</b> Anomaly Detection, Data Mining, Classification")
