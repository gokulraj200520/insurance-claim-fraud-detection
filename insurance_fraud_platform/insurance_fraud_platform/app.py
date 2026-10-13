"""Insurance Claim Fraud Detection Platform - Streamlit entry point.

Run locally:   streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Insurance Claim Fraud Detection Platform", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

from src import database as db  # noqa: E402
from src.pipeline import ensure_ready  # noqa: E402
from views import (about, alerts, dashboard, mining_lab, olap_explorer, profiling,  # noqa: E402
                   reports, scorer, ui, warehouse)

PAGES = {
    "Executive Dashboard": dashboard.render,
    "Fraud Alerts": alerts.render,
    "Risk Reports": reports.render,
    "Customer Profiling": profiling.render,
    "OLAP Explorer": olap_explorer.render,
    "Data Warehouse and ETL": warehouse.render,
    "Data Mining Lab": mining_lab.render,
    "Claim Risk Scorer": scorer.render,
    "About the Project": about.render,
}


def main() -> None:
    ui.inject_css()
    with st.sidebar:
        st.markdown('<div class="brand"><div class="t">Insurance Claim<br>Fraud Detection Platform</div>'
                    '<div class="s">Data Warehousing and Data Mining</div></div>', unsafe_allow_html=True)
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.markdown("---")
        st.caption("Team 13 - Capstone Project")

    if not db.is_ready():
        with st.spinner("First run: generating data, building the warehouse and scoring claims (about 30 seconds) ..."):
            ensure_ready()
        st.cache_data.clear()
        st.cache_resource.clear()
    PAGES[page]()


main()
