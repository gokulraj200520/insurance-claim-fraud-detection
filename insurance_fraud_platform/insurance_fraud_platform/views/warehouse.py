"""Data warehouse design, ETL monitoring and pipeline control."""
import plotly.graph_objects as go
import streamlit as st

from src import database as db
from src.pipeline import build_database

from . import data_access as da
from . import ui

STAR_SCHEMA_DOT = """
digraph star {
  rankdir=LR; bgcolor="white"; nodesep=0.5; ranksep=1.1;
  node [shape=record, fontname="Times New Roman", fontsize=12, style=filled, fillcolor="#F4F6F9", color="#1F3A5F"];
  edge [color="#2F6690", arrowsize=0.8];
  fact [fillcolor="#1F3A5F", fontcolor="white", label="fact_claims|claim_key (PK)\\ldate_key (FK)\\lcustomer_key (FK)\\lpolicy_key (FK)\\lprovider_key (FK)\\lclaim_type_key (FK)\\lclaim_amount\\lclaim_to_coverage_ratio\\ldays_since_policy_start\\ldays_to_report\\lnum_documents\\lpolice_report\\lwitness_present\\lprior_claims\\lis_fraud\\l"];
  d1 [label="dim_date|date_key (PK)\\lfull_date\\lmonth, quarter, year\\lday_name, is_weekend\\l"];
  d2 [label="dim_customer|customer_key (PK)\\lcustomer_id\\lage, age_group, gender\\lcity, occupation\\lincome_band, credit_score\\l"];
  d3 [label="dim_policy|policy_key (PK)\\lpolicy_id, policy_type\\lpremium, coverage_amount\\lstart_date\\l"];
  d4 [label="dim_provider|provider_key (PK)\\lprovider_id, provider_name\\lprovider_type\\l"];
  d5 [label="dim_claim_type|claim_type_key (PK)\\lclaim_type\\lclaim_category\\l"];
  d1 -> fact; d2 -> fact; d3 -> fact; d4 -> fact; d5 -> fact;
  scores [fillcolor="#FBEFC9", label="fraud_scores|claim_id (PK)\\lmodel_probability\\liso_score\\lz_flag, iqr_flag\\lrisk_score, risk_level\\ltop_reasons, status\\l"];
  fact -> scores [style=dashed, label="claim_id"];
}
"""


def render() -> None:
    ui.page_header("Data Warehouse and ETL", "Star schema design, ETL execution log and data quality monitoring")
    v = db.db_version()

    ui.section("Star Schema")
    st.graphviz_chart(STAR_SCHEMA_DOT)
    ui.note("The warehouse follows a <b>star schema</b>: one central fact table (<i>fact_claims</i>, grain = one claim) "
            "surrounded by five conformed dimensions. <i>fraud_scores</i> stores the mining output and "
            "<i>vw_claims_full</i> joins everything for analytics.")

    ui.section("Warehouse Contents")
    c1, c2 = st.columns([1, 2])
    counts = db.table_counts()
    with c1:
        ui.html_table(counts, max_height=420)
    with c2:
        sel = st.selectbox("Preview table", counts["Table"].tolist(), index=5)
        ui.html_table(db.query(f"SELECT * FROM {sel} LIMIT 15"), max_height=360)

    ui.section("ETL Pipeline")
    e1, e2, e3 = st.columns(3)
    for col, title, text in [
        (e1, "1. Extract", "Raw claim records are read from the operational CSV extract and staged unchanged in <i>stg_raw_claims</i>."),
        (e2, "2. Transform", "Duplicates removed, codes standardised, dates parsed, invalid values corrected, missing values imputed, features derived."),
        (e3, "3. Load", "Dimensions loaded first, surrogate keys resolved, then the fact table is populated and scored."),
    ]:
        col.markdown(f'<div class="kpi" style="height:150px"><div class="label">{title}</div><div class="sub" style="font-size:1rem;color:#1B1B1B;margin-top:6px">{text}</div></div>', unsafe_allow_html=True)

    log = db.query("SELECT stage AS Stage, metric AS Metric, CAST(value AS INTEGER) AS Value, run_at AS 'Run At' FROM etl_log")
    q = da.quality(v)
    a, b = st.columns([1, 1])
    with a:
        st.markdown("**Latest ETL run**")
        ui.html_table(log, max_height=300)
    with b:
        fig = go.Figure()
        fig.add_bar(x=q["Attribute"], y=q["Missing (raw)"], name="Raw (staging)", marker_color=ui.RISK_COLORS["High"])
        fig.add_bar(x=q["Attribute"], y=q["Missing (warehouse)"], name="Warehouse (cleaned)", marker_color=ui.RISK_COLORS["Low"])
        fig.update_layout(barmode="group", yaxis_title="Missing values")
        ui.show(ui.style_fig(fig, 330, "Data quality: missing values before and after ETL"), key="wh_quality")

    ui.section("Pipeline Control")
    st.write("Regenerate the synthetic raw data, re-run the ETL pipeline and re-score every claim. "
             "Investigation statuses are reset by this action.")
    if st.button("Rebuild data warehouse"):
        with st.spinner("Running ETL, training models and scoring claims ..."):
            summary = build_database(regenerate=True)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success(f"Rebuild complete: {summary['rows_loaded']:,} claims loaded, {summary['high_risk']:,} high-risk alerts.")
