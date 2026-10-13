"""Fraud alert queue and investigation workbench."""
import pandas as pd
import streamlit as st

from src import database as db
from src.config import STATUSES

from . import data_access as da
from . import ui


def render() -> None:
    ui.page_header("Fraud Alerts and Investigation", "Prioritised alert queue with explainable red flags and case workflow")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    df = da.claims(db.db_version())

    c1, c2, c3, c4 = st.columns(4)
    levels = c1.multiselect("Risk level", ui.RISK_ORDER[::-1], default=["High"])
    ptypes = c2.multiselect("Policy type (blank = all)", sorted(df["policy_type"].unique().tolist()))
    cities = c3.multiselect("City (blank = all)", sorted(df["city"].unique().tolist()))
    statuses = c4.multiselect("Case status (blank = all)", STATUSES)
    min_score = st.slider("Minimum risk score", 0, 100, 0)

    view = df[df["risk_level"].isin(levels) & (df["risk_score"] >= min_score)]
    if ptypes:
        view = view[view["policy_type"].isin(ptypes)]
    if cities:
        view = view[view["city"].isin(cities)]
    if statuses:
        view = view[view["status"].isin(statuses)]
    view = view.sort_values("risk_score", ascending=False)

    ui.kpi_row([
        ("Alerts in Queue", f"{len(view):,}", "After applying filters"),
        ("Claim Value", ui.fmt_inr(view["claim_amount"].sum()), "Total of filtered claims", "red"),
        ("Open Cases", f"{int((view['status'] == 'Open').sum()):,}", "Awaiting assignment"),
        ("Under Investigation", f"{int((view['status'] == 'Under Investigation').sum()):,}", "Active cases"),
        ("Confirmed Fraud", f"{int((view['status'] == 'Confirmed Fraud').sum()):,}", "Closed as fraud", "green"),
    ])

    ui.section("Alert Queue (top 100 by risk score)")
    if view.empty:
        st.info("No alerts match the selected filters.")
        return
    t = view.head(100).copy()
    t["claim_amount"] = t["claim_amount"].map(ui.inr_full)
    t["risk_score"] = t["risk_score"].round(1)
    ui.html_table(t[["claim_id", "claim_date", "customer_name", "policy_type", "claim_type", "provider_name",
                     "claim_amount", "risk_score", "risk_level", "status", "top_reasons"]].rename(columns={
        "claim_id": "Claim ID", "claim_date": "Date", "customer_name": "Customer", "policy_type": "Policy",
        "claim_type": "Claim Type", "provider_name": "Provider", "claim_amount": "Amount",
        "risk_score": "Risk Score", "risk_level": "Risk Level", "status": "Status",
        "top_reasons": "Key Red Flags"}), max_height=440)
    st.download_button("Download filtered alerts (CSV)",
                       view[["claim_id", "claim_date", "customer_id", "customer_name", "policy_type", "claim_type",
                             "provider_name", "claim_amount", "risk_score", "risk_level", "status",
                             "top_reasons"]].to_csv(index=False).encode("utf-8"),
                       file_name="fraud_alerts.csv", mime="text/csv")

    ui.section("Case Investigation")
    claim_id = st.selectbox("Select a claim to investigate", view["claim_id"].head(300).tolist())
    r = df[df["claim_id"] == claim_id].iloc[0]
    g1, g2, g3 = st.columns([1, 1, 1])
    with g1:
        ui.show(ui.gauge(r["risk_score"], "Composite Risk Score"), key="alert_gauge")
        ui.html_table(pd.DataFrame({"Component": ["Classifier probability", "Isolation Forest score",
                                                  "Z-score outlier", "IQR outlier"],
                                    "Value": [f"{r['model_probability'] * 100:.1f}%", f"{r['iso_score']:.1f} / 100",
                                              "Yes" if r["z_flag"] else "No", "Yes" if r["iqr_flag"] else "No"]}),
                      max_height=220)
    with g2:
        st.markdown("**Claim details**")
        ui.kv_table({"Claim ID": r["claim_id"], "Claim date": r["claim_date"], "Policy type": r["policy_type"],
                     "Claim type": r["claim_type"], "Claim amount": ui.inr_full(r["claim_amount"]),
                     "Coverage": ui.inr_full(r["coverage_amount"]),
                     "Claim / coverage": f"{r['claim_to_coverage_ratio'] * 100:.0f}%",
                     "Days since policy start": int(r["days_since_policy_start"]),
                     "Days to report": int(r["days_to_report"]), "Documents": int(r["num_documents"]),
                     "Police report": "Yes" if r["police_report"] else "No", "Provider": r["provider_name"]})
    with g3:
        st.markdown("**Customer profile**")
        ui.kv_table({"Customer": f"{r['customer_name']} ({r['customer_id']})", "Age": f"{int(r['age'])} ({r['age_group']})",
                     "Gender": r["gender"], "City": r["city"], "Occupation": r["occupation"],
                     "Annual income": ui.inr_full(r["annual_income"]), "Credit score": int(r["credit_score"]),
                     "Prior claims": int(r["prior_claims"]), "Risk level": r["risk_level"],
                     "Current status": r["status"]})

    ui.note("<b>Red flags detected:</b> " + str(r["top_reasons"]).replace(" | ", " &bull; "))
    s1, s2 = st.columns([2, 1])
    new_status = s1.selectbox("Update case status", STATUSES, index=STATUSES.index(r["status"]) if r["status"] in STATUSES else 0)
    s2.write("")
    s2.write("")
    if s2.button("Save status"):
        db.execute("UPDATE fraud_scores SET status = ? WHERE claim_id = ?", (new_status, claim_id))
        st.cache_data.clear()
        st.session_state["flash"] = f"Status of {claim_id} updated to '{new_status}'."
        st.rerun()
