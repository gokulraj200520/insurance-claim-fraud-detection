"""Customer profiling: individual risk profile and K-Means behavioural segmentation."""
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src import database as db
from src.config import RANDOM_STATE

from . import data_access as da
from . import ui

SEGMENT_NAMES = ["Segment A - Lowest Risk", "Segment B - Moderate", "Segment C - Elevated", "Segment D - Critical"]


@st.cache_data(show_spinner=False)
def customer_table(version: float) -> pd.DataFrame:
    df = da.claims(version)
    cust = df.groupby("customer_id").agg(
        customer_name=("customer_name", "first"), claims=("claim_id", "count"),
        total_claimed=("claim_amount", "sum"), avg_ratio=("claim_to_coverage_ratio", "mean"),
        avg_risk=("risk_score", "mean"), max_risk=("risk_score", "max"),
        high_risk_claims=("risk_level", lambda s: int((s == "High").sum())),
        credit_score=("credit_score", "first"), age=("age", "first"), city=("city", "first")).reset_index()
    X = StandardScaler().fit_transform(cust[["claims", "avg_ratio", "avg_risk", "credit_score"]])
    km = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit(X)
    cust["cluster"] = km.labels_
    order = cust.groupby("cluster")["avg_risk"].mean().sort_values().index.tolist()
    cust["Segment"] = cust["cluster"].map({c: SEGMENT_NAMES[i] for i, c in enumerate(order)})
    return cust


def render() -> None:
    ui.page_header("Customer Profiling", "Individual risk profiles and behavioural customer segmentation")
    v = db.db_version()
    df, cust = da.claims(v), customer_table(v)

    ui.section("Customer Risk Profile")
    ranked = cust.sort_values("max_risk", ascending=False)
    labels = {r.customer_id: f"{r.customer_id} - {r.customer_name} (max risk {r.max_risk:.0f})" for r in ranked.itertuples()}
    cid = st.selectbox("Select customer (sorted by highest claim risk)", ranked["customer_id"].tolist(),
                       format_func=lambda x: labels[x])
    c = df[df["customer_id"] == cid].sort_values("claim_date")
    p = c.iloc[0]
    seg = cust.loc[cust["customer_id"] == cid, "Segment"].iloc[0]
    ui.kpi_row([
        ("Customer Segment", seg.split(" - ")[1], seg.split(" - ")[0]),
        ("Total Claims", len(c), f"Policy {p['policy_id']} ({p['policy_type']})"),
        ("Total Claimed", ui.fmt_inr(c["claim_amount"].sum()), f"Coverage {ui.fmt_inr(p['coverage_amount'])}"),
        ("Highest Risk Score", f"{c['risk_score'].max():.1f}", f"Average {c['risk_score'].mean():.1f}", "red"),
        ("High-Risk Claims", int((c["risk_level"] == "High").sum()), "Claims rated High"),
    ])
    a, b = st.columns([1, 2])
    with a:
        ui.kv_table({"Name": p["customer_name"], "Age": f"{int(p['age'])} ({p['age_group']})", "Gender": p["gender"],
                     "City": p["city"], "Occupation": p["occupation"], "Annual income": ui.inr_full(p["annual_income"]),
                     "Income band": p["income_band"], "Credit score": int(p["credit_score"]),
                     "Policy start": p["policy_start_date"], "Premium": ui.inr_full(p["premium"])})
    with b:
        sc = px.scatter(c, x="claim_date", y="claim_amount", color="risk_level", size="risk_score", size_max=28,
                        color_discrete_map=ui.RISK_COLORS, category_orders={"risk_level": ui.RISK_ORDER},
                        labels={"claim_date": "Claim date", "claim_amount": "Claim amount (INR)", "risk_level": "Risk level"})
        ui.show(ui.style_fig(sc, 330, "Claim history (bubble size = risk score)"), key="prof_hist")
    t = c[["claim_id", "claim_date", "claim_type", "claim_amount", "risk_score", "risk_level", "status"]].copy()
    t["claim_amount"] = t["claim_amount"].map(ui.inr_full)
    t["risk_score"] = t["risk_score"].round(1)
    ui.html_table(t.rename(columns={"claim_id": "Claim ID", "claim_date": "Date", "claim_type": "Claim Type",
                                    "claim_amount": "Amount", "risk_score": "Risk Score", "risk_level": "Risk Level",
                                    "status": "Status"}), max_height=240)

    ui.section("Behavioural Segmentation (K-Means, k = 4)")
    c1, c2 = st.columns([3, 2])
    sample = cust.sample(min(len(cust), 1500), random_state=RANDOM_STATE)
    scat = px.scatter(sample, x="avg_ratio", y="avg_risk", color="Segment", opacity=0.75,
                      category_orders={"Segment": SEGMENT_NAMES},
                      color_discrete_sequence=["#2E7D32", "#4C8BC2", "#E0A100", "#B3261E"],
                      labels={"avg_ratio": "Average claim / coverage ratio", "avg_risk": "Average risk score"})
    with c1:
        ui.show(ui.style_fig(scat, 400, "Customer segments"), key="prof_seg")
    summ = cust.groupby("Segment").agg(Customers=("customer_id", "count"), Avg_Claims=("claims", "mean"),
                                       Avg_Ratio=("avg_ratio", "mean"), Avg_Risk=("avg_risk", "mean"),
                                       Credit=("credit_score", "mean")).reset_index().sort_values("Segment")
    summ = summ.rename(columns={"Avg_Claims": "Avg Claims", "Avg_Ratio": "Avg Ratio", "Avg_Risk": "Avg Risk",
                                "Credit": "Avg Credit Score"})
    with c2:
        st.markdown("**Segment summary**")
        ui.html_table(summ.round(2), max_height=300)

    c3, c4 = st.columns(2)
    for col, dim, key, label in [(c3, "age_group", "prof_age", "Age group"), (c4, "income_band", "prof_inc", "Income band")]:
        g = df.groupby(dim).agg(rate=("risk_level", lambda s: (s == "High").mean() * 100)).reset_index()
        bar = px.bar(g, x=dim, y="rate", labels={dim: label, "rate": "High-risk claims (%)"},
                     color_discrete_sequence=[ui.NAVY], text_auto=".1f")
        with col:
            ui.show(ui.style_fig(bar, 340, f"High-risk claim rate by {label.lower()}"), key=key)
