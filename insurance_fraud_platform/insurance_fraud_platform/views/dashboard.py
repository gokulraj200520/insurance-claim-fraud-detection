"""Executive dashboard: portfolio KPIs, trends and risk distribution."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import database as db

from . import data_access as da
from . import ui


def render() -> None:
    ui.page_header("Executive Dashboard", "Portfolio-wide view of claims, risk exposure and fraud alerts")
    df = da.claims(db.db_version())

    f1, f2, _ = st.columns([1, 1, 1])
    years = sorted(df["year"].unique().tolist())
    types = sorted(df["policy_type"].unique().tolist())
    sel_years = f1.multiselect("Year", years, default=years)
    sel_types = f2.multiselect("Policy type", types, default=types)
    df = df[df["year"].isin(sel_years) & df["policy_type"].isin(sel_types)]
    if df.empty:
        st.warning("No claims match the selected filters.")
        return

    high = df["risk_level"] == "High"
    ui.kpi_row([
        ("Total Claims", f"{len(df):,}", f"{df['year'].min()} - {df['year'].max()}"),
        ("Total Claim Value", ui.fmt_inr(df["claim_amount"].sum()), f"Average {ui.fmt_inr(df['claim_amount'].mean())}"),
        ("High-Risk Alerts", f"{int(high.sum()):,}", f"{high.mean() * 100:.1f}% of all claims", "red"),
        ("Amount at Risk", ui.fmt_inr(df.loc[high, "claim_amount"].sum()), "Value of high-risk claims", "red"),
        ("Average Risk Score", f"{df['risk_score'].mean():.1f}", "Scale 0 - 100"),
    ])

    ui.section("Claim Volume and Fraud Alert Trend")
    c1, c2 = st.columns([2, 1])
    m = df.groupby("year_month").agg(claims=("claim_id", "count"),
                                      alerts=("risk_level", lambda s: int((s == "High").sum()))).reset_index()
    m["month"] = pd.to_datetime(m["year_month"] + "-01")
    fig = go.Figure()
    fig.add_bar(x=m["month"], y=m["claims"], name="Total claims", marker_color=ui.LIGHT_BLUE)
    fig.add_scatter(x=m["month"], y=m["alerts"], name="High-risk alerts", mode="lines+markers",
                    line=dict(color=ui.RISK_COLORS["High"], width=2.5), yaxis="y2")
    fig.update_layout(yaxis=dict(title="Claims"),
                      yaxis2=dict(title="High-risk alerts", overlaying="y", side="right", showgrid=False))
    with c1:
        ui.show(ui.style_fig(fig, 380, "Monthly claims and high-risk alerts"), key="dash_trend")
    lv = df["risk_level"].value_counts().reindex(ui.RISK_ORDER).fillna(0).reset_index()
    lv.columns = ["Risk level", "Claims"]
    donut = px.pie(lv, names="Risk level", values="Claims", hole=0.55, color="Risk level",
                   color_discrete_map=ui.RISK_COLORS)
    donut.update_traces(textinfo="percent+label", sort=False)
    with c2:
        ui.show(ui.style_fig(donut, 380, "Risk level distribution"), key="dash_donut")

    ui.section("Risk Breakdown by Segment")
    c3, c4 = st.columns(2)
    ct = df.groupby(["policy_type", "risk_level"]).size().reset_index(name="Claims")
    bar = px.bar(ct, x="policy_type", y="Claims", color="risk_level", barmode="stack",
                 color_discrete_map=ui.RISK_COLORS, category_orders={"risk_level": ui.RISK_ORDER},
                 labels={"policy_type": "Policy type", "risk_level": "Risk level"})
    with c3:
        ui.show(ui.style_fig(bar, 360, "Claims by policy type and risk level"), key="dash_policy")
    city = df.groupby("city").agg(rate=("risk_level", lambda s: (s == "High").mean() * 100)).reset_index()
    city = city.sort_values("rate")
    cbar = px.bar(city, x="rate", y="city", orientation="h", labels={"rate": "High-risk claims (%)", "city": ""},
                  color_discrete_sequence=[ui.NAVY])
    with c4:
        ui.show(ui.style_fig(cbar, 360, "High-risk claim rate by city"), key="dash_city")

    c5, c6 = st.columns(2)
    pv = df.pivot_table(index="city", columns="policy_type", values="risk_score", aggfunc="mean").round(1)
    heat = px.imshow(pv, text_auto=".1f", aspect="auto", color_continuous_scale=["#FFFFFF", "#9DB7D5", "#1F3A5F"],
                     labels=dict(color="Avg risk"))
    with c5:
        ui.show(ui.style_fig(heat, 420, "Average risk score: city x policy type"), key="dash_heat")
    ptype = df.groupby("provider_type").agg(avg=("risk_score", "mean")).reset_index().sort_values("avg", ascending=False)
    pbar = px.bar(ptype, x="provider_type", y="avg", labels={"provider_type": "Provider type", "avg": "Average risk score"},
                  color_discrete_sequence=[ui.BLUE], text_auto=".1f")
    with c6:
        ui.show(ui.style_fig(pbar, 420, "Average risk score by provider type"), key="dash_provider")

    ui.section("Top 10 Highest-Risk Claims")
    top = df.sort_values("risk_score", ascending=False).head(10).copy()
    top["claim_amount"] = top["claim_amount"].map(ui.inr_full)
    top["risk_score"] = top["risk_score"].round(1)
    ui.html_table(top[["claim_id", "claim_date", "customer_name", "policy_type", "claim_type", "claim_amount",
                       "risk_score", "risk_level", "top_reasons"]].rename(columns={
        "claim_id": "Claim ID", "claim_date": "Date", "customer_name": "Customer", "policy_type": "Policy",
        "claim_type": "Claim Type", "claim_amount": "Amount", "risk_score": "Risk Score",
        "risk_level": "Risk Level", "top_reasons": "Key Red Flags"}), max_height=420)
