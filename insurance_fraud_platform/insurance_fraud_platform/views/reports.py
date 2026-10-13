"""Risk reports: segment analysis, provider risk, monthly report and written summary."""
import plotly.express as px
import streamlit as st

from src import database as db
from src import olap

from . import data_access as da
from . import ui

SEGMENTS = ["Policy Type", "Claim Type", "City", "Age Group", "Income Band", "Provider Type", "Day of Week"]


def render() -> None:
    ui.page_header("Risk Reports", "Management reports on where fraud risk is concentrated")
    df = da.claims(db.db_version())

    ui.section("Key Findings")
    seg = lambda col: df.groupby(col)["risk_score"].mean().sort_values(ascending=False)
    pt, ct, pr = seg("policy_type"), seg("city"), seg("provider_type")
    month = df.groupby("year_month")["risk_level"].apply(lambda s: (s == "High").sum()).sort_values(ascending=False)
    high = df[df["risk_level"] == "High"]
    st.markdown(
        f"- **{len(high):,}** of **{len(df):,}** claims ({len(high) / len(df) * 100:.1f}%) are rated *High* risk, "
        f"representing **{ui.fmt_inr(high['claim_amount'].sum())}** of claimed value.\n"
        f"- **{pt.index[0]}** policies carry the highest average risk score ({pt.iloc[0]:.1f}); "
        f"**{pt.index[-1]}** the lowest ({pt.iloc[-1]:.1f}).\n"
        f"- **{ct.index[0]}** is the highest-risk city ({ct.iloc[0]:.1f}); **{pr.index[0]}** is the highest-risk provider type.\n"
        f"- The peak month for high-risk alerts was **{month.index[0]}** with **{int(month.iloc[0])}** alerts.")

    tab1, tab2, tab3 = st.tabs(["Segment Analysis", "Provider Risk", "Monthly Report"])

    with tab1:
        dim = st.selectbox("Analyse risk by", SEGMENTS)
        res, _ = olap.cube_query([dim], ["Total Claims", "High-Risk Alerts", "Average Risk Score", "Fraud Rate (%)"])
        res["High-Risk Rate (%)"] = (res["High-Risk Alerts"] / res["Total Claims"] * 100).round(2)
        c1, c2 = st.columns(2)
        b1 = px.bar(res.sort_values("High-Risk Rate (%)"), x="High-Risk Rate (%)", y=dim, orientation="h",
                    color_discrete_sequence=[ui.NAVY], text_auto=".1f")
        with c1:
            ui.show(ui.style_fig(b1, 380, f"High-risk claim rate by {dim.lower()}"), key="rep_seg1")
        b2 = px.bar(res, x=dim, y="Average Risk Score", color_discrete_sequence=[ui.BLUE], text_auto=".1f")
        with c2:
            ui.show(ui.style_fig(b2, 380, f"Average risk score by {dim.lower()}"), key="rep_seg2")
        ui.html_table(res.sort_values("High-Risk Rate (%)", ascending=False), max_height=360)

    with tab2:
        pdf = df.groupby(["provider_id", "provider_name", "provider_type"]).agg(
            Claims=("claim_id", "count"), High_Risk=("risk_level", lambda s: int((s == "High").sum())),
            Avg_Risk=("risk_score", "mean"), Confirmed_Fraud=("is_fraud", "sum"),
            Claimed=("claim_amount", "sum")).reset_index()
        pdf = pdf[pdf["Claims"] >= 10].copy()
        pdf["High-Risk Rate (%)"] = (pdf["High_Risk"] / pdf["Claims"] * 100).round(1)
        pdf = pdf.sort_values(["High-Risk Rate (%)", "High_Risk"], ascending=False)
        top = pdf.head(15).sort_values("High-Risk Rate (%)")
        b = px.bar(top, x="High-Risk Rate (%)", y="provider_name", orientation="h",
                   color_discrete_sequence=[ui.RISK_COLORS["High"]], labels={"provider_name": ""}, text_auto=".1f")
        ui.show(ui.style_fig(b, 480, "Top 15 providers by high-risk claim rate (min. 10 claims)"), key="rep_prov")
        t = pdf.head(20).rename(columns={"provider_id": "Provider ID", "provider_name": "Provider",
                                         "provider_type": "Type", "High_Risk": "High-Risk Claims",
                                         "Avg_Risk": "Avg Risk Score", "Confirmed_Fraud": "Confirmed Fraud"})
        t["Claimed"] = t["Claimed"].map(ui.fmt_inr)
        t["Avg Risk Score"] = t["Avg Risk Score"].round(1)
        ui.html_table(t, max_height=420)

    with tab3:
        mm = df.groupby("year_month").agg(Claims=("claim_id", "count"), Alerts=("risk_level", lambda s: int((s == "High").sum())),
                                          Avg_Risk=("risk_score", "mean"), Value=("claim_amount", "sum")).reset_index()
        mm["Alert Rate (%)"] = (mm["Alerts"] / mm["Claims"] * 100).round(2)
        line = px.line(mm, x="year_month", y="Alert Rate (%)", markers=True, color_discrete_sequence=[ui.NAVY],
                       labels={"year_month": "Month"})
        ui.show(ui.style_fig(line, 360, "Monthly high-risk alert rate"), key="rep_month")
        out = mm.rename(columns={"year_month": "Month", "Avg_Risk": "Avg Risk Score", "Value": "Claim Value"})
        out["Avg Risk Score"] = out["Avg Risk Score"].round(1)
        csv = out.to_csv(index=False).encode("utf-8")
        out["Claim Value"] = out["Claim Value"].map(ui.fmt_inr)
        ui.html_table(out.sort_values("Month", ascending=False), max_height=380)
        st.download_button("Download monthly risk report (CSV)", csv, file_name="monthly_risk_report.csv", mime="text/csv")
