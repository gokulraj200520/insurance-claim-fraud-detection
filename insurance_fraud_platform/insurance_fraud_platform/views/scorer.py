"""Interactive claim scorer: estimate the fraud risk of a new claim with the trained model."""
import pandas as pd
import streamlit as st

from src import database as db
from src.config import RISK_HIGH, RISK_MEDIUM
from src.features import POLICY_TYPES, build_features
from src.scoring import explain, risk_level

from . import data_access as da
from . import ui


def render() -> None:
    ui.page_header("Claim Risk Scorer", "Score a new claim in real time using the trained Random Forest classifier")
    v = db.db_version()
    df = da.claims(v)
    model = da.trained(v)["final_model"]

    with st.container():
        c1, c2, c3 = st.columns(3)
        ptype = c1.selectbox("Policy type", POLICY_TYPES)
        coverage = c2.number_input("Coverage amount (INR)", 100_000, 10_000_000, 1_000_000, 50_000)
        amount = c3.number_input("Claim amount (INR)", 1_000, 10_000_000, 400_000, 10_000)
        c4, c5, c6 = st.columns(3)
        days_start = c4.slider("Days since policy start", 0, 2000, 30)
        days_report = c5.slider("Days taken to report the claim", 0, 60, 3)
        docs = c6.slider("Supporting documents submitted", 0, 10, 4)
        c7, c8, c9 = st.columns(3)
        prior = c7.slider("Previous claims", 0, 8, 0)
        police = c8.selectbox("Police report filed", ["Yes", "No"])
        witness = c9.selectbox("Witness present", ["Yes", "No"])
        c10, c11, c12 = st.columns(3)
        age = c10.slider("Customer age", 18, 80, 40)
        credit = c11.slider("Credit score", 300, 900, 700)
        income = c12.number_input("Annual income (INR)", 100_000, 10_000_000, 600_000, 50_000)

    row = pd.DataFrame([{
        "claim_amount": amount, "claim_to_coverage_ratio": min(amount / coverage, 1.5),
        "days_since_policy_start": days_start, "days_to_report": days_report, "num_documents": docs,
        "police_report": int(police == "Yes"), "witness_present": int(witness == "Yes"), "prior_claims": prior,
        "age": age, "credit_score": credit, "annual_income": income,
        "premium": coverage * 0.025, "coverage_amount": coverage, "is_weekend": 0,
        "provider_claim_count": df["provider_claim_count"].median(),
        "provider_avg_ratio": df["provider_avg_ratio"].median(), "policy_type": ptype,
    }])
    prob = float(model.predict_proba(build_features(row))[:, 1][0])
    score = prob * 100
    level = risk_level(score)
    row["z_flag"], row["iso_score"] = 0, 0

    ui.section("Assessment")
    a, b = st.columns([1, 2])
    with a:
        ui.show(ui.gauge(score, "Model Fraud Probability (%)"), key="scorer_gauge")
    with b:
        ui.kpi_row([("Risk Level", level, f"High >= {RISK_HIGH}, Medium >= {RISK_MEDIUM}", "red" if level == "High" else ""),
                    ("Fraud Probability", f"{prob * 100:.1f}%", "Random Forest estimate"),
                    ("Claim / Coverage", f"{amount / coverage * 100:.0f}%", "Claimed share of cover")])
        ui.note("<b>Red flags:</b> " + explain(row.iloc[0]).replace(" | ", " &bull; "))
        st.caption("This quick score uses the classifier only. Provider-specific features default to portfolio medians. "
                   "The full composite score on the alert pages also blends Isolation Forest and statistical outlier signals.")
