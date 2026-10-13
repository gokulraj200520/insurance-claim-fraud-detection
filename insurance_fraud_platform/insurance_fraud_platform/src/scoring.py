"""Composite fraud risk scoring: classifier + Isolation Forest + statistical outliers."""
from contextlib import closing
from datetime import datetime

import numpy as np
import pandas as pd

from . import database as db
from .anomaly import isolation_forest_scores, statistical_outliers
from .config import RISK_HIGH, RISK_MEDIUM, W_ISO, W_MODEL, W_STAT
from .features import add_provider_features
from .models import oof_probabilities


def risk_level(score: float) -> str:
    return "High" if score >= RISK_HIGH else "Medium" if score >= RISK_MEDIUM else "Low"


def explain(row) -> str:
    """Rule-based, human-readable red flags to support the investigator."""
    items = []
    if row["claim_to_coverage_ratio"] >= 0.5:
        items.append((1.0, f"Claim equals {row['claim_to_coverage_ratio'] * 100:.0f}% of insured coverage"))
    if row["days_since_policy_start"] <= 45:
        items.append((0.95, f"Filed only {int(row['days_since_policy_start'])} days after policy inception"))
    if row["days_to_report"] >= 12:
        items.append((0.85, f"Reported {int(row['days_to_report'])} days after the incident"))
    if row["prior_claims"] >= 3:
        items.append((0.8, f"{int(row['prior_claims'])} previous claims on record"))
    if row["num_documents"] <= 1:
        items.append((0.75, f"Only {int(row['num_documents'])} supporting document(s) submitted"))
    if row["police_report"] == 0 and row["policy_type"] in ("Auto", "Home"):
        items.append((0.6, "No police report filed"))
    if row.get("z_flag", 0):
        items.append((0.9, "Claim values are statistical outliers for the policy type"))
    if row.get("iso_score", 0) >= 70:
        items.append((0.7, "Unusual overall pattern flagged by Isolation Forest"))
    if not items:
        return "No significant red flags"
    items.sort(key=lambda t: -t[0])
    return " | ".join(text for _, text in items[:4])


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = add_provider_features(df)
    prob = oof_probabilities(df)
    iso, _ = isolation_forest_scores(df)
    stat = statistical_outliers(df)
    stat_component = (stat["z_flag"].astype(int) + stat["iqr_flag"].astype(int)) / 2 * 100
    risk = np.clip(W_MODEL * prob * 100 + W_ISO * iso + W_STAT * stat_component, 0, 100)

    work = df.copy()
    work["model_probability"] = prob
    work["iso_score"] = iso
    work["z_flag"] = stat["z_flag"].astype(int).values
    work["iqr_flag"] = stat["iqr_flag"].astype(int).values
    work["risk_score"] = risk.round(2)
    work["risk_level"] = work["risk_score"].map(risk_level)
    work["top_reasons"] = work.apply(explain, axis=1)
    work["status"] = np.where(work["risk_level"] == "Low", "Auto-Cleared", "Open")
    work["scored_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return work[["claim_id", "model_probability", "iso_score", "z_flag", "iqr_flag", "risk_score",
                 "risk_level", "top_reasons", "status", "scored_at"]]


def score_all_claims() -> dict:
    base = db.query("SELECT * FROM vw_claims_full")
    scores = compute_scores(base)
    with closing(db.get_connection()) as conn:
        conn.execute("DELETE FROM fraud_scores")
        scores.to_sql("fraud_scores", conn, if_exists="append", index=False)
        conn.commit()
    return {"claims_scored": len(scores),
            "high_risk": int((scores["risk_level"] == "High").sum()),
            "medium_risk": int((scores["risk_level"] == "Medium").sum()),
            "low_risk": int((scores["risk_level"] == "Low").sum())}
