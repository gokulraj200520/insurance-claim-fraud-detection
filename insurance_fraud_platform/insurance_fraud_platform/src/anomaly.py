"""Unsupervised anomaly detection: Isolation Forest and statistical outlier tests."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import FRAUD_RATE, RANDOM_STATE

ANOMALY_FEATURES = ["claim_amount", "claim_to_coverage_ratio", "days_since_policy_start",
                    "days_to_report", "num_documents", "prior_claims", "credit_score"]


def isolation_forest_scores(df: pd.DataFrame, contamination: float = FRAUD_RATE):
    """Returns (score 0-100 where higher = more anomalous, boolean flag)."""
    X = StandardScaler().fit_transform(df[ANOMALY_FEATURES])
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X)
    raw = -iso.score_samples(X)
    scaled = (raw - raw.min()) / (raw.max() - raw.min()) * 100
    return scaled, iso.predict(X) == -1


def statistical_outliers(df: pd.DataFrame, z_threshold: float = 3.0, iqr_k: float = 1.5) -> pd.DataFrame:
    """Z-score and IQR tests, computed within each policy type."""
    out = pd.DataFrame(index=df.index)
    grp = df.groupby("policy_type")

    def z(col):
        return grp[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))

    out["z_amount"] = z("claim_amount")
    out["z_ratio"] = z("claim_to_coverage_ratio")
    out["z_report"] = z("days_to_report")
    out["z_max"] = out[["z_amount", "z_ratio", "z_report"]].max(axis=1)
    out["z_flag"] = out["z_max"] > z_threshold

    def fence(col):
        q1 = grp[col].transform(lambda s: s.quantile(0.25))
        q3 = grp[col].transform(lambda s: s.quantile(0.75))
        return q3 + iqr_k * (q3 - q1)

    out["iqr_flag"] = (df["claim_to_coverage_ratio"] > fence("claim_to_coverage_ratio")) | \
                      (df["days_to_report"] > fence("days_to_report")) | \
                      (df["claim_amount"] > fence("claim_amount"))
    return out
