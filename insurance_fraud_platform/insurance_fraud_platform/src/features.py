"""Feature engineering shared by PCA, classification, anomaly detection and the claim scorer."""
import pandas as pd

from . import database as db

POLICY_TYPES = ["Auto", "Health", "Home", "Life"]
NUMERIC_FEATURES = [
    "claim_amount", "claim_to_coverage_ratio", "days_since_policy_start", "days_to_report",
    "num_documents", "police_report", "witness_present", "prior_claims", "age", "credit_score",
    "annual_income", "premium", "coverage_amount", "is_weekend", "provider_claim_count",
    "provider_avg_ratio",
]
FEATURE_NAMES = NUMERIC_FEATURES + [f"policy_{p}" for p in POLICY_TYPES]

FEATURE_LABELS = {
    "claim_amount": "Claim amount", "claim_to_coverage_ratio": "Claim / coverage ratio",
    "days_since_policy_start": "Days since policy start", "days_to_report": "Days to report",
    "num_documents": "Supporting documents", "police_report": "Police report filed",
    "witness_present": "Witness present", "prior_claims": "Prior claims", "age": "Customer age",
    "credit_score": "Credit score", "annual_income": "Annual income", "premium": "Premium",
    "coverage_amount": "Coverage amount", "is_weekend": "Weekend claim",
    "provider_claim_count": "Provider claim volume", "provider_avg_ratio": "Provider avg. ratio",
    "policy_Auto": "Policy: Auto", "policy_Health": "Policy: Health",
    "policy_Home": "Policy: Home", "policy_Life": "Policy: Life",
}


def add_provider_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("provider_id")
    df["provider_claim_count"] = g["claim_id"].transform("count")
    df["provider_avg_ratio"] = g["claim_to_coverage_ratio"].transform("mean")
    return df


def load_claims_frame() -> pd.DataFrame:
    return add_provider_features(db.query("SELECT * FROM vw_claims_full"))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES].copy()
    for p in POLICY_TYPES:
        X[f"policy_{p}"] = (df["policy_type"] == p).astype(int)
    return X.astype(float)[FEATURE_NAMES]
