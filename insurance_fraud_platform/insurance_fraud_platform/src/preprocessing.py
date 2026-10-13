"""Data-quality reporting and dimensionality reduction (PCA)."""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import database as db
from .config import RANDOM_STATE

QUALITY_COLUMNS = ["age", "gender", "city", "occupation", "credit_score", "num_documents",
                   "days_to_report", "police_report", "witness_present"]


def data_quality_report() -> pd.DataFrame:
    """Missing values in the raw staging table versus the cleaned warehouse view."""
    raw = db.query("SELECT * FROM stg_raw_claims")
    raw = raw.drop_duplicates("claim_id")
    clean = db.query("SELECT * FROM vw_claims_full")
    rows = []
    for col in QUALITY_COLUMNS:
        rows.append({"Attribute": col, "Missing (raw)": int(raw[col].isna().sum()),
                     "Missing (warehouse)": int(clean[col].isna().sum())})
    return pd.DataFrame(rows)


def run_pca(X: pd.DataFrame, n_components=None):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    comps = pca.fit_transform(Xs)
    return pca, comps
