"""ETL pipeline: Extract (CSV) -> Transform (clean, standardise, derive) -> Load (star schema)."""
from contextlib import closing
from datetime import datetime

import numpy as np
import pandas as pd

from .config import RAW_CSV_PATH
from .database import get_connection, init_schema

CATEGORY_MAP = {"Auto": "Vehicle", "Health": "Medical", "Home": "Property", "Life": "Life"}
AGE_BINS = [17, 25, 35, 45, 55, 65, 120]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
INCOME_BINS = [0, 300_000, 600_000, 1_000_000, np.inf]
INCOME_LABELS = ["Low", "Lower-Middle", "Upper-Middle", "High"]


# --------------------------------------------------------------------------- Extract
def extract(path=RAW_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=True)


# --------------------------------------------------------------------------- Transform
def _norm_gender(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    return "Male" if x.startswith("m") else "Female" if x.startswith("f") else np.nan


def _norm_flag(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in {"y", "yes", "true", "1"}:
        return 1
    if x in {"n", "no", "false", "0"}:
        return 0
    return np.nan


def _parse_dates(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        out[bad] = pd.to_datetime(s[bad], format="%d/%m/%Y", errors="coerce")
    return out


def transform(raw: pd.DataFrame):
    stats = {"rows_extracted": len(raw)}
    df = raw.copy()

    # 1. remove exact duplicate claim records
    n0 = len(df)
    df = df.drop_duplicates(subset="claim_id", keep="first").reset_index(drop=True)
    stats["duplicates_removed"] = n0 - len(df)

    # 2. standardise text, codes and data types
    for col in ["customer_name", "city", "occupation", "provider_name", "claim_type", "provider_type"]:
        df[col] = df[col].str.strip()
    df["policy_type"] = df["policy_type"].str.strip().str.title()
    df["gender"] = df["gender"].map(_norm_gender)
    df["police_report"] = df["police_report"].map(_norm_flag)
    df["witness_present"] = df["witness_present"].map(_norm_flag)
    df["claim_date"] = _parse_dates(df["claim_date"])
    df["policy_start_date"] = _parse_dates(df["policy_start_date"])
    numeric = ["age", "annual_income", "credit_score", "premium", "coverage_amount", "claim_amount",
               "days_to_report", "num_documents", "prior_claims", "is_fraud"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. invalid values -> treated as missing; unusable rows dropped
    invalid = (~df["age"].between(18, 90) & df["age"].notna()).sum()
    df.loc[~df["age"].between(18, 90), "age"] = np.nan
    invalid += (~df["credit_score"].between(300, 900) & df["credit_score"].notna()).sum()
    df.loc[~df["credit_score"].between(300, 900), "credit_score"] = np.nan
    stats["invalid_values_fixed"] = int(invalid)
    n0 = len(df)
    df = df[df["claim_amount"].gt(0) & df["claim_date"].notna() & df["policy_start_date"].notna()].copy()
    stats["invalid_rows_dropped"] = n0 - len(df)

    # 4. missing-value imputation
    fill_cols = ["age", "gender", "city", "occupation", "annual_income", "credit_score",
                 "num_documents", "days_to_report", "police_report", "witness_present", "prior_claims"]
    stats["missing_values_imputed"] = int(df[fill_cols].isna().sum().sum())
    for col in ["age", "gender", "city", "occupation", "annual_income", "credit_score"]:   # customer-level
        df[col] = df[col].fillna(df.groupby("customer_id")[col].transform("first"))
    for col in ["num_documents", "days_to_report"]:                                         # claim-level
        df[col] = df[col].fillna(df.groupby("policy_type")[col].transform("median"))
    for col in fill_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # 5. derived (feature-engineered) attributes
    df["claim_to_coverage_ratio"] = (df["claim_amount"] / df["coverage_amount"]).round(4)
    df["days_since_policy_start"] = (df["claim_date"] - df["policy_start_date"]).dt.days.clip(lower=0)
    df["age"] = df["age"].round().astype(int)
    df["age_group"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    df["income_band"] = pd.cut(df["annual_income"], INCOME_BINS, labels=INCOME_LABELS).astype(str)
    df["claim_category"] = df["policy_type"].map(CATEGORY_MAP)
    for col in ["credit_score", "days_to_report", "num_documents", "police_report", "witness_present",
                "prior_claims", "is_fraud", "days_since_policy_start"]:
        df[col] = df[col].round().astype(int)

    stats["rows_loaded"] = len(df)
    return df.reset_index(drop=True), stats


# --------------------------------------------------------------------------- Load
def _build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(df["claim_date"].min(), df["claim_date"].max(), freq="D")
    dd = pd.DataFrame({"full_date": dates})
    dd["date_key"] = dates.strftime("%Y%m%d").astype(int)
    dd["day"] = dates.day
    dd["month"] = dates.month
    dd["month_name"] = dates.month_name()
    dd["quarter"] = dates.quarter
    dd["year"] = dates.year
    dd["day_name"] = dates.day_name()
    dd["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    dd["full_date"] = dates.strftime("%Y-%m-%d")
    return dd


def load(df: pd.DataFrame, conn) -> None:
    _build_dim_date(df).to_sql("dim_date", conn, if_exists="append", index=False)

    dim_customer = df.drop_duplicates("customer_id")[
        ["customer_id", "customer_name", "age", "age_group", "gender", "city", "occupation",
         "annual_income", "income_band", "credit_score"]]
    dim_customer.to_sql("dim_customer", conn, if_exists="append", index=False)

    dim_policy = df.drop_duplicates("policy_id")[
        ["policy_id", "policy_type", "premium", "coverage_amount", "policy_start_date"]].rename(
        columns={"policy_start_date": "start_date"})
    dim_policy["start_date"] = dim_policy["start_date"].dt.strftime("%Y-%m-%d")
    dim_policy.to_sql("dim_policy", conn, if_exists="append", index=False)

    df.drop_duplicates("provider_id")[["provider_id", "provider_name", "provider_type"]].to_sql(
        "dim_provider", conn, if_exists="append", index=False)
    df.drop_duplicates("claim_type")[["claim_type", "claim_category"]].to_sql(
        "dim_claim_type", conn, if_exists="append", index=False)

    keys = {
        "customer_id": pd.read_sql_query("SELECT customer_key, customer_id FROM dim_customer", conn),
        "policy_id": pd.read_sql_query("SELECT policy_key, policy_id FROM dim_policy", conn),
        "provider_id": pd.read_sql_query("SELECT provider_key, provider_id FROM dim_provider", conn),
        "claim_type": pd.read_sql_query("SELECT claim_type_key, claim_type FROM dim_claim_type", conn),
    }
    fact = df.copy()
    for col, k in keys.items():
        fact = fact.merge(k, on=col, how="left")
    fact["date_key"] = fact["claim_date"].dt.strftime("%Y%m%d").astype(int)
    fact[["claim_id", "date_key", "customer_key", "policy_key", "provider_key", "claim_type_key",
          "claim_amount", "claim_to_coverage_ratio", "days_since_policy_start", "days_to_report",
          "num_documents", "police_report", "witness_present", "prior_claims", "is_fraud"]].to_sql(
        "fact_claims", conn, if_exists="append", index=False)


def run_etl(csv_path=RAW_CSV_PATH) -> dict:
    raw = extract(csv_path)
    clean, stats = transform(raw)
    with closing(get_connection()) as conn:
        init_schema(conn)
        raw.to_sql("stg_raw_claims", conn, if_exists="replace", index=False)
        load(clean, conn)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = [(stamp, "Extract", "rows_extracted", stats["rows_extracted"]),
               (stamp, "Transform", "duplicates_removed", stats["duplicates_removed"]),
               (stamp, "Transform", "invalid_values_fixed", stats["invalid_values_fixed"]),
               (stamp, "Transform", "invalid_rows_dropped", stats["invalid_rows_dropped"]),
               (stamp, "Transform", "missing_values_imputed", stats["missing_values_imputed"]),
               (stamp, "Load", "rows_loaded", stats["rows_loaded"])]
        conn.executemany("INSERT INTO etl_log (run_at, stage, metric, value) VALUES (?,?,?,?)", log)
        conn.commit()
    return stats
