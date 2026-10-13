"""SQLite data warehouse: star schema DDL and small query helpers."""
import sqlite3
from contextlib import closing

import pandas as pd

from .config import DATA_DIR, DB_PATH

SCHEMA_SQL = """
DROP VIEW  IF EXISTS vw_claims_full;
DROP TABLE IF EXISTS fraud_scores;
DROP TABLE IF EXISTS fact_claims;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_policy;
DROP TABLE IF EXISTS dim_provider;
DROP TABLE IF EXISTS dim_claim_type;
DROP TABLE IF EXISTS etl_log;

CREATE TABLE dim_date (
    date_key     INTEGER PRIMARY KEY,
    full_date    TEXT NOT NULL,
    day          INTEGER,
    month        INTEGER,
    month_name   TEXT,
    quarter      INTEGER,
    year         INTEGER,
    day_name     TEXT,
    is_weekend   INTEGER
);

CREATE TABLE dim_customer (
    customer_key  INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   TEXT UNIQUE NOT NULL,
    customer_name TEXT,
    age           INTEGER,
    age_group     TEXT,
    gender        TEXT,
    city          TEXT,
    occupation    TEXT,
    annual_income REAL,
    income_band   TEXT,
    credit_score  INTEGER
);

CREATE TABLE dim_policy (
    policy_key      INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id       TEXT UNIQUE NOT NULL,
    policy_type     TEXT,
    premium         REAL,
    coverage_amount REAL,
    start_date      TEXT
);

CREATE TABLE dim_provider (
    provider_key  INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id   TEXT UNIQUE NOT NULL,
    provider_name TEXT,
    provider_type TEXT
);

CREATE TABLE dim_claim_type (
    claim_type_key INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_type     TEXT UNIQUE NOT NULL,
    claim_category TEXT
);

CREATE TABLE fact_claims (
    claim_key               INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id                TEXT UNIQUE NOT NULL,
    date_key                INTEGER NOT NULL REFERENCES dim_date(date_key),
    customer_key            INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    policy_key              INTEGER NOT NULL REFERENCES dim_policy(policy_key),
    provider_key            INTEGER NOT NULL REFERENCES dim_provider(provider_key),
    claim_type_key          INTEGER NOT NULL REFERENCES dim_claim_type(claim_type_key),
    claim_amount            REAL,
    claim_to_coverage_ratio REAL,
    days_since_policy_start INTEGER,
    days_to_report          INTEGER,
    num_documents           INTEGER,
    police_report           INTEGER,
    witness_present         INTEGER,
    prior_claims            INTEGER,
    is_fraud                INTEGER
);

CREATE TABLE fraud_scores (
    claim_id          TEXT PRIMARY KEY,
    model_probability REAL,
    iso_score         REAL,
    z_flag            INTEGER,
    iqr_flag          INTEGER,
    risk_score        REAL,
    risk_level        TEXT,
    top_reasons       TEXT,
    status            TEXT,
    scored_at         TEXT
);

CREATE TABLE etl_log (
    log_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at  TEXT,
    stage   TEXT,
    metric  TEXT,
    value   REAL
);

CREATE VIEW vw_claims_full AS
SELECT
    f.claim_id, d.full_date AS claim_date, d.year, d.quarter, d.month, d.month_name,
    substr(d.full_date, 1, 7) AS year_month, d.day_name, d.is_weekend,
    c.customer_id, c.customer_name, c.age, c.age_group, c.gender, c.city, c.occupation,
    c.annual_income, c.income_band, c.credit_score,
    p.policy_id, p.policy_type, p.premium, p.coverage_amount, p.start_date AS policy_start_date,
    v.provider_id, v.provider_name, v.provider_type,
    t.claim_type, t.claim_category,
    f.claim_amount, f.claim_to_coverage_ratio, f.days_since_policy_start, f.days_to_report,
    f.num_documents, f.police_report, f.witness_present, f.prior_claims, f.is_fraud,
    s.model_probability, s.iso_score, s.z_flag, s.iqr_flag, s.risk_score, s.risk_level,
    s.top_reasons, s.status
FROM fact_claims f
JOIN dim_date       d ON d.date_key       = f.date_key
JOIN dim_customer   c ON c.customer_key   = f.customer_key
JOIN dim_policy     p ON p.policy_key     = f.policy_key
JOIN dim_provider   v ON v.provider_key   = f.provider_key
JOIN dim_claim_type t ON t.claim_type_key = f.claim_type_key
LEFT JOIN fraud_scores s ON s.claim_id    = f.claim_id;
"""

WAREHOUSE_TABLES = [
    "dim_date", "dim_customer", "dim_policy", "dim_provider",
    "dim_claim_type", "fact_claims", "fraud_scores", "etl_log", "stg_raw_claims",
]


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def query(sql: str, params=()) -> pd.DataFrame:
    with closing(get_connection()) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def execute(sql: str, params=()) -> None:
    with closing(get_connection()) as conn:
        conn.execute(sql, params)
        conn.commit()


def is_ready() -> bool:
    """True when the warehouse exists and has been scored."""
    if not DB_PATH.exists():
        return False
    try:
        return int(query("SELECT COUNT(*) AS n FROM fraud_scores")["n"].iloc[0]) > 0
    except Exception:
        return False


def db_version() -> float:
    """Modification time of the database file; used as a cache key."""
    return DB_PATH.stat().st_mtime if DB_PATH.exists() else 0.0


def table_counts() -> pd.DataFrame:
    rows = []
    with closing(get_connection()) as conn:
        for t in WAREHOUSE_TABLES:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.Error:
                n = 0
            rows.append({"Table": t, "Rows": n})
    return pd.DataFrame(rows)
