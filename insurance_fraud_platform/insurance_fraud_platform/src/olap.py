from __future__ import annotations

"""OLAP engine over the star schema (roll-up, drill-down, slice, dice, pivot) using SQL GROUP BY."""
import pandas as pd

from . import database as db

# Whitelisted dimension expressions (user input is never concatenated into SQL).
DIMENSIONS = {
    "Year": "year",
    "Quarter": "year || '-Q' || quarter",
    "Month": "year_month",
    "Day of Week": "day_name",
    "Policy Type": "policy_type",
    "Claim Type": "claim_type",
    "Claim Category": "claim_category",
    "City": "city",
    "Provider Type": "provider_type",
    "Age Group": "age_group",
    "Gender": "gender",
    "Income Band": "income_band",
}

MEASURES = {
    "Total Claims": "COUNT(*)",
    "Total Claim Amount (INR)": "SUM(claim_amount)",
    "Average Claim Amount (INR)": "AVG(claim_amount)",
    "Confirmed Fraud Cases": "SUM(is_fraud)",
    "Fraud Rate (%)": "100.0 * SUM(is_fraud) / COUNT(*)",
    "High-Risk Alerts": "SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END)",
    "Average Risk Score": "AVG(risk_score)",
}

TIME_HIERARCHY = ["Year", "Quarter", "Month"]


def dimension_values(dim: str) -> list:
    df = db.query(f"SELECT DISTINCT {DIMENSIONS[dim]} AS v FROM vw_claims_full ORDER BY 1")
    return df["v"].tolist()


def cube_query(group_dims: list, measures: list, filters: dict | None = None):
    """Return (result DataFrame, SQL text). `filters` maps dimension label -> list of allowed values."""
    select_dims = [f'{DIMENSIONS[d]} AS "{d}"' for d in group_dims]
    select_meas = [f'{MEASURES[m]} AS "{m}"' for m in measures]
    where, params = [], []
    for dim, values in (filters or {}).items():
        if values:
            where.append(f"{DIMENSIONS[dim]} IN ({','.join('?' * len(values))})")
            params.extend(values)
    sql = f"SELECT {', '.join(select_dims + select_meas)}\nFROM vw_claims_full"
    if where:
        sql += "\nWHERE " + "\n  AND ".join(where)
    if group_dims:
        sql += "\nGROUP BY " + ", ".join(f'"{d}"' for d in group_dims)
        sql += "\nORDER BY " + ", ".join(f'"{d}"' for d in group_dims)
    df = db.query(sql, params)
    for m in measures:
        df[m] = df[m].astype(float).round(2)
    shown = sql
    for v in params:                    # human-readable SQL for display only
        shown = shown.replace("?", repr(v), 1)
    return df, shown
