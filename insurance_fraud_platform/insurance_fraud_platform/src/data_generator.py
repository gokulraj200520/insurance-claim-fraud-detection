"""Synthetic raw insurance-claim generator.

Produces a deliberately *dirty* operational extract (duplicates, missing values,
inconsistent codes, invalid ages, mixed date formats) so that the ETL and data
cleaning stages have real work to do. Fraud labels follow realistic red-flag
patterns (claims filed soon after inception, very high claim-to-coverage ratio,
late reporting, few documents, repeat claimants, suspicious providers).
"""
import numpy as np
import pandas as pd

from .config import FRAUD_RATE, N_CLAIMS, RANDOM_STATE

FIRST_NAMES = ["Arun", "Karthik", "Suresh", "Ramesh", "Vignesh", "Prakash", "Manoj", "Dinesh", "Rahul",
               "Vijay", "Ajay", "Naveen", "Sathish", "Gopal", "Harish", "Priya", "Divya", "Lakshmi",
               "Meena", "Anitha", "Kavitha", "Revathi", "Deepa", "Swathi", "Nandhini", "Pooja", "Sangeetha",
               "Mohan", "Bala", "Yuvaraj"]
LAST_NAMES = ["Kumar", "Raj", "Selvam", "Murugan", "Iyer", "Nair", "Reddy", "Pillai", "Sharma", "Krishnan",
              "Subramanian", "Natarajan", "Rao", "Menon", "Gupta", "Chandran", "Babu", "Devi", "Ramesh"]
CITIES = ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Bengaluru",
          "Hyderabad", "Mumbai", "Delhi", "Kolkata", "Pune", "Kochi"]
OCCUPATIONS = ["Teacher", "Engineer", "Doctor", "Shop Owner", "Driver", "Farmer", "Accountant",
               "Software Developer", "Nurse", "Business Owner", "Clerk", "Contractor"]
POLICY_TYPES = ["Auto", "Health", "Home", "Life"]
POLICY_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
COVERAGE_RANGE = {"Auto": (300_000, 1_500_000), "Health": (200_000, 1_000_000),
                  "Home": (1_000_000, 5_000_000), "Life": (1_000_000, 10_000_000)}
CLAIM_TYPES = {"Auto": ["Collision", "Theft", "Third-Party Liability"],
               "Health": ["Hospitalization", "Surgery", "Outpatient Treatment"],
               "Home": ["Fire Damage", "Burglary", "Water Damage"],
               "Life": ["Death Benefit", "Critical Illness"]}
PROVIDER_TYPES = {"Auto": ["Garage", "Repair Shop"], "Health": ["Hospital", "Clinic"],
                  "Home": ["Repair Shop", "Assessor"], "Life": ["Hospital", "Assessor"]}
PROVIDER_PREFIX = ["Sri Lakshmi", "Green Valley", "Royal", "Metro", "Sunrise", "Heritage", "Star", "Lotus",
                   "Prime", "Unity", "Golden", "Elite", "Vijay", "Ganga", "Kaveri", "Annai", "Balaji", "Orient"]

DATA_START = np.datetime64("2022-01-01")
DATA_END = np.datetime64("2025-12-31")


def _make_customers(rng: np.random.Generator, m: int) -> pd.DataFrame:
    ptype = rng.choice(POLICY_TYPES, m, p=POLICY_WEIGHTS)
    coverage = np.zeros(m)
    for p, (lo, hi) in COVERAGE_RANGE.items():
        mask = ptype == p
        coverage[mask] = np.round(rng.uniform(lo, hi, mask.sum()), -4)
    premium = np.round(coverage * rng.uniform(0.01, 0.04, m), -2)
    start = np.datetime64("2019-01-01") + rng.integers(0, 2372, m).astype("timedelta64[D]")
    names = [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(m)]
    return pd.DataFrame({
        "customer_id": [f"CUST-{i + 1:05d}" for i in range(m)],
        "customer_name": names,
        "age": np.clip(rng.normal(41, 12, m), 21, 75).astype(int),
        "gender": rng.choice(["Male", "Female"], m),
        "city": rng.choice(CITIES, m),
        "occupation": rng.choice(OCCUPATIONS, m),
        "annual_income": np.round(rng.lognormal(np.log(600_000), 0.5, m), -3),
        "credit_score": np.clip(rng.normal(690, 70, m), 400, 850).astype(int),
        "policy_id": [f"POL-{i + 1:05d}" for i in range(m)],
        "policy_type": ptype,
        "premium": premium,
        "coverage_amount": coverage,
        "policy_start_date": start,
    })


def _make_providers(rng: np.random.Generator, k: int) -> pd.DataFrame:
    types = ["Hospital", "Clinic", "Garage", "Repair Shop", "Assessor"]
    ptype = rng.choice(types, k)
    city = rng.choice(CITIES, k)
    names = [f"{rng.choice(PROVIDER_PREFIX)} {t} - {c}" for t, c in zip(ptype, city)]
    return pd.DataFrame({
        "provider_id": [f"PRV-{i + 1:04d}" for i in range(k)],
        "provider_name": names,
        "provider_type": ptype,
        "suspicious": rng.random(k) < 0.12,
    })


def _inject_dirt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate typical data-quality problems of an operational source system."""
    df = df.copy()
    n = len(df)
    for col in ["age", "credit_score", "num_documents", "days_to_report"]:
        df[col] = df[col].astype(float)

    def pick(p):
        return rng.random(n) < p

    # missing values
    df.loc[pick(0.03), "age"] = np.nan
    df.loc[pick(0.02), "city"] = np.nan
    df.loc[pick(0.02), "occupation"] = np.nan
    df.loc[pick(0.02), "credit_score"] = np.nan
    df.loc[pick(0.025), "num_documents"] = np.nan
    df.loc[pick(0.015), "days_to_report"] = np.nan
    # impossible ages
    m = pick(0.005)
    df.loc[m, "age"] = rng.choice([-3, 0, 150, 999], m.sum())
    # inconsistent gender coding
    variants = {"Male": ["M", "male", "MALE", "m"], "Female": ["F", "female", "FEMALE", "f"]}
    m = pick(0.35)
    df.loc[m, "gender"] = [variants[x][rng.integers(0, 4)] for x in df.loc[m, "gender"]]
    # inconsistent policy type casing / whitespace
    m = pick(0.08)
    df.loc[m, "policy_type"] = [
        [x.lower(), x.upper(), f" {x} "][rng.integers(0, 3)] for x in df.loc[m, "policy_type"]]
    # Yes / No flags coded in several ways
    for col in ["police_report", "witness_present"]:
        flag = np.where(df[col].astype(int) == 1, "Yes", "No")
        m = pick(0.30)
        yes_v, no_v = ["Y", "yes", "YES", "True"], ["N", "no", "NO", "False"]
        flag = flag.astype(object)
        flag[m] = [yes_v[rng.integers(0, 4)] if f == "Yes" else no_v[rng.integers(0, 4)] for f in flag[m]]
        df[col] = flag
    # mixed date formats
    dates = pd.to_datetime(df["claim_date"])
    m = pick(0.04)
    df["claim_date"] = np.where(m, dates.dt.strftime("%d/%m/%Y"), dates.dt.strftime("%Y-%m-%d"))
    df["policy_start_date"] = pd.to_datetime(df["policy_start_date"]).dt.strftime("%Y-%m-%d")
    # exact duplicate records
    dup = df.sample(frac=0.02, random_state=RANDOM_STATE)
    df = pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def generate_raw_claims(n_claims: int = N_CLAIMS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = _make_customers(rng, int(n_claims * 0.33))
    providers = _make_providers(rng, 160)

    c = customers.iloc[rng.integers(0, len(customers), n_claims)].reset_index(drop=True)
    is_fraud = (rng.random(n_claims) < FRAUD_RATE).astype(int)
    fraud = is_fraud == 1
    # ~30% of fraud is "stealth" (looks like a normal claim); ~5% of genuine claims look suspicious
    look_fraud = (fraud & (rng.random(n_claims) > 0.30)) | (~fraud & (rng.random(n_claims) < 0.05))

    # ---- claim dates -------------------------------------------------------
    start = c["policy_start_date"].values.astype("datetime64[D]")
    span = int((DATA_END - DATA_START).astype(int)) + 1
    base = DATA_START + rng.integers(0, span, n_claims).astype("timedelta64[D]")
    base = np.where(base <= start, start + rng.integers(1, 400, n_claims).astype("timedelta64[D]"), base)
    short = start + rng.integers(3, 46, n_claims).astype("timedelta64[D]")
    use_short = look_fraud & (rng.random(n_claims) < 0.6) & (short >= DATA_START)
    claim_date = np.minimum(np.where(use_short, short, base), DATA_END)
    days_since = (claim_date - start).astype(int)

    # ---- amounts and behavioural indicators -------------------------------
    ratio = np.clip(np.where(look_fraud, rng.beta(6, 3, n_claims), rng.beta(2, 8, n_claims)), 0.01, 1.0)
    claim_amount = np.round(ratio * c["coverage_amount"].values, -2)

    legit_report = np.clip(rng.gamma(2, 2.0, n_claims), 0, 30).astype(int)
    days_to_report = np.where(look_fraud & (rng.random(n_claims) < 0.55), rng.integers(12, 46, n_claims), legit_report)
    num_docs = np.where(look_fraud, np.clip(rng.poisson(1.3, n_claims), 0, 5),
                        np.clip(rng.poisson(4, n_claims) + 1, 1, 9))
    police = (rng.random(n_claims) < np.where(look_fraud, 0.30, 0.75)).astype(int)
    witness = (rng.random(n_claims) < np.where(look_fraud, 0.15, 0.45)).astype(int)
    prior = np.where(look_fraud, np.clip(rng.poisson(2.2, n_claims), 0, 8), np.clip(rng.poisson(0.7, n_claims), 0, 6))

    # ---- claim type & provider (depend on policy type) --------------------
    claim_type = np.empty(n_claims, dtype=object)
    prov_id = np.empty(n_claims, dtype=object)
    prov_name = np.empty(n_claims, dtype=object)
    prov_type = np.empty(n_claims, dtype=object)
    for ptype in POLICY_TYPES:
        mask = (c["policy_type"] == ptype).values
        k = int(mask.sum())
        claim_type[mask] = rng.choice(CLAIM_TYPES[ptype], k)
        pool = providers[providers["provider_type"].isin(PROVIDER_TYPES[ptype])].reset_index(drop=True)
        w_legit = np.where(pool["suspicious"], 1.5, 1.0)
        w_fraud = np.where(pool["suspicious"], 9.0, 1.0)
        idx_legit = rng.choice(len(pool), k, p=w_legit / w_legit.sum())
        idx_fraud = rng.choice(len(pool), k, p=w_fraud / w_fraud.sum())
        chosen = np.where(fraud[mask], idx_fraud, idx_legit)
        prov_id[mask] = pool["provider_id"].values[chosen]
        prov_name[mask] = pool["provider_name"].values[chosen]
        prov_type[mask] = pool["provider_type"].values[chosen]

    df = pd.DataFrame({
        "claim_date": pd.to_datetime(claim_date),
        "customer_id": c["customer_id"], "customer_name": c["customer_name"], "age": c["age"],
        "gender": c["gender"], "city": c["city"], "occupation": c["occupation"],
        "annual_income": c["annual_income"], "credit_score": c["credit_score"],
        "policy_id": c["policy_id"], "policy_type": c["policy_type"], "premium": c["premium"],
        "coverage_amount": c["coverage_amount"], "policy_start_date": pd.to_datetime(start),
        "claim_type": claim_type, "provider_id": prov_id, "provider_name": prov_name,
        "provider_type": prov_type, "claim_amount": claim_amount, "days_to_report": days_to_report,
        "num_documents": num_docs, "police_report": police, "witness_present": witness,
        "prior_claims": prior, "is_fraud": is_fraud,
    })
    df = df.sort_values("claim_date").reset_index(drop=True)
    df.insert(0, "claim_id", [f"CLM-{i + 1:06d}" for i in range(len(df))])
    # sanity: derived field kept consistent with sorted order
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    return _inject_dirt(df, rng)
