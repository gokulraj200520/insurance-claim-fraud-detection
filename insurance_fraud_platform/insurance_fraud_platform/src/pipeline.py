"""End-to-end pipeline: generate raw data -> ETL into the warehouse -> score every claim."""
from . import database as db
from .config import DATA_DIR, N_CLAIMS, RAW_CSV_PATH
from .data_generator import generate_raw_claims
from .etl import run_etl
from .scoring import score_all_claims


def build_database(regenerate: bool = True, n_claims: int = N_CLAIMS) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if regenerate or not RAW_CSV_PATH.exists():
        generate_raw_claims(n_claims).to_csv(RAW_CSV_PATH, index=False)
    etl_stats = run_etl(RAW_CSV_PATH)
    score_stats = score_all_claims()
    return {**etl_stats, **score_stats}


def ensure_ready() -> None:
    if not db.is_ready():
        build_database(regenerate=not RAW_CSV_PATH.exists())
