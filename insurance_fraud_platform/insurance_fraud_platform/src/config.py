"""Central configuration for the Insurance Claim Fraud Detection Platform."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "insurance_dw.db"
RAW_CSV_PATH = DATA_DIR / "raw_claims.csv"

APP_TITLE = "Insurance Claim Fraud Detection Platform"

RANDOM_STATE = 42
N_CLAIMS = 6000          # number of synthetic raw claim records generated
FRAUD_RATE = 0.09        # share of fraudulent claims in the synthetic history

# Composite risk score thresholds (0 - 100)
RISK_HIGH = 60
RISK_MEDIUM = 35

# Composite risk score weights (must add up to 1.0)
W_MODEL = 0.60           # supervised classifier probability
W_ISO = 0.25             # Isolation Forest anomaly score
W_STAT = 0.15            # statistical outlier flags (Z-score / IQR)

STATUSES = ["Open", "Under Investigation", "Confirmed Fraud", "Cleared", "Auto-Cleared"]
