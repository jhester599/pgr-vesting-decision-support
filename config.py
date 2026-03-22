"""
Central configuration for the PGR Vesting Decision Support engine.

API keys are loaded from a .env file (never hardcoded).
All tuneable parameters for the backtest are defined here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API credentials
# Keys are read lazily via os.getenv so that imports never fail in test
# environments without a .env file. The API clients will raise a clear
# error if a key is None at the moment an actual HTTP call is made.
# ---------------------------------------------------------------------------
FMP_API_KEY: str | None = os.getenv("FMP_API_KEY")
AV_API_KEY: str | None = os.getenv("AV_API_KEY")

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------
FMP_BASE_URL: str = "https://financialmodelingprep.com/api"
AV_BASE_URL: str = "https://www.alphavantage.co/query"

# ---------------------------------------------------------------------------
# Rate limits (requests per day)
# ---------------------------------------------------------------------------
FMP_DAILY_LIMIT: int = 250
AV_DAILY_LIMIT: int = 25

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR: str = os.path.join("data", "raw")
DATA_PROCESSED_DIR: str = os.path.join("data", "processed")
REQUEST_COUNTS_FILE: str = os.path.join(DATA_RAW_DIR, ".request_counts.json")

# ---------------------------------------------------------------------------
# Target ticker
# ---------------------------------------------------------------------------
TICKER: str = "PGR"

# ---------------------------------------------------------------------------
# Known PGR corporate actions (used for validation in tests)
# ---------------------------------------------------------------------------
PGR_KNOWN_SPLITS: list[dict] = [
    {"date": "1992-12-09", "ratio": 3.0},
    {"date": "2002-04-23", "ratio": 3.0},
    {"date": "2006-05-19", "ratio": 4.0},
]

# ---------------------------------------------------------------------------
# Walk-Forward Optimization parameters
# ---------------------------------------------------------------------------
WFO_TRAIN_WINDOW_MONTHS: int = 60    # 5-year rolling training window
WFO_TEST_WINDOW_MONTHS: int = 6      # 6-month out-of-sample test period
WFO_EMBARGO_MONTHS: int = 1          # Gap between train end and test start
WFO_MIN_GAINSHARE_OBS: int = 60      # Min non-NaN rows to include Gainshare features

# ---------------------------------------------------------------------------
# Tax rates (federal maximums; add state rate in .env as needed)
# ---------------------------------------------------------------------------
LTCG_RATE: float = float(os.getenv("LTCG_RATE", "0.20"))
STCG_RATE: float = float(os.getenv("STCG_RATE", "0.37"))

# ---------------------------------------------------------------------------
# RSU vesting schedule
# ---------------------------------------------------------------------------
TIME_RSU_VEST_MONTH: int = 1    # January (time-based)
TIME_RSU_VEST_DAY: int = 19
PERF_RSU_VEST_MONTH: int = 7    # July (performance-based)
PERF_RSU_VEST_DAY: int = 17
