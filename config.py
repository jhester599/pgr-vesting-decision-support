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
# Cache paths (v1 JSON/parquet cache — retained for migration compatibility)
# ---------------------------------------------------------------------------
DATA_RAW_DIR: str = os.path.join("data", "raw")
DATA_PROCESSED_DIR: str = os.path.join("data", "processed")
REQUEST_COUNTS_FILE: str = os.path.join(DATA_RAW_DIR, ".request_counts.json")

# ---------------------------------------------------------------------------
# v2 SQLite database path
# ---------------------------------------------------------------------------
DB_PATH: str = os.path.join("data", "pgr_financials.db")

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
WFO_MIN_GAINSHARE_OBS: int = 60      # Min non-NaN rows to include Gainshare features

# v1 bug: WFO_EMBARGO_MONTHS = 1 was too short for a 6M overlapping target.
# v2 fix: embargo must equal the target horizon to prevent autocorrelation
# leakage between consecutive monthly observations.
WFO_EMBARGO_MONTHS: int = 1          # Retained for v1 backward-compat; DO NOT USE in v2
WFO_EMBARGO_MONTHS_6M: int = 6       # Correct embargo for 6-month target horizon
WFO_EMBARGO_MONTHS_12M: int = 12     # Correct embargo for 12-month target horizon
WFO_TARGET_HORIZONS: list[int] = [6, 12]

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

# ---------------------------------------------------------------------------
# v2 ETF benchmark universe
# These 22 ETFs are the alternative investment targets used for:
#   (a) ML relative return prediction (one WFO model per ETF)
#   (b) Historical vesting event backtest
# They overlap with but are distinct from the drift_analyzer's sector-overlap ETFs.
# ---------------------------------------------------------------------------
ETF_BENCHMARK_UNIVERSE: list[str] = [
    # US Broad Market
    "FZROX", "FNILX", "FSKAX", "VTI", "ITOT", "SCHB",
    # International
    "VXUS", "FZILX", "VEA", "IEMG",
    # Dividend
    "SCHD", "VIG", "DGRO",
    # Fixed Income
    "BND", "VBTLX", "VCIT", "LQD", "MBB",
    # Alternatives
    "VNQ", "GLD", "DBC", "VEE",
]

# ---------------------------------------------------------------------------
# ETF launch dates — tickers with limited history need proxy backfill
# ---------------------------------------------------------------------------
ETF_LAUNCH_DATES: dict[str, str] = {
    "FZROX": "2018-08-02",   # Fidelity Zero Total Market
    "FNILX": "2018-08-02",   # Fidelity Zero Large Cap
    "FZILX": "2018-08-02",   # Fidelity Zero International
    "DGRO":  "2014-06-11",   # iShares Core Dividend Growth
}

# ---------------------------------------------------------------------------
# Proxy map for pre-launch / limited-data tickers
# Key   = ticker with limited history
# Value = proxy ticker whose total return is substituted for pre-launch periods
# Rows filled from proxies are flagged proxy_fill=1 in the DB.
# ---------------------------------------------------------------------------
ETF_PROXY_MAP: dict[str, str] = {
    "FZROX": "VTI",    # Total US market — identical exposure
    "FNILX": "VTI",    # Large-cap US — effectively identical
    "FZILX": "VXUS",   # Total international — identical exposure
    "DGRO":  "VIG",    # Dividend growth — equivalent index
    "VBTLX": "BND",    # Same Bloomberg US Aggregate index, different wrapper
    "VEE":   "IEMG",   # Canadian TSX ETF; IEMG is the closest US-listed equivalent
}

