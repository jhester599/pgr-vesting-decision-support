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
FRED_API_KEY: str | None = os.getenv("FRED_API_KEY")

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------
FMP_BASE_URL: str = "https://financialmodelingprep.com/api"
AV_BASE_URL: str = "https://www.alphavantage.co/query"
FRED_BASE_URL: str = "https://api.stlouisfed.org/fred/series/observations"

# ---------------------------------------------------------------------------
# Rate limits (requests per day)
# ---------------------------------------------------------------------------
FMP_DAILY_LIMIT: int = 250
AV_DAILY_LIMIT: int = 25
# FRED is a free public API with no enforced daily limit; no budget tracking needed.

# ---------------------------------------------------------------------------
# FRED series lists
# ---------------------------------------------------------------------------
# v3.0: macro regime and credit spread features
FRED_SERIES_MACRO: list[str] = [
    "T10Y2Y",            # 10Y-2Y yield curve spread (recession predictor)
    "GS5",               # 5-Year Treasury Constant Maturity Rate
    "GS2",               # 2-Year Treasury Constant Maturity Rate
    "GS10",              # 10-Year Treasury Constant Maturity Rate
    "T10YIE",            # 10-Year Breakeven Inflation Rate
    "BAA10Y",            # Moody's Baa Corp Bond Yield minus 10Y Treasury (credit spread)
    "BAMLH0A0HYM2",      # ICE BofA High-Yield OAS (HY credit spread)
    "NFCI",              # Chicago Fed National Financial Conditions Index
    "VIXCLS",            # CBOE Volatility Index (VIX) — for regime classification
]
# v3.1: PGR-specific insurance and claims frequency features (defined here, fetched in v3.1)
FRED_SERIES_PGR: list[str] = [
    "CUSR0000SETC01",    # Motor vehicle insurance CPI (rate adequacy proxy)
    "TRFVOLUSM227NFWA",  # Vehicle miles traveled NSA (claims frequency proxy)
]

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

# v3.0: additional purge buffer beyond the target horizon to account for
# serial autocorrelation in monthly data (research report recommendation).
# Total gap = target_horizon + purge_buffer:
#   6M horizon  → gap = 6 + 2 = 8 months
#   12M horizon → gap = 12 + 3 = 15 months
WFO_PURGE_BUFFER_6M: int = 2
WFO_PURGE_BUFFER_12M: int = 3

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
# v2 ETF benchmark universe (20 ETFs)
# These ETFs are the alternative investment targets used for:
#   (a) ML relative return prediction (one WFO model per ETF)
#   (b) Historical vesting event backtest
#
# Design principles:
#   - Vanguard preferred for provider consistency and low expense ratios
#   - Sector ETFs added to capture industry-specific alternatives to PGR
#   - Redundant broad-market trackers removed (one total market, one S&P 500)
#   - All ETFs have pre-2014 history — no proxy backfill required
# ---------------------------------------------------------------------------
ETF_BENCHMARK_UNIVERSE: list[str] = [
    # US Broad Market (2)
    "VTI",              # Vanguard Total Stock Market
    "VOO",              # Vanguard S&P 500
    # US Sectors — Vanguard (6)
    "VGT",              # Vanguard Information Technology
    "VHT",              # Vanguard Health Care
    "VFH",              # Vanguard Financials (includes insurance, banks, asset managers)
    "VIS",              # Vanguard Industrials
    "VDE",              # Vanguard Energy
    "VPU",              # Vanguard Utilities
    # International (3)
    "VXUS",             # Vanguard Total International Stock
    "VEA",              # Vanguard Developed Markets ex-US
    "VWO",              # Vanguard Emerging Markets
    # Dividend-focused (2)
    "VIG",              # Vanguard Dividend Appreciation (dividend growth)
    "SCHD",             # Schwab US Dividend Equity (high yield; meaningfully differs from VIG)
    # Fixed Income (4)
    "BND",              # Vanguard Total Bond Market
    "BNDX",             # Vanguard Total International Bond
    "VCIT",             # Vanguard Intermediate-Term Corporate Bond
    "VMBS",             # Vanguard Mortgage-Backed Securities
    # Real Assets (3)
    "VNQ",              # Vanguard Real Estate
    "GLD",              # SPDR Gold Shares (no Vanguard equivalent)
    "DBC",              # Invesco DB Commodity Index (no Vanguard equivalent)
]

# ---------------------------------------------------------------------------
# v3.1 ensemble and Kelly sizing parameters
# ---------------------------------------------------------------------------
KELLY_FRACTION: float = 0.25          # quarter-Kelly to control risk
KELLY_MAX_POSITION: float = 0.30      # cap single-stock allocation at 30%
ENSEMBLE_MODELS: list[str] = ["elasticnet", "ridge", "bayesian_ridge"]

# ---------------------------------------------------------------------------
# v4.0 Tax-Loss Harvesting replacement map
# Maps tickers to a correlated-but-not-substantially-identical substitute.
# Wash-sale rule: must wait ≥ 31 days before repurchasing the original.
# ---------------------------------------------------------------------------
TLH_REPLACEMENT_MAP: dict[str, str] = {
    # PGR has no direct ETF substitute (individual stock)
    "VTI":  "ITOT",   # iShares Core S&P Total US Stock (highly correlated)
    "VOO":  "IVV",    # iShares Core S&P 500
    "VGT":  "QQQ",    # Invesco Nasdaq-100 (tech-heavy, not substantially identical)
    "VHT":  "XLV",    # Health Care Select Sector SPDR
    "VFH":  "XLF",    # Financial Select Sector SPDR
    "VIS":  "XLI",    # Industrial Select Sector SPDR
    "VDE":  "XLE",    # Energy Select Sector SPDR
    "VPU":  "XLU",    # Utilities Select Sector SPDR
    "VXUS": "IXUS",   # iShares Core MSCI Total International
    "VEA":  "EFA",    # iShares MSCI EAFE
    "VWO":  "EEM",    # iShares MSCI Emerging Markets
    "VIG":  "DGRO",   # iShares Core Dividend Growth
    "SCHD": "VYM",    # Vanguard High Dividend Yield
    "BND":  "AGG",    # iShares Core US Aggregate Bond
    "BNDX": "IAGG",   # iShares Core International Aggregate Bond
    "VCIT": "LQD",    # iShares iBoxx $ Investment Grade Corporate Bond
    "VMBS": "MBB",    # iShares MBS ETF
    "VNQ":  "IYR",    # iShares US Real Estate
    "GLD":  "IAU",    # iShares Gold Trust
    "DBC":  "PDBC",   # Invesco Optimum Yield Diversified Commodity
}

# v4.0 CPCV parameters
CPCV_N_FOLDS: int = 6         # Number of folds for CombinatorialPurgedCV
CPCV_N_TEST_FOLDS: int = 2    # Test folds per split; yields C(6,2)=15 splits, 5 paths

# v4.0 Black-Litterman parameters
BL_RISK_AVERSION: float = 2.5           # Moderate risk aversion (1=aggressive, 5=conservative)
BL_TAU: float = 0.05                    # Uncertainty in equilibrium returns (small = trust prior)
BL_VIEW_CONFIDENCE_SCALAR: float = 1.0  # Scales Ω = RMSE² × scalar

# v4.0 Tax-Loss Harvesting parameters
TLH_LOSS_THRESHOLD: float = -0.10       # Harvest when unrealized return < -10%
TLH_WASH_SALE_DAYS: int = 31            # Minimum days before repurchasing original

# v4.0 Fractional differentiation parameters
FRACDIFF_MAX_D: float = 0.5             # Maximum differentiation order (preserves memory)
FRACDIFF_CORR_THRESHOLD: float = 0.90   # Minimum correlation with original series
FRACDIFF_ADF_ALPHA: float = 0.05        # Stationarity significance level

# ---------------------------------------------------------------------------
# ETF launch dates — tickers with limited history need proxy backfill.
# All ETFs in ETF_BENCHMARK_UNIVERSE have pre-2014 history; no proxies are
# currently required.  BNDX (launched 2013-06-03) pre-dates all backtested
# vesting events (earliest: Jan 2014) by a sufficient margin.
# ---------------------------------------------------------------------------
ETF_LAUNCH_DATES: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Proxy map for pre-launch / limited-data tickers
# Key   = ticker with limited history
# Value = proxy ticker whose history is copied, flagged proxy_fill=1 in DB.
# Empty because all current benchmark ETFs have sufficient pre-2014 history.
# ---------------------------------------------------------------------------
ETF_PROXY_MAP: dict[str, str] = {}

