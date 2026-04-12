"""
FRED series lists, publication lags, ETF benchmark universe, peer ticker
universe, TLH replacement map, feature sets, data paths, DB path, ticker,
and PGR corporate actions.
"""

import os

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
# v3.1: PGR-specific insurance and claims frequency features
# v4.5: added used car CPI and medical CPI as direct claims severity predictors
# NOTE: CUSR0000SETC01 (motor vehicle insurance CPI) was removed on 2026-03-24 —
#       the series does not exist in FRED's observations endpoint (400 Bad Request).
#       The BLS publishes this data under series code SETE but FRED does not index it.
#       Re-add when a valid FRED ID is identified.
FRED_SERIES_PGR: list[str] = [
    "TRFVOLUSM227NFWA",  # Vehicle miles traveled NSA (claims frequency proxy)
    "CUSR0000SETA02",    # Used car & truck CPI (auto total-loss severity; v4.5)
    "CUSR0000SAM2",      # Medical care CPI (bodily injury / PIP severity; v4.5)
    # v4.5: PPI for Private Passenger Auto Insurance — replaces the originally
    # planned CUSR0000SETC01 (motor vehicle insurance CPI) which is not
    # available via FRED's observations endpoint.  The PPI series captures
    # cost-based pricing dynamics upstream of the CPI; validated 2026-03-29:
    # partial IC=0.353 (p<0.0001), standalone hit-rate 76.1%, 96.6% of
    # rolling 36M windows positive.  Monthly BLS release; ~1 month lag.
    "PCU5241265241261",  # PPI: Private Passenger Auto Insurance (v4.5)
]

# ---------------------------------------------------------------------------
# v19 research-only public macro series
# ---------------------------------------------------------------------------
# These are loaded from public endpoints during the v19 feature-completion
# cycle, then stored in fred_macro_monthly for point-in-time-safe reuse by the
# research harness. They are intentionally separate from the production FRED
# fetch lists because some come from BLS or Multpl rather than the FRED API.
V19_PUBLIC_MACRO_SERIES: list[str] = [
    "DTWEXBGS",                     # Broad trade-weighted USD index
    "DCOILWTICO",                   # WTI spot oil price
    "MORTGAGE30US",                 # 30-year fixed mortgage rate
    "WPU45110101",                  # PPI: legal services
    "PPIACO",                       # PPI: all commodities
    "MRTSSM447USN",                 # Retail sales: gasoline stations
    "THREEFYTP10",                  # ACM / NY Fed 10Y term premium
    "CUSR0000SETE",                 # CPI: motor vehicle insurance (BLS)
    "SP500_PE_RATIO_MULTPL",        # S&P 500 PE ratio (Multpl monthly table)
    "SP500_EARNINGS_YIELD_MULTPL",  # S&P 500 earnings yield (Multpl monthly table)
    "SP500_PRICE_TO_BOOK_MULTPL",   # S&P 500 price-to-book ratio (Multpl monthly table)
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
    # Insurance Industry (1) — v4.5: also used for pgr_vs_kie_6m relative strength feature
    "KIE",              # SPDR S&P Insurance ETF (pure-play insurance; inception 2005-11-08)
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
# v6.0 Peer ticker universe — insurance company peers for cross-asset signals
# (PGR vs. peer composite, residual momentum baseline).
# Fetched weekly by scripts/peer_fetch.py (Sunday 04:00 UTC cron — 30 hours
# after the main Friday 22:00 UTC weekly_fetch.py cron, to keep both runs
# within the 25 calls/day AV free-tier limit: Friday=24 calls, Sunday=8 calls).
# Bootstrap: run peer_bootstrap.yml once (workflow_dispatch) to load full history.
# ---------------------------------------------------------------------------
PEER_TICKER_UNIVERSE: list[str] = [
    "ALL",   # Allstate — closest business model comp (personal auto + home)
    "TRV",   # Travelers — large commercial + personal lines
    "CB",    # Chubb — global P&C, diversified
    "HIG",   # Hartford — personal + commercial + employee benefits
]

# ---------------------------------------------------------------------------
# ETF launch dates — tickers with limited history need proxy backfill.
# All ETFs in ETF_BENCHMARK_UNIVERSE have pre-2014 history; no proxies are
# currently required.  BNDX (launched 2013-06-03) pre-dates all backtested
# vesting events (earliest: Jan 2014) by a sufficient margin.
# KIE (launched 2005-11-08) has ~20 years of history; no proxy needed.
# ---------------------------------------------------------------------------
ETF_LAUNCH_DATES: dict[str, str] = {
    "KIE": "2005-11-08",    # SPDR S&P Insurance ETF; ~20 years history (v4.5)
}

# ---------------------------------------------------------------------------
# Proxy map for pre-launch / limited-data tickers
# Key   = ticker with limited history
# Value = proxy ticker whose history is copied, flagged proxy_fill=1 in DB.
# Empty because all current benchmark ETFs have sufficient pre-2014 history.
# ---------------------------------------------------------------------------
ETF_PROXY_MAP: dict[str, str] = {}

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
    "KIE":  "IAK",    # iShares U.S. Insurance ETF (diff. index: DJ vs S&P; v4.5)
}

# ---------------------------------------------------------------------------
# v4.1 — Data Integrity: Publication Lag Guards
# ---------------------------------------------------------------------------
# FRED publication lag (months). Prevents look-ahead bias from revised data.
# The feature matrix applies these lags when reading FRED data from the DB,
# so that month-T features only use data that was publicly available at month T.
FRED_DEFAULT_LAG_MONTHS: int = 1  # default for any series not in FRED_SERIES_LAGS
FRED_SERIES_LAGS: dict = {
    "NFCI":              2,   # weekly; revised for ~8 weeks after release
    "TRFVOLUSM227NFWA":  2,   # VMT NSA; 2-month publication delay
    "CUSR0000SETC01":    1,   # Motor vehicle insurance CPI; monthly release
    "CUSR0000SETA02":    1,   # Used car CPI; monthly BLS release (v4.5)
    "CUSR0000SAM2":      1,   # Medical care CPI; monthly BLS release (v4.5)
    "PCU5241265241261":  1,   # PPI: Private Passenger Auto Insurance (v4.5)
    "DTWEXBGS":          1,   # Broad trade-weighted USD index
    "DCOILWTICO":        1,   # WTI spot price
    "MORTGAGE30US":      1,   # Freddie Mac mortgage rate
    "WPU45110101":       1,   # Legal services PPI
    "PPIACO":            1,   # Broad PPI
    "MRTSSM447USN":      1,   # Gasoline retail sales
    "THREEFYTP10":       1,   # 10Y term premium
    "CUSR0000SETE":      1,   # Motor vehicle insurance CPI (BLS)
    "SP500_PE_RATIO_MULTPL": 1,          # Research-only monthly market valuation proxy
    "SP500_EARNINGS_YIELD_MULTPL": 1,    # Research-only monthly market valuation proxy
    "SP500_PRICE_TO_BOOK_MULTPL": 1,     # Research-only monthly market valuation proxy
    "BAA10Y":            1,
    "BAMLH0A0HYM2":      1,
    "T10Y2Y":            1,
    "GS5":               1,
    "GS2":               1,
    "GS10":              1,
    "T10YIE":            1,
    "VIXCLS":            1,
}

# EDGAR filing lag (months from period-end to public availability).
# PGR 10-Q is filed ~45 days after quarter end; 10-K ~60 days.
# Using 2 months as a conservative guard across all EDGAR quarterly data.
# Monthly 8-K data (combined ratio, PIF) is filed within the same month —
# however, applying the same lag is conservative and prevents any edge case.
EDGAR_FILING_LAG_MONTHS: int = 2

# ---------------------------------------------------------------------------
# v4.3 — Signal Quality + Confidence Layer
# ---------------------------------------------------------------------------
# Redundant features dropped from the final feature matrix to improve the
# obs/feature ratio from ~3.5:1 to ~4:1.
#   vol_21d:          highly correlated with vol_63d (same signal, shorter window)
#   credit_spread_ig: subset of credit_spread_hy (HY spread contains the IG
#                     signal plus the distress premium; IG is redundant)
FEATURES_TO_DROP: list[str] = ["vol_21d", "credit_spread_ig"]

# ---------------------------------------------------------------------------
# v11.0 Primary forecast universe — the 8 benchmarks selected in the v20/v21
# research cycle.  Provides better diversification coverage without the noise
# from 21-benchmark over-fitting.  ETF_BENCHMARK_UNIVERSE (above) still governs
# data ingestion; PRIMARY_FORECAST_UNIVERSE governs which benchmarks are used
# by the production WFO ensemble.
# ---------------------------------------------------------------------------
PRIMARY_FORECAST_UNIVERSE: list[str] = [
    "VOO",   # S&P 500 (core US equity)
    "VXUS",  # Total International
    "VWO",   # Emerging Markets
    "VMBS",  # Mortgage-Backed Securities
    "BND",   # Total Bond Market
    "GLD",   # Gold
    "DBC",   # Commodities
    "VDE",   # Energy
]

# ---------------------------------------------------------------------------
# Classification shadow: portfolio alignment (v124)
# ---------------------------------------------------------------------------

# Investable benchmarks for portfolio-weighted aggregate (v124: adds VGT + VIG).
# VIG serves as a proxy for SCHD (dividend/value sleeve) due to longer history
# (~228 months vs SCHD's ~168 months). SCHD added as separate per-benchmark
# classifier when history reaches ~185 months (~v135, late 2027).
INVESTABLE_CLASSIFIER_BENCHMARKS: list[str] = ["VOO", "VGT", "VIG", "VXUS", "VWO", "BND"]

# Fixed base weights from balanced_pref_95_5. Sum to exactly 1.0.
# VIG carries the SCHD sleeve weight (0.15). If SCHD is later added as a
# separate classifier, SCHD and VIG can share the 0.15 weight equally.
INVESTABLE_CLASSIFIER_BASE_WEIGHTS: dict[str, float] = {
    "VOO": 0.40,
    "VGT": 0.20,
    "VIG": 0.15,
    "VXUS": 0.10,
    "VWO": 0.10,
    "BND": 0.05,
}

# Contextual (non-investable) benchmarks retained for regime diagnostics.
# These continue to run as per-benchmark classifiers but are excluded from the
# primary investable-pool aggregate.
CONTEXTUAL_CLASSIFIER_BENCHMARKS: list[str] = ["DBC", "GLD", "VMBS", "VDE"]

# v11.0 promoted model-specific feature sets (established in v18/v20 research).
# Ridge v18: swaps yield_curvature for real_yield_change_6m, adds BVPS growth
#   and NPW growth to strengthen insurance-fundamental signal.
# GBT v18:   swaps vmt_yoy for vwo_vxus_spread_6m (EM/DM spread), adds
#   rate_adequacy_gap_yoy, pif_growth_yoy, investment_book_yield.
# The previous v8.6 overrides (elasticnet / bayesian_ridge) are retained for
# backward-compat but are no longer used by the primary production ensemble.
MODEL_FEATURE_BASE_GROUP_B: list[str] = [
    "mom_3m",
    "mom_6m",
    "mom_12m",
    "vol_63d",
    "yield_slope",
    "yield_curvature",
    "real_rate_10y",
    "credit_spread_hy",
    "nfci",
    "vix",
    "vmt_yoy",
]
MODEL_FEATURE_OVERRIDES: dict[str, list[str]] = {
    # v11.0 primary models (ridge_lean_v1__v18, gbt_lean_plus_two__v18)
    "ridge": [
        "mom_12m",
        "vol_63d",
        "yield_slope",
        "real_yield_change_6m",
        "real_rate_10y",
        "credit_spread_hy",
        "nfci",
        "vix",
        "combined_ratio_ttm",
        "investment_income_growth_yoy",
        "book_value_per_share_growth_yoy",
        "npw_growth_yoy",
    ],
    "gbt": [
        "mom_3m",
        "mom_6m",
        "mom_12m",
        "vol_63d",
        "yield_slope",
        "yield_curvature",
        "vwo_vxus_spread_6m",
        "credit_spread_hy",
        "nfci",
        "vix",
        "rate_adequacy_gap_yoy",
        "pif_growth_yoy",
        "investment_book_yield",
    ],
    # v8.6 overrides — retained for reference / research re-runs
    "elasticnet": MODEL_FEATURE_BASE_GROUP_B + [
        "investment_income_growth_yoy",
        "roe_net_income_ttm",
        "underwriting_income",
    ],
    "bayesian_ridge": MODEL_FEATURE_BASE_GROUP_B + [
        "combined_ratio_ttm",
        "investment_income_growth_yoy",
        "roe_net_income_ttm",
        "buyback_yield",
        "buyback_acceleration",
    ],
}
