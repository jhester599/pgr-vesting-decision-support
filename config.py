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
# FMP_API_KEY: DEPRECATED — FMP v3 endpoints (free tier) were retired on
# 2025-08-31. Quarterly fundamentals are now sourced from SEC EDGAR XBRL,
# which is free, authoritative, and requires no API key.
FMP_API_KEY: str | None = os.getenv("FMP_API_KEY")
AV_API_KEY: str | None = os.getenv("AV_API_KEY")
FRED_API_KEY: str | None = os.getenv("FRED_API_KEY")

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------
# FMP_BASE_URL: retained for fmp_client.py backward-compatibility only.
FMP_BASE_URL: str = "https://financialmodelingprep.com/api"
AV_BASE_URL: str = "https://www.alphavantage.co/query"
FRED_BASE_URL: str = "https://api.stlouisfed.org/fred/series/observations"

# SEC EDGAR XBRL — free, authoritative, no API key required.
# Required User-Agent header: descriptive name + contact email.
# Rate limit: 10 requests/second (enforced server-side).
EDGAR_BASE_URL: str = "https://data.sec.gov"
EDGAR_PGR_CIK: str = "CIK0000080661"
EDGAR_USER_AGENT_FALLBACK: str = (
    "PGR Vesting Decision Support contact@example.com"
)


def get_edgar_user_agent() -> str:
    """Return the SEC EDGAR User-Agent from env, or a generic fallback."""
    return os.getenv("EDGAR_USER_AGENT", EDGAR_USER_AGENT_FALLBACK)


def build_edgar_headers(host: str | None = None) -> dict[str, str]:
    """Build standard SEC EDGAR headers with the configured User-Agent."""
    headers = {
        "User-Agent": get_edgar_user_agent(),
        "Accept-Encoding": "gzip, deflate",
    }
    if host is not None:
        headers["Host"] = host
    return headers

# ---------------------------------------------------------------------------
# Rate limits (requests per day)
# ---------------------------------------------------------------------------
# FMP_DAILY_LIMIT: retained for the api_request_log schema but no longer
# consumed — FMP fundamentals fetches were replaced by EDGAR XBRL.
FMP_DAILY_LIMIT: int = 250
AV_DAILY_LIMIT: int = 25
# EDGAR and FRED are free public APIs with no enforced daily limit.

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
# v3.1 ensemble and Kelly sizing parameters
# ---------------------------------------------------------------------------
KELLY_FRACTION: float = 0.25          # quarter-Kelly to control risk
KELLY_MAX_POSITION: float = 0.20      # v4.1: reduced from 0.30 (Meulbroek 2005: 25% employer stock = 42% CE loss)
# v5.0: added shallow GBT as 4th ensemble member (max_depth=2, n_estimators=50)
ENSEMBLE_MODELS: list[str] = ["elasticnet", "ridge", "bayesian_ridge", "gbt"]

# ---------------------------------------------------------------------------
# v13 recommendation-layer promotion study
# ---------------------------------------------------------------------------
# Keep the live model stack unchanged, but allow the monthly report/email layer
# to include or eventually promote the simpler diversification-first baseline
# that performed best in the v11/v12 recommendation studies.
RECOMMENDATION_LAYER_MODE: str = os.getenv("RECOMMENDATION_LAYER_MODE", "shadow_promoted")
RECOMMENDATION_LAYER_VALID_MODES: tuple[str, ...] = (
    "live_only",
    "live_with_shadow",
    "shadow_promoted",
)
# v22: keep the active recommendation layer unchanged, but replace the
# previously displayed live-stack cross-check with the historically stronger
# reduced-universe Ridge+GBT candidate selected in v21.
V22_PROMOTED_CROSS_CHECK_CANDIDATE: str = os.getenv(
    "V22_PROMOTED_CROSS_CHECK_CANDIDATE",
    "ensemble_ridge_gbt_v18",
)
V13_SHADOW_BASELINE_STRATEGY: str = "historical_mean"
V13_SHADOW_BASELINE_POLICY: str = "neutral_band_3pct"
V13_SHADOW_FORECAST_UNIVERSE: list[str] = [
    "VOO",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
    "VFH",
]
V13_REDEPLOY_UNIVERSE: list[str] = [
    "VOO",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
]

# ---------------------------------------------------------------------------
# v4.4 — STCG Tax Boundary Guard
# ---------------------------------------------------------------------------
# Minimum predicted 6M alpha required to justify selling a lot still in the
# STCG zone (held 6–12 months) rather than waiting for LTCG qualification.
#
# Rationale: selling STCG vs. LTCG costs ~17–22pp in effective tax rate for
# most high-income earners (37% ordinary − 20% LTCG = 17pp; add 3.8% NIIT
# and state taxes for an upper bound near 22pp).  0.18 is the mid-range
# breakeven: if the model predicts less than 18% alpha, the tax savings from
# waiting a few weeks/months to cross the 365-day threshold likely exceed the
# opportunity cost of holding the concentrated position slightly longer.
#
# The 6–12 month zone is defined as: 180 < holding_days_at_vest <= 365.
# Lots held < 180 days have too long to wait; lots > 365 days are LTCG.
STCG_BREAKEVEN_THRESHOLD: float = 0.18
# Lower bound of the STCG boundary zone (days held, exclusive).
STCG_ZONE_MIN_DAYS: int = 180
# Upper bound of the STCG boundary zone — day 365 triggers LTCG.
STCG_ZONE_MAX_DAYS: int = 365

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

# v4.0 CPCV parameters — v5.0: upgraded from C(6,2)=15 paths to C(8,2)=28 paths
CPCV_N_FOLDS: int = 8         # Number of folds for CombinatorialPurgedCV (v5.0: was 6)
CPCV_N_TEST_FOLDS: int = 2    # Test folds per split; yields C(8,2)=28 paths (v5.0)

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
# v4.3 — Signal Quality + Confidence Layer
# ---------------------------------------------------------------------------
# Redundant features dropped from the final feature matrix to improve the
# obs/feature ratio from ~3.5:1 to ~4:1.
#   vol_21d:          highly correlated with vol_63d (same signal, shorter window)
#   credit_spread_ig: subset of credit_spread_hy (HY spread contains the IG
#                     signal plus the distress premium; IG is redundant)
FEATURES_TO_DROP: list[str] = ["vol_21d", "credit_spread_ig"]

# v8.6: model-specific production feature sets from the completed v7/v8
# ablation program.  GBT performs best with the lean Group B macro regime set.
# ElasticNet performs best with Group B plus a narrow investment/ROE/
# underwriting extension, avoiding the full Group E stack's low obs/feature ratio.
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
    "gbt": MODEL_FEATURE_BASE_GROUP_B,
    "elasticnet": MODEL_FEATURE_BASE_GROUP_B + [
        "investment_income_growth_yoy",
        "roe_net_income_ttm",
        "underwriting_income",
    ],
    "ridge": MODEL_FEATURE_BASE_GROUP_B + [
        "combined_ratio_ttm",
        "investment_income_growth_yoy",
        "roe_net_income_ttm",
    ],
    "bayesian_ridge": MODEL_FEATURE_BASE_GROUP_B + [
        "combined_ratio_ttm",
        "investment_income_growth_yoy",
        "roe_net_income_ttm",
        "buyback_yield",
        "buyback_acceleration",
    ],
}

# Use BayesianRidge posterior variance (σ²_pred) as the Ω diagonal in the
# Black-Litterman model instead of MAE².
BL_USE_BAYESIAN_VARIANCE: bool = True

# ---------------------------------------------------------------------------
# v4.3.1 — Diagnostic OOS Evaluation Report thresholds
# Used in _write_diagnostic_report() to flag model health vs. peer-review
# benchmarks (Harvey et al. 2016; Campbell & Thompson 2008; Gu et al. 2020).
# ---------------------------------------------------------------------------
# Campbell-Thompson OOS R²: >2% = good, 0.5–2% = marginal, <0% = failing.
DIAG_MIN_OOS_R2: float = 0.02
# Newey-West HAC-adjusted Spearman IC: >0.07 = good, 0.03–0.07 = marginal.
DIAG_MIN_IC: float = 0.07
# Hit rate (directional accuracy): >55% = good, 52–55% = marginal.
DIAG_MIN_HIT_RATE: float = 0.55
# CPCV positive paths (out of C(8,2)=28): ≥19 = good (~67%), 14–18 = marginal.
# v5.0: updated from C(6,2)=15 thresholds (was: ≥13/15 good, 10–12 marginal).
DIAG_CPCV_MIN_POSITIVE_PATHS: int = 19

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
# v5.1 — Probability Calibration
# ---------------------------------------------------------------------------
# Minimum OOS observations required before activating each calibration tier.
# At n < CALIBRATION_MIN_OBS_PLATT the raw BayesianRidge posterior is returned.
# At n >= CALIBRATION_MIN_OBS_ISOTONIC the two-stage Platt → Isotonic model
# is used, which is non-parametric and benefits from larger samples.
# ---------------------------------------------------------------------------
# v5.2 — Conformal Prediction Intervals
# ---------------------------------------------------------------------------
# Nominal coverage for the prediction interval shown in recommendation.md.
# 0.80 = 80% CI; interpretation: "over any 5 predictions, at least 4 will
# contain the true 6M relative return."
CONFORMAL_COVERAGE: float = 0.80
# Method: "aci" (Adaptive Conformal Inference; handles distribution shift)
# or "split" (standard split conformal; simpler, slightly narrower intervals).
CONFORMAL_METHOD: str = "aci"
# ACI step size γ — controls how fast α_t adapts to coverage misses.
# 0.05 is the default from Gibbs & Candès (2021); smaller = slower adaptation.
CONFORMAL_ACI_GAMMA: float = 0.05

# ---------------------------------------------------------------------------
# v5.1 — Probability Calibration
# ---------------------------------------------------------------------------
CALIBRATION_MIN_OBS_PLATT: int = 20     # Activate Platt scaling above this n
# Isotonic requires far more per-benchmark data to avoid plateau collapse on
# out-of-sample inputs.  With n=78–260 per benchmark (2026), the step function
# returns a single constant for most live predictions.  Re-evaluate at ~500+.
CALIBRATION_MIN_OBS_ISOTONIC: int = 500  # Per-benchmark isotonic threshold
# ECE computation parameters
CALIBRATION_N_BINS: int = 10            # Equal-width probability bins for ECE
CALIBRATION_BOOTSTRAP_REPS: int = 500   # Block bootstrap replications for ECE CI

# ---------------------------------------------------------------------------
# v6.0 — Beta-Transformed Linear Pool (BLP) aggregation
# Replaces naive equal-weight ensemble averaging with a calibrated pool.
# Ranjan & Gneiting (2010): any linear pool of calibrated forecasts is
# necessarily uncalibrated; BLP corrects this via a Beta CDF transformation.
#
# The BLP has 5 free parameters:
#   a, b   — Beta distribution shape parameters (the transformation)
#   w_1..3 — Linear pool weights for models 1-3; w_4 = 1 − Σw_1..3
#             (4 models → 3 independent weights)
#
# Parameter fitting uses maximum likelihood on OOS probability sequences.
# Requires BLP_MIN_OOS_MONTHS months of live OOS predictions to accumulate
# before fitting — guard enforced by BLPModel.fit().
# Target activation: Week 8 = 2026-05-20 (counting from first live run).
# ---------------------------------------------------------------------------
BLP_MIN_OOS_MONTHS: int = 12     # Minimum live OOS months required to fit BLP
BLP_N_PARAMS: int = 5            # 2 Beta shape + 3 independent weights
BLP_BETA_A_INIT: float = 1.0     # Initial Beta(a) shape (a=b=1 → uniform)
BLP_BETA_B_INIT: float = 1.0     # Initial Beta(b) shape
BLP_WEIGHT_INIT: float = 0.25    # Initial equal weight per model (4 models)

