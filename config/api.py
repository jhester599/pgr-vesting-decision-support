"""
API credentials, base URLs, EDGAR helpers, rate limits, HTTP retry settings,
and data freshness thresholds.
"""

import os

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
# Data freshness thresholds (peer-review operational safety checks)
# ---------------------------------------------------------------------------
DATA_FRESHNESS_MAX_PRICE_AGE_DAYS: int = 10
# FRED freshness is checked per series against the observation month the
# live decision row needs (feature month minus the series' publication lag;
# review F07). A series may lag that month by this many months before it is
# STALE.
DATA_FRESHNESS_FRED_GRACE_MONTHS: int = 0
DATA_FRESHNESS_MAX_EDGAR_AGE_DAYS: int = 35
DATA_FRESHNESS_PGR_EDGAR_FILING_GRACE_DAYS: int = 25
# Dividends are STALE when the latest ex-date is older than the latest price
# date minus this many usual payment intervals (review F08).
DIVIDEND_FRESHNESS_INTERVAL_MULTIPLE: float = 1.5
# Budget-aware dividend refresh (scripts/weekly_fetch.py --dividend-refresh):
# re-fetch a ticker once its last dividend fetch is this many days old.
# Monthly payers (usual interval <= 45 days) are refreshed weekly so that they
# stay within 1.5 intervals; everything else about monthly.
DIVIDEND_REFRESH_MIN_AGE_DAYS: int = 27
DIVIDEND_REFRESH_MIN_AGE_DAYS_MONTHLY_PAYER: int = 6
# AV calls kept in reserve when sizing the dividend refresh batch.
DIVIDEND_REFRESH_AV_RESERVE: int = 2
HTTP_RETRY_TOTAL: int = 3
HTTP_RETRY_BACKOFF_FACTOR: float = 1.0
