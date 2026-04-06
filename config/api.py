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
DATA_FRESHNESS_MAX_FRED_AGE_DAYS: int = 45
DATA_FRESHNESS_MAX_EDGAR_AGE_DAYS: int = 35
HTTP_RETRY_TOTAL: int = 3
HTTP_RETRY_BACKOFF_FACTOR: float = 1.0
