"""
Alpha Vantage REST API client.

Implements cache-first retrieval with a daily request counter guard.
All HTTP responses are cached to data/raw/ as JSON. The Alpha Vantage
free tier allows 25 requests/day. After initial population, subsequent
runs consume 0 API calls.

Cache invalidation:
  - Technical indicator endpoints: 24 hours (price series update daily)
"""

import json
import os
import time
from datetime import datetime
from typing import Any

import requests

import config
from src.ingestion.http_utils import build_retry_session
from src.ingestion.provider_registry import get_provider_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(function: str, extra_params: dict) -> str:
    """Return the local path for a cached JSON file."""
    param_str = "_".join(f"{k}{v}" for k, v in sorted(extra_params.items()))
    key = f"av_{function}_{param_str}" if param_str else f"av_{function}"
    key = key.replace("/", "_").replace("?", "_").replace("&", "_")
    return os.path.join(config.DATA_RAW_DIR, f"{key}.json")


def _load_counts() -> dict:
    if not os.path.exists(config.REQUEST_COUNTS_FILE):
        return {}
    with open(config.REQUEST_COUNTS_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_counts(counts: dict) -> None:
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    with open(config.REQUEST_COUNTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(counts, fh)


def _increment_av_count() -> int:
    """Increment today's Alpha Vantage request counter; raise if limit exceeded."""
    provider = get_provider_spec("av")
    counts = _load_counts()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    av_key = f"av_{today}"
    counts[av_key] = counts.get(av_key, 0) + 1
    limit = provider.daily_limit or config.AV_DAILY_LIMIT
    if provider.enforce_limit and counts[av_key] > limit:
        raise RuntimeError(
            f"Alpha Vantage daily request limit ({limit}) reached for {today}. "
            "Use cached data or wait until tomorrow."
        )
    _save_counts(counts)
    return counts[av_key]


def _is_cache_valid(path: str, max_age_hours: int) -> bool:
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < max_age_hours * 3600


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get(
    function: str,
    params: dict[str, Any] | None = None,
    cache_hours: int = 24,
) -> dict:
    """
    Fetch data from an Alpha Vantage endpoint, using a local cache when valid.

    Args:
        function:    Alpha Vantage function name, e.g. ``"SMA"``, ``"RSI"``.
        params:      Additional query-string parameters (function, symbol, apikey
                     are injected automatically; pass extras like time_period, interval).
        cache_hours: Cache TTL in hours (default 24).

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: If the Alpha Vantage daily request limit is exceeded.
        requests.HTTPError: On non-2xx HTTP responses.
        ValueError: If Alpha Vantage returns an error message in the payload.
    """
    if params is None:
        params = {}

    cache_key_params = {k: v for k, v in params.items() if k != "apikey"}
    path = _cache_path(function, cache_key_params)

    if _is_cache_valid(path, cache_hours):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # Cache miss — consume an API request.
    _increment_av_count()
    full_params = {
        "function": function,
        "symbol": config.TICKER,
        "apikey": config.AV_API_KEY,
        **params,
    }

    if config.AV_API_KEY is None:
        raise RuntimeError(
            "AV_API_KEY is not set. Add it to your .env file."
        )
    session = build_retry_session()
    response = session.get(config.AV_BASE_URL, params=full_params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Alpha Vantage signals errors inside the JSON body, not via HTTP status.
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate-limit note: {data['Note']}")

    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)

    return data
