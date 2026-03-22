"""
Financial Modeling Prep (FMP) REST API client.

Implements cache-first retrieval with a daily request counter guard.
All HTTP responses are cached to data/raw/ as JSON files. On subsequent
calls the cached file is returned without consuming an API request.

Cache invalidation policy:
  - Price / dividend / split endpoints: 24 hours
  - Fundamental endpoints (income-statement, key-metrics): 7 days
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any

import requests

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(endpoint_key: str) -> str:
    """Return the local path for a cached JSON file."""
    safe_key = endpoint_key.replace("/", "_").replace("?", "_").replace("&", "_")
    return os.path.join(config.DATA_RAW_DIR, f"fmp_{safe_key}.json")


def _load_counts() -> dict:
    if not os.path.exists(config.REQUEST_COUNTS_FILE):
        return {}
    with open(config.REQUEST_COUNTS_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_counts(counts: dict) -> None:
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    with open(config.REQUEST_COUNTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(counts, fh)


def _increment_fmp_count() -> int:
    """Increment today's FMP request counter; raise if limit exceeded."""
    counts = _load_counts()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fmp_key = f"fmp_{today}"
    counts[fmp_key] = counts.get(fmp_key, 0) + 1
    if counts[fmp_key] > config.FMP_DAILY_LIMIT:
        raise RuntimeError(
            f"FMP daily request limit ({config.FMP_DAILY_LIMIT}) reached for {today}. "
            "Use cached data or wait until tomorrow."
        )
    _save_counts(counts)
    return counts[fmp_key]


def _is_cache_valid(path: str, max_age_hours: int) -> bool:
    """Return True if a cached file exists and is younger than max_age_hours."""
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    age = time.time() - mtime
    return age < max_age_hours * 3600


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get(
    endpoint: str,
    params: dict[str, Any] | None = None,
    cache_hours: int = 24,
) -> Any:
    """
    Fetch data from a FMP REST endpoint, using a local cache when valid.

    Args:
        endpoint: FMP API path, e.g. ``"/v3/historical-price-full/PGR"``.
        params:   Additional query-string parameters (apikey is injected automatically).
        cache_hours: Cache TTL in hours. Use 24 for prices, 168 (7*24) for fundamentals.

    Returns:
        Parsed JSON response (dict or list depending on the endpoint).

    Raises:
        RuntimeError: If the FMP daily request limit is exceeded.
        requests.HTTPError: On non-2xx HTTP responses.
    """
    if params is None:
        params = {}

    # Build a deterministic cache key from endpoint + sorted params (excluding apikey).
    param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    endpoint_key = f"{endpoint}_{param_str}" if param_str else endpoint
    path = _cache_path(endpoint_key)

    if _is_cache_valid(path, cache_hours):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # Cache miss — consume an API request.
    _increment_fmp_count()
    full_params = {**params, "apikey": config.FMP_API_KEY}
    url = f"{config.FMP_BASE_URL}{endpoint}"

    if config.FMP_API_KEY is None:
        raise RuntimeError(
            "FMP_API_KEY is not set. Add it to your .env file."
        )
    response = requests.get(url, params=full_params, timeout=30)
    response.raise_for_status()
    data = response.json()

    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)

    return data
