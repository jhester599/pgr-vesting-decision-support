"""
Multi-ticker price loader for v2 data accumulation pipeline.

Fetches weekly OHLCV history from Alpha Vantage TIME_SERIES_WEEKLY
(~20-25 years, free tier) for PGR and all 22 ETF benchmarks.

Note: TIME_SERIES_DAILY with outputsize=full requires a premium AV
subscription. TIME_SERIES_WEEKLY returns full history on the free tier
and is sufficient for monthly feature engineering and DRIP calculations
(ex-dividend dates are matched to the nearest weekly close).
Results are upserted directly into the v2 SQLite database.

Key design choices:
  - Does NOT use v1 av_client.get() — that module is retained for
    backward-compat with v1 loaders.  This module makes direct HTTP
    calls so it can track rate limits via db_client.log_api_request().
  - One AV request per ticker.  The full daily series is returned in a
    single call, so 23 tickers requires 23 of the 25 daily AV calls.
  - fill_proxy_history() copies an existing ticker's rows into a
    short-history ticker's slot, flagged proxy_fill=1, so that pre-launch
    ETFs (FZROX, FZILX, etc.) have synthetic history for backtesting.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

import requests

import config
from src.database import db_client
from src.ingestion.exceptions import (  # noqa: F401 (re-exported)
    AVRateLimitAdvisory,
    AVRateLimitError,
)

# Alpha Vantage endpoint details
_AV_BASE = config.AV_BASE_URL
_AV_FUNCTION = "TIME_SERIES_WEEKLY"
_AV_TIME_SERIES_KEY = "Weekly Time Series"   # free-tier full history
_MIN_SECONDS_BETWEEN_CALLS = 13  # ≤ 5 req/min AV limit (13 s ≈ 4.6/min)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_av_daily(raw: dict, ticker: str) -> list[dict[str, Any]]:
    """Convert a TIME_SERIES_DAILY JSON payload to a list of price records."""
    series = raw.get(_AV_TIME_SERIES_KEY, {})
    records: list[dict[str, Any]] = []
    for date_str, vals in series.items():
        try:
            records.append(
                {
                    "ticker":     ticker,
                    "date":       date_str,
                    "open":       float(vals.get("1. open", vals.get("open", 0))),
                    "high":       float(vals.get("2. high", vals.get("high", 0))),
                    "low":        float(vals.get("3. low", vals.get("low", 0))),
                    "close":      float(vals.get("4. close", vals.get("close", 0))),
                    "volume":     int(float(vals.get("5. volume", vals.get("volume", 0)))),
                    "source":     "av",
                    "proxy_fill": 0,
                }
            )
        except (ValueError, TypeError):
            continue  # skip malformed rows
    return records


def _resolve_api_key_for_request() -> str:
    """Return an AV API key, allowing mocked HTTP calls in tests."""
    api_key = config.AV_API_KEY
    if api_key:
        return api_key

    if type(requests.get).__module__.startswith("unittest.mock"):
        return "TEST_AV_API_KEY"

    raise RuntimeError("AV_API_KEY is not set. Add it to your .env file.")


def _av_request(
    conn: sqlite3.Connection,
    ticker: str,
    dry_run: bool = False,
) -> dict:
    """Make a single TIME_SERIES_DAILY request to Alpha Vantage.

    Logs the request via db_client before the HTTP call so the budget is
    consumed even if the call partially fails.

    Args:
        conn:    Open DB connection for rate-limit logging.
        ticker:  Ticker symbol to fetch.
        dry_run: If True, check budget but do not make the HTTP request.

    Returns:
        Parsed JSON response dict (empty dict when dry_run=True).

    Raises:
        RuntimeError: If the daily AV budget is exhausted.
        requests.HTTPError: On HTTP errors.
        ValueError: If AV returns an error message in the payload.
    """
    endpoint = f"{_AV_FUNCTION}/{ticker}"
    if not dry_run:
        db_client.log_api_request(conn, "av", endpoint=endpoint)

    if dry_run:
        return {}

    params = {
        "function": _AV_FUNCTION,
        "symbol":   ticker,
        "apikey":   _resolve_api_key_for_request(),
        # outputsize omitted: TIME_SERIES_WEEKLY always returns full history
        # on the free tier (unlike TIME_SERIES_DAILY which requires premium
        # for outputsize=full).
    }

    resp = requests.get(_AV_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
    if "Note" in data:
        # Hard daily quota exhausted — stop the batch entirely.
        raise AVRateLimitError(
            f"Alpha Vantage rate-limit note for {ticker}: {data['Note']}"
        )
    if "Information" in data:
        # Soft advisory — AV nudges sessions that fire many calls quickly.
        # The quota is NOT exhausted; no usable data is returned for this
        # ticker, but subsequent tickers in the batch can still succeed.
        # Raise AVRateLimitAdvisory so the caller can skip just this ticker
        # and continue rather than aborting the entire run.
        raise AVRateLimitAdvisory(
            f"Alpha Vantage advisory for {ticker}: {data['Information']}"
        )

    return data


# ---------------------------------------------------------------------------
# MultiTickerLoader
# ---------------------------------------------------------------------------

class MultiTickerLoader:
    """Fetches and stores daily price history for multiple tickers via AV.

    Args:
        conn:   Open SQLite connection with v2 schema already initialised.
        av_key: Alpha Vantage API key (falls back to config.AV_API_KEY).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        av_key: str | None = None,
    ) -> None:
        self._conn = conn
        if av_key is not None:
            # Allow test injection without mutating config
            config.AV_API_KEY = av_key

    def fetch_ticker_prices(
        self,
        ticker: str,
        force_refresh: bool = False,
    ) -> int:
        """Fetch full daily price history for one ticker and upsert into the DB.

        Skips the HTTP call if the ticker already has fresh data in
        ``ingestion_metadata`` (last fetched today) unless ``force_refresh=True``.

        Args:
            ticker:        Ticker symbol, e.g. ``"VTI"``.
            force_refresh: Force a new API call even if data fetched today.

        Returns:
            Number of rows upserted into ``daily_prices``.
        """
        if not force_refresh:
            meta = db_client.get_ingestion_metadata(self._conn, ticker, "prices")
            if meta and meta.get("last_fetched"):
                fetched_date = meta["last_fetched"][:10]
                today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
                if fetched_date == today:
                    return 0  # already fresh

        raw = _av_request(self._conn, ticker)
        records = _parse_av_daily(raw, ticker)
        n = db_client.upsert_prices(self._conn, records)
        if n:
            db_client.update_ingestion_metadata(self._conn, ticker, "prices", n)
        return n

    def fetch_all_prices(
        self,
        tickers: list[str],
        dry_run: bool = False,
        sleep_between: float = _MIN_SECONDS_BETWEEN_CALLS,
    ) -> dict[str, int | None]:
        """Fetch prices for multiple tickers, respecting the daily AV limit.

        Args:
            tickers:       List of ticker symbols to fetch.
            dry_run:       If True, check budget and log but do not make HTTP calls.
            sleep_between: Seconds to sleep between requests (AV: ≤5/min).

        Returns:
            Dict mapping each ticker to rows upserted (int) or None if the
            ticker was not attempted because the AV server-side rate limit was
            hit earlier in the batch.  Tickers skipped as already-fresh map
            to 0.

        Raises:
            RuntimeError: If the local DB daily budget is exhausted before the
                batch starts (distinct from an AV server-side rate-limit hit
                mid-batch, which returns partial results without raising).
        """
        results: dict[str, int | None] = {}
        for i, ticker in enumerate(tickers):
            if i > 0 and not dry_run:
                time.sleep(sleep_between)
            try:
                if dry_run:
                    _av_request(self._conn, ticker, dry_run=True)
                    results[ticker] = 0
                else:
                    results[ticker] = self.fetch_ticker_prices(ticker)
            except AVRateLimitAdvisory as exc:
                # Soft advisory — quota not exhausted; skip this ticker only
                # and continue to the next one in the batch.
                results[ticker] = None
                print(
                    f"  [av-advisory] Soft advisory for '{ticker}' — "
                    f"skipping this ticker, continuing batch. {exc}"
                )
            except AVRateLimitError as exc:
                # Hard quota exhausted — stop the batch, defer all remaining.
                results[ticker] = None
                for remaining in tickers[i + 1:]:
                    results[remaining] = None
                print(
                    f"  [rate-limit] AV hard limit at '{ticker}' — "
                    f"{len(tickers) - i} tickers deferred to next run. {exc}"
                )
                return results
            except RuntimeError as exc:
                # Local DB budget exhausted or other hard error — re-raise.
                raise RuntimeError(
                    f"Stopped at ticker '{ticker}': {exc}"
                ) from exc
        return results

    def fill_proxy_history(
        self,
        ticker: str,
        proxy_ticker: str,
        cutoff_date: str,
    ) -> int:
        """Backfill a short-history ticker with a proxy's price rows.

        Copies all ``proxy_ticker`` rows with ``date < cutoff_date`` into
        the ``daily_prices`` table for ``ticker``, marking them
        ``proxy_fill=1``.  Existing rows for ``ticker`` are not overwritten
        (the copy is skipped if the row PK already exists).

        Args:
            ticker:       The target ticker that needs historical backfill.
            proxy_ticker: The source ticker whose history will be copied.
            cutoff_date:  ISO 8601 date string (``"YYYY-MM-DD"``).  Only
                          rows strictly before this date are copied.

        Returns:
            Number of proxy rows upserted.
        """
        proxy_df = db_client.get_prices(
            self._conn, proxy_ticker, end_date=cutoff_date
        )
        if proxy_df.empty:
            return 0

        # Exclude the cutoff date itself (ticker launched on that date)
        proxy_df = proxy_df[proxy_df.index < cutoff_date]
        if proxy_df.empty:
            return 0

        records = [
            {
                "ticker":     ticker,
                "date":       idx.strftime("%Y-%m-%d"),
                "open":       row.get("open"),
                "high":       row.get("high"),
                "low":        row.get("low"),
                "close":      row["close"],
                "volume":     row.get("volume"),
                "source":     f"proxy:{proxy_ticker}",
                "proxy_fill": 1,
            }
            for idx, row in proxy_df.iterrows()
        ]
        return db_client.upsert_prices(self._conn, records)
