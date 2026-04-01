"""
Multi-ticker dividend loader for the v2 data accumulation pipeline.

Fetches full dividend history from Alpha Vantage DIVIDENDS endpoint for
PGR and all 22 ETF benchmarks.  Results are upserted into the v2 SQLite
database for use by DRIP total-return calculations.

One AV request per ticker.  The DIVIDENDS endpoint returns all historical
ex-dividend events in a single response (no pagination).
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

import requests

import config
from src.database import db_client
from src.ingestion.exceptions import AVRateLimitAdvisory, AVRateLimitError

_AV_BASE = config.AV_BASE_URL
_AV_FUNCTION = "DIVIDENDS"
_MIN_SECONDS_BETWEEN_CALLS = 13  # ≤ 5 req/min AV limit


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_av_dividends(raw: dict, ticker: str) -> list[dict[str, Any]]:
    """Convert an AV DIVIDENDS JSON payload to a list of dividend records.

    AV response structure::

        {
            "symbol": "PGR",
            "data": [
                {
                    "ex_dividend_date": "2024-03-21",
                    "declaration_date": "...",
                    "record_date": "...",
                    "payment_date": "...",
                    "amount": "0.10"
                },
                ...
            ]
        }
    """
    records: list[dict[str, Any]] = []
    for item in raw.get("data", []):
        ex_date = item.get("ex_dividend_date", "")
        amount_str = item.get("amount", "0")
        if not ex_date or not amount_str:
            continue
        try:
            amount = float(amount_str)
        except ValueError:
            continue
        if amount <= 0:
            continue  # AV sometimes emits $0.00 placeholder rows
        records.append(
            {
                "ticker":  ticker,
                "ex_date": ex_date,
                "amount":  amount,
                "source":  "av",
            }
        )
    return records


def _av_dividend_request(
    conn: sqlite3.Connection,
    ticker: str,
    dry_run: bool = False,
) -> dict:
    """Make a single DIVIDENDS request to Alpha Vantage.

    Logs the request via db_client before the HTTP call.

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
        "apikey":   config.AV_API_KEY,
    }
    if config.AV_API_KEY is None:
        raise RuntimeError("AV_API_KEY is not set. Add it to your .env file.")

    resp = requests.get(_AV_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage dividend error for {ticker}: {data['Error Message']}")
    if "Note" in data:
        # Hard daily quota exhausted — stop the batch entirely.
        raise AVRateLimitError(
            f"Alpha Vantage rate-limit note for {ticker}: {data['Note']}"
        )
    if "Information" in data:
        # Soft advisory — quota not exhausted; no usable data for this ticker,
        # but subsequent tickers can still succeed.  Raise AVRateLimitAdvisory
        # so the caller can skip just this ticker and continue the batch.
        raise AVRateLimitAdvisory(
            f"Alpha Vantage advisory for {ticker}: {data['Information']}"
        )

    return data


# ---------------------------------------------------------------------------
# MultiDividendLoader
# ---------------------------------------------------------------------------

class MultiDividendLoader:
    """Fetches and stores full dividend history for multiple tickers via AV.

    Args:
        conn: Open SQLite connection with v2 schema initialised.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def fetch_dividends(
        self,
        ticker: str,
        force_refresh: bool = False,
    ) -> int:
        """Fetch full dividend history for one ticker and upsert into the DB.

        Skips the call if dividends were already fetched today (checked via
        ``ingestion_metadata``) unless ``force_refresh=True``.

        Args:
            ticker:        Ticker symbol, e.g. ``"PGR"``.
            force_refresh: Force API call even if data fetched today.

        Returns:
            Number of rows upserted into ``daily_dividends``.
        """
        if not force_refresh:
            meta = db_client.get_ingestion_metadata(self._conn, ticker, "dividends")
            if meta and meta.get("last_fetched"):
                fetched_date = meta["last_fetched"][:10]
                if fetched_date == datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"):
                    return 0  # already fresh

        raw = _av_dividend_request(self._conn, ticker)
        records = _parse_av_dividends(raw, ticker)
        n = db_client.upsert_dividends(self._conn, records)
        if n:
            db_client.update_ingestion_metadata(self._conn, ticker, "dividends", n)
        return n

    def fetch_for_tickers(
        self,
        tickers: list[str],
        dry_run: bool = False,
        sleep_between: float = _MIN_SECONDS_BETWEEN_CALLS,
    ) -> dict[str, int | None]:
        """Fetch dividends for multiple tickers.

        Args:
            tickers:       List of ticker symbols.
            dry_run:       Check budget and log but skip HTTP calls.
            sleep_between: Seconds between requests.

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
                    _av_dividend_request(self._conn, ticker, dry_run=True)
                    results[ticker] = 0
                else:
                    results[ticker] = self.fetch_dividends(ticker)
            except AVRateLimitAdvisory as exc:
                # Soft advisory — quota not exhausted; skip this ticker only.
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
                raise RuntimeError(
                    f"Stopped at ticker '{ticker}': {exc}"
                ) from exc
        return results
