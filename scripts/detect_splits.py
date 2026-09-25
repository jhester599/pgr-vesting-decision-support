"""
Detect splits from Alpha Vantage adjusted series and compare them with the registry.

Review 2026-09-25 (F03/F05): splits used to live only in hand-maintained
lists, and VOO's 2013 1-for-2 and VGT's 2026 8-for-1 splits were missing.
This script fetches ``TIME_SERIES_WEEKLY_ADJUSTED`` (free tier; one call per
ticker), recovers split coefficients from the adjusted-close factor (see
``src/ingestion/split_detector.py``) and reports every detection that is not
in ``config.KNOWN_SPLITS``.

It never writes ``split_history``: verified splits are added to
``config/splits.py`` by hand, then ``scripts/rebuild_relative_returns.py`` is
run.

Budget: one AV call per ticker, capped at ``--max-calls`` and at the AV calls
left today (``api_request_log``) minus a reserve of 2. Raw responses are
cached under ``data/raw/av_split_cache/`` (git-ignored) for the day, so a
re-run costs nothing. Each call is logged to ``api_request_log`` in ``--db``.

Usage:
    python scripts/detect_splits.py [--tickers VOO VGT] [--max-calls 10]
                                    [--db path/to/copy.db] [--dry-run]

Exit status is 1 when a detection is missing from, or disagrees with,
``config.KNOWN_SPLITS``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.ingestion.fetch_scheduler import get_all_price_tickers
from src.ingestion.http_utils import build_retry_session
from src.ingestion.split_detector import (
    detect_splits_from_adjusted_series,
    reconcile_detected_splits,
)
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)

_FUNCTION = "TIME_SERIES_WEEKLY_ADJUSTED"
_CACHE_DIR = Path(config.DATA_RAW_DIR) / "av_split_cache"
_SLEEP_SECONDS = 13  # <= 5 requests/minute
_RESERVE = 2


def _fetch_adjusted(conn, ticker: str, today: str) -> dict:
    cache = _CACHE_DIR / f"{ticker}_{today}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    db_client.log_api_request(conn, "av", endpoint=f"{_FUNCTION}/{ticker}")
    if not config.AV_API_KEY:
        raise RuntimeError("AV_API_KEY is not set.")
    resp = build_retry_session().get(
        config.AV_BASE_URL,
        params={"function": _FUNCTION, "symbol": ticker, "apikey": config.AV_API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    for key in ("Error Message", "Note", "Information"):
        if key in data:
            raise RuntimeError(f"Alpha Vantage {key} for {ticker}: {data[key]}")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect splits from AV adjusted series.")
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Tickers to check (default: PGR, ETF benchmarks, peers).")
    parser.add_argument("--max-calls", type=int, default=config.AV_DAILY_LIMIT)
    parser.add_argument("--db", default=config.DB_PATH,
                        help="DB used for the AV call budget log.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned tickers and budget; no HTTP calls.")
    args = parser.parse_args(argv)
    configure_logging()

    tickers = args.tickers or [*get_all_price_tickers(), *config.PEER_TICKER_UNIVERSE]
    # api_request_log is keyed by UTC date.
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    conn = db_client.get_connection(args.db, read_only=args.dry_run)
    used = db_client.get_api_request_count(conn, "av", today)
    budget = max(0, min(args.max_calls, config.AV_DAILY_LIMIT - used - _RESERVE))
    uncached = [t for t in tickers if not (_CACHE_DIR / f"{t}_{today}.json").exists()]
    planned = [t for t in tickers if t not in uncached] + uncached[:budget]
    skipped = uncached[budget:]
    logger.info("AV used today %s/%s; budget for this run %s calls.",
                used, config.AV_DAILY_LIMIT, budget)
    logger.info("Checking %s tickers: %s", len(planned), planned)
    if skipped:
        logger.warning("Over budget, not checked today: %s", skipped)
    if args.dry_run:
        conn.close()
        return 0

    detected: list[dict] = []
    first_call = True
    for ticker in planned:
        cached = (_CACHE_DIR / f"{ticker}_{today}.json").exists()
        if not cached:
            if not first_call:
                time.sleep(_SLEEP_SECONDS)
            first_call = False
        try:
            raw = _fetch_adjusted(conn, ticker, today)
        except RuntimeError as exc:
            logger.error("%s: %s", ticker, exc)
            continue
        found = detect_splits_from_adjusted_series(raw, ticker)
        for d in found:
            logger.info("%s: split %.4g (%s:%s) in week ending %s",
                        ticker, d["split_ratio"], d["numerator"], d["denominator"],
                        d["bar_date"])
        detected.extend(found)
    conn.close()

    issues = reconcile_detected_splits(detected, config.KNOWN_SPLITS)
    for issue in issues:
        logger.error(
            "%s %s: %s (detected %.4g, registry %s) - verify with the issuer "
            "notice and update config/splits.py",
            issue["ticker"], issue["bar_date"], issue["issue"],
            issue["detected_ratio"], issue["known_ratio"],
        )
    if not issues:
        logger.info("All %s detected splits match config.KNOWN_SPLITS.", len(detected))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
