"""
One-time (or quarterly) full data population script.

Fetches complete price and dividend history for the core ticker universe and
optionally loads full FRED history. Because the Alpha Vantage free tier allows
only 25 calls/day, price and dividend fetches are intended to run on separate
days.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.ingestion.fetch_scheduler import (
    get_all_dividend_tickers,
    get_all_price_tickers,
)
from src.ingestion.multi_dividend_loader import MultiDividendLoader
from src.ingestion.multi_ticker_loader import MultiTickerLoader
from src.logging_config import configure_logging


logger = logging.getLogger(__name__)

_DEFAULT_STATUS_FILE = os.path.join("data", "fetch_status.md")


class TickerResult(NamedTuple):
    ticker: str
    mode: str
    rows_upserted: int
    status: str
    detail: str


def _fetch_fred_step(conn, dry_run: bool = False) -> int:
    """Fetch full FRED macro history and upsert into fred_macro_monthly."""
    from src.ingestion.fred_loader import fetch_all_fred_macro, upsert_fred_to_db

    series_list = list(config.FRED_SERIES_MACRO) + list(
        getattr(config, "FRED_SERIES_PGR", [])
    )
    logger.info(
        "Fetching %s FRED series (no AV/FMP budget impact)...",
        len(series_list),
    )
    if dry_run:
        logger.info("[DRY RUN] Would fetch: %s", series_list)
        return 0

    if getattr(config, "FRED_API_KEY", None) is None:
        logger.warning("FRED_API_KEY not set - skipping FRED fetch.")
        return 0

    try:
        df = fetch_all_fred_macro(series_list)
        n = upsert_fred_to_db(conn, df)
        logger.info("FRED: %s rows upserted (%s series)", n, len(series_list))
        return n
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "FRED fetch failed. Continuing without FRED data. Error=%r",
            exc,
        )
        return 0


def main(
    do_prices: bool = False,
    do_dividends: bool = False,
    do_fred: bool = False,
    dry_run: bool = False,
    force: bool = False,
    status_file: str = _DEFAULT_STATUS_FILE,
) -> int:
    """Run the initial fetch and write a status report."""
    del force

    configure_logging()

    if not do_prices and not do_dividends and not do_fred:
        logger.error("Specify at least one of --prices, --dividends, --fred.")
        return 1

    run_start = datetime.now(timezone.utc)
    today = date.today()
    parts = []
    if do_prices:
        parts.append("prices")
    if do_dividends:
        parts.append("dividends")
    if do_fred:
        parts.append("FRED")
    mode_label = " + ".join(parts) if parts else "none"
    prefix = "[DRY RUN] " if dry_run else ""
    results: list[TickerResult] = []
    had_error = False

    logger.info("%sPGR v2 Initial Fetch - %s (%s)", prefix, today, mode_label)
    logger.info("Database: %s", config.DB_PATH)
    logger.info("Dry run: %s", dry_run)

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    all_tickers = get_all_price_tickers()

    if do_prices:
        logger.info("%sFetching prices for %s tickers...", prefix, len(all_tickers))
        try:
            loader = MultiTickerLoader(conn)
            price_results: dict[str, int] = loader.fetch_all_prices(
                all_tickers,
                dry_run=dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("MultiTickerLoader raised a fatal error. Error=%r", exc)
            had_error = True
            price_results = {t: 0 for t in all_tickers}

        for ticker, n_rows in price_results.items():
            if n_rows is None:
                status, detail = (
                    "DEFERRED",
                    "not attempted - AV rate limit reached earlier in batch",
                )
                rows = 0
            elif n_rows > 0:
                status, detail, rows = "OK", f"{n_rows} rows upserted", n_rows
            elif dry_run:
                status, detail, rows = "DRY-RUN", "no HTTP call made", 0
            else:
                status, detail, rows = "SKIPPED", "already fresh or no new data", 0
            results.append(TickerResult(ticker, "prices", rows, status, detail))
            marker = (
                "OK"
                if status == "OK"
                else ("--" if status in ("SKIPPED", "DRY-RUN", "DEFERRED") else "!!")
            )
            logger.info("[%s] %s %s", marker, f"{ticker:<6}", detail)

        price_ok = sum(1 for r in results if r.mode == "prices" and r.status == "OK")
        logger.info("Prices: %s/%s tickers loaded new data", price_ok, len(all_tickers))

    if do_dividends:
        div_tickers = get_all_dividend_tickers()
        logger.info(
            "%sFetching dividends for %s tickers...",
            prefix,
            len(div_tickers),
        )
        try:
            div_loader = MultiDividendLoader(conn)
            div_results: dict[str, int] = div_loader.fetch_for_tickers(
                div_tickers,
                dry_run=dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("MultiDividendLoader raised a fatal error. Error=%r", exc)
            had_error = True
            div_results = {t: 0 for t in div_tickers}

        for ticker, n_rows in div_results.items():
            if n_rows is None:
                status, detail = (
                    "DEFERRED",
                    "not attempted - AV rate limit reached earlier in batch",
                )
                rows = 0
            elif n_rows > 0:
                status, detail, rows = "OK", f"{n_rows} rows upserted", n_rows
            elif dry_run:
                status, detail, rows = "DRY-RUN", "no HTTP call made", 0
            else:
                status, detail, rows = "SKIPPED", "already fresh or no new data", 0
            results.append(TickerResult(ticker, "dividends", rows, status, detail))
            marker = (
                "OK"
                if status == "OK"
                else ("--" if status in ("SKIPPED", "DRY-RUN", "DEFERRED") else "!!")
            )
            logger.info("[%s] %s %s", marker, f"{ticker:<6}", detail)

        div_ok = sum(1 for r in results if r.mode == "dividends" and r.status == "OK")
        logger.info(
            "Dividends: %s/%s tickers loaded new data",
            div_ok,
            len(div_tickers),
        )

    if do_fred:
        _fetch_fred_step(conn, dry_run=dry_run)

    today_str = today.isoformat()
    if dry_run:
        av_projected = (
            (len(all_tickers) if do_prices else 0)
            + (len(get_all_dividend_tickers()) if do_dividends else 0)
        )
        logger.info("%sProjected AV calls: %s/%s", prefix, av_projected, config.AV_DAILY_LIMIT)
        if av_projected > config.AV_DAILY_LIMIT:
            logger.warning("Projected calls exceed daily limit.")
        av_used = av_projected
    else:
        av_used = db_client.get_api_request_count(conn, "av", today_str)
        logger.info("AV budget used today: %s/%s", av_used, config.AV_DAILY_LIMIT)

    conn.close()

    run_end = datetime.now(timezone.utc)
    duration_s = (run_end - run_start).total_seconds()
    overall_ok = not had_error and all(
        r.status in ("OK", "SKIPPED", "DRY-RUN", "DEFERRED") for r in results
    )
    n_ok = sum(1 for r in results if r.status == "OK")
    n_skip = sum(1 for r in results if r.status in ("SKIPPED", "DRY-RUN"))
    n_deferred = sum(1 for r in results if r.status == "DEFERRED")
    n_err = sum(1 for r in results if r.status == "ERROR")

    _write_status_file(
        status_file=status_file,
        run_dt=run_start,
        mode_label=mode_label,
        dry_run=dry_run,
        results=results,
        av_used=av_used,
        av_limit=config.AV_DAILY_LIMIT,
        duration_s=duration_s,
        overall_ok=overall_ok,
        n_deferred=n_deferred,
    )

    _write_github_step_summary(
        mode_label=mode_label,
        dry_run=dry_run,
        n_ok=n_ok,
        n_skip=n_skip,
        n_deferred=n_deferred,
        n_err=n_err,
        av_used=av_used,
        av_limit=config.AV_DAILY_LIMIT,
        results=results,
        overall_ok=overall_ok,
    )

    if n_deferred:
        badge = "[PARTIAL - RATE LIMITED]"
        deferred_note = (
            f", {n_deferred} deferred (re-run tomorrow for remaining tickers)"
        )
    else:
        badge = "[SUCCESS]" if overall_ok else "[PARTIAL FAILURE]"
        deferred_note = ""
    logger.info(
        "%s %s loaded, %s skipped, %s errors%s | %s/%s AV calls | %.0fs",
        badge,
        n_ok,
        n_skip,
        n_err,
        deferred_note,
        av_used,
        config.AV_DAILY_LIMIT,
        duration_s,
    )

    return 0 if overall_ok else 1


def _write_status_file(
    status_file: str,
    run_dt: datetime,
    mode_label: str,
    dry_run: bool,
    results: list[TickerResult],
    av_used: int,
    av_limit: int,
    duration_s: float,
    overall_ok: bool,
    n_deferred: int = 0,
) -> None:
    """Write (or append) a Markdown status report to ``status_file``."""
    Path(status_file).parent.mkdir(parents=True, exist_ok=True)

    if n_deferred:
        badge = "PARTIAL - RATE LIMITED"
    else:
        badge = "SUCCESS" if overall_ok else "PARTIAL FAILURE"
    dry_note = " *(dry run)*" if dry_run else ""
    ts = run_dt.strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "",
        f"## {ts} - `{mode_label}`{dry_note}  {badge}",
        "",
        f"- **AV calls used:** {av_used} / {av_limit}",
        f"- **Duration:** {duration_s:.0f}s",
        f"- **Tickers attempted:** {len(results)}",
        f"- **Loaded new data:** {sum(1 for r in results if r.status == 'OK')}",
        f"- **Skipped (no new data):** {sum(1 for r in results if r.status in ('SKIPPED', 'DRY-RUN'))}",
        f"- **Deferred (AV rate limit):** {n_deferred}" if n_deferred else "",
        f"- **Errors:** {sum(1 for r in results if r.status == 'ERROR')}",
        "",
        "| Ticker | Mode | Rows | Status | Detail |",
        "|--------|------|-----:|--------|--------|",
    ]
    for r in sorted(results, key=lambda x: (x.mode, x.ticker)):
        if r.status == "DEFERRED":
            icon = "PAUSE"
        elif r.status == "OK":
            icon = "OK"
        elif r.status in ("SKIPPED", "DRY-RUN"):
            icon = "SKIP"
        else:
            icon = "ERR"
        lines.append(
            f"| `{r.ticker}` | {r.mode} | {r.rows_upserted:,} "
            f"| {icon} {r.status} | {r.detail} |"
        )
    lines.append("")

    if not Path(status_file).exists():
        header = [
            "# PGR v2 Initial Fetch - Status Log",
            "",
            "Appended each time `initial_fetch.py` runs (via GitHub Actions or locally).",
            "Most-recent run appears **last**.",
            "",
        ]
        Path(status_file).write_text("\n".join(header + lines), encoding="utf-8")
    else:
        with open(status_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    logger.info("Status report written -> %s", status_file)


def _write_github_step_summary(
    mode_label: str,
    dry_run: bool,
    n_ok: int,
    n_skip: int,
    n_deferred: int,
    n_err: int,
    av_used: int,
    av_limit: int,
    results: list[TickerResult],
    overall_ok: bool,
) -> None:
    """Write a summary to ``$GITHUB_STEP_SUMMARY`` when running in Actions."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    if n_deferred:
        badge = "PARTIAL - RATE LIMITED"
    else:
        badge = "SUCCESS" if overall_ok else "PARTIAL FAILURE"
    dry_note = " *(dry run)*" if dry_run else ""
    deferred_row = (
        [f"| Tickers deferred (re-run tomorrow) | {n_deferred} |"]
        if n_deferred else []
    )
    lines = [
        f"# PGR Initial Fetch - `{mode_label}`{dry_note}",
        "",
        f"## {badge}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Tickers loaded | **{n_ok}** |",
        f"| Tickers skipped | {n_skip} |",
        *deferred_row,
        f"| Errors | {'**' + str(n_err) + '**' if n_err else '0'} |",
        f"| AV API calls | {av_used} / {av_limit} |",
        "",
        "<details><summary>Per-ticker detail</summary>",
        "",
        "| Ticker | Mode | Rows | Status |",
        "|--------|------|-----:|--------|",
    ]
    for r in sorted(results, key=lambda x: (x.mode, x.ticker)):
        if r.status == "DEFERRED":
            icon = "PAUSE"
        elif r.status == "OK":
            icon = "OK"
        elif r.status in ("SKIPPED", "DRY-RUN"):
            icon = "SKIP"
        else:
            icon = "ERR"
        lines.append(
            f"| `{r.ticker}` | {r.mode} | {r.rows_upserted:,} | {icon} {r.status} |"
        )
    lines += ["", "</details>", ""]

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGR v2 one-time full data population (run --prices and "
        "--dividends on separate days to stay within AV limits)."
    )
    parser.add_argument(
        "--prices",
        action="store_true",
        help="Fetch full price history for all tickers.",
    )
    parser.add_argument(
        "--dividends",
        action="store_true",
        help="Fetch full dividend history for all tickers.",
    )
    parser.add_argument(
        "--fred",
        action="store_true",
        help="Fetch full FRED macro history (free API, no AV budget impact).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without making HTTP calls.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if data was already fetched today.",
    )
    parser.add_argument(
        "--status-file",
        default=_DEFAULT_STATUS_FILE,
        help=f"Path for Markdown status report (default: {_DEFAULT_STATUS_FILE}).",
    )
    args = parser.parse_args()
    sys.exit(
        main(
            do_prices=args.prices,
            do_dividends=args.dividends,
            do_fred=args.fred,
            dry_run=args.dry_run,
            force=args.force,
            status_file=args.status_file,
        )
    )
