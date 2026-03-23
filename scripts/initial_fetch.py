"""
One-time (or quarterly) full data population script.

Fetches complete price AND dividend history for all 23 tickers
(PGR + 20 ETF benchmarks) from Alpha Vantage.  Because the free tier
allows only 25 calls/day, the two fetches must be run on separate days:

    Day 1:  python scripts/initial_fetch.py --prices
    Day 2:  python scripts/initial_fetch.py --dividends
    Day 3:  python scripts/initial_fetch.py --fred     # free API, any day

After both price and dividend days the database will have complete historical
data back to ~2000 for all tickers that AV covers.  FRED can be run any day
as it uses a separate free public API with no daily call limit.

Usage:
    python scripts/initial_fetch.py --prices     [--dry-run]
    python scripts/initial_fetch.py --dividends  [--dry-run]
    python scripts/initial_fetch.py --fred       [--dry-run]
    python scripts/initial_fetch.py --prices --dividends   # NOT recommended: exceeds 25 call/day

Options:
    --prices        Fetch full weekly price history for all 23 tickers (23 AV calls).
    --dividends     Fetch full dividend history for all 23 tickers (23 AV calls).
    --fred          Fetch full FRED macro history (free API, no AV/FMP budget impact).
    --dry-run       Log which tickers would be fetched but make no HTTP calls.
    --force         Re-fetch even if data was already fetched today.
    --status-file   Path to write a Markdown status report (default: data/fetch_status.md).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import db_client
from src.ingestion.fetch_scheduler import get_all_dividend_tickers, get_all_price_tickers
from src.ingestion.multi_dividend_loader import MultiDividendLoader
from src.ingestion.multi_ticker_loader import MultiTickerLoader


_DEFAULT_STATUS_FILE = os.path.join("data", "fetch_status.md")


class TickerResult(NamedTuple):
    ticker: str
    mode: str           # "prices" or "dividends"
    rows_upserted: int
    status: str         # "OK", "SKIPPED", "DEFERRED", "DRY-RUN", "ERROR"
    detail: str         # row count or error message


def _fetch_fred_step(conn, dry_run: bool = False) -> int:
    """Fetch full FRED macro history and upsert into fred_macro_monthly.

    Fetches all series in ``config.FRED_SERIES_MACRO`` + ``config.FRED_SERIES_PGR``.
    FRED is a free public API with no daily call limit; this step has no impact
    on the AV or FMP budget counters.

    Returns:
        Total rows upserted across all series.
    """
    from src.ingestion.fred_loader import fetch_all_fred_macro, upsert_fred_to_db

    series_list = list(config.FRED_SERIES_MACRO) + list(getattr(config, "FRED_SERIES_PGR", []))
    print(f"\nFetching {len(series_list)} FRED series (no AV/FMP budget impact)...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch: {series_list}")
        return 0

    if getattr(config, "FRED_API_KEY", None) is None:
        print("  WARNING: FRED_API_KEY not set — skipping FRED fetch.")
        return 0

    try:
        df = fetch_all_fred_macro(series_list)
        n = upsert_fred_to_db(conn, df)
        print(f"  FRED: {n} rows upserted ({len(series_list)} series)")
        return n
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: FRED fetch failed: {exc}. Continuing without FRED data.")
        return 0


def main(
    do_prices: bool = False,
    do_dividends: bool = False,
    do_fred: bool = False,
    dry_run: bool = False,
    force: bool = False,
    status_file: str = _DEFAULT_STATUS_FILE,
) -> int:
    """
    Run the initial fetch and write a status report.

    Returns:
        0 on full success, 1 if any ticker failed to load data.
    """
    if not do_prices and not do_dividends and not do_fred:
        print("Error: specify at least one of --prices, --dividends, --fred.")
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

    print(f"{prefix}PGR v2 Initial Fetch - {today} ({mode_label})")
    print(f"Database : {config.DB_PATH}")
    print(f"Dry run  : {dry_run}")
    print()

    conn = db_client.get_connection(config.DB_PATH)
    db_client.initialize_schema(conn)

    all_tickers = get_all_price_tickers()   # ["PGR"] + 20 ETFs = 21 tickers

    # -------------------------------------------------------------------------
    # Price fetch (23 AV calls)
    # -------------------------------------------------------------------------
    if do_prices:
        print(f"{prefix}Fetching prices for {len(all_tickers)} tickers...")
        try:
            loader = MultiTickerLoader(conn)
            price_results: dict[str, int] = loader.fetch_all_prices(
                all_tickers, dry_run=dry_run
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FATAL: MultiTickerLoader raised {exc}")
            had_error = True
            price_results = {t: 0 for t in all_tickers}

        for ticker, n_rows in price_results.items():
            if n_rows is None:
                status, detail = "DEFERRED", "not attempted — AV rate limit reached earlier in batch"
                rows = 0
            elif n_rows > 0:
                status, detail, rows = "OK", f"{n_rows} rows upserted", n_rows
            elif dry_run:
                status, detail, rows = "DRY-RUN", "no HTTP call made", 0
            else:
                status, detail, rows = "SKIPPED", "already fresh or no new data", 0
            results.append(TickerResult(ticker, "prices", rows, status, detail))
            marker = "OK" if status == "OK" else ("--" if status in ("SKIPPED", "DRY-RUN", "DEFERRED") else "!!")
            print(f"  [{marker}] {ticker:<6} {detail}")

        price_ok = sum(1 for r in results if r.mode == "prices" and r.status == "OK")
        print(f"\n  Prices: {price_ok}/{len(all_tickers)} tickers loaded new data")

    # -------------------------------------------------------------------------
    # Dividend fetch (23 AV calls)
    # -------------------------------------------------------------------------
    if do_dividends:
        div_tickers = get_all_dividend_tickers()
        print(f"\n{prefix}Fetching dividends for {len(div_tickers)} tickers...")
        try:
            div_loader = MultiDividendLoader(conn)
            div_results: dict[str, int] = div_loader.fetch_for_tickers(
                div_tickers, dry_run=dry_run
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FATAL: MultiDividendLoader raised {exc}")
            had_error = True
            div_results = {t: 0 for t in div_tickers}

        for ticker, n_rows in div_results.items():
            if n_rows is None:
                status, detail = "DEFERRED", "not attempted — AV rate limit reached earlier in batch"
                rows = 0
            elif n_rows > 0:
                status, detail, rows = "OK", f"{n_rows} rows upserted", n_rows
            elif dry_run:
                status, detail, rows = "DRY-RUN", "no HTTP call made", 0
            else:
                status, detail, rows = "SKIPPED", "already fresh or no new data", 0
            results.append(TickerResult(ticker, "dividends", rows, status, detail))
            marker = "OK" if status == "OK" else ("--" if status in ("SKIPPED", "DRY-RUN", "DEFERRED") else "!!")
            print(f"  [{marker}] {ticker:<6} {detail}")

        div_ok = sum(1 for r in results if r.mode == "dividends" and r.status == "OK")
        print(f"\n  Dividends: {div_ok}/{len(div_tickers)} tickers loaded new data")

    # -------------------------------------------------------------------------
    # FRED macro fetch (free API — no AV/FMP budget impact)
    # -------------------------------------------------------------------------
    if do_fred:
        _fetch_fred_step(conn, dry_run=dry_run)

    # -------------------------------------------------------------------------
    # Budget summary
    # -------------------------------------------------------------------------
    today_str = today.isoformat()
    if dry_run:
        av_projected = (len(all_tickers) if do_prices else 0) + \
                       (len(get_all_dividend_tickers()) if do_dividends else 0)
        print(f"\n{prefix}Projected AV calls: {av_projected}/{config.AV_DAILY_LIMIT}")
        if av_projected > config.AV_DAILY_LIMIT:
            print("  WARNING: projected calls exceed daily limit.")
        av_used = av_projected  # show projected, not actual
    else:
        av_used = db_client.get_api_request_count(conn, "av", today_str)
        print(f"\nAV budget used today: {av_used}/{config.AV_DAILY_LIMIT}")

    conn.close()

    # -------------------------------------------------------------------------
    # Status report
    # -------------------------------------------------------------------------
    run_end = datetime.now(timezone.utc)
    duration_s = (run_end - run_start).total_seconds()
    # DEFERRED = AV server-side rate limit hit mid-batch; not a genuine error.
    # The partial data already written is valid; remaining tickers need a re-run.
    overall_ok = not had_error and all(
        r.status in ("OK", "SKIPPED", "DRY-RUN", "DEFERRED") for r in results
    )
    n_ok       = sum(1 for r in results if r.status == "OK")
    n_skip     = sum(1 for r in results if r.status in ("SKIPPED", "DRY-RUN"))
    n_deferred = sum(1 for r in results if r.status == "DEFERRED")
    n_err      = sum(1 for r in results if r.status == "ERROR")

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

    # Console footer
    if n_deferred:
        badge = "[PARTIAL — RATE LIMITED]"
        deferred_note = f", {n_deferred} deferred (re-run tomorrow for remaining tickers)"
    else:
        badge = "[SUCCESS]" if overall_ok else "[PARTIAL FAILURE]"
        deferred_note = ""
    print(f"\n{badge}  {n_ok} loaded, {n_skip} skipped, {n_err} errors"
          f"{deferred_note} | {av_used}/{config.AV_DAILY_LIMIT} AV calls "
          f"| {duration_s:.0f}s")

    return 0 if overall_ok else 1


# ---------------------------------------------------------------------------
# Status report writers
# ---------------------------------------------------------------------------

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
        badge = "⚠️ PARTIAL — RATE LIMITED"
    else:
        badge = "✅ SUCCESS" if overall_ok else "❌ PARTIAL FAILURE"
    dry_note = " *(dry run)*" if dry_run else ""
    ts = run_dt.strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "",
        f"## {ts} — `{mode_label}`{dry_note}  {badge}",
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
            icon = "⏸️"
        elif r.status == "OK":
            icon = "✅"
        elif r.status in ("SKIPPED", "DRY-RUN"):
            icon = "⏭️"
        else:
            icon = "❌"
        lines.append(
            f"| `{r.ticker}` | {r.mode} | {r.rows_upserted:,} "
            f"| {icon} {r.status} | {r.detail} |"
        )
    lines.append("")

    # Prepend header if file is new, else append
    if not Path(status_file).exists():
        header = [
            "# PGR v2 Initial Fetch — Status Log",
            "",
            "Appended each time `initial_fetch.py` runs (via GitHub Actions or locally).",
            "Most-recent run appears **last**.",
            "",
        ]
        Path(status_file).write_text("\n".join(header + lines), encoding="utf-8")
    else:
        with open(status_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"Status report written -> {status_file}")


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
    """
    Write a rich summary to the GitHub Actions step summary file
    (``$GITHUB_STEP_SUMMARY``).  No-ops outside of GitHub Actions.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    if n_deferred:
        badge = "⚠️ PARTIAL — RATE LIMITED"
    else:
        badge = "✅ SUCCESS" if overall_ok else "❌ PARTIAL FAILURE"
    dry_note = " *(dry run)*" if dry_run else ""
    deferred_row = (
        [f"| Tickers deferred (re-run tomorrow) | ⚠️ **{n_deferred}** |"]
        if n_deferred else []
    )
    lines = [
        f"# PGR Initial Fetch — `{mode_label}`{dry_note}",
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
            icon = "⏸️"
        elif r.status == "OK":
            icon = "✅"
        elif r.status in ("SKIPPED", "DRY-RUN"):
            icon = "⏭️"
        else:
            icon = "❌"
        lines.append(f"| `{r.ticker}` | {r.mode} | {r.rows_upserted:,} | {icon} {r.status} |")
    lines += ["", "</details>", ""]

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PGR v2 one-time full data population (run --prices and "
                    "--dividends on separate days to stay within AV limits)."
    )
    parser.add_argument("--prices", action="store_true",
                        help="Fetch full price history for all tickers.")
    parser.add_argument("--dividends", action="store_true",
                        help="Fetch full dividend history for all tickers.")
    parser.add_argument("--fred", action="store_true",
                        help="Fetch full FRED macro history (free API, no AV budget impact).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions without making HTTP calls.")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if data was already fetched today.")
    parser.add_argument("--status-file", default=_DEFAULT_STATUS_FILE,
                        help=f"Path for Markdown status report (default: {_DEFAULT_STATUS_FILE}).")
    args = parser.parse_args()
    sys.exit(main(
        do_prices=args.prices,
        do_dividends=args.dividends,
        do_fred=args.fred,
        dry_run=args.dry_run,
        force=args.force,
        status_file=args.status_file,
    ))
