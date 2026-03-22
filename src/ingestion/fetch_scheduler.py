"""
Fetch scheduler for the v2 weekly data accumulation cron.

The weekly GitHub Actions run fetches the full price history for all 23 tickers
(PGR + 22 ETF benchmarks) in a single run.  TIME_SERIES_WEEKLY returns the
complete 20-25 year history in one call, so 23 tickers requires only 23 of the
25 daily AV calls — well under the free-tier limit.

Budget allocation per weekly run:
  23 calls — all ticker prices (PGR + 22 ETFs)
   1 call  — PGR dividends
  ─────────
  24 AV calls  (limit: 25/day)
   2 FMP calls — PGR key-metrics + income-statement (always, each weekly run)
  ─────────
   2 FMP calls (limit: 250/day)

ETF dividends are fetched by scripts/initial_fetch.py (one-time or quarterly
manual refresh) rather than on every weekly run, to stay within the 25 AV
call/day limit.
"""

from __future__ import annotations

import config


def get_all_price_tickers() -> list[str]:
    """Return all 23 tickers for the weekly price fetch.

    PGR is always first, followed by the 22 ETF benchmarks in their
    canonical order from config.ETF_BENCHMARK_UNIVERSE.

    Returns:
        List of 23 ticker symbols: [``"PGR"``] + all 22 ETF benchmarks.
    """
    return ["PGR"] + list(config.ETF_BENCHMARK_UNIVERSE)


def get_all_dividend_tickers() -> list[str]:
    """Return all 23 tickers for a full dividend fetch.

    Used by scripts/initial_fetch.py for the one-time bootstrap and
    for quarterly ETF dividend refreshes.  The weekly cron only fetches
    PGR dividends; this function is not called from weekly_fetch.py.

    Returns:
        List of 23 ticker symbols: [``"PGR"``] + all 22 ETF benchmarks.
    """
    return ["PGR"] + list(config.ETF_BENCHMARK_UNIVERSE)
