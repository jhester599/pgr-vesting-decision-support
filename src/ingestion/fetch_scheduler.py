"""
Fetch scheduler for the v2 weekly data accumulation cron.

The weekly GitHub Actions run fetches the full price history for all 23 tickers
(PGR + 22 ETF benchmarks) in a single run.  TIME_SERIES_WEEKLY returns the
complete 20-25 year history in one call, so 23 tickers requires only 23 of the
25 daily AV calls — well under the free-tier limit.

Budget allocation per weekly run:
  23 calls — all ticker prices (PGR + 22 ETFs)  [Friday 22:00 UTC]
   1 call  — PGR dividends                       [Friday 22:00 UTC]
  ─────────
  24 AV calls  (limit: 25/day)

Peer tickers (ALL, TRV, CB, HIG — v6.0 cross-asset signals) are fetched by
scripts/peer_fetch.py on a separate Sunday 04:00 UTC cron (30 hours after the
Friday run) to avoid competing for the same daily call budget.

  4 calls — peer prices   [Sunday 04:00 UTC]
  4 calls — peer dividends [Sunday 04:00 UTC]
  ─────────
  8 AV calls  (limit: 25/day; 17 calls of margin)

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


def get_peer_price_tickers() -> list[str]:
    """Return the four insurance-peer tickers for the Sunday peer price fetch.

    These tickers (ALL, TRV, CB, HIG) are fetched separately from the main
    ETF universe by ``scripts/peer_fetch.py`` on a Sunday 04:00 UTC cron
    (30 hours after the Friday 22:00 UTC main fetch) to avoid competing for
    the same 25 calls/day Alpha Vantage budget.

    Used for v6.0 cross-asset features:
      - ``pgr_vs_peers_6m``: PGR vs equal-weight peer composite relative return
      - Residual momentum baseline (Fama-French factor-neutral peer betas)

    Returns:
        List of 4 ticker symbols from ``config.PEER_TICKER_UNIVERSE``.
    """
    return list(config.PEER_TICKER_UNIVERSE)


def get_peer_dividend_tickers() -> list[str]:
    """Return the four insurance-peer tickers for peer dividend fetches.

    Dividend history is required to compute DRIP total returns for the peer
    composite, consistent with how the ETF benchmarks are handled throughout
    the rest of the pipeline.

    Returns:
        List of 4 ticker symbols from ``config.PEER_TICKER_UNIVERSE``.
    """
    return list(config.PEER_TICKER_UNIVERSE)
