"""Redeploy guidance and the Black-Litterman diagnostic for the monthly decision run."""

from __future__ import annotations

import logging
from datetime import date
from typing import cast

import pandas as pd

import config
from src.database import db_client
from src.portfolio.black_litterman import BLDiagnostics, build_bl_weights
from src.portfolio.diversification import score_benchmarks_against_pgr
from src.portfolio.redeploy_buckets import (
    add_destination_roles,
    recommend_redeploy_buckets,
)
from src.portfolio.redeploy_portfolio import (
    recommend_redeploy_portfolio,
    v27_investable_redeploy_universe,
)

logger = logging.getLogger(__name__)


def build_redeploy_guidance(conn) -> list[dict[str, object]]:
    """Build diversification-first redeploy buckets for production messaging."""
    investable_universe = v27_investable_redeploy_universe()
    scoreboard = score_benchmarks_against_pgr(conn, investable_universe)
    if scoreboard.empty:
        return []
    scoreboard["composite_score"] = 0.0
    scoreboard = add_destination_roles(scoreboard)
    return recommend_redeploy_buckets(scoreboard, investable_universe)


def load_etf_monthly_returns(
    conn,
    tickers: list[str],
    end_date: date,
) -> pd.DataFrame:
    """Load daily ETF prices, resample to month-end close, return pct_change matrix.

    Used by the Black-Litterman diagnostic call to build the covariance matrix.
    Tickers with no price data are silently excluded.

    Returns:
        DataFrame with tickers as columns and monthly returns as rows
        (DatetimeIndex).  May be empty if no data is available.
    """
    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            df = db_client.get_prices(conn, ticker, end_date=end_date.isoformat())
        except Exception:  # noqa: BLE001
            logger.warning("Could not load prices for %s; excluding from BL covariance", ticker, exc_info=True)
            continue
        if df.empty:
            continue
        monthly = df["close"].resample("ME").last().dropna()
        frames[ticker] = monthly.pct_change().dropna()
    if not frames:
        return pd.DataFrame()
    merged = pd.DataFrame(frames)
    return merged.dropna(how="all")


def build_redeploy_portfolio(
    conn,
    signals: pd.DataFrame,
    recommendation_mode: dict[str, str | float] | None,
) -> dict[str, object] | None:
    """Build the concrete monthly redeploy portfolio recommendation."""
    if signals.empty:
        return None
    investable_universe = v27_investable_redeploy_universe()
    scoreboard = score_benchmarks_against_pgr(conn, investable_universe)
    if scoreboard.empty:
        return None
    mode_label = (
        str(recommendation_mode.get("label", "DEFER-TO-TAX-DEFAULT"))
        if recommendation_mode is not None
        else "DEFER-TO-TAX-DEFAULT"
    )
    return recommend_redeploy_portfolio(
        signals=signals,
        diversification_scoreboard=scoreboard,
        recommendation_mode_label=mode_label,
    )


def build_bl_diagnostics(
    conn,
    ensemble_results: dict,
    as_of: date,
) -> BLDiagnostics | None:
    """Black-Litterman diagnostic shadow run (v34.0, Tier 1.4), or ``None``.

    Read-only: the result is shown in the Portfolio Optimizer Status section
    of ``recommendation.md`` only. It does NOT alter the primary
    recommendation or the redeploy portfolio weights.
    """
    bl_diagnostics: BLDiagnostics | None = None
    try:
        etf_returns_df = load_etf_monthly_returns(conn, config.ETF_BENCHMARK_UNIVERSE, as_of)
        if not etf_returns_df.empty and len(etf_returns_df) >= 12:
            # return_diagnostics=True always returns (weights, diagnostics).
            _, bl_diagnostics = cast(
                tuple[dict[str, float], BLDiagnostics],
                build_bl_weights(
                    ensemble_results,
                    etf_returns_df,
                    return_diagnostics=True,
                ),
            )
        else:
            logger.info(
                "BL diagnostic skipped: ETF return matrix has only %d rows (need ≥12).",
                len(etf_returns_df),
            )
    except Exception:
        logger.warning("BL diagnostic shadow run failed; Portfolio Optimizer section will show 'not run'", exc_info=True)
    return bl_diagnostics
