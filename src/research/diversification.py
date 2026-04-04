"""Diversification-aware scoring helpers for v11 research."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sqlite3

import numpy as np
import pandas as pd

from src.processing.multi_total_return import build_etf_monthly_returns


HIGH_CORRELATION_THRESHOLD = 0.75
MODERATE_CORRELATION_THRESHOLD = 0.45


@dataclass(frozen=True)
class DiversificationSnapshot:
    """Point-in-time diversification metrics for one alternative vs PGR."""

    ticker: str
    n_obs: int
    corr_to_pgr: float
    beta_to_pgr: float
    downside_corr_to_pgr: float
    diversification_score: float
    corr_bucket: str


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 3:
        return float("nan")
    if aligned.iloc[:, 0].nunique(dropna=False) <= 1:
        return 0.0
    if aligned.iloc[:, 1].nunique(dropna=False) <= 1:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return float(corr) if pd.notna(corr) else 0.0


def _safe_beta(x: pd.Series, y: pd.Series) -> float:
    """Return beta of x vs y using aligned monthly forward returns."""
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 3:
        return float("nan")
    var_y = float(aligned.iloc[:, 1].var(ddof=1))
    if not np.isfinite(var_y) or math.isclose(var_y, 0.0):
        return 0.0
    cov_xy = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]))
    return cov_xy / var_y


def classify_correlation_bucket(corr_to_pgr: float) -> str:
    """Bucket an alternative by how much it recreates PGR exposure."""
    if not np.isfinite(corr_to_pgr):
        return "unknown"
    if corr_to_pgr >= HIGH_CORRELATION_THRESHOLD:
        return "highly_correlated"
    if corr_to_pgr >= MODERATE_CORRELATION_THRESHOLD:
        return "moderately_correlated"
    return "diversifying"


def diversification_score(
    corr_to_pgr: float,
    downside_corr_to_pgr: float,
) -> float:
    """Return a bounded diversification score in [0, 1].

    The score intentionally rewards low and negative correlation to PGR, with a
    slightly stronger penalty when downside co-movement remains elevated.
    """
    corr_component = 0.5 * (1.0 - max(min(corr_to_pgr, 1.0), -1.0))
    downside_component = 0.5 * (1.0 - max(min(downside_corr_to_pgr, 1.0), -1.0))
    score = 0.7 * corr_component + 0.3 * downside_component
    return float(max(0.0, min(1.0, score)))


def build_monthly_return_matrix(
    conn: sqlite3.Connection,
    tickers: list[str],
) -> pd.DataFrame:
    """Load aligned forward 1M total returns for the supplied tickers."""
    series_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        returns = build_etf_monthly_returns(conn, ticker, forward_months=1)
        if returns.empty:
            continue
        series_map[ticker] = returns.rename(ticker)
    if not series_map:
        return pd.DataFrame()
    return pd.DataFrame(series_map)


def score_benchmarks_against_pgr(
    conn: sqlite3.Connection,
    tickers: list[str],
) -> pd.DataFrame:
    """Build diversification metrics for each benchmark relative to PGR."""
    matrix = build_monthly_return_matrix(conn, ["PGR", *tickers])
    if matrix.empty or "PGR" not in matrix.columns:
        return pd.DataFrame(
            columns=[
                "benchmark",
                "n_obs",
                "corr_to_pgr",
                "beta_to_pgr",
                "downside_corr_to_pgr",
                "diversification_score",
                "corr_bucket",
            ]
        )

    pgr = matrix["PGR"]
    rows: list[dict[str, float | int | str]] = []
    for ticker in tickers:
        if ticker not in matrix.columns:
            continue
        alt = matrix[ticker]
        aligned = pd.concat([alt, pgr], axis=1).dropna()
        if aligned.empty:
            continue
        corr = _safe_corr(alt, pgr)
        beta = _safe_beta(alt, pgr)
        downside_mask = (aligned.iloc[:, 1] < 0.0) | (aligned.iloc[:, 0] < 0.0)
        downside = aligned.loc[downside_mask]
        if downside.empty:
            downside_corr = corr
        else:
            downside_corr = _safe_corr(downside.iloc[:, 0], downside.iloc[:, 1])
        score = diversification_score(corr, downside_corr)
        rows.append(
            {
                "benchmark": ticker,
                "n_obs": int(len(aligned)),
                "corr_to_pgr": corr,
                "beta_to_pgr": beta,
                "downside_corr_to_pgr": downside_corr,
                "diversification_score": score,
                "corr_bucket": classify_correlation_bucket(corr),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["diversification_score", "corr_to_pgr"],
        ascending=[False, True],
    )
