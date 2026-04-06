"""
Historical vesting event backtest engine.

For each vesting event × benchmark × target_horizon combination:
  1. Slices the feature matrix to rows ≤ event_date (strict no-lookahead).
  2. Loads the pre-computed relative return series from the DB, sliced to
     rows ≤ event_date - embargo (prevents training on overlapping returns).
  3. Calls ``predict_current()`` using a freshly refitted model to generate a
     signal as of event_date.
  4. Loads the realized relative return for the forward window from the DB.
  5. Records whether the directional prediction was correct.

Temporal integrity:
  - Feature matrix slice: ``df.loc[: event_date]``
  - Relative return target slice: ``y.loc[: event_date - timedelta(embargo)]``
  - Realized outcome: from ``monthly_relative_returns`` table (pre-computed,
    no leakage by construction since forward returns require future prices)

Output: one BacktestEventResult per (event × benchmark × horizon) tuple.
``run_full_backtest()`` returns a flat DataFrame of all results.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
import logging
from typing import Literal, cast

import numpy as np
import pandas as pd

import config
from src.backtest.vesting_events import (
    VestingEvent,
    enumerate_monthly_evaluation_dates,
    enumerate_vesting_events,
)
from src.database import db_client
from src.models.wfo_engine import run_wfo, predict_current
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_X_y_relative,
)
from src.processing.multi_total_return import load_relative_return_matrix

logger = logging.getLogger(__name__)


@dataclass
class BacktestEventResult:
    """Result for one (vesting event × benchmark × target_horizon) cell."""
    event: VestingEvent
    benchmark: str
    target_horizon: int
    predicted_relative_return: float
    realized_relative_return: float
    signal_direction: Literal["OUTPERFORM", "UNDERPERFORM"]
    correct_direction: bool
    predicted_sell_pct: float
    ic_at_event: float
    hit_rate_at_event: float
    n_train_observations: int
    proxy_fill_fraction: float   # fraction of training obs using proxy-filled prices


def _signal_from_prediction(predicted: float) -> Literal["OUTPERFORM", "UNDERPERFORM"]:
    return "OUTPERFORM" if predicted >= 0 else "UNDERPERFORM"


def _realized_direction(realized: float) -> Literal["OUTPERFORM", "UNDERPERFORM"]:
    return "OUTPERFORM" if realized >= 0 else "UNDERPERFORM"


def _sell_pct_from_signal(
    predicted: float,
    ic: float,
    hit_rate: float,
) -> float:
    """
    Simple sell-percentage rule mirroring rebalancer._compute_sell_pct().

    Returns 0.0–1.0.
    """
    if ic < 0.05:
        return 0.50
    if predicted > 0.15 and ic > 0.10:
        return 0.25
    if predicted > 0.05:
        return 0.50
    return 1.00


def run_historical_backtest(
    conn: sqlite3.Connection,
    model_type: Literal["lasso", "ridge", "elasticnet"] = "elasticnet",
    target_horizon_months: int = 6,
    start_year: int = 2014,
    end_year: int | None = None,
    events_override: list[VestingEvent] | None = None,
) -> list[BacktestEventResult]:
    """
    Run walk-forward backtests for all vesting events and all ETF benchmarks.

    For each vesting event, the feature matrix and relative return series are
    sliced strictly to data available on or before the event date, so the
    model never sees future information during training.

    Args:
        conn:                  Open SQLite connection with v2 schema and
                               populated price / relative return tables.
        model_type:            ``"elasticnet"`` (default v3.0+), ``"lasso"``,
                               or ``"ridge"``.
        target_horizon_months: 6 or 12.
        start_year:            First vesting year to include (default 2014).
        end_year:              Last vesting year to include.  Defaults to
                               ``current_year - 2`` (ensures 12M returns
                               are fully realized for all events).
        events_override:       If provided, evaluate these events instead of
                               the default vesting event calendar.  Used by
                               ``run_monthly_stability_backtest()``.

    Returns:
        List of BacktestEventResult, one per (event × benchmark) pair where
        sufficient data exists to run the WFO model.
    """
    # Build the full feature matrix once (uses DB, not stale Parquet cache)
    df_full = build_feature_matrix_from_db(conn, force_refresh=True)

    # Load feature column names (excludes target)
    feature_cols = [c for c in df_full.columns if c != "target_6m_return"]
    X_full = df_full[feature_cols]

    if events_override is not None:
        vesting_events = events_override
    else:
        vesting_events = enumerate_vesting_events(
            start_year=start_year, end_year=end_year
        )

    results: list[BacktestEventResult] = []
    embargo = target_horizon_months   # months to lag the target slice

    for event in vesting_events:
        event_ts = pd.Timestamp(event.event_date)

        # Slice feature matrix to rows ≤ event_date
        X_event = X_full.loc[X_full.index <= event_ts]
        if X_event.empty:
            continue

        # Current observation = last row of the sliced matrix
        X_current = X_event.iloc[[-1]]

        for etf in config.ETF_BENCHMARK_UNIVERSE:
            # Load pre-computed relative return series for this ETF/horizon
            rel_full = load_relative_return_matrix(
                conn, etf, target_horizon_months
            )
            if rel_full.empty:
                continue

            # Embargo slice: exclude rows within `embargo` months of event_date
            # to prevent training on returns that overlap the prediction window
            embargo_cutoff = event_ts - pd.DateOffset(months=embargo)
            rel_train = rel_full.loc[rel_full.index <= embargo_cutoff]
            if rel_train.empty:
                continue

            # Align features to this (sliced) target series
            X_train_aligned = X_event.loc[X_event.index <= embargo_cutoff]
            if X_train_aligned.empty:
                continue

            try:
                _, y_aligned = get_X_y_relative(
                    X_train_aligned.assign(**{"__tmp_target": rel_train})
                    .drop(columns=["__tmp_target"]),
                    rel_train,
                    drop_na_target=True,
                )
                # Rebuild X aligned to y_aligned's index
                X_aligned = X_train_aligned.loc[y_aligned.index]
            except ValueError:
                continue

            if len(y_aligned) < (config.WFO_TRAIN_WINDOW_MONTHS
                                  + config.WFO_TEST_WINDOW_MONTHS):
                continue

            # Run WFO on the event-date-sliced data
            try:
                wfo_result = run_wfo(
                    X_aligned,
                    y_aligned,
                    model_type=model_type,
                    target_horizon_months=target_horizon_months,
                    benchmark=etf,
                )
            except ValueError:
                continue

            # Generate live prediction using a refit on the most recent window
            try:
                pred = predict_current(
                    X_full=X_aligned,
                    y_full=y_aligned,
                    X_current=X_current,
                    wfo_result=wfo_result,
                    model_type=model_type,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Skipping backtest prediction for benchmark %s on event %s due to live prediction failure. Error=%r",
                    etf,
                    event.event_date.isoformat(),
                    exc,
                )
                continue

            predicted = pred["predicted_return"]
            ic = pred["ic"]
            hit_rate = pred["hit_rate"]

            # Retrieve realized outcome from the pre-computed DB table
            # The realized return is the row nearest to event_date in the
            # monthly_relative_returns table for this benchmark/horizon.
            realized_series = load_relative_return_matrix(
                conn, etf, target_horizon_months,
                start_date=event.event_date.strftime("%Y-%m-%d"),
                end_date=event.horizon_6m_end.strftime("%Y-%m-%d")
                if target_horizon_months == 6
                else event.horizon_12m_end.strftime("%Y-%m-%d"),
            )

            if realized_series.empty:
                realized = np.nan
            else:
                realized = float(realized_series.iloc[0])

            # Proxy fill fraction: fraction of training obs with proxy prices
            # We approximate this from the DB; if the column is absent, use 0
            try:
                proxy_frac = _compute_proxy_fill_fraction(
                    conn, etf, X_aligned.index
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Could not compute proxy fill fraction for benchmark %s on event %s; defaulting to 0.0. Error=%r",
                    etf,
                    event.event_date.isoformat(),
                    exc,
                )
                proxy_frac = 0.0

            signal = _signal_from_prediction(predicted)
            realized_dir = _realized_direction(realized) if not np.isnan(realized) else None
            correct = (signal == realized_dir) if realized_dir is not None else False

            results.append(
                BacktestEventResult(
                    event=event,
                    benchmark=etf,
                    target_horizon=target_horizon_months,
                    predicted_relative_return=predicted,
                    realized_relative_return=realized,
                    signal_direction=signal,
                    correct_direction=correct,
                    predicted_sell_pct=_sell_pct_from_signal(predicted, ic, hit_rate),
                    ic_at_event=ic,
                    hit_rate_at_event=hit_rate,
                    n_train_observations=len(y_aligned),
                    proxy_fill_fraction=proxy_frac,
                )
            )

    return results


def run_monthly_stability_backtest(
    conn: sqlite3.Connection,
    model_type: Literal["lasso", "ridge", "elasticnet"] = "elasticnet",
    target_horizon_months: int = 6,
    start_year: int = 2014,
    end_year: int | None = None,
) -> list[BacktestEventResult]:
    """
    Run walk-forward backtests across all month-end evaluation dates (v3.0+).

    This expands evaluation from ~20 semi-annual vesting events to 120+
    monthly data points, enabling statistically meaningful assessment of model
    predictive skill.  The same temporal slicing logic as
    ``run_historical_backtest()`` is used — no lookahead leakage.

    Note on serial autocorrelation: consecutive monthly predictions with a
    6-month forward target share 5 of their 6 return months.  Use Newey-West
    standard errors (lag=5 for 6M, lag=11 for 12M) when computing significance
    from the full monthly series.

    Args:
        conn:                  Open SQLite connection.
        model_type:            ``"elasticnet"`` (default), ``"lasso"``, or
                               ``"ridge"``.
        target_horizon_months: 6 or 12.
        start_year:            First year to include (default 2014).
        end_year:              Last year to include.  Defaults to
                               ``current_year - 2``.

    Returns:
        List of BacktestEventResult for every month-end evaluation date where
        sufficient data exists, covering all ETF benchmarks.
    """
    monthly_dates = enumerate_monthly_evaluation_dates(
        start_year=start_year, end_year=end_year
    )
    return run_historical_backtest(
        conn=conn,
        model_type=model_type,
        target_horizon_months=target_horizon_months,
        start_year=start_year,
        end_year=end_year,
        events_override=monthly_dates,
    )


def run_full_backtest(
    conn: sqlite3.Connection,
    model_type: str = "elasticnet",
) -> pd.DataFrame:
    """
    Run the complete backtest for all vesting events, all benchmarks, and
    both target horizons (6M and 12M).

    Returns:
        DataFrame with one row per (event × benchmark × horizon) combination.
        Columns mirror BacktestEventResult fields plus derived columns:
          - ``event_date``, ``rsu_type``, ``year``
          - ``benchmark``, ``target_horizon``
          - ``predicted_relative_return``, ``realized_relative_return``
          - ``signal_direction``, ``correct_direction``
          - ``predicted_sell_pct``
          - ``ic_at_event``, ``hit_rate_at_event``
          - ``n_train_observations``, ``proxy_fill_fraction``
    """
    all_results = []
    for horizon in config.WFO_TARGET_HORIZONS:
        results = run_historical_backtest(
            conn,
            model_type=cast(Literal["lasso", "ridge", "elasticnet"], model_type),
            target_horizon_months=horizon,
        )
        all_results.extend(results)

    if not all_results:
        return pd.DataFrame()

    rows = []
    for r in all_results:
        rows.append({
            "event_date":                   r.event.event_date,
            "rsu_type":                     r.event.rsu_type,
            "year":                         r.event.event_date.year,
            "benchmark":                    r.benchmark,
            "target_horizon":               r.target_horizon,
            "predicted_relative_return":    r.predicted_relative_return,
            "realized_relative_return":     r.realized_relative_return,
            "signal_direction":             r.signal_direction,
            "correct_direction":            r.correct_direction,
            "predicted_sell_pct":           r.predicted_sell_pct,
            "ic_at_event":                  r.ic_at_event,
            "hit_rate_at_event":            r.hit_rate_at_event,
            "n_train_observations":         r.n_train_observations,
            "proxy_fill_fraction":          r.proxy_fill_fraction,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_proxy_fill_fraction(
    conn: sqlite3.Connection,
    ticker: str,
    date_index: pd.DatetimeIndex,
) -> float:
    """
    Estimate the fraction of training observations whose prices were
    proxy-filled (``proxy_fill=1`` in the daily_prices table).

    Approximates by comparing the count of proxy-fill rows in the ticker's
    price table that fall within the date_index range.
    """
    if date_index.empty:
        return 0.0

    start_str = date_index.min().strftime("%Y-%m-%d")
    end_str = date_index.max().strftime("%Y-%m-%d")

    cur = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN proxy_fill = 1 THEN 1 ELSE 0 END) AS proxy_count
        FROM daily_prices
        WHERE ticker = ?
          AND date BETWEEN ? AND ?
        """,
        (ticker, start_str, end_str),
    )
    row = cur.fetchone()
    if row is None or row[0] == 0:
        return 0.0
    return float(row[1]) / float(row[0])
