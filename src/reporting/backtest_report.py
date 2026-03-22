"""
Backtest reporting utilities for the v2 PGR vesting decision support engine.

Functions:
  - ``generate_backtest_table``: pivot table of realized relative returns by
    event date × ETF benchmark.
  - ``print_backtest_summary``: console summary of overall and per-benchmark
    hit rates, IC, and top/bottom performers.
  - ``export_backtest_to_csv``: full-detail dump for offline analysis.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.backtest.backtest_engine import BacktestEventResult


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_backtest_table(
    results: list["BacktestEventResult"],
    horizon: int,
) -> pd.DataFrame:
    """
    Pivot the backtest results into a readable event × benchmark table.

    Args:
        results: List of BacktestEventResult from run_historical_backtest().
        horizon: Target horizon in months (6 or 12) — filters the result list.

    Returns:
        DataFrame with index = event_date (date), columns = ETF benchmark
        tickers, values = realized_relative_return.  A companion attribute
        ``predictions`` holds the same pivot but for predicted_relative_return,
        and ``correct_direction`` holds a boolean pivot.

        Returns an empty DataFrame if no results match the horizon.
    """
    filtered = [r for r in results if r.target_horizon == horizon]
    if not filtered:
        return pd.DataFrame()

    rows = []
    for r in filtered:
        rows.append({
            "event_date":    r.event.event_date,
            "rsu_type":      r.event.rsu_type,
            "benchmark":     r.benchmark,
            "realized":      r.realized_relative_return,
            "predicted":     r.predicted_relative_return,
            "correct":       r.correct_direction,
        })

    df = pd.DataFrame(rows)

    realized_pivot = df.pivot_table(
        index="event_date",
        columns="benchmark",
        values="realized",
        aggfunc="first",
    )
    realized_pivot.columns.name = None
    realized_pivot.index.name = "event_date"

    return realized_pivot


def generate_prediction_table(
    results: list["BacktestEventResult"],
    horizon: int,
) -> pd.DataFrame:
    """
    Pivot of predicted_relative_return by event_date × benchmark.

    Same shape as ``generate_backtest_table``; values are model predictions.
    """
    filtered = [r for r in results if r.target_horizon == horizon]
    if not filtered:
        return pd.DataFrame()

    rows = [
        {
            "event_date": r.event.event_date,
            "benchmark":  r.benchmark,
            "predicted":  r.predicted_relative_return,
        }
        for r in filtered
    ]
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="event_date", columns="benchmark", values="predicted", aggfunc="first"
    )
    pivot.columns.name = None
    pivot.index.name = "event_date"
    return pivot


def generate_correct_direction_table(
    results: list["BacktestEventResult"],
    horizon: int,
) -> pd.DataFrame:
    """
    Pivot of correct_direction (bool) by event_date × benchmark.
    """
    filtered = [r for r in results if r.target_horizon == horizon]
    if not filtered:
        return pd.DataFrame()

    rows = [
        {
            "event_date": r.event.event_date,
            "benchmark":  r.benchmark,
            "correct":    r.correct_direction,
        }
        for r in filtered
    ]
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="event_date", columns="benchmark", values="correct", aggfunc="first"
    )
    pivot.columns.name = None
    pivot.index.name = "event_date"
    return pivot


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_backtest_summary(
    results: list["BacktestEventResult"],
    top_n: int = 5,
) -> None:
    """
    Print a formatted backtest summary to stdout.

    Covers:
      - Overall hit rate and mean IC across all results.
      - Hit rate broken down by target horizon (6M / 12M).
      - Hit rate by RSU type (time / performance).
      - Top N and bottom N benchmarks by hit rate.
      - Average predicted vs. realized return.
    """
    if not results:
        print("No backtest results to summarize.")
        return

    sep = "=" * 70
    print(sep)
    print("  PGR VESTING BACKTEST SUMMARY")
    print(sep)

    overall_hit = np.mean([r.correct_direction for r in results])
    overall_ic = np.mean([r.ic_at_event for r in results])
    print(f"  Total results      : {len(results)}")
    print(f"  Overall hit rate   : {overall_hit:.1%}")
    print(f"  Mean IC at event   : {overall_ic:.4f}")

    print()
    print("  By target horizon:")
    for horizon in sorted({r.target_horizon for r in results}):
        subset = [r for r in results if r.target_horizon == horizon]
        hr = np.mean([r.correct_direction for r in subset])
        print(f"    {horizon:2d}M : {hr:.1%} ({len(subset)} results)")

    print()
    print("  By RSU type:")
    for rsu_type in sorted({r.event.rsu_type for r in results}):
        subset = [r for r in results if r.event.rsu_type == rsu_type]
        hr = np.mean([r.correct_direction for r in subset])
        print(f"    {rsu_type:12s} : {hr:.1%} ({len(subset)} results)")

    print()
    print(f"  Top {top_n} benchmarks by hit rate:")
    by_bench: dict[str, list[bool]] = {}
    for r in results:
        by_bench.setdefault(r.benchmark, []).append(r.correct_direction)
    bench_hr = {b: np.mean(v) for b, v in by_bench.items()}
    sorted_bench = sorted(bench_hr.items(), key=lambda x: x[1], reverse=True)
    for bench, hr in sorted_bench[:top_n]:
        n = len(by_bench[bench])
        print(f"    {bench:<8} {hr:.1%}  (n={n})")

    print()
    print(f"  Bottom {top_n} benchmarks by hit rate:")
    for bench, hr in sorted_bench[-top_n:]:
        n = len(by_bench[bench])
        print(f"    {bench:<8} {hr:.1%}  (n={n})")

    realized_valid = [r.realized_relative_return for r in results
                      if not np.isnan(r.realized_relative_return)]
    if realized_valid:
        print()
        print(f"  Mean realized relative return : {np.mean(realized_valid):+.2%}")
        print(f"  Median realized rel. return   : {np.median(realized_valid):+.2%}")

    print(sep)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_backtest_to_csv(
    results: list["BacktestEventResult"],
    path: str,
) -> None:
    """
    Export full backtest results to a CSV file for offline analysis.

    Args:
        results: List of BacktestEventResult.
        path:    Destination file path (parent directories created if needed).
    """
    if not results:
        return

    rows = []
    for r in results:
        rows.append({
            "event_date":                 r.event.event_date,
            "rsu_type":                   r.event.rsu_type,
            "year":                       r.event.event_date.year,
            "benchmark":                  r.benchmark,
            "target_horizon":             r.target_horizon,
            "predicted_relative_return":  r.predicted_relative_return,
            "realized_relative_return":   r.realized_relative_return,
            "signal_direction":           r.signal_direction,
            "correct_direction":          r.correct_direction,
            "predicted_sell_pct":         r.predicted_sell_pct,
            "ic_at_event":                r.ic_at_event,
            "hit_rate_at_event":          r.hit_rate_at_event,
            "n_train_observations":       r.n_train_observations,
            "proxy_fill_fraction":        r.proxy_fill_fraction,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    df.to_csv(path, index=False)
