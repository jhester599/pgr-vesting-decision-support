"""
Backtest reporting utilities for the v2/v3 PGR vesting decision support engine.

Functions:
  - ``generate_backtest_table``: pivot table of realized relative returns by
    event date × ETF benchmark.
  - ``print_backtest_summary``: console summary of overall and per-benchmark
    hit rates, IC, and top/bottom performers.
  - ``export_backtest_to_csv``: full-detail dump for offline analysis.

v3.0 additions:
  - ``compute_oos_r_squared``: Campbell-Thompson OOS R² (model vs. naive mean).
  - ``apply_bhy_correction``: Benjamini-Hochberg-Yekutieli multiple testing
    correction across benchmarks (requires statsmodels).
  - ``compute_newey_west_ic``: HAC-corrected information coefficient and
    p-value (requires statsmodels).
  - ``generate_rolling_ic_series``: 24-month rolling IC / hit-rate time series.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.backtest.backtest_engine import BacktestEventResult

# ---------------------------------------------------------------------------
# v3.0 statistical metrics
# ---------------------------------------------------------------------------

def compute_oos_r_squared(
    predicted: pd.Series,
    realized: pd.Series,
) -> float:
    """
    Compute the Campbell-Thompson (2008) out-of-sample R².

    OOS R² = 1 - MSE_model / MSE_naive

    where MSE_naive uses the expanding historical mean return as the forecast.
    A positive OOS R² means the model beats the naive historical-average
    benchmark.  Values of 0.5–2.0% are economically significant for 6–12M
    return forecasting.

    Args:
        predicted: Series of model predictions (aligned index with ``realized``).
        realized:  Series of realized returns.

    Returns:
        OOS R² as a float.  Returns NaN if inputs have fewer than 2 observations.
    """
    aligned = pd.concat([predicted, realized], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")

    y_hat = aligned.iloc[:, 0].values
    y_true = aligned.iloc[:, 1].values

    # Expanding historical mean as the naive benchmark
    y_naive = np.array([np.mean(y_true[:i]) for i in range(1, len(y_true) + 1)])
    y_naive[0] = y_true[0]  # no prior history: use first realized value as naive

    mse_model = np.mean((y_true - y_hat) ** 2)
    mse_naive = np.mean((y_true - y_naive) ** 2)

    if mse_naive == 0.0:
        return float("nan")

    return float(1.0 - mse_model / mse_naive)


def apply_bhy_correction(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, bool]:
    """
    Apply Benjamini-Hochberg-Yekutieli (BHY) false discovery rate correction.

    Controls the false discovery rate at level ``alpha`` when testing multiple
    benchmarks simultaneously.  More powerful than Bonferroni while still
    providing FDR control.

    Requires ``statsmodels``.

    Args:
        p_values: Dict mapping benchmark ticker to raw p-value.
        alpha:    FDR level (default 0.05).

    Returns:
        Dict mapping benchmark ticker to bool: True if the null hypothesis is
        rejected after BHY correction (i.e., the IC is statistically significant).
    """
    from statsmodels.stats.multitest import multipletests

    if not p_values:
        return {}

    tickers = list(p_values.keys())
    raw_pvals = [p_values[t] for t in tickers]

    reject, _, _, _ = multipletests(raw_pvals, alpha=alpha, method="fdr_by")
    return dict(zip(tickers, reject.tolist()))


def compute_newey_west_ic(
    predicted: pd.Series,
    realized: pd.Series,
    lags: int,
) -> tuple[float, float]:
    """
    Compute the Spearman IC with Newey-West HAC standard errors.

    Newey-West correction accounts for serial autocorrelation in the IC series,
    which arises because consecutive monthly predictions with a 6M target share
    5 months of overlapping return windows.  Use ``lags = target_horizon - 1``
    (5 for 6M targets, 11 for 12M).

    Requires ``statsmodels``.

    Args:
        predicted: Series of model predictions.
        realized:  Series of realized returns.
        lags:      Number of autocorrelation lags for Newey-West adjustment.
                   Typically ``target_horizon_months - 1``.

    Returns:
        ``(ic, p_value)`` tuple where ``ic`` is the Spearman rank correlation
        and ``p_value`` is the HAC-adjusted p-value for H0: IC = 0.
    """
    from scipy.stats import rankdata
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.sandwich_covariance import cov_hac

    aligned = pd.concat([predicted, realized], axis=1).dropna()
    if len(aligned) < 4:
        return float("nan"), float("nan")

    y_hat = aligned.iloc[:, 0].values
    y_true = aligned.iloc[:, 1].values

    # Rank-transform both series (Spearman = Pearson on ranks)
    x_ranked = rankdata(y_hat).reshape(-1, 1)
    y_ranked = rankdata(y_true)

    # OLS of ranks on ranks (constant-free; IC is the slope if x is unit-scaled)
    # Add constant for valid OLS
    x_with_const = np.column_stack([np.ones(len(x_ranked)), x_ranked])
    model = OLS(y_ranked, x_with_const).fit()

    # HAC covariance with Newey-West
    hac_cov = cov_hac(model, nlags=lags)
    hac_se = np.sqrt(hac_cov[1, 1])
    slope = model.params[1]
    t_stat = slope / hac_se if hac_se > 0 else 0.0

    from scipy.stats import t as t_dist
    p_value = 2 * t_dist.sf(abs(t_stat), df=len(y_ranked) - 2)

    # Recover Spearman IC from the slope (scale by std(ranks))
    n = len(y_ranked)
    ic = float(np.corrcoef(rankdata(y_hat), rankdata(y_true))[0, 1])

    return ic, float(p_value)


def generate_rolling_ic_series(
    results: list["BacktestEventResult"],
    window_months: int = 24,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """
    Compute a rolling IC and hit-rate time series from monthly backtest results.

    Used to visualize whether predictive power is stable, trending, or regime-
    dependent.  Best used with output from ``run_monthly_stability_backtest()``.

    Args:
        results:       List of BacktestEventResult (typically monthly).
        window_months: Rolling window in months (default 24).
        benchmark:     If provided, filter to a single ETF benchmark.
                       If None, averages across all benchmarks per date.

    Returns:
        DataFrame with DatetimeIndex (event_date) and columns:
          ``ic_rolling``, ``hit_rate_rolling``, ``n_obs`` (observations in window).
        Sorted ascending by date.
    """
    if not results:
        return pd.DataFrame(
            columns=["ic_rolling", "hit_rate_rolling", "n_obs"]
        )

    rows = []
    for r in results:
        if benchmark is not None and r.benchmark != benchmark:
            continue
        rows.append({
            "event_date": pd.Timestamp(r.event.event_date),
            "ic":         r.ic_at_event,
            "correct":    float(r.correct_direction),
        })

    if not rows:
        return pd.DataFrame(columns=["ic_rolling", "hit_rate_rolling", "n_obs"])

    df = (
        pd.DataFrame(rows)
        .groupby("event_date")[["ic", "correct"]]
        .mean()
        .sort_index()
    )

    rolling = df.rolling(window=window_months, min_periods=window_months // 2)
    result = pd.DataFrame({
        "ic_rolling":       rolling["ic"].mean(),
        "hit_rate_rolling": rolling["correct"].mean(),
        "n_obs":            rolling["ic"].count(),
    })
    return result.dropna(how="all")


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
