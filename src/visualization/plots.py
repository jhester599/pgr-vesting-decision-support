"""
Visualization module for the PGR Vesting Decision Support engine.

All figures are saved to /plots/*.png.  plt.show() is never called
(headless-compatible).

v1 outputs:
  - plots/wfo_equity_curve.png    : Out-of-sample predicted vs. actual returns
  - plots/feature_importance.png  : Lasso coefficient magnitudes across WFO folds
  - plots/portfolio_drift.png     : Current vs. target sector allocation

v2 outputs (backtest & multi-benchmark):
  - plots/backtest_heatmap_<horizon>m.png  : Event × benchmark realized return grid
  - plots/hit_rate_by_benchmark.png        : Bar chart of directional hit rates
  - plots/predicted_vs_realized_<etf>.png  : Scatter per vesting event
  - plots/multi_benchmark_signals.png      : Current signal bar chart
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.models.wfo_engine import WFOResult
    from src.portfolio.drift_analyzer import PortfolioState

logger = logging.getLogger(__name__)

_PLOTS_DIR = "plots"


def _ensure_plots_dir() -> None:
    os.makedirs(_PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. WFO equity curve
# ---------------------------------------------------------------------------

def plot_wfo_equity_curve(wfo_result: "WFOResult") -> str:
    """
    Plot concatenated out-of-sample predicted vs. actual 6-month returns.

    Returns the path to the saved PNG.
    """
    _ensure_plots_dir()

    # Concatenate all out-of-sample folds
    dates, y_true_all, y_hat_all = [], [], []
    for fold in wfo_result.folds:
        n = len(fold.y_true)
        # Use the test period end dates if available; otherwise use fold index
        try:
            fold_dates = pd.date_range(
                start=fold.test_start, periods=n, freq="MS"
            )
        except Exception as exc:
            logger.exception(
                "Could not build plot dates for fold %s; using positional fallback. Error=%r",
                fold.fold_idx,
                exc,
            )
            fold_dates = range(len(dates), len(dates) + n)
        dates.extend(fold_dates)
        y_true_all.extend(fold.y_true.tolist())
        y_hat_all.extend(fold.y_hat.tolist())

    dates_arr = pd.to_datetime(dates) if isinstance(dates[0], pd.Timestamp) else dates
    y_true_arr = np.array(y_true_all)
    y_hat_arr = np.array(y_hat_all)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle(
        f"PGR Walk-Forward Optimization — Out-of-Sample Performance\n"
        f"IC={wfo_result.information_coefficient:.4f}  "
        f"Hit Rate={wfo_result.hit_rate:.1%}  "
        f"MAE={wfo_result.mean_absolute_error:.4f}",
        fontsize=12,
    )

    # Top panel: cumulative returns (actual vs. predicted signal direction)
    ax1 = axes[0]
    cum_actual = (1 + y_true_arr).cumprod() - 1
    cum_predicted = (1 + y_hat_arr).cumprod() - 1
    ax1.plot(cum_actual, label="Actual 6M Return (cumulative)", color="#1f77b4", linewidth=1.5)
    ax1.plot(cum_predicted, label="Predicted (model signal)", color="#ff7f0e",
             linewidth=1.5, linestyle="--")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Cumulative Out-of-Sample Returns")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: scatter actual vs. predicted
    ax2 = axes[1]
    ax2.scatter(y_hat_arr, y_true_arr, alpha=0.4, s=18, color="#2ca02c")
    # Add a regression line
    if len(y_hat_arr) > 2:
        m, b = np.polyfit(y_hat_arr, y_true_arr, 1)
        x_line = np.linspace(y_hat_arr.min(), y_hat_arr.max(), 100)
        ax2.plot(x_line, m * x_line + b, color="red", linewidth=1.0,
                 linestyle="--", label=f"OLS fit (slope={m:.2f})")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax2.set_xlabel("Predicted Return")
    ax2.set_ylabel("Actual Return")
    ax2.set_title("Predicted vs. Actual 6-Month Returns (Scatter)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_PLOTS_DIR, "wfo_equity_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(wfo_result: "WFOResult") -> str:
    """
    Plot Lasso coefficient magnitudes averaged across all WFO folds.

    Returns the path to the saved PNG.
    """
    _ensure_plots_dir()

    # Aggregate importances across folds
    all_features: dict[str, list[float]] = {}
    for fold in wfo_result.folds:
        for feat, coef in fold.feature_importances.items():
            all_features.setdefault(feat, []).append(abs(coef))

    if not all_features:
        return ""

    feat_names = list(all_features.keys())
    mean_coefs = [np.mean(all_features[f]) for f in feat_names]
    std_coefs = [np.std(all_features[f]) for f in feat_names]

    # Sort descending
    order = np.argsort(mean_coefs)[::-1]
    feat_names = [feat_names[i] for i in order]
    mean_coefs = [mean_coefs[i] for i in order]
    std_coefs = [std_coefs[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(feat_names) * 0.45)))
    y_pos = range(len(feat_names))
    bars = ax.barh(y_pos, mean_coefs, xerr=std_coefs, align="center",
                   color="#1f77b4", alpha=0.8, capsize=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feat_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("|Lasso Coefficient| (mean ± std across WFO folds)")
    ax.set_title(
        f"PGR Model — Feature Importance\n"
        f"({len(wfo_result.folds)} WFO folds, Lasso regularization)"
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_PLOTS_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Portfolio drift
# ---------------------------------------------------------------------------

def plot_portfolio_drift(portfolio_state: "PortfolioState") -> str:
    """
    Plot current sector weights vs. MSCI ACWI target equilibrium weights.

    Returns the path to the saved PNG.
    """
    _ensure_plots_dir()

    from src.portfolio.drift_analyzer import (
        compute_sector_weights,
        compute_sector_deviation,
        MARKET_EQUILIBRIUM_WEIGHTS,
    )

    current_weights = compute_sector_weights(portfolio_state)
    deviations = compute_sector_deviation(portfolio_state)

    sectors = sorted(MARKET_EQUILIBRIUM_WEIGHTS.keys())
    current = [current_weights.get(s, 0.0) for s in sectors]
    target = [MARKET_EQUILIBRIUM_WEIGHTS[s] for s in sectors]
    devs = [deviations.get(s, 0.0) for s in sectors]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PGR Portfolio — Sector Allocation vs. Target", fontsize=13)

    # Left panel: grouped bar chart
    ax1 = axes[0]
    x = np.arange(len(sectors))
    width = 0.35
    bars_curr = ax1.bar(x - width / 2, current, width, label="Current", color="#1f77b4", alpha=0.8)
    bars_tgt = ax1.bar(x + width / 2, target, width, label="Target (MSCI ACWI)", color="#ff7f0e", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sectors, rotation=30, ha="right", fontsize=9)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax1.set_ylabel("Portfolio Weight")
    ax1.set_title("Current vs. Target Sector Weights")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # Right panel: deviation bar chart (current - target)
    ax2 = axes[1]
    colors = ["#d62728" if d > 0 else "#2ca02c" for d in devs]
    ax2.barh(sectors, devs, color=colors, alpha=0.8)
    ax2.axvline(0, color="black", linewidth=1.0)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax2.set_xlabel("Deviation (Current - Target)")
    ax2.set_title("Sector Drift (red=overweight, green=underweight)")
    ax2.grid(True, axis="x", alpha=0.3)

    total_val = portfolio_state.total_value
    pgr_pct = portfolio_state.pgr_value / total_val if total_val > 0 else 0
    fig.text(
        0.5, 0.01,
        f"Total Portfolio: ${total_val:,.0f}  |  PGR Concentration: {pgr_pct:.1%}",
        ha="center", fontsize=10, style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(_PLOTS_DIR, "portfolio_drift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# v2 Plot 4: Backtest heatmap
# ---------------------------------------------------------------------------

def plot_backtest_heatmap(
    results: list,
    horizon: int,
) -> str:
    """
    Heatmap of realized relative returns: rows = vesting events, columns = ETF
    benchmarks.  Cells are color-coded by return magnitude; correct directional
    predictions are marked with a checkmark, incorrect with an X.

    Args:
        results: List of BacktestEventResult from run_historical_backtest().
        horizon: Target horizon in months (6 or 12).

    Returns:
        Path to the saved PNG, or ``""`` if no data.
    """
    from src.reporting.backtest_report import (
        generate_backtest_table,
        generate_correct_direction_table,
    )

    realized = generate_backtest_table(results, horizon)
    if realized.empty:
        return ""

    correct = generate_correct_direction_table(results, horizon)

    _ensure_plots_dir()

    n_events, n_bench = realized.shape
    fig_w = max(12, n_bench * 0.7)
    fig_h = max(6, n_events * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vals = realized.values.astype(float)
    vmax = np.nanpercentile(np.abs(vals), 95)
    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_bench))
    ax.set_xticklabels(realized.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_events))
    ax.set_yticklabels(
        [str(d) for d in realized.index], fontsize=8
    )

    # Overlay correctness markers
    for i, event_date in enumerate(realized.index):
        for j, bench in enumerate(realized.columns):
            if bench in correct.columns and event_date in correct.index:
                is_correct = correct.loc[event_date, bench]
                marker = "✓" if is_correct else "✗"
                color = "black"
                ax.text(j, i, marker, ha="center", va="center",
                        fontsize=7, color=color, alpha=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Realized Relative Return (PGR − ETF)", fontsize=9)
    ax.set_title(
        f"PGR Backtest — Realized Relative Return Heatmap ({horizon}M horizon)\n"
        "✓ = correct directional prediction, ✗ = incorrect",
        fontsize=11,
    )
    plt.tight_layout()

    path = os.path.join(_PLOTS_DIR, f"backtest_heatmap_{horizon}m.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# v2 Plot 5: Hit rate by benchmark
# ---------------------------------------------------------------------------

def plot_hit_rate_by_benchmark(
    results: list,
) -> str:
    """
    Horizontal bar chart of directional hit rate per ETF benchmark, with
    separate bars for the 6M and 12M target horizons.

    Args:
        results: List of BacktestEventResult (all horizons combined).

    Returns:
        Path to the saved PNG, or ``""`` if no data.
    """
    if not results:
        return ""

    _ensure_plots_dir()
    import collections

    # hit_rates[horizon][benchmark] = (n_correct, n_total)
    hit_data: dict[int, dict[str, list[bool]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for r in results:
        hit_data[r.target_horizon][r.benchmark].append(r.correct_direction)

    horizons = sorted(hit_data.keys())
    if not horizons:
        return ""

    # Collect benchmark names across all horizons
    all_benchmarks = sorted(
        {b for h in hit_data.values() for b in h.keys()}
    )

    n_bench = len(all_benchmarks)
    fig, ax = plt.subplots(figsize=(10, max(5, n_bench * 0.45)))

    bar_height = 0.35
    y = np.arange(n_bench)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, horizon in enumerate(horizons):
        hit_rates = []
        for bench in all_benchmarks:
            vals = hit_data[horizon].get(bench, [])
            hit_rates.append(np.mean(vals) if vals else np.nan)
        offset = (i - len(horizons) / 2 + 0.5) * bar_height
        ax.barh(
            y + offset, hit_rates, bar_height,
            label=f"{horizon}M", color=colors[i % len(colors)], alpha=0.85,
        )

    ax.axvline(0.5, color="black", linewidth=1.0, linestyle="--",
               label="50% baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(all_benchmarks, fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Directional Hit Rate")
    ax.set_title(
        "PGR Backtest — Directional Hit Rate by Benchmark\n"
        "(fraction of vesting events where PGR direction was correctly predicted)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    path = os.path.join(_PLOTS_DIR, "hit_rate_by_benchmark.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# v2 Plot 6: Predicted vs. realized scatter
# ---------------------------------------------------------------------------

def plot_predicted_vs_realized_scatter(
    results: list,
    benchmark: str,
) -> str:
    """
    Scatter plot of predicted vs. realized relative return for one benchmark,
    one point per vesting event, colored by RSU type (time/performance).

    Args:
        results:   List of BacktestEventResult.
        benchmark: ETF ticker to filter on (e.g. ``"VTI"``).

    Returns:
        Path to the saved PNG, or ``""`` if no data for this benchmark.
    """
    subset = [r for r in results if r.benchmark == benchmark]
    if not subset:
        return ""

    _ensure_plots_dir()

    time_r = [r for r in subset if r.event.rsu_type == "time"]
    perf_r = [r for r in subset if r.event.rsu_type == "performance"]

    fig, ax = plt.subplots(figsize=(8, 7))

    for group, color, label in [
        (time_r, "#1f77b4", "Time RSU (January)"),
        (perf_r, "#ff7f0e", "Performance RSU (July)"),
    ]:
        if group:
            ax.scatter(
                [r.predicted_relative_return for r in group],
                [r.realized_relative_return for r in group],
                color=color, alpha=0.75, s=60, label=label,
            )

    all_vals = (
        [r.predicted_relative_return for r in subset]
        + [r.realized_relative_return for r in subset]
    )
    all_vals = [v for v in all_vals if not np.isnan(v)]
    if all_vals:
        lim = max(abs(min(all_vals)), abs(max(all_vals))) * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.8, alpha=0.4, label="Perfect prediction")

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.set_xlabel("Predicted Relative Return (PGR − ETF)")
    ax.set_ylabel("Realized Relative Return (PGR − ETF)")
    ax.set_title(f"PGR vs. {benchmark} — Predicted vs. Realized\n(all vesting events)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # IC annotation
    predicted = [r.predicted_relative_return for r in subset]
    realized = [r.realized_relative_return for r in subset
                if not np.isnan(r.realized_relative_return)]
    if len(predicted) == len(realized) and len(realized) > 2:
        from scipy.stats import spearmanr
        ic, _ = spearmanr(predicted, realized)
        ax.text(
            0.04, 0.96, f"IC (Spearman) = {ic:.3f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    safe_bench = benchmark.replace("/", "_")
    path = os.path.join(_PLOTS_DIR, f"predicted_vs_realized_{safe_bench}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# v2 Plot 7: Multi-benchmark current signals
# ---------------------------------------------------------------------------

def plot_multi_benchmark_signals(
    signals_df: pd.DataFrame,
) -> str:
    """
    Horizontal bar chart of the current model predictions across all ETF
    benchmarks.  Bars are colored green (OUTPERFORM), red (UNDERPERFORM), or
    grey (NEUTRAL).

    Args:
        signals_df: DataFrame from ``get_current_signals()`` with index =
                    benchmark ticker and columns including
                    ``predicted_relative_return`` and ``signal``.

    Returns:
        Path to the saved PNG, or ``""`` if signals_df is empty.
    """
    if signals_df.empty or "predicted_relative_return" not in signals_df.columns:
        return ""

    _ensure_plots_dir()

    df = signals_df.sort_values("predicted_relative_return", ascending=True)
    benchmarks = df.index.tolist()
    values = df["predicted_relative_return"].values

    signal_col = df["signal"] if "signal" in df.columns else pd.Series(
        ["NEUTRAL"] * len(df), index=df.index
    )
    colors = []
    for sig in signal_col.loc[benchmarks]:
        if sig == "OUTPERFORM":
            colors.append("#2ca02c")
        elif sig == "UNDERPERFORM":
            colors.append("#d62728")
        else:
            colors.append("#7f7f7f")

    fig, ax = plt.subplots(figsize=(10, max(5, len(benchmarks) * 0.45)))
    ax.barh(benchmarks, values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax.set_xlabel("Predicted Relative Return (PGR − ETF)")
    ax.set_title(
        "PGR Current Signals — Live Model Predictions\n"
        "Green = OUTPERFORM (hold PGR), Red = UNDERPERFORM (consider selling), "
        "Grey = NEUTRAL"
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_PLOTS_DIR, "multi_benchmark_signals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
