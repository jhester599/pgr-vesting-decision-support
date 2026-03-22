"""
Visualization module for the PGR Vesting Decision Support engine.

All figures are saved to /plots/*.png.  plt.show() is never called
(headless-compatible).

Three outputs:
  - plots/wfo_equity_curve.png    : Out-of-sample predicted vs. actual returns
  - plots/feature_importance.png  : Lasso coefficient magnitudes across WFO folds
  - plots/portfolio_drift.png     : Current vs. target sector allocation
"""

from __future__ import annotations

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
        except Exception:
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
