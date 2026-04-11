"""Reusable forecast diagnostics for ensemble evaluation and reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

from src.reporting.backtest_report import compute_newey_west_ic, compute_oos_r_squared


@dataclass(frozen=True)
class ClarkWestResult:
    """Clark-West MSFE-adjusted test summary."""

    n_obs: int
    t_stat: float
    p_value: float
    mean_adjusted_differential: float


def expanding_mean_benchmark(realized: pd.Series) -> pd.Series:
    """Return the expanding historical-mean forecast used as the naive benchmark."""
    clean = realized.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float, name="benchmark_forecast")

    values = clean.to_numpy(dtype=float)
    benchmark = np.empty(len(values), dtype=float)
    benchmark[0] = values[0]
    for idx in range(1, len(values)):
        benchmark[idx] = float(values[:idx].mean())
    return pd.Series(benchmark, index=clean.index, name="benchmark_forecast")


def compute_clark_west_result(
    predicted: pd.Series,
    realized: pd.Series,
    lags: int,
) -> ClarkWestResult:
    """Run a one-sided Clark-West test versus the expanding historical mean."""
    aligned = pd.concat([predicted, realized], axis=1).dropna()
    if len(aligned) < 4:
        return ClarkWestResult(
            n_obs=int(len(aligned)),
            t_stat=float("nan"),
            p_value=float("nan"),
            mean_adjusted_differential=float("nan"),
        )

    model_pred = aligned.iloc[:, 0].astype(float)
    y_true = aligned.iloc[:, 1].astype(float)
    benchmark_pred = expanding_mean_benchmark(y_true)

    error_benchmark = y_true - benchmark_pred
    error_model = y_true - model_pred
    adjusted_diff = error_benchmark.pow(2) - (
        error_model.pow(2) - (model_pred - benchmark_pred).pow(2)
    )

    x_values = np.ones((len(adjusted_diff), 1), dtype=float)
    model = OLS(adjusted_diff.to_numpy(dtype=float), x_values).fit()
    hac_cov = cov_hac(model, nlags=max(1, lags))
    hac_se = float(np.sqrt(hac_cov[0, 0]))
    mean_adjusted_diff = float(adjusted_diff.mean())
    t_stat = mean_adjusted_diff / hac_se if hac_se > 0 else 0.0

    from scipy.stats import t as t_dist

    p_value = float(t_dist.sf(t_stat, df=len(adjusted_diff) - 1))
    return ClarkWestResult(
        n_obs=int(len(adjusted_diff)),
        t_stat=float(t_stat),
        p_value=p_value,
        mean_adjusted_differential=mean_adjusted_diff,
    )


def summarize_prediction_diagnostics(
    predicted: pd.Series,
    realized: pd.Series,
    target_horizon_months: int = 6,
) -> dict[str, float | int]:
    """Compute OOS R², Newey-West IC, hit rate, and Clark-West diagnostics."""
    aligned = pd.concat([predicted, realized], axis=1).dropna()
    if aligned.empty:
        return {
            "n_obs": 0,
            "oos_r2": float("nan"),
            "nw_ic": float("nan"),
            "nw_p_value": float("nan"),
            "hit_rate": float("nan"),
            "cw_t_stat": float("nan"),
            "cw_p_value": float("nan"),
            "cw_mean_adjusted_differential": float("nan"),
        }

    y_hat = aligned.iloc[:, 0]
    y_true = aligned.iloc[:, 1]
    lags = max(1, target_horizon_months - 1)
    nw_ic, nw_p_value = compute_newey_west_ic(y_hat, y_true, lags=lags)
    cw = compute_clark_west_result(y_hat, y_true, lags=lags)

    return {
        "n_obs": int(len(aligned)),
        "oos_r2": float(compute_oos_r_squared(y_hat, y_true)),
        "nw_ic": float(nw_ic),
        "nw_p_value": float(nw_p_value),
        "hit_rate": float(np.mean(np.sign(y_true) == np.sign(y_hat))),
        "cw_t_stat": float(cw.t_stat),
        "cw_p_value": float(cw.p_value),
        "cw_mean_adjusted_differential": float(cw.mean_adjusted_differential),
    }
