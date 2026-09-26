"""Date-clustered (Driscoll-Kraay) inference for pooled forecast evaluation.

Review 2026-09-25, F13: the pooled OOS rows share dates. The eight benchmark
forecasts made on one date share the PGR leg of their targets, and consecutive
6-month targets overlap by five months. The old pooled IC p-value applied a
Newey-West correction with 5 lags over date-sorted rows, which spans less than
one date, so it treated the same-date rows as independent (p = 2.5e-5 against
0.015 when clustered by date).

Driscoll-Kraay sums each observation's regression score within its calendar
month and applies a Bartlett-kernel HAC to the monthly sums. It covers the
cross-sectional dependence and the overlap. With one observation per month it
reduces to Newey-West. p-values use a t distribution with (number of months
with data - 1) degrees of freedom.

Tests provided:

- ``rank_ic_test``: Spearman IC and the slope test of rank(y) on rank(y_hat).
- ``pesaran_timmermann_test``: directional accuracy. The regression of
  1{y > 0} on 1{y_hat > 0} (Pesaran and Timmermann 2009, which reduces to the
  1992 test for serially independent data). The slope is zero for any
  predictor whose calls are independent of the outcome, including one that
  always predicts the same sign, whatever the base rate.
- ``clark_west_test``: MSFE-adjusted test of the model against a benchmark
  forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import t as t_dist

from src.models.prequential import month_ordinals

Alternative = Literal["two-sided", "greater"]


@dataclass(frozen=True)
class RegressionTest:
    """Slope (or mean) estimate with a Driscoll-Kraay t-test."""

    coef: float
    se: float
    t_stat: float
    p_value: float
    n_obs: int
    n_periods: int


@dataclass(frozen=True)
class DirectionalAccuracyResult:
    """Hit rate against the base rate, with the Pesaran-Timmermann test.

    Rates are over the rows with a directional call (y_hat != 0); ``n_obs``
    counts all rows and ``n_calls`` the called ones.
    """

    n_obs: int
    hit_rate: float
    base_rate: float
    constant_rule_hit_rate: float
    excess_over_base_rate: float
    chance_hit_rate: float
    pt_stat: float
    pt_p_value: float
    n_calls: int = 0


@dataclass(frozen=True)
class ClarkWestResult:
    """Clark-West MSFE-adjusted test summary (one-sided: model beats benchmark)."""

    n_obs: int
    t_stat: float
    p_value: float
    mean_adjusted_differential: float


_NAN_TEST = RegressionTest(
    coef=float("nan"),
    se=float("nan"),
    t_stat=float("nan"),
    p_value=float("nan"),
    n_obs=0,
    n_periods=0,
)


def _time_codes(dates: pd.Index | pd.Series | np.ndarray | list | None, n_obs: int) -> np.ndarray:
    """Integer month codes starting at 0 (positions when no dates are given)."""
    if dates is None:
        return np.arange(n_obs, dtype=np.int64)
    months = month_ordinals(pd.Index(dates))
    if len(months) != n_obs:
        raise ValueError(f"dates has {len(months)} entries for {n_obs} observations.")
    return months - months.min() if len(months) else months


def driscoll_kraay_covariance(
    exog: np.ndarray,
    resid: np.ndarray,
    time_codes: np.ndarray,
    lags: int,
) -> np.ndarray:
    """Driscoll-Kraay (Bartlett) covariance of OLS coefficients.

    Scores ``x_i * u_i`` are summed within each period code; months without
    data contribute zero, so lags count calendar months. Uses the cluster
    small-sample factor ``G / (G - 1) * (N - 1) / (N - k)`` (G = periods with
    data), matching ``statsmodels`` ``cov_type="hac-groupsum"`` defaults.
    """
    exog = np.asarray(exog, dtype=float)
    resid = np.asarray(resid, dtype=float)
    n_obs, k_params = exog.shape
    scores = exog * resid[:, None]
    n_periods = int(time_codes.max()) + 1 if len(time_codes) else 0
    period_scores = np.zeros((n_periods, k_params), dtype=float)
    np.add.at(period_scores, time_codes, scores)
    omega = period_scores.T @ period_scores
    for lag in range(1, max(int(lags), 0) + 1):
        if lag >= n_periods:
            break
        weight = 1.0 - lag / (lags + 1.0)
        gamma = period_scores[lag:].T @ period_scores[:-lag]
        omega += weight * (gamma + gamma.T)
    bread = np.linalg.inv(exog.T @ exog)
    cov = bread @ omega @ bread
    n_groups = len(np.unique(time_codes))
    if n_groups > 1 and n_obs > k_params:
        cov *= n_groups / (n_groups - 1.0) * (n_obs - 1.0) / (n_obs - k_params)
    return cov


def _p_value(t_stat: float, df: int, alternative: Alternative) -> float:
    if not np.isfinite(t_stat):
        if np.isnan(t_stat):
            return float("nan")
        if alternative == "greater":
            return 0.0 if t_stat > 0 else 1.0
        return 0.0
    df = max(int(df), 1)
    if alternative == "greater":
        return float(t_dist.sf(t_stat, df=df))
    return float(2.0 * t_dist.sf(abs(t_stat), df=df))


def driscoll_kraay_ols(
    y: np.ndarray,
    x: np.ndarray | None,
    dates: pd.Index | pd.Series | np.ndarray | list | None,
    lags: int,
    alternative: Alternative = "two-sided",
) -> RegressionTest:
    """Test the slope of ``y`` on ``x`` (or the mean of ``y`` when ``x`` is None)."""
    y = np.asarray(y, dtype=float)
    n_obs = len(y)
    codes = _time_codes(dates, n_obs)
    if x is None:
        exog = np.ones((n_obs, 1), dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        exog = np.column_stack([np.ones(n_obs, dtype=float), x])
    valid = np.isfinite(y) & np.all(np.isfinite(exog), axis=1)
    y, exog, codes = y[valid], exog[valid], codes[valid]
    n_obs = len(y)
    n_periods = int(len(np.unique(codes)))
    k = exog.shape[1]
    if n_obs < max(4, k + 2) or n_periods < 3:
        return RegressionTest(float("nan"), float("nan"), float("nan"), float("nan"), n_obs, n_periods)
    if k == 2 and np.nanstd(exog[:, 1]) == 0.0:
        return RegressionTest(float("nan"), float("nan"), float("nan"), float("nan"), n_obs, n_periods)
    params, *_ = np.linalg.lstsq(exog, y, rcond=None)
    resid = y - exog @ params
    codes = codes - codes.min()
    cov = driscoll_kraay_covariance(exog, resid, codes, lags)
    coef = float(params[-1])
    var = float(cov[-1, -1])
    se = float(np.sqrt(var)) if var > 0 else 0.0
    if se <= 1e-15:
        t_stat = float("nan") if abs(coef) <= 1e-15 else float(np.sign(coef) * np.inf)
    else:
        t_stat = coef / se
    return RegressionTest(
        coef=coef,
        se=se,
        t_stat=float(t_stat),
        p_value=_p_value(float(t_stat), n_periods - 1, alternative),
        n_obs=n_obs,
        n_periods=n_periods,
    )


def _aligned(
    predicted: pd.Series | np.ndarray,
    realized: pd.Series | np.ndarray,
    dates: pd.Index | pd.Series | np.ndarray | list | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pred = np.asarray(predicted, dtype=float)
    real = np.asarray(realized, dtype=float)
    if len(pred) != len(real):
        raise ValueError("predicted and realized must have the same length.")
    date_arr = None
    if dates is not None:
        date_arr = np.asarray(pd.Index(dates))
        if len(date_arr) != len(pred):
            raise ValueError("dates must have the same length as predicted.")
    valid = np.isfinite(pred) & np.isfinite(real)
    return pred[valid], real[valid], (date_arr[valid] if date_arr is not None else None)


def rank_ic_test(
    predicted: pd.Series | np.ndarray,
    realized: pd.Series | np.ndarray,
    dates: pd.Index | pd.Series | np.ndarray | list | None,
    lags: int,
) -> tuple[float, float]:
    """Spearman IC and its two-sided Driscoll-Kraay p-value."""
    pred, real, date_arr = _aligned(predicted, realized, dates)
    if len(pred) < 4:
        return float("nan"), float("nan")
    x_rank = rankdata(pred)
    y_rank = rankdata(real)
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return 0.0, float("nan")
    ic = float(np.corrcoef(x_rank, y_rank)[0, 1])
    test = driscoll_kraay_ols(y_rank, x_rank, date_arr, lags, alternative="two-sided")
    return (ic if np.isfinite(ic) else 0.0), test.p_value


def pesaran_timmermann_test(
    predicted: pd.Series | np.ndarray,
    realized: pd.Series | np.ndarray,
    dates: pd.Index | pd.Series | np.ndarray | list | None,
    lags: int,
) -> DirectionalAccuracyResult:
    """Directional accuracy against the base rate, with a one-sided PT test.

    Only rows with a directional call count: a zero forecast (e.g. while the
    prequential shrinkage alpha is 0) makes no call. Over those rows,
    ``hit_rate`` is the share where sign(y_hat) == sign(y), ``base_rate`` is
    P(y > 0) and ``constant_rule_hit_rate`` is max(p, 1 - p), the hit rate of
    always calling the more frequent sign. The test regresses 1{y > 0} on
    1{y_hat > 0}; its slope equals P(y > 0 | up call) - P(y > 0 | down call),
    and ``chance_hit_rate`` is the PT expectation p_y p_x + (1 - p_y)(1 - p_x)
    under independence.
    """
    pred, real, date_arr = _aligned(predicted, realized, dates)
    n_obs = len(pred)
    called = pred != 0.0
    n_calls = int(called.sum())
    if n_calls == 0:
        nan = float("nan")
        return DirectionalAccuracyResult(n_obs, nan, nan, nan, nan, nan, nan, nan, 0)
    pred_c, real_c = pred[called], real[called]
    hit_rate = float(np.mean(np.sign(pred_c) == np.sign(real_c)))
    up_call = (pred_c > 0).astype(float)
    up_outcome = (real_c > 0).astype(float)
    base_rate = float(np.mean(up_outcome))
    call_rate = float(np.mean(up_call))
    constant_rule = max(base_rate, 1.0 - base_rate)
    chance = base_rate * call_rate + (1.0 - base_rate) * (1.0 - call_rate)
    if date_arr is not None:
        called_dates = date_arr[called]
    else:
        called_dates = np.flatnonzero(called)
    test = driscoll_kraay_ols(up_outcome, up_call, called_dates, lags, alternative="greater")
    return DirectionalAccuracyResult(
        n_obs=n_obs,
        hit_rate=hit_rate,
        base_rate=base_rate,
        constant_rule_hit_rate=float(constant_rule),
        excess_over_base_rate=float(hit_rate - constant_rule),
        chance_hit_rate=float(chance),
        pt_stat=test.t_stat,
        pt_p_value=test.p_value,
        n_calls=n_calls,
    )


def clark_west_test(
    predicted: pd.Series | np.ndarray,
    realized: pd.Series | np.ndarray,
    benchmark_forecast: pd.Series | np.ndarray,
    dates: pd.Index | pd.Series | np.ndarray | list | None,
    lags: int,
) -> ClarkWestResult:
    """One-sided Clark-West test of the model against ``benchmark_forecast``."""
    pred = np.asarray(predicted, dtype=float)
    real = np.asarray(realized, dtype=float)
    bench = np.asarray(benchmark_forecast, dtype=float)
    date_arr = np.asarray(pd.Index(dates)) if dates is not None else None
    valid = np.isfinite(pred) & np.isfinite(real) & np.isfinite(bench)
    pred, real, bench = pred[valid], real[valid], bench[valid]
    date_arr = date_arr[valid] if date_arr is not None else None
    if len(pred) < 4:
        return ClarkWestResult(int(len(pred)), float("nan"), float("nan"), float("nan"))
    adjusted = (real - bench) ** 2 - ((real - pred) ** 2 - (pred - bench) ** 2)
    test = driscoll_kraay_ols(adjusted, None, date_arr, lags, alternative="greater")
    return ClarkWestResult(
        n_obs=int(len(adjusted)),
        t_stat=test.t_stat,
        p_value=test.p_value,
        mean_adjusted_differential=float(np.mean(adjusted)),
    )


def oos_r2_against(
    predicted: pd.Series | np.ndarray,
    realized: pd.Series | np.ndarray,
    benchmark_forecast: pd.Series | np.ndarray,
) -> float:
    """``1 - SSE(model) / SSE(benchmark)`` over rows where all three exist."""
    pred = np.asarray(predicted, dtype=float)
    real = np.asarray(realized, dtype=float)
    bench = np.asarray(benchmark_forecast, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(real) & np.isfinite(bench)
    if valid.sum() < 2:
        return float("nan")
    sse_naive = float(np.sum((real[valid] - bench[valid]) ** 2))
    if sse_naive <= 0.0:
        return float("nan")
    sse_model = float(np.sum((real[valid] - pred[valid]) ** 2))
    return float(1.0 - sse_model / sse_naive)


__all__ = [
    "ClarkWestResult",
    "DirectionalAccuracyResult",
    "RegressionTest",
    "clark_west_test",
    "driscoll_kraay_covariance",
    "driscoll_kraay_ols",
    "oos_r2_against",
    "pesaran_timmermann_test",
    "rank_ic_test",
]
