"""
v5.2 — Conformal Prediction Intervals for PGR Ensemble Predictions.

Produces distribution-free prediction intervals with marginal coverage
guarantees for time-series data.  Two methods are provided:

  1. Split Conformal (Papadopoulos et al. 2002 / Vovk et al. 2005):
     Uses WFO OOS residuals as the calibration set.  Marginal coverage
     guarantee: P(y ∈ CI) ≥ 1-α, with finite-sample correction +1/(n+1).
     Simple, interpretable, no additional model fitting required.

  2. Adaptive Conformal Inference — ACI (Gibbs & Candès 2021):
     Adjusts the effective coverage level α_t at each chronological step
     based on whether the prior fold's prediction interval covered the
     true outcome.  Provides valid marginal coverage under distribution
     shift and non-stationarity — critical for 6-month overlapping return
     windows where the data-generating process changes over time.

     Update rule:  α_{t+1} = clip(α_t + γ(α_nominal − err_t), 0.01, 0.99)
     where err_t = 0 if covered, 1 if not covered.
     γ = CONFORMAL_ACI_GAMMA (default 0.05).

Both methods use symmetric intervals (±q̂ around ŷ) which are easier to
interpret in a financial context ("predicted +3.5% ± 8.2%").

MAPIE (≥1.3.0) is listed in requirements.txt and used here for the
TimeSeriesRegressor validation path; the production monthly pipeline uses
the native split/ACI implementation to avoid the latency of a full MAPIE
refit during the monthly batch run.

References:
  Papadopoulos et al. (2002): "Inductive Confidence Machines for Regression"
  Vovk et al. (2005): "Algorithmic Learning in a Random World"
  Gibbs & Candès (2021): "Adaptive Conformal Inference Under Distribution Shift"
  Xu & Xie (2021): "Conformal Prediction Interval for Dynamic Time-Series"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConformalResult:
    """
    Prediction interval produced by conformal calibration.

    Attributes:
        lower:              Lower bound of the prediction interval.
        upper:              Upper bound of the prediction interval.
        width:              ``upper − lower`` (total CI width).
        coverage_level:     Nominal coverage (e.g., 0.80 for an 80% CI).
        empirical_coverage: Fraction of calibration residuals inside the
                            interval — should be ≥ ``coverage_level``.
        n_calibration:      Number of calibration residuals used.
        method:             ``"split"`` or ``"aci"``.
    """
    lower: float
    upper: float
    width: float
    coverage_level: float
    empirical_coverage: float
    n_calibration: int
    method: str


@dataclass(frozen=True)
class ConformalCoverageBacktest:
    """Empirical interval coverage measured over historical sequential OOS points.

    ``covered``, ``widths`` and ``evaluated_dates`` describe each scored point
    (dates are positions when no dates were given).
    """

    n_evaluated: int
    empirical_coverage: float
    target_coverage: float
    coverage_gap: float
    trailing_n: int
    trailing_empirical_coverage: float
    trailing_coverage_gap: float
    method: str
    covered: tuple[bool, ...] = ()
    widths: tuple[float, ...] = ()
    evaluated_dates: tuple = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conformal_quantile_level(n: int, alpha: float) -> float:
    """
    Finite-sample adjusted quantile level for split conformal.

    Returns ``ceil((1-α)(n+1)) / n``, capped at 1.0.  This ensures
    marginal coverage P(y ∈ CI) ≥ 1-α for any calibration set of size n.

    Args:
        n:     Number of calibration residuals.
        alpha: Miscoverage level (= 1 - coverage_level).

    Returns:
        Adjusted quantile level ∈ (0, 1].
    """
    return min(float(np.ceil((1.0 - alpha) * (n + 1)) / n), 1.0)


# ---------------------------------------------------------------------------
# Split conformal
# ---------------------------------------------------------------------------

def split_conformal_interval(
    y_hat_current: float,
    residuals: np.ndarray,
    coverage: float = 0.80,
) -> ConformalResult:
    """
    Symmetric split conformal prediction interval.

    Calibrates on the absolute WFO OOS residuals ``|y_true − ŷ|`` and
    produces a symmetric interval ``[ŷ − q̂, ŷ + q̂]`` where ``q̂`` is
    the finite-sample corrected quantile of the absolute residuals.

    Marginal coverage guarantee: P(y ∈ CI) ≥ 1-α on exchangeable data.
    For time-series with distribution shift, coverage is approximate —
    use ``aci_adjusted_interval()`` for a stronger guarantee.

    Args:
        y_hat_current:  Ensemble point prediction for the current period.
        residuals:      Signed WFO OOS residuals (y_true − ŷ), shape (n,).
                        The absolute values are used for the symmetric CI.
        coverage:       Nominal coverage level (default: 0.80).

    Returns:
        ``ConformalResult`` with the symmetric prediction interval.

    Raises:
        ValueError: If ``residuals`` is empty.
    """
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) == 0:
        raise ValueError("residuals must be non-empty for conformal calibration.")

    alpha = 1.0 - coverage
    abs_resid = np.abs(residuals)
    n = len(abs_resid)

    q_level = _conformal_quantile_level(n, alpha)
    q_hat = float(np.quantile(abs_resid, q_level))

    empirical = float(np.mean(abs_resid <= q_hat))

    return ConformalResult(
        lower=float(y_hat_current) - q_hat,
        upper=float(y_hat_current) + q_hat,
        width=2.0 * q_hat,
        coverage_level=coverage,
        empirical_coverage=empirical,
        n_calibration=n,
        method="split",
    )


# ---------------------------------------------------------------------------
# Adaptive Conformal Inference (ACI)
# ---------------------------------------------------------------------------

def aci_adjusted_interval(
    y_hat_current: float,
    residuals: np.ndarray,
    nominal_coverage: float = 0.80,
    gamma: float = 0.05,
) -> ConformalResult:
    """
    Adaptive Conformal Inference (ACI) prediction interval.

    Runs ACI forward through the chronological WFO fold residuals to arrive
    at the ACI-adjusted coverage level α_T, then applies split conformal
    with that effective coverage.

    ACI update rule at each step t:
        err_t  = 0 if |e_t| ≤ q_t (covered), 1 otherwise
        α_{t+1} = clip(α_t + γ(α_nominal − err_t), 0.01, 0.99)

    where ``q_t`` is the split conformal quantile built from residuals
    before step t.  When the model covers more than expected, α increases
    (CI narrows); when it misses, α decreases (CI widens).

    Args:
        y_hat_current:   Current ensemble point prediction.
        residuals:       Signed WFO OOS residuals in chronological order,
                         shape (n,).
        nominal_coverage: Target coverage (default: 0.80).
        gamma:           ACI step size (default: 0.05).

    Returns:
        ``ConformalResult`` using the ACI-adjusted effective coverage.
        Falls back to split conformal when n < 4.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)

    if n < 4:
        result = split_conformal_interval(y_hat_current, residuals, nominal_coverage)
        return ConformalResult(
            lower=result.lower, upper=result.upper, width=result.width,
            coverage_level=nominal_coverage, empirical_coverage=result.empirical_coverage,
            n_calibration=n, method="aci",
        )

    alpha_nominal = 1.0 - nominal_coverage
    alpha_t = alpha_nominal
    abs_resid = np.abs(residuals)

    # Walk forward chronologically, each step using all prior residuals
    for t in range(1, n):
        calib_so_far = abs_resid[:t]
        q_level = _conformal_quantile_level(len(calib_so_far), alpha_t)
        q_t = float(np.quantile(calib_so_far, q_level))
        # err_t: 1 if not covered, 0 if covered
        err_t = float(abs_resid[t] > q_t)
        alpha_t = float(np.clip(alpha_t + gamma * (alpha_nominal - err_t), 0.01, 0.99))

    # Apply final α_T to the full residual set
    effective_coverage = 1.0 - alpha_t
    q_level_final = _conformal_quantile_level(n, alpha_t)
    q_hat = float(np.quantile(abs_resid, q_level_final))
    empirical = float(np.mean(abs_resid <= q_hat))

    return ConformalResult(
        lower=float(y_hat_current) - q_hat,
        upper=float(y_hat_current) + q_hat,
        width=2.0 * q_hat,
        coverage_level=effective_coverage,
        empirical_coverage=empirical,
        n_calibration=n,
        method="aci",
    )


# ---------------------------------------------------------------------------
# Main entry point: per-benchmark interval
# ---------------------------------------------------------------------------

def conformal_interval_from_ensemble(
    y_hat_current: float,
    y_hat_oos: np.ndarray,
    y_true_oos: np.ndarray,
    coverage: float = 0.80,
    method: str = "aci",
    gamma: float = 0.05,
) -> ConformalResult:
    """
    Compute a conformal prediction interval for the current ensemble prediction.

    Uses the WFO OOS ensemble predictions and realized returns (in
    chronological order) as the calibration set.

    Args:
        y_hat_current: Current inverse-variance ensemble point prediction.
        y_hat_oos:     Historical OOS ensemble predictions, shape (n,),
                       in chronological order.
        y_true_oos:    Realized relative returns corresponding to
                       ``y_hat_oos``, shape (n,), same order.
        coverage:      Nominal coverage (default: 0.80 = 80% CI).
        method:        ``"split"`` or ``"aci"`` (default: ``"aci"``).
        gamma:         ACI step size (only used when ``method="aci"``).

    Returns:
        ``ConformalResult`` with the prediction interval for the current
        period and metadata (method, n_calibration, empirical_coverage).

    Raises:
        ValueError: If ``y_hat_oos`` and ``y_true_oos`` have different lengths
                    or if ``method`` is not ``"split"`` or ``"aci"``.
    """
    y_hat_oos = np.asarray(y_hat_oos, dtype=float)
    y_true_oos = np.asarray(y_true_oos, dtype=float)

    if len(y_hat_oos) != len(y_true_oos):
        raise ValueError(
            f"y_hat_oos and y_true_oos must have the same length; "
            f"got {len(y_hat_oos)} and {len(y_true_oos)}."
        )
    if method not in ("split", "aci"):
        raise ValueError(f"method must be 'split' or 'aci'; got '{method}'.")

    residuals = y_true_oos - y_hat_oos  # signed, chronological

    if method == "split":
        return split_conformal_interval(y_hat_current, residuals, coverage)
    else:
        return aci_adjusted_interval(y_hat_current, residuals, coverage, gamma)


def backtest_conformal_coverage(
    y_hat_oos: np.ndarray,
    y_true_oos: np.ndarray,
    coverage: float = 0.80,
    method: str = "aci",
    gamma: float = 0.05,
    trailing_window: int = 12,
    min_calibration: int | None = None,
    dates=None,
    horizon_months: int = 1,
) -> ConformalCoverageBacktest:
    """
    Evaluate realized conformal coverage over sequential historical OOS points.

    For each chronological point ``t`` this builds the interval for
    ``y_hat_oos[t]`` from the residuals that had been realised by ``t`` only:
    a residual dated ``d`` counts when month(d) + ``horizon_months`` <=
    month(t). With 6-month targets the five preceding residuals are not yet
    known at ``t`` (review 2026-09-25, F13: they used to be included). Points
    with fewer than ``min_calibration`` realised residuals are skipped.

    ``dates`` (sorted ascending) gives each point's date; without it, points
    are consecutive months. ``horizon_months=1`` reproduces "all prior points".

    The trailing value answers the production question: "Over the most recent
    12 OOS points, how often would our stated conformal interval have actually
    covered the realized return?"
    """
    from src.models.prequential import month_ordinals, realised_counts

    y_hat_arr = np.asarray(y_hat_oos, dtype=float)
    y_true_arr = np.asarray(y_true_oos, dtype=float)

    if len(y_hat_arr) != len(y_true_arr):
        raise ValueError(
            f"y_hat_oos and y_true_oos must have the same length; got "
            f"{len(y_hat_arr)} and {len(y_true_arr)}."
        )
    if method not in ("split", "aci"):
        raise ValueError(f"method must be 'split' or 'aci'; got '{method}'.")
    if trailing_window <= 0:
        raise ValueError("trailing_window must be positive.")

    if min_calibration is None:
        min_calibration = 4 if method == "aci" else 1
    if min_calibration <= 0:
        raise ValueError("min_calibration must be positive.")

    if dates is None:
        date_values = np.arange(len(y_hat_arr))
    else:
        date_values = np.asarray(pd.Index(dates))
        if len(date_values) != len(y_hat_arr):
            raise ValueError("dates must have the same length as y_hat_oos.")

    valid = np.isfinite(y_hat_arr) & np.isfinite(y_true_arr)
    y_hat_valid = y_hat_arr[valid]
    y_true_valid = y_true_arr[valid]
    dates_valid = date_values[valid]
    months = month_ordinals(pd.Index(dates_valid))
    if len(months) > 1 and np.any(np.diff(months) < 0):
        raise ValueError("dates must be sorted ascending.")
    realised = realised_counts(months, months, horizon_months)

    covered: list[bool] = []
    widths: list[float] = []
    evaluated: list = []
    for idx in range(len(y_hat_valid)):
        n_known = int(realised[idx])
        if n_known < min_calibration:
            continue
        interval = conformal_interval_from_ensemble(
            y_hat_current=float(y_hat_valid[idx]),
            y_hat_oos=y_hat_valid[:n_known],
            y_true_oos=y_true_valid[:n_known],
            coverage=coverage,
            method=method,
            gamma=gamma,
        )
        covered.append(bool(interval.lower <= float(y_true_valid[idx]) <= interval.upper))
        widths.append(float(interval.width))
        evaluated.append(dates_valid[idx])

    if not covered:
        return ConformalCoverageBacktest(
            n_evaluated=0,
            empirical_coverage=float("nan"),
            target_coverage=coverage,
            coverage_gap=float("nan"),
            trailing_n=0,
            trailing_empirical_coverage=float("nan"),
            trailing_coverage_gap=float("nan"),
            method=method,
        )

    empirical = float(np.mean(covered))
    trailing_slice = covered[-trailing_window:]
    trailing_empirical = float(np.mean(trailing_slice))
    if dates is not None:
        evaluated = [pd.Timestamp(value) for value in evaluated]
    return ConformalCoverageBacktest(
        n_evaluated=len(covered),
        empirical_coverage=empirical,
        target_coverage=coverage,
        coverage_gap=empirical - coverage,
        trailing_n=len(trailing_slice),
        trailing_empirical_coverage=trailing_empirical,
        trailing_coverage_gap=trailing_empirical - coverage,
        method=method,
        covered=tuple(covered),
        widths=tuple(widths),
        evaluated_dates=tuple(evaluated),
    )
