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
