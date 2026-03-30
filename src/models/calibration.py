"""
v5.1 — Probability Calibration for PGR Ensemble Predictions.

Implements Platt scaling (logistic regression) and isotonic regression to
convert raw ensemble regression scores into well-calibrated P(outperform)
probabilities.

The raw ``prob_outperform`` values produced by the ensemble are derived from
BayesianRidge posterior standard deviations via ``norm.cdf(y_hat / y_std)``.
These are Phase 1 / uncalibrated estimates.  Phase 2 (this module) applies
an expanding-window calibration model trained on all available historical
OOS fold predictions.

Calibration method transitions:
  - n < CALIBRATION_MIN_OBS_PLATT  (< 20):  uncalibrated (return 0.5)
  - n ≥ CALIBRATION_MIN_OBS_PLATT  (≥ 20):  Platt scaling
  - n ≥ CALIBRATION_MIN_OBS_ISOTONIC (≥ 60): Platt → Isotonic (two-stage)

The Expected Calibration Error (ECE) measures reliability:
  ECE = Σ_b (|B_b| / n) × |accuracy_b − confidence_b|
A 95% CI is produced via block bootstrap with block_len = prediction horizon,
preserving autocorrelation structure from the overlapping return windows.

References:
  Platt (1999): "Probabilistic Outputs for Support Vector Machines"
  Zadrozny & Elkan (2002): "Transforming Classifier Scores into Accurate
    Multiclass Probability Estimates"
  Niculescu-Mizil & Caruana (2005): "Predicting Good Probabilities With
    Supervised Learning"
  Politis & Romano (1994): "The Stationary Bootstrap" (block bootstrap CI)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Summary of a fitted calibration model evaluated on the training data.

    Attributes:
        n_obs:          Number of OOS observations used to fit the model.
        method:         ``"uncalibrated"``, ``"platt"``, or ``"isotonic"``.
        ece:            Expected Calibration Error on the training sample.
        ece_ci_lower:   Lower bound of the 95% block-bootstrap CI for ECE.
        ece_ci_upper:   Upper bound of the 95% block-bootstrap CI for ECE.
    """
    n_obs: int
    method: str
    ece: float
    ece_ci_lower: float
    ece_ci_upper: float


# ---------------------------------------------------------------------------
# ECE helpers
# ---------------------------------------------------------------------------

def compute_ece(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute the Expected Calibration Error (ECE).

    Bins probabilities into ``n_bins`` equal-width buckets and measures the
    weighted average gap between mean predicted probability and empirical
    positive frequency.

    Args:
        probs:    Predicted probabilities ∈ [0, 1], shape (n,).
        outcomes: Binary outcomes (1 = outperformed, 0 = did not), shape (n,).
        n_bins:   Number of equal-width bins.

    Returns:
        ECE ∈ [0, 1].  Lower is better (0 = perfect calibration).
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(probs)
    if n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Include the right edge in the final bin to capture prob == 1.0
        if hi == edges[-1]:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        mean_prob = float(probs[mask].mean())
        mean_out = float(outcomes[mask].mean())
        ece += (n_bin / n) * abs(mean_prob - mean_out)
    return float(ece)


def block_bootstrap_ece_ci(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    block_len: int = 6,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Compute a 95% block-bootstrap confidence interval for ECE.

    Uses the circular block bootstrap (Politis & Romano 1994) with
    ``block_len = prediction horizon`` to preserve autocorrelation from
    overlapping return windows.

    Args:
        probs:       Predicted probabilities, shape (n,).
        outcomes:    Binary outcomes, shape (n,).
        n_bins:      Number of ECE bins.
        block_len:   Length of each bootstrap block (= target horizon in months).
        n_bootstrap: Number of bootstrap replications.
        rng:         Optional numpy Generator for reproducibility.

    Returns:
        ``(ci_lower, ci_upper)`` — 2.5th and 97.5th percentiles of bootstrap ECE.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(probs)

    if n < block_len * 2:
        # Insufficient data for meaningful blocking — return degenerate CI
        ece = compute_ece(probs, outcomes, n_bins)
        return 0.0, min(1.0, ece * 2)

    n_blocks_needed = max(1, n // block_len)
    # Circular block bootstrap: wrap-around indices keep all positions equally probable
    boot_eces: list[float] = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks_needed)
        idx_list: list[np.ndarray] = []
        for s in starts:
            end = s + block_len
            if end <= n:
                idx_list.append(np.arange(s, end))
            else:
                # Wrap around
                idx_list.append(np.concatenate([np.arange(s, n), np.arange(0, end - n)]))
        all_idx = np.concatenate(idx_list)[:n]
        boot_eces.append(compute_ece(probs[all_idx], outcomes[all_idx], n_bins))

    arr = np.array(boot_eces)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_calibration_model(
    y_hat_hist: np.ndarray,
    outcomes: np.ndarray,
    min_obs_platt: int = 20,
    min_obs_isotonic: int = 60,
    n_bins: int = 10,
    block_len: int = 6,
    n_bootstrap: int = 500,
) -> tuple[object | None, CalibrationResult]:
    """
    Fit a calibration model on historical OOS ensemble predictions.

    The model transitions through three phases as more OOS data accumulates:

      n < min_obs_platt   → uncalibrated (no model fitted)
      n ≥ min_obs_platt   → Platt scaling (logistic regression on y_hat)
      n ≥ min_obs_isotonic → Platt → Isotonic (two-stage non-parametric)

    The returned ``fitted_model`` is passed directly to
    ``calibrate_prediction()``.  It can be:
      - ``None``                          (uncalibrated)
      - ``Pipeline``                      (Platt only)
      - ``(Pipeline, IsotonicRegression)`` (Platt + Isotonic)

    Args:
        y_hat_hist:        Historical OOS ensemble regression scores, shape (n,).
        outcomes:          Binary outcomes: 1 if PGR outperformed the benchmark,
                           0 otherwise.  Shape (n,).
        min_obs_platt:     Minimum observations to activate Platt scaling.
        min_obs_isotonic:  Minimum observations to switch to isotonic.
        n_bins:            Number of ECE bins.
        block_len:         Bootstrap block length (= prediction horizon).
        n_bootstrap:       Bootstrap replications for ECE CI.

    Returns:
        ``(fitted_model, CalibrationResult)``.
    """
    y_hat_hist = np.asarray(y_hat_hist, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n = len(y_hat_hist)

    # Guard: need at least 2 classes to fit a classifier
    if n < min_obs_platt or len(np.unique(outcomes)) < 2:
        ece = compute_ece(np.full(n, 0.5), outcomes, n_bins) if n >= 2 else 0.0
        return None, CalibrationResult(
            n_obs=n,
            method="uncalibrated",
            ece=ece,
            ece_ci_lower=0.0,
            ece_ci_upper=1.0,
        )

    # --- Platt scaling: logistic regression with no regularization ---
    # StandardScaler normalises y_hat across benchmarks (they have different scales).
    # C=1e10 makes the logistic effectively unregularised — standard Platt scaling.
    platt = Pipeline([
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                C=1e10,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])
    platt.fit(y_hat_hist.reshape(-1, 1), outcomes)

    if n >= min_obs_isotonic:
        # --- Two-stage: Platt then Isotonic ---
        # Platt probabilities become the input to isotonic regression, which
        # enforces monotonicity and fits the actual reliability curve.
        platt_probs = platt.predict_proba(y_hat_hist.reshape(-1, 1))[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(platt_probs, outcomes)

        fitted_probs = np.clip(iso.predict(platt_probs), 0.0, 1.0)
        ece = compute_ece(fitted_probs, outcomes, n_bins)
        ci_lo, ci_hi = block_bootstrap_ece_ci(
            fitted_probs, outcomes, n_bins, block_len, n_bootstrap
        )
        return (platt, iso), CalibrationResult(
            n_obs=n,
            method="isotonic",
            ece=ece,
            ece_ci_lower=ci_lo,
            ece_ci_upper=ci_hi,
        )

    # --- Platt-only ---
    fitted_probs = platt.predict_proba(y_hat_hist.reshape(-1, 1))[:, 1]
    ece = compute_ece(fitted_probs, outcomes, n_bins)
    ci_lo, ci_hi = block_bootstrap_ece_ci(
        fitted_probs, outcomes, n_bins, block_len, n_bootstrap
    )
    return platt, CalibrationResult(
        n_obs=n,
        method="platt",
        ece=ece,
        ece_ci_lower=ci_lo,
        ece_ci_upper=ci_hi,
    )


# ---------------------------------------------------------------------------
# Live prediction calibration
# ---------------------------------------------------------------------------

def calibrate_prediction(
    fitted_model: object | None,
    y_hat_current: float,
) -> float:
    """
    Apply a fitted calibration model to the current ensemble prediction.

    Args:
        fitted_model:   Output of ``fit_calibration_model()``.  Can be:
                        ``None``, a Platt ``Pipeline``, or a
                        ``(Pipeline, IsotonicRegression)`` tuple.
        y_hat_current:  Current inverse-variance weighted ensemble prediction
                        (``predicted_relative_return``).

    Returns:
        Calibrated P(outperform) ∈ [0, 1].  Returns 0.5 when the model is
        ``None`` (insufficient data for calibration).
    """
    if fitted_model is None:
        return 0.5

    x = np.array([[float(y_hat_current)]])

    if isinstance(fitted_model, tuple):
        platt, iso = fitted_model
        platt_prob = float(platt.predict_proba(x)[0, 1])
        calibrated = float(iso.predict([platt_prob])[0])
    else:
        calibrated = float(fitted_model.predict_proba(x)[0, 1])

    return float(np.clip(calibrated, 0.0, 1.0))
