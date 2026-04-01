"""
Beta-Transformed Linear Pool (BLP) aggregation for ensemble probability forecasts.

Replaces naive equal-weight ensemble averaging with a calibrated combination.

Motivation (Ranjan & Gneiting 2010):
    Any linear pool of individually calibrated forecasts is necessarily
    *uncalibrated* as a combined forecast — the pooled distribution is too
    wide relative to the ideal.  The Beta-Transformed Linear Pool corrects
    this by passing the linear combination through a Beta CDF, which can
    sharpen or widen the pooled forecast to achieve calibration.

Model definition:
    Given K probability forecasts p_1, ..., p_K and weights w_1, ..., w_K:

        p_bar   = Σ w_k * p_k          (linear pool)
        BLP(p)  = F_Beta(p_bar; a, b)  (Beta CDF transformation)

    Parameters (5 total for K=4 models):
        a, b          — Beta distribution shape parameters
        w_1, w_2, w_3 — Independent weights; w_4 = 1 − w_1 − w_2 − w_3
                         (weights are constrained to [0, 1] and sum to 1)

    The likelihood is:
        log L = Σ_t log f_Beta(p_bar_t; a, b)  (Beta PDF evaluated at p_bar_t)

    A neutral starting point is a = b = 1, equal weights (Beta(1,1) = uniform),
    which is equivalent to unweighted linear pooling — fitting starts here and
    gradient-based optimisation (L-BFGS-B) searches for the MLE.

Data requirements:
    BLP parameter fitting requires at least ``config.BLP_MIN_OOS_MONTHS`` months
    of live out-of-sample predictions to avoid overfitting to a short history.
    ``BLPModel.fit()`` raises ``InsufficientDataError`` when this threshold is
    not met.  Target activation: 2026-05-20 (12 months after first live run).

Usage::

    from src.models.blp import BLPModel

    model = BLPModel(n_models=4)
    # oos_probs: shape (T, 4) — one column per ensemble model, one row per period
    # oos_outcomes: shape (T,) — binary (1 = outperform, 0 = underperform)
    model.fit(oos_probs, oos_outcomes)
    blp_probs = model.transform(live_probs)   # shape (T,)

References:
    Ranjan, R. & Gneiting, T. (2010). Combining probability forecasts.
    Journal of the Royal Statistical Society: Series B, 72(1), 71–91.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta as beta_dist

import config


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InsufficientDataError(ValueError):
    """Raised when fewer than ``BLP_MIN_OOS_MONTHS`` observations are provided.

    BLP parameter fitting via MLE is unreliable on short histories.  This
    exception signals that the caller should fall back to the existing
    inverse-variance weighted ensemble until sufficient OOS data accumulates.
    """


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class BLPParams:
    """Fitted BLP parameters.

    Attributes:
        a:       Beta distribution shape parameter a  (> 0).
        b:       Beta distribution shape parameter b  (> 0).
        weights: Linear pool weights, length == n_models, sums to 1.0.
        log_likelihood: MLE objective value at the fitted params (lower is
                        better; stored as the negative log-likelihood returned
                        by scipy.optimize.minimize).
        converged: True if the optimizer reported successful convergence.
    """

    a: float
    b: float
    weights: list[float]
    log_likelihood: float = 0.0
    converged: bool = False

    @property
    def n_models(self) -> int:
        """Number of ensemble models in the pool."""
        return len(self.weights)


# ---------------------------------------------------------------------------
# BLP model
# ---------------------------------------------------------------------------

class BLPModel:
    """Beta-Transformed Linear Pool for combining K ensemble probability forecasts.

    Args:
        n_models: Number of ensemble members (default 4: elasticnet, ridge,
                  bayesian_ridge, gbt — matching ``config.ENSEMBLE_MODELS``).
        min_obs:  Minimum OOS observations required before fitting.
                  Defaults to ``config.BLP_MIN_OOS_MONTHS``.

    Attributes:
        params_: Fitted :class:`BLPParams`.  ``None`` until :meth:`fit` succeeds.
    """

    def __init__(
        self,
        n_models: int = 4,
        min_obs: int | None = None,
    ) -> None:
        self.n_models = n_models
        self.min_obs: int = (
            min_obs if min_obs is not None else config.BLP_MIN_OOS_MONTHS
        )
        self.params_: BLPParams | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        oos_probs: np.ndarray,
        oos_outcomes: np.ndarray,
    ) -> "BLPModel":
        """Fit BLP parameters by maximum likelihood on OOS probability sequences.

        Args:
            oos_probs:    Shape ``(T, n_models)``.  Each column is the OOS
                          calibrated probability sequence for one ensemble model.
                          Values must be strictly in (0, 1) — probabilities of
                          PGR outperforming a given benchmark.
            oos_outcomes: Shape ``(T,)``.  Binary: 1 = PGR outperformed, 0 = not.
                          Used as the observed event for log-likelihood evaluation.

        Returns:
            ``self`` (for method chaining).

        Raises:
            InsufficientDataError: If ``T < self.min_obs``.
            ValueError: If shapes are inconsistent or weights become infeasible.
        """
        oos_probs = np.asarray(oos_probs, dtype=float)
        oos_outcomes = np.asarray(oos_outcomes, dtype=float)

        if oos_probs.ndim == 1:
            oos_probs = oos_probs.reshape(-1, 1)

        T, K = oos_probs.shape
        if K != self.n_models:
            raise ValueError(
                f"oos_probs has {K} columns but n_models={self.n_models}."
            )
        if T < self.min_obs:
            raise InsufficientDataError(
                f"BLP fitting requires at least {self.min_obs} OOS observations; "
                f"only {T} provided.  Wait until {self.min_obs - T} more monthly "
                f"predictions accumulate before activating BLP aggregation."
            )

        # Clip probabilities away from 0/1 to keep log-likelihood finite.
        oos_probs = np.clip(oos_probs, 1e-6, 1.0 - 1e-6)

        # Parameterisation: x = [log(a), log(b), logit(w_1), ..., logit(w_{K-1})]
        # This ensures a > 0, b > 0 and weights in (0, 1) without explicit bounds.
        def _unpack(x: np.ndarray) -> tuple[float, float, np.ndarray]:
            a_val = float(np.exp(x[0]))
            b_val = float(np.exp(x[1]))
            # x[2:] has K-1 free parameters; append 0 as the reference logit
            # for the K-th model so that softmax produces K weights summing to 1.
            raw_w = np.concatenate([x[2:], [0.0]])
            w = np.exp(raw_w - raw_w.max())
            w /= w.sum()
            return a_val, b_val, w

        def _neg_log_likelihood(x: np.ndarray) -> float:
            a_val, b_val, w = _unpack(x)
            p_bar = oos_probs @ w                      # shape (T,)
            p_bar = np.clip(p_bar, 1e-6, 1.0 - 1e-6)
            # Log-likelihood under Beta(a, b) evaluated at p_bar
            # (treating p_bar as a Beta-distributed random variable)
            ll = (a_val - 1.0) * np.log(p_bar) + (b_val - 1.0) * np.log(1.0 - p_bar)
            ll -= betaln(a_val, b_val)
            return -ll.sum()

        # Initial parameters: a=b=1 (uniform), equal weights
        x0_weights = np.zeros(K - 1)  # softmax of zeros → equal weights
        x0 = np.concatenate([
            [np.log(config.BLP_BETA_A_INIT), np.log(config.BLP_BETA_B_INIT)],
            x0_weights,
        ])

        result = minimize(
            _neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-8},
        )

        a_fit, b_fit, w_fit = _unpack(result.x)
        self.params_ = BLPParams(
            a=a_fit,
            b=b_fit,
            weights=w_fit.tolist(),
            log_likelihood=float(result.fun),
            converged=bool(result.success),
        )
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply the fitted BLP transform to a new set of probability forecasts.

        Args:
            probs: Shape ``(n_models,)`` or ``(T, n_models)``.  Each column is
                   the current-period probability for one ensemble model.

        Returns:
            BLP-aggregated probabilities.  Shape ``(T,)`` or scalar depending
            on input shape.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.params_ is None:
            raise RuntimeError(
                "BLPModel.transform() called before fit().  "
                "Call fit(oos_probs, oos_outcomes) first, or check is_fitted()."
            )
        probs = np.asarray(probs, dtype=float)
        squeeze = probs.ndim == 1
        if squeeze:
            probs = probs.reshape(1, -1)

        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        w = np.asarray(self.params_.weights)
        p_bar = probs @ w                              # shape (T,)
        p_bar = np.clip(p_bar, 1e-6, 1.0 - 1e-6)
        blp_probs = beta_dist.cdf(p_bar, self.params_.a, self.params_.b)

        return float(blp_probs[0]) if squeeze else blp_probs

    def is_fitted(self) -> bool:
        """Return True if the model has been successfully fitted."""
        return self.params_ is not None
