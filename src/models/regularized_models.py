"""
L1/L2/ElasticNet/BayesianRidge regularized regression wrappers for the WFO engine.

Rationale for model selection (per CLAUDE.md and Gemini peer review):
  - Dataset has ~180 monthly observations after feature engineering — a
    "Large P, Small N" regime where complex models (XGBoost, deep trees)
    overfit catastrophically.
  - Lasso (L1) performs automatic feature selection by driving noisy
    feature coefficients to exactly zero, addressing multicollinearity.
  - Ridge (L2) retains all features but shrinks coefficients proportionally,
    providing stable predictions when features are correlated.
  - ElasticNet blends L1+L2, handling correlated predictors more gracefully.
  - BayesianRidge provides predictive uncertainty estimates (posterior std),
    which are used by the v3.1 fractional Kelly sizing logic.

All models use alpha tuning via a nested inner TimeSeriesSplit (not KFold)
to prevent any temporal leakage during hyperparameter selection.

IMPORTANT: StandardScaler is applied inside each WFO fold only, never
across the full dataset. This is enforced here by design — the scaler
is part of a Pipeline object that is fit exclusively on X_train.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import BayesianRidge, ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def _make_inner_cv() -> TimeSeriesSplit:
    """Inner cross-validator for alpha tuning — uses TimeSeriesSplit, never KFold."""
    return TimeSeriesSplit(n_splits=3)


def build_lasso_pipeline(alphas: np.ndarray | None = None) -> Pipeline:
    """
    Return a sklearn Pipeline of [StandardScaler → LassoCV].

    The scaler is fit only on the training fold data when pipeline.fit() is
    called, ensuring no data leakage across WFO folds.

    Args:
        alphas: Array of alpha values to search. Defaults to a log-spaced
                grid from 1e-4 to 1e2.

    Returns:
        Unfitted sklearn Pipeline.
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 50)

    lasso = LassoCV(
        alphas=alphas,
        cv=_make_inner_cv(),
        max_iter=10_000,
        fit_intercept=True,
        selection="cyclic",
    )
    return Pipeline([("scaler", StandardScaler()), ("model", lasso)])


def build_ridge_pipeline(alphas: np.ndarray | None = None) -> Pipeline:
    """
    Return a sklearn Pipeline of [StandardScaler → RidgeCV].

    Args:
        alphas: Array of alpha values to search. Defaults to a log-spaced
                grid from 1e-4 to 1e4.

    Returns:
        Unfitted sklearn Pipeline.
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 50)

    ridge = RidgeCV(
        alphas=alphas,
        cv=_make_inner_cv(),
        fit_intercept=True,
    )
    return Pipeline([("scaler", StandardScaler()), ("model", ridge)])


def build_elasticnet_pipeline(
    l1_ratios: list[float] | None = None,
    alphas: np.ndarray | None = None,
    cv_splits: int = 3,
) -> Pipeline:
    """
    Return a sklearn Pipeline of [StandardScaler → ElasticNetCV].

    ElasticNet blends L1 (Lasso) feature selection with L2 (Ridge) coefficient
    shrinkage.  This handles correlated predictors more gracefully than pure
    Lasso and is the recommended default for v3.0+ runs (research report §2).

    The scaler is fit only on the training fold data when pipeline.fit() is
    called, ensuring no data leakage across WFO folds.

    Args:
        l1_ratios: List of L1/L2 mixing ratios to grid search.  A ratio of 1.0
                   recovers pure Lasso; 0.0 recovers pure Ridge.  Defaults to
                   ``[0.1, 0.5, 0.9, 0.95, 1.0]`` per research report guidance.
        alphas:    Array of alpha (regularization strength) values to search.
                   Defaults to 50 log-spaced values from 1e-4 to 1e2.
        cv_splits: Number of inner TimeSeriesSplit folds for hyperparameter
                   selection.  Default: 3 (matches Lasso/Ridge convention).

    Returns:
        Unfitted sklearn Pipeline.
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9, 0.95, 1.0]
    if alphas is None:
        alphas = np.logspace(-4, 2, 50)

    enet = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        max_iter=10_000,
        fit_intercept=True,
        selection="cyclic",
    )
    return Pipeline([("scaler", StandardScaler()), ("model", enet)])


class UncertaintyPipeline(Pipeline):
    """
    Pipeline subclass that exposes ``predict_with_std()`` for models that
    support uncertainty quantification (e.g., BayesianRidge).

    Usage::

        pipe = build_bayesian_ridge_pipeline()
        pipe.fit(X_train, y_train)
        y_pred, y_std = pipe.predict_with_std(X_test)

    The scaler transform is applied before calling the final estimator's
    ``predict(return_std=True)`` method.  For models that do not support
    ``return_std``, ``predict_with_std`` falls back to standard predict with
    ``y_std = np.zeros_like(y_pred)`` (no uncertainty).
    """

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_pred, y_std) after applying all pipeline transforms."""
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)

        final_estimator = self._final_estimator
        if hasattr(final_estimator, "predict") and callable(
            getattr(final_estimator, "predict", None)
        ):
            try:
                y_pred, y_std = final_estimator.predict(Xt, return_std=True)
                return np.asarray(y_pred), np.asarray(y_std)
            except TypeError:
                pass

        y_pred = final_estimator.predict(Xt)
        return np.asarray(y_pred), np.zeros_like(y_pred)


def build_bayesian_ridge_pipeline(
    alpha_1: float = 1e-6,
    alpha_2: float = 1e-6,
    lambda_1: float = 1e-6,
    lambda_2: float = 1e-6,
) -> UncertaintyPipeline:
    """
    Return an UncertaintyPipeline of [StandardScaler → BayesianRidge].

    BayesianRidge places a Gaussian prior over the model weights, providing
    both point predictions and predictive uncertainty (posterior standard
    deviation).  The uncertainty is used by the fractional Kelly sizing
    formula: ``f* = kelly_fraction × signal / prediction_variance``.

    The returned pipeline exposes ``predict_with_std(X)`` which returns
    ``(y_pred, y_std)`` for use in the ensemble sizing logic.

    Args:
        alpha_1, alpha_2: Shape and rate parameters for the Gamma prior over
                          noise precision α.  Defaults follow sklearn convention.
        lambda_1, lambda_2: Shape and rate parameters for the Gamma prior over
                            weight precision λ.  Defaults follow sklearn convention.

    Returns:
        Unfitted UncertaintyPipeline.
    """
    model = BayesianRidge(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        fit_intercept=True,
    )
    return UncertaintyPipeline([("scaler", StandardScaler()), ("model", model)])


def get_feature_importances(pipeline: Pipeline, feature_names: list[str]) -> dict[str, float]:
    """
    Extract feature coefficients from a fitted Lasso or Ridge pipeline.

    Args:
        pipeline:      Fitted sklearn Pipeline with a 'model' step.
        feature_names: List of feature column names in the same order as X.

    Returns:
        Dict mapping feature name to its coefficient magnitude, sorted
        descending by absolute value.
    """
    model = pipeline.named_steps["model"]
    coefs = model.coef_

    result = {name: float(coef) for name, coef in zip(feature_names, coefs)}
    return dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))
