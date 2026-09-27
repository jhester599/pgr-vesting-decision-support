"""
Walk-Forward Optimization (WFO) engine.

Implements rolling-window time-series cross-validation using
sklearn.model_selection.TimeSeriesSplit with a defined embargo/purge
period to prevent autocorrelation leakage between adjacent folds.

Configuration (from config.py):
  WFO_TRAIN_WINDOW_MONTHS = 60  (5-year rolling training window)
  WFO_TEST_WINDOW_MONTHS  = 6   (6-month out-of-sample test period)

v2 embargo fix: the gap between train and test must equal the forward return
horizon being predicted.  With a 6-month target, consecutive monthly
observations share 5 months of overlapping return window — a gap of 1
month (the v1 default) does not purge this autocorrelation.

    target horizon = 6M  →  gap = 6 months  (WFO_EMBARGO_MONTHS_6M)
    target horizon = 12M →  gap = 12 months (WFO_EMBARGO_MONTHS_12M)

The legacy ``WFO_EMBARGO_MONTHS = 1`` constant is retained in config.py
only for v1 backward compatibility.

WFO procedure per fold:
  1. Slice X_train, y_train using the rolling window index.
  2. Fit the pipeline (StandardScaler inside, then LassoCV/RidgeCV)
     ONLY on X_train — scaler is never fit on test data.
  3. Generate y_hat on X_test (purely out-of-sample).
  4. Store fold results: test dates, y_true, y_hat, optimal alpha,
     feature coefficients.

Final metrics are computed only from the concatenated out-of-sample folds.
No in-sample data contaminates the performance statistics.

PROHIBITED: K-Fold cross-validation. Validation uses TimeSeriesSplit as
mandated by AGENTS.md. The combinatorial CPCV diagnostic (a K-fold: it trained
on folds after its test folds, review 2026-09-25 F02) is retired; ``run_cpcv``
remains only as a stub that raises ``UnsupportedValidationMethodError``
(pre-v200 remediation R3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Literal, NoReturn
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

import config
from src.models.regularized_models import (
    build_bayesian_ridge_pipeline,
    build_elasticnet_pipeline,
    build_gbt_pipeline,
    build_lasso_pipeline,
    build_ridge_pipeline,
    get_feature_importances,
)
from src.processing.feature_engineering import get_model_feature_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Results for a single WFO fold."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    y_true: np.ndarray
    y_hat: np.ndarray
    optimal_alpha: float
    feature_importances: dict[str, float]
    n_train: int
    n_test: int
    _test_dates: list[pd.Timestamp] = field(default_factory=list)


@dataclass
class WFOResult:
    """Aggregated results across all WFO folds."""
    folds: list[FoldResult] = field(default_factory=list)
    benchmark: str = ""          # ETF ticker this model was trained against (v2)
    target_horizon: int = 6      # Forward return horizon in months (v2)
    model_type: str = "lasso"    # "lasso", "ridge", "elasticnet", or "bayesian_ridge"

    @property
    def y_true_all(self) -> np.ndarray:
        return np.concatenate([f.y_true for f in self.folds])

    @property
    def y_hat_all(self) -> np.ndarray:
        return np.concatenate([f.y_hat for f in self.folds])

    @property
    def test_dates_all(self) -> pd.DatetimeIndex:
        all_dates = []
        for fold in self.folds:
            all_dates.extend(fold._test_dates)
        return pd.DatetimeIndex(all_dates)

    @property
    def information_coefficient(self) -> float:
        """Spearman rank correlation between y_true and y_hat (out-of-sample)."""
        return _safe_spearman_ic(self.y_true_all, self.y_hat_all)

    @property
    def mean_absolute_error(self) -> float:
        return float(mean_absolute_error(self.y_true_all, self.y_hat_all))

    @property
    def hit_rate(self) -> float:
        """Fraction of correct directional predictions."""
        return float(np.mean(np.sign(self.y_true_all) == np.sign(self.y_hat_all)))

    def summary(self) -> dict:
        return {
            "n_folds": len(self.folds),
            "n_total_oos_observations": len(self.y_true_all),
            "information_coefficient": self.information_coefficient,
            "mean_absolute_error": self.mean_absolute_error,
            "hit_rate": self.hit_rate,
            "benchmark": self.benchmark,
            "target_horizon": self.target_horizon,
            "model_type": self.model_type,
        }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def _safe_spearman_ic(
    y_true: np.ndarray | list[float],
    y_hat: np.ndarray | list[float],
) -> float:
    """Return a warning-free Spearman IC for possibly constant arrays.

    scipy.stats.spearmanr emits ConstantInputWarning when either side is constant.
    In this project that condition simply means the fold/path has no rank
    discrimination, so we treat it as IC = 0.0 rather than surfacing log noise.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_hat_arr = np.asarray(y_hat, dtype=float)

    valid = np.isfinite(y_true_arr) & np.isfinite(y_hat_arr)
    if valid.sum() < 2:
        return float("nan")

    y_true_clean = y_true_arr[valid]
    y_hat_clean = y_hat_arr[valid]

    if np.nanstd(y_true_clean) == 0.0 or np.nanstd(y_hat_clean) == 0.0:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        corr, _ = spearmanr(y_true_clean, y_hat_clean)
    return float(corr) if np.isfinite(corr) else 0.0


def _min_required_observations(total_gap: int) -> int:
    """Return the minimum rows for a WFO run: two test windows.

    ``TimeSeriesSplit`` needs ``n_splits >= 2``, so train + gap + one test
    window was not enough: every row count below train + gap + two test
    windows failed with sklearn's "n_splits=2 or more" error (review
    2026-09-25, step 9). Every fold still trains on the full
    ``WFO_TRAIN_WINDOW_MONTHS`` rows: the oldest train set has
    ``train + (available mod test)`` rows before the cap.
    """
    return (
        config.WFO_TRAIN_WINDOW_MONTHS
        + total_gap
        + 2 * config.WFO_TEST_WINDOW_MONTHS
    )

def run_wfo(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: Literal["lasso", "ridge", "elasticnet", "bayesian_ridge", "gbt"] = "elasticnet",
    target_horizon_months: int = 6,
    benchmark: str = "",
    purge_buffer: int | None = None,
    feature_columns: list[str] | None = None,
) -> WFOResult:
    """
    Run Walk-Forward Optimization on the feature matrix.

    The embargo (gap between training end and test start) is:
        gap = target_horizon_months + purge_buffer

    The ``purge_buffer`` provides additional separation beyond the target
    horizon to account for serial autocorrelation in monthly data (v3.0+).
    Default values (from config) are 2 months for 6M targets and 3 months
    for 12M targets, giving total gaps of 8 and 15 months respectively.

    Args:
        X:                     Feature DataFrame (no NaN in price-derived
                               columns; may have NaN in optional fundamentals).
        y:                     Target Series (forward total return or relative
                               return). Must have no NaN — call
                               get_X_y(drop_na_target=True) first.
        model_type:            ``"elasticnet"`` (default v3.0+; L1+L2 blend),
                               ``"lasso"`` (L1 feature selection), or
                               ``"ridge"`` (L2 shrinkage).
        target_horizon_months: Forward return horizon in months.  Contributes
                               to the embargo gap.  Use 6 for 6M models, 12
                               for 12M.  Default: 6.
        benchmark:             ETF ticker this model is trained against
                               (e.g. ``"VTI"``).  Stored in WFOResult for
                               identification; does not affect computation.
        purge_buffer:          Additional months of gap beyond the target
                               horizon.  If None, uses
                               ``config.WFO_PURGE_BUFFER_6M`` (2) for 6M
                               horizons and ``config.WFO_PURGE_BUFFER_12M``
                               (3) for 12M horizons.  Pass ``0`` to reproduce
                               v2.7 behavior (gap = target_horizon only).
        feature_columns:       Optional explicit feature subset for research
                               experiments. When provided, bypasses the
                               model-specific production feature overrides.

    Returns:
        WFOResult containing per-fold FoldResults and aggregate metrics.
        ``result.target_horizon`` and ``result.benchmark`` are set from args.

    Raises:
        ValueError: If y contains NaN values (caller must drop them first).
        ValueError: If the dataset is too small to form even one WFO fold.
    """
    if y.isna().any():
        raise ValueError(
            "y contains NaN values. Call get_X_y(df, drop_na_target=True) first."
        )

    if feature_columns is not None:
        selected_cols = [col for col in feature_columns if col in X.columns]
        if not selected_cols:
            raise ValueError("feature_columns did not match any columns in X.")
    else:
        selected_cols = get_model_feature_columns(X, model_type=model_type)
    X = X[selected_cols].copy()

    # v3.0: resolve purge buffer — default from config based on horizon
    if purge_buffer is None:
        purge_buffer = (
            config.WFO_PURGE_BUFFER_6M
            if target_horizon_months <= 6
            else config.WFO_PURGE_BUFFER_12M
        )
    total_gap = target_horizon_months + purge_buffer

    n = len(X)
    # Subtract the embargo from available rows to avoid requesting more folds
    # than the data can support with the new (larger) gap.
    available = n - config.WFO_TRAIN_WINDOW_MONTHS - total_gap
    n_splits = max(
        1,
        available // config.WFO_TEST_WINDOW_MONTHS,
    )

    min_required = _min_required_observations(total_gap)
    if n < min_required:
        raise ValueError(
            f"Dataset has only {n} observations. Need at least "
            f"{min_required} "
            f"(TRAIN_WINDOW={config.WFO_TRAIN_WINDOW_MONTHS} + "
            f"GAP={total_gap} + 2 x TEST_WINDOW={config.WFO_TEST_WINDOW_MONTHS})."
        )

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=config.WFO_TRAIN_WINDOW_MONTHS,
        test_size=config.WFO_TEST_WINDOW_MONTHS,
        gap=total_gap,  # v3.0: gap = horizon + purge_buffer (default 8M for 6M, 15M for 12M)
    )

    X_arr = X.values
    y_arr = y.values
    feature_names = list(X.columns)
    dates = X.index

    result = WFOResult(
        benchmark=benchmark,
        target_horizon=target_horizon_months,
        model_type=model_type,
    )

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
        X_train, X_test = X_arr[train_idx].copy(), X_arr[test_idx].copy()
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Impute NaN with training-fold column medians (no leakage).
        train_medians = np.nanmedian(X_train, axis=0)
        train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
        for col_i in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, col_i]), col_i] = train_medians[col_i]
            X_test[np.isnan(X_test[:, col_i]), col_i] = train_medians[col_i]

        if model_type == "elasticnet":
            pipeline: Pipeline = build_elasticnet_pipeline(
                target_horizon_months=target_horizon_months,
                purge_buffer=purge_buffer,
            )
        elif model_type == "lasso":
            pipeline = build_lasso_pipeline(
                target_horizon_months=target_horizon_months,
                purge_buffer=purge_buffer,
            )
        elif model_type == "bayesian_ridge":
            pipeline = build_bayesian_ridge_pipeline()
        elif model_type == "gbt":
            pipeline = build_gbt_pipeline()
        else:
            pipeline = build_ridge_pipeline(
                target_horizon_months=target_horizon_months,
                purge_buffer=purge_buffer,
            )
        pipeline.fit(X_train, y_train)
        y_hat = pipeline.predict(X_test)

        model_step = pipeline.named_steps["model"]
        # GBT's .alpha attribute is the Huber loss quantile (default 0.9), not a
        # regularisation strength.  Store 0.0 as a sentinel for non-linear models.
        if model_type == "gbt":
            optimal_alpha = 0.0
        else:
            optimal_alpha = float(
                getattr(model_step, "alpha_", None)
                or getattr(model_step, "alpha", 0.0)
                or 0.0
            )

        importances = get_feature_importances(pipeline, feature_names)

        fold = FoldResult(
            fold_idx=fold_idx,
            train_start=dates[train_idx[0]],
            train_end=dates[train_idx[-1]],
            test_start=dates[test_idx[0]],
            test_end=dates[test_idx[-1]],
            y_true=y_test.copy(),
            y_hat=y_hat.copy(),
            optimal_alpha=optimal_alpha,
            feature_importances=importances,
            n_train=len(train_idx),
            n_test=len(test_idx),
        )
        fold._test_dates = list(dates[test_idx])
        result.folds.append(fold)

    return result


def predict_current(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_current: pd.DataFrame,
    wfo_result: WFOResult,
    model_type: Literal["lasso", "ridge", "elasticnet", "bayesian_ridge", "gbt"] = "elasticnet",
    train_window_months: int | None = None,
) -> dict:
    """
    Generate a live prediction for the current observation by refitting on
    the most recent ``train_window_months`` of data.

    Unlike the v1 placeholder (which returned the last fold's y_hat mean),
    this function actually trains a fresh model on the most recent window
    and predicts on ``X_current``.  The WFO IC and hit_rate from the
    completed out-of-sample folds are retained as confidence metrics.

    Args:
        X_full:              Complete feature DataFrame (same X used in run_wfo).
        y_full:              Complete target Series (same y used in run_wfo,
                             including NaN rows which are dropped here).
        X_current:           Single-row DataFrame with current features.
        wfo_result:          Completed WFOResult from run_wfo() — provides IC,
                             hit_rate, benchmark, and target_horizon metadata.
        model_type:          Must match the model_type used in run_wfo().
        train_window_months: Number of most-recent months to refit on.
                             Defaults to ``config.WFO_TRAIN_WINDOW_MONTHS`` (60).

    Returns:
        Dict with keys:
          - ``predicted_return`` (float): fresh model prediction on X_current
          - ``prediction_std`` (float): posterior std (BayesianRidge only; 0 otherwise)
          - ``ic`` (float): out-of-sample IC from wfo_result
          - ``hit_rate`` (float): directional hit rate from wfo_result
          - ``benchmark`` (str): ETF ticker (from wfo_result)
          - ``target_horizon`` (int): forward months (from wfo_result)
          - ``top_features`` (list): top 5 (name, coef) from refitted model
    """
    if train_window_months is None:
        train_window_months = config.WFO_TRAIN_WINDOW_MONTHS

    selected_cols = get_model_feature_columns(X_full, model_type=model_type)
    X_full = X_full[selected_cols].copy()
    X_current = X_current[selected_cols].copy()

    # Align X and y; drop NaN targets (same as get_X_y with drop_na_target=True)
    aligned = X_full.join(y_full, how="inner")
    aligned = aligned.dropna(subset=[y_full.name])
    recent = aligned.iloc[-train_window_months:]

    feature_cols = list(X_full.columns)
    X_recent = recent[feature_cols].values.copy()
    y_recent = recent[y_full.name].values

    # Impute NaN with training medians (same as per-fold logic in run_wfo)
    train_medians = np.nanmedian(X_recent, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
    for col_i in range(X_recent.shape[1]):
        X_recent[np.isnan(X_recent[:, col_i]), col_i] = train_medians[col_i]

    X_curr_arr = X_current[feature_cols].values.copy()
    for col_i in range(X_curr_arr.shape[1]):
        X_curr_arr[np.isnan(X_curr_arr[:, col_i]), col_i] = train_medians[col_i]

    prediction_std = 0.0
    if model_type == "elasticnet":
        pipeline: Pipeline = build_elasticnet_pipeline(
            target_horizon_months=wfo_result.target_horizon,
        )
    elif model_type == "lasso":
        pipeline = build_lasso_pipeline(
            target_horizon_months=wfo_result.target_horizon,
        )
    elif model_type == "bayesian_ridge":
        pipeline = build_bayesian_ridge_pipeline()
    elif model_type == "gbt":
        pipeline = build_gbt_pipeline()
    else:
        pipeline = build_ridge_pipeline(
            target_horizon_months=wfo_result.target_horizon,
        )
    pipeline.fit(X_recent, y_recent)

    if model_type == "bayesian_ridge" and hasattr(pipeline, "predict_with_std"):
        y_pred_arr, y_std_arr = pipeline.predict_with_std(X_curr_arr)
        predicted = float(y_pred_arr[0])
        prediction_std = float(y_std_arr[0])
    else:
        predicted = float(pipeline.predict(X_curr_arr)[0])

    importances = get_feature_importances(pipeline, feature_cols)

    return {
        "predicted_return":  predicted,
        "prediction_std":    prediction_std,
        "ic":                wfo_result.information_coefficient,
        "hit_rate":          wfo_result.hit_rate,
        "benchmark":         wfo_result.benchmark,
        "target_horizon":    wfo_result.target_horizon,
        "top_features":      list(importances.items())[:5],
    }


# ---------------------------------------------------------------------------
# Retired: Combinatorial Purged Cross-Validation (CPCV)
# ---------------------------------------------------------------------------

class UnsupportedValidationMethodError(ValueError):
    """Raised when a caller asks for a validation method AGENTS.md prohibits."""


def run_cpcv(*args: object, **kwargs: object) -> NoReturn:
    """Retired entry point; always raises ``UnsupportedValidationMethodError``.

    CPCV trains on folds that come after its test folds, i.e. it is a
    combinatorial K-fold, which AGENTS.md prohibits. It was a monthly
    stability diagnostic until the pre-v200 remediation R3 (2026-09-27)
    removed it from active execution; the decision now uses walk-forward
    validation only (``run_wfo``) with an explicit ``wfo_completed`` gate.
    This stub keeps old imports failing loudly instead of running K-fold.
    """
    del args, kwargs
    raise UnsupportedValidationMethodError(
        "CPCV (combinatorial purged K-fold) is retired: AGENTS.md prohibits "
        "K-fold validation. Use walk-forward validation (run_wfo)."
    )
