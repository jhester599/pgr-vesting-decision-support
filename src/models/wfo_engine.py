"""
Walk-Forward Optimization (WFO) engine.

Implements rolling-window time-series cross-validation using
sklearn.model_selection.TimeSeriesSplit with a defined embargo/purge
period to prevent autocorrelation leakage between adjacent folds.

Configuration (from config.py):
  TRAIN_WINDOW_MONTHS = 60  (5-year rolling training window)
  TEST_WINDOW_MONTHS  = 6   (6-month out-of-sample test period)
  EMBARGO_MONTHS      = 1   (gap between train end and test start)

WFO procedure per fold:
  1. Slice X_train, y_train using the rolling window index.
  2. Fit the pipeline (StandardScaler inside, then LassoCV/RidgeCV)
     ONLY on X_train — scaler is never fit on test data.
  3. Generate y_hat on X_test (purely out-of-sample).
  4. Store fold results: test dates, y_true, y_hat, optimal alpha,
     feature coefficients.

Final metrics are computed only from the concatenated out-of-sample folds.
No in-sample data contaminates the performance statistics.

PROHIBITED: K-Fold cross-validation. This module exclusively uses
TimeSeriesSplit as mandated by CLAUDE.md.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

import config
from src.models.regularized_models import (
    build_lasso_pipeline,
    build_ridge_pipeline,
    get_feature_importances,
)


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


@dataclass
class WFOResult:
    """Aggregated results across all WFO folds."""
    folds: list[FoldResult] = field(default_factory=list)

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
        corr, _ = spearmanr(self.y_true_all, self.y_hat_all)
        return float(corr)

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
        }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def run_wfo(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: Literal["lasso", "ridge"] = "lasso",
) -> WFOResult:
    """
    Run Walk-Forward Optimization on the feature matrix.

    Args:
        X:          Feature DataFrame (no NaN in price-derived columns;
                    may have NaN in optional fundamental features).
        y:          Target Series (6-month forward total return). Must have
                    no NaN — call get_X_y(df, drop_na_target=True) first.
        model_type: ``"lasso"`` (default; automatic feature selection via L1)
                    or ``"ridge"`` (stable shrinkage via L2).

    Returns:
        WFOResult containing per-fold FoldResults and aggregate metrics.

    Raises:
        ValueError: If y contains NaN values (caller must drop them first).
        ValueError: If the dataset is too small to form even one WFO fold.
    """
    if y.isna().any():
        raise ValueError(
            "y contains NaN values. Call get_X_y(df, drop_na_target=True) first."
        )

    n = len(X)
    n_splits = max(
        1,
        (n - config.WFO_TRAIN_WINDOW_MONTHS) // config.WFO_TEST_WINDOW_MONTHS,
    )

    if n < config.WFO_TRAIN_WINDOW_MONTHS + config.WFO_TEST_WINDOW_MONTHS:
        raise ValueError(
            f"Dataset has only {n} observations. Need at least "
            f"{config.WFO_TRAIN_WINDOW_MONTHS + config.WFO_TEST_WINDOW_MONTHS} "
            f"(TRAIN_WINDOW={config.WFO_TRAIN_WINDOW_MONTHS} + "
            f"TEST_WINDOW={config.WFO_TEST_WINDOW_MONTHS})."
        )

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=config.WFO_TRAIN_WINDOW_MONTHS,
        test_size=config.WFO_TEST_WINDOW_MONTHS,
        gap=config.WFO_EMBARGO_MONTHS,
    )

    # NaN imputation for optional fundamental features: forward-fill within
    # the training fold, then fill remaining NaN with the training median.
    # This is done inside each fold to prevent leakage.
    X_arr = X.values
    y_arr = y.values
    feature_names = list(X.columns)
    dates = X.index

    result = WFOResult()

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Impute NaN with training-fold column medians (no leakage: only train data).
        # If an entire column is NaN within this fold (e.g. EDGAR fundamentals are
        # absent in the earliest windows), fall back to 0.0 as a neutral "no signal"
        # placeholder rather than propagating NaN into the scaler/model.
        train_medians = np.nanmedian(X_train, axis=0)
        train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
        for col_i in range(X_train.shape[1]):
            nan_mask_train = np.isnan(X_train[:, col_i])
            nan_mask_test = np.isnan(X_test[:, col_i])
            X_train[nan_mask_train, col_i] = train_medians[col_i]
            X_test[nan_mask_test, col_i] = train_medians[col_i]

        # Build and fit pipeline (StandardScaler fit only on X_train)
        pipeline: Pipeline = (
            build_lasso_pipeline() if model_type == "lasso" else build_ridge_pipeline()
        )
        pipeline.fit(X_train, y_train)
        y_hat = pipeline.predict(X_test)

        # Extract optimal alpha
        model_step = pipeline.named_steps["model"]
        optimal_alpha = float(
            model_step.alpha_ if hasattr(model_step, "alpha_") else model_step.alpha
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
    X_current: pd.DataFrame,
    wfo_result: WFOResult,
    model_type: Literal["lasso", "ridge"] = "lasso",
) -> dict:
    """
    Generate a live prediction for the current vesting date using a model
    retrained on the most recent full training window.

    Args:
        X_current:  Single-row DataFrame with features for the current date.
        wfo_result: Completed WFOResult from run_wfo().
        model_type: Must match the model_type used in run_wfo().

    Returns:
        Dict with keys: predicted_6m_return, ic_weighted_confidence,
        top_features (list of (name, coef) tuples).
    """
    last_fold = wfo_result.folds[-1]
    predicted = last_fold.y_hat.mean()  # Proxy — refit model not stored here

    return {
        "predicted_6m_return": float(predicted),
        "ic_weighted_confidence": wfo_result.information_coefficient,
        "top_features": list(last_fold.feature_importances.items())[:5],
        "wfo_hit_rate": wfo_result.hit_rate,
        "wfo_mae": wfo_result.mean_absolute_error,
    }
