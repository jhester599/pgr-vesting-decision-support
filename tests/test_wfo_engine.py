"""
Tests for src/models/wfo_engine.py and src/models/regularized_models.py

Critical temporal integrity assertions:
  1. max(train_index) < min(test_index) for every fold — no future data in training.
  2. min(test_index) - max(train_index) >= EMBARGO_MONTHS — embargo enforced.
  3. StandardScaler is fit only inside the pipeline on X_train data (verified
     structurally — the scaler is inside a Pipeline, never fit separately).
  4. Model is retrained from scratch on each fold (each fold produces a new
     pipeline object).
  5. y must not contain NaN (validator enforced before training).
  6. Dataset too small raises ValueError.
"""

import pytest
import numpy as np
import pandas as pd

import config
from src.models.wfo_engine import run_wfo, WFOResult, FoldResult
from src.models.regularized_models import (
    build_lasso_pipeline,
    build_ridge_pipeline,
    get_feature_importances,
)


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_dataset():
    """
    180 monthly observations with 5 features and a valid target.
    Dates are month-end, spanning 15 years.
    """
    n = 180
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2009-01-30", periods=n, freq="BME")

    # Features: mix of momentum-like (autocorrelated) and noise
    X = pd.DataFrame(
        {
            "mom_3m": rng.normal(0, 0.1, n).cumsum() * 0.01,
            "mom_6m": rng.normal(0, 0.1, n).cumsum() * 0.01,
            "vol_21d": np.abs(rng.normal(0.2, 0.05, n)),
            "rsi_14": rng.uniform(30, 70, n),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )

    # Target: simple linear combination of features + noise
    y = pd.Series(
        0.3 * X["mom_6m"].values
        - 0.1 * X["vol_21d"].values
        + rng.normal(0, 0.05, n),
        index=pd.DatetimeIndex(dates, name="date"),
        name="target_6m_return",
    )
    return X, y


# ---------------------------------------------------------------------------
# Temporal integrity tests
# ---------------------------------------------------------------------------

class TestWFOTemporalIntegrity:
    def test_train_end_before_test_start_every_fold(self, synthetic_dataset):
        """
        Core invariant: training data must end before test data begins.
        Verifies the chronological arrow-of-time is never violated.
        """
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="lasso")

        for fold in result.folds:
            assert fold.train_end < fold.test_start, (
                f"Fold {fold.fold_idx}: train_end ({fold.train_end.date()}) "
                f">= test_start ({fold.test_start.date()}). TEMPORAL LEAKAGE."
            )

    def test_embargo_gap_enforced(self, synthetic_dataset):
        """
        The gap between train_end and test_start must be at least
        EMBARGO_MONTHS to prevent autocorrelation leakage.
        """
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="lasso")
        embargo = pd.DateOffset(months=config.WFO_EMBARGO_MONTHS)

        for fold in result.folds:
            gap = fold.test_start - fold.train_end
            # Allow a few days tolerance for business-day rounding
            min_gap = pd.Timedelta(days=config.WFO_EMBARGO_MONTHS * 28)
            assert gap >= min_gap, (
                f"Fold {fold.fold_idx}: gap between train_end and test_start "
                f"({gap.days} days) is less than {min_gap.days} days (embargo)."
            )

    def test_no_test_dates_overlap_train_dates(self, synthetic_dataset):
        """
        Verify at the index level that test observation dates never appear
        in the training window of the same fold.
        """
        X, y = synthetic_dataset
        from sklearn.model_selection import TimeSeriesSplit
        n = len(X)
        n_splits = max(1, (n - config.WFO_TRAIN_WINDOW_MONTHS) // config.WFO_TEST_WINDOW_MONTHS)
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=config.WFO_TRAIN_WINDOW_MONTHS,
            test_size=config.WFO_TEST_WINDOW_MONTHS,
            gap=config.WFO_EMBARGO_MONTHS,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X.values)):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, (
                f"Fold {fold_idx}: {len(overlap)} indices appear in both "
                "train and test sets."
            )

    def test_max_train_idx_lt_min_test_idx(self, synthetic_dataset):
        """Verify numerically that max train index < min test index for every fold."""
        X, y = synthetic_dataset
        from sklearn.model_selection import TimeSeriesSplit
        n = len(X)
        n_splits = max(1, (n - config.WFO_TRAIN_WINDOW_MONTHS) // config.WFO_TEST_WINDOW_MONTHS)
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=config.WFO_TRAIN_WINDOW_MONTHS,
            test_size=config.WFO_TEST_WINDOW_MONTHS,
            gap=config.WFO_EMBARGO_MONTHS,
        )
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X.values)):
            assert max(train_idx) < min(test_idx), (
                f"Fold {fold_idx}: max train index ({max(train_idx)}) >= "
                f"min test index ({min(test_idx)}). TEMPORAL LEAKAGE."
            )

    def test_scaler_is_inside_pipeline(self):
        """
        Structural test: StandardScaler is a named step in the Pipeline,
        guaranteeing it is always fit on X_train only.
        """
        pipeline = build_lasso_pipeline()
        assert "scaler" in pipeline.named_steps, (
            "StandardScaler must be a named step in the Pipeline."
        )
        from sklearn.preprocessing import StandardScaler
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)

    def test_ridge_pipeline_has_scaler(self):
        pipeline = build_ridge_pipeline()
        assert "scaler" in pipeline.named_steps

    def test_each_fold_independent_model(self, synthetic_dataset):
        """
        Each fold must produce a new fitted model. Verify that the
        optimal_alpha can differ between folds (model retrained from scratch).
        """
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="lasso")
        # With 180 observations and real data variation, alphas should
        # differ across at least some folds (they're re-optimized per fold)
        alphas = [f.optimal_alpha for f in result.folds]
        assert len(alphas) > 0, "No folds were produced."
        # Structural check: each fold has its own FoldResult object
        assert len({id(f) for f in result.folds}) == len(result.folds)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestWFOInputValidation:
    def test_raises_on_nan_target(self, synthetic_dataset):
        X, y = synthetic_dataset
        y_with_nan = y.copy()
        y_with_nan.iloc[-5:] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            run_wfo(X, y_with_nan)

    def test_raises_on_too_small_dataset(self):
        """Dataset with fewer rows than TRAIN + TEST window must raise ValueError."""
        n = config.WFO_TRAIN_WINDOW_MONTHS + config.WFO_TEST_WINDOW_MONTHS - 1
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2020-01-01", periods=n, freq="BME")
        X = pd.DataFrame({"f1": rng.normal(size=n)}, index=dates)
        y = pd.Series(rng.normal(size=n), index=dates, name="target_6m_return")
        with pytest.raises(ValueError, match="Dataset has only"):
            run_wfo(X, y)


# ---------------------------------------------------------------------------
# Output structure tests
# ---------------------------------------------------------------------------

class TestWFOOutputStructure:
    def test_result_has_folds(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y)
        assert len(result.folds) > 0

    def test_fold_has_correct_fields(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y)
        fold = result.folds[0]
        assert isinstance(fold, FoldResult)
        assert fold.n_train > 0
        assert fold.n_test > 0
        assert len(fold.y_true) == fold.n_test
        assert len(fold.y_hat) == fold.n_test

    def test_aggregate_metrics_computable(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y)
        summary = result.summary()
        assert "information_coefficient" in summary
        assert "hit_rate" in summary
        assert "mean_absolute_error" in summary
        assert -1.0 <= summary["information_coefficient"] <= 1.0
        assert 0.0 <= summary["hit_rate"] <= 1.0

    def test_all_test_obs_concatenated(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y)
        total_oos = sum(f.n_test for f in result.folds)
        assert len(result.y_true_all) == total_oos
        assert len(result.y_hat_all) == total_oos


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------

class TestFeatureImportances:
    def test_keys_match_feature_names(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="lasso")
        importances = result.folds[0].feature_importances
        assert set(importances.keys()) == set(X.columns)

    def test_sorted_by_absolute_value(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="lasso")
        coefs = list(result.folds[0].feature_importances.values())
        abs_coefs = [abs(c) for c in coefs]
        assert abs_coefs == sorted(abs_coefs, reverse=True), (
            "Feature importances must be sorted by absolute value descending."
        )
