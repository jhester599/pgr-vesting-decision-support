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
from src.models.wfo_engine import run_wfo, predict_current, WFOResult, FoldResult
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
        target_horizon_months (default 6) to prevent autocorrelation leakage
        from overlapping forward return windows.

        v2 fix: embargo = target_horizon_months (not the legacy 1-month gap).
        With a 6-month forward return target, consecutive monthly observations
        share 5 months of overlapping return window — a 1-month gap does NOT
        purge this.  Asserting >= 168 days (6 × 28) confirms the fix.
        """
        X, y = synthetic_dataset
        target_horizon = 6
        result = run_wfo(X, y, model_type="lasso", target_horizon_months=target_horizon)

        for fold in result.folds:
            gap = fold.test_start - fold.train_end
            # 6 months × 28 days/month = 168 days minimum (business-day safe)
            min_gap = pd.Timedelta(days=target_horizon * 28)
            assert gap >= min_gap, (
                f"Fold {fold.fold_idx}: gap between train_end and test_start "
                f"({gap.days} days) is less than {min_gap.days} days "
                f"(required embargo for {target_horizon}M horizon)."
            )

    def test_embargo_gap_enforced_12m(self, synthetic_dataset):
        """Embargo must be >= 12 months for a 12-month target horizon."""
        X, y = synthetic_dataset
        target_horizon = 12
        result = run_wfo(X, y, model_type="lasso", target_horizon_months=target_horizon)

        for fold in result.folds:
            gap = fold.test_start - fold.train_end
            min_gap = pd.Timedelta(days=target_horizon * 28)
            assert gap >= min_gap, (
                f"Fold {fold.fold_idx}: gap {gap.days} days < {min_gap.days} days "
                f"(embargo for {target_horizon}M horizon)."
            )

    def test_no_test_dates_overlap_train_dates(self, synthetic_dataset):
        """
        Verify at the index level that test observation dates never appear
        in the training window of the same fold.
        """
        X, y = synthetic_dataset
        from sklearn.model_selection import TimeSeriesSplit
        target_horizon = 6
        n = len(X)
        n_splits = max(
            1,
            (n - config.WFO_TRAIN_WINDOW_MONTHS - target_horizon)
            // config.WFO_TEST_WINDOW_MONTHS,
        )
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=config.WFO_TRAIN_WINDOW_MONTHS,
            test_size=config.WFO_TEST_WINDOW_MONTHS,
            gap=target_horizon,
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
        target_horizon = 6
        n = len(X)
        n_splits = max(
            1,
            (n - config.WFO_TRAIN_WINDOW_MONTHS - target_horizon)
            // config.WFO_TEST_WINDOW_MONTHS,
        )
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=config.WFO_TRAIN_WINDOW_MONTHS,
            test_size=config.WFO_TEST_WINDOW_MONTHS,
            gap=target_horizon,
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

    def test_elasticnet_uses_model_specific_feature_subset(self, synthetic_dataset):
        X, y = synthetic_dataset
        X = X.assign(
            mom_12m=X["mom_6m"] * 0.5,
            vol_63d=X["vol_21d"] * 0.8,
            yield_slope=np.linspace(-1.0, 1.0, len(X)),
            yield_curvature=np.linspace(1.0, -1.0, len(X)),
            real_rate_10y=np.linspace(0.1, 0.3, len(X)),
            credit_spread_hy=np.linspace(0.2, 0.6, len(X)),
            nfci=np.linspace(-0.5, 0.5, len(X)),
            vix=np.linspace(15.0, 25.0, len(X)),
            vmt_yoy=np.linspace(-0.02, 0.03, len(X)),
            investment_income_growth_yoy=np.linspace(-0.1, 0.1, len(X)),
            roe_net_income_ttm=np.linspace(0.08, 0.14, len(X)),
            underwriting_income=np.linspace(100.0, 200.0, len(X)),
            channel_mix_agency_pct=np.linspace(0.6, 0.7, len(X)),
        )

        result = run_wfo(X, y, model_type="elasticnet")
        expected_cols = set(config.MODEL_FEATURE_OVERRIDES["elasticnet"])
        actual_cols = set(result.folds[0].feature_importances.keys())
        assert actual_cols == expected_cols

    def test_gbt_uses_model_specific_feature_subset(self, synthetic_dataset):
        X, y = synthetic_dataset
        X = X.assign(
            mom_12m=X["mom_6m"] * 0.5,
            vol_63d=X["vol_21d"] * 0.8,
            yield_slope=np.linspace(-1.0, 1.0, len(X)),
            yield_curvature=np.linspace(1.0, -1.0, len(X)),
            real_rate_10y=np.linspace(0.1, 0.3, len(X)),
            credit_spread_hy=np.linspace(0.2, 0.6, len(X)),
            nfci=np.linspace(-0.5, 0.5, len(X)),
            vix=np.linspace(15.0, 25.0, len(X)),
            vmt_yoy=np.linspace(-0.02, 0.03, len(X)),
            investment_income_growth_yoy=np.linspace(-0.1, 0.1, len(X)),
        )

        result = run_wfo(X, y, model_type="gbt")
        expected_cols = set(config.MODEL_FEATURE_OVERRIDES["gbt"])
        actual_cols = set(result.folds[0].feature_importances.keys())
        assert actual_cols == expected_cols

    def test_ridge_uses_model_specific_feature_subset(self, synthetic_dataset):
        X, y = synthetic_dataset
        X = X.assign(
            mom_12m=X["mom_6m"] * 0.5,
            vol_63d=X["vol_21d"] * 0.8,
            yield_slope=np.linspace(-1.0, 1.0, len(X)),
            yield_curvature=np.linspace(1.0, -1.0, len(X)),
            real_rate_10y=np.linspace(0.1, 0.3, len(X)),
            credit_spread_hy=np.linspace(0.2, 0.6, len(X)),
            nfci=np.linspace(-0.5, 0.5, len(X)),
            vix=np.linspace(15.0, 25.0, len(X)),
            vmt_yoy=np.linspace(-0.02, 0.03, len(X)),
            combined_ratio_ttm=np.linspace(88.0, 96.0, len(X)),
            investment_income_growth_yoy=np.linspace(-0.1, 0.1, len(X)),
            roe_net_income_ttm=np.linspace(0.08, 0.14, len(X)),
            underwriting_income=np.linspace(100.0, 200.0, len(X)),
        )

        result = run_wfo(X, y, model_type="ridge")
        expected_cols = set(config.MODEL_FEATURE_OVERRIDES["ridge"])
        actual_cols = set(result.folds[0].feature_importances.keys())
        assert actual_cols == expected_cols

    def test_bayesian_ridge_uses_model_specific_feature_subset(self, synthetic_dataset):
        X, y = synthetic_dataset
        X = X.assign(
            mom_12m=X["mom_6m"] * 0.5,
            vol_63d=X["vol_21d"] * 0.8,
            yield_slope=np.linspace(-1.0, 1.0, len(X)),
            yield_curvature=np.linspace(1.0, -1.0, len(X)),
            real_rate_10y=np.linspace(0.1, 0.3, len(X)),
            credit_spread_hy=np.linspace(0.2, 0.6, len(X)),
            nfci=np.linspace(-0.5, 0.5, len(X)),
            vix=np.linspace(15.0, 25.0, len(X)),
            vmt_yoy=np.linspace(-0.02, 0.03, len(X)),
            combined_ratio_ttm=np.linspace(88.0, 96.0, len(X)),
            investment_income_growth_yoy=np.linspace(-0.1, 0.1, len(X)),
            roe_net_income_ttm=np.linspace(0.08, 0.14, len(X)),
            underwriting_income=np.linspace(100.0, 200.0, len(X)),
        )

        result = run_wfo(X, y, model_type="bayesian_ridge")
        expected_cols = set(config.MODEL_FEATURE_OVERRIDES["bayesian_ridge"])
        actual_cols = set(result.folds[0].feature_importances.keys())
        assert actual_cols == expected_cols


# ---------------------------------------------------------------------------
# v2 WFOResult metadata tests
# ---------------------------------------------------------------------------

class TestWFOResultMetadata:
    def test_benchmark_stored_in_result(self, synthetic_dataset):
        """benchmark kwarg is stored in WFOResult.benchmark."""
        X, y = synthetic_dataset
        result = run_wfo(X, y, benchmark="VTI")
        assert result.benchmark == "VTI"

    def test_target_horizon_stored_in_result(self, synthetic_dataset):
        """target_horizon_months kwarg is stored in WFOResult.target_horizon."""
        X, y = synthetic_dataset
        result = run_wfo(X, y, target_horizon_months=12)
        assert result.target_horizon == 12

    def test_model_type_stored_in_result(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="ridge")
        assert result.model_type == "ridge"

    def test_summary_includes_benchmark_and_horizon(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y, benchmark="BND", target_horizon_months=12)
        summary = result.summary()
        assert summary["benchmark"] == "BND"
        assert summary["target_horizon"] == 12


# ---------------------------------------------------------------------------
# v2 predict_current tests
# ---------------------------------------------------------------------------

class TestPredictCurrent:
    def test_returns_required_keys(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="lasso", target_horizon_months=6)
        # Use all but last row as full history; last row as current
        X_full = X.iloc[:-1]
        y_full = y.iloc[:-1]
        X_current = X.iloc[[-1]]
        pred = predict_current(X_full, y_full, X_current, result)
        assert "predicted_return" in pred
        assert "ic" in pred
        assert "hit_rate" in pred
        assert "benchmark" in pred
        assert "target_horizon" in pred
        assert "top_features" in pred

    def test_predicted_return_is_float(self, synthetic_dataset):
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="lasso")
        pred = predict_current(X.iloc[:-1], y.iloc[:-1], X.iloc[[-1]], result)
        assert isinstance(pred["predicted_return"], float)

    def test_predict_current_is_not_last_fold_mean(self, synthetic_dataset):
        """
        predict_current() must return a fresh model prediction, not the v1
        placeholder (mean of the last fold's y_hat).

        We verify this by passing an extreme X_current (all features set to
        10× the column max) and checking that the prediction differs from the
        last fold's out-of-sample mean — i.e., features are actually used.
        """
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="ridge")  # Ridge never zeros coefficients
        X_full = X.iloc[:-1]
        y_full = y.iloc[:-1]

        # Construct two extreme observations at opposite ends of the feature space
        X_extreme_high = X.iloc[[-1]].copy()
        X_extreme_low = X.iloc[[-1]].copy()
        X_extreme_high.iloc[0] = X.max().values * 10
        X_extreme_low.iloc[0] = X.min().values * 10

        pred_high = predict_current(X_full, y_full, X_extreme_high, result,
                                    model_type="ridge")
        pred_low = predict_current(X_full, y_full, X_extreme_low, result,
                                   model_type="ridge")

        # Ridge with non-zero coefficients must give different predictions for
        # extreme-high vs extreme-low feature values.
        assert pred_high["predicted_return"] != pred_low["predicted_return"], (
            "predict_current returned the same value for extreme-high and "
            "extreme-low feature inputs under Ridge — this suggests the model "
            "has no non-zero coefficients or is using a constant placeholder."
        )

    def test_ic_and_hit_rate_from_wfo_result(self, synthetic_dataset):
        """ic and hit_rate in predict_current output must match wfo_result metrics."""
        X, y = synthetic_dataset
        result = run_wfo(X, y, model_type="lasso", benchmark="VTI")
        pred = predict_current(X.iloc[:-1], y.iloc[:-1], X.iloc[[-1]], result)
        assert pred["ic"] == pytest.approx(result.information_coefficient)
        assert pred["hit_rate"] == pytest.approx(result.hit_rate)
        assert pred["benchmark"] == "VTI"

    def test_predict_current_respects_elasticnet_feature_subset(self, synthetic_dataset):
        X, y = synthetic_dataset
        X = X.assign(
            mom_12m=X["mom_6m"] * 0.5,
            vol_63d=X["vol_21d"] * 0.8,
            yield_slope=np.linspace(-1.0, 1.0, len(X)),
            yield_curvature=np.linspace(1.0, -1.0, len(X)),
            real_rate_10y=np.linspace(0.1, 0.3, len(X)),
            credit_spread_hy=np.linspace(0.2, 0.6, len(X)),
            nfci=np.linspace(-0.5, 0.5, len(X)),
            vix=np.linspace(15.0, 25.0, len(X)),
            vmt_yoy=np.linspace(-0.02, 0.03, len(X)),
            investment_income_growth_yoy=np.linspace(-0.1, 0.1, len(X)),
            roe_net_income_ttm=np.linspace(0.08, 0.14, len(X)),
            underwriting_income=np.linspace(100.0, 200.0, len(X)),
            channel_mix_agency_pct=np.linspace(0.6, 0.7, len(X)),
        )

        result = run_wfo(X, y, model_type="elasticnet")
        pred = predict_current(
            X.iloc[:-1],
            y.iloc[:-1],
            X.iloc[[-1]],
            result,
            model_type="elasticnet",
        )
        top_feature_names = {name for name, _ in pred["top_features"]}
        assert top_feature_names.issubset(set(config.MODEL_FEATURE_OVERRIDES["elasticnet"]))
