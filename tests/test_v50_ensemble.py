"""
Tests for v5.0 ensemble upgrades.

Validates:
  - build_gbt_pipeline()          (shallow GBT as 4th ensemble member)
  - get_feature_importances()     (handles feature_importances_ for tree models)
  - CPCV config upgrade           (CPCV_N_FOLDS=8 → C(8,2)=28 paths)
  - ENSEMBLE_MODELS contains 'gbt'
  - DIAG_CPCV_MIN_POSITIVE_PATHS updated to 19
  - run_wfo() accepts model_type="gbt"
  - Inverse-variance ensemble weighting in get_ensemble_signals()
  - Full 4-model ensemble run returns EnsembleWFOResult per benchmark

All tests are DB-independent (synthetic data only).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.models.regularized_models import (
    build_gbt_pipeline,
    build_bayesian_ridge_pipeline,
    build_elasticnet_pipeline,
    build_ridge_pipeline,
    get_feature_importances,
)
from src.models.wfo_engine import WFOResult, FoldResult, run_wfo
from src.models.multi_benchmark_wfo import (
    EnsembleWFOResult,
    get_ensemble_signals,
    run_ensemble_benchmarks,
)


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _make_Xy(n: int = 100, p: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic monthly feature matrix and target series."""
    dates = pd.date_range("2015-01-31", periods=n, freq="ME")
    X = pd.DataFrame(
        _RNG.normal(0, 1, (n, p)),
        index=dates,
        columns=[f"f{i}" for i in range(p)],
    )
    y = pd.Series(_RNG.normal(0, 0.05, n), index=dates, name="rel_return")
    return X, y


def _make_wfo_result(
    mae: float = 0.05,
    model_type: str = "elasticnet",
    n_obs: int = 20,
) -> WFOResult:
    """
    Stub WFOResult with a precisely controlled OOS MAE.

    Constructs y_true as a constant vector (0.05) and y_hat = y_true + mae,
    so sklearn.metrics.mean_absolute_error(y_true, y_hat) == mae exactly.
    This makes inverse-variance weight tests deterministic without mocking.
    """
    y_true = np.full(n_obs, 0.05)
    y_hat = y_true + mae   # |y_true - y_hat| = mae for every observation

    fold = FoldResult(
        fold_idx=0,
        train_start=pd.Timestamp("2015-01-31"),
        train_end=pd.Timestamp("2019-12-31"),
        test_start=pd.Timestamp("2020-01-31"),
        test_end=pd.Timestamp("2020-10-31"),
        y_true=y_true,
        y_hat=y_hat,
        optimal_alpha=0.01,
        feature_importances={},
        n_train=60,
        n_test=n_obs,
    )
    fold._test_dates = list(pd.date_range("2020-01-31", periods=n_obs, freq="ME"))

    result = WFOResult(benchmark="VTI", target_horizon=6, model_type=model_type)
    result.folds = [fold]
    return result


# ===========================================================================
# Config constants
# ===========================================================================

class TestConfigV50:
    def test_cpcv_n_folds_is_8(self) -> None:
        assert config.CPCV_N_FOLDS == 8

    def test_cpcv_n_test_folds_is_2(self) -> None:
        assert config.CPCV_N_TEST_FOLDS == 2

    def test_cpcv_path_count(self) -> None:
        """C(8, 2) = 28 paths."""
        import math
        n_paths = math.comb(config.CPCV_N_FOLDS, config.CPCV_N_TEST_FOLDS)
        assert n_paths == 28

    def test_diag_cpcv_min_positive_paths(self) -> None:
        assert config.DIAG_CPCV_MIN_POSITIVE_PATHS == 19

    def test_diag_threshold_fraction_consistent(self) -> None:
        """≥19/28 ≈ 67.9% — same ballpark as the former ≥13/15 ≈ 86.7% threshold."""
        n_paths = math.comb(config.CPCV_N_FOLDS, config.CPCV_N_TEST_FOLDS)
        frac = config.DIAG_CPCV_MIN_POSITIVE_PATHS / n_paths
        assert 0.60 <= frac <= 0.80, (
            f"Threshold fraction {frac:.1%} outside expected 60–80% range"
        )

    def test_gbt_in_ensemble_models(self) -> None:
        assert "gbt" in config.ENSEMBLE_MODELS

    def test_ensemble_has_four_models(self) -> None:
        assert len(config.ENSEMBLE_MODELS) == 4

    def test_all_original_models_present(self) -> None:
        for m in ["elasticnet", "ridge", "bayesian_ridge"]:
            assert m in config.ENSEMBLE_MODELS, f"{m} missing from ENSEMBLE_MODELS"


# ===========================================================================
# GBT pipeline
# ===========================================================================

class TestBuildGbtPipeline:
    def test_returns_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline
        pipe = build_gbt_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_model_step(self) -> None:
        pipe = build_gbt_pipeline()
        assert "model" in pipe.named_steps

    def test_model_is_gbt(self) -> None:
        from sklearn.ensemble import GradientBoostingRegressor
        pipe = build_gbt_pipeline()
        assert isinstance(pipe.named_steps["model"], GradientBoostingRegressor)

    def test_default_max_depth(self) -> None:
        pipe = build_gbt_pipeline()
        assert pipe.named_steps["model"].max_depth == 2

    def test_default_n_estimators(self) -> None:
        pipe = build_gbt_pipeline()
        assert pipe.named_steps["model"].n_estimators == 50

    def test_fits_and_predicts(self) -> None:
        X, y = _make_Xy(n=80)
        pipe = build_gbt_pipeline()
        pipe.fit(X.values, y.values)
        preds = pipe.predict(X.values[:5])
        assert preds.shape == (5,)
        assert np.isfinite(preds).all()

    def test_custom_params(self) -> None:
        pipe = build_gbt_pipeline(max_depth=3, n_estimators=100)
        assert pipe.named_steps["model"].max_depth == 3
        assert pipe.named_steps["model"].n_estimators == 100


# ===========================================================================
# get_feature_importances — tree model support
# ===========================================================================

class TestGetFeatureImportancesGbt:
    def _fitted_gbt(self) -> tuple:
        X, y = _make_Xy(n=80, p=5)
        pipe = build_gbt_pipeline()
        pipe.fit(X.values, y.values)
        return pipe, list(X.columns)

    def test_returns_dict(self) -> None:
        pipe, names = self._fitted_gbt()
        result = get_feature_importances(pipe, names)
        assert isinstance(result, dict)

    def test_all_features_present(self) -> None:
        pipe, names = self._fitted_gbt()
        result = get_feature_importances(pipe, names)
        assert set(result.keys()) == set(names)

    def test_importances_non_negative(self) -> None:
        """GBT feature_importances_ are always >= 0."""
        pipe, names = self._fitted_gbt()
        result = get_feature_importances(pipe, names)
        assert all(v >= 0 for v in result.values())

    def test_importances_sum_to_one(self) -> None:
        """GBT importances normalise to 1.0."""
        pipe, names = self._fitted_gbt()
        result = get_feature_importances(pipe, names)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_sorted_descending(self) -> None:
        pipe, names = self._fitted_gbt()
        result = get_feature_importances(pipe, names)
        vals = list(result.values())
        assert vals == sorted(vals, reverse=True)

    def test_linear_model_still_works(self) -> None:
        """Regression: linear models (coef_) still return non-empty dict."""
        X, y = _make_Xy(n=80, p=5)
        pipe = build_elasticnet_pipeline()
        pipe.fit(X.values, y.values)
        result = get_feature_importances(pipe, list(X.columns))
        assert len(result) == 5

    def test_no_coef_no_importances_returns_empty(self) -> None:
        """Model with neither coef_ nor feature_importances_ returns {}."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        class _DummyModel:
            def fit(self, X, y): ...
            def predict(self, X): return np.zeros(len(X))

        pipe = Pipeline([("scaler", StandardScaler()), ("model", _DummyModel())])
        pipe.fit(np.ones((10, 3)), np.ones(10))
        result = get_feature_importances(pipe, ["a", "b", "c"])
        assert result == {}


# ===========================================================================
# run_wfo with model_type="gbt"
# ===========================================================================

class TestRunWfoGbt:
    def test_gbt_runs_and_returns_result(self) -> None:
        X, y = _make_Xy(n=150)
        result = run_wfo(X, y, model_type="gbt", target_horizon_months=6, benchmark="VTI")
        assert len(result.folds) > 0

    def test_gbt_result_has_ic(self) -> None:
        X, y = _make_Xy(n=150)
        result = run_wfo(X, y, model_type="gbt", target_horizon_months=6)
        assert np.isfinite(result.information_coefficient)

    def test_gbt_result_model_type_recorded(self) -> None:
        X, y = _make_Xy(n=150)
        result = run_wfo(X, y, model_type="gbt", target_horizon_months=6)
        assert result.model_type == "gbt"

    def test_gbt_optimal_alpha_is_zero_sentinel(self) -> None:
        """GBT has no alpha; wfo_engine stores 0.0 as sentinel."""
        X, y = _make_Xy(n=150)
        result = run_wfo(X, y, model_type="gbt", target_horizon_months=6)
        for fold in result.folds:
            assert fold.optimal_alpha == pytest.approx(0.0)


# ===========================================================================
# Inverse-variance weighting in get_ensemble_signals()
# ===========================================================================

class TestInverseVarianceWeighting:
    """
    Unit-test the weighting logic by constructing controlled EnsembleWFOResult
    stubs and verifying that lower-MAE models get higher weight.
    """

    def _build_signals(
        self,
        maes: dict[str, float],
        preds: dict[str, float],
        X: pd.DataFrame,
        y: pd.Series,
        etf: str = "VTI",
    ) -> pd.DataFrame:
        """
        Build a synthetic EnsembleWFOResult with custom per-model MAE and
        controlled live prediction values, then call get_ensemble_signals().

        Predictions are injected by monkeypatching predict_current to return
        per-model constant values.
        """
        import src.models.multi_benchmark_wfo as mbw
        from src.models.wfo_engine import WFOResult

        # Build stub WFOResults with the requested MAE values
        model_results: dict[str, WFOResult] = {}
        for mtype, mae_val in maes.items():
            wfo_r = _make_wfo_result(mae=mae_val, model_type=mtype)
            model_results[mtype] = wfo_r

        ens = EnsembleWFOResult(
            benchmark=etf,
            target_horizon=6,
            mean_ic=0.10,
            mean_hit_rate=0.55,
            mean_mae=float(np.mean(list(maes.values()))),
            model_results=model_results,
        )

        rel_matrix = pd.DataFrame({etf: y}, index=y.index)
        X_current = X.iloc[[-1]]

        # Patch predict_current to return the controlled prediction values
        original_pc = mbw.predict_current

        def _mock_pc(X_full, y_full, X_current, wfo_result, model_type, **kwargs):
            return {
                "predicted_return": preds.get(model_type, 0.0),
                "prediction_std": 0.05,
                "ic": 0.10,
                "hit_rate": 0.55,
                "benchmark": etf,
                "target_horizon": 6,
                "top_features": [],
            }

        mbw.predict_current = _mock_pc
        try:
            signals = get_ensemble_signals(
                X_full=X,
                relative_return_matrix=rel_matrix,
                ensemble_results={etf: ens},
                X_current=X_current,
            )
        finally:
            mbw.predict_current = original_pc

        return signals

    def test_equal_mae_produces_equal_weight_mean(self) -> None:
        """When all MAEs are equal, weighted avg = simple mean."""
        X, y = _make_Xy(n=120)
        maes = {"elasticnet": 0.05, "ridge": 0.05, "bayesian_ridge": 0.05}
        preds = {"elasticnet": 0.10, "ridge": 0.20, "bayesian_ridge": 0.30}
        signals = self._build_signals(maes, preds, X, y)
        expected = (0.10 + 0.20 + 0.30) / 3
        assert signals.loc["VTI", "point_prediction"] == pytest.approx(expected, rel=1e-6)

    def test_lower_mae_gets_higher_weight(self) -> None:
        """
        Model A has MAE=0.01, Model B has MAE=0.10.
        Weight_A / Weight_B = (0.10)² / (0.01)² = 100.
        Weighted avg should be much closer to pred_A than pred_B.
        """
        X, y = _make_Xy(n=120)
        maes = {"elasticnet": 0.01, "ridge": 0.10}
        preds = {"elasticnet": 1.0, "ridge": -1.0}
        signals = self._build_signals(maes, preds, X, y)
        result = signals.loc["VTI", "point_prediction"]
        # Weight_A = 1/0.0001 = 10000, Weight_B = 1/0.01 = 100
        # expected ≈ (1.0 * 10000 + (-1.0) * 100) / 10100 ≈ 0.9802
        expected = (1.0 * 10000 + (-1.0) * 100) / (10000 + 100)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_single_model_returns_that_prediction(self) -> None:
        X, y = _make_Xy(n=120)
        maes = {"elasticnet": 0.05}
        preds = {"elasticnet": 0.123}
        signals = self._build_signals(maes, preds, X, y)
        assert signals.loc["VTI", "point_prediction"] == pytest.approx(0.123, rel=1e-6)

    def test_zero_mae_fallback_weight_one(self) -> None:
        """
        MAE=0 edge case: weight falls back to 1.0 to avoid division by zero.
        Two models with MAE=0 → equal weight → simple mean.
        """
        X, y = _make_Xy(n=120)
        maes = {"elasticnet": 0.0, "ridge": 0.0}
        preds = {"elasticnet": 0.10, "ridge": 0.20}
        signals = self._build_signals(maes, preds, X, y)
        assert signals.loc["VTI", "point_prediction"] == pytest.approx(0.15, rel=1e-6)


# ===========================================================================
# Full 4-model ensemble integration (run_ensemble_benchmarks)
# ===========================================================================

class TestFourModelEnsemble:
    def test_four_models_in_result(self) -> None:
        """run_ensemble_benchmarks returns a result with all 4 model types."""
        X, y = _make_Xy(n=150)
        rel_matrix = pd.DataFrame({"VTI": y})
        results = run_ensemble_benchmarks(X, rel_matrix, target_horizon_months=6)
        assert "VTI" in results
        model_keys = set(results["VTI"].model_results.keys())
        assert "gbt" in model_keys, f"'gbt' missing from model_results: {model_keys}"

    def test_all_four_model_types_present(self) -> None:
        X, y = _make_Xy(n=150)
        rel_matrix = pd.DataFrame({"VTI": y})
        results = run_ensemble_benchmarks(X, rel_matrix, target_horizon_months=6)
        model_keys = set(results["VTI"].model_results.keys())
        for expected in config.ENSEMBLE_MODELS:
            assert expected in model_keys, f"Model '{expected}' missing"

    def test_ensemble_result_has_finite_ic(self) -> None:
        X, y = _make_Xy(n=150)
        rel_matrix = pd.DataFrame({"VTI": y})
        results = run_ensemble_benchmarks(X, rel_matrix, target_horizon_months=6)
        assert np.isfinite(results["VTI"].mean_ic)

    def test_ensemble_result_has_finite_hit_rate(self) -> None:
        X, y = _make_Xy(n=150)
        rel_matrix = pd.DataFrame({"VTI": y})
        results = run_ensemble_benchmarks(X, rel_matrix, target_horizon_months=6)
        hr = results["VTI"].mean_hit_rate
        assert 0.0 <= hr <= 1.0


# ===========================================================================
# ETF description lookup (monthly_decision._ETF_DESCRIPTIONS)
# ===========================================================================

class TestEtfDescriptions:
    def _get_desc(self) -> dict[str, str]:
        import scripts.monthly_decision as md
        return md._ETF_DESCRIPTIONS

    def test_all_benchmark_etfs_have_description(self) -> None:
        desc = self._get_desc()
        missing = [t for t in config.ETF_BENCHMARK_UNIVERSE if t not in desc]
        assert missing == [], f"ETFs missing descriptions: {missing}"

    def test_no_banned_words(self) -> None:
        """Descriptions must not contain 'Vanguard', 'Fund', or 'ETF'."""
        desc = self._get_desc()
        banned = ["Vanguard", "Fund", "ETF"]
        violations = [
            (ticker, d, word)
            for ticker, d in desc.items()
            for word in banned
            if word.lower() in d.lower()
        ]
        assert violations == [], f"Banned words found: {violations}"

    def test_descriptions_are_non_empty(self) -> None:
        desc = self._get_desc()
        empty = [t for t, d in desc.items() if not d.strip()]
        assert empty == [], f"Empty descriptions: {empty}"

    def test_vti_description(self) -> None:
        desc = self._get_desc()
        assert "Total Stock Market" in desc["VTI"]

    def test_kie_description(self) -> None:
        desc = self._get_desc()
        assert "Insurance" in desc["KIE"]
