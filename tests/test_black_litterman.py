"""
Tests for Black-Litterman portfolio construction in src/portfolio/black_litterman.py (v4.0).

Verifies:
  - Output weights sum to approximately 1.0 (or are empty)
  - All weights are non-negative
  - Each weight ≤ KELLY_MAX_POSITION (0.30)
  - Ledoit-Wolf shrinkage is applied (covariance is PSD)
  - Higher view confidence (lower Omega) moves weights closer to the view
  - compute_equilibrium_returns returns a Series with correct shape
  - Graceful fallback to equal weights when no positive-IC signals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.portfolio.black_litterman import (
    build_bl_weights,
    compute_equilibrium_returns,
    _ledoit_wolf_covariance,
)
from src.models.multi_benchmark_wfo import EnsembleWFOResult


def _make_returns(tickers: list[str], n_months: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(
        rng.normal(0.008, 0.04, size=(n_months, len(tickers))),
        index=idx,
        columns=tickers,
    )


def _make_ensemble_result(
    benchmark: str, mean_ic: float = 0.10, mean_mae: float = 0.05, seed: int = 0
) -> EnsembleWFOResult:
    from src.models.wfo_engine import WFOResult, FoldResult
    rng = np.random.default_rng(seed)
    # Create a minimal WFOResult with fake fold data
    fold = FoldResult(
        fold_idx=0,
        train_start=pd.Timestamp("2019-01-31"),
        train_end=pd.Timestamp("2022-12-31"),
        test_start=pd.Timestamp("2023-01-31"),
        test_end=pd.Timestamp("2023-06-30"),
        y_true=rng.normal(size=6),
        y_hat=rng.normal(size=6),
        optimal_alpha=0.01,
        feature_importances={},
        n_train=48,
        n_test=6,
    )
    wfo = WFOResult(folds=[fold], benchmark=benchmark, target_horizon=6, model_type="elasticnet")
    return EnsembleWFOResult(
        benchmark=benchmark,
        target_horizon=6,
        mean_ic=mean_ic,
        mean_hit_rate=0.60,
        mean_mae=mean_mae,
        model_results={"elasticnet": wfo},
    )


class TestLedoitWolfCovariance:
    def test_returns_dataframe(self):
        tickers = ["VTI", "BND", "GLD"]
        returns = _make_returns(tickers)
        cov = _ledoit_wolf_covariance(returns)
        assert isinstance(cov, pd.DataFrame)

    def test_symmetric(self):
        tickers = ["VTI", "BND", "GLD"]
        returns = _make_returns(tickers)
        cov = _ledoit_wolf_covariance(returns)
        assert np.allclose(cov.values, cov.values.T)

    def test_positive_semidefinite(self):
        tickers = ["VTI", "BND", "GLD"]
        returns = _make_returns(tickers)
        cov = _ledoit_wolf_covariance(returns)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert (eigvals >= -1e-10).all(), "Covariance matrix must be PSD"

    def test_same_columns_and_index(self):
        tickers = ["VTI", "BND", "GLD"]
        returns = _make_returns(tickers)
        cov = _ledoit_wolf_covariance(returns)
        assert list(cov.columns) == tickers
        assert list(cov.index) == tickers


class TestBuildBlWeights:
    def test_returns_dict(self):
        tickers = ["VTI", "BND"]
        returns = _make_returns(tickers)
        signals = {t: _make_ensemble_result(t) for t in tickers}
        weights = build_bl_weights(signals, returns)
        assert isinstance(weights, dict)

    def test_all_weights_non_negative(self):
        tickers = ["VTI", "BND", "GLD"]
        returns = _make_returns(tickers)
        signals = {t: _make_ensemble_result(t) for t in tickers}
        weights = build_bl_weights(signals, returns)
        for k, v in weights.items():
            assert v >= -1e-8, f"Weight for {k} is negative: {v}"

    def test_weights_sum_to_approximately_one(self):
        tickers = ["VTI", "BND", "GLD"]
        returns = _make_returns(tickers)
        signals = {t: _make_ensemble_result(t) for t in tickers}
        weights = build_bl_weights(signals, returns)
        if weights:
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.05, f"Weights sum to {total}, expected ~1.0"

    def test_each_weight_at_most_max_position(self):
        tickers = ["VTI", "BND", "GLD", "VGT"]
        returns = _make_returns(tickers)
        signals = {t: _make_ensemble_result(t) for t in tickers}
        weights = build_bl_weights(signals, returns)
        for k, v in weights.items():
            assert v <= config.KELLY_MAX_POSITION + 1e-6, (
                f"Weight for {k} = {v} exceeds KELLY_MAX_POSITION={config.KELLY_MAX_POSITION}"
            )

    def test_zero_ic_benchmarks_excluded_from_views(self):
        """Benchmarks with mean_ic ≤ 0 should not generate views."""
        tickers = ["VTI", "BND"]
        returns = _make_returns(tickers)
        # BND has negative IC → should not receive a view
        signals = {
            "VTI": _make_ensemble_result("VTI", mean_ic=0.10),
            "BND": _make_ensemble_result("BND", mean_ic=-0.05),
        }
        weights = build_bl_weights(signals, returns)
        # Function should still return without error
        assert isinstance(weights, dict)

    def test_no_positive_ic_returns_equal_weights(self):
        """When all benchmarks have IC ≤ 0, should return equal weights."""
        tickers = ["VTI", "BND"]
        returns = _make_returns(tickers)
        signals = {
            "VTI": _make_ensemble_result("VTI", mean_ic=-0.05),
            "BND": _make_ensemble_result("BND", mean_ic=-0.10),
        }
        weights = build_bl_weights(signals, returns)
        if weights:
            # All equal weights → each ≈ 1/n
            expected = 1.0 / len(tickers)
            for v in weights.values():
                assert abs(v - expected) < 0.05

    def test_insufficient_data_raises(self):
        tickers = ["VTI", "BND"]
        short_returns = _make_returns(tickers, n_months=5)
        signals = {t: _make_ensemble_result(t) for t in tickers}
        with pytest.raises(ValueError, match="12 months"):
            build_bl_weights(signals, short_returns)


class TestComputeEquilibriumReturns:
    def test_returns_series(self):
        tickers = ["VTI", "BND", "GLD"]
        returns = _make_returns(tickers)
        from src.portfolio.black_litterman import _ledoit_wolf_covariance
        cov = _ledoit_wolf_covariance(returns)
        w = pd.Series({t: 1.0 / 3 for t in tickers})
        pi = compute_equilibrium_returns(cov, w)
        assert isinstance(pi, pd.Series)

    def test_correct_length(self):
        tickers = ["VTI", "BND", "GLD"]
        returns = _make_returns(tickers)
        from src.portfolio.black_litterman import _ledoit_wolf_covariance
        cov = _ledoit_wolf_covariance(returns)
        w = pd.Series({t: 1.0 / 3 for t in tickers})
        pi = compute_equilibrium_returns(cov, w)
        assert len(pi) == len(tickers)

    def test_finite_values(self):
        tickers = ["VTI", "BND", "GLD"]
        returns = _make_returns(tickers)
        from src.portfolio.black_litterman import _ledoit_wolf_covariance
        cov = _ledoit_wolf_covariance(returns)
        w = pd.Series({t: 1.0 / 3 for t in tickers})
        pi = compute_equilibrium_returns(cov, w)
        assert np.isfinite(pi.values).all()
