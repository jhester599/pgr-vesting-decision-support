"""
Tests for src/models/multi_benchmark_wfo.py

Verifies that run_all_benchmarks() and get_current_signals() correctly:
  - Train one WFO model per benchmark column.
  - Skip benchmarks with insufficient overlapping data (no crash).
  - Propagate the benchmark ticker and target_horizon into each WFOResult.
  - Classify signals correctly (OUTPERFORM / UNDERPERFORM / NEUTRAL).
  - Handle empty inputs gracefully.
"""

import numpy as np
import pandas as pd
import pytest

import config
from src.models.multi_benchmark_wfo import run_all_benchmarks, get_current_signals


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_feature_matrix():
    """
    180 monthly observations × 5 features.
    Returns (X, dates) so callers can build y series from the same index.
    """
    n = 180
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2009-01-30", periods=n, freq="BME")
    X = pd.DataFrame(
        {
            "mom_3m":  rng.normal(0, 0.05, n),
            "mom_6m":  rng.normal(0, 0.08, n),
            "vol_21d": np.abs(rng.normal(0.2, 0.05, n)),
            "rsi_14":  rng.uniform(30, 70, n),
            "pe_ratio": rng.uniform(10, 30, n),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return X, pd.DatetimeIndex(dates, name="date")


@pytest.fixture()
def small_relative_return_matrix(synthetic_feature_matrix):
    """
    3-column relative return matrix aligned to the feature matrix dates.
    Each column is one synthetic ETF benchmark.
    """
    X, dates = synthetic_feature_matrix
    rng = np.random.default_rng(99)
    # Drop last 6 to leave room for forward-return NaN (match typical real usage)
    valid_dates = dates[:-6]
    rel = pd.DataFrame(
        {
            "ETF_A": rng.normal(0.01, 0.05, len(valid_dates)),
            "ETF_B": rng.normal(-0.005, 0.04, len(valid_dates)),
            "ETF_C": rng.normal(0.0, 0.06, len(valid_dates)),
        },
        index=valid_dates,
    )
    return rel


@pytest.fixture()
def run_result(synthetic_feature_matrix, small_relative_return_matrix):
    """Pre-computed run_all_benchmarks result used by multiple tests."""
    X, _ = synthetic_feature_matrix
    return run_all_benchmarks(
        X,
        small_relative_return_matrix,
        model_type="lasso",
        target_horizon_months=6,
    )


# ---------------------------------------------------------------------------
# run_all_benchmarks tests
# ---------------------------------------------------------------------------

class TestRunAllBenchmarks:
    def test_returns_dict_keyed_by_etf(self, run_result):
        assert isinstance(run_result, dict)
        for key in run_result:
            assert isinstance(key, str)

    def test_one_result_per_benchmark_column(
        self, run_result, small_relative_return_matrix
    ):
        """Every column in the matrix should produce one WFOResult."""
        assert set(run_result.keys()) == set(small_relative_return_matrix.columns)

    def test_each_result_has_folds(self, run_result):
        for etf, res in run_result.items():
            assert len(res.folds) > 0, f"{etf}: WFOResult has no folds."

    def test_benchmark_stored_in_wfo_result(self, run_result):
        for etf, res in run_result.items():
            assert res.benchmark == etf, (
                f"WFOResult.benchmark should be '{etf}' but got '{res.benchmark}'."
            )

    def test_target_horizon_stored_in_wfo_result(self, run_result):
        for etf, res in run_result.items():
            assert res.target_horizon == 6

    def test_no_train_data_shares_test_obs(self, run_result):
        """
        Core temporal integrity: max train index < min test index for every
        fold in every model.
        """
        for etf, res in run_result.items():
            for fold in res.folds:
                assert fold.train_end < fold.test_start, (
                    f"{etf} fold {fold.fold_idx}: train_end >= test_start "
                    "(temporal leakage)."
                )

    def test_skips_benchmark_with_no_overlap(self, synthetic_feature_matrix):
        """
        A benchmark column whose dates do not overlap the feature matrix index
        should be silently skipped (not raise, not produce empty WFOResult).
        """
        X, dates = synthetic_feature_matrix
        rng = np.random.default_rng(0)
        # Dates entirely in the future — no overlap with X
        future_dates = pd.bdate_range("2040-01-01", periods=50, freq="BME")
        rel = pd.DataFrame(
            {"NO_OVERLAP": rng.normal(0, 0.05, 50)},
            index=pd.DatetimeIndex(future_dates, name="date"),
        )
        result = run_all_benchmarks(X, rel, target_horizon_months=6)
        assert "NO_OVERLAP" not in result

    def test_raises_on_empty_relative_return_matrix(self, synthetic_feature_matrix):
        X, _ = synthetic_feature_matrix
        with pytest.raises(ValueError, match="empty"):
            run_all_benchmarks(X, pd.DataFrame(), target_horizon_months=6)

    def test_ridge_model_type_stored(self, synthetic_feature_matrix, small_relative_return_matrix):
        X, _ = synthetic_feature_matrix
        result = run_all_benchmarks(
            X, small_relative_return_matrix,
            model_type="ridge",
            target_horizon_months=6,
        )
        for etf, res in result.items():
            assert res.model_type == "ridge"

    def test_12m_horizon_stored(self, synthetic_feature_matrix, small_relative_return_matrix):
        X, _ = synthetic_feature_matrix
        # Need larger dataset for 12M embargo; use 180 rows which is sufficient
        result = run_all_benchmarks(
            X, small_relative_return_matrix,
            model_type="lasso",
            target_horizon_months=12,
        )
        for etf, res in result.items():
            assert res.target_horizon == 12


# ---------------------------------------------------------------------------
# get_current_signals tests
# ---------------------------------------------------------------------------

class TestGetCurrentSignals:
    def test_returns_dataframe(
        self, synthetic_feature_matrix, small_relative_return_matrix, run_result
    ):
        X, dates = synthetic_feature_matrix
        X_current = X.iloc[[-1]]
        signals = get_current_signals(
            X_full=X,
            relative_return_matrix=small_relative_return_matrix,
            wfo_results=run_result,
            X_current=X_current,
        )
        assert isinstance(signals, pd.DataFrame)

    def test_one_row_per_benchmark(
        self, synthetic_feature_matrix, small_relative_return_matrix, run_result
    ):
        X, _ = synthetic_feature_matrix
        X_current = X.iloc[[-1]]
        signals = get_current_signals(
            X_full=X,
            relative_return_matrix=small_relative_return_matrix,
            wfo_results=run_result,
            X_current=X_current,
        )
        assert set(signals.index) == set(run_result.keys())

    def test_required_columns_present(
        self, synthetic_feature_matrix, small_relative_return_matrix, run_result
    ):
        X, _ = synthetic_feature_matrix
        signals = get_current_signals(
            X_full=X,
            relative_return_matrix=small_relative_return_matrix,
            wfo_results=run_result,
            X_current=X.iloc[[-1]],
        )
        for col in ["predicted_relative_return", "ic", "hit_rate", "signal", "top_feature"]:
            assert col in signals.columns, f"Missing column: {col}"

    def test_signal_values_are_valid(
        self, synthetic_feature_matrix, small_relative_return_matrix, run_result
    ):
        X, _ = synthetic_feature_matrix
        signals = get_current_signals(
            X_full=X,
            relative_return_matrix=small_relative_return_matrix,
            wfo_results=run_result,
            X_current=X.iloc[[-1]],
        )
        valid_signals = {"OUTPERFORM", "UNDERPERFORM", "NEUTRAL"}
        for etf, row in signals.iterrows():
            assert row["signal"] in valid_signals, (
                f"{etf}: signal '{row['signal']}' is not in {valid_signals}"
            )

    def test_empty_wfo_results_returns_empty_df(
        self, synthetic_feature_matrix, small_relative_return_matrix
    ):
        X, _ = synthetic_feature_matrix
        signals = get_current_signals(
            X_full=X,
            relative_return_matrix=small_relative_return_matrix,
            wfo_results={},
            X_current=X.iloc[[-1]],
        )
        assert signals.empty

    def test_neutral_when_ic_below_threshold(
        self, synthetic_feature_matrix, small_relative_return_matrix
    ):
        """
        If IC is below _IC_THRESHOLD, signal must be NEUTRAL regardless of
        predicted return magnitude.
        """
        from src.models.multi_benchmark_wfo import _IC_THRESHOLD, _SIGNAL_NEUTRAL
        from unittest.mock import patch

        X, _ = synthetic_feature_matrix
        run_res = run_all_benchmarks(
            X, small_relative_return_matrix,
            model_type="lasso", target_horizon_months=6,
        )

        # Patch predict_current to return IC below threshold + large predicted return
        mock_pred = {
            "predicted_return":  0.5,          # large positive — would be OUTPERFORM
            "ic":                _IC_THRESHOLD - 0.001,  # just below threshold
            "hit_rate":          0.6,
            "benchmark":         "ETF_A",
            "target_horizon":    6,
            "top_features":      [("mom_6m", 0.3)],
        }
        with patch(
            "src.models.multi_benchmark_wfo.predict_current",
            return_value=mock_pred,
        ):
            signals = get_current_signals(
                X_full=X,
                relative_return_matrix=small_relative_return_matrix,
                wfo_results=run_res,
                X_current=X.iloc[[-1]],
            )

        for etf, row in signals.iterrows():
            assert row["signal"] == _SIGNAL_NEUTRAL, (
                f"{etf}: expected NEUTRAL (IC below threshold) but got '{row['signal']}'"
            )
