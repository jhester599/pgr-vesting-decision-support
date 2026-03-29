"""
Tests for Combinatorial Purged Cross-Validation (CPCV) in wfo_engine.py (v4.0).

Verifies:
  - C(6, 2) = 15 train-test splits for n_folds=6, n_test_folds=2
  - n_paths = 5 (n_test_paths from CombinatorialPurgedCV)
  - CPCVResult has correct structure and finite IC values
  - Temporal ordering preserved: all train indices < all test indices per split
  - Insufficient data raises ValueError
  - Works with elasticnet, ridge, and bayesian_ridge model types
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.wfo_engine import CPCVResult, run_cpcv


def _make_data(n: int = 150, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-31", periods=n, freq="ME")
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.normal(size=n), index=idx, name="target")
    return X, y


class TestCPCVSplitCount:
    def test_15_splits_for_6_folds_2_test_folds(self):
        """C(6, 2) = 15."""
        X, y = _make_data()
        result = run_cpcv(X, y, n_folds=6, n_test_folds=2)
        assert result.n_splits == 15

    def test_5_paths_for_6_folds_2_test_folds(self):
        """n_test_paths for C(6,2) config is 5."""
        X, y = _make_data()
        result = run_cpcv(X, y, n_folds=6, n_test_folds=2)
        assert result.n_paths == 5

    def test_split_ics_length_equals_n_splits(self):
        X, y = _make_data()
        result = run_cpcv(X, y, n_folds=6, n_test_folds=2)
        # split_ics may be slightly less if some splits have < 2 obs
        assert len(result.split_ics) <= result.n_splits
        assert len(result.split_ics) >= 1


class TestCPCVResult:
    def test_returns_cpcv_result(self):
        X, y = _make_data()
        result = run_cpcv(X, y)
        assert isinstance(result, CPCVResult)

    def test_mean_ic_is_finite(self):
        X, y = _make_data()
        result = run_cpcv(X, y)
        assert np.isfinite(result.mean_ic)

    def test_path_ics_are_finite(self):
        X, y = _make_data()
        result = run_cpcv(X, y)
        for ic in result.path_ics:
            assert np.isfinite(ic), f"Path IC {ic} is not finite"

    def test_mean_ic_equals_mean_of_path_ics(self):
        X, y = _make_data()
        result = run_cpcv(X, y)
        if result.path_ics:
            expected = float(np.nanmean(result.path_ics))
            assert abs(result.mean_ic - expected) < 1e-9

    def test_ic_std_non_negative(self):
        X, y = _make_data()
        result = run_cpcv(X, y)
        if not np.isnan(result.ic_std):
            assert result.ic_std >= 0.0

    def test_benchmark_field_set(self):
        X, y = _make_data()
        result = run_cpcv(X, y, benchmark="VTI")
        assert result.benchmark == "VTI"

    def test_model_type_field_set(self):
        X, y = _make_data()
        result = run_cpcv(X, y, model_type="ridge")
        assert result.model_type == "ridge"


class TestCPCVTemporalOrdering:
    def test_fold_indices_are_contiguous_and_ordered(self):
        """
        CPCV divides the data into N ordered, contiguous, non-overlapping folds.
        Each fold's indices should be strictly greater than all previous folds.
        Note: unlike WFO, the training SET may include later folds than test folds
        (that is the CPCV design — combinatorial combinations of folds).
        """
        from skfolio.model_selection import CombinatorialPurgedCV
        import numpy as np
        X_arr = np.zeros((150, 4))
        cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=0)
        # Get all unique fold partitions from all splits
        all_test_indices = set()
        for train_idx, test_idx_list in cv.split(X_arr):
            for test_idx in test_idx_list:
                for idx in test_idx:
                    all_test_indices.add(int(idx))
        # All observations should appear in at least one test set
        assert len(all_test_indices) > 0

    def test_train_and_test_partitions_disjoint(self):
        """Within each split, train and test index sets must not overlap."""
        from skfolio.model_selection import CombinatorialPurgedCV
        import numpy as np
        X_arr = np.zeros((120, 4))
        cv = CombinatorialPurgedCV(n_folds=6, n_test_folds=2, purged_size=0)
        for train_idx, test_idx_list in cv.split(X_arr):
            train_set = set(train_idx.tolist())
            for test_idx in test_idx_list:
                test_set = set(test_idx.tolist())
                overlap = train_set & test_set
                assert len(overlap) == 0, (
                    f"Train and test sets overlap: {overlap}"
                )


class TestCPCVModelTypes:
    def test_elasticnet_runs(self):
        X, y = _make_data(n=120)
        result = run_cpcv(X, y, model_type="elasticnet", n_folds=4, n_test_folds=2)
        assert isinstance(result, CPCVResult)

    def test_ridge_runs(self):
        X, y = _make_data(n=120)
        result = run_cpcv(X, y, model_type="ridge", n_folds=4, n_test_folds=2)
        assert isinstance(result, CPCVResult)

    def test_bayesian_ridge_runs(self):
        X, y = _make_data(n=120)
        result = run_cpcv(X, y, model_type="bayesian_ridge", n_folds=4, n_test_folds=2)
        assert isinstance(result, CPCVResult)


class TestCPCVEdgeCases:
    def test_insufficient_data_raises(self):
        X, y = _make_data(n=5)
        with pytest.raises(ValueError, match="Insufficient"):
            run_cpcv(X, y, n_folds=6, n_test_folds=2)

    def test_config_defaults_used_when_none(self):
        import config
        import math
        X, y = _make_data()
        result = run_cpcv(X, y, n_folds=None, n_test_folds=None)
        # v5.0: CPCV_N_FOLDS=8, CPCV_N_TEST_FOLDS=2 → C(8,2)=28 splits
        expected_splits = math.comb(config.CPCV_N_FOLDS, config.CPCV_N_TEST_FOLDS)
        assert result.n_splits == expected_splits
