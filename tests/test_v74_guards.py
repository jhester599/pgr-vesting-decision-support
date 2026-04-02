"""
Tests for v7.4 — CPCV Path Stability Guard + Obs/Feature Ratio guard.

CPCVResult stability properties (8 tests):
  1.  test_n_positive_paths_all_positive
  2.  test_n_positive_paths_none_positive
  3.  test_n_positive_paths_mixed
  4.  test_positive_path_fraction_all_positive
  5.  test_positive_path_fraction_empty
  6.  test_stability_verdict_good
  7.  test_stability_verdict_marginal
  8.  test_stability_verdict_fail
  9.  test_stability_verdict_unknown_when_empty

compute_obs_feature_ratio() (8 tests):
  10. test_ratio_ok
  11. test_ratio_warning
  12. test_ratio_fail_below_2
  13. test_no_features_returns_fail
  14. test_per_fold_ratio_computed_correctly
  15. test_warning_emitted_when_below_min_ratio
  16. test_no_warning_when_ok
  17. test_verdict_ok_exact_boundary
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import field

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.models.wfo_engine import CPCVResult
from src.processing.feature_engineering import compute_obs_feature_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpcv(path_ics: list[float]) -> CPCVResult:
    return CPCVResult(
        model_type="elasticnet",
        benchmark="VTI",
        n_splits=28,
        n_paths=len(path_ics),
        path_ics=path_ics,
        mean_ic=float(np.nanmean(path_ics)) if path_ics else float("nan"),
        ic_std=float(np.nanstd(path_ics)) if len(path_ics) > 1 else float("nan"),
        split_ics=[],
    )


def _df(n_obs: int, n_features: int) -> pd.DataFrame:
    """Synthetic feature DataFrame (no NaN)."""
    idx = pd.date_range("2015-01-31", periods=n_obs, freq="ME")
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(np.random.randn(n_obs, n_features), index=idx, columns=cols)


# ---------------------------------------------------------------------------
# CPCVResult stability property tests (1–9)
# ---------------------------------------------------------------------------

class TestCPCVStability:

    def test_n_positive_paths_all_positive(self):
        result = _cpcv([0.10, 0.05, 0.08, 0.12])
        assert result.n_positive_paths == 4

    def test_n_positive_paths_none_positive(self):
        result = _cpcv([-0.10, -0.05, -0.08, 0.0])
        # IC == 0 is not > 0
        assert result.n_positive_paths == 0

    def test_n_positive_paths_mixed(self):
        result = _cpcv([0.10, -0.05, 0.08, -0.12, 0.03])
        assert result.n_positive_paths == 3

    def test_positive_path_fraction_all_positive(self):
        result = _cpcv([0.10, 0.05, 0.08, 0.12])
        assert result.positive_path_fraction == pytest.approx(1.0)

    def test_positive_path_fraction_empty(self):
        result = _cpcv([])
        assert result.positive_path_fraction != result.positive_path_fraction  # NaN

    def test_stability_verdict_good(self):
        """19+ positive paths out of 28 → GOOD."""
        path_ics = [0.05] * config.DIAG_CPCV_MIN_POSITIVE_PATHS + [-0.01] * (28 - config.DIAG_CPCV_MIN_POSITIVE_PATHS)
        result = _cpcv(path_ics)
        assert result.stability_verdict == "GOOD"

    def test_stability_verdict_marginal(self):
        """Between floor//2 and DIAG_CPCV_MIN_POSITIVE_PATHS → MARGINAL."""
        good_thresh = config.DIAG_CPCV_MIN_POSITIVE_PATHS   # 19
        marginal_thresh = good_thresh // 2                    # 9
        # Use marginal_thresh positive paths (just at the marginal floor)
        n_pos = marginal_thresh
        path_ics = [0.05] * n_pos + [-0.01] * (28 - n_pos)
        result = _cpcv(path_ics)
        assert result.stability_verdict == "MARGINAL"

    def test_stability_verdict_fail(self):
        """Fewer than floor//2 positive paths → FAIL."""
        good_thresh = config.DIAG_CPCV_MIN_POSITIVE_PATHS   # 19
        marginal_thresh = good_thresh // 2                    # 9
        n_pos = marginal_thresh - 1                           # 8
        path_ics = [0.05] * n_pos + [-0.01] * (28 - n_pos)
        result = _cpcv(path_ics)
        assert result.stability_verdict == "FAIL"

    def test_stability_verdict_unknown_when_empty(self):
        result = _cpcv([])
        assert result.stability_verdict == "UNKNOWN"


# ---------------------------------------------------------------------------
# compute_obs_feature_ratio() tests (10–17)
# ---------------------------------------------------------------------------

class TestObsFeatureRatio:

    def test_ratio_ok(self):
        """Both full-matrix and per-fold ratios above min_ratio → verdict OK.

        With n_features=10: per_fold_ratio = WFO_TRAIN_WINDOW_MONTHS(60) / 10 = 6.0 ≥ 4.0.
        With n_obs=100: full-matrix ratio = 10.0 ≥ 4.0.
        """
        df = _df(n_obs=100, n_features=10)
        result = compute_obs_feature_ratio(df, min_ratio=4.0, warn=False)
        assert result["verdict"] == "OK"
        assert result["ratio"] == pytest.approx(100 / 10)

    def test_ratio_warning(self):
        """Ratio between 2.0 and min_ratio → verdict WARNING."""
        df = _df(n_obs=60, n_features=20)  # full-matrix ratio = 3.0
        result = compute_obs_feature_ratio(df, min_ratio=4.0, warn=False)
        assert result["verdict"] == "WARNING"

    def test_ratio_fail_below_2(self):
        """Ratio below 2.0 → verdict FAIL."""
        df = _df(n_obs=30, n_features=20)  # ratio = 1.5
        result = compute_obs_feature_ratio(df, min_ratio=4.0, warn=False)
        assert result["verdict"] == "FAIL"

    def test_no_features_returns_fail(self):
        """Empty DataFrame (no feature columns) → verdict FAIL."""
        df = pd.DataFrame(index=pd.date_range("2020-01-31", periods=10, freq="ME"))
        result = compute_obs_feature_ratio(df, warn=False)
        assert result["verdict"] == "FAIL"
        assert result["n_features"] == 0

    def test_per_fold_ratio_computed_correctly(self):
        """per_fold_ratio = WFO_TRAIN_WINDOW_MONTHS / n_features."""
        df = _df(n_obs=200, n_features=25)
        result = compute_obs_feature_ratio(df, warn=False)
        expected = config.WFO_TRAIN_WINDOW_MONTHS / 25
        assert result["per_fold_ratio"] == pytest.approx(expected)

    def test_warning_emitted_when_below_min_ratio(self):
        """UserWarning is emitted when ratio < min_ratio."""
        df = _df(n_obs=60, n_features=20)  # ratio = 3.0 < 4.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_obs_feature_ratio(df, min_ratio=4.0, warn=True)
        assert any(issubclass(w.category, UserWarning) for w in caught), (
            "Expected UserWarning for ratio below min_ratio"
        )

    def test_no_warning_when_ok(self):
        """No warning emitted when both ratios are above min_ratio.

        n_features=10: per_fold_ratio = 6.0 ≥ 4.0; n_obs=100: ratio = 10.0 ≥ 4.0.
        """
        df = _df(n_obs=100, n_features=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_obs_feature_ratio(df, min_ratio=4.0, warn=True)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_verdict_ok_exact_boundary(self):
        """Both ratios exactly at min_ratio → OK (boundary is inclusive).

        n_features=15: per_fold_ratio = 60/15 = 4.0 exactly.
        n_obs=60:       full ratio = 60/15 = 4.0 exactly.
        """
        df = _df(n_obs=60, n_features=15)
        result = compute_obs_feature_ratio(df, min_ratio=4.0, warn=False)
        assert result["ratio"] == pytest.approx(4.0)
        assert result["per_fold_ratio"] == pytest.approx(4.0)
        assert result["verdict"] == "OK"
