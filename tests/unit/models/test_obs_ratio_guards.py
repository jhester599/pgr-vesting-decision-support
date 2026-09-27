"""
Tests for the v7.4 observation-to-feature ratio guard.

The v7.4 CPCV path-stability tests that shared this file were removed with
the CPCV diagnostic itself (pre-v200 remediation R3: CPCV is a combinatorial
K-fold, which AGENTS.md prohibits); ``test_cpcv_retired.py`` replaces them.

compute_obs_feature_ratio():
  1. test_ratio_ok
  2. test_ratio_warning
  3. test_ratio_fail_below_2
  4. test_no_features_returns_fail
  5. test_per_fold_ratio_computed_correctly
  6. test_warning_emitted_when_below_min_ratio
  7. test_no_warning_when_ok
  8. test_verdict_ok_exact_boundary
"""

from __future__ import annotations

import os
import warnings
from dataclasses import field

import numpy as np
import pandas as pd
import pytest


import config
from src.processing.feature_engineering import compute_obs_feature_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(n_obs: int, n_features: int) -> pd.DataFrame:
    """Synthetic feature DataFrame (no NaN)."""
    idx = pd.date_range("2015-01-31", periods=n_obs, freq="ME")
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(np.random.randn(n_obs, n_features), index=idx, columns=cols)


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
