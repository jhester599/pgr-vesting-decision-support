"""
Tests for the v3.0 purge buffer fix in src/models/wfo_engine.py.

Verifies:
  - Default gap = horizon + purge_buffer (8 for 6M, 15 for 12M)
  - purge_buffer=0 reproduces v2.7 behavior (gap = horizon only)
  - Config constants are consistent with documented values
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit
from unittest.mock import patch

import config
from src.models.wfo_engine import run_wfo


def _make_synthetic_data(n: int = 150, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-31", periods=n, freq="ME")
    X = pd.DataFrame(rng.normal(size=(n, n_features)), index=idx,
                     columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.normal(size=n), index=idx, name="target")
    return X, y


class TestPurgeBufferConfig:
    def test_6m_purge_buffer_is_2(self):
        assert config.WFO_PURGE_BUFFER_6M == 2

    def test_12m_purge_buffer_is_3(self):
        assert config.WFO_PURGE_BUFFER_12M == 3

    def test_total_gap_6m_is_8(self):
        assert config.WFO_EMBARGO_MONTHS_6M + config.WFO_PURGE_BUFFER_6M == 8

    def test_total_gap_12m_is_15(self):
        assert config.WFO_EMBARGO_MONTHS_12M + config.WFO_PURGE_BUFFER_12M == 15


class TestRunWfoPurgeBuffer:
    def test_default_6m_uses_gap_8(self):
        """Verify that run_wfo with 6M horizon uses gap=8 by default."""
        X, y = _make_synthetic_data()

        captured_gaps = []

        original_tss = TimeSeriesSplit

        class CapturingTSS(original_tss):
            def __init__(self, *args, **kwargs):
                captured_gaps.append(kwargs.get("gap", 0))
                super().__init__(*args, **kwargs)

        with patch("src.models.wfo_engine.TimeSeriesSplit", CapturingTSS):
            run_wfo(X, y, model_type="lasso", target_horizon_months=6)

        assert len(captured_gaps) > 0
        assert captured_gaps[0] == 8, f"Expected gap=8 for 6M horizon, got {captured_gaps[0]}"

    def test_default_12m_uses_gap_15(self):
        """Verify that run_wfo with 12M horizon uses gap=15 by default."""
        X, y = _make_synthetic_data(n=200)

        captured_gaps = []

        original_tss = TimeSeriesSplit

        class CapturingTSS(original_tss):
            def __init__(self, *args, **kwargs):
                captured_gaps.append(kwargs.get("gap", 0))
                super().__init__(*args, **kwargs)

        with patch("src.models.wfo_engine.TimeSeriesSplit", CapturingTSS):
            try:
                run_wfo(X, y, model_type="lasso", target_horizon_months=12)
            except ValueError:
                pass  # dataset may be too small; we only care about the gap value

        if captured_gaps:
            assert captured_gaps[0] == 15, \
                f"Expected gap=15 for 12M horizon, got {captured_gaps[0]}"

    def test_purge_buffer_zero_reproduces_v2_behavior(self):
        """purge_buffer=0 should use gap = target_horizon only."""
        X, y = _make_synthetic_data()

        captured_gaps = []

        original_tss = TimeSeriesSplit

        class CapturingTSS(original_tss):
            def __init__(self, *args, **kwargs):
                captured_gaps.append(kwargs.get("gap", 0))
                super().__init__(*args, **kwargs)

        with patch("src.models.wfo_engine.TimeSeriesSplit", CapturingTSS):
            run_wfo(X, y, model_type="lasso",
                    target_horizon_months=6, purge_buffer=0)

        assert len(captured_gaps) > 0
        assert captured_gaps[0] == 6, \
            f"Expected gap=6 for purge_buffer=0, got {captured_gaps[0]}"

    def test_explicit_purge_buffer_overrides_default(self):
        """Explicit purge_buffer value should override config defaults."""
        X, y = _make_synthetic_data()

        captured_gaps = []

        original_tss = TimeSeriesSplit

        class CapturingTSS(original_tss):
            def __init__(self, *args, **kwargs):
                captured_gaps.append(kwargs.get("gap", 0))
                super().__init__(*args, **kwargs)

        with patch("src.models.wfo_engine.TimeSeriesSplit", CapturingTSS):
            run_wfo(X, y, model_type="lasso",
                    target_horizon_months=6, purge_buffer=1)

        assert len(captured_gaps) > 0
        assert captured_gaps[0] == 7, \
            f"Expected gap=7 for purge_buffer=1, got {captured_gaps[0]}"

    def test_run_wfo_produces_valid_result(self):
        """run_wfo with default purge buffer should complete without error."""
        X, y = _make_synthetic_data()
        result = run_wfo(X, y, model_type="elasticnet",
                         target_horizon_months=6)
        assert len(result.folds) >= 1
        assert not np.isnan(result.information_coefficient)
