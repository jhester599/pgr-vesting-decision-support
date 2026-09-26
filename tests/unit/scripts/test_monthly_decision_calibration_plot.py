"""Tests for the calibration reliability diagram in scripts/monthly_decision.py (v6.5, P2.7).

 1.  _plot_calibration_curve returns None when fewer than 4 observations
 2.  _plot_calibration_curve returns None when method is "uncalibrated"
 3.  _plot_calibration_curve writes a PNG file when data is sufficient

Split out of the old ``tests/test_v65_p26_p27_p28.py`` (review 2026-09-25, step 11).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np


class TestPlotCalibrationCurve:
    """Tests for _plot_calibration_curve in monthly_decision.py."""

    def _import(self):
        """Import _plot_calibration_curve lazily (avoids matplotlib import at collection time)."""
        import importlib
        # monthly_decision imports config and src modules; mock heavy dependencies
        with patch.dict("os.environ", {"AV_API_KEY": "test", "FMP_API_KEY": "test",
                                        "FRED_API_KEY": "test"}):
            mod = importlib.import_module("scripts.monthly_decision")
        return mod._plot_calibration_curve

    def test_returns_none_when_fewer_than_4_obs(self, tmp_path):
        plot_fn = self._import()
        from src.models.calibration import CalibrationResult
        cal = CalibrationResult(n_obs=3, method="platt", ece=0.05,
                                ece_ci_lower=0.02, ece_ci_upper=0.10)
        result = plot_fn(tmp_path, np.array([0.6, 0.7, 0.8]), np.array([1, 0, 1]), cal)
        assert result is None

    def test_returns_none_when_uncalibrated(self, tmp_path):
        plot_fn = self._import()
        from src.models.calibration import CalibrationResult
        cal = CalibrationResult(n_obs=50, method="uncalibrated", ece=0.05,
                                ece_ci_lower=0.02, ece_ci_upper=0.10)
        probs = np.random.default_rng(42).uniform(0.3, 0.8, 50)
        outcomes = (probs > 0.5).astype(int)
        result = plot_fn(tmp_path, probs, outcomes, cal)
        assert result is None

    def test_writes_png_when_sufficient_data(self, tmp_path):
        plot_fn = self._import()
        from src.models.calibration import CalibrationResult
        cal = CalibrationResult(n_obs=100, method="platt", ece=0.03,
                                ece_ci_lower=0.01, ece_ci_upper=0.07)
        rng = np.random.default_rng(0)
        probs = rng.uniform(0.2, 0.9, 100)
        outcomes = (probs > 0.5).astype(int)
        save_path = plot_fn(tmp_path, probs, outcomes, cal)
        assert save_path is not None
        assert Path(save_path).exists()
        assert Path(save_path).suffix == ".png"
