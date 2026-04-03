"""
Tests for the v4.3.1 diagnostic OOS evaluation report.

Validates:
  - _flag() threshold logic (✅ / ⚠️ / ❌)
  - _write_diagnostic_report() file creation and content
  - Aggregate OOS R² and Newey-West IC integration
  - Empty / insufficient-data guard paths
  - config.py diagnostic threshold constants
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import config
from src.models.wfo_engine import CPCVResult

# ---------------------------------------------------------------------------
# Import the functions under test (path manipulation matches project layout)
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.monthly_decision import _flag, _write_diagnostic_report


# ---------------------------------------------------------------------------
# Minimal WFOResult / EnsembleWFOResult stubs
# ---------------------------------------------------------------------------

@dataclass
class _FoldStub:
    y_true: np.ndarray
    y_hat: np.ndarray
    _test_dates: list[pd.Timestamp] = field(default_factory=list)


@dataclass
class _WFOResultStub:
    """Minimal WFOResult-compatible stub for testing."""

    folds: list[_FoldStub] = field(default_factory=list)
    benchmark: str = ""
    target_horizon: int = 6
    model_type: str = "elasticnet"

    @property
    def y_true_all(self) -> np.ndarray:
        return np.concatenate([f.y_true for f in self.folds]) if self.folds else np.array([])

    @property
    def y_hat_all(self) -> np.ndarray:
        return np.concatenate([f.y_hat for f in self.folds]) if self.folds else np.array([])

    @property
    def test_dates_all(self) -> pd.DatetimeIndex:
        all_dates: list[pd.Timestamp] = []
        for fold in self.folds:
            all_dates.extend(fold._test_dates)
        return pd.DatetimeIndex(all_dates)


@dataclass
class _EnsembleWFOResultStub:
    model_results: dict[str, _WFOResultStub] = field(default_factory=dict)


def _make_ensemble_results(
    n_obs: int = 40,
    ic_level: float = 0.5,
    benchmarks: list[str] | None = None,
) -> dict[str, _EnsembleWFOResultStub]:
    """
    Build synthetic ensemble_results dict with controllable IC level.

    Simulates n_obs monthly OOS predictions for each benchmark.  A higher
    ``ic_level`` (0–1) shifts the y_hat closer to y_true (better signal).
    """
    rng = np.random.default_rng(42)
    if benchmarks is None:
        benchmarks = ["VOO", "VTI"]

    results: dict[str, _EnsembleWFOResultStub] = {}
    dates = pd.date_range("2019-01-31", periods=n_obs, freq="ME")

    for etf in benchmarks:
        y_true = rng.normal(0.0, 0.10, size=n_obs)
        noise = rng.normal(0.0, 0.10, size=n_obs)
        y_hat = ic_level * y_true + (1 - ic_level) * noise

        fold = _FoldStub(
            y_true=y_true,
            y_hat=y_hat,
            _test_dates=list(dates),
        )
        wfo = _WFOResultStub(folds=[fold], benchmark=etf, model_type="elasticnet")
        results[etf] = _EnsembleWFOResultStub(model_results={"elasticnet": wfo})

    return results


# ===========================================================================
# Tests for _flag()
# ===========================================================================

class TestFlag:
    """_flag(value, good, marginal, higher_is_better) → ✅ / ⚠️ / ❌"""

    def test_higher_is_better_good(self) -> None:
        assert _flag(0.10, good=0.07, marginal=0.03) == "✅"

    def test_higher_is_better_marginal(self) -> None:
        assert _flag(0.05, good=0.07, marginal=0.03) == "⚠️"

    def test_higher_is_better_fail(self) -> None:
        assert _flag(0.01, good=0.07, marginal=0.03) == "❌"

    def test_higher_is_better_exactly_good_threshold(self) -> None:
        assert _flag(0.07, good=0.07, marginal=0.03) == "✅"

    def test_higher_is_better_exactly_marginal_threshold(self) -> None:
        assert _flag(0.03, good=0.07, marginal=0.03) == "⚠️"

    def test_lower_is_better_good(self) -> None:
        assert _flag(0.01, good=0.05, marginal=0.10, higher_is_better=False) == "✅"

    def test_lower_is_better_marginal(self) -> None:
        assert _flag(0.07, good=0.05, marginal=0.10, higher_is_better=False) == "⚠️"

    def test_lower_is_better_fail(self) -> None:
        assert _flag(0.15, good=0.05, marginal=0.10, higher_is_better=False) == "❌"

    def test_oos_r2_good(self) -> None:
        """OOS R² of 3% exceeds the 2% good threshold."""
        assert _flag(0.03, good=config.DIAG_MIN_OOS_R2, marginal=0.005) == "✅"

    def test_oos_r2_marginal(self) -> None:
        """OOS R² of 1% is between 0.5% marginal and 2% good."""
        assert _flag(0.01, good=config.DIAG_MIN_OOS_R2, marginal=0.005) == "⚠️"

    def test_oos_r2_fail(self) -> None:
        """Negative OOS R² is failing."""
        assert _flag(-0.01, good=config.DIAG_MIN_OOS_R2, marginal=0.005) == "❌"

    def test_hit_rate_good(self) -> None:
        assert _flag(0.60, good=config.DIAG_MIN_HIT_RATE, marginal=0.52) == "✅"

    def test_hit_rate_fail(self) -> None:
        assert _flag(0.48, good=config.DIAG_MIN_HIT_RATE, marginal=0.52) == "❌"


# ===========================================================================
# Tests for config constants
# ===========================================================================

class TestConfigConstants:
    """Verify the v4.3.1 diagnostic constants are correctly defined."""

    def test_diag_min_oos_r2(self) -> None:
        assert config.DIAG_MIN_OOS_R2 == pytest.approx(0.02)

    def test_diag_min_ic(self) -> None:
        assert config.DIAG_MIN_IC == pytest.approx(0.07)

    def test_diag_min_hit_rate(self) -> None:
        assert config.DIAG_MIN_HIT_RATE == pytest.approx(0.55)

    def test_diag_cpcv_min_positive_paths(self) -> None:
        # v5.0: upgraded to 19 (≥19/28 ≈ 67% positive paths; was 10/15 under C(6,2))
        assert config.DIAG_CPCV_MIN_POSITIVE_PATHS == 19

    def test_constants_are_numeric(self) -> None:
        for attr in (
            "DIAG_MIN_OOS_R2",
            "DIAG_MIN_IC",
            "DIAG_MIN_HIT_RATE",
        ):
            assert isinstance(getattr(config, attr), float), f"{attr} should be float"

    def test_cpcv_threshold_is_int(self) -> None:
        assert isinstance(config.DIAG_CPCV_MIN_POSITIVE_PATHS, int)


# ===========================================================================
# Tests for _write_diagnostic_report()
# ===========================================================================

class TestWriteDiagnosticReport:
    """Integration tests for the diagnostic report writer."""

    def test_creates_diagnostic_md(self, tmp_path: Path) -> None:
        """Report file is created when ensemble_results contains valid data."""
        ensemble = _make_ensemble_results(n_obs=40)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        assert (tmp_path / "diagnostic.md").exists()

    def test_report_contains_month_header(self, tmp_path: Path) -> None:
        ensemble = _make_ensemble_results(n_obs=40)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "March 2026" in content

    def test_report_contains_oos_r2(self, tmp_path: Path) -> None:
        ensemble = _make_ensemble_results(n_obs=40)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "OOS R²" in content

    def test_report_contains_newey_west_ic(self, tmp_path: Path) -> None:
        ensemble = _make_ensemble_results(n_obs=40)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "Newey-West" in content

    def test_report_contains_per_benchmark_table(self, tmp_path: Path) -> None:
        ensemble = _make_ensemble_results(n_obs=40, benchmarks=["VOO", "VTI", "VGT"])
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "VOO" in content
        assert "VTI" in content
        assert "VGT" in content

    def test_report_mentions_cpcv_deferred(self, tmp_path: Path) -> None:
        ensemble = _make_ensemble_results(n_obs=40)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "Phase 1" in content or "N/A" in content

    def test_report_includes_runtime_governance_metrics_when_provided(self, tmp_path: Path) -> None:
        ensemble = _make_ensemble_results(n_obs=40)
        cpcv_result = CPCVResult(
            model_type="elasticnet",
            benchmark="VTI",
            n_splits=28,
            n_paths=28,
            path_ics=[0.05] * 20 + [-0.01] * 8,
            mean_ic=0.032,
            ic_std=0.041,
            split_ics=[],
        )
        obs_feature_report = {
            "n_obs": 120,
            "n_features": 15,
            "ratio": 8.0,
            "per_fold_ratio": 4.0,
            "verdict": "OK",
            "message": "obs/feature ratio healthy.",
        }
        _write_diagnostic_report(
            tmp_path,
            date(2026, 3, 26),
            ensemble,
            obs_feature_report=obs_feature_report,
            representative_cpcv=cpcv_result,
        )
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "Feature Governance" in content
        assert "20/28" in content
        assert "Representative CPCV" in content

    def test_report_contains_threshold_reference(self, tmp_path: Path) -> None:
        ensemble = _make_ensemble_results(n_obs=40)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "Campbell" in content

    def test_good_signal_shows_green_flag(self, tmp_path: Path) -> None:
        """High ic_level should produce ✅ flags in the per-benchmark table."""
        ensemble = _make_ensemble_results(n_obs=80, ic_level=0.9)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "✅" in content

    def test_poor_signal_shows_red_flag(self, tmp_path: Path) -> None:
        """Near-zero ic_level should produce ❌ flags in the per-benchmark table."""
        ensemble = _make_ensemble_results(n_obs=40, ic_level=0.0)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "❌" in content

    def test_insufficient_data_guard(self, tmp_path: Path) -> None:
        """With only 2 OOS observations the function writes a minimal report."""
        ensemble = _make_ensemble_results(n_obs=2)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        path = tmp_path / "diagnostic.md"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        # Minimal report should still have the month header
        assert "March 2026" in content

    def test_empty_ensemble_results(self, tmp_path: Path) -> None:
        """Empty ensemble dict writes a minimal report without crashing."""
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), {})
        assert (tmp_path / "diagnostic.md").exists()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Output dir is created if it doesn't exist."""
        nested = tmp_path / "nested" / "dir"
        ensemble = _make_ensemble_results(n_obs=20)
        _write_diagnostic_report(nested, date(2026, 3, 26), ensemble)
        assert (nested / "diagnostic.md").exists()

    def test_horizon_affects_nw_lags(self, tmp_path: Path) -> None:
        """12M horizon should show 11 Newey-West lags in the report."""
        ensemble = _make_ensemble_results(n_obs=50)
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble, target_horizon_months=12)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        assert "11" in content  # 12 - 1 = 11 lags

    def test_elasticnet_preferred_over_ridge(self, tmp_path: Path) -> None:
        """When both elasticnet and ridge results exist, elasticnet is used."""
        rng = np.random.default_rng(7)
        dates = pd.date_range("2020-01-31", periods=30, freq="ME")
        y_true = rng.normal(0, 0.1, 30)
        # elasticnet: high IC;  ridge: near-zero IC
        fold_en = _FoldStub(y_true=y_true, y_hat=0.8 * y_true, _test_dates=list(dates))
        fold_ri = _FoldStub(y_true=y_true, y_hat=rng.normal(0, 0.1, 30), _test_dates=list(dates))
        ens = _EnsembleWFOResultStub(model_results={
            "elasticnet": _WFOResultStub(folds=[fold_en], benchmark="VOO"),
            "ridge":      _WFOResultStub(folds=[fold_ri], benchmark="VOO"),
        })
        ensemble = {"VOO": ens}
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        content = (tmp_path / "diagnostic.md").read_text(encoding="utf-8")
        # High-IC elasticnet path should produce a ✅ flag
        assert "✅" in content

    def test_fallback_to_first_model_when_no_elasticnet(self, tmp_path: Path) -> None:
        """Falls back to first model result when elasticnet key is absent."""
        rng = np.random.default_rng(9)
        dates = pd.date_range("2020-01-31", periods=30, freq="ME")
        y_true = rng.normal(0, 0.1, 30)
        fold = _FoldStub(y_true=y_true, y_hat=0.7 * y_true, _test_dates=list(dates))
        ens = _EnsembleWFOResultStub(model_results={
            "lasso": _WFOResultStub(folds=[fold], benchmark="VTI"),
        })
        ensemble = {"VTI": ens}
        _write_diagnostic_report(tmp_path, date(2026, 3, 26), ensemble)
        assert (tmp_path / "diagnostic.md").exists()
