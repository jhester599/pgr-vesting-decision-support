"""Tests for v32.2+v32.3 — decision policy backtest in monthly report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.monthly_decision as md
from src.models.wfo_engine import FoldResult, WFOResult
from src.models.multi_benchmark_wfo import EnsembleWFOResult
from src.research.policy_metrics import FIXED_POLICIES, SIGNAL_POLICIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fold(
    fold_idx: int,
    y_hat: list[float],
    y_true: list[float],
) -> FoldResult:
    n = len(y_hat)
    ts = pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(fold_idx * 6)
    fold = FoldResult(
        fold_idx=fold_idx,
        train_start=ts - pd.DateOffset(years=2),
        train_end=ts - pd.DateOffset(months=7),
        test_start=ts - pd.DateOffset(months=6),
        test_end=ts,
        y_true=np.array(y_true),
        y_hat=np.array(y_hat),
        optimal_alpha=0.01,
        feature_importances={"f1": 0.5, "f2": 0.3},
        n_train=24,
        n_test=n,
    )
    fold._test_dates = [ts - pd.DateOffset(months=i) for i in range(n)]  # type: ignore[attr-defined]
    return fold


def _make_ensemble(y_hat: list[float], y_true: list[float]) -> EnsembleWFOResult:
    """Build a minimal EnsembleWFOResult with one elasticnet model."""
    fold = _make_fold(0, y_hat, y_true)
    wfo = WFOResult(
        folds=[fold], benchmark="VTI", target_horizon=6, model_type="elasticnet"
    )
    return EnsembleWFOResult(
        benchmark="VTI",
        target_horizon=6,
        mean_ic=0.1,
        mean_hit_rate=0.6,
        mean_mae=0.05,
        model_results={"elasticnet": wfo},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_policy_summary_returns_all_policies():
    """All fixed and signal policies should be present in the result."""
    y_hat = [0.05, -0.02, 0.08, -0.04, 0.03, 0.01, -0.06, 0.09]
    y_true = [0.04, -0.03, 0.07, -0.05, 0.02, -0.01, 0.03, 0.06]
    ensemble = _make_ensemble(y_hat, y_true)
    result = md._compute_policy_summary({"VTI": ensemble})
    assert result is not None
    for p in list(FIXED_POLICIES) + list(SIGNAL_POLICIES):
        assert p in result, f"Missing policy: {p}"


def test_compute_policy_summary_returns_none_for_insufficient_data():
    """Fewer than 4 OOS observations → None."""
    y_hat = [0.05, -0.02, 0.08]
    y_true = [0.04, -0.03, 0.07]
    ensemble = _make_ensemble(y_hat, y_true)
    result = md._compute_policy_summary({"VTI": ensemble})
    assert result is None


def test_always_sell_all_mean_return_is_zero():
    """'always_sell_100' always holds 0% → mean return should be 0."""
    y_hat = [0.05, -0.02, 0.08, -0.04]
    y_true = [0.04, -0.03, 0.07, -0.05]
    ensemble = _make_ensemble(y_hat, y_true)
    result = md._compute_policy_summary({"VTI": ensemble})
    assert result is not None
    assert abs(result["always_sell_100"].mean_policy_return) < 1e-9


def test_always_hold_all_mean_return_equals_mean_realized():
    """'always_hold_100' holds everything → mean return = mean realized return."""
    y_hat = [0.05, -0.02, 0.08, -0.04, 0.03]
    y_true = [0.04, -0.03, 0.07, -0.05, 0.02]
    ensemble = _make_ensemble(y_hat, y_true)
    result = md._compute_policy_summary({"VTI": ensemble})
    assert result is not None
    expected = float(np.mean(y_true))
    assert abs(result["always_hold_100"].mean_policy_return - expected) < 1e-9


def test_policy_backtest_section_appears_in_recommendation(tmp_path: Path):
    """recommendation.md should contain the Decision Policy Backtest section."""
    y_hat = [0.05, -0.02, 0.08, -0.04, 0.03, 0.01, -0.06, 0.09]
    y_true = [0.04, -0.03, 0.07, -0.05, 0.02, -0.01, 0.03, 0.06]
    ensemble = _make_ensemble(y_hat, y_true)
    policy_summary = md._compute_policy_summary({"VTI": ensemble})
    assert policy_summary is not None

    # Build minimal inputs required by _write_recommendation_md
    signals = pd.DataFrame({
        "signal": ["OUTPERFORM"],
        "predicted_relative_return": [0.04],
        "confidence_tier": ["MODERATE"],
        "prob_outperform": [0.6],
    }, index=["VTI"])
    from datetime import date
    from unittest.mock import MagicMock

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    conn.execute.return_value.fetchone.return_value = None

    md._write_recommendation_md(
        tmp_path,
        as_of=date(2026, 4, 1),
        run_date=date(2026, 4, 6),
        conn=conn,
        signals=signals,
        consensus="OUTPERFORM",
        mean_predicted=0.04,
        mean_ic=0.10,
        mean_hr=0.60,
        sell_pct=0.0,
        dry_run=True,
        policy_summary=policy_summary,
        freshness_report={"warnings": []},
        recommendation_mode={
            "mode": "monitoring-only",
            "label": "Monitoring Only",
            "sell_pct": 0.5,
            "summary": "Test stub summary.",
        },
    )

    rec_path = tmp_path / "recommendation.md"
    assert rec_path.exists()
    content = rec_path.read_text(encoding="utf-8")
    assert "Decision Policy Backtest" in content
    assert "Fixed Heuristic Baselines" in content
    assert "Model-Driven Policies vs. Heuristics" in content
    # Uplift columns should be present
    assert "Uplift vs Sell-All" in content or "uplift" in content.lower()


def test_policy_backtest_section_absent_when_none(tmp_path: Path):
    """If policy_summary is None, the section should not appear."""
    signals = pd.DataFrame({
        "signal": ["OUTPERFORM"],
        "predicted_relative_return": [0.04],
        "confidence_tier": ["MODERATE"],
        "prob_outperform": [0.6],
    }, index=["VTI"])
    from datetime import date
    from unittest.mock import MagicMock

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    conn.execute.return_value.fetchone.return_value = None

    md._write_recommendation_md(
        tmp_path,
        as_of=date(2026, 4, 1),
        run_date=date(2026, 4, 6),
        conn=conn,
        signals=signals,
        consensus="OUTPERFORM",
        mean_predicted=0.04,
        mean_ic=0.10,
        mean_hr=0.60,
        sell_pct=0.0,
        dry_run=True,
        policy_summary=None,
        freshness_report={"warnings": []},
        recommendation_mode={
            "mode": "monitoring-only",
            "label": "Monitoring Only",
            "sell_pct": 0.5,
            "summary": "Test stub summary.",
        },
    )

    rec_path = tmp_path / "recommendation.md"
    assert rec_path.exists()
    content = rec_path.read_text(encoding="utf-8")
    assert "Decision Policy Backtest" not in content
