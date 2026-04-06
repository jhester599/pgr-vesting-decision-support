"""Tests for v32.0 — feature importance stability across WFO folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.wfo_engine import FoldResult, WFOResult
from src.research.evaluation import (
    FeatureImportanceStability,
    compute_feature_importance_stability,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fold(
    fold_idx: int,
    importances: dict[str, float],
    n: int = 10,
) -> FoldResult:
    """Construct a minimal FoldResult with the given feature importances."""
    ts = pd.Timestamp("2020-01-31") + pd.offsets.MonthEnd(fold_idx)
    return FoldResult(
        fold_idx=fold_idx,
        train_start=ts - pd.DateOffset(years=2),
        train_end=ts - pd.DateOffset(months=7),
        test_start=ts - pd.DateOffset(months=6),
        test_end=ts,
        y_true=np.zeros(n),
        y_hat=np.zeros(n),
        optimal_alpha=0.01,
        feature_importances=importances,
        n_train=24,
        n_test=n,
    )


def _make_result(folds: list[FoldResult]) -> WFOResult:
    r = WFOResult(folds=folds, benchmark="VTI", target_horizon=6, model_type="lasso")
    for fold in r.folds:
        fold._test_dates = [fold.test_end]  # type: ignore[attr-defined]
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_perfect_stability_returns_one():
    """Identical importance rankings across folds → stability score ≈ 1.0."""
    imp = {"alpha": 0.9, "beta": 0.5, "gamma": 0.2}
    folds = [_make_fold(i, imp) for i in range(4)]
    result = _make_result(folds)
    stab = compute_feature_importance_stability(result)
    assert stab is not None
    assert abs(stab.stability_score - 1.0) < 1e-9
    assert stab.verdict == "STABLE"
    assert stab.n_folds == 4


def test_reversed_ranking_returns_negative():
    """Reversed importance rankings on alternating folds → negative correlation."""
    imp_a = {"alpha": 0.9, "beta": 0.5, "gamma": 0.1}
    imp_b = {"alpha": 0.1, "beta": 0.5, "gamma": 0.9}
    folds = [_make_fold(i, imp_a if i % 2 == 0 else imp_b) for i in range(4)]
    result = _make_result(folds)
    stab = compute_feature_importance_stability(result)
    assert stab is not None
    assert stab.stability_score < 0.0


def test_single_fold_returns_none():
    """A single fold cannot produce a stability metric → None."""
    folds = [_make_fold(0, {"alpha": 0.5, "beta": 0.3})]
    result = _make_result(folds)
    assert compute_feature_importance_stability(result) is None


def test_empty_folds_returns_none():
    """No folds → None."""
    result = _make_result([])
    assert compute_feature_importance_stability(result) is None


def test_empty_importances_skipped():
    """Folds with empty importance dicts are excluded; if < 2 remain → None."""
    empty_fold = _make_fold(0, {})
    good_fold = _make_fold(1, {"alpha": 0.8, "beta": 0.4})
    result = _make_result([empty_fold, good_fold])
    # Only one fold with data → None
    assert compute_feature_importance_stability(result) is None


def test_per_feature_dataframe_shape():
    """per_feature DataFrame should have expected columns and index."""
    imp_a = {"alpha": 0.9, "beta": 0.4, "gamma": 0.1}
    imp_b = {"alpha": 0.8, "beta": 0.5, "gamma": 0.2}
    folds = [_make_fold(i, imp_a if i == 0 else imp_b) for i in range(3)]
    result = _make_result(folds)
    stab = compute_feature_importance_stability(result)
    assert stab is not None
    assert set(stab.per_feature.columns) == {"mean_rank", "rank_std", "mean_importance"}
    assert set(stab.per_feature.index) == {"alpha", "beta", "gamma"}


def test_union_of_feature_names():
    """Features absent in some folds should appear in the summary with NaN.

    We need ≥ 2 features in common between folds for the Spearman computation
    to succeed.  Here fold 0 has alpha/beta/delta and fold 1 has alpha/beta/gamma;
    the two shared features (alpha, beta) allow a rank correlation, and the
    fold-specific features (delta, gamma) appear in the union with NaN ranks.
    """
    imp_a = {"alpha": 0.9, "beta": 0.4, "delta": 0.2}
    imp_b = {"alpha": 0.7, "beta": 0.5, "gamma": 0.3}
    folds = [_make_fold(0, imp_a), _make_fold(1, imp_b)]
    result = _make_result(folds)
    stab = compute_feature_importance_stability(result)
    assert stab is not None
    assert "gamma" in stab.per_feature.index
    assert "beta" in stab.per_feature.index
    assert "delta" in stab.per_feature.index


def test_verdict_thresholds():
    """Verify STABLE / MARGINAL / UNSTABLE verdicts match documented thresholds."""
    imp_a = {"f1": 1.0, "f2": 0.8, "f3": 0.6, "f4": 0.4}
    imp_b = {"f1": 0.9, "f2": 0.7, "f3": 0.5, "f4": 0.3}
    folds = [_make_fold(i, imp_a if i % 2 == 0 else imp_b) for i in range(6)]
    result = _make_result(folds)
    stab = compute_feature_importance_stability(result)
    assert stab is not None
    # Near-identical rankings should be STABLE
    assert stab.verdict == "STABLE"
    assert stab.stability_score >= 0.7


def test_stability_score_range():
    """Stability score must be in [-1, 1]."""
    imp_a = {"a": 0.5, "b": 0.3, "c": 0.2}
    imp_b = {"a": 0.2, "b": 0.4, "c": 0.6}
    folds = [_make_fold(i, imp_a if i % 2 == 0 else imp_b) for i in range(4)]
    result = _make_result(folds)
    stab = compute_feature_importance_stability(result)
    assert stab is not None
    assert -1.0 <= stab.stability_score <= 1.0
