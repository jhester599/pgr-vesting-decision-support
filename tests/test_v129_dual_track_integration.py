"""v129 dual-track integration smoke tests (no DB connection required)."""
from __future__ import annotations

from pathlib import Path
import pytest

from config.features import DUAL_TRACK_LEAN_BASELINE_OVERRIDES, INVESTABLE_CLASSIFIER_BENCHMARKS
from src.models.v129_feature_map import load_v128_feature_map, resolve_benchmark_features

REAL_MAP = Path("results/research/v128_benchmark_feature_map.csv")
LEAN_BASELINE = ["mom_12m", "vol_63d", "yield_slope", "real_yield_change_6m",
                 "real_rate_10y", "credit_spread_hy", "nfci", "vix",
                 "combined_ratio_ttm", "investment_income_growth_yoy",
                 "book_value_per_share_growth_yoy", "npw_growth_yoy"]


def test_bnd_dbc_vig_have_distinct_features_from_lean_baseline() -> None:
    fmap = load_v128_feature_map(REAL_MAP)
    for bm in ["BND", "DBC", "VIG"]:
        features = resolve_benchmark_features(bm, fmap, lean_baseline=LEAN_BASELINE,
                                               overrides=DUAL_TRACK_LEAN_BASELINE_OVERRIDES)
        assert features != LEAN_BASELINE, f"{bm} should have distinct features"
        assert len(features) > 0


def test_vgt_always_resolves_to_lean_baseline() -> None:
    fmap = load_v128_feature_map(REAL_MAP)
    features = resolve_benchmark_features("VGT", fmap, lean_baseline=LEAN_BASELINE,
                                           overrides=DUAL_TRACK_LEAN_BASELINE_OVERRIDES)
    assert features == LEAN_BASELINE


def test_lean_baseline_benchmarks_resolve_correctly() -> None:
    fmap = load_v128_feature_map(REAL_MAP)
    for bm in ["GLD", "VDE", "VMBS", "VOO", "VWO", "VXUS"]:
        features = resolve_benchmark_features(bm, fmap, lean_baseline=LEAN_BASELINE,
                                               overrides=DUAL_TRACK_LEAN_BASELINE_OVERRIDES)
        assert features == LEAN_BASELINE, f"{bm} should use lean_baseline"


def test_dual_track_delta_structure() -> None:
    """dual_track_delta keys must be a subset of INVESTABLE_CLASSIFIER_BENCHMARKS."""
    from src.models.classification_shadow import ClassificationShadowSummary
    import dataclasses
    fields = {f.name for f in dataclasses.fields(ClassificationShadowSummary)}
    assert "dual_track_delta" in fields


def test_benchmark_specific_columns_in_schema() -> None:
    from src.reporting.classification_artifacts import CLASSIFICATION_SHADOW_COLUMNS
    assert "benchmark_specific_features" in CLASSIFICATION_SHADOW_COLUMNS
    assert "benchmark_specific_prob_actionable_sell" in CLASSIFICATION_SHADOW_COLUMNS
    assert "benchmark_specific_tier" in CLASSIFICATION_SHADOW_COLUMNS
