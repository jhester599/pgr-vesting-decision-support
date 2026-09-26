"""Tests for classification portfolio-alignment config constants."""
import pytest
from config.features import (
    ETF_BENCHMARK_UNIVERSE,
    INVESTABLE_CLASSIFIER_BENCHMARKS,
    INVESTABLE_CLASSIFIER_BASE_WEIGHTS,
    CONTEXTUAL_CLASSIFIER_BENCHMARKS,
    PRIMARY_FORECAST_UNIVERSE,
)


def test_investable_benchmarks_subset_of_universe() -> None:
    for ticker in INVESTABLE_CLASSIFIER_BENCHMARKS:
        assert ticker in ETF_BENCHMARK_UNIVERSE, (
            f"{ticker} in INVESTABLE_CLASSIFIER_BENCHMARKS but not in ETF_BENCHMARK_UNIVERSE"
        )


def test_investable_base_weights_keys_match_benchmarks() -> None:
    assert set(INVESTABLE_CLASSIFIER_BASE_WEIGHTS.keys()) == set(INVESTABLE_CLASSIFIER_BENCHMARKS)


def test_investable_base_weights_sum_to_one() -> None:
    total = sum(INVESTABLE_CLASSIFIER_BASE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


def test_investable_base_weights_all_positive() -> None:
    for ticker, w in INVESTABLE_CLASSIFIER_BASE_WEIGHTS.items():
        assert w > 0, f"Weight for {ticker} is {w}, expected positive"


def test_contextual_no_overlap_with_investable() -> None:
    overlap = set(CONTEXTUAL_CLASSIFIER_BENCHMARKS) & set(INVESTABLE_CLASSIFIER_BENCHMARKS)
    assert overlap == set(), f"Overlap between investable and contextual: {overlap}"


def test_contextual_benchmarks_subset_of_primary_forecast() -> None:
    for ticker in CONTEXTUAL_CLASSIFIER_BENCHMARKS:
        assert ticker in PRIMARY_FORECAST_UNIVERSE, (
            f"{ticker} in CONTEXTUAL_CLASSIFIER_BENCHMARKS but not in PRIMARY_FORECAST_UNIVERSE"
        )


def test_investable_benchmarks_subset_of_primary_forecast() -> None:
    # v124: VGT and VIG are investable classifiers but intentionally outside the
    # 8-benchmark WFO primary ensemble (PRIMARY_FORECAST_UNIVERSE). The subset
    # check is relaxed to exclude those two tickers; all others must remain present.
    v124_investable_only = {"VGT", "VIG"}
    for ticker in INVESTABLE_CLASSIFIER_BENCHMARKS:
        if ticker in v124_investable_only:
            continue
        assert ticker in PRIMARY_FORECAST_UNIVERSE, (
            f"{ticker} in INVESTABLE_CLASSIFIER_BENCHMARKS but not in PRIMARY_FORECAST_UNIVERSE"
        )


def test_investable_and_contextual_partition_primary_forecast() -> None:
    # v124: VGT and VIG expanded the investable set beyond PRIMARY_FORECAST_UNIVERSE.
    # The invariant is now: PRIMARY_FORECAST_UNIVERSE ⊆ investable ∪ contextual
    # (the union is a superset of PRIMARY_FORECAST_UNIVERSE, not an exact partition).
    primary = set(PRIMARY_FORECAST_UNIVERSE)
    union = set(INVESTABLE_CLASSIFIER_BENCHMARKS) | set(CONTEXTUAL_CLASSIFIER_BENCHMARKS)
    assert primary.issubset(union), (
        f"PRIMARY_FORECAST_UNIVERSE not covered by investable + contextual. "
        f"Missing: {primary - union}"
    )


def test_vgt_in_investable_benchmarks() -> None:
    assert "VGT" in INVESTABLE_CLASSIFIER_BENCHMARKS


def test_vig_in_investable_benchmarks() -> None:
    assert "VIG" in INVESTABLE_CLASSIFIER_BENCHMARKS


def test_v124_weights_include_vgt_and_vig() -> None:
    assert "VGT" in INVESTABLE_CLASSIFIER_BASE_WEIGHTS
    assert "VIG" in INVESTABLE_CLASSIFIER_BASE_WEIGHTS


def test_v124_weights_sum_to_one() -> None:
    total = sum(INVESTABLE_CLASSIFIER_BASE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6


def test_v124_vgt_weight() -> None:
    assert abs(INVESTABLE_CLASSIFIER_BASE_WEIGHTS["VGT"] - 0.20) < 1e-6


def test_v124_vig_weight() -> None:
    assert abs(INVESTABLE_CLASSIFIER_BASE_WEIGHTS["VIG"] - 0.15) < 1e-6


def test_v128_feature_map_path_constant_exists() -> None:
    from config import features as f
    assert hasattr(f, "V128_BENCHMARK_FEATURE_MAP_PATH")
    assert str(f.V128_BENCHMARK_FEATURE_MAP_PATH).endswith("v128_benchmark_feature_map.csv")


def test_dual_track_lean_baseline_overrides_contains_vgt() -> None:
    from config import features as f
    assert hasattr(f, "DUAL_TRACK_LEAN_BASELINE_OVERRIDES")
    assert "VGT" in f.DUAL_TRACK_LEAN_BASELINE_OVERRIDES
