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
    for ticker in INVESTABLE_CLASSIFIER_BENCHMARKS:
        assert ticker in PRIMARY_FORECAST_UNIVERSE, (
            f"{ticker} in INVESTABLE_CLASSIFIER_BENCHMARKS but not in PRIMARY_FORECAST_UNIVERSE"
        )


def test_investable_and_contextual_partition_primary_forecast() -> None:
    union = set(INVESTABLE_CLASSIFIER_BENCHMARKS) | set(CONTEXTUAL_CLASSIFIER_BENCHMARKS)
    assert union == set(PRIMARY_FORECAST_UNIVERSE), (
        f"Investable + contextual does not equal PRIMARY_FORECAST_UNIVERSE. "
        f"Diff: {union.symmetric_difference(set(PRIMARY_FORECAST_UNIVERSE))}"
    )
