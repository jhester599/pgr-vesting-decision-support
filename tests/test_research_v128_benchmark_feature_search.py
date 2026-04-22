from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "research"
        / "v128_benchmark_feature_search.py"
    )
    spec = importlib.util.spec_from_file_location(
        "v128_benchmark_feature_search",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_candidate_feature_columns_excludes_target_column() -> None:
    feature_df, _ = MODULE.load_v128_inputs(benchmarks=["VOO"])
    features = MODULE.candidate_feature_columns(feature_df)
    assert "target_6m_return" not in features
    assert features == [
        column for column in feature_df.columns if column != "target_6m_return"
    ]
    assert len(features) > 0


def test_build_consensus_subset_is_capped_and_deterministic() -> None:
    rows = []
    for fold in range(3):
        for feature, abs_coef in (
            ("f_a", 0.50),
            ("f_b", 0.40),
            ("f_c", 0.40),
            ("f_d", 0.30),
        ):
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "abs_coef": abs_coef,
                    "selected": True,
                }
            )
    selection_df = pd.DataFrame(rows)
    subset = MODULE.build_consensus_subset(selection_df, max_features=3)
    assert subset == ["f_a", "f_b", "f_c"]


def test_passes_forward_gate_requires_real_improvement() -> None:
    reference = pd.Series(
        {
            "balanced_accuracy_covered": 0.55,
            "ece_10": 0.08,
            "brier_score": 0.20,
        }
    )
    better = pd.Series(
        {
            "balanced_accuracy_covered": 0.556,
            "ece_10": 0.09,
            "brier_score": 0.205,
        }
    )
    worse = pd.Series(
        {
            "balanced_accuracy_covered": 0.553,
            "ece_10": 0.08,
            "brier_score": 0.20,
        }
    )
    assert MODULE._passes_forward_gate(better, reference) is True
    assert MODULE._passes_forward_gate(worse, reference) is False


def test_select_benchmark_winner_falls_back_to_baseline_when_guardrails_fail() -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "benchmark": "VOO",
                "method": "lean_baseline",
                "production_eligible": True,
                "n_features": 12,
                "features": "a|b",
                "balanced_accuracy_covered": 0.60,
                "ece_10": 0.08,
                "brier_score": 0.19,
                "log_loss": 0.55,
            },
            {
                "benchmark": "VOO",
                "method": "forward_stepwise",
                "production_eligible": True,
                "n_features": 5,
                "features": "a|c",
                "balanced_accuracy_covered": 0.61,
                "ece_10": 0.20,
                "brier_score": 0.25,
                "log_loss": 0.54,
            },
        ]
    )
    winners = MODULE.select_benchmark_winner(comparison_df)
    assert winners.iloc[0]["method"] == "lean_baseline"


@pytest.mark.slow
def test_run_feature_search_smoke_covers_all_benchmarks_with_small_subset() -> None:
    feature_df, _ = MODULE.load_v128_inputs()
    candidate_features = MODULE.candidate_feature_columns(feature_df)[:2]
    benchmarks = MODULE.benchmark_universe()[:3]
    artifacts = MODULE.run_feature_search(
        benchmarks=benchmarks,
        candidate_features=candidate_features,
        max_features=2,
        write_outputs=False,
    )
    feature_map = artifacts["feature_map"]
    assert set(feature_map["benchmark"]) == set(benchmarks)
    assert feature_map["n_features"].max() <= 12
    assert not artifacts["single_feature"].empty
    assert not artifacts["comparison"].empty
