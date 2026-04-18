"""Tests for the v156 USD index momentum evaluation harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_usd_features_in_registry() -> None:
    from src.research.v87_utils import available_feature_families
    import pandas as pd
    stub_cols = ["usd_broad_return_3m", "usd_momentum_6m", "yield_slope", "vix", "nfci", "mom_12m"]
    stub_df = pd.DataFrame({c: [0.0] for c in stub_cols})
    families = available_feature_families(stub_df)
    macro = families.get("macro_rates_spreads", [])
    assert "usd_broad_return_3m" in macro
    assert "usd_momentum_6m" in macro


def test_usd_target_benchmarks_constant() -> None:
    from results.research.v156_usd_momentum_eval import TARGET_BENCHMARKS
    for bm in ["BND", "VXUS", "VWO"]:
        assert bm in TARGET_BENCHMARKS


def test_build_usd_augmented_cols_no_duplicates() -> None:
    from results.research.v156_usd_momentum_eval import build_augmented_feature_cols
    base = ["a", "b", "usd_broad_return_3m"]
    result = build_augmented_feature_cols(base, extra=["usd_broad_return_3m", "usd_momentum_6m"])
    assert result.count("usd_broad_return_3m") == 1
    assert "usd_momentum_6m" in result


@pytest.mark.slow
def test_run_usd_evaluation_writes_candidate(tmp_path: Path) -> None:
    from results.research.v156_usd_momentum_eval import run_usd_evaluation
    candidate_path = tmp_path / "v156_usd_candidate.json"
    result = run_usd_evaluation(benchmarks=["BND"], candidate_path=candidate_path)
    assert candidate_path.exists()
    assert "usd_winners" in result


def test_candidate_file_schema() -> None:
    import json
    from results.research.v156_usd_momentum_eval import DEFAULT_CANDIDATE_PATH
    if not DEFAULT_CANDIDATE_PATH.exists():
        pytest.skip("candidate file not yet generated — run harness first")
    data = json.loads(DEFAULT_CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert "usd_winners" in data
    assert "recommendation" in data
