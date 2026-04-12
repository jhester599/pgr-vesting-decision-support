from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results") / "research"


def test_v110_results_have_expected_columns() -> None:
    df = pd.read_csv(RESULTS_DIR / "v110_gemini_veto_gate_results.csv")
    assert {
        "variant",
        "gate_style",
        "threshold",
        "mean_policy_return",
        "agreement_with_regression_rate",
        "unnecessary_action_rate",
    }.issubset(df.columns)


def test_v111_results_have_expected_columns() -> None:
    df = pd.read_csv(RESULTS_DIR / "v111_permission_overlay_results.csv")
    assert {
        "variant",
        "gate_style",
        "threshold",
        "mean_policy_return",
        "agreement_with_regression_rate",
        "action_month_rate",
    }.issubset(df.columns)


def test_v112_results_have_expected_columns() -> None:
    df = pd.read_csv(RESULTS_DIR / "v112_target_reformulation_results.csv")
    assert {
        "variant",
        "gate_style",
        "threshold",
        "mean_policy_return",
        "agreement_with_regression_rate",
    }.issubset(df.columns)


def test_v113_results_have_promotion_flags() -> None:
    df = pd.read_csv(RESULTS_DIR / "v113_constrained_candidate_selection_results.csv")
    assert {
        "variant",
        "source_version",
        "gate_style",
        "uplift_vs_regression",
        "promotion_eligible",
    }.issubset(df.columns)


def test_v114_v117_summary_outputs_exist() -> None:
    expected = [
        RESULTS_DIR / "v114_shadow_gate_overlay_summary_results.csv",
        RESULTS_DIR / "v115_classifier_monitoring_summary_results.csv",
        RESULTS_DIR / "v116_limited_gate_candidate_results.csv",
        RESULTS_DIR / "v117_primary_mode_selector_evaluation_results.csv",
    ]
    for path in expected:
        assert path.exists()
