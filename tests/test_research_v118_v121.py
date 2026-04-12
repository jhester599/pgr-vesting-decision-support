from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.v118_utils import disagreement_label, recommendation_mode_from_hold_fraction


RESULTS_DIR = Path("results") / "research"


def test_recommendation_mode_from_hold_fraction_maps_defer_and_actionable() -> None:
    assert recommendation_mode_from_hold_fraction(0.5) == "DEFER-TO-TAX-DEFAULT"
    assert recommendation_mode_from_hold_fraction(0.0) == "ACTIONABLE"
    assert recommendation_mode_from_hold_fraction(1.0) == "ACTIONABLE"


def test_disagreement_label_identifies_veto_sell_to_defer() -> None:
    assert disagreement_label(0.0, 0.5) == "veto_sell_to_defer"
    assert disagreement_label(0.5, 0.5) == "no_change"


def test_v118_results_have_expected_columns() -> None:
    df = pd.read_csv(RESULTS_DIR / "v118_prospective_shadow_replay_results.csv")
    assert {
        "date",
        "live_hold_fraction",
        "shadow_hold_fraction",
        "would_change",
        "shadow_minus_live",
        "expanding_agreement_rate",
        "selected_variant",
    }.issubset(df.columns)


def test_v119_v121_outputs_have_expected_columns() -> None:
    v119 = pd.read_csv(RESULTS_DIR / "v119_disagreement_scorecard_results.csv")
    assert {
        "selected_variant",
        "agreement_rate",
        "disagreement_months",
        "cumulative_shadow_minus_live_all",
        "max_consecutive_disagreements",
    }.issubset(v119.columns)

    v120 = pd.read_csv(RESULTS_DIR / "v120_prospective_shadow_gate_assessment_results.csv")
    assert {
        "selected_variant",
        "decision",
        "agreement_pass",
        "uplift_pass",
        "matured_live_monitoring_n",
    }.issubset(v120.columns)

    v121 = pd.read_csv(RESULTS_DIR / "v121_prospective_shadow_phase_summary_results.csv")
    assert {
        "selected_variant",
        "agreement_rate",
        "next_step_decision",
    }.issubset(v121.columns)
