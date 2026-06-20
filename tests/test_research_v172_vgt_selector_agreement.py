"""Tests for v172 VGT selector-agreement gate logic.

Covers:
- run_regularized_gate: gate_passed logic, candidate agreement, insufficient data
- determine_synthesis_verdict: CONDITIONAL_SHADOW vs REJECT paths and boundary cases
- Pipeline builders: correct solver, penalty, and C value plumbing
- V129_RESULTS / AS_OF_DATES / VGT_CANDIDATE_FEATURES: constant integrity
- Artifact checks: CSV and summary markdown existence (skipped if script not run)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v172_vgt_selector_agreement import (
    AS_OF_DATES,
    MIN_BA_ADVANTAGE,
    RESULTS_CSV,
    SUMMARY_MD,
    V129_RESULTS,
    VGT_CANDIDATE_FEATURES,
    _build_en_pipeline,
    _build_l1_pipeline,
    determine_synthesis_verdict,
    run_regularized_gate,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic X/y data
# ---------------------------------------------------------------------------

def _make_xy(
    n: int = 80,
    features: list[str] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    if features is None:
        features = [
            "rate_adequacy_gap_yoy",
            "severity_index_yoy",
            "buyback_acceleration",
            "pb_ratio",
            "yield_curvature",
        ]
    x = pd.DataFrame(rng.standard_normal((n, len(features))), columns=features)
    y = pd.Series(rng.integers(0, 2, n), name="actionable_sell_3pct")
    return x, y


def _make_xy_with_signal(
    n: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a dataset where rate_adequacy_gap_yoy has a real relationship with y."""
    rng = np.random.default_rng(seed)
    features = [
        "rate_adequacy_gap_yoy",
        "severity_index_yoy",
        "buyback_acceleration",
        "pb_ratio",
        "yield_curvature",
    ]
    x = pd.DataFrame(rng.standard_normal((n, len(features))), columns=features)
    # Strong linear signal: y ~ 1 iff rate_adequacy_gap_yoy > 0.5
    logit = 3.0 * x["rate_adequacy_gap_yoy"]
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = pd.Series((prob > 0.5).astype(int), name="actionable_sell_3pct")
    return x, y


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_as_of_dates_length(self) -> None:
        assert len(AS_OF_DATES) == 3

    def test_as_of_dates_values(self) -> None:
        assert AS_OF_DATES == ["2022-03-31", "2023-03-31", "2024-03-31"]

    def test_candidate_features_length(self) -> None:
        assert len(VGT_CANDIDATE_FEATURES) == 2

    def test_candidate_features_values(self) -> None:
        assert "rate_adequacy_gap_yoy" in VGT_CANDIDATE_FEATURES
        assert "severity_index_yoy" in VGT_CANDIDATE_FEATURES

    def test_v129_results_length(self) -> None:
        assert len(V129_RESULTS) == 3

    def test_v129_results_dates_match_as_of_dates(self) -> None:
        for row, expected in zip(V129_RESULTS, AS_OF_DATES):
            assert row["as_of_date"] == expected

    def test_v129_results_have_required_keys(self) -> None:
        required = {"as_of_date", "ba_2feat", "ba_baseline", "n_covered", "top3_forward"}
        for row in V129_RESULTS:
            assert required.issubset(row.keys())

    def test_v129_ba_values_are_valid(self) -> None:
        for row in V129_RESULTS:
            assert 0.0 <= float(row["ba_2feat"]) <= 1.0
            assert 0.0 <= float(row["ba_baseline"]) <= 1.0

    def test_v129_2feat_exceeds_baseline_at_all_dates(self) -> None:
        for row in V129_RESULTS:
            assert float(row["ba_2feat"]) > float(row["ba_baseline"])


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


class TestPipelineBuilders:
    def test_l1_pipeline_has_two_steps(self) -> None:
        pipe = _build_l1_pipeline(0.25)
        assert len(pipe.steps) == 2

    def test_l1_pipeline_solver_is_saga(self) -> None:
        pipe = _build_l1_pipeline(0.25)
        assert pipe.named_steps["logistic"].solver == "saga"

    def test_l1_pipeline_c_value_set(self) -> None:
        pipe = _build_l1_pipeline(0.10)
        assert pipe.named_steps["logistic"].C == pytest.approx(0.10)

    def test_l1_pipeline_l1_ratio_is_one(self) -> None:
        pipe = _build_l1_pipeline(0.25)
        assert pipe.named_steps["logistic"].l1_ratio == pytest.approx(1.0)

    def test_en_pipeline_has_two_steps(self) -> None:
        pipe = _build_en_pipeline(0.25, 0.50)
        assert len(pipe.steps) == 2

    def test_en_pipeline_solver_is_saga(self) -> None:
        pipe = _build_en_pipeline(0.25, 0.50)
        assert pipe.named_steps["logistic"].solver == "saga"

    def test_en_pipeline_c_and_l1_ratio_set(self) -> None:
        pipe = _build_en_pipeline(0.10, 0.80)
        lr = pipe.named_steps["logistic"]
        assert lr.C == pytest.approx(0.10)
        assert lr.l1_ratio == pytest.approx(0.80)

    def test_en_pipeline_different_l1_ratio_values(self) -> None:
        for ratio in (0.20, 0.50, 0.80):
            pipe = _build_en_pipeline(0.25, ratio)
            assert pipe.named_steps["logistic"].l1_ratio == pytest.approx(ratio)


# ---------------------------------------------------------------------------
# run_regularized_gate — structure and gate_passed logic
# ---------------------------------------------------------------------------


class TestRunRegularizedGate:
    def test_returns_required_keys(self) -> None:
        x, y = _make_xy()
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        required = {
            "eligible_features",
            "l1_selected",
            "en_selected",
            "candidate_l1_agreement",
            "candidate_en_agreement",
            "gate_passed",
        }
        assert required.issubset(result.keys())

    def test_candidate_l1_agreement_keys_match_candidates(self) -> None:
        x, y = _make_xy()
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert set(result["candidate_l1_agreement"].keys()) == set(VGT_CANDIDATE_FEATURES)

    def test_candidate_en_agreement_keys_match_candidates(self) -> None:
        x, y = _make_xy()
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert set(result["candidate_en_agreement"].keys()) == set(VGT_CANDIDATE_FEATURES)

    def test_l1_selected_is_list(self) -> None:
        x, y = _make_xy()
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert isinstance(result["l1_selected"], list)

    def test_en_selected_is_list(self) -> None:
        x, y = _make_xy()
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert isinstance(result["en_selected"], list)

    def test_gate_passed_is_bool(self) -> None:
        x, y = _make_xy()
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert isinstance(result["gate_passed"], bool)

    def test_candidate_agreement_values_are_bool(self) -> None:
        x, y = _make_xy()
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        for v in result["candidate_l1_agreement"].values():
            assert isinstance(v, bool)
        for v in result["candidate_en_agreement"].values():
            assert isinstance(v, bool)

    def test_gate_passed_when_no_candidates_selected(self) -> None:
        """gate_passed is False when neither candidate appears in either selector."""
        x, y = _make_xy()
        # Use a candidate feature list that is NOT in x's columns
        result = run_regularized_gate(x, y, ["nonexistent_feat_a", "nonexistent_feat_b"])
        assert result["gate_passed"] is False

    def test_gate_passed_requires_both_l1_and_en_agreement(self) -> None:
        """gate_passed=True requires at least one candidate in BOTH l1 and en sets."""
        x, y = _make_xy_with_signal()
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        # Verify the invariant directly
        l1_any = any(result["candidate_l1_agreement"].values())
        en_any = any(result["candidate_en_agreement"].values())
        assert result["gate_passed"] == (l1_any and en_any)

    def test_insufficient_obs_returns_gate_false(self) -> None:
        """Fewer than 30 aligned observations → gate cannot be evaluated."""
        x, y = _make_xy(n=20)
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert result["gate_passed"] is False

    def test_single_class_target_returns_gate_false(self) -> None:
        """If y is all-zeros, regularization cannot run → gate_passed False."""
        x, y = _make_xy(n=60)
        y[:] = 0
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert result["gate_passed"] is False

    def test_missing_values_in_x_handled(self) -> None:
        """NaN values in feature columns are median-imputed; result is not an error."""
        x, y = _make_xy(n=80)
        x.iloc[::5, 0] = np.nan  # introduce NaNs in first column
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert isinstance(result["gate_passed"], bool)

    def test_empty_dataframe_returns_safe_defaults(self) -> None:
        x = pd.DataFrame(columns=["rate_adequacy_gap_yoy", "severity_index_yoy"])
        y = pd.Series(name="actionable_sell_3pct", dtype=int)
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert result["gate_passed"] is False
        assert result["l1_selected"] == []
        assert result["en_selected"] == []

    def test_strong_signal_feature_survives_l1(self) -> None:
        """With a strong linear signal in rate_adequacy_gap_yoy, L1 should keep it."""
        x, y = _make_xy_with_signal(n=120)
        result = run_regularized_gate(x, y, VGT_CANDIDATE_FEATURES)
        assert result["candidate_l1_agreement"]["rate_adequacy_gap_yoy"] is True


# ---------------------------------------------------------------------------
# determine_synthesis_verdict
# ---------------------------------------------------------------------------


class TestDetermineSynthesisVerdict:
    @staticmethod
    def _gate(passed: bool) -> dict[str, object]:
        return {
            "gate_passed": passed,
            "candidate_l1_agreement": {},
            "candidate_en_agreement": {},
            "l1_selected": [],
            "en_selected": [],
        }

    def test_conditional_shadow_when_all_criteria_met(self) -> None:
        """BA advantage >= MIN_BA_ADVANTAGE at all 3 dates + gate passed >= 2."""
        v129 = [
            {"as_of_date": "2022-03-31", "ba_2feat": 0.70, "ba_baseline": 0.58},
            {"as_of_date": "2023-03-31", "ba_2feat": 0.80, "ba_baseline": 0.58},
            {"as_of_date": "2024-03-31", "ba_2feat": 0.90, "ba_baseline": 0.58},
        ]
        gates = [self._gate(True), self._gate(True), self._gate(False)]
        assert determine_synthesis_verdict(v129, gates) == "CONDITIONAL_SHADOW"

    def test_reject_when_gate_passes_only_once(self) -> None:
        v129 = [
            {"as_of_date": "2022-03-31", "ba_2feat": 0.70, "ba_baseline": 0.58},
            {"as_of_date": "2023-03-31", "ba_2feat": 0.80, "ba_baseline": 0.58},
            {"as_of_date": "2024-03-31", "ba_2feat": 0.90, "ba_baseline": 0.58},
        ]
        gates = [self._gate(False), self._gate(True), self._gate(False)]
        assert determine_synthesis_verdict(v129, gates) == "REJECT"

    def test_reject_when_ba_advantage_too_small_at_one_date(self) -> None:
        v129 = [
            {"as_of_date": "2022-03-31", "ba_2feat": 0.70, "ba_baseline": 0.68},  # delta=0.02
            {"as_of_date": "2023-03-31", "ba_2feat": 0.80, "ba_baseline": 0.58},
            {"as_of_date": "2024-03-31", "ba_2feat": 0.90, "ba_baseline": 0.58},
        ]
        gates = [self._gate(True), self._gate(True), self._gate(True)]
        assert determine_synthesis_verdict(v129, gates) == "REJECT"

    def test_reject_when_all_gates_fail(self) -> None:
        v129 = [
            {"as_of_date": "2022-03-31", "ba_2feat": 0.70, "ba_baseline": 0.58},
            {"as_of_date": "2023-03-31", "ba_2feat": 0.80, "ba_baseline": 0.58},
            {"as_of_date": "2024-03-31", "ba_2feat": 0.90, "ba_baseline": 0.58},
        ]
        gates = [self._gate(False), self._gate(False), self._gate(False)]
        assert determine_synthesis_verdict(v129, gates) == "REJECT"

    def test_conditional_shadow_exact_gate_threshold(self) -> None:
        """Exactly 2 gate passes is sufficient for CONDITIONAL_SHADOW."""
        v129 = [
            {"as_of_date": "2022-03-31", "ba_2feat": 0.70, "ba_baseline": 0.58},
            {"as_of_date": "2023-03-31", "ba_2feat": 0.80, "ba_baseline": 0.58},
            {"as_of_date": "2024-03-31", "ba_2feat": 0.90, "ba_baseline": 0.58},
        ]
        gates = [self._gate(True), self._gate(True), self._gate(False)]
        assert determine_synthesis_verdict(v129, gates) == "CONDITIONAL_SHADOW"

    def test_reject_when_gates_passed_is_zero(self) -> None:
        v129 = [
            {"as_of_date": "2022-03-31", "ba_2feat": 0.70, "ba_baseline": 0.58},
            {"as_of_date": "2023-03-31", "ba_2feat": 0.80, "ba_baseline": 0.58},
            {"as_of_date": "2024-03-31", "ba_2feat": 0.90, "ba_baseline": 0.58},
        ]
        assert determine_synthesis_verdict(v129, []) == "REJECT"

    def test_ba_advantage_exactly_at_min_threshold(self) -> None:
        """BA advantage = MIN_BA_ADVANTAGE exactly should count as meeting the requirement."""
        v129 = [
            {"as_of_date": "2022-03-31",
             "ba_2feat": 0.58 + MIN_BA_ADVANTAGE, "ba_baseline": 0.58},
            {"as_of_date": "2023-03-31",
             "ba_2feat": 0.58 + MIN_BA_ADVANTAGE, "ba_baseline": 0.58},
            {"as_of_date": "2024-03-31",
             "ba_2feat": 0.58 + MIN_BA_ADVANTAGE, "ba_baseline": 0.58},
        ]
        gates = [self._gate(True), self._gate(True), self._gate(False)]
        assert determine_synthesis_verdict(v129, gates) == "CONDITIONAL_SHADOW"

    def test_v129_actual_data_produces_a_valid_verdict(self) -> None:
        """The hardcoded V129_RESULTS are valid input for determine_synthesis_verdict."""
        gates = [self._gate(True), self._gate(True), self._gate(True)]
        result = determine_synthesis_verdict(V129_RESULTS, gates)
        assert result in {"CONDITIONAL_SHADOW", "REJECT"}

    def test_v129_actual_data_ba_advantage_qualifies(self) -> None:
        """All V129 dates have BA advantage well above the minimum threshold."""
        for row in V129_RESULTS:
            delta = float(row["ba_2feat"]) - float(row["ba_baseline"])
            assert delta >= MIN_BA_ADVANTAGE, (
                f"V129 date {row['as_of_date']} BA delta {delta:.4f} < MIN_BA_ADVANTAGE"
            )


# ---------------------------------------------------------------------------
# Artifact checks (skipped if script has not been run)
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_results_csv_exists(self) -> None:
        if not RESULTS_CSV.exists():
            pytest.skip("v172 results CSV not generated — run v172 script first")
        df = pd.read_csv(RESULTS_CSV)
        assert len(df) > 0

    def test_results_csv_has_expected_columns(self) -> None:
        if not RESULTS_CSV.exists():
            pytest.skip("v172 results CSV not generated — run v172 script first")
        df = pd.read_csv(RESULTS_CSV)
        required = [
            "as_of_date", "gate_passed",
            "rate_adequacy_l1", "rate_adequacy_en",
            "severity_l1", "severity_en",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_results_csv_has_three_rows(self) -> None:
        if not RESULTS_CSV.exists():
            pytest.skip("v172 results CSV not generated — run v172 script first")
        df = pd.read_csv(RESULTS_CSV)
        assert len(df) == 3

    def test_results_csv_as_of_dates_match_constants(self) -> None:
        if not RESULTS_CSV.exists():
            pytest.skip("v172 results CSV not generated — run v172 script first")
        df = pd.read_csv(RESULTS_CSV)
        for expected in AS_OF_DATES:
            assert expected in df["as_of_date"].values

    def test_summary_md_exists(self) -> None:
        if not SUMMARY_MD.exists():
            pytest.skip("v172 summary MD not generated — run v172 script first")

    def test_summary_md_contains_verdict(self) -> None:
        if not SUMMARY_MD.exists():
            pytest.skip("v172 summary MD not generated — run v172 script first")
        text = SUMMARY_MD.read_text(encoding="utf-8")
        assert "CONDITIONAL_SHADOW" in text or "REJECT" in text

    def test_summary_md_contains_governance_section(self) -> None:
        if not SUMMARY_MD.exists():
            pytest.skip("v172 summary MD not generated — run v172 script first")
        text = SUMMARY_MD.read_text(encoding="utf-8")
        assert "Regularized Selector Gate" in text
