"""Tests for v130 Path B temperature scaling adoption analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _import_evaluate():
    """Import evaluate_adoption_criteria from v130 script."""
    import importlib.util

    script_path = (
        PROJECT_ROOT / "results" / "research" / "v130_path_b_temp_scaling_adoption.py"
    )
    spec = importlib.util.spec_from_file_location(
        "v130_path_b_temp_scaling_adoption", script_path
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.evaluate_adoption_criteria


def test_adoption_criteria_all_pass() -> None:
    """When all three criteria pass, verdict should be ADOPT."""
    evaluate_adoption_criteria = _import_evaluate()
    result = evaluate_adoption_criteria(
        ba_temp=0.60,
        ba_path_a=0.50,
        brier_temp=0.18,
        brier_path_a=0.20,
        ece_temp=0.10,
        ece_path_a=0.12,
    )
    assert result["criterion_a"] is True, "BA delta 0.10 >= 0.03 should pass"
    assert result["criterion_b"] is True, "Brier 0.18 <= 0.22 should pass"
    assert result["criterion_c"] is True, "ECE 0.10 <= 0.18 should pass"
    assert result["adopt"] is True


def test_adoption_criteria_fails_criterion_a() -> None:
    """When BA delta < 0.03, should not adopt."""
    evaluate_adoption_criteria = _import_evaluate()
    result = evaluate_adoption_criteria(
        ba_temp=0.52,
        ba_path_a=0.50,
        brier_temp=0.18,
        brier_path_a=0.20,
        ece_temp=0.10,
        ece_path_a=0.12,
    )
    assert result["criterion_a"] is False, "BA delta 0.02 < 0.03 should fail"
    assert result["adopt"] is False


def test_adoption_criteria_fails_criterion_b() -> None:
    """When Brier degrades by more than 0.02 vs Path A, should not adopt."""
    evaluate_adoption_criteria = _import_evaluate()
    result = evaluate_adoption_criteria(
        ba_temp=0.60,
        ba_path_a=0.50,
        brier_temp=0.24,
        brier_path_a=0.20,
        ece_temp=0.10,
        ece_path_a=0.12,
    )
    assert result["criterion_b"] is False, "Brier 0.24 > 0.22 should fail"
    assert result["adopt"] is False


def test_adoption_criteria_fails_criterion_c() -> None:
    """When ECE is more than 50% worse than Path A, should not adopt."""
    evaluate_adoption_criteria = _import_evaluate()
    result = evaluate_adoption_criteria(
        ba_temp=0.60,
        ba_path_a=0.50,
        brier_temp=0.18,
        brier_path_a=0.20,
        ece_temp=0.20,
        ece_path_a=0.12,
    )
    assert result["criterion_c"] is False, "ECE 0.20 > 0.18 should fail"
    assert result["adopt"] is False


def test_adoption_criteria_boundary_ba() -> None:
    """BA delta exactly 0.03 should pass criterion A."""
    evaluate_adoption_criteria = _import_evaluate()
    result = evaluate_adoption_criteria(
        ba_temp=0.53,
        ba_path_a=0.50,
        brier_temp=0.18,
        brier_path_a=0.20,
        ece_temp=0.10,
        ece_path_a=0.12,
    )
    assert result["criterion_a"] is True, "BA delta 0.03 exactly >= 0.03 should pass"


def test_adoption_criteria_boundary_ece() -> None:
    """ECE exactly 1.5x Path A should pass criterion C."""
    evaluate_adoption_criteria = _import_evaluate()
    result = evaluate_adoption_criteria(
        ba_temp=0.60,
        ba_path_a=0.50,
        brier_temp=0.18,
        brier_path_a=0.20,
        ece_temp=0.18,
        ece_path_a=0.12,
    )
    assert result["criterion_c"] is True, "ECE 0.18 == 0.12 * 1.5 exactly should pass"


def test_output_csv_has_required_columns() -> None:
    """Output CSV must have model, balanced_accuracy_covered, brier_score, ece_10, adopt."""
    p = PROJECT_ROOT / "results" / "research" / "v130_path_b_temp_scaling_results.csv"
    if not p.exists():
        pytest.skip("Output CSV not generated yet")
    df = pd.read_csv(p)
    for col in ["model", "balanced_accuracy_covered", "brier_score", "ece_10", "adopt"]:
        assert col in df.columns, f"Missing column: {col}"


def test_output_csv_has_three_models() -> None:
    """Output CSV must have exactly three model rows."""
    p = PROJECT_ROOT / "results" / "research" / "v130_path_b_temp_scaling_results.csv"
    if not p.exists():
        pytest.skip("Output CSV not generated yet")
    df = pd.read_csv(p)
    assert len(df) == 3, f"Expected 3 model rows, got {len(df)}"


def test_summary_md_exists() -> None:
    """Summary markdown must exist after script runs."""
    p = PROJECT_ROOT / "results" / "research" / "v130_path_b_temp_scaling_summary.md"
    if not p.exists():
        pytest.skip("Summary markdown not generated yet")
    text = p.read_text(encoding="utf-8")
    assert "ADOPT" in text or "DO NOT ADOPT" in text, "Summary must contain adoption verdict"
