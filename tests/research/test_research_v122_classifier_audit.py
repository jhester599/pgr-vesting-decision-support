from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.research.study_paths import study_output_path

# Stored-artifact tests: they read committed data (review F28, step 9).
pytestmark = pytest.mark.artifact




def test_v122_classifier_audit_outputs_have_expected_columns() -> None:
    coeff_df = pd.read_csv(study_output_path("v122_classifier_audit_coefficients.csv"))
    assert {
        "benchmark",
        "feature",
        "coefficient",
        "feature_std",
        "standardized_abs_coef",
        "train_n",
        "classifier_weight",
        "weighted_importance",
    }.issubset(coeff_df.columns)

    totals_df = pd.read_csv(study_output_path("v122_classifier_audit_feature_totals.csv"))
    assert {"feature", "weighted_importance"}.issubset(totals_df.columns)

    summary_path = study_output_path("v122_classifier_audit_summary.md")
    assert summary_path.exists()
