from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from config.features import CONTEXTUAL_CLASSIFIER_BENCHMARKS
from src.reporting.classification_artifacts import (
    CLASSIFICATION_SHADOW_COLUMNS,
    DECISION_OVERLAY_COLUMNS,
    append_classifier_history,
    build_classifier_history_entry,
    classification_history_path,
    write_classification_shadow_csv,
    write_decision_overlays_csv,
)


def test_write_classification_shadow_csv_enforces_columns(tmp_path: Path) -> None:
    path = write_classification_shadow_csv(
        tmp_path,
        pd.DataFrame(
            {
                "benchmark": ["VOO"],
                "classifier_raw_prob_actionable_sell": [0.32],
                "classifier_prob_actionable_sell": [0.28],
                "classifier_history_obs": [36],
                "classifier_weight": [0.5],
                "classifier_weighted_contribution": [0.14],
                "classifier_shadow_tier": ["HIGH"],
            }
        ),
    )
    written = pd.read_csv(path)
    assert list(written.columns) == CLASSIFICATION_SHADOW_COLUMNS


def test_write_decision_overlays_csv_enforces_columns(tmp_path: Path) -> None:
    path = write_decision_overlays_csv(
        tmp_path,
        pd.DataFrame(
            {
                "variant": ["live", "shadow_gate"],
                "recommendation_mode": ["DEFER-TO-TAX-DEFAULT", "ACTIONABLE"],
                "recommended_sell_pct": [0.5, 0.75],
                "would_change": [False, True],
                "reason": ["live production path", "classifier granted permission to deviate"],
                "classifier_prob_actionable_sell": [0.72, 0.72],
            }
        ),
    )
    written = pd.read_csv(path)
    assert list(written.columns) == DECISION_OVERLAY_COLUMNS


def test_classifier_history_append_round_trip(tmp_path: Path) -> None:
    entry = build_classifier_history_entry(
        as_of_date=date(2026, 4, 11),
        run_date=date(2026, 4, 11),
        feature_anchor_date="2026-03-31",
        forecast_horizon_months=6,
        classification_shadow_summary={
            "probability_actionable_sell": 0.28,
            "stance": "NON-ACTIONABLE",
            "confidence_tier": "HIGH",
        },
        live_recommendation_mode="DEFER-TO-TAX-DEFAULT",
        live_sell_pct=0.5,
        shadow_gate_overlay={
            "variant": "permission_overlay",
            "recommendation_mode": "DEFER-TO-TAX-DEFAULT",
            "recommended_sell_pct": 0.5,
            "would_change": False,
        },
    )
    path = append_classifier_history(base_dir=tmp_path, entry=entry)
    assert path == classification_history_path(tmp_path)
    written = pd.read_csv(path)
    assert len(written) == 1
    assert written.loc[0, "classifier_prob_actionable_sell"] == 0.28


# ---------------------------------------------------------------------------
# Tests for is_contextual column in classification_shadow.csv (Task 4 / v123)
# ---------------------------------------------------------------------------


def test_classification_shadow_columns_includes_is_contextual() -> None:
    assert "is_contextual" in CLASSIFICATION_SHADOW_COLUMNS


def test_write_classification_shadow_csv_adds_is_contextual_column(
    tmp_path,
) -> None:
    detail_df = pd.DataFrame({
        "benchmark": ["VOO", "VXUS", "VWO", "BND", "GLD", "DBC"],
        "classifier_raw_prob_actionable_sell": [0.5] * 6,
        "classifier_prob_actionable_sell": [0.5] * 6,
        "classifier_history_obs": [200] * 6,
        "classifier_weight": [0.1] * 6,
        "classifier_weighted_contribution": [0.05] * 6,
        "classifier_shadow_tier": ["MODERATE"] * 6,
    })
    out_path = write_classification_shadow_csv(tmp_path, detail_df)
    written = pd.read_csv(out_path)
    assert "is_contextual" in written.columns
    contextual_map = written.set_index("benchmark")["is_contextual"]
    assert not contextual_map["VOO"]
    assert not contextual_map["VXUS"]
    assert not contextual_map["VWO"]
    assert contextual_map["GLD"]
    assert contextual_map["DBC"]
    # All CONTEXTUAL_CLASSIFIER_BENCHMARKS in the df should be True
    for ticker in ["GLD", "DBC"]:
        assert contextual_map[ticker] == True


def test_write_classification_shadow_csv_empty_df_has_is_contextual_column(
    tmp_path,
) -> None:
    out_path = write_classification_shadow_csv(tmp_path, None)
    written = pd.read_csv(out_path)
    assert "is_contextual" in written.columns
    assert len(written) == 0


def test_classification_shadow_columns_has_dual_track_columns() -> None:
    assert "benchmark_specific_features" in CLASSIFICATION_SHADOW_COLUMNS
    assert "benchmark_specific_prob_actionable_sell" in CLASSIFICATION_SHADOW_COLUMNS
    assert "benchmark_specific_tier" in CLASSIFICATION_SHADOW_COLUMNS


def test_write_classification_shadow_csv_writes_dual_track_columns(tmp_path) -> None:
    import pandas as pd
    from src.reporting.classification_artifacts import write_classification_shadow_csv
    detail_df = pd.DataFrame({
        "benchmark": ["BND", "VGT"],
        "classifier_raw_prob_actionable_sell": [0.40, 0.30],
        "classifier_prob_actionable_sell": [0.38, 0.29],
        "classifier_history_obs": [120, 150],
        "classifier_weight": [0.1, 0.2],
        "classifier_weighted_contribution": [0.038, 0.058],
        "classifier_shadow_tier": ["MODERATE", "HIGH"],
        "is_contextual": [False, False],
        "benchmark_specific_features": ["pb_ratio|npw_per_pif_yoy", "lean_baseline"],
        "benchmark_specific_prob_actionable_sell": [0.44, 0.29],
        "benchmark_specific_tier": ["MODERATE", "HIGH"],
    })
    path = write_classification_shadow_csv(tmp_path, detail_df)
    written = pd.read_csv(path)
    assert "benchmark_specific_features" in written.columns
    assert "benchmark_specific_prob_actionable_sell" in written.columns
    assert "benchmark_specific_tier" in written.columns
    assert "classifier_prob_actionable_sell" in written.columns
