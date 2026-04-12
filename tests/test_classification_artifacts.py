from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

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
