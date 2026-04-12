"""Artifact writers and history helpers for classifier shadow outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from config.features import CONTEXTUAL_CLASSIFIER_BENCHMARKS


CLASSIFICATION_SHADOW_COLUMNS = [
    "benchmark",
    "classifier_raw_prob_actionable_sell",
    "classifier_prob_actionable_sell",
    "classifier_history_obs",
    "classifier_weight",
    "classifier_weighted_contribution",
    "classifier_shadow_tier",
    "is_contextual",
]

DECISION_OVERLAY_COLUMNS = [
    "variant",
    "recommendation_mode",
    "recommended_sell_pct",
    "would_change",
    "reason",
    "classifier_prob_actionable_sell",
]

CLASSIFICATION_HISTORY_COLUMNS = [
    "as_of_date",
    "run_date",
    "feature_anchor_date",
    "forecast_horizon_months",
    "mature_on_date",
    "is_horizon_mature",
    "classifier_prob_actionable_sell",
    "classifier_stance",
    "classifier_confidence_tier",
    "live_recommendation_mode",
    "live_sell_pct",
    "shadow_overlay_mode",
    "shadow_overlay_sell_pct",
    "shadow_overlay_would_change",
    "shadow_overlay_variant",
    "actual_actionable_sell",
    "actual_basket_relative_return",
]


@dataclass(frozen=True)
class ClassifierHistoryEntry:
    """Append-only monthly classifier history row."""

    as_of_date: str
    run_date: str
    feature_anchor_date: str | None
    forecast_horizon_months: int
    mature_on_date: str | None
    is_horizon_mature: bool
    classifier_prob_actionable_sell: float | None
    classifier_stance: str | None
    classifier_confidence_tier: str | None
    live_recommendation_mode: str
    live_sell_pct: float
    shadow_overlay_mode: str | None
    shadow_overlay_sell_pct: float | None
    shadow_overlay_would_change: bool | None
    shadow_overlay_variant: str | None
    actual_actionable_sell: float | None = None
    actual_basket_relative_return: float | None = None

    def to_row(self) -> dict[str, Any]:
        """Return a flat CSV-ready row."""
        return asdict(self)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a frame containing exactly the requested columns."""
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result.loc[:, columns]


def write_classification_shadow_csv(
    out_dir: Path,
    detail_df: pd.DataFrame | None,
) -> Path:
    """Write the per-benchmark classifier shadow detail artifact."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "classification_shadow.csv"
    if detail_df is None or detail_df.empty:
        pd.DataFrame(columns=CLASSIFICATION_SHADOW_COLUMNS).to_csv(path, index=False)
        return path
    df = detail_df.copy()
    df["is_contextual"] = df["benchmark"].isin(CONTEXTUAL_CLASSIFIER_BENCHMARKS)
    _ensure_columns(df, CLASSIFICATION_SHADOW_COLUMNS).to_csv(path, index=False)
    return path


def write_decision_overlays_csv(
    out_dir: Path,
    overlay_df: pd.DataFrame | None,
) -> Path:
    """Write the shadow gate overlay artifact."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "decision_overlays.csv"
    if overlay_df is None or overlay_df.empty:
        pd.DataFrame(columns=DECISION_OVERLAY_COLUMNS).to_csv(path, index=False)
        return path
    _ensure_columns(overlay_df, DECISION_OVERLAY_COLUMNS).to_csv(path, index=False)
    return path


def classification_history_path(base_dir: Path | None = None) -> Path:
    """Return the append-only classifier history artifact path."""
    if base_dir is None:
        base_dir = Path("results") / "monthly_decisions"
    return base_dir / "classification_shadow_history.csv"


def append_classifier_history(
    *,
    base_dir: Path,
    entry: ClassifierHistoryEntry,
) -> Path:
    """Append or upsert one classifier-history row keyed by as_of_date."""
    path = classification_history_path(base_dir)
    if path.exists():
        history_df = pd.read_csv(path)
    else:
        history_df = pd.DataFrame(columns=CLASSIFICATION_HISTORY_COLUMNS)

    row_df = pd.DataFrame([entry.to_row()])
    history_df = history_df[history_df["as_of_date"].astype(str) != entry.as_of_date]
    if history_df.empty:
        history_df = row_df
    else:
        history_df = pd.concat([history_df, row_df], ignore_index=True)
    history_df = _ensure_columns(history_df, CLASSIFICATION_HISTORY_COLUMNS)
    history_df = history_df.sort_values("as_of_date").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(path, index=False)
    return path


def build_classifier_history_entry(
    *,
    as_of_date: date,
    run_date: date,
    feature_anchor_date: str | None,
    forecast_horizon_months: int,
    classification_shadow_summary: dict[str, Any] | None,
    live_recommendation_mode: str,
    live_sell_pct: float,
    shadow_gate_overlay: dict[str, Any] | None,
) -> ClassifierHistoryEntry:
    """Build one history row from the monthly classifier and overlay payloads."""
    mature_on_date = None
    is_horizon_mature = False
    if feature_anchor_date:
        mature_on_ts = pd.Timestamp(feature_anchor_date) + pd.DateOffset(
            months=forecast_horizon_months
        )
        mature_on_date = mature_on_ts.date().isoformat()
        is_horizon_mature = run_date >= mature_on_ts.date()

    return ClassifierHistoryEntry(
        as_of_date=as_of_date.isoformat(),
        run_date=run_date.isoformat(),
        feature_anchor_date=feature_anchor_date,
        forecast_horizon_months=forecast_horizon_months,
        mature_on_date=mature_on_date,
        is_horizon_mature=is_horizon_mature,
        classifier_prob_actionable_sell=(
            float(classification_shadow_summary["probability_actionable_sell"])
            if isinstance(classification_shadow_summary, dict)
            and classification_shadow_summary.get("probability_actionable_sell") is not None
            else None
        ),
        classifier_stance=(
            str(classification_shadow_summary.get("stance"))
            if isinstance(classification_shadow_summary, dict)
            and classification_shadow_summary.get("stance") is not None
            else None
        ),
        classifier_confidence_tier=(
            str(classification_shadow_summary.get("confidence_tier"))
            if isinstance(classification_shadow_summary, dict)
            and classification_shadow_summary.get("confidence_tier") is not None
            else None
        ),
        live_recommendation_mode=live_recommendation_mode,
        live_sell_pct=float(live_sell_pct),
        shadow_overlay_mode=(
            str(shadow_gate_overlay.get("recommendation_mode"))
            if isinstance(shadow_gate_overlay, dict)
            and shadow_gate_overlay.get("recommendation_mode") is not None
            else None
        ),
        shadow_overlay_sell_pct=(
            float(shadow_gate_overlay["recommended_sell_pct"])
            if isinstance(shadow_gate_overlay, dict)
            and shadow_gate_overlay.get("recommended_sell_pct") is not None
            else None
        ),
        shadow_overlay_would_change=(
            bool(shadow_gate_overlay.get("would_change"))
            if isinstance(shadow_gate_overlay, dict)
            and shadow_gate_overlay.get("would_change") is not None
            else None
        ),
        shadow_overlay_variant=(
            str(shadow_gate_overlay.get("variant"))
            if isinstance(shadow_gate_overlay, dict)
            and shadow_gate_overlay.get("variant") is not None
            else None
        ),
    )
