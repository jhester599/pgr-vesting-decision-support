"""Research-only x7 targeted TA replacement utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class X7TAVariant:
    """One bounded TA replacement variant for x-series research."""

    variant: str
    feature_swaps: dict[str, str]
    experiment_mode: str
    notes: str


def apply_feature_swaps(
    baseline_features: list[str],
    feature_swaps: dict[str, str],
) -> list[str]:
    """Apply replacement-only feature swaps while preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for feature in baseline_features:
        replacement = feature_swaps.get(feature, feature)
        if replacement in seen:
            continue
        result.append(replacement)
        seen.add(replacement)
    return result


def build_x7_ta_variants() -> list[X7TAVariant]:
    """Return the pre-registered bounded x7 TA replacement variants."""
    return [
        X7TAVariant(
            variant="x2_core_baseline",
            feature_swaps={},
            experiment_mode="replacement",
            notes="Existing x2 conservative feature set.",
        ),
        X7TAVariant(
            variant="ta_minimal_replacement",
            feature_swaps={
                "mom_12m": "ta_pgr_obv_detrended",
                "vol_63d": "ta_pgr_natr_63d",
            },
            experiment_mode="replacement",
            notes="v164 minimal OBV/NATR replacement candidate.",
        ),
        X7TAVariant(
            variant="ta_minimal_plus_vwo_pct_b",
            feature_swaps={
                "mom_12m": "ta_pgr_obv_detrended",
                "vol_63d": "ta_pgr_natr_63d",
                "vix": "ta_ratio_bb_pct_b_6m_vwo",
            },
            experiment_mode="replacement",
            notes="Minimal replacement plus one representative VWO Bollinger %B.",
        ),
        X7TAVariant(
            variant="ta_bollinger_width_probe",
            feature_swaps={
                "mom_12m": "ta_pgr_obv_detrended",
                "vol_63d": "ta_pgr_natr_63d",
                "vix": "ta_ratio_bb_width_6m_voo",
            },
            experiment_mode="replacement",
            notes="Minimal replacement plus one VOO Bollinger-width probe.",
        ),
    ]


def attach_baseline_deltas(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Attach horizon-specific deltas versus the x2 core baseline."""
    if detail_df.empty:
        return detail_df.copy()
    metric_cols = [
        "balanced_accuracy",
        "brier_score",
        "log_loss",
        "accuracy",
        "precision",
        "recall",
    ]
    present = [column for column in metric_cols if column in detail_df.columns]
    baseline = detail_df.loc[
        detail_df["variant"].eq("x2_core_baseline"),
        ["horizon_months", *present],
    ].drop_duplicates(["horizon_months"])
    baseline = baseline.rename(
        columns={column: f"baseline_{column}" for column in present}
    )
    merged = detail_df.merge(baseline, on="horizon_months", how="left")
    for column in present:
        merged[f"delta_{column}"] = (
            merged[column] - merged[f"baseline_{column}"]
        ).round(12)
    merged["clears_x7_gate"] = (
        merged.get("delta_balanced_accuracy", pd.Series(0.0, index=merged.index))
        .fillna(0.0)
        .gt(0.0)
        & merged.get("delta_brier_score", pd.Series(0.0, index=merged.index))
        .fillna(0.0)
        .lt(0.0)
    )
    return merged


def summarize_ta_variants(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize x7 variants across horizons."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variant, group in detail_df.groupby("variant", dropna=False):
        rows.append(
            {
                "variant": variant,
                "tested_horizon_count": int(group["horizon_months"].nunique()),
                "cleared_horizon_count": int(
                    group.loc[
                        group.get(
                            "clears_x7_gate",
                            pd.Series(False, index=group.index),
                        ).fillna(False),
                        "horizon_months",
                    ].nunique()
                ),
                "mean_delta_balanced_accuracy": float(
                    group.get(
                        "delta_balanced_accuracy",
                        pd.Series(dtype=float),
                    ).mean()
                ),
                "mean_delta_brier_score": float(
                    group.get("delta_brier_score", pd.Series(dtype=float)).mean()
                ),
                "mean_balanced_accuracy": float(
                    group.get("balanced_accuracy", pd.Series(dtype=float)).mean()
                ),
                "mean_brier_score": float(
                    group.get("brier_score", pd.Series(dtype=float)).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "cleared_horizon_count",
            "mean_delta_balanced_accuracy",
            "mean_delta_brier_score",
            "variant",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
