"""Research-only x15 P/B regime overlay utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_pb_regime_targets(
    current_pb: pd.Series,
    future_pb: pd.Series,
    *,
    hurdle: float,
) -> pd.DataFrame:
    """Build bounded up/down/neutral P/B regime targets."""
    aligned = pd.concat(
        [
            pd.to_numeric(current_pb, errors="coerce").rename("current_pb"),
            pd.to_numeric(future_pb, errors="coerce").rename("future_pb"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    normalized_delta = aligned["future_pb"] / aligned["current_pb"] - 1.0
    target_up = (normalized_delta > hurdle).astype(int)
    target_down = (normalized_delta < -hurdle).astype(int)
    target_regime = np.where(
        target_up == 1,
        "up",
        np.where(target_down == 1, "down", "neutral"),
    )
    return pd.DataFrame(
        {
            "current_pb": aligned["current_pb"],
            "future_pb": aligned["future_pb"],
            "normalized_delta": normalized_delta,
            "target_up": target_up,
            "target_down": target_down,
            "target_regime": target_regime,
        },
        index=aligned.index,
    )


def apply_pb_overlay(
    *,
    current_pb: float,
    up_prob: float,
    down_prob: float,
    positive_shift: float,
    negative_shift: float,
    confidence_threshold: float,
) -> tuple[float, str]:
    """Map up/down probabilities into a bounded P/B overlay prediction."""
    action = "neutral"
    shift = 0.0
    if up_prob >= confidence_threshold and up_prob > down_prob:
        action = "up"
        shift = positive_shift
    elif down_prob >= confidence_threshold and down_prob > up_prob:
        action = "down"
        shift = negative_shift
    predicted_pb = max(float(current_pb) * (1.0 + float(shift)), 0.01)
    return float(predicted_pb), action


def summarize_x15_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x15 overlay rows within horizon and compare to no-change."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in detail_df.groupby("horizon_months", dropna=False):
        ranked = group.sort_values(
            ["pb_mae", "pb_rmse", "overlay_action_rate", "model_name"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        no_change = group[group["model_name"] == "no_change_pb_overlay"]
        baseline_mae = (
            float(no_change["pb_mae"].iloc[0])
            if not no_change.empty
            else float("nan")
        )
        baseline_rmse = (
            float(no_change["pb_rmse"].iloc[0])
            if not no_change.empty
            else float("nan")
        )
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            out = row.copy()
            out["rank"] = int(rank)
            out["beats_no_change_pb"] = bool(row["pb_mae"] < baseline_mae)
            out["beats_no_change_pb"] = bool(
                row["pb_mae"] < baseline_mae
                and row["pb_rmse"] <= baseline_rmse
            )
            rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-safe records."""
    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")
