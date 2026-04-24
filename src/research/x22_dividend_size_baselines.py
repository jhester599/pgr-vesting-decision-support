"""Research-only x22 helpers for dividend size baseline challengers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.research.x21_dividend_target_scales import (
    back_transform_scaled_prediction,
    build_scaled_size_targets,
    iter_positive_expanding_splits,
)


def candidate_target_scales() -> dict[str, tuple[str, str]]:
    """Return the bounded x22 target-scale set."""
    return {
        "raw_dollars": ("target_raw_dollars", "unit_scale"),
        "to_current_bvps": ("target_to_current_bvps", "current_bvps"),
        "to_persistent_bvps": ("target_to_persistent_bvps", "persistent_bvps"),
    }


def baseline_from_history(history: pd.Series, *, mode: str) -> float:
    """Return one low-complexity baseline prediction from prior history."""
    numeric = pd.to_numeric(history, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    if mode == "historical_mean":
        return float(numeric.mean())
    if mode == "prior_positive_year":
        return float(numeric.iloc[-1])
    if mode == "trailing_2_mean":
        return float(numeric.tail(2).mean())
    if mode == "trailing_2_median":
        return float(numeric.tail(2).median())
    raise ValueError(f"Unsupported x22 baseline mode '{mode}'.")


def evaluate_x22_baseline(
    frame: pd.DataFrame,
    *,
    target_scale_name: str,
    target_column: str,
    scale_column: str,
    mode: str,
    min_train_years: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one positive-only annual baseline after back-transforming to dollars."""
    annual = build_scaled_size_targets(frame.copy())
    annual["unit_scale"] = 1.0
    aligned = annual[[target_column, scale_column, "special_dividend_excess"]].copy()
    aligned = aligned.sort_index()
    aligned[target_column] = pd.to_numeric(aligned[target_column], errors="coerce")
    aligned[scale_column] = pd.to_numeric(aligned[scale_column], errors="coerce")
    aligned["special_dividend_excess"] = pd.to_numeric(
        aligned["special_dividend_excess"],
        errors="coerce",
    )
    aligned = aligned[aligned["special_dividend_excess"] > 0.0]
    aligned = aligned.dropna(subset=[target_column, scale_column, "special_dividend_excess"])
    aligned = aligned[aligned[scale_column] > 1e-12]
    rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        iter_positive_expanding_splits(aligned, min_train_years=min_train_years)
    ):
        history = aligned[target_column].iloc[train_idx]
        scaled_pred = baseline_from_history(history, mode=mode)
        scale_value = float(aligned[scale_column].iloc[test_idx[0]])
        rows.append(
            {
                "snapshot_date": aligned.index[test_idx[0]],
                "fold_idx": int(fold_idx),
                "target_scale": target_scale_name,
                "scale_column": scale_column,
                "model_name": mode,
                "actual_scaled_target": float(aligned[target_column].iloc[test_idx[0]]),
                "predicted_scaled_target": float(max(scaled_pred, 0.0)),
                "actual_dollars": float(aligned["special_dividend_excess"].iloc[test_idx[0]]),
                "predicted_dollars": float(
                    back_transform_scaled_prediction(scaled_pred, scale_value)
                ),
            }
        )
    detail = pd.DataFrame(rows)
    metrics = {
        "target_scale": target_scale_name,
        "scale_column": scale_column,
        "model_name": mode,
        "n_obs": int(len(detail)),
        "dollar_mae": float(np.mean(np.abs(detail["predicted_dollars"] - detail["actual_dollars"]))),
        "dollar_rmse": float(
            np.sqrt(np.mean(np.square(detail["predicted_dollars"] - detail["actual_dollars"])))
        ),
        "scaled_mae": float(
            np.mean(np.abs(detail["predicted_scaled_target"] - detail["actual_scaled_target"]))
        ),
        "mean_actual_dollars": float(detail["actual_dollars"].mean()),
        "mean_predicted_dollars": float(detail["predicted_dollars"].mean()),
    }
    return detail, metrics


def summarize_x22_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x22 rows by dollar MAE."""
    if detail_df.empty:
        return pd.DataFrame()
    ranked = detail_df.copy()
    if "scaled_mae" not in ranked.columns:
        ranked["scaled_mae"] = np.nan
    ranked = ranked.sort_values(
        ["dollar_mae", "scaled_mae", "target_scale", "model_name"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked
