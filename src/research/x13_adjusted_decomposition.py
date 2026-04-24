"""Research-only x13 adjusted decomposition utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

X13_HORIZONS: tuple[int, int] = (3, 6)


def combine_adjusted_decomposition_predictions(
    bvps_predictions: pd.DataFrame,
    pb_predictions: pd.DataFrame,
    *,
    horizon_months: int,
    bvps_model_name: str,
    pb_model_name: str,
    target_variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine BVPS and P/B predictions into an implied future price."""
    merged = bvps_predictions.merge(
        pb_predictions,
        on=["date", "fold_idx"],
        how="inner",
        suffixes=("_bvps", "_pb"),
    )
    if merged.empty:
        raise ValueError("No aligned BVPS and P/B predictions to combine.")
    merged["implied_future_price"] = merged["implied_future_bvps"] * merged["y_pred_pb"]
    merged["true_future_price"] = merged["true_future_bvps"] * merged["y_true_pb"]
    price_error = merged["implied_future_price"] - merged["true_future_price"]
    current_price = merged["current_bvps"] * merged["current_pb"]
    true_up = merged["true_future_price"] > current_price
    predicted_up = merged["implied_future_price"] > current_price
    model_name = f"{bvps_model_name}__{pb_model_name}"
    metrics = {
        "horizon_months": int(horizon_months),
        "model_name": model_name,
        "bvps_model_name": bvps_model_name,
        "pb_model_name": pb_model_name,
        "target_variant": target_variant,
        "n_obs": int(len(merged)),
        "implied_price_mae": float(np.mean(np.abs(price_error))),
        "implied_price_rmse": float(np.sqrt(np.mean(np.square(price_error)))),
        "directional_hit_rate": float(np.mean(predicted_up == true_up)),
    }
    return merged.sort_values("date").reset_index(drop=True), metrics


def summarize_x13_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x13 rows within each target variant and horizon."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in detail_df.groupby(["target_variant", "horizon_months"]):
        ranked = group.sort_values(
            ["implied_price_mae", "implied_price_rmse", "directional_hit_rate", "model_name"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            out = row.copy()
            out["rank"] = int(rank)
            rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-safe records."""
    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")
