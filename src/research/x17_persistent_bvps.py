"""Research-only x17 persistent BVPS helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_persistent_bvps_series(
    current_bvps: pd.Series,
    monthly_dividends: pd.Series,
) -> pd.Series:
    """Build a dividend-persistent BVPS level series."""
    bvps = pd.to_numeric(current_bvps.copy(), errors="coerce")
    bvps.index = pd.DatetimeIndex(pd.to_datetime(bvps.index))
    bvps = bvps.sort_index()
    dividends = pd.to_numeric(monthly_dividends.copy(), errors="coerce").reindex(
        bvps.index,
        fill_value=0.0,
    )
    dividends = dividends.sort_index().fillna(0.0)
    persistent = bvps + dividends.cumsum()
    persistent.name = "persistent_bvps"
    return persistent


def build_persistent_bvps_targets(
    persistent_bvps: pd.Series,
    *,
    horizons: tuple[int, ...] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Build future persistent BVPS level and growth targets."""
    persistent = pd.to_numeric(persistent_bvps.copy(), errors="coerce")
    persistent.index = pd.DatetimeIndex(pd.to_datetime(persistent.index))
    persistent = persistent.sort_index()
    result = pd.DataFrame({"persistent_bvps": persistent}, index=persistent.index)
    for horizon in horizons:
        future_level = persistent.shift(-horizon)
        future_growth = future_level / persistent - 1.0
        result[f"target_{horizon}m_persistent_bvps"] = future_level
        result[f"target_{horizon}m_persistent_bvps_growth"] = future_growth
    return result


def summarize_x17_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x17 rows within target variant and horizon."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in detail_df.groupby(["target_variant", "horizon_months"], dropna=False):
        ranked = group.sort_values(
            ["future_bvps_mae", "growth_rmse", "model_name"],
            ascending=[True, True, True],
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
