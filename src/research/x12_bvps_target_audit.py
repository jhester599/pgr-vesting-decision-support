"""Research-only x12 BVPS target audit utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_monthly_dividend_series(
    dividends: pd.DataFrame,
    monthly_index: pd.DatetimeIndex,
) -> pd.Series:
    """Aggregate dividend history to business month-end per-share totals."""
    amount_col = "amount" if "amount" in dividends.columns else "dividend"
    amounts = pd.to_numeric(dividends[amount_col], errors="coerce").copy()
    amounts.index = pd.DatetimeIndex(pd.to_datetime(amounts.index))
    monthly = amounts.groupby(amounts.index.to_period("M")).sum(min_count=1)
    monthly.index = monthly.index.to_timestamp("M")
    monthly.index = pd.DatetimeIndex(
        [pd.offsets.BMonthEnd().rollback(ts) for ts in monthly.index]
    )
    monthly = monthly.groupby(level=0).sum(min_count=1)
    result = monthly.reindex(pd.DatetimeIndex(monthly_index), fill_value=0.0)
    result.name = "monthly_dividend_per_share"
    return result.astype(float)


def build_adjusted_bvps_targets(
    current_bvps: pd.Series,
    monthly_dividends: pd.Series,
    *,
    horizons: tuple[int, ...] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Build raw and dividend-adjusted future BVPS targets."""
    bvps = pd.to_numeric(current_bvps.copy(), errors="coerce")
    bvps.index = pd.DatetimeIndex(pd.to_datetime(bvps.index))
    bvps = bvps.sort_index()
    dividends = pd.to_numeric(monthly_dividends.copy(), errors="coerce").reindex(
        bvps.index,
        fill_value=0.0,
    )
    dividends = dividends.sort_index().fillna(0.0)
    result = pd.DataFrame({"current_bvps": bvps}, index=bvps.index)
    for horizon in horizons:
        future_bvps = bvps.shift(-horizon)
        cumulative = pd.Series(0.0, index=bvps.index)
        for step in range(1, horizon + 1):
            cumulative = cumulative + dividends.shift(-step).fillna(0.0)
        adjusted_future = future_bvps + cumulative
        growth = future_bvps / bvps - 1.0
        adjusted_growth = adjusted_future / bvps - 1.0
        result[f"target_{horizon}m_bvps"] = future_bvps
        result[f"target_{horizon}m_bvps_growth"] = growth
        result[f"target_{horizon}m_adjusted_bvps"] = adjusted_future
        result[f"target_{horizon}m_adjusted_bvps_growth"] = adjusted_growth
        result[f"target_{horizon}m_forward_dividends"] = cumulative
    return result


def identify_bvps_discontinuities(
    current_bvps: pd.Series,
    monthly_dividends: pd.Series,
    *,
    pct_threshold: float = -0.08,
) -> pd.DataFrame:
    """Flag large negative BVPS changes with nearby dividend support."""
    bvps = pd.to_numeric(current_bvps.copy(), errors="coerce")
    bvps.index = pd.DatetimeIndex(pd.to_datetime(bvps.index))
    bvps = bvps.sort_index()
    dividends = pd.to_numeric(monthly_dividends.copy(), errors="coerce").reindex(
        bvps.index,
        fill_value=0.0,
    )
    pct_change = bvps.pct_change(fill_method=None)
    frame = pd.DataFrame(
        {
            "date": bvps.index,
            "current_bvps": bvps.values,
            "bvps_change": bvps.diff().values,
            "bvps_pct_change": pct_change.values,
            "monthly_dividend": dividends.values,
        }
    )
    frame["has_dividend_support"] = (
        frame["monthly_dividend"] > 0.0
    ).map(bool).astype(object)
    result = frame[frame["bvps_pct_change"] <= pct_threshold].copy()
    return result.reset_index(drop=True)


def summarize_x12_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x12 rows within each target variant and horizon."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in detail_df.groupby(["target_variant", "horizon_months"]):
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
