"""Research-only x18 dividend policy regime utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POLICY_CHANGE_DATE = pd.Timestamp("2018-12-01")


def _amount_column(dividends: pd.DataFrame) -> str:
    for column in ("amount", "dividend"):
        if column in dividends.columns:
            return column
    raise ValueError("dividends must include an 'amount' or 'dividend' column")


def classify_dividend_regime(
    dividends: pd.DataFrame,
    *,
    policy_change_date: pd.Timestamp = POLICY_CHANGE_DATE,
) -> pd.DataFrame:
    """Classify dividend rows into pre/post policy regimes."""
    amount_col = _amount_column(dividends)
    result = dividends.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    result = result.sort_index()
    result[amount_col] = pd.to_numeric(result[amount_col], errors="coerce")
    result["policy_regime"] = np.where(
        result.index < policy_change_date,
        "quantitative_annual",
        "regular_plus_special",
    )
    return result


def infer_post_policy_regular_baseline(
    dividends: pd.Series,
    *,
    as_of_date: pd.Timestamp,
    lookback_months: int = 24,
    regular_cap: float = 0.25,
) -> float:
    """Infer the post-policy regular quarterly dividend from prior small payments."""
    lookback_start = as_of_date - pd.DateOffset(months=lookback_months)
    prior = dividends.loc[
        (dividends.index < as_of_date) & (dividends.index >= lookback_start)
    ].dropna()
    regular = prior[(prior > 0.0) & (prior <= regular_cap)]
    if regular.empty:
        return float("nan")
    return float(regular.median())


def _window_bounds(snapshot_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=int(snapshot_date.year), month=12, day=1)
    end = pd.Timestamp(year=int(snapshot_date.year + 1), month=2, day=28)
    if end.is_leap_year:
        end = pd.Timestamp(year=end.year, month=2, day=29)
    return start, end


def build_regime_aware_dividend_targets(
    monthly_features: pd.DataFrame,
    dividends: pd.DataFrame,
    *,
    policy_change_date: pd.Timestamp = POLICY_CHANGE_DATE,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Build regime-aware annual dividend targets from November snapshots."""
    if monthly_features.empty:
        return pd.DataFrame()
    amount_col = _amount_column(dividends)
    div = classify_dividend_regime(
        dividends[[amount_col]].copy(),
        policy_change_date=policy_change_date,
    )
    amounts = pd.to_numeric(div[amount_col], errors="coerce")
    snapshots = monthly_features.copy()
    snapshots.index = pd.DatetimeIndex(pd.to_datetime(snapshots.index))
    snapshots = snapshots.sort_index()
    snapshots = snapshots[snapshots.index.month == 11]

    rows: list[dict[str, Any]] = []
    for snapshot_date, snapshot in snapshots.iterrows():
        window_start, window_end = _window_bounds(snapshot_date)
        window_values = amounts.loc[
            (amounts.index >= window_start) & (amounts.index <= window_end)
        ]
        has_complete_target = not amounts.empty and amounts.index.max() >= window_end
        total = float(window_values.sum()) if has_complete_target else float("nan")
        regime = (
            "quantitative_annual"
            if window_start < policy_change_date
            else "regular_plus_special"
        )
        if regime == "regular_plus_special":
            baseline = infer_post_policy_regular_baseline(amounts, as_of_date=window_start)
            excess = (
                max(total - baseline, 0.0)
                if np.isfinite(total) and np.isfinite(baseline)
                else float("nan")
            )
            occurred = int(excess > tolerance) if np.isfinite(excess) else float("nan")
            eligible = 1
        else:
            baseline = 0.0
            excess = float("nan")
            occurred = float("nan")
            eligible = 0
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "snapshot_year": int(snapshot_date.year),
                "policy_regime": regime,
                "target_window_start": window_start,
                "target_window_end": window_end,
                "window_dividend_total": total,
                "regular_baseline_dividend": baseline,
                "special_dividend_excess_regime": excess,
                "special_dividend_occurred_regime": occurred,
                "model_eligible_post_policy": eligible,
                "snapshot_bvps": float(snapshot.get("book_value_per_share", np.nan)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("snapshot_date")


def summarize_x18_policy_targets(targets: pd.DataFrame) -> dict[str, Any]:
    """Summarize the policy-regime target frame."""
    post_policy = targets[targets["policy_regime"] == "regular_plus_special"]
    return {
        "snapshot_count": int(len(targets)),
        "post_policy_snapshot_count": int(len(post_policy)),
        "first_post_policy_snapshot": (
            None if post_policy.empty else str(post_policy.index.min().date())
        ),
        "mean_post_policy_window_total": (
            None if post_policy.empty else float(post_policy["window_dividend_total"].mean())
        ),
        "mean_post_policy_special_excess": (
            None
            if post_policy.empty
            else float(post_policy["special_dividend_excess_regime"].mean())
        ),
    }
