"""Helpers for tracking rolling model-health drift over monthly runs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config


@dataclass(frozen=True)
class ModelDriftSummary:
    """Latest rolling model-health view used by monthly monitoring/reporting."""

    as_of_month: str | None
    window_months: int
    history_months: int
    rolling_ic: float
    rolling_hit_rate: float
    rolling_ece: float
    ic_below_threshold_streak: int
    drift_flag: bool


def add_rolling_model_health(
    history: pd.DataFrame,
    window_months: int = 12,
    ic_threshold: float = config.DIAG_MIN_IC,
) -> pd.DataFrame:
    """Return monthly history with rolling health metrics and IC breach flags."""
    if history.empty:
        return history.copy()

    required = {"month_end", "aggregate_nw_ic", "aggregate_hit_rate", "ece"}
    missing = required - set(history.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"history is missing required columns: {missing_str}")

    enriched = history.copy()
    enriched["month_end"] = pd.to_datetime(enriched["month_end"])
    enriched = enriched.sort_values("month_end").reset_index(drop=True)

    enriched["rolling_12_ic"] = (
        enriched["aggregate_nw_ic"].rolling(window=window_months, min_periods=1).mean()
    )
    enriched["rolling_12_hit_rate"] = (
        enriched["aggregate_hit_rate"].rolling(window=window_months, min_periods=1).mean()
    )
    enriched["rolling_12_ece"] = enriched["ece"].rolling(window=window_months, min_periods=1).mean()
    enriched["rolling_ic_below_threshold"] = enriched["rolling_12_ic"] < ic_threshold

    return enriched


def summarize_latest_model_drift(
    history: pd.DataFrame,
    window_months: int = 12,
    ic_threshold: float = config.DIAG_MIN_IC,
    min_consecutive_breaches: int = 3,
) -> ModelDriftSummary | None:
    """Summarize the latest rolling drift state from monthly performance history."""
    if history.empty:
        return None

    enriched = add_rolling_model_health(
        history,
        window_months=window_months,
        ic_threshold=ic_threshold,
    )
    latest = enriched.iloc[-1]

    streak = 0
    for below in reversed(enriched["rolling_ic_below_threshold"].tolist()):
        if not bool(below):
            break
        streak += 1

    return ModelDriftSummary(
        as_of_month=str(latest["month_end"].date()),
        window_months=window_months,
        history_months=int(len(enriched)),
        rolling_ic=float(latest["rolling_12_ic"]),
        rolling_hit_rate=float(latest["rolling_12_hit_rate"]),
        rolling_ece=float(latest["rolling_12_ece"]),
        ic_below_threshold_streak=streak,
        drift_flag=streak >= min_consecutive_breaches,
    )
