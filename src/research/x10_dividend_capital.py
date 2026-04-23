"""Research-only x10 capital features for special-dividend testing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def build_dividend_capital_features(monthly_features: pd.DataFrame) -> pd.DataFrame:
    """Add x9-derived capital-generation proxies for dividend research."""
    result = monthly_features.copy()
    result["capital_generation_proxy"] = (
        _numeric(result, "bvps_growth_ytd")
        + _numeric(result, "underwriting_margin_ttm")
        + _numeric(result, "unrealized_gain_pct_equity")
        - _numeric(result, "buyback_yield")
    )
    result["excess_capital_proxy"] = (
        _numeric(result, "unrealized_gain_pct_equity")
        + _numeric(result, "bvps_growth_ytd")
        - 0.02 * _numeric(result, "pgr_premium_to_surplus")
    )
    denominator = result["capital_generation_proxy"].where(
        result["capital_generation_proxy"].abs() > 1e-12
    )
    result["buyback_vs_capital_generation"] = (
        _numeric(result, "buyback_yield") / denominator
    )
    result["premium_growth_x_margin_for_dividend"] = (
        _numeric(result, "npw_growth_yoy")
        * _numeric(result, "underwriting_margin_ttm")
    )
    result["november_snapshot_flag"] = (result.index.month == 11).astype(int)
    return result


def november_capital_snapshots(monthly_features: pd.DataFrame) -> pd.DataFrame:
    """Return November-only annual capital snapshots."""
    enriched = build_dividend_capital_features(monthly_features)
    return enriched[enriched.index.month == 11].copy()


def build_x10_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return low-count x10 annual feature sets."""
    candidates = {
        "x6_reference": [
            "book_value_per_share_growth_yoy",
            "pb_ratio",
            "pgr_premium_to_surplus",
            "combined_ratio_ttm",
            "underwriting_margin_ttm",
            "npw_growth_yoy",
            "pif_growth_yoy",
            "unrealized_gain_pct_equity",
            "buyback_yield",
            "real_rate_10y",
        ],
        "x9_capital_generation": [
            "current_bvps",
            "bvps_growth_ytd",
            "bvps_growth_3m",
            "bvps_yoy_dollar_change",
            "capital_generation_proxy",
            "excess_capital_proxy",
            "buyback_vs_capital_generation",
            "premium_growth_x_margin_for_dividend",
            "premium_to_surplus_x_cr_delta",
            "buyback_yield_x_pb_ratio",
            "pgr_premium_to_surplus",
            "unrealized_gain_pct_equity",
        ],
    }
    return {
        name: [column for column in columns if column in frame.columns]
        for name, columns in candidates.items()
    }
