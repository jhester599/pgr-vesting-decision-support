"""Research-only x19 post-policy dividend helpers."""

from __future__ import annotations

import pandas as pd


def filter_post_policy_annual_frame(annual: pd.DataFrame) -> pd.DataFrame:
    """Keep only eligible post-policy annual snapshots."""
    result = annual.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    result = result.sort_index()
    if "model_eligible_post_policy" in result.columns:
        result = result[result["model_eligible_post_policy"] == 1]
    return result


def build_x19_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return bounded post-policy feature sets for x19."""
    candidates = {
        "x10_capital_generation": [
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
        "persistent_capital_generation": [
            "persistent_bvps",
            "persistent_bvps_growth_ytd",
            "persistent_bvps_growth_3m",
            "persistent_bvps_yoy_dollar_change",
            "capital_generation_proxy",
            "excess_capital_proxy",
            "buyback_vs_capital_generation",
            "premium_growth_x_margin_for_dividend",
            "pgr_premium_to_surplus",
            "unrealized_gain_pct_equity",
            "book_value_creation_proxy",
            "persistent_bvps_to_price",
        ],
    }
    return {
        name: [column for column in columns if column in frame.columns]
        for name, columns in candidates.items()
    }
