from __future__ import annotations

import pandas as pd

from src.processing.feature_engineering import truncate_relative_target_for_asof


def test_truncate_relative_target_for_asof_masks_future_known_targets() -> None:
    index = pd.to_datetime(
        ["2025-09-30", "2025-10-31", "2025-11-30", "2025-12-31", "2026-01-31"]
    )
    series = pd.Series([0.01, -0.02, 0.03, -0.01, 0.00], index=index, name="VOO_6m")
    truncated = truncate_relative_target_for_asof(
        series,
        as_of=pd.Timestamp("2026-04-11"),
        horizon_months=6,
    )
    assert pd.notna(truncated.loc[pd.Timestamp("2025-09-30")])
    assert pd.notna(truncated.loc[pd.Timestamp("2025-10-31")])
    assert pd.isna(truncated.loc[pd.Timestamp("2025-11-30")])
    assert pd.isna(truncated.loc[pd.Timestamp("2026-01-31")])
