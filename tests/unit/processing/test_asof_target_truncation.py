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
    # 2025-09-30 + 6M ends 2026-03-31, before the as-of date: known.
    assert pd.notna(truncated.loc[pd.Timestamp("2025-09-30")])
    # 2025-10-31 + 6M ends 2026-04-30, after the as-of date 2026-04-11: its
    # realised value was not knowable yet (review F22), so it must be hidden.
    assert pd.isna(truncated.loc[pd.Timestamp("2025-10-31")])
    assert pd.isna(truncated.loc[pd.Timestamp("2025-11-30")])
    assert pd.isna(truncated.loc[pd.Timestamp("2026-01-31")])


def test_every_retained_target_window_ends_on_or_before_asof() -> None:
    from src.processing.total_return import forward_window_end

    index = pd.date_range("2018-01-31", "2026-08-31", freq="BME")
    series = pd.Series(0.01, index=index, name="VTI_6m")
    for as_of in pd.date_range("2020-01-01", "2026-09-25", freq="7D"):
        for horizon in (6, 12):
            truncated = truncate_relative_target_for_asof(series, as_of, horizon)
            kept = truncated.dropna().index
            assert all(forward_window_end(t, horizon) <= as_of for t in kept)
            # ...and nothing knowable is dropped.
            dropped = truncated.index[truncated.isna()]
            assert all(forward_window_end(t, horizon) > as_of for t in dropped)
