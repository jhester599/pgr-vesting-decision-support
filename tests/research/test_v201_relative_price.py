"""Hand-calculated split-neutral relative trends and causal perturbations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pgr_vds.research_lib import price_macro


def test_relative_trends_cancel_two_assets_split_actions() -> None:
    dates = pd.date_range("2010-01-01", "2012-12-31", freq="W-FRI")
    origins = pd.date_range("2010-01-29", "2011-12-30", freq="BME")
    pgr = pd.Series(100.0, index=dates)
    voo = pgr.copy()
    pgr.loc[pgr.index >= "2011-01-07"] = 50.0
    voo.loc[voo.index >= "2011-06-03"] = 200.0
    splits = {
        "PGR": pd.Series([2.0], index=pd.to_datetime(["2011-01-07"])),
        "VOO": pd.Series([0.5], index=pd.to_datetime(["2011-06-03"])),
    }
    result = price_macro.relative_trends(pgr, voo, splits, origins)
    # Both adjusted closes remain 100: ratio=1, EMA gap=0 and flat RSI=50.
    assert result.loc["2011-12-30", "pm_ema12_voo"] == pytest.approx(0)
    assert result.loc["2011-12-30", "pm_rsi6_voo"] == pytest.approx(50)
    assert result.pm_ema12_voo.iloc[:11].isna().all()
    mutated = pgr.copy()
    mutated.loc[mutated.index > origins.max()] = 99999.0
    future = {
        **splits,
        "PGR": pd.concat(
            [
                splits["PGR"],
                pd.Series([10.0], index=pd.to_datetime(["2012-03-02"])),
            ]
        ),
    }
    pd.testing.assert_frame_equal(
        result,
        price_macro.relative_trends(mutated, voo, future, origins),
    )
    missing = voo.copy()
    missing.loc[missing.index.to_period("M") == pd.Period("2011-11")] = np.nan
    unknown = price_macro.relative_trends(pgr, missing, splits, origins)
    assert np.isnan(unknown.loc["2011-11-30", "pm_ema12_voo"])
