"""Independent expected outputs for the frozen price/macro recipes."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest


def library() -> object:
    return import_module("pgr_vds.research_lib.price_macro")


def test_split_continuity_calendar_momentum_and_future_actions() -> None:
    dates = pd.date_range("1999-01-01", "2002-12-27", freq="W-FRI")
    underlying = np.where(dates.year < 2001, 100.0, 200.0)
    raw = pd.Series(underlying / np.where(dates >= "2000-05-19", 4, 1), dates)
    splits = pd.Series([4.0], index=pd.to_datetime(["2000-05-19"]))
    origins = pd.date_range("2000-01-31", "2001-12-31", freq="BME")
    result = library().price_features(raw, splits, origins)
    assert result.loc["2000-06-30", "pm_mom12"] == pytest.approx(0)
    assert result.loc["2001-06-29", "pm_mom12"] == pytest.approx(1)
    assert result.loc["2001-06-29", "pm_mom3"] == pytest.approx(0)
    raw.loc[raw.index > origins[-1]] = 999999
    splits.loc[pd.Timestamp("2002-05-01")] = 99
    pd.testing.assert_frame_equal(
        result, library().price_features(raw, splits, origins)
    )


def test_weekly_volatility_and_high_have_known_scale() -> None:
    dates = pd.date_range("2000-01-07", periods=60, freq="W-FRI")
    continuous = np.where(np.arange(60) % 2, 110.0, 100.0)
    raw = pd.Series(continuous / np.where(np.arange(60) >= 30, 2, 1), dates)
    splits = pd.Series([2.0], index=[dates[30]])
    origin = pd.DatetimeIndex([dates[-1] + pd.offsets.BMonthEnd(0)])
    result = library().price_features(raw, splits, origin).iloc[0]
    # Last 13 returns: seven +.1 and six -1/11, sample variance /12.
    mean = (7 * 0.1 - 6 / 11) / 13
    variance = (7 * (0.1 - mean) ** 2 + 6 * (-1 / 11 - mean) ** 2) / 12
    assert result.pm_vol13w == pytest.approx(np.sqrt(52 * variance))
    assert result.pm_high52w == pytest.approx(0)


def test_monthly_trend_flat_and_monotone_ratio() -> None:
    dates = pd.date_range("2000-01-31", periods=15, freq="BME")
    rising = pd.Series(np.arange(1, 16, dtype=float), dates)
    result = library().trend_features(rising)
    ema = 1.0
    for value in range(2, 16):
        ema = (2 / 13) * value + (11 / 13) * ema
    assert result.iloc[-1].pm_ema12_voo == pytest.approx(15 / ema - 1)
    assert result.iloc[-1].pm_rsi6_voo == pytest.approx(100)
    assert library().trend_features(rising * 0 + 2).iloc[-1].pm_rsi6_voo == 50
    assert result.iloc[:11].pm_ema12_voo.isna().all()


def test_one_calendar_lag_missing_period_and_duplicate_rejection() -> None:
    raw = pd.DataFrame(
        {
            "series_id": ["VIXCLS"] * 3,
            "month_end": pd.to_datetime(
                ["2020-03-31", "2020-04-30", "2020-06-30"]
            ),
            "value": [53.54, 34.15, 30.43],
        }
    )
    origins = pd.date_range("2020-04-30", periods=4, freq="BME")
    values, ledger = library().available_macro(raw, origins, {"VIXCLS": 1})
    assert values.loc["2020-04-30", "VIXCLS"] == 53.54
    assert values.loc["2020-05-29", "VIXCLS"] == 34.15
    assert np.isnan(values.loc["2020-06-30", "VIXCLS"])
    assert (
        ledger.loc[ledger.origin == origins[2], "status"].iloc[0] == "missing"
    )
    raw.loc[raw.month_end > "2020-04-30", "value"] = 1e9
    perturbed, _ = library().available_macro(raw, origins, {"VIXCLS": 1})
    pd.testing.assert_frame_equal(values.iloc[:2], perturbed.iloc[:2])
    with pytest.raises(ValueError, match="Duplicate"):
        library().available_macro(
            pd.concat([raw, raw.iloc[:1]]), origins, {"VIXCLS": 1}
        )
    with pytest.raises(ValueError, match="raw"):
        library().available_macro(
            raw, origins, {"VIXCLS": 1}, raw_unlagged=False
        )


def test_macro_differences_gap_and_future_values() -> None:
    dates = pd.date_range("2000-01-31", periods=19, freq="BME")
    frame = pd.DataFrame(
        {
            "T10Y2Y": 2.0,
            "GS10": np.arange(19, dtype=float),
            "T10YIE": 1.0,
            "VIXCLS": 20.0,
            "NFCI": 0.2,
            "BAMLH0A0HYM2": 3.0,
            "PCU5241265241261": np.arange(100, 119, dtype=float),
            "CUSR0000SETA02": 100.0,
            "CUSR0000SAM2": 100.0,
        },
        index=dates,
    )
    output = library().macro_features(frame)
    assert output.iloc[12].pm_rate_gap == pytest.approx(0.12)
    assert output.iloc[12].pm_real_change6 == pytest.approx(6)
    assert output.iloc[12].pm_slope == pytest.approx(2)
    frame.iloc[14:] = 9999
    pd.testing.assert_frame_equal(
        output.iloc[:14], library().macro_features(frame).iloc[:14]
    )


def test_paired_date_blocks_known_improvement_and_replication() -> None:
    dates = pd.date_range("2000-01-31", periods=12, freq="BME")
    control = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "benchmark": ["A", "B"] * 12,
            "y_true": 1.0,
            "y_hat": 0.0,
            "naive": 0.0,
        }
    )
    candidate = control.copy()
    candidate["y_hat"] = 0.5
    output = library().paired_comparison(control, candidate, 6)
    assert output["delta_r2"] == pytest.approx(0.75)
    assert output["mean_date_loss_gain"] == pytest.approx(0.75)
    assert output["primary_p"] == pytest.approx(1 / 2001)
    assert output["delta_r2_ci"] == pytest.approx([0.75, 0.75])
    assert library().paired_comparison(candidate, control, 6)["primary_p"] == 1
    bad = candidate.iloc[:-1]
    with pytest.raises(ValueError, match="support"):
        library().paired_comparison(control, bad, 6)


def test_calendar_gaps_remain_gaps() -> None:
    dates = pd.to_datetime(
        ["2000-01-31", "2000-02-29", "2000-04-28", "2000-05-31"]
    )
    with pytest.raises(ValueError, match="monthly"):
        library().trend_features(pd.Series([1, 2, 3, 4], dates))


def test_invalid_final_week_cannot_reuse_earlier_month_value() -> None:
    """A 14-day endpoint gap invalidates the latest return/window."""
    dates = pd.date_range("2000-01-07", periods=60, freq="W-FRI")
    dates = dates.delete(-2)
    prices = pd.Series(np.arange(len(dates), dtype=float) + 100, dates)
    origins = pd.date_range(dates[0], dates[-1], freq="BME")
    origins = origins.append(
        pd.DatetimeIndex(
            [
                dates[-1] + pd.offsets.BMonthEnd(0),
            ]
        )
    ).unique()
    result = library().price_features(
        prices,
        pd.Series(dtype=float),
        origins,
    )
    assert np.isnan(result.iloc[-1].pm_vol13w)
    assert np.isnan(result.iloc[-1].pm_high52w)
