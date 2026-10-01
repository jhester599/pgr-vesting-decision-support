"""Independent hand-calculated fixtures for v202 insurance mathematics."""

import numpy as np
import pandas as pd
import pytest

from pgr_vds.research_lib.insurance_valuation import (
    fiscal_month_shift,
    insurance_features,
    paired_delta_bootstrap,
    release_month,
    select_causal_recipe,
    split_rebased_growth,
    trailing_twelve_growth,
)


def test_release_month_uses_actual_date_and_conservative_fallback() -> None:
    assert release_month("2020-01-31", "2020-02-13") == pd.Timestamp(
        "2020-02-28"
    )
    assert release_month("2020-01-31", None) == pd.Timestamp(
        "2020-03-31"
    )


def test_split_rebased_growth_removes_four_for_one_discontinuity() -> None:
    assert split_rebased_growth(30.0, 8.0, 4.0) == pytest.approx(
        8.0 / 7.5 - 1.0
    )
    assert np.isnan(split_rebased_growth(30.0, 8.0, 0.0))


def test_trailing_twelve_growth_requires_complete_calendar() -> None:
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    values = pd.Series([10.0] * 12 + [12.0] * 12, index=index)
    assert trailing_twelve_growth(values).iloc[-1] == pytest.approx(0.2)
    values.iloc[4] = np.nan
    assert np.isnan(trailing_twelve_growth(values).iloc[-1])


def test_fiscal_month_shift_is_pattern_not_yoy_volatility() -> None:
    rows = []
    for year in range(2016, 2027):
        for month in (2, 3):
            ratio = (
                (1.20 if month == 2 else 1.05)
                if year < 2024
                else (1.10 if month == 2 else 1.32)
            )
            rows.append((year, month, ratio * 100.0, 100.0))
    frame = pd.DataFrame(rows, columns=["year", "month", "npw", "npe"])
    result = fiscal_month_shift(frame)
    assert result["confirmed"] is True
    assert result["pre_peak_month"] == 2
    assert result["post_peak_month"] == 3


def test_filing_gate_calendar_yoy_and_negative_earnings() -> None:
    months = pd.date_range("2019-01-31", periods=25, freq="ME")
    rows = pd.DataFrame({
        "month_end": months,
        "filing_date": months + pd.Timedelta(days=15),
        "combined_ratio": [90.0] * 25,
        "pif_agency_auto": [100.0] * 25,
        "pif_direct_auto": [50.0] * 25,
        "pif_commercial_lines": [50.0] * 25,
        "net_premiums_written": [10.0] * 12 + [12.0] * 13,
        "investment_income": [2.0] * 25,
        "investment_book_yield": [4.0] * 25,
        "book_value_per_share": [8.0] * 25,
        "shareholders_equity": [800.0] * 25,
        "common_shares_outstanding": [100.0] * 25,
        "net_income": [1.0] * 24 + [-20.0],
    })
    rows.loc[rows.index[-1], "filing_date"] = pd.Timestamp("2021-03-20")
    origins = pd.DatetimeIndex(["2021-02-26", "2021-03-31"])
    price = pd.Series([80.0, 80.0], index=origins)
    features, ledger = insurance_features(rows, pd.Series(dtype=float), origins, price)
    assert features.loc[origins[0], "iv_npw_t12_yoy"] == pytest.approx(0.2)
    assert features.loc[origins[1], "iv_npw_t12_yoy"] == pytest.approx(
        144.0 / 122.0 - 1.0
    )
    assert np.isnan(features.loc[origins[1], "iv_pe"])
    assert features.loc[origins[1], "iv_pb"] == pytest.approx(10.0)
    assert ledger.loc[
        (ledger.origin == origins[0]) & (ledger.feature == "iv_pb"),
        "report_month",
    ].iloc[0] == pd.Timestamp("2020-12-31")


def test_upstream_selection_uses_only_mature_inner_validation() -> None:
    dates = pd.date_range("2010-01-29", periods=60, freq="BME")
    rows = []
    for date in dates:
        rows.extend([
            (date, "A", 1.0, 1.0, date + pd.offsets.BMonthEnd(6)),
            (date, "B", 1.0, -1.0, date + pd.offsets.BMonthEnd(6)),
        ])
    frame = pd.DataFrame(
        rows, columns=["date", "recipe", "y_true", "y_hat", "available"]
    )
    origin = dates[-1] + pd.offsets.BMonthEnd(12)
    selected = select_causal_recipe(frame, dates, origin, horizon=6)
    assert selected == "A"
    frame.loc[frame.available > origin, "y_true"] = -99.0
    assert select_causal_recipe(frame, dates, origin, horizon=6) == "A"


def test_paired_delta_bootstrap_reports_zero_spread_for_equal_forecasts() -> None:
    dates = pd.date_range("2010-01-29", periods=24, freq="BME")
    frame = pd.DataFrame({
        "date": dates,
        "benchmark": "VOO",
        "y_true": np.linspace(-0.2, 0.2, 24),
        "naive": 0.0,
        "y_hat": 0.05,
    })
    result = paired_delta_bootstrap(frame, frame, 6, 20, 20260926)
    assert result["delta_r2"] == pytest.approx(0.0)
    assert result["bootstrap_delta_r2_sd"] == pytest.approx(0.0)
    assert result["approx_first_holm_delta_r2"] == pytest.approx(0.0)
