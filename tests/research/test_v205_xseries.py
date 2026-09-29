"""Hand-calculated expected outputs for the v205 dividend/BVPS lanes.

Every expected number below was computed by hand from the fixture inputs,
not by calling the module under test. The fixtures cover the x-series
defects named in review findings F15, F16 and F25: split-inconsistent
BVPS, 13-month "year-over-year" gaps and December specials missing from a
January-March window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pgr_vds.research_lib import xseries


def _series(values: dict[str, float]) -> pd.Series:
    """Date-indexed float series in chronological order."""
    return pd.Series(
        list(values.values()),
        index=pd.DatetimeIndex(list(values.keys())),
        dtype=float,
    ).sort_index()


def _reports(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Monthly reports with an explicit actual filing date."""
    return pd.DataFrame(
        {
            "month_end": pd.to_datetime([row[0] for row in rows]),
            "filing_date": pd.to_datetime([row[1] for row in rows]),
            "book_value_per_share": [row[2] for row in rows],
        }
    )


SPLIT_2006 = _series({"2006-05-19": 4.0})
NO_SPLITS = pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def test_share_factor_counts_only_actions_after_start() -> None:
    assert xseries.share_factor(
        SPLIT_2006, pd.Timestamp("2006-04-30"), pd.Timestamp("2006-05-31")
    ) == 4.0
    assert xseries.share_factor(
        SPLIT_2006, pd.Timestamp("2006-05-19"), pd.Timestamp("2006-06-30")
    ) == 1.0
    with pytest.raises(ValueError):
        xseries.share_factor(
            _series({"2006-05-19": 0.0}),
            pd.Timestamp("2006-01-31"),
            pd.Timestamp("2006-12-29"),
        )


def test_cash_windows_use_the_origin_share_basis() -> None:
    dividends = _series(
        {"2006-01-10": 1.00, "2006-06-01": 0.10, "2006-12-01": 0.20}
    )
    value, end = xseries.next_cash_12m(
        dividends, SPLIT_2006, pd.Timestamp("2005-12-30")
    )
    # One pre-split origin share becomes four shares after 2006-05-19:
    # 1.00 + 4 * 0.10 + 4 * 0.20.
    assert value == pytest.approx(2.2, abs=1e-12)
    assert end == pd.Timestamp("2006-12-29")
    # One post-split origin share held a quarter share before the split.
    assert xseries.past_cash(
        dividends, SPLIT_2006, pd.Timestamp("2006-12-29"), 12, 0
    ) == pytest.approx(0.55, abs=1e-12)
    assert xseries.past_cash(
        dividends, SPLIT_2006, pd.Timestamp("2006-12-29"), 24, 12
    ) == 0.0


def test_bvps_growth_crosses_the_2006_split_on_one_share_basis() -> None:
    reports = _reports(
        [
            ("2006-03-31", "2006-04-12", 30.0),
            ("2006-04-30", "2006-05-17", 32.0),
            ("2007-04-30", "2007-05-16", 9.0),
        ]
    )
    target = xseries.bvps_growth_target(
        reports, SPLIT_2006, pd.Timestamp("2006-05-31"), 12
    )
    assert target is not None
    # 32 / 4 = 8 on the origin basis; 9 / 8 - 1. The raw F15 value
    # 9 / 32 - 1 = -0.71875 must never appear.
    assert target["value"] == pytest.approx(0.125, abs=1e-12)
    assert target["report_month"] == pd.Timestamp("2006-04-30")
    assert target["target_end"] == pd.Timestamp("2007-04-30")
    assert target["available"] == pd.Timestamp("2007-05-16")
    # At 2006-05-16 only March was filed and March 2007 is absent.
    assert xseries.bvps_growth_target(
        reports, SPLIT_2006, pd.Timestamp("2006-05-16"), 12
    ) is None


def test_adjusted_bvps_growth_includes_december_special() -> None:
    reports = _reports(
        [
            ("2019-10-31", "2019-11-13", 20.0),
            ("2020-04-30", "2020-05-15", 21.0),
        ]
    )
    dividends = _series(
        {
            "2019-10-04": 0.10,
            "2019-12-17": 0.50,
            "2020-01-07": 2.35,
            "2020-04-06": 0.10,
        }
    )
    target = xseries.adjusted_bvps_growth_target(
        reports, dividends, NO_SPLITS, pd.Timestamp("2019-11-29"), 6
    )
    assert target is not None
    # (21 + .50 + 2.35 + .10) / 20 - 1; the October payment precedes
    # the report month and is excluded.
    assert target["value"] == pytest.approx(0.1975, abs=1e-12)
    assert target["target_end"] == pd.Timestamp("2020-04-30")
    assert target["available"] == pd.Timestamp("2020-05-15")
    split_reports = _reports(
        [
            ("2019-10-31", "2019-11-13", 20.0),
            ("2020-04-30", "2020-05-15", 10.5),
        ]
    )
    split = xseries.adjusted_bvps_growth_target(
        split_reports,
        dividends,
        _series({"2020-02-01": 2.0}),
        pd.Timestamp("2019-11-29"),
        6,
    )
    assert split is not None
    # 10.5 * 2 + .50 + 2.35 + 2 * .10 = 24.05; 24.05 / 20 - 1.
    assert split["value"] == pytest.approx(0.2025, abs=1e-12)


def test_annual_excess_uses_december_to_february_and_x23_baseline() -> None:
    dividends = _series(
        {
            "2020-04-06": 0.10,
            "2020-07-06": 0.10,
            "2020-10-06": 0.10,
            "2021-01-07": 4.60,
            "2021-04-06": 0.10,
            "2021-07-06": 0.10,
            "2021-10-06": 0.10,
            "2021-12-17": 1.50,
            "2022-01-06": 0.10,
        }
    )
    reports = _reports([("2021-10-31", "2021-11-17", 30.0)])
    target = xseries.annual_excess_target(
        dividends, NO_SPLITS, reports, pd.Timestamp("2021-11-30")
    )
    assert target is not None
    # Ordinary baseline: median of positive payments <= .25 in the prior
    # 24 months is .10. December-February cash 1.60; excess 1.50 / 30.
    assert target["ordinary"] == pytest.approx(0.10, abs=1e-12)
    assert target["window_cash"] == pytest.approx(1.60, abs=1e-12)
    assert target["value"] == pytest.approx(0.05, abs=1e-12)
    assert target["positive"] is True
    assert target["available"] == pd.Timestamp("2022-02-28")
    assert target["event_id"] == "annual-dec-feb:2022"
    assert xseries.annual_window(pd.Timestamp("2019-11-29")) == (
        pd.Timestamp("2019-12-01"),
        pd.Timestamp("2020-02-29"),
    )
    # Pre-policy November origins are outside the x18/x23 endpoint.
    assert xseries.annual_excess_target(
        dividends, NO_SPLITS, reports, pd.Timestamp("2017-11-30")
    ) is None


def _feature_fixture(drop: str | None = None) -> tuple[
    pd.DataFrame, pd.Series
]:
    months = pd.date_range("2019-01-31", "2020-12-31", freq="ME")
    rows = []
    for index, month in enumerate(months):
        if drop is not None and month == pd.Timestamp(drop):
            continue
        in_2020 = month.year == 2020
        rows.append(
            {
                "month_end": month,
                "filing_date": month + pd.Timedelta(days=15),
                "book_value_per_share": (
                    20.0 + 0.5 * (index - 11) if in_2020 else 20.0
                ),
                "combined_ratio": 95.0 if in_2020 else 90.0,
                "pif_total": 110.0 if in_2020 else 100.0,
                "net_premiums_written": 12.0 if in_2020 else 10.0,
                "roe_net_income_ttm": 25.0,
                "gainshare_estimate": 1.5,
                "investment_book_yield": 2.5,
            }
        )
    # Filed after the 2021-01-29 origin, so it must not be used.
    rows.append(
        {
            "month_end": pd.Timestamp("2021-01-31"),
            "filing_date": pd.Timestamp("2021-02-15"),
            "book_value_per_share": 999.0,
            "combined_ratio": 999.0,
            "pif_total": 999.0,
            "net_premiums_written": 999.0,
            "roe_net_income_ttm": 999.0,
            "gainshare_estimate": 999.0,
            "investment_book_yield": 999.0,
        }
    )
    dividends = _series(
        {
            "2019-04-04": 0.10,
            "2019-07-03": 0.10,
            "2019-10-04": 0.10,
            "2020-01-07": 4.60,
            "2020-04-06": 0.10,
            "2020-07-06": 0.10,
            "2020-10-06": 0.10,
            "2021-01-07": 0.10,
            "2021-02-05": 9.99,
        }
    )
    return pd.DataFrame(rows), dividends


def test_rolling_monthly_features_use_only_filed_reports() -> None:
    reports, dividends = _feature_fixture()
    features = xseries.lane_features(
        reports, dividends, NO_SPLITS, pd.Timestamp("2021-01-29")
    )
    expected = {
        "report_month": pd.Timestamp("2020-12-31"),
        "report_filing_date": pd.Timestamp("2021-01-15"),
        "past12_cash": 0.4,
        "prior_past12_cash": 4.9,
        "cr_ttm": 95.0,
        "pif_growth_yoy_cal": 0.1,
        "npw_growth_ttm": 0.2,
        "current_bvps": 26.0,
        "bvps_growth_1m": 26.0 / 25.5 - 1.0,
        "bvps_growth_3m": 26.0 / 24.5 - 1.0,
        "bvps_growth_6m": 26.0 / 23.0 - 1.0,
        "bvps_growth_yoy": 0.3,
        "bvps_growth_ytd": 26.0 / 20.5 - 1.0,
        "bvps_yoy_dollar_change": 6.0,
        "roe_ttm": 25.0,
        "gainshare_estimate": 1.5,
        "investment_book_yield": 2.5,
        "month_of_year": 1.0,
        "q4_flag": 0.0,
        "dividend_season_flag": 1.0,
    }
    for name, value in expected.items():
        if isinstance(value, pd.Timestamp):
            assert features[name] == value, name
        else:
            assert features[name] == pytest.approx(value, abs=1e-12), name


def test_missing_month_gives_nan_not_a_thirteen_month_change() -> None:
    reports, dividends = _feature_fixture(drop="2020-05-31")
    features = xseries.lane_features(
        reports, dividends, NO_SPLITS, pd.Timestamp("2021-01-29")
    )
    assert np.isnan(features["cr_ttm"])
    assert np.isnan(features["npw_growth_ttm"])
    assert features["pif_growth_yoy_cal"] == pytest.approx(0.1, abs=1e-12)
    assert features["bvps_growth_yoy"] == pytest.approx(0.3, abs=1e-12)
    early = xseries.lane_features(
        reports, dividends, NO_SPLITS, pd.Timestamp("2019-06-28")
    )
    assert np.isnan(early["bvps_growth_yoy"])
    assert np.isnan(early["pif_growth_yoy_cal"])


def _ridge_frame(n: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=n, freq="BME")
    rng = np.random.default_rng(7)
    x = rng.normal(size=n)
    ends = [xseries.label_end(date, 12) for date in dates]
    return pd.DataFrame(
        {
            "x1": x,
            "y": 2.0 * x + 1.0,
            "label_end": ends,
            "available": ends,
        },
        index=dates,
    )


def test_nested_ridge_selects_least_shrinkage_for_exact_signal() -> None:
    frame = _ridge_frame()
    train = np.arange(0, 120)
    test = np.arange(144, 150)
    result = xseries.nested_ridge_forecast(
        frame, ["x1"], "y", train, test, 12, (1.0, 10.0, 100.0)
    )
    assert result["status"] == "scorable"
    assert result["alpha"] == 1.0
    assert len(result["inner"]) == 3
    assert all(entry["test_size"] == 6 for entry in result["inner"])
    truth = frame["y"].iloc[test].to_numpy()
    assert np.max(np.abs(result["prediction"] - truth)) < 0.1


def test_nested_ridge_ignores_labels_unavailable_at_the_test_origin() -> None:
    clean = _ridge_frame()
    clean.loc[clean.index[80:90], "y"] = np.nan
    poisoned = _ridge_frame()
    poisoned.loc[poisoned.index[80:90], "y"] = 1e6
    poisoned.loc[poisoned.index[80:90], "available"] = pd.Timestamp(
        "2100-01-29"
    )
    train = np.arange(0, 120)
    test = np.arange(144, 150)
    first = xseries.nested_ridge_forecast(
        clean, ["x1"], "y", train, test, 12, (1.0, 10.0, 100.0)
    )
    second = xseries.nested_ridge_forecast(
        poisoned, ["x1"], "y", train, test, 12, (1.0, 10.0, 100.0)
    )
    assert first["status"] == second["status"] == "scorable"
    np.testing.assert_allclose(first["prediction"], second["prediction"])
    assert first["alpha"] == second["alpha"]


def test_nested_ridge_ties_choose_more_shrinkage_and_flags_support() -> None:
    frame = _ridge_frame()
    frame["y"] = 0.0
    result = xseries.nested_ridge_forecast(
        frame, ["x1"], "y", np.arange(0, 120), np.arange(144, 150), 12,
        (1.0, 10.0, 100.0),
    )
    assert result["alpha"] == 100.0
    sparse = _ridge_frame()
    sparse.loc[sparse.index[:70], "y"] = np.nan
    unsupported = xseries.nested_ridge_forecast(
        sparse, ["x1"], "y", np.arange(0, 120), np.arange(144, 150), 12,
        (1.0, 10.0, 100.0),
    )
    assert unsupported["status"] == "unscorable"
    assert np.isnan(unsupported["prediction"]).all()


def test_paired_block_test_is_one_sided_and_centered() -> None:
    dates = pd.date_range("2015-01-31", periods=24, freq="BME")
    gain = xseries.paired_block_test(
        dates, np.full(24, 0.5), 12, replicates=200, seed=20260926
    )
    assert gain["observed"] == pytest.approx(0.5)
    assert gain["inference_supported"] is True
    # Every centered draw is 0, never >= .5.
    assert gain["p_value"] == pytest.approx(1.0 / 201.0)
    loss = xseries.paired_block_test(
        dates, np.full(24, -0.1), 12, replicates=200, seed=20260926
    )
    assert loss["p_value"] == 1.0
    short = xseries.paired_block_test(
        dates[:20], np.full(20, 0.5), 12, replicates=200, seed=20260926
    )
    assert short["inference_supported"] is False
    assert np.isnan(short["p_value"])


def test_prequential_stream_uses_only_matured_residuals() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="BME")
    residuals = [-2.0, -1.0, 0.0, 1.0, 2.0, 0.5]
    frame = pd.DataFrame(
        {
            "date": dates,
            "available": [date + pd.offsets.BMonthEnd(1) for date in dates],
            "y_hat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            "threshold": 0.0,
        }
    )
    frame["y_true"] = frame["y_hat"] + residuals
    stream = xseries.prequential_stream(frame, min_support=5)
    assert stream["warmup"].tolist() == [True] * 5 + [False]
    last = stream.iloc[-1]
    # Linear quantiles of [-2,-1,0,1,2] at .1/.9 are -1.6/+1.6.
    assert last["lower"] == pytest.approx(0.5 - 1.6)
    assert last["upper"] == pytest.approx(0.5 + 1.6)
    # .5 + r > 0 for r in {0,1,2}: 3 of 5.
    assert last["probability"] == pytest.approx(0.6)
    assert bool(last["covered"]) is True
    assert stream["probability"].iloc[:5].isna().all()


def test_proper_scores_and_equal_width_ece() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1])
    assert xseries.ece_equal_width(p, y) == pytest.approx(0.25)
    scores = xseries.probability_scores(p, y)
    assert scores["brier"] == pytest.approx(0.21)
    expected_log = -np.mean(
        [np.log(0.9), np.log(0.1), np.log(0.9), np.log(0.9)]
    )
    assert scores["log_loss"] == pytest.approx(expected_log)


def test_past_majority_direction_uses_mature_labels_only() -> None:
    dates = pd.date_range("2020-01-31", periods=4, freq="BME")
    available = pd.DatetimeIndex(
        [date + pd.offsets.BMonthEnd(1) for date in dates]
    )
    base = xseries.majority_base(
        dates,
        available,
        np.array([1, 0, 1, 1]),
        pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-31"]),
    )
    assert np.isnan(base[0])
    assert base[1] == 1.0
    # Mature [1, 0]: a .5 rate predicts the event.
    assert base[2] == 1.0


def _passing() -> dict[str, float | bool]:
    return {
        "relative_mae_reduction": 0.12,
        "adjusted_p": 0.01,
        "delta_r2": 0.02,
        "hit_rate": 0.70,
        "control_hit_rate": 0.60,
        "brier": 0.20,
        "control_brier": 0.21,
        "coverage_80": 0.80,
        "control_coverage_80": 0.70,
        "independent_blocks": 6.0,
        "inference_supported": True,
    }


def test_preregistered_disposition_requires_every_gate() -> None:
    assert xseries.lane_disposition(_passing())["passes"] is True
    failures = {
        "relative_mae_reduction": 0.09,
        "adjusted_p": 0.05,
        "delta_r2": 0.009,
        "hit_rate": 0.59,
        "brier": 0.22,
        "coverage_80": 0.60,
        "independent_blocks": 4.9,
        "inference_supported": False,
    }
    for name, value in failures.items():
        summary = _passing()
        summary[name] = value
        result = xseries.lane_disposition(summary)
        assert result["passes"] is False, name
        assert result["failed"], name
    summary = _passing()
    summary["brier"] = float("nan")
    assert xseries.lane_disposition(summary)["passes"] is False


def test_campaign_holm_places_v205_in_slots_29_to_34() -> None:
    adjusted = xseries.campaign_holm([0.001, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert adjusted[0] == pytest.approx(0.038)
    assert adjusted[1:] == [1.0] * 5
    two = xseries.campaign_holm([0.001, 0.002, 1.0, 1.0, 1.0, 1.0])
    assert two[:2] == pytest.approx([0.038, 0.074])
    with pytest.raises(ValueError):
        xseries.campaign_holm([0.5] * 5)


def test_at_most_one_winner_is_nominated() -> None:
    rows = [
        {"candidate": "B1", "passes": True, "adjusted_p": 0.02,
         "relative_mae_reduction": 0.2},
        {"candidate": "D1", "passes": True, "adjusted_p": 0.01,
         "relative_mae_reduction": 0.1},
        {"candidate": "B2", "passes": False, "adjusted_p": 0.001,
         "relative_mae_reduction": 0.5},
    ]
    assert xseries.nominate_winner(rows) == "D1"
    assert xseries.nominate_winner(
        [dict(row, passes=False) for row in rows]
    ) is None


def test_block_spearman_reports_centered_two_sided_ic() -> None:
    dates = pd.date_range("2015-01-31", periods=24, freq="BME")
    values = np.arange(24, dtype=float)
    result = xseries.block_spearman(
        dates, values, 2.0 * values + 1.0, 12, replicates=200, seed=1
    )
    # A monotone forecast has IC 1 in every resample, so no centered draw
    # reaches |1|.
    assert result["ic"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(1.0 / 201.0)
    reverse = xseries.block_spearman(
        dates, values, -values, 12, replicates=200, seed=1
    )
    assert reverse["ic"] == pytest.approx(-1.0)
    flat = xseries.block_spearman(
        dates, values, np.zeros(24), 12, replicates=200, seed=1
    )
    assert np.isnan(flat["ic"]) and np.isnan(flat["p_value"])


def test_restate_to_basis_puts_levels_on_one_share_basis() -> None:
    origins = pd.DatetimeIndex(["2006-04-28", "2006-06-30"])
    values = np.array([32.0, 8.5])
    # After the 4-for-1 split one April share is four June shares.
    np.testing.assert_allclose(
        xseries.restate_to_basis(
            values, origins, SPLIT_2006, pd.Timestamp("2006-06-30")
        ),
        [8.0, 8.5],
    )
    np.testing.assert_allclose(
        xseries.restate_to_basis(
            values, origins, SPLIT_2006, pd.Timestamp("2006-04-28")
        ),
        [32.0, 34.0],
    )
