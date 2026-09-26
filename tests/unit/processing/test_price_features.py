"""WP2 (review 2026-09-25, F01/F15): price features on weekly, split-affected bars.

``daily_prices`` holds one unadjusted bar per week. These fixtures use weekly
bars with a 4-for-1 split so that a feature that counts rows as trading days,
or compares raw closes across the split, fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import src.processing.feature_engineering as fe
from src.processing.feature_engineering import (
    build_feature_matrix,
    build_feature_matrix_from_db,
)

SPLIT_DATE = pd.Timestamp("2006-05-19")
PRICE_FEATURES = ["mom_3m", "mom_6m", "mom_12m", "vol_63d", "high_52w"]


def _splits(split_date: pd.Timestamp | None = SPLIT_DATE, ratio: float = 4.0) -> pd.DataFrame:
    if split_date is None:
        return pd.DataFrame(
            {"split_ratio": [], "numerator": [], "denominator": []},
            index=pd.DatetimeIndex([], name="split_date"),
        )
    return pd.DataFrame(
        {"split_ratio": [ratio], "numerator": [ratio], "denominator": [1.0]},
        index=pd.DatetimeIndex([split_date], name="split_date"),
    )


def _no_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        {"dividend": pd.Series(dtype=float)},
        index=pd.DatetimeIndex([], name="ex_date"),
    )


def _adjusted_weekly(
    start: str = "2003-01-03",
    end: str = "2010-12-31",
    annual_growth: float = 0.10,
    wiggle: float = 0.0,
    base: float = 20.0,
) -> pd.Series:
    """Weekly Friday closes on the latest share basis.

    With ``wiggle == 0`` the close is constant within each calendar month and
    grows by exactly ``annual_growth`` every 12 months, so every calendar
    momentum is known in closed form.
    """
    dates = pd.date_range(start, end, freq="W-FRI")
    months = (dates.year - dates[0].year) * 12 + (dates.month - 1)
    close = base * (1.0 + annual_growth) ** (months / 12.0)
    if wiggle:
        close = close * (1.0 + wiggle * np.sin(np.arange(len(dates)) * 0.9))
    return pd.Series(close, index=pd.DatetimeIndex(dates, name="date"), name="close")


def _raw_from_adjusted(
    adjusted: pd.Series,
    split_date: pd.Timestamp | None = SPLIT_DATE,
    ratio: float = 4.0,
) -> pd.Series:
    """Unadjusted closes: bars before the split trade ``ratio`` times higher."""
    if split_date is None:
        return adjusted.copy()
    return adjusted.where(adjusted.index >= split_date, adjusted * ratio)


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
        },
        index=close.index,
    )


@pytest.fixture(autouse=True)
def _no_repo_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(fe, "_PROCESSED_PATH", str(tmp_path / "feature_matrix.parquet"))


def _build(close: pd.Series, splits: pd.DataFrame) -> pd.DataFrame:
    return build_feature_matrix(
        _ohlcv(close), _no_dividends(), splits, force_refresh=True
    )


# ---------------------------------------------------------------------------
# Calendar momentum on weekly bars with a 4:1 split
# ---------------------------------------------------------------------------

def test_mom_12m_equals_true_calendar_return_across_split() -> None:
    adjusted = _adjusted_weekly()
    df = _build(_raw_from_adjusted(adjusted), _splits())

    rows = df.loc["2004-01":"2010-12"]
    assert rows["mom_12m"].notna().all()
    np.testing.assert_allclose(rows["mom_12m"], 0.10, atol=1e-12)
    np.testing.assert_allclose(rows["mom_6m"], 1.10 ** 0.5 - 1.0, atol=1e-12)
    np.testing.assert_allclose(rows["mom_3m"], 1.10 ** 0.25 - 1.0, atol=1e-12)


def test_no_feature_is_discontinuous_across_split() -> None:
    """The split must have no effect on any feature (targets included)."""
    adjusted = _adjusted_weekly(wiggle=0.03)
    with_split = _build(_raw_from_adjusted(adjusted), _splits())
    pre_adjusted = _build(adjusted, _splits(None))

    assert list(with_split.columns) == list(pre_adjusted.columns)
    pd.testing.assert_frame_equal(with_split, pre_adjusted, rtol=1e-9, atol=1e-12)

    # And explicitly: month-to-month changes around the split look like the
    # rest of the sample.
    around = with_split.loc["2006-03":"2006-08", PRICE_FEATURES]
    elsewhere = with_split.loc["2008-01":"2010-12", PRICE_FEATURES]
    assert (around.diff().abs().max() <= elsewhere.diff().abs().max() * 1.5 + 1e-9).all()


def test_vol_63d_is_13_weekly_returns_annualised_by_sqrt_52() -> None:
    adjusted = _adjusted_weekly(wiggle=0.03)
    df = _build(_raw_from_adjusted(adjusted), _splits())

    for month_end in ["2006-05-31", "2006-06-30", "2009-03-31"]:
        bars = adjusted.loc[:month_end]
        log_ret = np.log(bars / bars.shift(1)).iloc[-13:]
        expected = float(log_ret.std(ddof=1) * np.sqrt(52))
        assert df.loc[month_end, "vol_63d"] == pytest.approx(expected, rel=1e-9)


def test_high_52w_uses_52_weekly_bars() -> None:
    adjusted = _adjusted_weekly(wiggle=0.03)
    df = _build(_raw_from_adjusted(adjusted), _splits())

    for month_end in ["2006-05-31", "2006-06-30", "2009-03-31"]:
        bars = adjusted.loc[:month_end]
        expected = float(bars.iloc[-1] / bars.iloc[-52:].max())
        assert df.loc[month_end, "high_52w"] == pytest.approx(expected, rel=1e-9)


def test_builder_rejects_bars_coarser_than_weekly() -> None:
    adjusted = _adjusted_weekly()
    monthly_bars = adjusted.groupby(adjusted.index.to_period("M")).tail(1)
    with pytest.raises(ValueError, match="weekly bars"):
        _build(monthly_bars, _splits(None))


# ---------------------------------------------------------------------------
# build_feature_matrix_from_db: per-share basis and synthetic spreads
# ---------------------------------------------------------------------------

# Splits in the synthetic DB. PGR's is real; the others are placed inside the
# fixture window so every spread straddles at least one split.
_DB_SPLITS: dict[str, tuple[str, float]] = {
    "PGR": ("2006-05-19", 4.0),
    "KIE": ("2007-12-07", 3.0),
    "CB": ("2006-04-21", 2.0),
    "VOO": ("2008-10-24", 0.5),
    "VWO": ("2008-06-20", 2.0),
}


def _db_prices(ticker: str) -> pd.DataFrame:
    adjusted = _adjusted_weekly(start="2002-01-04", end="2010-12-31")
    split = _DB_SPLITS.get(ticker)
    raw = (
        _raw_from_adjusted(adjusted, pd.Timestamp(split[0]), split[1])
        if split
        else adjusted
    )
    frame = _ohlcv(raw)
    frame["source"] = "test"
    frame["proxy_fill"] = 0
    return frame


def _db_splits(ticker: str) -> pd.DataFrame:
    split = _DB_SPLITS.get(ticker)
    if split is None:
        return pd.DataFrame()
    return _splits(pd.Timestamp(split[0]), split[1])


def _edgar_monthly() -> pd.DataFrame:
    """8-K rows on the share basis in effect at each report period."""
    months = pd.date_range("2002-01-31", "2010-12-31", freq="ME")
    step = np.arange(len(months))
    pre_split = months < SPLIT_DATE
    per_share = np.where(pre_split, 4.0, 1.0)
    bvps_latest = 8.0 * 1.01 ** step
    equity = 5_000.0 * 1.01 ** step
    return pd.DataFrame(
        {
            "book_value_per_share": bvps_latest * per_share,
            "eps_basic": 0.10 * 1.01 ** step * per_share,
            "shareholders_equity": equity,
            "shares_repurchased": 1.0 / per_share,
            "avg_cost_per_share": 20.0 * 1.01 ** step * per_share,
        },
        index=pd.DatetimeIndex(months, name="month_end"),
    )


@pytest.fixture()
def db_features(monkeypatch) -> pd.DataFrame:
    from src.database import db_client

    empty_dividends = pd.DataFrame(
        columns=["amount", "source"], index=pd.DatetimeIndex([], name="ex_date")
    )
    monkeypatch.setattr(
        db_client, "get_prices", lambda conn, ticker, *a, **k: _db_prices(ticker)
    )
    monkeypatch.setattr(db_client, "get_dividends", lambda conn, ticker: empty_dividends)
    monkeypatch.setattr(db_client, "get_splits", lambda conn, ticker: _db_splits(ticker))
    monkeypatch.setattr(db_client, "get_pgr_fundamentals", lambda conn: pd.DataFrame())
    monkeypatch.setattr(db_client, "get_pgr_edgar_monthly", lambda conn: _edgar_monthly())
    monkeypatch.setattr(db_client, "get_fred_macro", lambda conn, series: pd.DataFrame())
    return build_feature_matrix_from_db(conn=None, force_refresh=True)


def test_pb_ratio_log_change_is_small_across_2006_split(db_features: pd.DataFrame) -> None:
    pb = db_features["pb_ratio"].loc["2005-06":"2007-12"]
    assert pb.notna().all()
    assert np.log(pb).diff().abs().max() < 0.3


def test_bvps_growth_yoy_is_continuous_across_2006_split(db_features: pd.DataFrame) -> None:
    growth = db_features["book_value_per_share_growth_yoy"].loc["2005-06":"2007-12"]
    assert growth.notna().all()
    assert growth.abs().max() < 0.5
    np.testing.assert_allclose(growth, 1.01 ** 12 - 1.0, atol=1e-9)


def test_buyback_yield_is_continuous_across_2006_split(db_features: pd.DataFrame) -> None:
    buyback = db_features["buyback_yield"].loc["2005-06":"2007-12"]
    assert buyback.notna().all()
    assert np.log(buyback).diff().abs().max() < 0.3


@pytest.mark.parametrize(
    "feature",
    [
        "pgr_vs_kie_6m",
        "pgr_vs_peers_6m",
        "pgr_vs_vfh_6m",
        "commodity_equity_momentum",
        "vwo_vxus_spread_6m",
    ],
)
def test_synthetic_spreads_ignore_splits(db_features: pd.DataFrame, feature: str) -> None:
    """Every ticker has the same split-adjusted path, so every spread is 0."""
    spread = db_features[feature].loc["2003-01":"2010-12"]
    assert spread.notna().all()
    assert spread.abs().max() < 1e-9


def test_db_momentum_uses_split_adjusted_calendar_closes(db_features: pd.DataFrame) -> None:
    rows = db_features.loc["2003-01":"2010-12"]
    np.testing.assert_allclose(rows["mom_12m"], 0.10, atol=1e-12)


# ---------------------------------------------------------------------------
# TA shadow features (F24) and Monte Carlo volatility (F19)
# ---------------------------------------------------------------------------

def _ta_frames(split: bool) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    pgr_adjusted = _adjusted_weekly(wiggle=0.03)
    vwo_adjusted = _adjusted_weekly(annual_growth=0.05, wiggle=0.02, base=40.0)
    if split:
        pgr = _ohlcv(_raw_from_adjusted(pgr_adjusted))
        pgr["volume"] = np.where(pgr.index >= SPLIT_DATE, 4e6, 1e6)
        # Alpha Vantage's split-week bar mixes bases: open and high come from
        # the pre-split days, low and close from the post-split days
        # (PGR 2006-05-19: open 107.40, high 108.63, low 26.79, close 27.24).
        pgr.loc[SPLIT_DATE, ["open", "high"]] *= 4.0
        return {"PGR": pgr, "VWO": _ohlcv(vwo_adjusted)}, {"PGR": _splits()}
    pgr = _ohlcv(pgr_adjusted)
    pgr["volume"] = 4e6
    return {"PGR": pgr, "VWO": _ohlcv(vwo_adjusted)}, {}


def test_ta_features_ignore_splits_on_weekly_bars() -> None:
    from src.research.v160_ta_features import build_ta_feature_matrix

    raw_map, split_map = _ta_frames(split=True)
    adjusted_map, _ = _ta_frames(split=False)
    with_split = build_ta_feature_matrix(
        raw_map, benchmarks=("VWO",), peer_tickers=(), split_map=split_map
    )
    pre_adjusted = build_ta_feature_matrix(adjusted_map, benchmarks=("VWO",), peer_tickers=())

    pd.testing.assert_frame_equal(with_split, pre_adjusted, rtol=1e-9, atol=1e-12)
    # The 13-week NATR at the split month is ordinary (the high/low band is
    # ±1 %, so NATR is about 0.02 plus the weekly move).
    assert with_split.loc["2006-05-31", "ta_pgr_natr_63d"] < 0.1
    assert with_split.loc["2006-03":"2006-08", "ta_ratio_roc_6m_vwo"].abs().max() < 0.2


def test_split_week_bar_with_mixed_bases_is_restated() -> None:
    """Each of open/high/low takes the basis nearest the bar's close."""
    from src.processing.price_adjustment import split_adjusted_ohlcv

    bars = pd.DataFrame(
        {
            "open": [107.96, 107.40, 27.38],
            "high": [108.24, 108.63, 27.65],
            "low": [106.52, 26.79, 26.90],
            "close": [107.20, 27.24, 27.15],
            "volume": [1e6, 2e6, 4e6],
        },
        index=pd.DatetimeIndex(["2006-05-12", "2006-05-19", "2006-05-26"]),
    )
    adjusted = split_adjusted_ohlcv(bars, _splits())
    split_bar = adjusted.loc["2006-05-19"]
    assert split_bar["open"] == pytest.approx(107.40 / 4)
    assert split_bar["high"] == pytest.approx(108.63 / 4)
    assert split_bar["low"] == pytest.approx(26.79)
    assert split_bar["close"] == pytest.approx(27.24)
    assert adjusted.loc["2006-05-12", "high"] == pytest.approx(108.24 / 4)
    assert adjusted.loc["2006-05-26", "low"] == pytest.approx(26.90)

    # A 1-for-2 reverse split (VOO 2013-10-24): the low is pre-split.
    voo = pd.DataFrame(
        {
            "open": [77.51, 79.92, 161.15],
            "high": [79.93, 161.20, 162.62],
            "low": [77.44, 79.6861, 160.53],
            "close": [79.87, 161.20, 161.34],
        },
        index=pd.DatetimeIndex(["2013-10-18", "2013-10-25", "2013-11-01"]),
    )
    voo_adj = split_adjusted_ohlcv(voo, _splits(pd.Timestamp("2013-10-24"), 0.5))
    assert voo_adj.loc["2013-10-25", "open"] == pytest.approx(79.92 * 2)
    assert voo_adj.loc["2013-10-25", "high"] == pytest.approx(161.20)
    assert voo_adj.loc["2013-10-25", "low"] == pytest.approx(79.6861 * 2)


def test_ta_windows_count_weekly_bars() -> None:
    from src.research.v160_ta_features import build_ta_feature_matrix

    adjusted_map, _ = _ta_frames(split=False)
    features = build_ta_feature_matrix(adjusted_map, benchmarks=("VWO",), peer_tickers=())
    pgr = adjusted_map["PGR"]["close"]
    vwo = adjusted_map["VWO"]["close"]
    ratio = (pgr / vwo).loc[:"2009-03-31"]
    expected = float(ratio.iloc[-1] / ratio.iloc[-27] - 1.0)  # 26 weekly bars
    assert features.loc["2009-03-31", "ta_ratio_roc_6m_vwo"] == pytest.approx(expected)


def test_ta_builder_rejects_bars_coarser_than_weekly() -> None:
    from src.research.v160_ta_features import build_ta_feature_matrix

    adjusted_map, _ = _ta_frames(split=False)
    monthly = {
        ticker: frame.groupby(frame.index.to_period("M")).tail(1)
        for ticker, frame in adjusted_map.items()
    }
    with pytest.raises(ValueError, match="weekly bars"):
        build_ta_feature_matrix(monthly, benchmarks=("VWO",), peer_tickers=())


def test_monte_carlo_vol_uses_recent_adjusted_weekly_returns() -> None:
    from src.tax.monte_carlo import estimate_annual_vol_weekly

    # Weekly log returns alternate +r / -r, so the sample std is known; a
    # 4-for-1 split sits inside the lookback window.
    r = 0.02
    dates = pd.date_range("2005-06-03", periods=80, freq="W-FRI")
    log_path = np.cumsum(np.where(np.arange(80) % 2 == 0, r, -r))
    adjusted = pd.Series(30.0 * np.exp(log_path), index=dates)
    raw = _raw_from_adjusted(adjusted)
    assert dates[0] < SPLIT_DATE < dates[-1]

    vol = estimate_annual_vol_weekly(raw, _splits(), lookback_weeks=52)
    returns = np.diff(np.log(adjusted.to_numpy()))[-52:]
    expected = float(np.std(returns, ddof=1) * np.sqrt(52))
    assert vol == pytest.approx(expected, rel=1e-12)
    assert vol == pytest.approx(r * np.sqrt(52), rel=0.02)

    # Only the recent window counts: a turbulent earlier year is ignored.
    turbulent = adjusted.copy()
    turbulent.iloc[:20] *= np.where(np.arange(20) % 2 == 0, 1.3, 0.7)
    assert estimate_annual_vol_weekly(
        _raw_from_adjusted(turbulent), _splits(), lookback_weeks=52
    ) == pytest.approx(vol, rel=1e-12)
