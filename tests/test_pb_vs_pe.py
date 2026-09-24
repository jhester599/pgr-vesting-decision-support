"""Tests for src/research/pb_vs_pe.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.research.pb_vs_pe import (
    block_bootstrap_ic,
    build_asof_signals,
    expanding_residual,
    forward_log_returns,
    noise_diagnostics,
    one_way_stats,
    oos_predictions,
    oos_r2,
    window_ic,
)

NO_SPLITS = pd.DataFrame(columns=["split_ratio"], index=pd.DatetimeIndex([]))


def _valuation(periods: list[str], bvps: list[float], ttm: list[float], avail: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_end": periods,
            "data_available_date": avail,
            "book_value_per_share": bvps,
            "eps_basic_ttm": ttm,
        }
    )


def test_asof_signals_wait_for_data_available_date() -> None:
    valuation = _valuation(
        ["2020-01-31", "2020-02-29", "2020-03-31"],
        bvps=[10.0, 20.0, 40.0],
        ttm=[1.0, 2.0, 4.0],
        avail=["2020-02-15", "2020-04-20", "2020-04-15"],
    )
    prices = pd.DataFrame(
        {"close": [30.0, 30.0, 30.0, 30.0]},
        index=pd.to_datetime(["2020-02-28", "2020-03-31", "2020-04-17", "2020-04-30"]),
    )
    months = pd.DatetimeIndex(pd.to_datetime(["2020-02-28", "2020-03-31", "2020-04-17", "2020-04-30"]))

    out = build_asof_signals(valuation, prices, NO_SPLITS, months)

    # Feb and Mar only know the January filing.
    assert out["period"].iloc[0] == pd.Timestamp("2020-01-31")
    assert out["period"].iloc[1] == pd.Timestamp("2020-01-31")
    # Apr 17: March is public (Apr 15) even though February is not (Apr 20).
    assert out["period"].iloc[2] == pd.Timestamp("2020-03-31")
    assert out["pb"].iloc[2] == pytest.approx(30.0 / 40.0)
    assert out["pe"].iloc[0] == pytest.approx(30.0)
    assert out["bp"].iloc[0] == pytest.approx(-np.log(3.0))
    assert out["ep"].iloc[0] == pytest.approx(1.0 / 30.0)


def test_asof_signals_restate_old_filing_across_split() -> None:
    split_date = pd.Timestamp("2006-05-19")
    splits = pd.DataFrame({"split_ratio": [4.0]}, index=pd.DatetimeIndex([split_date]))
    # April filing (pre-split basis) is still the latest one known at May-end.
    valuation = _valuation(["2006-04-30"], bvps=[32.0], ttm=[9.6], avail=["2006-05-17"])
    prices = pd.DataFrame({"close": [27.0]}, index=pd.to_datetime(["2006-05-31"]))

    out = build_asof_signals(valuation, prices, splits, pd.DatetimeIndex(["2006-05-31"]))

    assert out["pb"].iloc[0] == pytest.approx(27.0 / 8.0)
    assert out["pe"].iloc[0] == pytest.approx(27.0 / 2.4)


def test_expanding_residual_is_exact_and_uses_no_future_data() -> None:
    idx = pd.date_range("2000-01-31", periods=80, freq="ME")
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=80), index=idx)
    y = 2.0 + 3.0 * x
    y.iloc[70] += 1.0

    resid = expanding_residual(y, x, min_obs=60)

    assert resid.iloc[:59].isna().all()
    assert resid.iloc[59:70].abs().max() == pytest.approx(0.0, abs=1e-9)
    y_changed = y.copy()
    y_changed.iloc[75:] += 50.0
    assert expanding_residual(y_changed, x, min_obs=60).iloc[:75].equals(resid.iloc[:75])


def test_forward_log_returns() -> None:
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    pgr = {12: pd.Series([0.10, -0.20], index=idx)}
    mkt = {12: pd.Series([0.05, 0.00], index=idx)}

    out = forward_log_returns(pgr, mkt)

    assert out["abs_12m"].tolist() == pytest.approx([np.log(1.10), np.log(0.80)])
    assert out["rel_12m"].tolist() == pytest.approx([np.log(1.10 / 1.05), np.log(0.80)])


def _synthetic_frame(n: int = 200, signal_strength: float = 1.0, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"sig": x, "abs_6m": signal_strength * x + noise, "rel_6m": noise}, index=idx)


def test_one_way_stats_detects_positive_signal() -> None:
    frame = _synthetic_frame()

    table = one_way_stats(frame, frame, ["sig"], [6], n_boot=200).set_index("target")

    assert table.loc["abs_6m", "ic"] > 0.9
    assert table.loc["abs_6m", "nw_t"] > 5
    assert table.loc["abs_6m", "ic_lo90"] > 0.8
    assert table.loc["abs_6m", "n_independent"] == pytest.approx(200 / 6)
    assert abs(table.loc["rel_6m", "ic"]) < 0.2


def test_block_bootstrap_is_reproducible_and_brackets_estimate() -> None:
    frame = _synthetic_frame(signal_strength=0.3)
    x, y = frame["sig"].to_numpy(), frame["abs_6m"].to_numpy()

    first = block_bootstrap_ic(x, y, block=6, n_boot=300, seed=7)
    second = block_bootstrap_ic(x, y, block=6, n_boot=300, seed=7)

    point = stats.spearmanr(x, y)[0]
    assert first == second
    assert first[0] < point < first[1]


def test_oos_walk_forward_rewards_real_signal_and_penalizes_noise() -> None:
    frame = _synthetic_frame()

    real = oos_predictions(frame, ["sig"], "abs_6m", horizon=6)
    noise = oos_predictions(frame, ["sig"], "rel_6m", horizon=6)

    # 60 training months + 6-month purge before the first 12-month test block.
    assert real.index.min() == frame.index[200 - 12 * ((200 - 60 - 6) // 12)]
    assert real.index.min() >= frame.index[60 + 6]
    assert oos_r2(real) > 0.9
    assert oos_r2(noise) < 0.05


def test_noise_diagnostics_price_share_is_one_when_denominator_constant() -> None:
    idx = pd.date_range("2010-01-31", periods=40, freq="ME")
    price = pd.Series(np.linspace(20, 40, 40), index=idx) * (1 + 0.05 * np.sin(np.arange(40)))
    signals = pd.DataFrame(
        {"price": price, "bvps": 10.0, "ttm_eps": 2.0, "pb": price / 10.0, "pe": price / 2.0}
    )

    table = noise_diagnostics(signals).set_index("multiple")

    assert table.loc["P/B", "price_share_of_variance"] == pytest.approx(1.0)
    assert table.loc["P/B", "denominator_log_change_sd"] == pytest.approx(0.0)
    assert table.loc["P/E", "median"] == pytest.approx((price / 2.0).median())


def test_window_ic_selects_by_signal_date() -> None:
    frame = _synthetic_frame()
    frame.loc["2012-01-01":, "abs_6m"] *= -1

    table = window_ic(
        frame, ["sig"], ["abs_6m"], [("2005-01-01", "2011-12-31"), ("2012-01-01", "2030-01-01")]
    )

    assert table["ic"].iloc[0] > 0.9
    assert table["ic"].iloc[1] < -0.9
