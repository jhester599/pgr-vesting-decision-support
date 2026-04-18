"""Tests for the v160 technical-analysis feature factory."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _price_frame(values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    close = pd.Series(values, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, len(values)),
        },
        index=dates,
    )


def test_rsi_is_centered_and_bounded() -> None:
    from src.research.v160_ta_features import relative_strength_index

    series = pd.Series(
        [10.0, 11.0, 10.5, 12.0, 13.0, 12.0, 14.0, 15.0],
        index=pd.bdate_range("2020-01-01", periods=8),
    )
    rsi = relative_strength_index(series, window=3)

    assert rsi.dropna().between(-1.0, 1.0).all()
    assert rsi.name == "rsi_centered"


def test_ema_gap_is_dimensionless_and_scale_invariant() -> None:
    from src.research.v160_ta_features import ema_gap

    series = pd.Series(
        [100.0, 102.0, 104.0, 106.0, 108.0],
        index=pd.bdate_range("2020-01-01", periods=5),
    )
    gap = ema_gap(series, span=3)
    scaled_gap = ema_gap(series * 10.0, span=3)

    pd.testing.assert_series_equal(gap, scaled_gap)
    assert gap.name == "ema_gap"


def test_bollinger_percent_b_uses_actual_close_not_middle_band() -> None:
    from src.research.v160_ta_features import bollinger_percent_b

    close = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 20.0],
        index=pd.bdate_range("2020-01-01", periods=5),
    )
    pct_b, bandwidth = bollinger_percent_b(close, window=5, num_std=2.0)

    middle = close.rolling(5, min_periods=5).mean()
    std = close.rolling(5, min_periods=5).std(ddof=0)
    lower = middle - 2.0 * std
    upper = middle + 2.0 * std
    expected = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

    assert abs(float(pct_b.iloc[-1]) - float(expected)) < 1e-12
    assert float(pct_b.iloc[-1]) != 0.5
    assert float(bandwidth.iloc[-1]) > 0.0


def test_natr_is_scale_invariant() -> None:
    from src.research.v160_ta_features import normalized_average_true_range

    prices = _price_frame([100.0, 101.0, 103.0, 102.0, 105.0, 108.0])
    natr = normalized_average_true_range(prices, window=3)
    scaled = prices.copy()
    scaled[["open", "high", "low", "close"]] *= 10.0
    scaled_natr = normalized_average_true_range(scaled, window=3)

    pd.testing.assert_series_equal(natr, scaled_natr)


def test_obv_detrending_removes_raw_cumulative_scale() -> None:
    from src.research.v160_ta_features import detrended_obv

    prices = _price_frame([100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 103.0, 105.0])
    obv_feature = detrended_obv(prices, span=3)
    raw_obv_abs_max = prices["volume"].cumsum().abs().max()

    assert obv_feature.dropna().abs().max() < raw_obv_abs_max / 10.0
    assert obv_feature.name == "obv_detrended"


def test_ratio_features_align_to_month_end_without_forward_lookup() -> None:
    from src.research.v160_ta_features import build_ta_feature_matrix

    pgr = _price_frame([100.0 + i for i in range(90)])
    voo = _price_frame([200.0 + i * 0.5 for i in range(90)])
    features = build_ta_feature_matrix(
        {"PGR": pgr, "VOO": voo},
        benchmarks=["VOO"],
        peer_tickers=[],
    )

    assert "ta_ratio_roc_6m_voo" in features.columns
    assert "ta_ratio_bb_pct_b_6m_voo" in features.columns
    assert features.index.is_monotonic_increasing
    assert all(ts == pd.offsets.BMonthEnd().rollback(ts) for ts in features.index)
    assert features.index.max() <= pgr.index.max()
