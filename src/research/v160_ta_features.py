"""Research-only technical-analysis feature factory for v160-v164.

The functions in this module intentionally avoid TA-Lib/pandas-ta and operate
only on point-in-time price frames already available in the project database.
They are not wired into the production monthly decision path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

PRIMARY_TA_BENCHMARKS: tuple[str, ...] = (
    "VOO",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
)
PEER_TICKERS: tuple[str, ...] = ("ALL", "TRV", "CB", "HIG")


def _price_column(price_df: pd.DataFrame) -> pd.Series:
    """Return adjusted close when present, otherwise close."""
    column = "adjusted_close" if "adjusted_close" in price_df.columns else "close"
    series = price_df[column].astype(float).copy()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index))
    return series.sort_index()


def _monthly_last(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Sample a daily/irregular series on the last business day of each month."""
    result = values.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    result = result.sort_index()
    month_end = result.resample("BME").last()
    return month_end.loc[month_end.index <= result.index.max()]


def ema_gap(close: pd.Series, span: int) -> pd.Series:
    """Return ``(close - EMA) / EMA`` as a dimensionless trend-gap feature."""
    close = close.astype(float)
    ema = close.ewm(span=span, adjust=False, min_periods=max(1, span // 2)).mean()
    gap = (close - ema) / ema.replace(0.0, np.nan)
    gap.name = "ema_gap"
    return gap


def relative_strength_index(close: pd.Series, window: int) -> pd.Series:
    """Return RSI centered to ``[-1, 1]`` via ``(RSI - 50) / 50``."""
    close = close.astype(float)
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(window, min_periods=max(1, window // 2)).mean()
    avg_loss = losses.rolling(window, min_periods=max(1, window // 2)).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    centered = ((rsi - 50.0) / 50.0).clip(-1.0, 1.0)
    centered.name = "rsi_centered"
    return centered


def bollinger_percent_b(
    close: pd.Series,
    window: int,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """Return Bollinger ``%B`` and bandwidth from the actual close series."""
    close = close.astype(float)
    middle = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    pct_b = (close - lower) / width.replace(0.0, np.nan)
    bandwidth = width / middle.replace(0.0, np.nan)
    pct_b.name = "bb_pct_b"
    bandwidth.name = "bb_width"
    return pct_b, bandwidth


def normalized_average_true_range(price_df: pd.DataFrame, window: int) -> pd.Series:
    """Return rolling ATR divided by close for scale-invariant volatility."""
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    close = price_df["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window, min_periods=max(1, window // 2)).mean()
    natr = atr / close.replace(0.0, np.nan)
    natr.name = "natr"
    return natr


def average_directional_index(price_df: pd.DataFrame, window: int) -> pd.Series:
    """Return ADX scaled to ``[0, 1]``."""
    high = price_df["high"].astype(float)
    low = price_df["low"].astype(float)
    close = price_df["close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_sum = true_range.rolling(window, min_periods=max(1, window // 2)).sum()
    plus_di = 100.0 * plus_dm.rolling(window, min_periods=max(1, window // 2)).sum() / tr_sum
    minus_di = 100.0 * minus_dm.rolling(window, min_periods=max(1, window // 2)).sum() / tr_sum
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.rolling(window, min_periods=max(1, window // 2)).mean() / 100.0
    adx.name = "adx"
    return adx.clip(0.0, 1.0)


def macd_histogram_normalized(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """Return MACD histogram divided by close for scale invariance."""
    close = close.astype(float)
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=max(1, fast // 2)).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=max(1, slow // 2)).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=max(1, signal // 2)).mean()
    hist = (macd - signal_line) / close.replace(0.0, np.nan)
    hist.name = "macd_hist_norm"
    return hist


def detrended_obv(price_df: pd.DataFrame, span: int = 63) -> pd.Series:
    """Return OBV distance from its EMA, normalized by trailing volume."""
    close = price_df["close"].astype(float)
    volume = price_df["volume"].fillna(0.0).astype(float)
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    obv_ema = obv.ewm(span=span, adjust=False, min_periods=max(1, span // 2)).mean()
    volume_scale = volume.rolling(span, min_periods=max(1, span // 2)).mean() * span
    feature = (obv - obv_ema) / volume_scale.replace(0.0, np.nan)
    feature.name = "obv_detrended"
    return feature


def _ratio_series(pgr: pd.Series, benchmark: pd.Series) -> pd.Series:
    aligned = pd.concat([pgr.rename("pgr"), benchmark.rename("benchmark")], axis=1).dropna()
    ratio = aligned["pgr"] / aligned["benchmark"].replace(0.0, np.nan)
    ratio.name = "ratio"
    return ratio


def _pc_tech(close: pd.Series, price_df: pd.DataFrame | None = None) -> pd.Series:
    components = [
        close.pct_change(126, fill_method=None).rename("roc"),
        ema_gap(close, span=252).rename("ema_gap"),
        relative_strength_index(close, window=126).rename("rsi"),
        bollinger_percent_b(close, window=126)[0].rename("bb_pct_b"),
    ]
    if price_df is not None and {"high", "low", "close"}.issubset(price_df.columns):
        components.append(normalized_average_true_range(price_df, window=63).rename("natr"))
    frame = pd.concat(components, axis=1)
    z = (frame - frame.rolling(252, min_periods=63).mean()) / frame.rolling(
        252,
        min_periods=63,
    ).std(ddof=0)
    pc = z.mean(axis=1, skipna=True)
    pc.name = "pc_tech"
    return pc


def build_ta_feature_matrix(
    price_map: Mapping[str, pd.DataFrame],
    benchmarks: Sequence[str] = PRIMARY_TA_BENCHMARKS,
    peer_tickers: Sequence[str] = PEER_TICKERS,
) -> pd.DataFrame:
    """Build monthly research-only TA features from available price frames."""
    if "PGR" not in price_map:
        raise ValueError("price_map must include PGR.")

    pgr_df = price_map["PGR"].copy()
    pgr_close = _price_column(pgr_df)
    monthly_index = pd.DatetimeIndex(_monthly_last(pgr_close).index)
    features = pd.DataFrame(index=monthly_index)
    features.index.name = "date"

    pgr_natr = normalized_average_true_range(pgr_df, window=63)
    pgr_adx = average_directional_index(pgr_df, window=63)
    features["ta_pgr_natr_63d"] = _monthly_last(pgr_natr).reindex(monthly_index)
    features["ta_pgr_adx_63d"] = _monthly_last(pgr_adx).reindex(monthly_index)
    features["ta_pgr_macd_hist_norm"] = _monthly_last(
        macd_histogram_normalized(pgr_close)
    ).reindex(monthly_index)
    if "volume" in pgr_df.columns:
        features["ta_pgr_obv_detrended"] = _monthly_last(
            detrended_obv(pgr_df, span=63)
        ).reindex(monthly_index)

    for benchmark in benchmarks:
        if benchmark not in price_map:
            continue
        suffix = benchmark.lower()
        benchmark_close = _price_column(price_map[benchmark])
        ratio = _ratio_series(pgr_close, benchmark_close)
        ratio_roc = ratio.pct_change(126, fill_method=None)
        ratio_rsi = relative_strength_index(ratio, window=126)
        ratio_bb_pct_b, ratio_bb_width = bollinger_percent_b(ratio, window=126)
        features[f"ta_ratio_roc_6m_{suffix}"] = _monthly_last(ratio_roc).reindex(
            monthly_index
        )
        features[f"ta_ratio_ema_gap_12m_{suffix}"] = _monthly_last(
            ema_gap(ratio, span=252)
        ).reindex(monthly_index)
        features[f"ta_ratio_rsi_6m_{suffix}"] = _monthly_last(ratio_rsi).reindex(
            monthly_index
        )
        features[f"ta_ratio_bb_pct_b_6m_{suffix}"] = _monthly_last(
            ratio_bb_pct_b
        ).reindex(monthly_index)
        features[f"ta_ratio_bb_width_6m_{suffix}"] = _monthly_last(
            ratio_bb_width
        ).reindex(monthly_index)
        features[
            f"ta_ratio_roc_6m_{suffix}__x__ta_pgr_natr_63d"
        ] = features[f"ta_ratio_roc_6m_{suffix}"] * features["ta_pgr_natr_63d"]
        features[
            f"ta_ratio_roc_6m_{suffix}__x__ta_pgr_adx_63d"
        ] = features[f"ta_ratio_roc_6m_{suffix}"] * features["ta_pgr_adx_63d"]

    for ticker in ("VOO", "BND", "GLD", "DBC", "VDE"):
        if ticker not in price_map:
            continue
        close = _price_column(price_map[ticker])
        suffix = ticker.lower()
        if ticker == "VOO":
            features["ta_voo_pc_tech"] = _monthly_last(
                _pc_tech(close, price_map[ticker])
            ).reindex(monthly_index)
        elif ticker == "BND":
            features["ta_bnd_macd_hist_norm"] = _monthly_last(
                macd_histogram_normalized(close)
            ).reindex(monthly_index)
        else:
            features[f"ta_{suffix}_roc_6m"] = _monthly_last(
                close.pct_change(126, fill_method=None)
            ).reindex(monthly_index)

    peer_frames = [
        _price_column(price_map[ticker]).rename(ticker)
        for ticker in peer_tickers
        if ticker in price_map
    ]
    if peer_frames:
        peer_close = pd.concat(peer_frames, axis=1).dropna(how="all").mean(axis=1)
        peer_ratio = _ratio_series(pgr_close, peer_close)
        features["ta_peer_ratio_roc_6m"] = _monthly_last(
            peer_ratio.pct_change(126, fill_method=None)
        ).reindex(monthly_index)
        features["ta_peer_ratio_rsi_6m"] = _monthly_last(
            relative_strength_index(peer_ratio, window=126)
        ).reindex(monthly_index)
        features["ta_peer_ratio_ema_gap_12m"] = _monthly_last(
            ema_gap(peer_ratio, span=252)
        ).reindex(monthly_index)

    return features.astype("float64")
