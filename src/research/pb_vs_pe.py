"""
Research: is price-to-book or price-to-earnings the better PGR valuation gauge?

"Better" means: a cheaper reading predicts higher forward returns.  Every
signal here is oriented so that a larger value means *cheaper* (a yield or a
negated premium), so a useful signal has a positive relationship with
forward returns.

Signals are observed as-of each month-end: a filing is only used once its
``data_available_date`` has passed, and per-share figures are restated onto
the share basis of the price they are compared with.

Signals:
    ``bp``          log book-to-price (log of 1 / P/B)
    ``ep``          earnings yield, TTM EPS / price (can be negative)
    ``bp_z60``      ``bp`` relative to its own trailing 60-month history
    ``ep_z60``      ``ep`` relative to its own trailing 60-month history
    ``pb_roe_adj``  negated residual of log P/B on trailing ROE, fitted on an
                    expanding window (cheap relative to profitability)
    ``ep_6m``       earnings yield on the last 6 months of EPS, annualized
    ``ep_3m``       earnings yield on the last 3 months of EPS, annualized
    ``ep_norm``     earnings yield on normalized EPS: trailing 60-month
                    average ROE times current book value per share
    ``roe_gap``     trailing ROE minus its own trailing 60-month average (not
                    a valuation signal: where profitability sits in its cycle)

Because P/E = P/B / ROE, ``ep`` = ROE x book-to-price: the earnings yield
mixes a price-level component (``bp``) with a profitability component.
``ep_norm`` removes the cyclical part of ROE; ``roe_gap`` isolates it.

Validation follows the project rules: in-sample statistics use
Newey-West (HAC) errors for overlapping returns, and out-of-sample tests use
``TimeSeriesSplit`` with a gap equal to the forecast horizon (purge).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.processing.valuation_multiples import (
    latest_share_basis_factor,
    share_basis_factor,
)

SIGNALS: list[str] = [
    "bp",
    "ep",
    "bp_z60",
    "ep_z60",
    "pb_roe_adj",
    "ep_6m",
    "ep_3m",
    "ep_norm",
    "roe_gap",
]
SIGNAL_LABELS: dict[str, str] = {
    "bp": "Book-to-price (1 / P/B)",
    "ep": "Earnings yield (1 / P/E)",
    "bp_z60": "Book-to-price vs own 5y history",
    "ep_z60": "Earnings yield vs own 5y history",
    "pb_roe_adj": "P/B cheapness given ROE",
    "ep_6m": "Earnings yield, last 6 months annualized",
    "ep_3m": "Earnings yield, last 3 months annualized",
    "ep_norm": "Earnings yield on normalized earnings",
    "roe_gap": "ROE minus its 5-year average",
}
HORIZONS: list[int] = [1, 3, 6, 12, 24, 36]
ZSCORE_WINDOW = 60
ZSCORE_MIN_PERIODS = 36
ROE_ADJ_MIN_OBS = 60
OOS_MIN_TRAIN = 60
OOS_TEST_SIZE = 12


def build_asof_signals(
    valuation: pd.DataFrame,
    prices: pd.DataFrame,
    split_history: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build valuation signals as they were knowable at each month-end.

    Args:
        valuation: Output of ``build_monthly_valuation_multiples`` (one row
            per period month-end, with ``data_available_date``).
        prices: Unadjusted PGR prices indexed by date with ``close``.
        split_history: Splits indexed by split date with ``split_ratio``.
        month_ends: Observation dates (e.g. business month-ends).

    Returns:
        DataFrame indexed by ``month_ends`` with ``period`` (the filing
        period used), ``bvps``, ``ttm_eps`` and ``price`` on the latest share
        basis, ``pb``, ``pe``, ``roe`` and every signal in ``SIGNALS``.
    """
    val = valuation.copy()
    val["month_end"] = pd.to_datetime(val["month_end"])
    val["data_available_date"] = pd.to_datetime(val["data_available_date"])
    val = val.set_index("month_end").sort_index()

    latest = latest_share_basis_factor(split_history)
    period_factor = share_basis_factor(pd.DatetimeIndex(val.index), split_history)
    bvps_latest = val["book_value_per_share"] * period_factor / latest
    ttm_latest = val["eps_basic_ttm"] * period_factor / latest
    eps_month_latest = val["eps_basic"] * period_factor / latest
    eps_6m_latest = eps_month_latest.rolling(6, min_periods=6).sum() * 2
    eps_3m_latest = eps_month_latest.rolling(3, min_periods=3).sum() * 4
    # ROE: TTM earnings over average of opening and closing book value.
    avg_book = (bvps_latest + bvps_latest.shift(12)) / 2
    roe = ttm_latest / avg_book

    closes = prices["close"].dropna().sort_index()
    close_factor = share_basis_factor(pd.DatetimeIndex(closes.index), split_history)
    close_latest = closes * close_factor / latest

    records: list[dict[str, object]] = []
    for t in month_ends:
        available = val.index[val["data_available_date"] <= t]
        price = close_latest.asof(t)
        if len(available) == 0 or pd.isna(price):
            records.append({"date": t})
            continue
        period = available.max()
        records.append(
            {
                "date": t,
                "period": period,
                "bvps": bvps_latest[period],
                "ttm_eps": ttm_latest[period],
                "eps_6m": eps_6m_latest[period],
                "eps_3m": eps_3m_latest[period],
                "roe": roe[period],
                "price": price,
            }
        )
    cols = ["period", "bvps", "ttm_eps", "eps_6m", "eps_3m", "roe", "price"]
    frame = pd.DataFrame(records).set_index("date").reindex(columns=cols)

    out = pd.DataFrame(index=pd.DatetimeIndex(month_ends, name="date"))
    for col in cols:
        out[col] = frame[col]
    out["pb"] = frame["price"] / frame["bvps"].where(frame["bvps"] > 0)
    out["pe"] = frame["price"] / frame["ttm_eps"].where(frame["ttm_eps"] > 0)
    out["bp"] = -np.log(out["pb"])
    out["ep"] = frame["ttm_eps"] / frame["price"]
    for col in ("bp", "ep"):
        rolling = out[col].rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS)
        out[f"{col}_z60"] = (out[col] - rolling.mean()) / rolling.std()
    out["pb_roe_adj"] = expanding_residual(-out["bp"], out["roe"], ROE_ADJ_MIN_OBS) * -1
    out["ep_6m"] = frame["eps_6m"] / frame["price"]
    out["ep_3m"] = frame["eps_3m"] / frame["price"]
    out["roe_avg60"] = out["roe"].rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
    out["ep_norm"] = out["roe_avg60"] * frame["bvps"] / frame["price"]
    out["roe_gap"] = out["roe"] - out["roe_avg60"]
    return out


def expanding_residual(
    y: pd.Series,
    x: pd.Series,
    min_obs: int,
) -> pd.Series:
    """Residual of ``y`` on ``x`` at each date, fitted only on data up to it.

    Args:
        y: Dependent series.
        x: Single regressor, same index as ``y``.
        min_obs: Minimum complete observations before a residual is produced.

    Returns:
        Series of ``y_t - (a_t + b_t * x_t)`` where ``a_t, b_t`` come from an
        OLS fit on observations up to and including ``t``.
    """
    resid = pd.Series(np.nan, index=y.index, dtype=float)
    ys, xs = y.to_numpy(dtype=float), x.to_numpy(dtype=float)
    for i in range(len(y)):
        mask = ~(np.isnan(ys[: i + 1]) | np.isnan(xs[: i + 1]))
        if mask.sum() < min_obs or np.isnan(ys[i]) or np.isnan(xs[i]):
            continue
        slope, intercept = np.polyfit(xs[: i + 1][mask], ys[: i + 1][mask], 1)
        resid.iloc[i] = ys[i] - (intercept + slope * xs[i])
    return resid


def forward_log_returns(
    pgr_forward: dict[int, pd.Series],
    market_forward: dict[int, pd.Series],
) -> pd.DataFrame:
    """Combine simple forward total returns into log absolute/excess targets.

    Args:
        pgr_forward: Horizon (months) -> PGR forward total return series.
        market_forward: Horizon -> market forward total return series.

    Returns:
        DataFrame with ``abs_{h}m`` (log PGR return) and ``rel_{h}m`` (log PGR
        minus log market return) columns.
    """
    cols = {}
    for h, series in pgr_forward.items():
        cols[f"abs_{h}m"] = np.log1p(series)
        cols[f"rel_{h}m"] = np.log1p(series) - np.log1p(market_forward[h])
    return pd.DataFrame(cols)


def _hac_fit(y: np.ndarray, x: np.ndarray, lags: int):  # noqa: ANN202
    model = sm.OLS(y, sm.add_constant(x))
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def block_bootstrap_ic(
    x: np.ndarray,
    y: np.ndarray,
    block: int,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """90% moving-block bootstrap interval for the Spearman correlation.

    Args:
        x: Signal values (no NaN).
        y: Forward returns aligned with ``x`` (no NaN).
        block: Block length; set to the forecast horizon so overlapping
            return windows stay together.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        (5th percentile, 95th percentile) of the bootstrap distribution.
    """
    n = len(x)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    draws = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        draws[b] = stats.spearmanr(x[idx], y[idx])[0]
    return float(np.nanpercentile(draws, 5)), float(np.nanpercentile(draws, 95))


def one_way_stats(
    signals: pd.DataFrame,
    targets: pd.DataFrame,
    signal_names: list[str],
    horizons: list[int],
    start: str | None = None,
    end: str | None = None,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """In-sample one-variable predictive statistics for each signal/target.

    Returns one row per (signal, target) with the observation count, the
    approximate number of non-overlapping observations, Spearman rank IC and
    its block-bootstrap 90% interval, and the standardized OLS slope with its
    Newey-West t-statistic (lags = horizon).
    """
    records = []
    for signal in signal_names:
        for h in horizons:
            for kind in ("abs", "rel"):
                target = f"{kind}_{h}m"
                pair = pd.concat([signals[signal], targets[target]], axis=1).dropna()
                if start is not None:
                    pair = pair[pair.index >= pd.Timestamp(start)]
                if end is not None:
                    pair = pair[pair.index <= pd.Timestamp(end)]
                if len(pair) < 3 * h:
                    continue
                x = pair[signal].to_numpy()
                y = pair[target].to_numpy()
                ic = stats.spearmanr(x, y)[0]
                lo, hi = block_bootstrap_ic(x, y, block=h, n_boot=n_boot)
                zx = (x - x.mean()) / x.std()
                fit = _hac_fit(y, zx, lags=h)
                records.append(
                    {
                        "signal": signal,
                        "target": target,
                        "kind": kind,
                        "horizon": h,
                        "n_obs": len(pair),
                        "n_independent": len(pair) / h,
                        "ic": ic,
                        "ic_lo90": lo,
                        "ic_hi90": hi,
                        "slope_per_sd": fit.params[1],
                        "nw_t": fit.tvalues[1],
                        "r2": fit.rsquared,
                        "first_date": pair.index.min(),
                        "last_date": pair.index.max(),
                    }
                )
    return pd.DataFrame(records)


def oos_predictions(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    horizon: int,
) -> pd.DataFrame:
    """Walk-forward out-of-sample predictions with a purge gap.

    Uses ``TimeSeriesSplit`` (expanding training window, ``gap=horizon`` so
    no training target window overlaps the test period, 12-month test
    blocks, at least 60 training months).  Scaling is fitted inside each
    training fold only.

    Args:
        frame: Signals and targets indexed by date.
        features: Signal columns to use.
        target: Target column.
        horizon: Forecast horizon in months (purge length).

    Returns:
        DataFrame indexed by test date with ``actual``, ``model`` (Ridge
        prediction) and ``mean`` (training-mean benchmark) columns.
    """
    data = frame[features + [target]].dropna()
    n = len(data)
    n_splits = (n - OOS_MIN_TRAIN - horizon) // OOS_TEST_SIZE
    if n_splits < 2:
        return pd.DataFrame(columns=["actual", "model", "mean"])
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=OOS_TEST_SIZE, gap=horizon)
    X = data[features].to_numpy()
    y = data[target].to_numpy()
    parts = []
    for train_idx, test_idx in splitter.split(X):
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(X[train_idx], y[train_idx])
        parts.append(
            pd.DataFrame(
                {
                    "actual": y[test_idx],
                    "model": model.predict(X[test_idx]),
                    "mean": y[train_idx].mean(),
                },
                index=data.index[test_idx],
            )
        )
    return pd.concat(parts)


def oos_r2(preds: pd.DataFrame) -> float:
    """Out-of-sample R^2 relative to the training-mean forecast."""
    if preds.empty:
        return float("nan")
    sse_model = float(((preds["actual"] - preds["model"]) ** 2).sum())
    sse_mean = float(((preds["actual"] - preds["mean"]) ** 2).sum())
    return 1.0 - sse_model / sse_mean


def oos_table(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    horizons: list[int],
) -> pd.DataFrame:
    """Out-of-sample R^2 and prediction IC for each feature set and target."""
    records = []
    for name, features in feature_sets.items():
        for h in horizons:
            for kind in ("abs", "rel"):
                target = f"{kind}_{h}m"
                preds = oos_predictions(frame, features, target, h)
                if preds.empty:
                    continue
                records.append(
                    {
                        "model": name,
                        "target": target,
                        "kind": kind,
                        "horizon": h,
                        "n_test": len(preds),
                        "oos_r2": oos_r2(preds),
                        "oos_ic": stats.spearmanr(preds["model"], preds["actual"])[0],
                        "first_test": preds.index.min(),
                    }
                )
    return pd.DataFrame(records)


def encompassing_table(
    frame: pd.DataFrame,
    pairs: list[tuple[str, str]],
    horizons: list[int],
) -> pd.DataFrame:
    """Joint regression of the target on two standardized signals.

    Each row reports both signals' per-SD slopes and Newey-West t-stats when
    included together, over their common sample.
    """
    records = []
    for a, b in pairs:
        for h in horizons:
            for kind in ("abs", "rel"):
                target = f"{kind}_{h}m"
                data = frame[[a, b, target]].dropna()
                if len(data) < 3 * h:
                    continue
                X = data[[a, b]].to_numpy()
                X = (X - X.mean(axis=0)) / X.std(axis=0)
                fit = _hac_fit(data[target].to_numpy(), X, lags=h)
                records.append(
                    {
                        "signal_a": a,
                        "signal_b": b,
                        "target": target,
                        "kind": kind,
                        "horizon": h,
                        "n_obs": len(data),
                        "slope_a": fit.params[1],
                        "nw_t_a": fit.tvalues[1],
                        "slope_b": fit.params[2],
                        "nw_t_b": fit.tvalues[2],
                        "corr_ab": float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]),
                        "r2": fit.rsquared,
                    }
                )
    return pd.DataFrame(records)


def noise_diagnostics(signals: pd.DataFrame) -> pd.DataFrame:
    """Describe how each multiple and its denominator move month to month.

    Returns one row per multiple with its level distribution, the SD of
    monthly log changes in the multiple and in its denominator (book value
    or TTM EPS, only while positive), the share of the multiple's monthly
    variance that comes from price moves rather than the denominator, and
    the 12-month autocorrelation of the log level (persistence).
    """
    log_price = np.log(signals["price"])
    records = []
    for name, level, denom in (
        ("P/B", signals["pb"], signals["bvps"]),
        ("P/E", signals["pe"], signals["ttm_eps"]),
    ):
        log_level = np.log(level)
        log_denom = np.log(denom.where(denom > 0))
        both = pd.concat(
            [log_level.diff(), log_price.diff(), log_denom.diff()], axis=1
        ).dropna()
        both.columns = ["multiple", "price", "denom"]
        records.append(
            {
                "multiple": name,
                "months": int(level.notna().sum()),
                "median": float(level.median()),
                "p10": float(level.quantile(0.10)),
                "p90": float(level.quantile(0.90)),
                "monthly_log_change_sd": float(both["multiple"].std()),
                "denominator_log_change_sd": float(both["denom"].std()),
                "price_share_of_variance": float(
                    both["price"].var() / both["multiple"].var()
                ),
                "autocorr_12m": float(log_level.autocorr(12)),
            }
        )
    return pd.DataFrame(records)


def window_ic(
    frame: pd.DataFrame,
    signal_names: list[str],
    targets: list[str],
    windows: list[tuple[str, str]],
) -> pd.DataFrame:
    """Spearman IC of each signal/target inside each (start, end) window.

    Windows select on the observation (signal) date, inclusive of both ends.
    """
    records = []
    for start, end in windows:
        sub = frame.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        for signal in signal_names:
            for target in targets:
                pair = sub[[signal, target]].dropna()
                if len(pair) < 24:
                    continue
                records.append(
                    {
                        "window_start": start,
                        "window_end": end,
                        "signal": signal,
                        "target": target,
                        "n_obs": len(pair),
                        "ic": stats.spearmanr(pair[signal], pair[target])[0],
                    }
                )
    return pd.DataFrame(records)


def annual_sample_ic(
    frame: pd.DataFrame,
    signal_names: list[str],
    target: str,
) -> pd.DataFrame:
    """Spearman IC using one observation per year, for each calendar month.

    With a 12-month target this removes overlap between return windows
    entirely; repeating it for all 12 starting months shows how much the
    answer depends on which month is sampled.
    """
    records = []
    for signal in signal_names:
        for month in range(1, 13):
            sub = frame[frame.index.month == month]
            pair = sub[[signal, target]].dropna()
            if len(pair) < 8:
                continue
            records.append(
                {
                    "signal": signal,
                    "target": target,
                    "month": month,
                    "n_obs": len(pair),
                    "ic": stats.spearmanr(pair[signal], pair[target])[0],
                }
            )
    return pd.DataFrame(records)


def return_decomposition(
    total_return_index: pd.Series,
    price: pd.Series,
    bvps: pd.Series,
    periods: list[tuple[str, str]],
) -> pd.DataFrame:
    """Split annualized total return into book growth, re-rating and dividends.

    In logs, total return = change in book value per share + change in P/B +
    the dividend contribution (total return less price return).  All three
    inputs must share a per-share basis (e.g. restated across splits).

    Args:
        total_return_index: Dividend-reinvested value index by month-end.
        price: Share price by month-end (same basis as ``bvps``).
        bvps: Book value per share by month-end.
        periods: (start, end) month-end pairs.

    Returns:
        One row per period with annualized log contributions and their sum.
    """
    records = []
    for start, end in periods:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        years = (e - s).days / 365.25
        tr = float(np.log(total_return_index.asof(e) / total_return_index.asof(s)))
        pr = float(np.log(price.asof(e) / price.asof(s)))
        book = float(np.log(bvps.asof(e) / bvps.asof(s)))
        records.append(
            {
                "start": start,
                "end": end,
                "years": years,
                "total": tr / years,
                "book_growth": book / years,
                "rerating": (pr - book) / years,
                "dividends": (tr - pr) / years,
                "pb_start": float(price.asof(s) / bvps.asof(s)),
                "pb_end": float(price.asof(e) / bvps.asof(e)),
            }
        )
    return pd.DataFrame(records)


def implied_roe(pb: float, cost_of_equity: float, growth: float) -> float:
    """Long-run ROE implied by a P/B under the Gordon growth model.

    Justified P/B = (ROE - g) / (r - g), so ROE = g + P/B x (r - g).
    """
    return growth + pb * (cost_of_equity - growth)
