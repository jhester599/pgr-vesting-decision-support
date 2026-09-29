"""Frozen v201 causal features and paired development inference.

Prices are raw weekly bars. Share factors accumulate only past splits;
feature ratios are invariant to the arbitrary initial share basis. Macro
inputs must be raw calendar observations, never an already lagged cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pgr_vds.research_lib.metrics import honest_r2


def monthly_index(index: pd.DatetimeIndex) -> None:
    """Require a complete ordered business-month-end grid."""
    if not len(index) or not index.equals(
        pd.date_range(index.min(), index.max(), freq="BME")
    ):
        raise ValueError("Require a complete ordered monthly origin grid")


def price_features(
    close: pd.Series,
    splits: pd.Series,
    origins: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Calendar momentum, 13 weekly returns and 52 weekly close highs.

    A split multiplies subsequent raw closes by the accumulated share count.
    Dividends are intentionally absent from technical price ratios; forward
    targets separately retain v200's manual fractional-share DRIP convention.
    Missing weekly/monthly observations are not forward-filled.
    """
    monthly_index(origins)
    raw = close.loc[close.index <= origins.max()].sort_index().astype(float)
    if raw.index.has_duplicates or (raw <= 0).any():
        raise ValueError("Require positive unique weekly price bars")
    if len(raw) > 1 and raw.index.to_series().diff().dt.days.median() != 7:
        raise ValueError("Require weekly bars; do not treat rows as days")
    if splits.index.has_duplicates or (splits <= 0).any():
        raise ValueError("Require positive unique split actions")
    factor = pd.Series(1.0, index=raw.index)
    for date, ratio in splits.sort_index().items():
        if date <= origins.max():
            factor.loc[factor.index >= date] *= float(ratio)
    adjusted = raw * factor
    monthly = adjusted.resample("BME").last(skipna=False)
    output = pd.DataFrame(index=origins)
    for horizon in (3, 6, 12):
        output[f"pm_mom{horizon}"] = (
            monthly / monthly.shift(horizon) - 1
        ).reindex(origins)
    returns = adjusted.pct_change(fill_method=None)
    returns = returns.where(raw.index.to_series().diff().dt.days <= 10)
    weekly = pd.DataFrame(
        {
            "pm_vol13w": returns.rolling(13, min_periods=13).std()
            * np.sqrt(52),
            "pm_high52w": adjusted
            / adjusted.rolling("364D", min_periods=52).max()
            - 1,
        }
    )
    output = output.join(
        weekly.resample("BME").last(skipna=False).reindex(origins)
    )
    return output


def trend_features(ratio: pd.Series) -> pd.DataFrame:
    """VOO price-ratio EMA distance and simple six-calendar-month RSI.

    EMA span12, adjust=False and a 12-month warmup are frozen. RSI uses six
    monthly changes, no Wilder/daily substitution. Flat ratios score50.
    """
    monthly_index(ratio.index)
    ema = ratio.ewm(span=12, adjust=False, min_periods=12).mean()
    change = ratio.diff()
    gain = change.clip(lower=0).rolling(6, min_periods=6).mean()
    loss = (-change.clip(upper=0)).rolling(6, min_periods=6).mean()
    rsi = 100 * gain / (gain + loss)
    rsi = rsi.where((gain + loss) != 0, 50)
    return pd.DataFrame(
        {
            "pm_ema12_voo": ratio / ema - 1,
            "pm_rsi6_voo": rsi,
        }
    )


def available_macro(
    raw: pd.DataFrame,
    origins: pd.DatetimeIndex,
    lags: dict[str, int],
    raw_unlagged: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select exactly origin-period minus one frozen publication rule.

    Historical timestamps/vintages are unavailable. Calendar availability is
    a conservative proxy, not an observed release certification. Each series
    gets one rule; missing periods stay NaN, including interior/stale gaps.
    """
    monthly_index(origins)
    if not raw_unlagged:
        raise ValueError("Only raw unlagged observations may be consumed")
    work = raw.copy()
    work["period"] = pd.to_datetime(work["month_end"]).dt.to_period("M")
    if work.duplicated(["series_id", "period"]).any():
        raise ValueError("Duplicate macro calendar periods")
    values = pd.DataFrame(index=origins, columns=list(lags), dtype=float)
    records: list[dict] = []
    for series, lag in lags.items():
        if type(lag) is not int or lag < 1:
            raise ValueError("Require one fixed positive calendar lag")
        observations = work.loc[work.series_id == series].set_index("period")
        for origin in origins:
            period = origin.to_period("M") - lag
            value = (
                float(observations.loc[period, "value"])
                if period in observations.index
                else np.nan
            )
            values.loc[origin, series] = value
            records.append(
                {
                    "origin": origin,
                    "series_id": series,
                    "observation_period": str(period),
                    "available": (period + lag).to_timestamp()
                    + pd.offsets.BMonthEnd(0),
                    "publication_rule_months": lag,
                    "lag_applications": 1,
                    "value": value,
                    "status": "usable" if np.isfinite(value) else "missing",
                    "vintage": "latest;not point-in-time",
                    "release_timestamp": None,
                }
            )
    return values, pd.DataFrame(records)


def macro_features(values: pd.DataFrame) -> pd.DataFrame:
    """Frozen slopes, real-yield changes, stress levels and price-cost gap."""
    monthly_index(values.index)
    real = values["GS10"] - values["T10YIE"]
    costs = (
        values["CUSR0000SETA02"].pct_change(12, fill_method=None)
        + values["CUSR0000SAM2"].pct_change(12, fill_method=None)
    ) / 2
    return pd.DataFrame(
        {
            "pm_slope": values["T10Y2Y"],
            "pm_real_change6": real - real.shift(6),
            "pm_vix": values["VIXCLS"],
            "pm_nfci": values["NFCI"],
            "pm_credit_hy": values["BAMLH0A0HYM2"],
            "pm_rate_gap": values["PCU5241265241261"].pct_change(
                12, fill_method=None
            )
            - costs,
        }
    )


def paired_comparison(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
    horizon: int,
    replicates: int = 2000,
    seed: int = 20260926,
) -> dict:
    """One-sided paired loss test, equal weight per calendar origin.

    R² differences use the identical realised mature-mean denominator.
    Resample contiguous monthly dates; all benchmark rows travel together.
    Missing calendar origins are retained as empty dates in sampling blocks.
    """
    keys = ["date", "benchmark"]
    left = control.sort_values(keys).reset_index(drop=True)
    right = candidate.sort_values(keys).reset_index(drop=True)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Duplicate support keys")
    if not left[keys].equals(right[keys]) or not np.allclose(
        left[["y_true", "naive"]],
        right[["y_true", "naive"]],
        atol=0,
        rtol=0,
    ):
        raise ValueError(
            "Candidate must use identical control support/targets"
        )
    if horizon not in (6, 12) or replicates < 1 or left.empty:
        raise ValueError("Require a supported horizon and nonempty panel")
    y = left.y_true.to_numpy()
    p, q = left.y_hat.to_numpy(), right.y_hat.to_numpy()
    naive = left.naive.to_numpy()
    if not np.isfinite(np.column_stack([y, p, q, naive])).all():
        raise ValueError("Nonfinite paired support")
    gain = (y - p) ** 2 - (y - q) ** 2
    dates = pd.date_range(left.date.min(), left.date.max(), freq="BME")
    rows = [
        np.flatnonzero(left.date.to_numpy() == d.to_datetime64())
        for d in dates
    ]
    date_gain = np.array([gain[r].mean() if len(r) else np.nan for r in rows])
    observed = float(np.nanmean(date_gain))
    delta = honest_r2(y, q, naive) - honest_r2(y, p, naive)
    draws, r2_draws = [], []
    if np.isfinite(date_gain).sum() >= 2 * horizon:
        rng = np.random.default_rng(seed)
        for _ in range(replicates):
            starts = rng.integers(
                0,
                len(dates) - horizon + 1,
                size=int(np.ceil(len(dates) / horizon)),
            )
            indices = np.concatenate(
                [np.arange(s, s + horizon) for s in starts]
            )[: len(dates)]
            sampled = np.concatenate([rows[i] for i in indices])
            if not len(sampled):
                continue
            draws.append(float(np.nanmean(date_gain[indices])))
            r2_draws.append(
                honest_r2(y[sampled], q[sampled], naive[sampled])
                - honest_r2(y[sampled], p[sampled], naive[sampled])
            )
    pvalue = (
        float(
            (1 + np.sum(np.asarray(draws) - observed >= observed))
            / (1 + len(draws))
        )
        if draws and observed > 0
        else 1.0
    )
    return {
        "delta_r2": delta,
        "mean_date_loss_gain": observed,
        "primary_p": pvalue,
        "delta_r2_ci": np.quantile(r2_draws, [0.025, 0.975]).tolist()
        if r2_draws
        else [None, None],
        "n_rows": len(left),
        "n_dates": int(np.isfinite(date_gain).sum()),
        "block_length": horizon,
        "bootstrap_replicates": replicates,
        "seed": seed,
        "inference_supported": bool(draws),
    }
