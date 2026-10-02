"""Causal, calendar-aligned insurance and valuation research calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from pgr_vds.research_lib.price_macro import paired_comparison


def release_month(report_month: object, filing_date: object | None) -> pd.Timestamp:
    """Return first business month end after a filing, or a two-month fallback."""
    report = pd.Timestamp(report_month)
    if pd.isna(filing_date):
        return (report.to_period("M") + 2).to_timestamp() + pd.offsets.BMonthEnd(0)
    filing = pd.Timestamp(filing_date)
    if filing < report:
        raise ValueError("Filing precedes its report month")
    return filing + pd.offsets.BMonthEnd(0)


def split_rebased_growth(
    previous_bvps: float, current_bvps: float, split_factor: float
) -> float:
    """Compare per-share book values on today's share basis."""
    if previous_bvps <= 0 or current_bvps <= 0 or split_factor <= 0:
        return float("nan")
    return current_bvps / (previous_bvps / split_factor) - 1.0


def trailing_twelve_growth(values: pd.Series) -> pd.Series:
    """Annual growth in twelve complete calendar-month sums."""
    series = values.astype(float).sort_index()
    if series.index.has_duplicates or not series.index.equals(
        pd.date_range(series.index.min(), series.index.max(), freq="ME")
    ):
        raise ValueError("Require a complete unique monthly calendar")
    total = series.rolling(12, min_periods=12).sum()
    previous = total.shift(12)
    return (total / previous - 1.0).where(previous > 0)


def fiscal_month_shift(frame: pd.DataFrame) -> dict[str, Any]:
    """Test February versus March NPW/NPE seasonality across fixed eras."""
    needed = {"year", "month", "npw", "npe"}
    if not needed.issubset(frame.columns):
        raise ValueError("Fiscal audit requires year, month, NPW and NPE")
    work = frame.loc[frame.month.isin((2, 3))].copy()
    work["ratio"] = work.npw / work.npe
    work = work.loc[np.isfinite(work.ratio) & (work.npe > 0)]
    medians = {
        era: work.loc[work.year.between(*years)].groupby("month").ratio.median()
        for era, years in (("pre", (2016, 2023)), ("post", (2024, 2026)))
    }
    if any(set(part.index) != {2, 3} for part in medians.values()):
        raise ValueError("Both eras require February and March observations")
    pre = medians["pre"]
    post = medians["post"]
    return {
        "confirmed": bool(pre.loc[2] > pre.loc[3] and post.loc[3] > post.loc[2]),
        "pre_peak_month": int(pre.idxmax()),
        "post_peak_month": int(post.idxmax()),
        "pre_feb_minus_march": float(pre.loc[2] - pre.loc[3]),
        "post_march_minus_feb": float(post.loc[3] - post.loc[2]),
        "pre_years": [2016, 2023],
        "post_years": [2024, 2026],
    }


def insurance_features(
    monthly: pd.DataFrame,
    splits: pd.Series,
    origins: pd.DatetimeIndex,
    raw_price: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build eight filing-gated monthly blocks without value forward filling.

    A stale report can be used for at most three calendar months. Rolling
    calculations require every constituent month; missing source months stay
    missing. Original-value vintages are unavailable in the database.
    """
    if origins.has_duplicates or not origins.is_monotonic_increasing:
        raise ValueError("Require ordered unique monthly origins")
    raw = monthly.copy()
    raw["month_end"] = pd.to_datetime(raw["month_end"])
    if raw.month_end.duplicated().any():
        raise ValueError("Duplicate EDGAR report month")
    raw = raw.set_index("month_end").sort_index()
    calendar = pd.date_range(raw.index.min(), raw.index.max(), freq="ME")
    raw = raw.reindex(calendar)
    filing = pd.to_datetime(raw["filing_date"])
    release = pd.Series(
        [release_month(month, filed) for month, filed in filing.items()],
        index=calendar,
    )
    splits = splits.copy()
    splits.index = pd.DatetimeIndex(splits.index)
    if splits.index.has_duplicates or (splits <= 0).any():
        raise ValueError("Invalid split actions")
    numbers = raw
    columns: dict[str, pd.Series] = {}
    lookbacks: dict[str, int] = {}

    def add(name: str, values: pd.Series, lookback: int) -> None:
        columns[name] = values.replace([np.inf, -np.inf], np.nan)
        lookbacks[name] = lookback

    cr = pd.to_numeric(numbers.combined_ratio, errors="coerce")
    cr_ttm = cr.rolling(12, min_periods=12).mean()
    add("iv_cr_ttm", cr_ttm, 12)
    add("iv_cr_change", cr_ttm - cr_ttm.shift(12), 24)
    stable = numbers[
        ["pif_agency_auto", "pif_direct_auto", "pif_commercial_lines"]
    ].sum(axis=1, min_count=3)
    pif_growth = (stable / stable.shift(12) - 1.0).where(stable.shift(12) > 0)
    add("iv_pif_growth", pif_growth, 13)
    add("iv_gainshare", (100.0 - cr_ttm).clip(lower=0) * pif_growth, 24)
    income = pd.to_numeric(numbers.investment_income, errors="coerce")
    add("iv_income_growth", trailing_twelve_growth(income), 24)
    add("iv_book_yield", numbers.investment_book_yield.astype(float), 1)

    bvps = numbers.book_value_per_share.astype(float)
    split_values = []
    for current, previous in zip(calendar, calendar.shift(-12)):
        factor = splits.loc[(splits.index > previous) & (splits.index <= current)]
        split_values.append(float(factor.prod()) if len(factor) else 1.0)
    growth = pd.Series(
        [
            split_rebased_growth(float(old), float(new), factor)
            for old, new, factor in zip(bvps.shift(12), bvps, split_values)
        ],
        index=calendar,
    )
    add("iv_bvps_growth", growth, 13)
    ni = numbers.net_income.astype(float)
    equity = numbers.shareholders_equity.astype(float)
    average_equity = (equity + equity.shift(12)) / 2.0
    roe = ni.rolling(12, min_periods=12).sum() / average_equity
    add("iv_roe", roe.where(average_equity > 0), 13)
    shares = numbers.common_shares_outstanding.astype(float)
    add("iv_bvps", bvps, 1)
    add("iv_eps_ttm", ni.rolling(12, min_periods=12).sum() / shares, 12)

    npw = numbers.net_premiums_written.astype(float)
    add("iv_npw_t12_yoy", trailing_twelve_growth(npw), 24)
    source = pd.DataFrame(columns, index=calendar)
    availability = {}
    release_ns = release.map(lambda timestamp: float(timestamp.value))
    for name, window in lookbacks.items():
        latest = release_ns.rolling(window, min_periods=window).max()
        availability[name] = pd.to_datetime(latest)
    available = pd.DataFrame(availability, index=calendar)
    out = pd.DataFrame(index=origins)
    ledger = []
    for name in source.columns:
        values = []
        for origin in origins:
            eligible = (
                (calendar <= origin)
                & (available[name].to_numpy() <= origin)
            )
            positions = np.flatnonzero(eligible)
            position = int(positions[-1]) if len(positions) else -1
            report = calendar[position] if position >= 0 else pd.NaT
            age = (
                origin.to_period("M") - report.to_period("M")
            ).n if position >= 0 else 999
            valid = position >= 0 and age <= 3
            values.append(source[name].iloc[position] if valid else np.nan)
            ledger.append({
                "origin": origin,
                "feature": name,
                "report_month": report if valid else pd.NaT,
                "available": available[name].iloc[position] if valid else pd.NaT,
                "fallback": bool(valid and pd.isna(filing.iloc[position])),
                "age_months": age if valid else None,
            })
        out[name] = values
    for name in ("iv_bvps", "iv_eps_ttm"):
        source_dates = [
            row["report_month"]
            for row in ledger
            if row["feature"] == name
        ]
        factors = [
            float(splits.loc[(splits.index > report) & (splits.index <= origin)].prod())
            if pd.notna(report) else np.nan
            for report, origin in zip(source_dates, origins)
        ]
        out[name] = out[name] / factors
    out["iv_pb"] = raw_price.reindex(origins) / out.iv_bvps
    out["iv_pe"] = (raw_price.reindex(origins) / out.iv_eps_ttm).where(
        out.iv_eps_ttm > 0
    )
    out["iv_pb_pe_spread"] = np.log(out.iv_pb) - np.log(out.iv_pe)
    for name, parent in (
        ("iv_pb", "iv_bvps"),
        ("iv_pe", "iv_eps_ttm"),
        ("iv_pb_pe_spread", "iv_eps_ttm"),
    ):
        ledger.extend(
            {**row, "feature": name}
            for row in ledger
            if row["feature"] == parent
        )
    return out, pd.DataFrame(ledger)


def select_causal_recipe(
    panel: pd.DataFrame,
    training_dates: pd.DatetimeIndex,
    outer_origin: pd.Timestamp,
    horizon: int,
) -> str | None:
    """Select an upstream preregistered recipe on inner mature OOS dates.

    The panel contains upstream forecasts that were individually fitted only
    on history prior to their dates. Candidate rows stay grouped by month.
    """
    window = {6: 60, 12: 120}[horizon]
    minimum = {6: 24, 12: 60}[horizon]
    dates = pd.DatetimeIndex(training_dates).sort_values().unique()
    if len(dates) < window or dates.max() >= outer_origin:
        return None
    splitter = TimeSeriesSplit(
        n_splits=3,
        test_size=6,
        gap=2 * horizon,
        max_train_size=window,
    )
    work = panel.copy()
    work["date"] = pd.to_datetime(work.date)
    work["available"] = pd.to_datetime(work.available)
    work = work.loc[work.available <= outer_origin]
    validation = []
    for train_index, test_index in splitter.split(dates):
        train_dates = dates[train_index]
        usable = work.loc[
            work.date.isin(train_dates)
            & (work.available <= dates[test_index[0]])
        ]
        if usable.date.nunique() < minimum:
            return None
        validation.extend(dates[test_index])
    scored = work.loc[work.date.isin(validation)].copy()
    losses = {}
    required_dates = len(set(validation))
    for recipe, group in scored.groupby("recipe", sort=True):
        if (
            group.date.nunique() != required_dates
            or group[["y_true", "y_hat"]].isna().any().any()
        ):
            continue
        losses[str(recipe)] = float(
            np.mean((group.y_true.to_numpy() - group.y_hat.to_numpy()) ** 2)
        )
    return min(losses, key=lambda recipe: losses[recipe]) if losses else None


def paired_delta_bootstrap(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
    horizon: int,
    replicates: int = 2000,
    seed: int = 20260926,
) -> dict[str, Any]:
    """Add D9's descriptive first-Holm spread to the frozen paired test."""
    result = paired_comparison(control, candidate, horizon, replicates, seed)
    keys = ["date", "benchmark"]
    left = control.sort_values(keys).reset_index(drop=True)
    right = candidate.sort_values(keys).reset_index(drop=True)
    dates = pd.date_range(left.date.min(), left.date.max(), freq="BME")
    row_dates = pd.to_datetime(left.date).to_numpy()
    groups = [
        np.flatnonzero(row_dates == date.to_datetime64()) for date in dates
    ]
    y = left.y_true.to_numpy(dtype=float)
    old = left.y_hat.to_numpy(dtype=float)
    new = right.y_hat.to_numpy(dtype=float)
    naive = left.naive.to_numpy(dtype=float)
    from pgr_vds.research_lib.metrics import honest_r2

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        starts = rng.integers(
            0, len(dates) - horizon + 1,
            size=int(np.ceil(len(dates) / horizon)),
        )
        sampled_dates = np.concatenate(
            [np.arange(start, start + horizon) for start in starts]
        )[:len(dates)]
        rows = np.concatenate([groups[index] for index in sampled_dates])
        draws.append(
            honest_r2(y[rows], new[rows], naive[rows])
            - honest_r2(y[rows], old[rows], naive[rows])
        )
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    spread = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    result["bootstrap_delta_r2_sd"] = spread
    result["approx_first_holm_delta_r2"] = 3.01 * spread
    return result
