"""Research-only target builders for the x-series PGR forecasting lane."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_datetime_series(values: pd.Series, name: str) -> pd.Series:
    """Return a sorted numeric series with a DatetimeIndex."""
    result = pd.to_numeric(values.copy(), errors="coerce")
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    result = result.sort_index()
    result.name = name
    return result


def _positive_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return numerator / denominator where both sides are positive."""
    valid = (numerator > 0.0) & (denominator > 0.0)
    return (numerator / denominator).where(valid)


def build_forward_return_targets(
    price_or_value: pd.Series,
    horizons: tuple[int, ...] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Build multi-horizon absolute return, log-return, and direction targets.

    Args:
        price_or_value: Monthly price or total-return value series indexed by
            month-end date.
        horizons: Forward horizons in monthly observations.

    Returns:
        DataFrame indexed like ``price_or_value`` with current/future values and
        target columns for each requested horizon. Tail rows without a complete
        future horizon are left as NaN.
    """
    current = _as_datetime_series(price_or_value, "current_price")
    result = pd.DataFrame({"current_price": current}, index=current.index)

    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        future = current.shift(-horizon)
        simple_return = _positive_ratio(future, current) - 1.0
        result[f"future_{horizon}m_price"] = future
        result[f"target_{horizon}m_return"] = simple_return
        result[f"target_{horizon}m_log_return"] = np.log1p(simple_return)
        result[f"target_{horizon}m_up"] = (simple_return > 0.0).where(
            simple_return.notna()
        )

    return result


def build_decomposition_targets(
    price: pd.Series,
    bvps: pd.Series,
    horizons: tuple[int, ...] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Build BVPS and P/B decomposition targets for x-series research.

    Args:
        price: Monthly PGR price series indexed by month-end date.
        bvps: Monthly book value per share series indexed by month-end date.
        horizons: Forward horizons in monthly observations.

    Returns:
        DataFrame with future BVPS, BVPS growth, log BVPS growth, future P/B,
        and log future P/B targets for each horizon.
    """
    current_price = _as_datetime_series(price, "current_price")
    current_bvps = _as_datetime_series(bvps, "current_bvps")
    aligned = pd.concat([current_price, current_bvps], axis=1, join="inner")
    result = aligned.copy()
    current_pb = _positive_ratio(aligned["current_price"], aligned["current_bvps"])
    result["current_pb"] = current_pb

    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        future_price = aligned["current_price"].shift(-horizon)
        future_bvps = aligned["current_bvps"].shift(-horizon)
        bvps_ratio = _positive_ratio(future_bvps, aligned["current_bvps"])
        future_pb = _positive_ratio(future_price, future_bvps)

        result[f"target_{horizon}m_bvps"] = future_bvps
        result[f"target_{horizon}m_bvps_growth"] = bvps_ratio - 1.0
        result[f"target_{horizon}m_log_bvps_growth"] = np.log(bvps_ratio)
        result[f"target_{horizon}m_pb"] = future_pb
        result[f"target_{horizon}m_log_pb"] = np.log(future_pb)

    return result


def _dividend_amount_column(dividends: pd.DataFrame) -> str:
    """Return the dividend amount column used by repo loaders/tests."""
    for column in ("amount", "dividend"):
        if column in dividends.columns:
            return column
    raise ValueError("dividends must include an 'amount' or 'dividend' column")


def _infer_regular_baseline(
    dividends: pd.Series,
    target_start: pd.Timestamp,
    lookback_months: int = 24,
) -> float:
    """Infer a normal quarterly dividend from pre-target dividend history."""
    lookback_start = target_start - pd.DateOffset(months=lookback_months)
    prior = dividends.loc[
        (dividends.index < target_start) & (dividends.index >= lookback_start)
    ].dropna()
    prior = prior[prior > 0.0].tail(8)
    if prior.empty:
        return float("nan")

    first_pass = float(prior.median())
    if not np.isfinite(first_pass) or first_pass <= 0.0:
        return float("nan")
    regular = prior[prior <= first_pass * 2.0]
    if regular.empty:
        regular = prior
    return float(regular.median())


def _series_value(row: pd.Series, columns: tuple[str, ...]) -> float:
    """Return the first finite row value among candidate columns."""
    for column in columns:
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return float("nan")


def build_special_dividend_targets(
    monthly_features: pd.DataFrame,
    dividends: pd.DataFrame,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Build annual November-snapshot Q1 special-dividend targets.

    The target for snapshot year ``Y`` is the total dividend paid in Q1 of
    ``Y + 1`` above a regular quarterly baseline inferred only from dividend
    records before that target Q1 window.

    Args:
        monthly_features: Monthly feature or EDGAR/price frame. Only November
            rows are used as prediction snapshots.
        dividends: Dividend history indexed by ex-dividend date.
        tolerance: Minimum excess required to mark an occurrence.

    Returns:
        Annual DataFrame indexed by November snapshot date.
    """
    if monthly_features.empty:
        return pd.DataFrame()
    amount_col = _dividend_amount_column(dividends)
    div_amounts = pd.to_numeric(dividends[amount_col], errors="coerce")
    div_amounts.index = pd.DatetimeIndex(pd.to_datetime(div_amounts.index))
    div_amounts = div_amounts.sort_index()

    snapshots = monthly_features.copy()
    snapshots.index = pd.DatetimeIndex(pd.to_datetime(snapshots.index))
    snapshots = snapshots.sort_index()
    snapshots = snapshots[snapshots.index.month == 11]

    rows: list[dict[str, float | int | str]] = []
    for snapshot_date, snapshot in snapshots.iterrows():
        target_year = int(snapshot_date.year + 1)
        target_start = pd.Timestamp(year=target_year, month=1, day=1)
        target_end = pd.Timestamp(year=target_year, month=3, day=31)
        baseline = _infer_regular_baseline(div_amounts, target_start)

        q1_values = div_amounts.loc[
            (div_amounts.index >= target_start)
            & (div_amounts.index <= target_end)
        ]
        has_complete_target = (
            not div_amounts.empty and div_amounts.index.max() >= target_end
        )
        q1_total = float(q1_values.sum()) if has_complete_target else float("nan")
        if np.isfinite(q1_total) and np.isfinite(baseline):
            excess = max(q1_total - baseline, 0.0)
            occurred = int(excess > tolerance)
        else:
            excess = float("nan")
            occurred = float("nan")

        snapshot_bvps = _series_value(
            snapshot,
            ("book_value_per_share", "current_bvps", "bvps"),
        )
        snapshot_price = _series_value(
            snapshot,
            ("close_price", "current_price", "price", "close"),
        )
        net_income = _series_value(
            snapshot,
            ("net_income_ttm_per_share", "ttm_net_income_per_share"),
        )

        rows.append(
            {
                "snapshot_date": snapshot_date,
                "snapshot_year": int(snapshot_date.year),
                "snapshot_month": int(snapshot_date.month),
                "target_year": target_year,
                "regular_baseline_dividend": baseline,
                "q1_dividend_total": q1_total,
                "special_dividend_occurred": occurred,
                "special_dividend_excess": excess,
                "snapshot_bvps": snapshot_bvps,
                "snapshot_price": snapshot_price,
                "snapshot_net_income_ttm_per_share": net_income,
                "special_dividend_excess_to_bvps": (
                    excess / snapshot_bvps
                    if np.isfinite(excess) and snapshot_bvps > 0.0
                    else float("nan")
                ),
                "special_dividend_excess_to_price": (
                    excess / snapshot_price
                    if np.isfinite(excess) and snapshot_price > 0.0
                    else float("nan")
                ),
                "special_dividend_excess_to_net_income": (
                    excess / net_income
                    if np.isfinite(excess) and net_income > 0.0
                    else float("nan")
                ),
            }
        )

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).set_index("snapshot_date")
    result.index.name = "snapshot_date"
    return result
