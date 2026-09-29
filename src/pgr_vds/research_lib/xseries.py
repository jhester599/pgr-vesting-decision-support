"""Causal targets, features and scoring for the v205 dividend/BVPS lanes.

Every per-share quantity is restated to one origin-date share with the
explicit split history, so the 2006 4-for-1 split is never read as a capital
event (review F15). Monthly report features use only reports actually
filed by the origin, and year-over-year or trailing windows require every
calendar month in the window, so a missing month gives NaN rather than a
13-month change (F16). The annual dividend window is December through
February, so December specials are not lost to a January-March window
(F25). Nothing here reads a database, fetches data or selects a model from
outer test outcomes; study runners control input access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pgr_vds.research_lib.adapters import (
    excess_ratio,
    regular_dividend_baseline,
    ridge_pipeline,
)
from pgr_vds.research_lib.metrics import holm38
from pgr_vds.research_lib.temporal import chronological_splits, label_end

__all__ = [
    "ALPHA_GRID",
    "CAMPAIGN_SLOTS",
    "POLICY_CHANGE",
    "THRESHOLDS",
    "adjusted_bvps_growth_target",
    "annual_excess_target",
    "annual_window",
    "block_spearman",
    "bvps_growth_target",
    "campaign_holm",
    "ece_equal_width",
    "filed_reports",
    "label_end",
    "lane_disposition",
    "lane_features",
    "majority_base",
    "nested_ridge_forecast",
    "next_cash_12m",
    "nominate_winner",
    "origin_basis_cash",
    "paired_block_test",
    "past_cash",
    "prequential_stream",
    "probability_scores",
    "restate_to_basis",
    "share_factor",
]

POLICY_CHANGE = pd.Timestamp("2018-12-01")
ALPHA_GRID = (1.0, 10.0, 100.0)
# Zero-based positions of campaign slots 29-34 (v201 6, v202 8, v203 8,
# v204 6, then v205 6 and v206 4) in the frozen 38-test Holm family.
CAMPAIGN_SLOTS = tuple(range(28, 34))
THRESHOLDS = {
    "relative_mae_reduction": 0.10,
    "adjusted_p": 0.05,
    "delta_r2": 0.01,
    "coverage_tolerance": 0.05,
    "independent_blocks": 5.0,
}
FEATURE_COLUMNS = [
    "report_month",
    "report_filing_date",
    "past12_cash",
    "prior_past12_cash",
    "cr_ttm",
    "pif_growth_yoy_cal",
    "npw_growth_ttm",
    "current_bvps",
    "bvps_growth_1m",
    "bvps_growth_3m",
    "bvps_growth_6m",
    "bvps_growth_yoy",
    "bvps_growth_ytd",
    "bvps_yoy_dollar_change",
    "roe_ttm",
    "gainshare_estimate",
    "investment_book_yield",
    "month_of_year",
    "q4_flag",
    "dividend_season_flag",
]


def share_factor(
    splits: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
) -> float:
    """Shares held at ``end`` per share held at ``start``: (start, end]."""
    values = splits.loc[(splits.index > start) & (splits.index <= end)]
    values = values.astype(float)
    if (~np.isfinite(values)).any() or (values <= 0).any():
        raise ValueError("Invalid split ratio in the explicit split history")
    return float(values.prod())


def origin_basis_cash(
    dividends: pd.Series, splits: pd.Series, origin: pd.Timestamp,
) -> pd.Series:
    """Raw per-share payments restated to one origin-date share."""
    origin = pd.Timestamp(origin)
    amounts = []
    for date, amount in dividends.items():
        stamp = pd.Timestamp(date)
        if stamp <= origin:
            amounts.append(float(amount) / share_factor(splits, stamp, origin))
        else:
            amounts.append(float(amount) * share_factor(splits, origin, stamp))
    return pd.Series(amounts, index=dividends.index, dtype=float)


def _cash_between(
    cash: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
) -> float:
    """Cash with ex-dates in (start, end]."""
    return float(cash.loc[(cash.index > start) & (cash.index <= end)].sum())


def next_cash_12m(
    dividends: pd.Series, splits: pd.Series, origin: pd.Timestamp,
) -> tuple[float, pd.Timestamp]:
    """Next-12M cash per origin share to the labelled business month-end."""
    origin = pd.Timestamp(origin)
    end = label_end(origin, 12)
    cash = origin_basis_cash(dividends, splits, origin)
    return _cash_between(cash, origin, end), end


def past_cash(
    dividends: pd.Series,
    splits: pd.Series,
    origin: pd.Timestamp,
    start_months: int,
    end_months: int,
) -> float:
    """Cash per origin share in (origin-start_months, origin-end_months]."""
    if start_months <= end_months or end_months < 0:
        raise ValueError("Use a positive backward window before the origin")
    origin = pd.Timestamp(origin)
    cash = origin_basis_cash(dividends, splits, origin)
    return _cash_between(
        cash,
        origin - pd.DateOffset(months=start_months),
        origin - pd.DateOffset(months=end_months),
    )


def filed_reports(reports: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Reports whose month and actual filing date are both by the origin."""
    origin = pd.Timestamp(origin)
    filed = reports.loc[
        (pd.to_datetime(reports["month_end"]) <= origin)
        & (pd.to_datetime(reports["filing_date"]) <= origin)
    ].copy()
    filed["month_end"] = pd.to_datetime(filed["month_end"])
    filed["filing_date"] = pd.to_datetime(filed["filing_date"])
    filed = filed.sort_values(["month_end", "filing_date"], kind="stable")
    filed = filed.drop_duplicates("month_end", keep="first")
    filed.index = pd.PeriodIndex(filed["month_end"], freq="M")
    return filed


def _on_origin_basis(
    value: float,
    month_end: pd.Timestamp,
    splits: pd.Series,
    origin: pd.Timestamp,
) -> float:
    """Restate a per-share report value to one origin-date share."""
    if month_end <= origin:
        return float(value) / share_factor(splits, month_end, origin)
    return float(value) * share_factor(splits, origin, month_end)


def restate_to_basis(
    values: np.ndarray | Sequence[float],
    origins: pd.DatetimeIndex | Sequence[object],
    splits: pd.Series,
    basis: pd.Timestamp,
) -> np.ndarray:
    """Restate per-origin-share dollar values to one ``basis``-date share.

    Runners use the fold's first test origin as the basis, so only splits
    known by that decision date rescale the training history.
    """
    basis = pd.Timestamp(basis)
    return np.array([
        _on_origin_basis(value, pd.Timestamp(origin), splits, basis)
        if np.isfinite(value) else float("nan")
        for value, origin in zip(
            np.asarray(values, dtype=float), pd.DatetimeIndex(origins)
        )
    ])


def _future_report(
    reports: pd.DataFrame, month: pd.Period,
) -> pd.Series | None:
    """First filed report for an exact calendar month, if any."""
    frame = reports.copy()
    frame["month_end"] = pd.to_datetime(frame["month_end"])
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    rows = frame.loc[frame["month_end"].dt.to_period("M") == month]
    if rows.empty:
        return None
    return rows.sort_values("filing_date", kind="stable").iloc[0]


def _growth_parts(
    reports: pd.DataFrame,
    splits: pd.Series,
    origin: pd.Timestamp,
    months: int,
) -> tuple[pd.Series, pd.Series, float, float] | None:
    """Latest filed report at origin and its exact report-month+h report."""
    origin = pd.Timestamp(origin)
    filed = filed_reports(reports, origin)
    if filed.empty:
        return None
    current_row = filed.iloc[-1]
    future_row = _future_report(reports, filed.index[-1] + months)
    if future_row is None:
        return None
    current = _on_origin_basis(
        current_row["book_value_per_share"],
        current_row["month_end"],
        splits,
        origin,
    )
    future = _on_origin_basis(
        future_row["book_value_per_share"],
        future_row["month_end"],
        splits,
        origin,
    )
    if not (np.isfinite(current) and np.isfinite(future)) or current <= 0:
        return None
    return current_row, future_row, current, future


def _target_record(
    value: float, current_row: pd.Series, future_row: pd.Series,
) -> dict[str, Any]:
    end = pd.offsets.BMonthEnd().rollback(
        future_row["month_end"].to_period("M").to_timestamp(how="end")
        .normalize()
    )
    return {
        "value": float(value),
        "report_month": pd.Timestamp(current_row["month_end"]),
        "future_month": pd.Timestamp(future_row["month_end"]),
        "target_end": pd.Timestamp(end),
        "available": pd.Timestamp(future_row["filing_date"]),
    }


def bvps_growth_target(
    reports: pd.DataFrame,
    splits: pd.Series,
    origin: pd.Timestamp,
    months: int,
) -> dict[str, Any] | None:
    """Latest filed BVPS to report month+h on the origin share basis."""
    parts = _growth_parts(reports, splits, origin, months)
    if parts is None:
        return None
    current_row, future_row, current, future = parts
    return _target_record(future / current - 1.0, current_row, future_row)


def adjusted_bvps_growth_target(
    reports: pd.DataFrame,
    dividends: pd.Series,
    splits: pd.Series,
    origin: pd.Timestamp,
    months: int,
) -> dict[str, Any] | None:
    """Archived x12/x16 dividend-adjusted BVPS growth, split-consistent.

    (future BVPS + cash with ex-dates after the current report month through
    the future report month) / current BVPS - 1, all per origin share.
    """
    parts = _growth_parts(reports, splits, origin, months)
    if parts is None:
        return None
    current_row, future_row, current, future = parts
    cash = origin_basis_cash(dividends, splits, pd.Timestamp(origin))
    paid = _cash_between(
        cash, current_row["month_end"], future_row["month_end"]
    )
    return _target_record(
        (future + paid) / current - 1.0, current_row, future_row
    )


def annual_window(origin: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Archived x18 December 1 through next February 28/29 window."""
    year = pd.Timestamp(origin).year
    start = pd.Timestamp(year, 12, 1)
    end = pd.Timestamp(year + 1, 3, 1) - pd.Timedelta(days=1)
    return start, end


def annual_excess_target(
    dividends: pd.Series,
    splits: pd.Series,
    reports: pd.DataFrame,
    origin: pd.Timestamp,
) -> dict[str, Any] | None:
    """Archived x23 excess dividend / current BVPS at a November origin.

    The ordinary baseline is the v200-frozen median of positive payments
    <= .25 in the 24 months to the origin. Pre-policy origins, origins
    without an ordinary baseline and origins without a filed BVPS are
    outside the endpoint; no fallback is invented.
    """
    origin = pd.Timestamp(origin)
    start, end = annual_window(origin)
    if origin.month != 11 or start < POLICY_CHANGE:
        return None
    cash = origin_basis_cash(dividends, splits, origin)
    ordinary = regular_dividend_baseline(cash, origin)
    filed = filed_reports(reports, origin)
    if not np.isfinite(ordinary) or filed.empty:
        return None
    latest = filed.iloc[-1]
    current = _on_origin_basis(
        latest["book_value_per_share"], latest["month_end"], splits, origin
    )
    if not np.isfinite(current) or current <= 0:
        return None
    total = _cash_between(cash, start - pd.Timedelta(days=1), end)
    ratio = excess_ratio(total, ordinary, current)
    return {
        "value": float(ratio),
        "ordinary": float(ordinary),
        "window_cash": float(total),
        "current_bvps": float(current),
        "report_month": pd.Timestamp(latest["month_end"]),
        "positive": bool(ratio > 0),
        "target_end": end,
        "available": end,
        "event_id": f"annual-dec-feb:{origin.year + 1}",
    }


def _window_values(
    filed: pd.DataFrame, column: str, last: pd.Period, length: int,
) -> np.ndarray | None:
    """Exact contiguous calendar months ending at ``last``, else None."""
    months = pd.period_range(last - (length - 1), last, freq="M")
    if column not in filed or not set(months).issubset(filed.index):
        return None
    values = pd.to_numeric(filed.loc[months, column], errors="coerce")
    array = values.to_numpy(dtype=float)
    return array if np.isfinite(array).all() else None


def _ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return float("nan")
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator - 1.0)


def lane_features(
    reports: pd.DataFrame,
    dividends: pd.Series,
    splits: pd.Series,
    origin: pd.Timestamp,
) -> dict[str, Any]:
    """Rolling monthly lane features known at one business month-end."""
    origin = pd.Timestamp(origin)
    result: dict[str, Any] = {name: float("nan") for name in FEATURE_COLUMNS}
    result.update(
        report_month=pd.NaT,
        report_filing_date=pd.NaT,
        past12_cash=past_cash(dividends, splits, origin, 12, 0),
        prior_past12_cash=past_cash(dividends, splits, origin, 24, 12),
        month_of_year=float(origin.month),
        q4_flag=float(origin.quarter == 4),
        dividend_season_flag=float(origin.month in (11, 12, 1)),
    )
    filed = filed_reports(reports, origin)
    if filed.empty:
        return result
    last = filed.index[-1]
    latest = filed.iloc[-1]
    result.update(
        report_month=pd.Timestamp(latest["month_end"]),
        report_filing_date=pd.Timestamp(latest["filing_date"]),
    )

    def value(column: str, lag: int) -> float:
        month = last - lag
        if column not in filed or month not in filed.index:
            return float("nan")
        return float(pd.to_numeric(filed.loc[month, column], errors="coerce"))

    def bvps(lag: int) -> float:
        month = last - lag
        raw = value("book_value_per_share", lag)
        if not np.isfinite(raw):
            return raw
        return _on_origin_basis(
            raw, month.to_timestamp(how="end").normalize(), splits, origin
        )

    ratios = _window_values(filed, "combined_ratio", last, 12)
    result["cr_ttm"] = float(ratios.mean()) if ratios is not None else np.nan
    result["pif_growth_yoy_cal"] = _ratio(
        value("pif_total", 0), value("pif_total", 12)
    )
    premiums = _window_values(filed, "net_premiums_written", last, 24)
    if premiums is not None:
        result["npw_growth_ttm"] = _ratio(
            float(premiums[12:].sum()), float(premiums[:12].sum())
        )
    current = bvps(0)
    result["current_bvps"] = current
    for name, lag in (
        ("bvps_growth_1m", 1),
        ("bvps_growth_3m", 3),
        ("bvps_growth_6m", 6),
        ("bvps_growth_yoy", 12),
    ):
        result[name] = _ratio(current, bvps(lag))
    result["bvps_growth_ytd"] = _ratio(current, bvps(last.month - 1))
    result["bvps_yoy_dollar_change"] = current - bvps(12)
    result["roe_ttm"] = value("roe_net_income_ttm", 0)
    result["gainshare_estimate"] = value("gainshare_estimate", 0)
    result["investment_book_yield"] = value("investment_book_yield", 0)
    return result


def nested_ridge_forecast(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    train_index: np.ndarray,
    test_index: np.ndarray,
    horizon: int,
    grid: Sequence[float] = ALPHA_GRID,
) -> dict[str, Any]:
    """Select one Ridge penalty in three inner folds, then forecast.

    ``frame`` is indexed by contiguous business month-end origins and holds
    the features, the training target and each label's explicit
    ``label_end`` and ``available`` dates. Training rows must have ended and
    arrived by the relevant (inner or outer) test origin; inner validation
    labels must also have arrived by the outer test origin. The inner loss
    is summed absolute error of the supplied training target (the scored
    endpoint, or a registered proxy label); exact ties choose the larger
    penalty. The outer training fold also needs the minimum usable support.
    Unsupported histories are never shortened.
    """
    if horizon not in (6, 12):
        raise ValueError("The registered horizons are six and twelve months.")
    dates = pd.DatetimeIndex(frame.index)
    minimum = 24 if horizon == 6 else 60
    train_index = np.asarray(train_index, dtype=int)
    test_index = np.asarray(test_index, dtype=int)
    first_test = dates[test_index].min()
    x = frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    finite_x = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    y = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
    ends = pd.DatetimeIndex(pd.to_datetime(frame["label_end"]))
    arrivals = pd.DatetimeIndex(pd.to_datetime(frame["available"]))
    labelled = np.isfinite(y) & finite_x & ends.notna() & arrivals.notna()

    def usable(rows: np.ndarray, cutoff: pd.Timestamp) -> np.ndarray:
        keep = (
            labelled[rows]
            & (dates[rows] < cutoff)
            & (ends[rows] <= cutoff)
            & (arrivals[rows] <= cutoff)
        )
        return rows[np.asarray(keep, dtype=bool)]

    result: dict[str, Any] = {
        "status": "unscorable",
        "reason": "",
        "alpha": float("nan"),
        "prediction": np.full(len(test_index), np.nan),
        "inner": [],
        "inner_absolute_error": {},
        "n_train": 0,
        "train_start": str(dates[train_index[0]].date()),
        "train_end": str(dates[train_index[-1]].date()),
        "test_start": str(dates[test_index[0]].date()),
        "test_end": str(dates[test_index[-1]].date()),
    }
    outer = usable(train_index, first_test)
    result["n_train"] = len(outer)
    if len(outer) < minimum:
        result["reason"] = "Insufficient mature outer training labels"
        return result
    splits = chronological_splits(dates[train_index], horizon, inner=True)
    if len(splits) != 3:
        result["reason"] = "Unconstructible three-fold inner history"
        return result
    errors = np.zeros(len(grid))
    supported = True
    for inner_fold, (inner_train, validation) in enumerate(splits):
        absolute_train = train_index[inner_train]
        absolute_validation = train_index[validation]
        cutoff = dates[absolute_validation[0]]
        rows = usable(absolute_train, cutoff)
        keep = (
            labelled[absolute_validation]
            & (ends[absolute_validation] <= first_test)
            & (arrivals[absolute_validation] <= first_test)
        )
        checked = absolute_validation[np.asarray(keep, dtype=bool)]
        entry: dict[str, Any] = {
            "inner_fold": inner_fold,
            "test_size": len(validation),
            "gap": 2 * horizon,
            "n_train": len(rows),
            "n_test_usable": len(checked),
            "train_start": str(dates[absolute_train[0]].date()),
            "train_end": str(dates[absolute_train[-1]].date()),
            "test_start": str(cutoff.date()),
            "test_end": str(dates[absolute_validation[-1]].date()),
            "status": "scorable",
        }
        if len(rows) < minimum or not len(checked):
            entry["status"] = "unscorable"
            supported = False
            result["inner"].append(entry)
            continue
        for position, alpha in enumerate(grid):
            model = ridge_pipeline(float(alpha))
            model.fit(x.iloc[rows], y[rows])
            predicted = model.predict(x.iloc[checked])
            errors[position] += float(np.abs(y[checked] - predicted).sum())
        result["inner"].append(entry)
    if not supported:
        result["reason"] = "Unsupported fixed three-fold inner history"
        return result
    best = float(errors.min())
    tolerance = 1e-12 * max(1.0, abs(best))
    alpha = max(
        float(a) for a, error in zip(grid, errors) if error <= best + tolerance
    )
    model = ridge_pipeline(alpha)
    model.fit(x.iloc[outer], y[outer])
    prediction = np.full(len(test_index), np.nan)
    ready = finite_x[test_index]
    if ready.any():
        prediction[ready] = model.predict(x.iloc[test_index[ready]])
    result.update(
        status="scorable",
        reason="Nested alpha selected on inner absolute error",
        alpha=alpha,
        prediction=prediction,
        inner_absolute_error={
            str(float(a)): float(error) for a, error in zip(grid, errors)
        },
    )
    return result


def _block_draws(
    n: int, h: int, replicates: int, seed: int,
) -> list[np.ndarray]:
    """Moving blocks of h consecutive observed dates, trimmed to n."""
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / h))
    draws = []
    for _ in range(replicates):
        starts = rng.integers(0, n - h + 1, size=blocks)
        draws.append(
            np.concatenate([np.arange(s, s + h) for s in starts])[:n]
        )
    return draws


def paired_block_test(
    dates: pd.DatetimeIndex | Sequence[object],
    improvement: np.ndarray | Sequence[float],
    h: int,
    replicates: int = 2000,
    seed: int = 20260926,
) -> dict[str, Any]:
    """One-sided centered moving-date-block test of mean improvement > 0.

    One value per monthly date (already averaged within the date). At least
    two full blocks are required; otherwise no inference is reported.
    """
    index = pd.DatetimeIndex(dates)
    values = np.asarray(improvement, dtype=float)
    if len(index) != len(values) or index.has_duplicates:
        raise ValueError("Supply one improvement per unique date")
    order = np.argsort(index.asi8, kind="stable")
    values = values[order]
    if not np.isfinite(values).all():
        raise ValueError("Improvements must be finite")
    observed = float(values.mean()) if len(values) else float("nan")
    result: dict[str, Any] = {
        "observed": observed,
        "n_dates": len(values),
        "block_length": h,
        "replicates": replicates,
        "seed": seed,
        "inference_supported": len(values) >= 2 * h,
        "p_value": float("nan"),
        "ci": [float("nan"), float("nan")],
    }
    if not result["inference_supported"]:
        return result
    means = np.array([
        values[rows].mean() for rows in _block_draws(
            len(values), h, replicates, seed
        )
    ])
    result["ci"] = np.quantile(means, [0.025, 0.975]).tolist()
    if observed <= 0.0:
        result["p_value"] = 1.0
    else:
        exceed = (means - observed) >= observed
        result["p_value"] = float((1 + exceed.sum()) / (1 + replicates))
    return result


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or np.ptp(y) == 0.0 or np.ptp(p) == 0.0:
        return float("nan")
    ranked_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    ranked_p = pd.Series(p).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(ranked_y, ranked_p)[0, 1])


def block_spearman(
    dates: pd.DatetimeIndex | Sequence[object],
    y: np.ndarray | Sequence[float],
    p: np.ndarray | Sequence[float],
    h: int,
    replicates: int = 2000,
    seed: int = 20260926,
) -> dict[str, Any]:
    """Spearman IC with a two-sided centered moving-date-block p-value."""
    index = pd.DatetimeIndex(dates)
    order = np.argsort(index.asi8, kind="stable")
    actual = np.asarray(y, dtype=float)[order]
    predicted = np.asarray(p, dtype=float)[order]
    observed = _spearman(actual, predicted)
    result: dict[str, Any] = {
        "ic": observed,
        "n_dates": len(actual),
        "block_length": h,
        "inference_supported": len(actual) >= 2 * h,
        "p_value": float("nan"),
        "ci": [float("nan"), float("nan")],
    }
    if not result["inference_supported"] or not np.isfinite(observed):
        return result
    draws = np.array([
        _spearman(actual[rows], predicted[rows])
        for rows in _block_draws(len(actual), h, replicates, seed)
    ])
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return result
    result["ci"] = np.quantile(draws, [0.025, 0.975]).tolist()
    exceed = np.abs(draws - observed) >= abs(observed)
    result["p_value"] = float((1 + exceed.sum()) / (1 + len(draws)))
    return result


def prequential_stream(
    frame: pd.DataFrame, min_support: int,
) -> pd.DataFrame:
    """Nominal 80% intervals and event probabilities from matured residuals.

    Each row uses only residuals of earlier forecasts in the same stream
    whose labels had arrived by its origin. Rows with fewer than
    ``min_support`` matured residuals are warmup and stay unevaluated.
    The event is ``y_true > threshold``; its probability is the matured
    residual frequency of ``y_hat + residual > threshold``.
    """
    required = {"date", "available", "y_true", "y_hat", "threshold"}
    if not required.issubset(frame.columns):
        raise ValueError("Stream needs date, available, y_true, y_hat, threshold")
    output = frame.sort_values("date", kind="stable").copy()
    dates = pd.DatetimeIndex(pd.to_datetime(output["date"]))
    arrivals = pd.DatetimeIndex(pd.to_datetime(output["available"]))
    actual = output["y_true"].to_numpy(dtype=float)
    forecast = output["y_hat"].to_numpy(dtype=float)
    threshold = output["threshold"].to_numpy(dtype=float)
    residual = actual - forecast
    lower = np.full(len(output), np.nan)
    upper = np.full(len(output), np.nan)
    probability = np.full(len(output), np.nan)
    count = np.zeros(len(output), dtype=int)
    for row in range(len(output)):
        past = (
            (dates < dates[row])
            & (arrivals <= dates[row])
            & np.isfinite(residual)
        )
        history = residual[np.asarray(past, dtype=bool)]
        count[row] = len(history)
        if len(history) < min_support or not np.isfinite(forecast[row]):
            continue
        low, high = np.quantile(history, [0.1, 0.9])
        lower[row] = forecast[row] + low
        upper[row] = forecast[row] + high
        probability[row] = float(
            np.mean(forecast[row] + history > threshold[row])
        )
    output["lower"] = lower
    output["upper"] = upper
    output["probability"] = probability
    output["n_matured_residuals"] = count
    output["warmup"] = ~np.isfinite(probability)
    output["event"] = (actual > threshold).astype(float)
    covered = (actual >= lower) & (actual <= upper)
    output["covered"] = np.where(output["warmup"], np.nan, covered)
    return output


def ece_equal_width(
    probability: np.ndarray | Sequence[float],
    outcome: np.ndarray | Sequence[float],
    bins: int = 10,
) -> float:
    """Expected calibration error over fixed equal-width probability bins."""
    p = np.asarray(probability, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if not len(p):
        return float("nan")
    index = np.minimum(np.floor(p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            gap = abs(p[members].mean() - y[members].mean())
            total += members.sum() / len(p) * gap
    return float(total)


def probability_scores(
    probability: np.ndarray | Sequence[float],
    outcome: np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Brier, clipped log loss and equal-width ECE for binary events."""
    p = np.asarray(probability, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if not len(p):
        nan = float("nan")
        return {"brier": nan, "log_loss": nan, "ece": nan, "n": 0}
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(
            -np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
        ),
        "ece": ece_equal_width(p, y),
        "n": len(p),
    }


def majority_base(
    dates: pd.DatetimeIndex | Sequence[object],
    available: pd.DatetimeIndex | Sequence[object],
    events: np.ndarray | Sequence[float],
    origins: pd.DatetimeIndex | Sequence[object],
) -> np.ndarray:
    """Constant majority event rule learned from labels matured by origin."""
    label_dates = pd.DatetimeIndex(dates)
    arrivals = pd.DatetimeIndex(available)
    values = np.asarray(events, dtype=float)
    result = []
    for origin in pd.DatetimeIndex(origins):
        keep = (
            (label_dates < origin)
            & (arrivals <= origin)
            & np.isfinite(values)
        )
        mature = values[np.asarray(keep, dtype=bool)]
        result.append(float(mature.mean() >= 0.5) if len(mature) else np.nan)
    return np.asarray(result, dtype=float)


def lane_disposition(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered v205 lane threshold; NaN never passes."""

    def finite(*names: str) -> bool:
        return all(
            isinstance(summary.get(name), (int, float))
            and np.isfinite(float(summary[name]))
            for name in names
        )

    target = 0.80
    gates = {
        "mae_reduction_at_least_10pct": finite("relative_mae_reduction")
        and summary["relative_mae_reduction"]
        >= THRESHOLDS["relative_mae_reduction"],
        "holm_adjusted_p_below_05": finite("adjusted_p")
        and summary["adjusted_p"] < THRESHOLDS["adjusted_p"],
        "delta_r2_at_least_01": finite("delta_r2")
        and summary["delta_r2"] >= THRESHOLDS["delta_r2"],
        "no_directional_deterioration": finite("hit_rate", "control_hit_rate")
        and summary["hit_rate"] >= summary["control_hit_rate"],
        "no_brier_deterioration": finite("brier", "control_brier")
        and summary["brier"] <= summary["control_brier"],
        "no_coverage_deterioration": finite(
            "coverage_80", "control_coverage_80"
        )
        and abs(summary["coverage_80"] - target)
        <= abs(summary["control_coverage_80"] - target)
        + THRESHOLDS["coverage_tolerance"],
        "sufficient_independent_events": finite("independent_blocks")
        and summary["independent_blocks"]
        >= THRESHOLDS["independent_blocks"],
        "inference_supported": summary.get("inference_supported") is True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {"passes": not failed, "failed": failed, "gates": gates}


def campaign_holm(v205_p: Sequence[float]) -> list[float]:
    """Holm-adjust v205's six primary p-values inside the 38-test family.

    Pending or unused slots of other steps are conservatively p=1; v207
    completes the campaign adjustment.
    """
    values = [float(value) for value in v205_p]
    if len(values) != len(CAMPAIGN_SLOTS):
        raise ValueError("v205 has exactly six registered campaign slots")
    family = [1.0] * 38
    for slot, value in zip(CAMPAIGN_SLOTS, values):
        family[slot] = value
    adjusted = holm38(family)
    return [adjusted[slot] for slot in CAMPAIGN_SLOTS]


def nominate_winner(rows: Sequence[Mapping[str, Any]]) -> str | None:
    """At most one passing lane: lowest adjusted p, then largest MAE cut."""
    passing = [row for row in rows if row.get("passes") is True]
    if not passing:
        return None
    best = min(
        passing,
        key=lambda row: (
            float(row["adjusted_p"]), -float(row["relative_mae_reduction"])
        ),
    )
    return str(best["candidate"])
