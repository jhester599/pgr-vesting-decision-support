"""Prequential (realised-only) reconstruction of the ensemble's OOS record.

Review 2026-09-25, F13 and F04: the monthly health metrics used to be computed
with information a forecaster could not have had at the time. The inverse-MAE
ensemble weights came from errors over the whole OOS history, the shrinkage
alpha (0.50) was chosen on that same history (v38), and the OOS-R^2 naive
benchmark included the target it was compared with.

Everything here follows one rule. A target labelled ``d`` (the start of its
forward window) is realised ``horizon_months`` calendar months later, so a
forecaster standing at date ``t`` may use it only when::

    month(d) + horizon_months <= month(t)

Under that rule, for every OOS row at date ``t``:

- the Ridge/GBT weights are ``1 / MAE^2`` over the realised OOS rows of that
  benchmark (equal weights before any are realised);
- the shrinkage alpha is v38's rule applied to the realised OOS rows of all
  benchmarks: the grid value (``config.ENSEMBLE_SHRINKAGE_ALPHA_GRID``) that
  minimises ``sum((y - alpha * z)^2)`` (1.0, i.e. no shrinkage, until
  ``min_obs`` rows are realised);
- the naive benchmark is the mean of that benchmark's realised targets,
  including the training history (Campbell-Thompson prevailing mean).

A live forecast at the as-of date uses the same rule; every OOS row is realised
by then, because the target matrix is truncated to windows that ended on or
before the as-of date.

For a non-datetime index, positions stand for consecutive months.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import config

if TYPE_CHECKING:
    from src.models.multi_benchmark_wfo import EnsembleWFOResult

PANEL_BASE_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "date",
    "y_true",
    "z",
    "alpha",
    "y_hat",
    "naive",
)


def month_ordinals(index: pd.Index | pd.Series | np.ndarray | list) -> np.ndarray:
    """Return ``year * 12 + month`` for dates.

    An integer index is taken as consecutive month numbers (its labels, so
    dropping rows does not shift the months); any other index uses positions.
    """
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    if isinstance(index, pd.DatetimeIndex):
        return (index.year * 12 + index.month - 1).to_numpy(dtype=np.int64)
    if len(index) and isinstance(index[0], (pd.Timestamp, np.datetime64)):
        dt_index = pd.DatetimeIndex(index)
        return (dt_index.year * 12 + dt_index.month - 1).to_numpy(dtype=np.int64)
    if pd.api.types.is_integer_dtype(index.dtype):
        return index.to_numpy(dtype=np.int64)
    return np.arange(len(index), dtype=np.int64)


def realised_counts(
    history_months: np.ndarray,
    forecast_months: np.ndarray,
    horizon_months: int,
) -> np.ndarray:
    """Number of (sorted) history rows realised at each forecast month.

    ``history_months`` must be sorted ascending. A history row is realised at
    forecast month ``m`` when ``history_month + horizon_months <= m``.
    """
    return np.searchsorted(
        np.asarray(history_months, dtype=np.int64),
        np.asarray(forecast_months, dtype=np.int64) - int(horizon_months),
        side="right",
    )


def prevailing_mean_forecast(
    forecast_index: pd.Index,
    target_history: pd.Series,
    horizon_months: int,
) -> pd.Series:
    """Mean of the targets realised by each forecast date (NaN before any).

    ``target_history`` is the benchmark's full target series, including the
    training history. Its index uses the same convention as ``forecast_index``
    (dates, or positions standing for consecutive months).
    """
    history = target_history.dropna().astype(float).sort_index()
    history_months = month_ordinals(history.index)
    forecast_months = month_ordinals(forecast_index)
    counts = realised_counts(history_months, forecast_months, horizon_months)
    cumulative = np.concatenate([[0.0], np.cumsum(history.to_numpy(dtype=float))])
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(counts > 0, cumulative[counts] / np.maximum(counts, 1), np.nan)
    return pd.Series(values, index=forecast_index, name="naive", dtype=float)


def component_oos_frame(ens_result: "EnsembleWFOResult") -> pd.DataFrame:
    """Return one row per OOS date: ``y_true`` plus each model's prediction."""
    columns: dict[str, pd.Series] = {}
    y_true: pd.Series | None = None
    for model_type, result in ens_result.model_results.items():
        if not result.folds:
            continue
        dates = pd.DatetimeIndex(result.test_dates_all)
        pred = pd.Series(result.y_hat_all, index=dates, dtype=float)
        truth = pd.Series(result.y_true_all, index=dates, dtype=float)
        if pred.index.has_duplicates:
            raise ValueError(
                f"{ens_result.benchmark}/{model_type}: duplicate OOS dates in WFO folds."
            )
        columns[str(model_type)] = pred
        y_true = truth if y_true is None else y_true
    if not columns or y_true is None:
        return pd.DataFrame(columns=["y_true"])
    frame = pd.DataFrame(columns).dropna(how="any")
    frame.insert(0, "y_true", y_true.reindex(frame.index))
    frame = frame.dropna(subset=["y_true"]).sort_index()
    frame.index.name = "date"
    return frame


def inverse_mae_weights(abs_errors: pd.DataFrame | dict[str, float]) -> dict[str, float]:
    """Normalised ``1 / MAE^2`` weights (weight 1.0 when an MAE is ~0)."""
    if isinstance(abs_errors, pd.DataFrame):
        maes = {str(col): float(abs_errors[col].mean()) for col in abs_errors.columns}
    else:
        maes = {str(key): float(value) for key, value in abs_errors.items()}
    raw = {key: (1.0 / (mae**2) if mae > 1e-9 else 1.0) for key, mae in maes.items()}
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def prequential_inverse_mae_weights(
    frame: pd.DataFrame,
    model_cols: list[str],
    horizon_months: int,
) -> pd.DataFrame:
    """Per-row ``1 / MAE^2`` weights from each model's realised OOS errors."""
    months = month_ordinals(frame.index)
    counts = realised_counts(months, months, horizon_months)
    raw = np.ones((len(frame), len(model_cols)), dtype=float)
    for col_i, col in enumerate(model_cols):
        abs_err = np.abs(frame["y_true"].to_numpy(dtype=float) - frame[col].to_numpy(dtype=float))
        cumulative = np.concatenate([[0.0], np.cumsum(abs_err)])
        with np.errstate(invalid="ignore", divide="ignore"):
            mae = np.where(counts > 0, cumulative[counts] / np.maximum(counts, 1), np.nan)
        has_mae = counts > 0
        weight = np.ones(len(frame), dtype=float)
        positive = has_mae & (mae > 1e-9)
        weight[positive] = 1.0 / (mae[positive] ** 2)
        raw[:, col_i] = weight
    # Rows with nothing realised yet get equal weights.
    raw[counts == 0, :] = 1.0
    weights = raw / raw.sum(axis=1, keepdims=True)
    return pd.DataFrame(weights, index=frame.index, columns=list(model_cols))


def _grid(alpha_grid: tuple[float, ...] | list[float] | None) -> np.ndarray:
    grid = config.ENSEMBLE_SHRINKAGE_ALPHA_GRID if alpha_grid is None else alpha_grid
    values = np.asarray(sorted(float(value) for value in grid), dtype=float)
    if values.size == 0:
        raise ValueError("alpha_grid must not be empty.")
    return values


def best_grid_shrinkage(
    sum_zy: float,
    sum_zz: float,
    alpha_grid: tuple[float, ...] | list[float] | None = None,
) -> float:
    """Grid alpha minimising ``sum((y - alpha z)^2)`` from the sufficient sums.

    ``SSE(alpha) = sum(y^2) - 2 alpha sum(zy) + alpha^2 sum(z^2)``; the
    ``sum(y^2)`` term does not depend on alpha, so it is left out. Ties go to
    the larger alpha (less shrinkage).
    """
    grid = _grid(alpha_grid)
    sse = -2.0 * grid * float(sum_zy) + grid**2 * float(sum_zz)
    best = np.flatnonzero(sse <= sse.min() + 1e-15)
    return float(grid[best[-1]])


def least_squares_shrinkage(
    z: np.ndarray,
    y: np.ndarray,
    min_obs: int,
    alpha_grid: tuple[float, ...] | list[float] | None = None,
) -> float:
    """v38 grid alpha minimising the squared error; 1.0 (no shrinkage) below ``min_obs``."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(z) & np.isfinite(y)
    z, y = z[valid], y[valid]
    sum_zz = float(np.sum(z * z))
    if len(z) < max(int(min_obs), 1) or sum_zz <= 1e-18:
        return 1.0
    return best_grid_shrinkage(float(np.sum(z * y)), sum_zz, alpha_grid)


def prequential_shrinkage_alphas(
    dates: pd.Index | pd.Series,
    z: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    horizon_months: int,
    min_obs: int | None = None,
    alpha_grid: tuple[float, ...] | list[float] | None = None,
) -> np.ndarray:
    """Grid shrinkage alpha per row from the rows realised by that row's date.

    Rows may come from several benchmarks (the alpha is pooled), so dates can
    repeat. The result is aligned with the input order.
    """
    if min_obs is None:
        min_obs = config.ENSEMBLE_SHRINKAGE_MIN_REALIZED_OBS
    months = month_ordinals(pd.Index(dates))
    z_arr = np.asarray(z, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    order = np.argsort(months, kind="mergesort")
    sorted_months = months[order]
    zy = np.concatenate([[0.0], np.cumsum(z_arr[order] * y_arr[order])])
    zz = np.concatenate([[0.0], np.cumsum(z_arr[order] ** 2)])
    counts = realised_counts(sorted_months, months, horizon_months)
    alphas = np.ones(len(months), dtype=float)
    cache: dict[int, float] = {}
    for row, count in enumerate(counts):
        count = int(count)
        if count < max(int(min_obs), 1) or zz[count] <= 1e-18:
            continue
        if count not in cache:
            cache[count] = best_grid_shrinkage(zy[count], zz[count], alpha_grid)
        alphas[row] = cache[count]
    return alphas


def _ensemble_horizon(ens_result: "EnsembleWFOResult") -> int:
    """Target horizon of an ensemble result (its members' horizon as fallback)."""
    horizon = getattr(ens_result, "target_horizon", None)
    if horizon is None:
        for result in getattr(ens_result, "model_results", {}).values():
            horizon = getattr(result, "target_horizon", None)
            if horizon is not None:
                break
    return int(horizon if horizon is not None else 6)


def build_prequential_panel(
    ensemble_results: dict[str, "EnsembleWFOResult"],
    horizon_months: int | None = None,
    min_alpha_obs: int | None = None,
) -> pd.DataFrame:
    """Rebuild every benchmark's ensemble OOS record with realised-only choices.

    Returns one row per (benchmark, OOS date) with columns ``benchmark``,
    ``date``, ``y_true``, ``z`` (weighted ensemble before shrinkage), ``alpha``,
    ``y_hat`` (``alpha * z``), ``naive`` (prevailing mean), and per model
    ``pred_<model>`` and ``w_<model>``. Sorted by date, then benchmark.
    """
    parts: list[pd.DataFrame] = []
    horizons: set[int] = set()
    for benchmark, ens_result in ensemble_results.items():
        frame = component_oos_frame(ens_result)
        if frame.empty:
            continue
        horizon = int(
            horizon_months if horizon_months is not None else _ensemble_horizon(ens_result)
        )
        horizons.add(horizon)
        model_cols = [col for col in frame.columns if col != "y_true"]
        weights = prequential_inverse_mae_weights(frame, model_cols, horizon)
        z = (frame[model_cols] * weights).sum(axis=1)
        history = getattr(ens_result, "target_history", None)
        if history is None or len(history) == 0:
            history = frame["y_true"]
        naive = prevailing_mean_forecast(frame.index, history, horizon)
        part = pd.DataFrame(
            {
                "benchmark": str(benchmark),
                "date": frame.index,
                "y_true": frame["y_true"].to_numpy(dtype=float),
                "z": z.to_numpy(dtype=float),
                "naive": naive.to_numpy(dtype=float),
            }
        )
        for col in model_cols:
            part[f"pred_{col}"] = frame[col].to_numpy(dtype=float)
            part[f"w_{col}"] = weights[col].to_numpy(dtype=float)
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=list(PANEL_BASE_COLUMNS))
    if len(horizons) > 1:
        raise ValueError(f"Cannot pool benchmarks with different horizons: {sorted(horizons)}")
    horizon = horizons.pop()

    panel = pd.concat(parts, ignore_index=True)
    panel["alpha"] = prequential_shrinkage_alphas(
        panel["date"],
        panel["z"].to_numpy(dtype=float),
        panel["y_true"].to_numpy(dtype=float),
        horizon_months=horizon,
        min_obs=min_alpha_obs,
    )
    panel["y_hat"] = panel["alpha"] * panel["z"]
    panel = panel.sort_values(["date", "benchmark"], kind="mergesort").reset_index(drop=True)
    panel.attrs["horizon_months"] = horizon
    ordered = [*PANEL_BASE_COLUMNS, *[c for c in panel.columns if c not in PANEL_BASE_COLUMNS]]
    return panel[ordered]


def live_shrinkage_alpha(panel: pd.DataFrame, min_obs: int | None = None) -> float:
    """Shrinkage alpha for the live forecast: every OOS row is realised by then."""
    if min_obs is None:
        min_obs = config.ENSEMBLE_SHRINKAGE_MIN_REALIZED_OBS
    if panel.empty:
        return 1.0
    return least_squares_shrinkage(
        panel["z"].to_numpy(dtype=float),
        panel["y_true"].to_numpy(dtype=float),
        min_obs=min_obs,
    )


def benchmark_series(panel: pd.DataFrame, benchmark: str, column: str) -> pd.Series:
    """Return one benchmark's panel column as a date-indexed Series."""
    rows = panel[panel["benchmark"] == str(benchmark)]
    return pd.Series(
        rows[column].to_numpy(dtype=float),
        index=pd.DatetimeIndex(rows["date"]),
        name=column,
    )


__all__ = [
    "PANEL_BASE_COLUMNS",
    "benchmark_series",
    "build_prequential_panel",
    "component_oos_frame",
    "inverse_mae_weights",
    "best_grid_shrinkage",
    "least_squares_shrinkage",
    "live_shrinkage_alpha",
    "month_ordinals",
    "prequential_inverse_mae_weights",
    "prequential_shrinkage_alphas",
    "prevailing_mean_forecast",
    "realised_counts",
]
