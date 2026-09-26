"""Reusable forecast diagnostics for ensemble evaluation and reporting.

Review 2026-09-25 (F04, F13):

- The naive benchmark behind OOS R^2 and Clark-West is the prevailing mean of
  the targets realised by each forecast date, per benchmark, including the
  training history (``src.models.prequential.prevailing_mean_forecast``).
- Pooled (multi-benchmark) significance is Driscoll-Kraay by date
  (``src.models.robust_inference``). A single benchmark has one row per date,
  where it reduces to Newey-West.
- Directional accuracy is reported against the base rate, with the
  Pesaran-Timmermann test that the gate uses, over the rows with a call
  (a zero forecast makes none).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.prequential import prevailing_mean_forecast
from src.models.robust_inference import (
    ClarkWestResult,
    clark_west_test,
    oos_r2_against,
    pesaran_timmermann_test,
    rank_ic_test,
)

IC_P_VALUE_METHOD = "driscoll-kraay (clustered by date)"


def expanding_mean_benchmark(realized: pd.Series) -> pd.Series:
    """Legacy one-step expanding mean (row 0 uses its own value).

    Kept for research callers of ``compute_clark_west_result`` without a
    horizon. Production uses ``prevailing_mean_forecast``, which counts only
    targets realised before each forecast date.
    """
    clean = realized.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float, name="benchmark_forecast")

    values = clean.to_numpy(dtype=float)
    benchmark = np.empty(len(values), dtype=float)
    benchmark[0] = values[0]
    for idx in range(1, len(values)):
        benchmark[idx] = float(values[:idx].mean())
    return pd.Series(benchmark, index=clean.index, name="benchmark_forecast")


def compute_clark_west_result(
    predicted: pd.Series,
    realized: pd.Series,
    lags: int,
    benchmark_forecast: pd.Series | None = None,
    horizon_months: int | None = None,
) -> ClarkWestResult:
    """Run a one-sided Clark-West test against a naive benchmark forecast.

    With ``benchmark_forecast`` given, it is used as is (aligned with
    ``realized``). Otherwise, with ``horizon_months`` the benchmark is the
    prevailing mean of the targets realised before each date; without it, the
    legacy one-step expanding mean.
    """
    aligned = pd.concat([predicted, realized], axis=1).dropna()
    if len(aligned) < 4:
        return ClarkWestResult(
            n_obs=int(len(aligned)),
            t_stat=float("nan"),
            p_value=float("nan"),
            mean_adjusted_differential=float("nan"),
        )

    model_pred = aligned.iloc[:, 0].astype(float)
    y_true = aligned.iloc[:, 1].astype(float)
    if benchmark_forecast is not None:
        benchmark_pred = pd.Series(benchmark_forecast).reindex(aligned.index).astype(float)
    elif horizon_months is not None:
        benchmark_pred = prevailing_mean_forecast(
            aligned.index, realized.dropna(), horizon_months
        )
    else:
        benchmark_pred = expanding_mean_benchmark(y_true)
    dates = aligned.index if isinstance(aligned.index, pd.DatetimeIndex) else None
    return clark_west_test(model_pred, y_true, benchmark_pred, dates, lags=max(1, lags))


def _series_dates(index: pd.Index) -> pd.Index | None:
    return index if isinstance(index, pd.DatetimeIndex) else None


def summarize_prediction_diagnostics(
    predicted: pd.Series,
    realized: pd.Series,
    target_horizon_months: int = 6,
    benchmark_forecast: pd.Series | None = None,
    target_history: pd.Series | None = None,
) -> dict[str, float | int]:
    """Honest diagnostics for one forecast series (one row per date).

    OOS R^2 and Clark-West use ``benchmark_forecast`` when given, otherwise
    the prevailing mean of ``target_history`` (default: ``realized``) realised
    by each date. IC significance is Newey-West over dates (lag h-1).
    """
    aligned = pd.concat([predicted, realized], axis=1).dropna()
    if aligned.empty:
        return _empty_summary()

    y_hat = aligned.iloc[:, 0].astype(float)
    y_true = aligned.iloc[:, 1].astype(float)
    lags = max(1, target_horizon_months - 1)
    if benchmark_forecast is not None:
        naive = pd.Series(benchmark_forecast).reindex(aligned.index).astype(float)
    else:
        history = realized if target_history is None else target_history
        naive = prevailing_mean_forecast(aligned.index, history, target_horizon_months)
    dates = _series_dates(aligned.index)
    return _summarize_arrays(
        y_hat.to_numpy(dtype=float),
        y_true.to_numpy(dtype=float),
        naive.to_numpy(dtype=float),
        dates,
        lags,
    )


def _empty_summary() -> dict[str, float | int]:
    nan = float("nan")
    return {
        "n_obs": 0,
        "oos_r2": nan,
        "nw_ic": nan,
        "nw_p_value": nan,
        "hit_rate": nan,
        "n_calls": 0,
        "base_rate": nan,
        "constant_rule_hit_rate": nan,
        "hit_rate_excess": nan,
        "pt_stat": nan,
        "pt_p_value": nan,
        "cw_t_stat": nan,
        "cw_p_value": nan,
        "cw_mean_adjusted_differential": nan,
    }


def _summarize_arrays(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    naive: np.ndarray,
    dates: pd.Index | np.ndarray | None,
    lags: int,
    score: np.ndarray | None = None,
) -> dict[str, float | int]:
    # The rank IC measures ordering, so it uses the ensemble score before the
    # (time-varying) shrinkage when one is given; magnitude-based metrics use
    # the issued forecast.
    ic, ic_p = rank_ic_test(y_hat if score is None else score, y_true, dates, lags)
    direction = pesaran_timmermann_test(y_hat, y_true, dates, lags)
    cw = clark_west_test(y_hat, y_true, naive, dates, lags)
    return {
        "n_obs": int(len(y_true)),
        "oos_r2": float(oos_r2_against(y_hat, y_true, naive)),
        "nw_ic": float(ic),
        "nw_p_value": float(ic_p),
        "hit_rate": float(direction.hit_rate),
        "n_calls": int(direction.n_calls),
        "base_rate": float(direction.base_rate),
        "constant_rule_hit_rate": float(direction.constant_rule_hit_rate),
        "hit_rate_excess": float(direction.excess_over_base_rate),
        "pt_stat": float(direction.pt_stat),
        "pt_p_value": float(direction.pt_p_value),
        "cw_t_stat": float(cw.t_stat),
        "cw_p_value": float(cw.p_value),
        "cw_mean_adjusted_differential": float(cw.mean_adjusted_differential),
    }


def summarize_panel_diagnostics(
    panel: pd.DataFrame,
    target_horizon_months: int = 6,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Pooled and per-benchmark diagnostics from a prequential panel.

    ``panel`` is ``build_prequential_panel`` output (columns ``benchmark``,
    ``date``, ``y_true``, ``y_hat``, ``naive`` and, optionally, ``z``). Pooled
    R^2 sums each benchmark's squared errors against its own prevailing mean;
    pooled IC, hit-rate and Clark-West significance are Driscoll-Kraay by date.

    The rank IC is computed on ``z``, the prequential ensemble score before
    shrinkage, when present. The shrinkage alpha changes over time, and
    rescaling forecasts date by date changes their pooled ranks without
    changing the model's ordering (v38 relied on IC being invariant to the
    scale). OOS R^2, Clark-West and the directional calls use ``y_hat``.

    Returns ``(pooled, per_benchmark)``; ``pooled`` also carries ``n_dates``.
    """
    lags = max(1, target_horizon_months - 1)
    if panel is None or panel.empty:
        pooled = _empty_summary()
        pooled["n_dates"] = 0
        return pooled, pd.DataFrame(columns=["benchmark", *pooled.keys()])

    rows: list[dict[str, float | int | str]] = []
    for benchmark, part in panel.groupby("benchmark", sort=False):
        part = part.sort_values("date", kind="mergesort")
        summary = _summarize_arrays(
            part["y_hat"].to_numpy(dtype=float),
            part["y_true"].to_numpy(dtype=float),
            part["naive"].to_numpy(dtype=float),
            pd.DatetimeIndex(part["date"]),
            lags,
            score=part["z"].to_numpy(dtype=float) if "z" in part.columns else None,
        )
        rows.append({"benchmark": str(benchmark), **summary})

    pooled = _summarize_arrays(
        panel["y_hat"].to_numpy(dtype=float),
        panel["y_true"].to_numpy(dtype=float),
        panel["naive"].to_numpy(dtype=float),
        pd.DatetimeIndex(panel["date"]),
        lags,
        score=panel["z"].to_numpy(dtype=float) if "z" in panel.columns else None,
    )
    pooled["n_dates"] = int(pd.Index(panel["date"]).nunique())
    return pooled, pd.DataFrame(rows)


__all__ = [
    "ClarkWestResult",
    "IC_P_VALUE_METHOD",
    "compute_clark_west_result",
    "expanding_mean_benchmark",
    "summarize_panel_diagnostics",
    "summarize_prediction_diagnostics",
]
