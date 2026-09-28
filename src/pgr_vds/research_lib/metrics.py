"""Honest endpoint metrics with benchmark-preserving monthly inference."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def honest_r2(y: ArrayLike, p: ArrayLike, naive: ArrayLike) -> float:
    """Compute 1-SSE/SSE_naive using one finite mask for all three arrays.

    The caller supplies already-honest benchmark-specific prevailing means.
    This allows panels to pool errors without pooling different past means.
    """
    actual = np.asarray(y, dtype=float)
    predicted = np.asarray(p, dtype=float)
    comparator = np.asarray(naive, dtype=float)
    if (
        actual.ndim != 1
        or actual.shape != predicted.shape
        or actual.shape != comparator.shape
    ):
        raise ValueError(
            "Outcome, prediction, and naive must be aligned 1D arrays."
        )
    keep = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & np.isfinite(comparator)
    )
    if keep.sum() < 2:
        return float("nan")
    denominator = float(np.sum((actual[keep] - comparator[keep]) ** 2))
    if denominator <= 0.0:
        return float("nan")
    numerator = float(np.sum((actual[keep] - predicted[keep]) ** 2))
    return float(1.0 - numerator / denominator)


def prevailing_mean(
    origins: Iterable[object], history: pd.DataFrame,
) -> NDArray[np.float64]:
    """Mean finite earlier targets whose explicit availability has arrived."""
    required = {"date", "y_true", "available"}
    if not required.issubset(history.columns):
        raise ValueError(
            "History requires date, y_true, and available columns."
        )
    dates = pd.to_datetime(history["date"])
    arrivals = pd.to_datetime(history["available"])
    values = pd.to_numeric(history["y_true"], errors="coerce").to_numpy(
        dtype=float
    )
    results = []
    for origin in pd.DatetimeIndex(origins):
        keep = (dates < origin) & (arrivals <= origin) & np.isfinite(values)
        mean = float(values[keep].mean()) if keep.any() else float("nan")
        results.append(mean)
    return np.asarray(results, dtype=float)


def _spearman(y: NDArray[np.float64], p: NDArray[np.float64]) -> float:
    """Rank correlation without row-pooled significance calculations."""
    if len(y) < 3 or np.ptp(y) == 0.0 or np.ptp(p) == 0.0:
        return float("nan")
    ranked_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    ranked_p = pd.Series(p).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(ranked_y, ranked_p)[0, 1])


def _equal_weight_ic(
    y: NDArray[np.float64], p: NDArray[np.float64], benchmarks: NDArray[Any],
) -> float:
    """Average each benchmark's IC equally, omitting undefined correlations."""
    correlations = [
        _spearman(y[benchmarks == name], p[benchmarks == name])
        for name in np.unique(benchmarks)
    ]
    valid = np.asarray(correlations, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float(valid.mean()) if len(valid) else float("nan")


def _interval(values: list[float]) -> list[float]:
    """Return a central 95% percentile moving-date-block interval."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan")] * 2
    return np.quantile(finite, [0.025, 0.975]).tolist()


def _centered_p(
    observed: float, bootstrap: list[float], two_sided: bool = False,
) -> float:
    """Test a zero improvement using centered date-block bootstrap draws."""
    values = np.asarray(bootstrap, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(observed):
        return float("nan")
    if observed <= 0.0 and not two_sided:
        return 1.0
    centered = values - observed
    exceed = (
        np.abs(centered) >= abs(observed)
        if two_sided else centered >= observed
    )
    return float((1 + exceed.sum()) / (1 + len(values)))


def panel_summary(
    frame: pd.DataFrame,
    h: int,
    replicates: int = 2000,
    seed: int = 20260926,
) -> dict[str, Any]:
    """Summarize development forecasts with all benchmarks kept by date.

    The preregistered primary test is a one-sided paired squared-error
    improvement against the honest naive. Every bootstrap block contains h
    consecutive observed monthly dates. This function never reads inputs or
    selects a model. Warmup rows are excluded through their missing naive or
    base prediction. At least two full date blocks are required for inference.
    """
    required = {
        "date", "benchmark", "y_true", "y_hat", "naive", "base_prediction",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            "Panel requires date, benchmark, y_true, y_hat, naive, and "
            "base_prediction."
        )
    if h not in (6, 12) or replicates < 1:
        raise ValueError(
            "Use a registered horizon and positive bootstrap replicates."
        )
    work = frame[list(required)].copy()
    work["date"] = pd.to_datetime(work["date"])
    if work.duplicated(["date", "benchmark"]).any():
        raise ValueError(
            "A panel must have one forecast per date and benchmark."
        )
    numeric = ["y_true", "y_hat", "naive", "base_prediction"]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(work[numeric]).all(axis=1) & work["date"].notna()
    work = work.loc[finite]
    work = work.sort_values(["date", "benchmark"], kind="stable")
    y = work["y_true"].to_numpy(dtype=float)
    p = work["y_hat"].to_numpy(dtype=float)
    naive = work["naive"].to_numpy(dtype=float)
    base = work["base_prediction"].to_numpy(dtype=float)
    benchmarks = work["benchmark"].to_numpy()
    hits = ((y > 0) == (p > 0)).astype(float)
    base_hits = ((y > 0) == (base > 0)).astype(float)
    loss_gain = (y - naive) ** 2 - (y - p) ** 2
    dates = pd.DatetimeIndex(work["date"].unique())
    row_dates = work["date"].to_numpy()
    date_rows = [
        np.flatnonzero(row_dates == date.to_datetime64()) for date in dates
    ]
    gain_by_date = np.asarray([loss_gain[rows].mean() for rows in date_rows])
    skill_by_date = np.asarray([
        (hits[rows] - base_hits[rows]).mean() for rows in date_rows
    ])
    observed_gain = float(gain_by_date.mean()) if len(dates) else float("nan")
    observed_skill = (
        float(skill_by_date.mean()) if len(dates) else float("nan")
    )
    result: dict[str, Any] = {
        "n_rows": len(work),
        "n_dates": len(dates),
        "n_benchmarks": len(np.unique(benchmarks)),
        "oos_r2": honest_r2(y, p, naive),
        "equal_weight_ic": _equal_weight_ic(y, p, benchmarks),
        "panel_ic": _spearman(y, p),
        "hit_rate": float(hits.mean()) if len(y) else float("nan"),
        "base_hit_rate": float(base_hits.mean()) if len(y) else float("nan"),
        "base_rate": float((y > 0).mean()) if len(y) else float("nan"),
        "directional_skill": observed_skill,
        "mean_squared_error_improvement": observed_gain,
        "block_length": h, "bootstrap_replicates": replicates, "seed": seed,
        "inference_supported": len(dates) >= 2 * h,
    }
    draws: dict[str, list[float]] = {name: [] for name in (
        "oos_r2", "equal_weight_ic", "panel_ic", "loss_gain", "skill",
    )}
    if result["inference_supported"]:
        rng = np.random.default_rng(seed)
        blocks = int(np.ceil(len(dates) / h))
        for _ in range(replicates):
            starts = rng.integers(0, len(dates) - h + 1, size=blocks)
            sampled_dates = np.concatenate([
                np.arange(start, start + h) for start in starts
            ])[:len(dates)]
            sampled_rows = np.concatenate([
                date_rows[index] for index in sampled_dates
            ])
            sample_y, sample_p = y[sampled_rows], p[sampled_rows]
            draws["oos_r2"].append(
                honest_r2(sample_y, sample_p, naive[sampled_rows])
            )
            draws["equal_weight_ic"].append(_equal_weight_ic(
                sample_y, sample_p, benchmarks[sampled_rows]
            ))
            draws["panel_ic"].append(_spearman(sample_y, sample_p))
            draws["loss_gain"].append(
                float(gain_by_date[sampled_dates].mean())
            )
            draws["skill"].append(float(skill_by_date[sampled_dates].mean()))
    for name in ("oos_r2", "equal_weight_ic", "panel_ic"):
        result[name + "_ci"] = _interval(draws[name])
    result["primary_p"] = _centered_p(observed_gain, draws["loss_gain"])
    result["directional_skill_p"] = _centered_p(observed_skill, draws["skill"])
    result["equal_weight_ic_p"] = _centered_p(
        result["equal_weight_ic"], draws["equal_weight_ic"], True
    )
    result["panel_ic_p"] = _centered_p(
        result["panel_ic"], draws["panel_ic"], True
    )
    result["inference_method"] = (
        "paired moving-date-block bootstrap; all benchmarks retained"
    )
    return result


def holm38(pvalues: Sequence[float]) -> list[float]:
    """Holm-adjust primary p-values with unused campaign slots set to one."""
    values = np.asarray(pvalues, dtype=float)
    if (
        len(values) > 38
        or np.any(~np.isfinite(values))
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("Supply at most 38 finite probabilities in [0,1].")
    padded = np.concatenate([values, np.ones(38 - len(values))])
    order = np.argsort(padded, kind="stable")
    adjusted = np.minimum(
        1.0, np.maximum.accumulate(padded[order] * np.arange(38, 0, -1))
    )
    restored = np.empty(38, dtype=float)
    restored[order] = adjusted
    return restored[:len(values)].tolist()
