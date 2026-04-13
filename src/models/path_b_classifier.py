"""Path B composite portfolio-target classifier with prequential temperature scaling.

v131 -- Wires the temperature-scaled Path B composite classifier into production
shadow output. Extracted and hardened from v125/v127/v130 research scripts.

Public API
----------
build_composite_return_series  -- weighted composite relative-return series
fit_path_b_classifier          -- train Path B logistic on composite target
apply_prequential_temperature_scaling -- prequential temperature calibration
PATH_B_THRESHOLD               -- composite return threshold (0.03)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config.features import INVESTABLE_CLASSIFIER_BASE_WEIGHTS
from src.processing.feature_engineering import truncate_relative_target_for_asof
from src.processing.multi_total_return import load_relative_return_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATH_B_THRESHOLD: float = 0.03
_HORIZON_MONTHS: int = 6
_GRID_TEMPERATURES: np.ndarray = np.concatenate(
    [
        np.linspace(0.50, 0.95, 10),
        np.linspace(1.0, 3.0, 41),
    ]
)


# ---------------------------------------------------------------------------
# Temperature scaling utilities (verbatim from v130 for reproducibility)
# ---------------------------------------------------------------------------


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    """Clip probabilities away from 0 and 1 for stable log transforms."""
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(probs: np.ndarray) -> np.ndarray:
    """Return the logit transform of probabilities."""
    clipped = _clip_probs(probs)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the logistic sigmoid of arbitrary values."""
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _apply_temperature(prob: float, temperature: float) -> float:
    """Apply temperature scaling to one binary probability."""
    temp = max(float(temperature), 1e-6)
    value = float(_sigmoid(np.array([_logit(np.array([prob]))[0] / temp]))[0])
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _fit_temperature_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Fit temperature on historical OOS points via log-loss grid search.

    Parameters
    ----------
    y_true:
        Integer binary labels (0/1).
    y_prob:
        Raw probabilities aligned with y_true.

    Returns
    -------
    float
        Best temperature from a fixed candidate grid.
    """
    y_hist = np.asarray(y_true, dtype=int)
    p_hist = _clip_probs(y_prob)
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in _GRID_TEMPERATURES:
        scaled = _sigmoid(_logit(p_hist) / float(temperature))
        # manual log-loss to avoid sklearn dependency here
        eps = 1e-6
        scaled = np.clip(scaled, eps, 1.0 - eps)
        loss = -float(np.mean(
            y_hist * np.log(scaled) + (1 - y_hist) * np.log(1.0 - scaled)
        ))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


# ---------------------------------------------------------------------------
# Public: prequential temperature scaling
# ---------------------------------------------------------------------------


def apply_prequential_temperature_scaling(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    warmup: int = 24,
) -> np.ndarray:
    """Apply prequential temperature scaling to an array of OOS probabilities.

    For each index ``t >= warmup``, fits a temperature scalar on all prior
    observations ``probs[:t]`` / ``labels[:t]`` and applies it to ``probs[t]``.
    Observations before the warmup window are returned unchanged.

    Parameters
    ----------
    probs:
        Array of raw OOS probabilities in chronological order.
    labels:
        Integer binary labels (0/1) aligned with probs.
    warmup:
        Minimum number of prior observations required before calibration is
        applied. Observations at indices ``< warmup`` are returned unchanged.

    Returns
    -------
    np.ndarray
        Calibrated probabilities with the same shape as ``probs``.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    calibrated = _clip_probs(probs).copy()

    for idx in range(len(probs)):
        if idx < warmup or len(np.unique(labels[:idx])) < 2:
            continue
        temperature = _fit_temperature_grid(labels[:idx], probs[:idx])
        calibrated[idx] = _apply_temperature(float(probs[idx]), temperature)

    return np.clip(calibrated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public: composite return series builder
# ---------------------------------------------------------------------------


def build_composite_return_series(
    conn: object,
    weights: dict[str, float] | None = None,
    as_of: date | None = None,
    horizon_months: int = _HORIZON_MONTHS,
) -> pd.Series:
    """Build a weighted-composite relative-return series across benchmarks.

    Loads per-benchmark relative return series, aligns them on a common date
    index (dropping any row with any NaN), and returns the weighted sum.

    Parameters
    ----------
    conn:
        Open database connection forwarded to ``load_relative_return_matrix``.
    weights:
        Mapping of ticker -> portfolio weight.  Defaults to
        ``INVESTABLE_CLASSIFIER_BASE_WEIGHTS``.
    as_of:
        If provided, truncates each series so that the horizon does not extend
        beyond the as-of date (prevents look-ahead).
    horizon_months:
        Forward horizon in months used when loading relative return matrices.

    Returns
    -------
    pd.Series
        Named ``"composite_relative_return"``, indexed by month-end date.

    Raises
    ------
    ValueError
        If no relative return series can be loaded for any benchmark.
    """
    if weights is None:
        weights = INVESTABLE_CLASSIFIER_BASE_WEIGHTS

    rel_series_list: list[pd.Series] = []
    weight_list: list[float] = []

    for ticker, weight in weights.items():
        try:
            rel = load_relative_return_matrix(conn, ticker, horizon_months)
            if rel is not None and not rel.empty:
                if as_of is not None:
                    rel = truncate_relative_target_for_asof(
                        rel,
                        as_of=pd.Timestamp(as_of),
                        horizon_months=horizon_months,
                    )
                rel_series_list.append(rel.rename(ticker))
                weight_list.append(weight)
        except Exception:
            continue

    if not rel_series_list:
        raise ValueError(
            "No relative return series could be loaded for composite target."
        )

    total_w = sum(weight_list)
    norm_weights = [w / total_w for w in weight_list]

    combined = pd.concat(rel_series_list, axis=1).dropna(how="any")
    composite: pd.Series = sum(  # type: ignore[assignment]
        combined[s.name] * w
        for s, w in zip(rel_series_list, norm_weights)
        if s.name in combined.columns
    )
    composite.name = "composite_relative_return"
    return composite


# ---------------------------------------------------------------------------
# Public: Path B classifier
# ---------------------------------------------------------------------------


def fit_path_b_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
) -> float | None:
    """Train the Path B logistic classifier on all available history and return
    the current-month (last-row) probability.

    Uses ``LogisticRegression(C=0.5, class_weight='balanced', solver='lbfgs')``
    consistent with the Path B research specification.

    Parameters
    ----------
    X:
        Feature DataFrame indexed by month-end date.
    y:
        Binary target Series (1 = actionable sell, 0 = non-actionable).
    feature_cols:
        Ordered list of feature column names to use from ``X``.

    Returns
    -------
    float or None
        Probability of the actionable-sell class for the last row of ``X``,
        or ``None`` if training is not possible (< 30 observations or fewer
        than 2 unique classes).
    """
    usable = [c for c in feature_cols if c in X.columns]
    if not usable:
        return None

    X_sub = X[usable].copy()
    aligned_index = X_sub.index.intersection(y.index)
    X_sub = X_sub.loc[aligned_index]
    y_sub = y.loc[aligned_index]

    if len(X_sub) < 30:
        return None
    if len(np.unique(y_sub.to_numpy(dtype=int))) < 2:
        return None

    x_vals = X_sub.to_numpy(dtype=float).copy()  # .copy() ensures writable (pandas 3.0+)
    medians = np.nanmedian(x_vals, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for col_idx in range(x_vals.shape[1]):
        x_vals[np.isnan(x_vals[:, col_idx]), col_idx] = medians[col_idx]

    try:
        model = LogisticRegression(
            C=0.5,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(x_vals, y_sub.to_numpy(dtype=int))
        last_row = X.iloc[[-1]][usable].to_numpy(dtype=float).copy()  # writable
        for col_idx in range(last_row.shape[1]):
            last_row[np.isnan(last_row[:, col_idx]), col_idx] = medians[col_idx]
        return float(model.predict_proba(last_row)[:, 1][0])
    except Exception:
        return None
