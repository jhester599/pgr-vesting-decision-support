"""
Multi-benchmark Walk-Forward Optimization runner for v2 relative return models.

Runs one independent LassoCV/RidgeCV WFO model per ETF benchmark, using the
PGR-minus-ETF relative return as the target variable.  Each model is entirely
separate — no regularization or feature weights are shared across benchmarks.

Primary entry points:
  - ``run_all_benchmarks``: trains 20 WFO models (one per ETF column in the
    relative return matrix) and returns a dict keyed by ETF ticker.
  - ``get_current_signals``: given a completed set of WFO results, refits each
    model on the most recent data and generates a prediction for today.

Computational note:
  Each fold takes < 2 s on ~180 rows × 15 features.  20 models × ~8 folds ≈
  160 fits ≈ 60 s total — well within acceptable limits for a batch run.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.models.wfo_engine import WFOResult, run_wfo, predict_current
from src.processing.feature_engineering import get_X_y_relative


# ---------------------------------------------------------------------------
# Multi-benchmark training
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    X: pd.DataFrame,
    relative_return_matrix: pd.DataFrame,
    model_type: Literal["lasso", "ridge"] = "lasso",
    target_horizon_months: int = 6,
) -> dict[str, WFOResult]:
    """
    Train one WFO model per ETF benchmark.

    For each ETF column in ``relative_return_matrix``, aligns the feature
    matrix to the relative return series, drops NaN targets, and calls
    ``run_wfo()`` with the correct embargo for the target horizon.

    Args:
        X:                      Feature DataFrame from
                                ``build_feature_matrix_from_db()`` (or
                                ``build_feature_matrix()``).  Monthly
                                DatetimeIndex.
        relative_return_matrix: DataFrame with one column per ETF benchmark,
                                values = PGR return − ETF return.  Typically
                                the output of
                                ``build_relative_return_targets()`` or loaded
                                via ``load_relative_return_matrix()`` for each
                                ETF.  Index must overlap with ``X``.
        model_type:             ``"lasso"`` (default) or ``"ridge"``.
        target_horizon_months:  Forward return horizon in months (6 or 12).
                                Sets the WFO embargo gap.

    Returns:
        Dict mapping ETF ticker → WFOResult.  Benchmarks skipped due to
        insufficient overlapping data are absent from the dict (no empty
        WFOResult placeholders).

    Raises:
        ValueError: If ``relative_return_matrix`` is empty.
    """
    if relative_return_matrix.empty:
        raise ValueError("relative_return_matrix is empty — no benchmarks to train.")

    results: dict[str, WFOResult] = {}

    for etf in relative_return_matrix.columns:
        rel_series = relative_return_matrix[etf].rename(f"{etf}_{target_horizon_months}m")

        try:
            X_aligned, y_aligned = get_X_y_relative(
                X, rel_series, drop_na_target=True
            )
        except ValueError:
            # No overlapping data for this benchmark — skip silently.
            continue

        if len(y_aligned) == 0:
            continue

        try:
            wfo_result = run_wfo(
                X_aligned,
                y_aligned,
                model_type=model_type,
                target_horizon_months=target_horizon_months,
                benchmark=etf,
            )
        except ValueError:
            # Dataset too small for this benchmark — skip.
            continue

        results[etf] = wfo_result

    return results


# ---------------------------------------------------------------------------
# Current-period signal generation
# ---------------------------------------------------------------------------

_SIGNAL_OUTPERFORM = "OUTPERFORM"
_SIGNAL_UNDERPERFORM = "UNDERPERFORM"
_SIGNAL_NEUTRAL = "NEUTRAL"

_IC_THRESHOLD = 0.05          # minimum IC for a signal to be non-NEUTRAL
_RETURN_THRESHOLD = 0.01      # minimum |predicted return| for directional signal


def get_current_signals(
    X_full: pd.DataFrame,
    relative_return_matrix: pd.DataFrame,
    wfo_results: dict[str, WFOResult],
    X_current: pd.DataFrame,
    model_type: Literal["lasso", "ridge"] = "lasso",
    train_window_months: int | None = None,
) -> pd.DataFrame:
    """
    Generate live predictions for the current observation across all benchmarks.

    For each ETF in ``wfo_results``, calls ``predict_current()`` to refit the
    model on the most recent training window and predict on ``X_current``.

    Signal classification:
      - ``OUTPERFORM``:   predicted relative return > threshold AND IC ≥ 0.05
      - ``UNDERPERFORM``: predicted relative return < -threshold AND IC ≥ 0.05
      - ``NEUTRAL``:      |predicted return| < threshold OR IC < 0.05

    Args:
        X_full:                 Complete feature DataFrame (same as used in
                                ``run_all_benchmarks``).
        relative_return_matrix: DataFrame of relative returns (one column per
                                ETF).  Used to build the aligned y_full for
                                each benchmark.
        wfo_results:            Dict from ``run_all_benchmarks()`` mapping
                                ETF ticker → WFOResult.
        X_current:              Single-row DataFrame with current features.
        model_type:             Must match the model_type used in training.
        train_window_months:    Refit window size.  Defaults to
                                ``config.WFO_TRAIN_WINDOW_MONTHS``.

    Returns:
        DataFrame with one row per benchmark and columns:
          - ``benchmark``: ETF ticker
          - ``predicted_relative_return``: live model prediction
          - ``ic``: out-of-sample IC from WFO
          - ``hit_rate``: directional hit rate from WFO
          - ``signal``: "OUTPERFORM", "UNDERPERFORM", or "NEUTRAL"
          - ``top_feature``: name of the highest-weight feature (absolute coef)
    """
    rows = []

    for etf, wfo_result in wfo_results.items():
        if etf not in relative_return_matrix.columns:
            continue

        horizon = wfo_result.target_horizon
        rel_series = relative_return_matrix[etf].rename(f"{etf}_{horizon}m")

        try:
            _, y_full = get_X_y_relative(X_full, rel_series, drop_na_target=True)
        except ValueError:
            continue

        try:
            pred = predict_current(
                X_full=X_full,
                y_full=y_full,
                X_current=X_current,
                wfo_result=wfo_result,
                model_type=model_type,
                train_window_months=train_window_months,
            )
        except Exception:  # noqa: BLE001
            continue

        predicted = pred["predicted_return"]
        ic = pred["ic"]
        hit_rate = pred["hit_rate"]
        top_features = pred["top_features"]
        top_feature = top_features[0][0] if top_features else ""

        if ic < _IC_THRESHOLD or abs(predicted) < _RETURN_THRESHOLD:
            signal = _SIGNAL_NEUTRAL
        elif predicted > 0:
            signal = _SIGNAL_OUTPERFORM
        else:
            signal = _SIGNAL_UNDERPERFORM

        rows.append({
            "benchmark":                  etf,
            "predicted_relative_return":  predicted,
            "ic":                         ic,
            "hit_rate":                   hit_rate,
            "signal":                     signal,
            "top_feature":                top_feature,
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "benchmark", "predicted_relative_return", "ic",
                "hit_rate", "signal", "top_feature",
            ]
        )

    df = pd.DataFrame(rows).set_index("benchmark")
    return df
