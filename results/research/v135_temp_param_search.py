"""v135 -- Path B temperature-parameter search harness.

Tunes the prequential temperature-scaling configuration used by the Path B
classifier while keeping the asymmetric abstention thresholds fixed at the
v131 candidate pair (low=0.15, high=0.70).

Usage
-----
python results/research/v135_temp_param_search.py --temp-min 0.5 --temp-max 3.0 --warmup 24

Outputs
-------
covered_ba=X.XXXX
coverage=X.XXXX

Exit codes
----------
0 -- success
1 -- coverage < 0.20 (insufficient predictions to be decision-useful)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.calibration import compute_ece

from results.research.v131_threshold_sweep_eval import (
    FOLD_DETAIL_PATH,
    MIN_COVERAGE,
    evaluate_thresholds,
)

DEFAULT_TEMP_MIN: float = 0.50
DEFAULT_TEMP_MAX: float = 3.0
DEFAULT_N_TEMPS: int = 51
DEFAULT_WARMUP: int = 24
DEFAULT_LOW_THRESH: float = 0.15
DEFAULT_HIGH_THRESH: float = 0.70
MIN_WARMUP: int = 6
MAX_WARMUP: int = 48


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
    """Apply temperature scaling to a single probability."""
    temp = max(float(temperature), 1e-6)
    value = float(_sigmoid(np.array([_logit(np.array([prob]))[0] / temp]))[0])
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _build_temperature_grid(
    temp_min: float,
    temp_max: float,
    n_temps: int,
) -> np.ndarray:
    """Build the candidate temperature grid."""
    if temp_min <= 0.0:
        raise ValueError("temp_min must be positive")
    if temp_min > temp_max:
        raise ValueError("temp_min must be <= temp_max")
    if n_temps < 10:
        raise ValueError("n_temps must be >= 10")
    if temp_max / temp_min > 10.0:
        return np.logspace(np.log10(temp_min), np.log10(temp_max), n_temps)
    return np.linspace(temp_min, temp_max, n_temps)


def _fit_temperature_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    temperature_grid: np.ndarray,
) -> float:
    """Fit temperature on historical OOS points via grid-search log-loss."""
    y_hist = np.asarray(y_true, dtype=int)
    p_hist = _clip_probs(y_prob)
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in temperature_grid:
        scaled = _sigmoid(_logit(p_hist) / float(temperature))
        loss = float(log_loss(y_hist, scaled, labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


def apply_prequential_temperature_scaling(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    temperature_grid: np.ndarray,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply prequential temperature scaling using a custom grid and warmup."""
    if warmup < MIN_WARMUP:
        raise ValueError(f"warmup must be >= {MIN_WARMUP}")
    if warmup > MAX_WARMUP:
        raise ValueError(f"warmup must be <= {MAX_WARMUP}")

    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    calibrated = _clip_probs(probs).copy()
    temperatures = np.ones(len(probs), dtype=float)

    for idx in range(len(probs)):
        if idx < warmup or len(np.unique(labels[:idx])) < 2:
            continue
        temperature = _fit_temperature_grid(
            labels[:idx],
            probs[:idx],
            temperature_grid,
        )
        temperatures[idx] = temperature
        calibrated[idx] = _apply_temperature(float(probs[idx]), temperature)

    return calibrated, temperatures


def evaluate_temperature_config(
    temp_min: float,
    temp_max: float,
    n_temps: int = DEFAULT_N_TEMPS,
    warmup: int = DEFAULT_WARMUP,
    low_thresh: float = DEFAULT_LOW_THRESH,
    high_thresh: float = DEFAULT_HIGH_THRESH,
    fold_detail_path: str | None = None,
) -> dict[str, float]:
    """Evaluate a Path B temperature-search configuration."""
    grid = _build_temperature_grid(temp_min=temp_min, temp_max=temp_max, n_temps=n_temps)

    csv_path = Path(fold_detail_path) if fold_detail_path is not None else FOLD_DETAIL_PATH
    df = pd.read_csv(csv_path)
    df["test_date"] = pd.to_datetime(df["test_date"])
    df = df.sort_values("test_date").reset_index(drop=True)

    y_true = df["y_true"].to_numpy(dtype=int)
    raw_probs = df["path_b_prob"].to_numpy(dtype=float)
    calibrated_probs, temperatures = apply_prequential_temperature_scaling(
        raw_probs,
        y_true,
        temperature_grid=grid,
        warmup=warmup,
    )

    threshold_metrics = evaluate_thresholds(
        y_true,
        calibrated_probs,
        low=low_thresh,
        high=high_thresh,
    )

    covered_mask = (calibrated_probs < low_thresh) | (calibrated_probs > high_thresh)
    covered_probs = calibrated_probs[covered_mask]
    covered_true = y_true[covered_mask]

    if len(covered_probs) > 0:
        covered_brier = float(brier_score_loss(covered_true, covered_probs))
        covered_log_loss = float(log_loss(covered_true, covered_probs, labels=[0, 1]))
        covered_ece = float(compute_ece(covered_probs, covered_true, n_bins=10))
    else:
        covered_brier = float("nan")
        covered_log_loss = float("nan")
        covered_ece = float("nan")

    return {
        "covered_ba": float(threshold_metrics["covered_ba"]),
        "coverage": float(threshold_metrics["coverage"]),
        "brier": covered_brier,
        "log_loss": covered_log_loss,
        "ece": covered_ece,
        "mean_temperature": float(np.mean(temperatures)),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Path B temperature-scaling search parameters."
    )
    parser.add_argument("--temp-min", type=float, default=DEFAULT_TEMP_MIN)
    parser.add_argument("--temp-max", type=float, default=DEFAULT_TEMP_MAX)
    parser.add_argument("--n-temps", type=int, default=DEFAULT_N_TEMPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--low", type=float, default=DEFAULT_LOW_THRESH)
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH_THRESH)
    parser.add_argument("--fold-detail-path", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns shell exit code."""
    args = _parse_args(argv)
    metrics = evaluate_temperature_config(
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        n_temps=args.n_temps,
        warmup=args.warmup,
        low_thresh=args.low,
        high_thresh=args.high,
        fold_detail_path=args.fold_detail_path,
    )
    print(f"covered_ba={metrics['covered_ba']:.4f}")
    print(f"coverage={metrics['coverage']:.4f}")
    if metrics["coverage"] < MIN_COVERAGE:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
