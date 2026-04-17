"""v148 -- Positive-class weight replay proxy on preserved Path B probabilities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v131_threshold_sweep_eval import FOLD_DETAIL_PATH, MIN_COVERAGE, evaluate_thresholds
from results.research.v135_temp_param_search import DEFAULT_TEMP_MIN, DEFAULT_N_TEMPS, _build_temperature_grid, apply_prequential_temperature_scaling
from results.research.v146_threshold_sweep import DEFAULT_TEMP_MAX_PATH, DEFAULT_WARMUP_PATH

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v148_class_weight_candidate.txt"


def _apply_positive_class_weight(probabilities: np.ndarray, positive_weight: float) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    odds = probs / (1.0 - probs)
    adjusted_odds = odds * positive_weight
    return adjusted_odds / (1.0 + adjusted_odds)


def evaluate_class_weight_proxy(
    positive_weight: float,
    low: float = 0.15,
    high: float = 0.70,
    fold_detail_path: str | None = None,
) -> dict[str, float]:
    """Evaluate a class-weight odds-adjustment proxy on the preserved Path B frame."""
    weight_value = float(positive_weight)
    if not 0.25 <= weight_value <= 4.0:
        raise ValueError(f"positive_weight must be in [0.25, 4.0], got {weight_value}")

    temp_max = float(DEFAULT_TEMP_MAX_PATH.read_text(encoding="utf-8").strip())
    warmup = int(DEFAULT_WARMUP_PATH.read_text(encoding="utf-8").strip())
    grid = _build_temperature_grid(DEFAULT_TEMP_MIN, temp_max, DEFAULT_N_TEMPS)
    csv_path = Path(fold_detail_path) if fold_detail_path is not None else FOLD_DETAIL_PATH
    df = pd.read_csv(csv_path)
    df["test_date"] = pd.to_datetime(df["test_date"])
    df = df.sort_values("test_date").reset_index(drop=True)

    y_true = df["y_true"].to_numpy(dtype=int)
    raw_probs = df["path_b_prob"].to_numpy(dtype=float)
    weighted_probs = _apply_positive_class_weight(raw_probs, weight_value)
    calibrated_probs, _ = apply_prequential_temperature_scaling(
        weighted_probs,
        y_true,
        temperature_grid=grid,
        warmup=warmup,
    )
    threshold_metrics = evaluate_thresholds(y_true, calibrated_probs, low=low, high=high)
    return {
        "covered_ba": float(threshold_metrics["covered_ba"]),
        "coverage": float(threshold_metrics["coverage"]),
        "mean_probability": float(np.mean(calibrated_probs)),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate positive-class weight proxy.")
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    weight = args.weight
    if weight is None:
        weight = float(Path(args.candidate_file).read_text(encoding="utf-8").strip())
    metrics = evaluate_class_weight_proxy(float(weight))
    print(f"covered_ba={metrics['covered_ba']:.4f}")
    print(f"coverage={metrics['coverage']:.4f}")
    if metrics["coverage"] < MIN_COVERAGE:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
