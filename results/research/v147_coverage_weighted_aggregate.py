"""v147 -- Coverage-weighted Path A / Path B probability aggregation proxy."""

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

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v147_path_b_multiplier_candidate.txt"


def evaluate_coverage_weighted_aggregate(
    path_b_multiplier: float,
    low: float = 0.15,
    high: float = 0.70,
    fold_detail_path: str | None = None,
) -> dict[str, float]:
    """Evaluate a simple coverage-weighted blend of Path A and tuned Path B."""
    multiplier = float(path_b_multiplier)
    if not 0.25 <= multiplier <= 4.0:
        raise ValueError(f"path_b_multiplier must be in [0.25, 4.0], got {multiplier}")

    temp_max = float(DEFAULT_TEMP_MAX_PATH.read_text(encoding="utf-8").strip())
    warmup = int(DEFAULT_WARMUP_PATH.read_text(encoding="utf-8").strip())
    grid = _build_temperature_grid(DEFAULT_TEMP_MIN, temp_max, DEFAULT_N_TEMPS)
    csv_path = Path(fold_detail_path) if fold_detail_path is not None else FOLD_DETAIL_PATH
    df = pd.read_csv(csv_path)
    df["test_date"] = pd.to_datetime(df["test_date"])
    df = df.sort_values("test_date").reset_index(drop=True)

    y_true = df["y_true"].to_numpy(dtype=int)
    path_a = df["path_a_prob"].to_numpy(dtype=float)
    path_b_raw = df["path_b_prob"].to_numpy(dtype=float)
    path_b, _ = apply_prequential_temperature_scaling(
        path_b_raw,
        y_true,
        temperature_grid=grid,
        warmup=warmup,
    )

    path_a_coverage = float(((path_a < low) | (path_a > high)).mean())
    path_b_coverage = float(((path_b < low) | (path_b > high)).mean())
    weight_a = max(path_a_coverage, 1e-6)
    weight_b = max(path_b_coverage * multiplier, 1e-6)
    aggregate_prob = (weight_a * path_a + weight_b * path_b) / (weight_a + weight_b)
    threshold_metrics = evaluate_thresholds(y_true, aggregate_prob, low=low, high=high)
    return {
        "covered_ba": float(threshold_metrics["covered_ba"]),
        "coverage": float(threshold_metrics["coverage"]),
        "path_a_weight": float(weight_a / (weight_a + weight_b)),
        "path_b_weight": float(weight_b / (weight_a + weight_b)),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coverage-weighted Path A/B aggregation.")
    parser.add_argument("--path-b-multiplier", type=float, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    multiplier = args.path_b_multiplier
    if multiplier is None:
        multiplier = float(Path(args.candidate_file).read_text(encoding="utf-8").strip())
    metrics = evaluate_coverage_weighted_aggregate(float(multiplier))
    print(f"covered_ba={metrics['covered_ba']:.4f}")
    print(f"coverage={metrics['coverage']:.4f}")
    if metrics["coverage"] < MIN_COVERAGE:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
