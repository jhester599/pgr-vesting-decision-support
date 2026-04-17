"""v146 -- Threshold sweep on top of the current v135 temperature baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from results.research.v131_threshold_sweep_eval import FOLD_DETAIL_PATH, MIN_COVERAGE, evaluate_thresholds
from results.research.v135_temp_param_search import DEFAULT_TEMP_MIN, DEFAULT_N_TEMPS, _build_temperature_grid, apply_prequential_temperature_scaling

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v146_threshold_candidate.json"
DEFAULT_TEMP_MAX_PATH = PROJECT_ROOT / "results" / "research" / "v135_temp_max_candidate.txt"
DEFAULT_WARMUP_PATH = PROJECT_ROOT / "results" / "research" / "v135_warmup_candidate.txt"


def evaluate_threshold_candidate(
    low: float,
    high: float,
    temp_max: float | None = None,
    warmup: int | None = None,
    fold_detail_path: str | None = None,
) -> dict[str, float]:
    """Evaluate one abstention-band pair on the tuned v135 temperature path."""
    low_value = float(low)
    high_value = float(high)
    if not 0.0 <= low_value <= 0.50:
        raise ValueError(f"low must be in [0.0, 0.50], got {low_value}")
    if not 0.50 <= high_value <= 1.0:
        raise ValueError(f"high must be in [0.50, 1.0], got {high_value}")
    if low_value >= high_value:
        raise ValueError("low must be < high")

    temp_max_value = float(
        temp_max
        if temp_max is not None
        else DEFAULT_TEMP_MAX_PATH.read_text(encoding="utf-8").strip()
    )
    warmup_value = int(
        warmup
        if warmup is not None
        else DEFAULT_WARMUP_PATH.read_text(encoding="utf-8").strip()
    )
    grid = _build_temperature_grid(DEFAULT_TEMP_MIN, temp_max_value, DEFAULT_N_TEMPS)
    csv_path = Path(fold_detail_path) if fold_detail_path is not None else FOLD_DETAIL_PATH
    df = pd.read_csv(csv_path)
    df["test_date"] = pd.to_datetime(df["test_date"])
    df = df.sort_values("test_date").reset_index(drop=True)

    y_true = df["y_true"].to_numpy(dtype=int)
    raw_probs = df["path_b_prob"].to_numpy(dtype=float)
    calibrated_probs, _ = apply_prequential_temperature_scaling(
        raw_probs,
        y_true,
        temperature_grid=grid,
        warmup=warmup_value,
    )
    threshold_metrics = evaluate_thresholds(y_true, calibrated_probs, low=low_value, high=high_value)
    covered_mask = (calibrated_probs < low_value) | (calibrated_probs > high_value)
    covered_probs = calibrated_probs[covered_mask]
    covered_true = y_true[covered_mask]
    covered_brier = (
        float(brier_score_loss(covered_true, covered_probs)) if len(covered_probs) > 0 else float("nan")
    )
    return {
        "covered_ba": float(threshold_metrics["covered_ba"]),
        "coverage": float(threshold_metrics["coverage"]),
        "brier": covered_brier,
    }


def _parse_candidate(candidate_file: str) -> tuple[float, float]:
    payload = json.loads(Path(candidate_file).read_text(encoding="utf-8"))
    return float(payload["low"]), float(payload["high"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v146 threshold candidates.")
    parser.add_argument("--low", type=float, default=None)
    parser.add_argument("--high", type=float, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    low = args.low
    high = args.high
    if low is None or high is None:
        low, high = _parse_candidate(args.candidate_file)
    metrics = evaluate_threshold_candidate(float(low), float(high))
    print(f"covered_ba={metrics['covered_ba']:.4f}")
    print(f"coverage={metrics['coverage']:.4f}")
    if metrics["coverage"] < MIN_COVERAGE:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
