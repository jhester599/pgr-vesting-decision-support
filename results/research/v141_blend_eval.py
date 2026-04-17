"""v141 -- Fixed Ridge-vs-GBT ensemble blend evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.research.v139_utils import DEFAULT_BENCHMARKS, evaluate_ensemble_configuration, reconstruct_two_model_blend_predictions

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v141_blend_weight_candidate.txt"


def evaluate_blend_weight(
    ridge_weight: float,
    benchmarks: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a fixed Ridge-vs-GBT blend weight on the current frame."""
    weight_value = float(ridge_weight)
    if not 0.0 <= weight_value <= 1.0:
        raise ValueError(f"ridge_weight must be in [0, 1], got {weight_value}")
    return evaluate_ensemble_configuration(
        benchmarks=benchmarks or DEFAULT_BENCHMARKS,
        prediction_builder=lambda ens_result: reconstruct_two_model_blend_predictions(
            ens_result,
            ridge_weight=weight_value,
            shrinkage_alpha=config.ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA,
        ),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed Ridge-vs-GBT blend weights.")
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    weight = args.weight
    if weight is None:
        weight = float(Path(args.candidate_file).read_text(encoding="utf-8").strip())
    metrics = evaluate_blend_weight(float(weight))
    print(f"pooled_oos_r2={metrics['pooled_oos_r2']:.4f}")
    print(f"pooled_ic={metrics['pooled_ic']:.4f}")
    print(f"pooled_hit_rate={metrics['pooled_hit_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
