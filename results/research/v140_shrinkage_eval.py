"""v140 -- Standalone shrinkage evaluation on the current ensemble frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.v139_utils import DEFAULT_BENCHMARKS, evaluate_ensemble_configuration

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v140_shrinkage_candidate.txt"


def evaluate_shrinkage(
    shrinkage: float,
    benchmarks: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate one ensemble shrinkage alpha on the current research frame."""
    shrinkage_value = float(shrinkage)
    if not 0.0 <= shrinkage_value <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage_value}")
    return evaluate_ensemble_configuration(
        benchmarks=benchmarks or DEFAULT_BENCHMARKS,
        config_overrides={"ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA": shrinkage_value},
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ensemble shrinkage.")
    parser.add_argument("--shrinkage", type=float, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    shrinkage = args.shrinkage
    if shrinkage is None:
        shrinkage = float(Path(args.candidate_file).read_text(encoding="utf-8").strip())
    metrics = evaluate_shrinkage(float(shrinkage))
    print(f"pooled_oos_r2={metrics['pooled_oos_r2']:.4f}")
    print(f"pooled_ic={metrics['pooled_ic']:.4f}")
    print(f"pooled_hit_rate={metrics['pooled_hit_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
