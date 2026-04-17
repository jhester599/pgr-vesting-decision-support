"""v143 -- Correlation-pruned feature-set evaluation on the current frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.features import MODEL_FEATURE_OVERRIDES
from src.research.v139_utils import DEFAULT_BENCHMARKS, evaluate_ensemble_configuration, prune_feature_overrides

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v143_corr_prune_candidate.txt"


def evaluate_corr_pruning(
    rho_threshold: float,
    benchmarks: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate greedy within-model correlation pruning."""
    baseline = evaluate_ensemble_configuration(benchmarks=benchmarks or DEFAULT_BENCHMARKS)
    pruned_overrides = prune_feature_overrides(
        baseline["feature_df"],
        MODEL_FEATURE_OVERRIDES,
        rho_threshold=float(rho_threshold),
    )
    return evaluate_ensemble_configuration(
        benchmarks=benchmarks or DEFAULT_BENCHMARKS,
        model_feature_overrides=pruned_overrides,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate correlation-pruned feature sets.")
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rho = args.rho
    if rho is None:
        rho = float(Path(args.candidate_file).read_text(encoding="utf-8").strip())
    metrics = evaluate_corr_pruning(float(rho))
    print(f"pooled_oos_r2={metrics['pooled_oos_r2']:.4f}")
    print(f"pooled_ic={metrics['pooled_ic']:.4f}")
    print(f"pooled_hit_rate={metrics['pooled_hit_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
