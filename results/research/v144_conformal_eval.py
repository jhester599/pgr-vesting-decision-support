"""v144 -- Conformal coverage backtest on the current pooled ensemble frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.conformal import backtest_conformal_coverage
from src.research.v139_utils import DEFAULT_BENCHMARKS, evaluate_ensemble_configuration

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v144_conformal_candidate.json"


def evaluate_conformal_config(
    coverage: float,
    gamma: float,
    benchmarks: list[str] | None = None,
) -> dict[str, float]:
    """Backtest realized conformal coverage on pooled ensemble OOS predictions."""
    coverage_value = float(coverage)
    gamma_value = float(gamma)
    if not 0.50 <= coverage_value < 1.0:
        raise ValueError(f"coverage must be in [0.50, 1.0), got {coverage_value}")
    if not 0.0 < gamma_value < 1.0:
        raise ValueError(f"gamma must be in (0, 1), got {gamma_value}")
    metrics = evaluate_ensemble_configuration(benchmarks=benchmarks or DEFAULT_BENCHMARKS)
    backtest = backtest_conformal_coverage(
        y_hat_oos=metrics["pooled_predictions"],
        y_true_oos=metrics["pooled_realized"],
        coverage=coverage_value,
        method="aci",
        gamma=gamma_value,
        trailing_window=12,
    )
    return {
        "coverage": float(backtest.empirical_coverage),
        "target_coverage": float(backtest.target_coverage),
        "coverage_gap": float(backtest.coverage_gap),
        "trailing_coverage": float(backtest.trailing_empirical_coverage),
    }


def _parse_candidate(params_file: str) -> tuple[float, float]:
    payload = json.loads(Path(params_file).read_text(encoding="utf-8"))
    return float(payload["coverage"]), float(payload["aci_gamma"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conformal coverage settings.")
    parser.add_argument("--coverage", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    coverage = args.coverage
    gamma = args.gamma
    if coverage is None or gamma is None:
        coverage, gamma = _parse_candidate(args.candidate_file)
    metrics = evaluate_conformal_config(float(coverage), float(gamma))
    print(f"coverage={metrics['coverage']:.4f}")
    print(f"target_coverage={metrics['target_coverage']:.4f}")
    print(f"coverage_gap={metrics['coverage_gap']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
