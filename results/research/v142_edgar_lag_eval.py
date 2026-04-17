"""v142 -- EDGAR filing-lag evaluation on the current ensemble frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.v139_utils import DEFAULT_BENCHMARKS, evaluate_ensemble_configuration

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v142_edgar_lag_candidate.txt"


def evaluate_edgar_lag(
    lag: int,
    benchmarks: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate one EDGAR lag value on the current research frame."""
    lag_value = int(lag)
    if not 0 <= lag_value <= 3:
        raise ValueError(f"lag must be in [0, 3], got {lag_value}")
    return evaluate_ensemble_configuration(
        benchmarks=benchmarks or DEFAULT_BENCHMARKS,
        config_overrides={"EDGAR_FILING_LAG_MONTHS": lag_value},
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EDGAR lag values.")
    parser.add_argument("--lag", type=int, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    lag = args.lag
    if lag is None:
        lag = int(Path(args.candidate_file).read_text(encoding="utf-8").strip())
    metrics = evaluate_edgar_lag(int(lag))
    print(f"pooled_oos_r2={metrics['pooled_oos_r2']:.4f}")
    print(f"pooled_ic={metrics['pooled_ic']:.4f}")
    print(f"pooled_hit_rate={metrics['pooled_hit_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
