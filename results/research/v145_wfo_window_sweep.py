"""v145 -- WFO train/test window sweep on the current ensemble frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.v139_utils import DEFAULT_BENCHMARKS, evaluate_ensemble_configuration

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "results" / "research" / "v145_wfo_candidate.json"


def evaluate_wfo_windows(
    train_months: int,
    test_months: int,
    benchmarks: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a bounded WFO train/test window pair."""
    train_value = int(train_months)
    test_value = int(test_months)
    if not 36 <= train_value <= 72:
        raise ValueError(f"train_months must be in [36, 72], got {train_value}")
    if not 3 <= test_value <= 12:
        raise ValueError(f"test_months must be in [3, 12], got {test_value}")
    if train_value <= test_value:
        raise ValueError("train_months must be greater than test_months.")
    return evaluate_ensemble_configuration(
        benchmarks=benchmarks or DEFAULT_BENCHMARKS,
        config_overrides={
            "WFO_TRAIN_WINDOW_MONTHS": train_value,
            "WFO_TEST_WINDOW_MONTHS": test_value,
        },
    )


def _parse_candidate(candidate_file: str) -> tuple[int, int]:
    payload = json.loads(Path(candidate_file).read_text(encoding="utf-8"))
    return int(payload["train"]), int(payload["test"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WFO train/test windows.")
    parser.add_argument("--train", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    train_months = args.train
    test_months = args.test
    if train_months is None or test_months is None:
        train_months, test_months = _parse_candidate(args.candidate_file)
    metrics = evaluate_wfo_windows(int(train_months), int(test_months))
    print(f"pooled_oos_r2={metrics['pooled_oos_r2']:.4f}")
    print(f"pooled_ic={metrics['pooled_ic']:.4f}")
    print(f"pooled_hit_rate={metrics['pooled_hit_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
