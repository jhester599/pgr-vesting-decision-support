"""v154 -- Firth logistic evaluation for short-history benchmarks (CLS-02)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.research.v37_utils import BENCHMARKS, RESULTS_DIR, RIDGE_FEATURES_12
from src.research.v154_utils import (
    FIRTH_THIN_THRESHOLD,
    compare_logistic_vs_firth,
    load_research_inputs_for_classification,
)

DEFAULT_CANDIDATE_PATH = RESULTS_DIR / "v154_firth_candidate.json"
WIN_THRESHOLD_BA = 0.02


def run_firth_evaluation(
    benchmarks: list[str] | None = None,
    feature_cols: list[str] | None = None,
    candidate_path: Path = DEFAULT_CANDIDATE_PATH,
) -> dict:
    """Run Firth vs standard logistic comparison and write candidate file."""
    bms = list(benchmarks or BENCHMARKS)
    cols = list(feature_cols or RIDGE_FEATURES_12)

    feature_df, rel_map = load_research_inputs_for_classification()
    rows = compare_logistic_vs_firth(feature_df, rel_map, bms, cols)

    winners: list[str] = []
    for row in rows:
        delta = row["delta_ba_covered"]
        if not np.isnan(delta) and delta >= WIN_THRESHOLD_BA:
            winners.append(row["benchmark"])

    result = {
        "benchmarks_tested": bms,
        "firth_thin_threshold": FIRTH_THIN_THRESHOLD,
        "win_threshold_ba": WIN_THRESHOLD_BA,
        "rows": rows,
        "firth_winners": winners,
        "recommendation": (
            "adopt_firth_for_thin_benchmarks" if winners else "no_benefit"
        ),
    }
    candidate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Firth logistic for CLS-02.")
    parser.add_argument(
        "--benchmarks", nargs="+", default=None,
        help="Benchmarks to test (default: all 8)",
    )
    parser.add_argument(
        "--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_firth_evaluation(
        benchmarks=args.benchmarks,
        candidate_path=Path(args.candidate_file),
    )
    for row in result["rows"]:
        bm = row["benchmark"]
        thin = "THIN" if row["is_thin"] else "    "
        std = row["std_ba_covered"]
        firth = row["firth_ba_covered"]
        delta = row["delta_ba_covered"]
        print(
            f"{bm:6s} {thin}  "
            f"std_ba={std:.4f}  firth_ba={firth:.4f}  delta={delta:+.4f}"
        )
    print(f"\nFirth winners (delta >= {result['win_threshold_ba']}): {result['firth_winners']}")
    print(f"Recommendation: {result['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
