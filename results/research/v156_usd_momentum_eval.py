"""v156 -- USD index momentum evaluation for BND/VXUS/VWO (FEAT-01)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.research.v37_utils import RESULTS_DIR, RIDGE_FEATURES_12
from src.research.v154_utils import (
    evaluate_classifier_wfo,
    load_research_inputs_for_classification,
)
from results.research.v155_wti_momentum_eval import build_augmented_feature_cols

DEFAULT_CANDIDATE_PATH = RESULTS_DIR / "v156_usd_candidate.json"
TARGET_BENCHMARKS: list[str] = ["BND", "VXUS", "VWO"]
USD_FEATURES: list[str] = ["usd_broad_return_3m", "usd_momentum_6m"]
WIN_THRESHOLD_BA: float = 0.03


def run_usd_evaluation(
    benchmarks: list[str] | None = None,
    candidate_path: Path = DEFAULT_CANDIDATE_PATH,
) -> dict:
    """Compare lean_baseline vs lean_baseline+USD_features for currency-sensitive ETFs."""
    bms = list(benchmarks or TARGET_BENCHMARKS)
    base_cols = list(RIDGE_FEATURES_12)

    feature_df, rel_map = load_research_inputs_for_classification()
    available_usd = [f for f in USD_FEATURES if f in feature_df.columns]

    rows: list[dict] = []
    for bm in bms:
        if bm not in rel_map or rel_map[bm].empty:
            continue
        aug_cols = build_augmented_feature_cols(base_cols, extra=available_usd)
        base_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=base_cols,
            benchmark=bm,
            use_firth=False,
        )
        aug_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=aug_cols,
            benchmark=bm,
            use_firth=False,
        )
        delta = float("nan")
        if not base_result.get("skipped") and not aug_result.get("skipped"):
            base_ba = base_result.get("ba_covered", float("nan"))
            aug_ba = aug_result.get("ba_covered", float("nan"))
            if not (np.isnan(base_ba) or np.isnan(aug_ba)):
                delta = aug_ba - base_ba
        rows.append({
            "benchmark": bm,
            "usd_features_added": available_usd,
            "base_ba_covered": base_result.get("ba_covered", float("nan")),
            "aug_ba_covered": aug_result.get("ba_covered", float("nan")),
            "delta_ba_covered": delta,
            "base_coverage": base_result.get("coverage", float("nan")),
            "aug_coverage": aug_result.get("coverage", float("nan")),
        })

    winners = [
        r["benchmark"] for r in rows
        if not np.isnan(r["delta_ba_covered"]) and r["delta_ba_covered"] >= WIN_THRESHOLD_BA
    ]
    recommendation = "adopt_usd_for_targets" if winners else (
        "features_missing_check_pipeline" if not available_usd else "no_benefit"
    )
    result = {
        "benchmarks_tested": bms,
        "usd_features_requested": USD_FEATURES,
        "usd_features_available": available_usd,
        "win_threshold_ba": WIN_THRESHOLD_BA,
        "rows": rows,
        "usd_winners": winners,
        "recommendation": recommendation,
    }
    candidate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate USD momentum features (FEAT-01).")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_usd_evaluation(
        benchmarks=args.benchmarks,
        candidate_path=Path(args.candidate_file),
    )
    print(f"USD features available: {result['usd_features_available']}")
    for row in result["rows"]:
        bm = row["benchmark"]
        base = row["base_ba_covered"]
        aug = row["aug_ba_covered"]
        delta = row["delta_ba_covered"]
        print(f"{bm:6s}  base_ba={base:.4f}  +usd_ba={aug:.4f}  delta={delta:+.4f}")
    print(f"\nUSD winners (delta >= {result['win_threshold_ba']}): {result['usd_winners']}")
    print(f"Recommendation: {result['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
