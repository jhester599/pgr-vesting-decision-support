"""v155 -- WTI 3M momentum evaluation for DBC/VDE classification (FEAT-02)."""

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

DEFAULT_CANDIDATE_PATH = RESULTS_DIR / "v155_wti_candidate.json"
TARGET_BENCHMARKS: list[str] = ["DBC", "VDE"]
WTI_FEATURE: str = "wti_return_3m"
WIN_THRESHOLD_BA: float = 0.04


def build_augmented_feature_cols(
    base_cols: list[str],
    extra: list[str],
) -> list[str]:
    """Return base_cols with extra features appended, no duplicates."""
    seen = set(base_cols)
    result = list(base_cols)
    for col in extra:
        if col not in seen:
            result.append(col)
            seen.add(col)
    return result


def run_wti_evaluation(
    benchmarks: list[str] | None = None,
    candidate_path: Path = DEFAULT_CANDIDATE_PATH,
) -> dict:
    """Compare lean_baseline vs lean_baseline+WTI for targeted benchmarks."""
    bms = list(benchmarks or TARGET_BENCHMARKS)
    base_cols = list(RIDGE_FEATURES_12)
    augmented_cols = build_augmented_feature_cols(base_cols, extra=[WTI_FEATURE])

    feature_df, rel_map = load_research_inputs_for_classification()

    rows: list[dict] = []
    for bm in bms:
        if bm not in rel_map or rel_map[bm].empty:
            continue
        aug_cols = build_augmented_feature_cols(
            base_cols,
            extra=[WTI_FEATURE] if WTI_FEATURE in feature_df.columns else [],
        )
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
            "base_ba_covered": base_result.get("ba_covered", float("nan")),
            "aug_ba_covered": aug_result.get("ba_covered", float("nan")),
            "delta_ba_covered": delta,
            "base_coverage": base_result.get("coverage", float("nan")),
            "aug_coverage": aug_result.get("coverage", float("nan")),
            "wti_feature_available": WTI_FEATURE in feature_df.columns,
        })

    winners = [
        r["benchmark"] for r in rows
        if not np.isnan(r["delta_ba_covered"]) and r["delta_ba_covered"] >= WIN_THRESHOLD_BA
    ]
    result = {
        "benchmarks_tested": bms,
        "wti_feature": WTI_FEATURE,
        "win_threshold_ba": WIN_THRESHOLD_BA,
        "rows": rows,
        "wti_winners": winners,
        "recommendation": "adopt_wti_for_targets" if winners else "no_benefit",
    }
    candidate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WTI momentum feature (FEAT-02).")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_wti_evaluation(
        benchmarks=args.benchmarks,
        candidate_path=Path(args.candidate_file),
    )
    for row in result["rows"]:
        bm = row["benchmark"]
        base = row["base_ba_covered"]
        aug = row["aug_ba_covered"]
        delta = row["delta_ba_covered"]
        avail = row["wti_feature_available"]
        print(
            f"{bm:6s}  base_ba={base:.4f}  +wti_ba={aug:.4f}  delta={delta:+.4f}  "
            f"wti_in_matrix={avail}"
        )
    print(f"\nWTI winners (delta >= {result['win_threshold_ba']}): {result['wti_winners']}")
    print(f"Recommendation: {result['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
