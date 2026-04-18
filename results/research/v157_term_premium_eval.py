"""v157 -- Term premium 3M differential evaluation (FEAT-03)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.research.v37_utils import BENCHMARKS, RESULTS_DIR, RIDGE_FEATURES_12
from src.research.v154_utils import (
    evaluate_classifier_wfo,
    load_research_inputs_for_classification,
)
from results.research.v155_wti_momentum_eval import build_augmented_feature_cols

DEFAULT_CANDIDATE_PATH = RESULTS_DIR / "v157_term_premium_candidate.json"
SOURCE_FEATURE: str = "term_premium_10y"
DERIVED_FEATURE: str = "term_premium_diff_3m"
WIN_THRESHOLD_BA: float = 0.02


def compute_term_premium_diff(series: pd.Series, periods: int = 3) -> pd.Series:
    """Return n-period change in term premium as a new named series."""
    if periods <= 0:
        raise ValueError(f"periods must be > 0, got {periods}")
    result = series.diff(periods)
    result.name = DERIVED_FEATURE
    return result


def augment_with_term_diff(
    feature_df: pd.DataFrame,
    periods: int = 3,
) -> pd.DataFrame:
    """Add term_premium_diff_3m column to feature_df if source column exists."""
    if SOURCE_FEATURE not in feature_df.columns:
        return feature_df
    df = feature_df.copy()
    df[DERIVED_FEATURE] = compute_term_premium_diff(df[SOURCE_FEATURE], periods=periods)
    return df


def run_term_premium_evaluation(
    benchmarks: list[str] | None = None,
    candidate_path: Path = DEFAULT_CANDIDATE_PATH,
) -> dict:
    """Compare lean_baseline vs lean_baseline+term_premium_diff_3m."""
    bms = list(benchmarks or BENCHMARKS)
    base_cols = list(RIDGE_FEATURES_12)

    feature_df, rel_map = load_research_inputs_for_classification()
    augmented_df = augment_with_term_diff(feature_df)

    source_available = SOURCE_FEATURE in feature_df.columns
    derived_available = DERIVED_FEATURE in augmented_df.columns
    aug_cols = (
        build_augmented_feature_cols(base_cols, extra=[DERIVED_FEATURE])
        if derived_available
        else base_cols
    )

    rows: list[dict] = []
    for bm in bms:
        if bm not in rel_map or rel_map[bm].empty:
            continue
        base_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=base_cols,
            benchmark=bm,
            use_firth=False,
        )
        if not derived_available:
            rows.append({
                "benchmark": bm,
                "skipped": True,
                "reason": "term_premium_10y not in feature matrix",
            })
            continue
        aug_result = evaluate_classifier_wfo(
            feature_df=augmented_df,
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
            "skipped": False,
        })

    winners = [
        r["benchmark"] for r in rows
        if not r.get("skipped")
        and not np.isnan(r.get("delta_ba_covered", float("nan")))
        and r.get("delta_ba_covered", -999.0) >= WIN_THRESHOLD_BA
    ]
    result = {
        "benchmarks_tested": bms,
        "source_feature": SOURCE_FEATURE,
        "derived_feature": DERIVED_FEATURE,
        "source_available": source_available,
        "derived_available": derived_available,
        "win_threshold_ba": WIN_THRESHOLD_BA,
        "rows": rows,
        "term_premium_winners": winners,
        "recommendation": "adopt_term_premium_diff" if winners else "no_benefit",
    }
    candidate_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate term premium 3M diff (FEAT-03).")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--candidate-file", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_term_premium_evaluation(
        benchmarks=args.benchmarks,
        candidate_path=Path(args.candidate_file),
    )
    print(f"Source feature available: {result['source_available']}")
    print(f"Derived feature created:  {result['derived_available']}")
    for row in result["rows"]:
        if row.get("skipped"):
            print(f"{row['benchmark']:6s}  SKIPPED: {row.get('reason', 'unknown')}")
            continue
        bm = row["benchmark"]
        base = row["base_ba_covered"]
        aug = row["aug_ba_covered"]
        delta = row["delta_ba_covered"]
        print(f"{bm:6s}  base_ba={base:.4f}  +tp_diff={aug:.4f}  delta={delta:+.4f}")
    print(f"\nTerm premium winners: {result['term_premium_winners']}")
    print(f"Recommendation: {result['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
