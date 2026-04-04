"""v9.4 benchmark-universe reduction analysis."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from src.database import db_client
from src.research.benchmark_sets import BENCHMARK_FAMILIES
from scripts.benchmark_suite import run_benchmark_suite


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")


def build_benchmark_scorecard(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Score each benchmark by ensemble and average model quality."""
    model_df = detail_df[detail_df["item_type"] == "model"]
    ensemble_df = detail_df[detail_df["item_type"] == "ensemble"]

    model_agg = (
        model_df.groupby("benchmark", dropna=False)[["ic", "hit_rate", "oos_r2", "mae"]]
        .mean()
        .rename(
            columns={
                "ic": "model_mean_ic",
                "hit_rate": "model_mean_hit_rate",
                "oos_r2": "model_mean_oos_r2",
                "mae": "model_mean_mae",
            }
        )
    )
    ensemble_agg = (
        ensemble_df.groupby("benchmark", dropna=False)[["ic", "hit_rate", "oos_r2", "mae"]]
        .mean()
        .rename(
            columns={
                "ic": "ensemble_ic",
                "hit_rate": "ensemble_hit_rate",
                "oos_r2": "ensemble_oos_r2",
                "mae": "ensemble_mae",
            }
        )
    )
    scorecard = model_agg.join(ensemble_agg, how="outer").reset_index()
    scorecard["family"] = scorecard["benchmark"].map(BENCHMARK_FAMILIES).fillna("other")

    rank_cols = [
        "model_mean_ic",
        "model_mean_hit_rate",
        "model_mean_oos_r2",
        "ensemble_ic",
        "ensemble_hit_rate",
        "ensemble_oos_r2",
    ]
    for col in rank_cols:
        scorecard[f"{col}_rank"] = scorecard[col].rank(method="average", ascending=False)
    scorecard["composite_score"] = scorecard[[f"{col}_rank" for col in rank_cols]].mean(axis=1)
    return scorecard.sort_values(by="composite_score", ascending=True)


def build_candidate_universes(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Create candidate reduced benchmark universes from the scorecard."""
    candidates: list[dict[str, str]] = []

    top8 = scorecard.head(8)["benchmark"].tolist()
    candidates.append(
        {
            "universe_name": "top8_composite",
            "benchmarks": ",".join(top8),
            "n_benchmarks": str(len(top8)),
        }
    )

    diversified: list[str] = []
    for family, family_df in scorecard.groupby("family", dropna=False):
        diversified.extend(family_df.head(1)["benchmark"].tolist())
    remaining = [
        benchmark
        for benchmark in scorecard["benchmark"].tolist()
        if benchmark not in diversified
    ]
    for benchmark in remaining:
        family = BENCHMARK_FAMILIES.get(benchmark, "other")
        if sum(BENCHMARK_FAMILIES.get(existing, "other") == family for existing in diversified) >= 2:
            continue
        diversified.append(benchmark)
        if len(diversified) >= 8:
            break
    candidates.append(
        {
            "universe_name": "diversified_top8",
            "benchmarks": ",".join(diversified[:8]),
            "n_benchmarks": str(len(diversified[:8])),
        }
    )

    sectors = scorecard[scorecard["family"] == "sector"].head(2)["benchmark"].tolist()
    broad = scorecard[scorecard["family"] == "broad_equity"].head(2)["benchmark"].tolist()
    fixed_income = scorecard[scorecard["family"] == "fixed_income"].head(2)["benchmark"].tolist()
    real_assets = scorecard[scorecard["family"] == "real_asset"].head(1)["benchmark"].tolist()
    balanced = broad + sectors + fixed_income + real_assets
    candidates.append(
        {
            "universe_name": "balanced_core7",
            "benchmarks": ",".join(balanced),
            "n_benchmarks": str(len(balanced)),
        }
    )

    return pd.DataFrame(candidates)


def run_benchmark_reduction(
    conn: Any,
    detail_csv_path: str,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build scorecards, candidate universes, and retest them."""
    detail_df = pd.read_csv(detail_csv_path)
    scorecard = build_benchmark_scorecard(detail_df)
    candidates = build_candidate_universes(scorecard)

    retest_rows: list[pd.DataFrame] = []
    for candidate in candidates.itertuples(index=False):
        benchmarks = [value.strip() for value in candidate.benchmarks.split(",") if value.strip()]
        _, summary_df = run_benchmark_suite(
            conn=conn,
            benchmarks=benchmarks,
            horizons=[6],
            model_types=list(config.ENSEMBLE_MODELS),
            baseline_strategies=["historical_mean", "last_value", "zero"],
            output_dir=os.path.join(output_dir, "candidate_runs"),
        )
        summary_df = summary_df.copy()
        summary_df["universe_name"] = candidate.universe_name
        summary_df["benchmarks"] = candidate.benchmarks
        retest_rows.append(summary_df)

    candidate_summary = pd.concat(retest_rows, ignore_index=True) if retest_rows else pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    scorecard_path = os.path.join(output_dir, f"benchmark_reduction_scorecard_{stamp}.csv")
    candidate_path = os.path.join(output_dir, f"benchmark_reduction_candidates_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"benchmark_reduction_candidate_summary_{stamp}.csv")
    scorecard.to_csv(scorecard_path, index=False)
    candidates.to_csv(candidate_path, index=False)
    candidate_summary.to_csv(summary_path, index=False)
    print(f"Wrote benchmark scorecard to {scorecard_path}")
    print(f"Wrote candidate universes to {candidate_path}")
    print(f"Wrote candidate summary to {summary_path}")
    return scorecard, candidates, candidate_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 benchmark-universe reduction analysis.")
    parser.add_argument(
        "--detail-csv",
        default=os.path.join(DEFAULT_OUTPUT_DIR, "benchmark_suite_detail_20260403.csv"),
        help="Path to benchmark suite detail CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    conn = db_client.get_connection(config.DB_PATH)
    run_benchmark_reduction(
        conn=conn,
        detail_csv_path=args.detail_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
