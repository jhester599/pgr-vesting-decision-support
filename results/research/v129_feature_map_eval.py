"""v129 -- Benchmark feature-map evaluation harness.

This is a re-scoped implementation of the 2026-04-13 Target 1 plan. The plan
assumed `v125_portfolio_target_fold_detail.csv` stored per-benchmark logits or
coefficients that could be re-routed without retraining. The current repo does
not contain those artifacts, so this harness evaluates candidate feature maps
by replaying the exact v128 benchmark-specific WFO classifier family on the
stored research feature matrix.

Usage
-----
python results/research/v129_feature_map_eval.py --strategy lean_baseline
python results/research/v129_feature_map_eval.py --strategy v128_map
python results/research/v129_feature_map_eval.py --strategy file:results/research/v129_candidate_map.csv

Outputs
-------
covered_ba=X.XXXX
coverage=X.XXXX

Exit codes
----------
0 -- success
1 -- coverage < 0.20
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.features import V128_BENCHMARK_FEATURE_MAP_PATH
from src.models.calibration import compute_ece
from src.research.v87_utils import feature_set_from_name

from results.research.v128_benchmark_feature_search import (
    CachedEvaluator,
    benchmark_universe,
    build_benchmark_datasets,
    candidate_feature_columns,
    load_v128_inputs,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "research"
V128_COMPARISON_PATH = RESULTS_DIR / "v128_benchmark_feature_search_comparison.csv"
V128_MAP_PATH = PROJECT_ROOT / V128_BENCHMARK_FEATURE_MAP_PATH
V129_CANDIDATE_MAP_PATH = RESULTS_DIR / "v129_candidate_map.csv"

MIN_COVERAGE: float = 0.20
DEFAULT_LOW_THRESH: float = 0.30
DEFAULT_HIGH_THRESH: float = 0.70


@lru_cache(maxsize=1)
def _candidate_universe() -> set[str]:
    """Return the eligible feature universe for candidate-map validation."""
    feature_df, _ = load_v128_inputs()
    return set(candidate_feature_columns(feature_df))


@lru_cache(maxsize=None)
def _load_saved_pooled_metrics(method_name: str) -> dict[str, float]:
    """Load the canonical v128 pooled metrics for known built-in strategies."""
    df = pd.read_csv(V128_COMPARISON_PATH)
    row = df[(df["benchmark"] == "POOLED") & (df["method"] == method_name)].iloc[0]
    return {
        "covered_ba": float(row["balanced_accuracy_covered"]),
        "coverage": float(row["coverage"]),
        "brier": float(row["brier_score"]),
        "ece_10": float(row["ece_10"]),
        "log_loss": float(row["log_loss"]),
    }


def _validate_candidate_map(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a candidate feature-map DataFrame."""
    required = {"benchmark", "feature_set", "feature_list"}
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"candidate map missing required columns: {sorted(missing)}")

    normalized = candidate_df.copy()
    normalized["benchmark"] = normalized["benchmark"].astype(str).str.strip()
    normalized["feature_set"] = normalized["feature_set"].astype(str).str.strip()
    normalized["feature_list"] = normalized["feature_list"].astype(str).str.strip()

    expected_benchmarks = set(benchmark_universe())
    present_benchmarks = set(normalized["benchmark"])
    if present_benchmarks != expected_benchmarks:
        raise ValueError(
            "candidate map benchmarks must exactly match benchmark_universe(): "
            f"expected={sorted(expected_benchmarks)} got={sorted(present_benchmarks)}"
        )

    eligible = _candidate_universe()
    for _, row in normalized.iterrows():
        features = [part.strip() for part in str(row["feature_list"]).split(",") if part.strip()]
        if len(features) > 12:
            raise ValueError(f"{row['benchmark']} has {len(features)} features; max is 12")
        invalid = [feature for feature in features if feature not in eligible]
        if invalid:
            raise ValueError(f"{row['benchmark']} contains ineligible features: {invalid}")

    return normalized.sort_values("benchmark").reset_index(drop=True)


def _load_candidate_map_from_file(path: Path) -> pd.DataFrame:
    """Load and validate a file-based candidate map."""
    return _validate_candidate_map(pd.read_csv(path))


def _candidate_map_from_v128() -> pd.DataFrame:
    """Convert the v128 source-of-truth map into the v129 candidate schema."""
    source = pd.read_csv(V128_MAP_PATH)
    rows: list[dict[str, str]] = []
    for _, row in source.iterrows():
        rows.append(
            {
                "benchmark": str(row["benchmark"]),
                "feature_set": str(row["selected_method"]),
                "feature_list": str(row["selected_features"]).replace("|", ", "),
            }
        )
    return _validate_candidate_map(pd.DataFrame(rows))


def _load_strategy_map(strategy: str) -> pd.DataFrame:
    """Load one feature-map strategy into normalized v129 candidate format."""
    feature_df, _ = load_v128_inputs(benchmarks=benchmark_universe())
    lean_baseline = feature_set_from_name(feature_df, "lean_baseline")
    if strategy == "lean_baseline":
        rows = [
            {
                "benchmark": benchmark,
                "feature_set": "lean_baseline",
                "feature_list": ", ".join(lean_baseline),
            }
            for benchmark in benchmark_universe()
        ]
        return _validate_candidate_map(pd.DataFrame(rows))
    if strategy == "v128_map":
        return _candidate_map_from_v128()
    if strategy.startswith("file:"):
        file_path = PROJECT_ROOT / strategy.removeprefix("file:")
        return _load_candidate_map_from_file(file_path)
    raise ValueError(f"unsupported strategy: {strategy}")


def _compute_metrics_from_predictions(
    pooled_df: pd.DataFrame,
    low_thresh: float,
    high_thresh: float,
) -> dict[str, float]:
    """Compute pooled coverage-aware metrics from benchmark prediction rows."""
    y_true = pooled_df["y_true"].to_numpy(dtype=int)
    y_prob = np.clip(pooled_df["y_prob"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    covered_mask = (y_prob <= low_thresh) | (y_prob >= high_thresh)
    coverage = float(covered_mask.mean()) if len(y_prob) > 0 else 0.0
    if covered_mask.sum() == 0 or len(np.unique(y_true[covered_mask])) < 2:
        covered_ba = 0.5
    else:
        covered_ba = float(
            balanced_accuracy_score(
                y_true[covered_mask],
                (y_prob[covered_mask] >= 0.5).astype(int),
            )
        )
    return {
        "covered_ba": covered_ba,
        "coverage": coverage,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece_10": float(compute_ece(y_prob, y_true, n_bins=10)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def _evaluate_candidate_map(
    candidate_df: pd.DataFrame,
    low_thresh: float,
    high_thresh: float,
) -> dict[str, float]:
    """Evaluate a candidate map by replaying the v128 benchmark classifier path."""
    feature_df, rel_map = load_v128_inputs(benchmarks=benchmark_universe())
    datasets = build_benchmark_datasets(feature_df, rel_map)
    evaluator = CachedEvaluator()

    pooled_frames: list[pd.DataFrame] = []
    for _, row in candidate_df.iterrows():
        benchmark = str(row["benchmark"])
        features = [part.strip() for part in str(row["feature_list"]).split(",") if part.strip()]
        dataset = datasets[benchmark]
        filtered = [feature for feature in features if feature in dataset.x_df.columns]
        _, pred_df = evaluator.evaluate(dataset, filtered, method="candidate_map")
        if pred_df.empty:
            continue
        frame = pred_df.copy()
        frame["benchmark"] = benchmark
        pooled_frames.append(frame)

    if not pooled_frames:
        return {
            "covered_ba": 0.5,
            "coverage": 0.0,
            "brier": float("nan"),
            "ece_10": float("nan"),
            "log_loss": float("nan"),
        }

    pooled = (
        pd.concat(pooled_frames, ignore_index=True)
        .sort_values(["date", "benchmark"])
        .reset_index(drop=True)
    )
    return _compute_metrics_from_predictions(
        pooled,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )


@lru_cache(maxsize=None)
def evaluate_feature_map(
    strategy: str,
    fold_detail_path: str | None = None,
    low_thresh: float = DEFAULT_LOW_THRESH,
    high_thresh: float = DEFAULT_HIGH_THRESH,
    benchmarks: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Evaluate one candidate feature-routing strategy."""
    _ = fold_detail_path  # retained for CLI compatibility with the original plan
    selected_benchmarks = tuple(benchmarks) if benchmarks is not None else None
    if strategy == "lean_baseline" and selected_benchmarks is None:
        return _load_saved_pooled_metrics("lean_baseline")
    if strategy == "v128_map" and selected_benchmarks is None:
        return _load_saved_pooled_metrics("final_feature_map")
    candidate_df = _load_strategy_map(strategy)
    if selected_benchmarks is not None:
        candidate_df = candidate_df[candidate_df["benchmark"].isin(selected_benchmarks)]
    return _evaluate_candidate_map(
        candidate_df,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate benchmark feature-map routing.")
    parser.add_argument("--strategy", required=True, type=str)
    parser.add_argument("--fold-detail-path", type=str, default=None)
    parser.add_argument("--low", type=float, default=DEFAULT_LOW_THRESH)
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH_THRESH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    metrics = evaluate_feature_map(
        strategy=args.strategy,
        fold_detail_path=args.fold_detail_path,
        low_thresh=args.low,
        high_thresh=args.high,
    )
    print(f"covered_ba={metrics['covered_ba']:.4f}")
    print(f"coverage={metrics['coverage']:.4f}")
    if metrics["coverage"] < MIN_COVERAGE:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
