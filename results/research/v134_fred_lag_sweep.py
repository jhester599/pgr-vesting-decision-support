"""v134 -- FRED publication lag sweep against the production ensemble path.

This is a live-state implementation of Target 4 from the 2026-04-13
autoresearch plan. The plan's success criteria were written against the v38
promotion snapshot, so this harness first reuses the same pre-holdout research
frame and the current production ensemble plumbing to establish a reproducible
baseline under today's repo state.

Usage:
    python results/research/v134_fred_lag_sweep.py
    python results/research/v134_fred_lag_sweep.py --lag-overrides "{\"GS10\": 0}"
    python results/research/v134_fred_lag_sweep.py --params-file results/research/v134_lag_candidate.json

Outputs:
    pooled_oos_r2=X.XXXX
    pooled_ic=X.XXXX
    pooled_hit_rate=X.XXXX
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import features as config_features
from config.features import MODEL_FEATURE_OVERRIDES, PRIMARY_FORECAST_UNIVERSE
from src.models.evaluation import reconstruct_ensemble_oos_predictions
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.research.v37_utils import (
    compute_metrics,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
)

DEFAULT_CANDIDATE_PATH = (
    PROJECT_ROOT / "results" / "research" / "v134_lag_candidate.json"
)
DEFAULT_BENCHMARKS = list(PRIMARY_FORECAST_UNIVERSE)


def _validate_lag_overrides(lag_overrides: dict[str, int]) -> dict[str, int]:
    """Validate the caller-supplied lag overrides."""
    validated: dict[str, int] = {}
    known_series = set(config.FRED_SERIES_LAGS)
    for series_id, lag in lag_overrides.items():
        if series_id not in known_series:
            raise KeyError(f"unknown FRED series: {series_id}")
        if not isinstance(lag, int):
            raise ValueError(f"lag for {series_id} must be an int, got {lag!r}")
        if lag < 0 or lag > 3:
            raise ValueError(f"lag for {series_id} must be in [0, 3], got {lag}")
        validated[str(series_id)] = int(lag)
    return validated


@contextmanager
def _temporary_fred_lags(lag_overrides: dict[str, int]) -> Iterator[None]:
    """Temporarily patch both config re-export surfaces used by the builder."""
    original_root = dict(config.FRED_SERIES_LAGS)
    original_features = dict(config_features.FRED_SERIES_LAGS)
    patched = dict(original_root)
    patched.update(lag_overrides)
    config.FRED_SERIES_LAGS = patched
    config_features.FRED_SERIES_LAGS = dict(patched)
    try:
        yield
    finally:
        config.FRED_SERIES_LAGS = original_root
        config_features.FRED_SERIES_LAGS = original_features


def _load_relative_matrix(
    conn: sqlite3.Connection,
    benchmarks: list[str],
) -> pd.DataFrame:
    """Load the pre-holdout relative-return matrix for the requested benchmarks."""
    rel_map: dict[str, pd.Series] = {}
    for benchmark in benchmarks:
        rel_series = load_relative_series(conn, benchmark, horizon=6)
        if not rel_series.empty:
            rel_map[benchmark] = rel_series
    if not rel_map:
        raise RuntimeError("No relative return series available for the requested benchmarks.")
    return pd.DataFrame(rel_map)


def run_lag_sweep(
    lag_overrides: dict[str, int],
    benchmarks: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate one lag configuration on the production ensemble research frame."""
    validated = _validate_lag_overrides(lag_overrides)
    selected_benchmarks = list(benchmarks or DEFAULT_BENCHMARKS)

    conn = get_connection()
    try:
        with _temporary_fred_lags(validated):
            feature_df = load_feature_matrix(conn)
        relative_return_matrix = _load_relative_matrix(conn, selected_benchmarks)
        ensemble_results = run_ensemble_benchmarks(
            feature_df,
            relative_return_matrix,
            target_horizon_months=6,
            model_feature_overrides=MODEL_FEATURE_OVERRIDES,
        )

        rows: list[dict[str, object]] = []
        for benchmark in selected_benchmarks:
            ens_result = ensemble_results.get(benchmark)
            if ens_result is None:
                continue
            y_hat, y_true = reconstruct_ensemble_oos_predictions(
                ens_result,
                shrinkage_alpha=config.ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA,
            )
            if y_hat.empty or y_true.empty:
                continue
            metrics = compute_metrics(
                y_true.to_numpy(dtype=float),
                y_hat.to_numpy(dtype=float),
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    **metrics,
                    "_y_true": y_true.to_numpy(dtype=float),
                    "_y_hat": y_hat.to_numpy(dtype=float),
                }
            )

        if not rows:
            raise RuntimeError("Lag sweep produced no benchmark rows.")

        pooled = pool_metrics(rows)
        return {
            "pooled_oos_r2": float(pooled["r2"]),
            "pooled_ic": float(pooled["ic"]),
            "pooled_hit_rate": float(pooled["hit_rate"]),
            "n_benchmarks": float(len(rows)),
        }
    finally:
        conn.close()


def _parse_lag_overrides(raw: str | None, params_file: str | None) -> dict[str, int]:
    """Parse the lag override payload from CLI inputs."""
    if params_file:
        payload = json.loads(Path(params_file).read_text(encoding="utf-8"))
    elif raw:
        payload = json.loads(raw)
    else:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("lag overrides payload must decode to a JSON object.")
    return {str(key): int(value) for key, value in payload.items()}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate FRED lag overrides.")
    parser.add_argument("--lag-overrides", type=str, default=None)
    parser.add_argument("--params-file", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point."""
    args = _parse_args(argv)
    overrides = _parse_lag_overrides(args.lag_overrides, args.params_file)
    metrics = run_lag_sweep(overrides)
    print(f"pooled_oos_r2={metrics['pooled_oos_r2']:.4f}")
    print(f"pooled_ic={metrics['pooled_ic']:.4f}")
    print(f"pooled_hit_rate={metrics['pooled_hit_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
