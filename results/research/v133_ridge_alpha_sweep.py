"""v133 -- Ridge alpha-grid sweep on the production research frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from config.features import MODEL_FEATURE_OVERRIDES, PRIMARY_FORECAST_UNIVERSE
from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    compute_metrics,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    pool_metrics,
)
from src.research.v37_utils import custom_wfo
from src.models.regularized_models import build_ridge_pipeline

DEFAULT_BENCHMARKS = list(PRIMARY_FORECAST_UNIVERSE)


def _validate_alpha_inputs(
    alpha_min: float,
    alpha_max: float,
    n_alpha: int,
) -> None:
    """Validate sweep bounds before running expensive WFO work."""
    if alpha_min <= 0.0:
        raise ValueError("alpha_min must be > 0.")
    if alpha_max <= 0.0:
        raise ValueError("alpha_max must be > 0.")
    if alpha_max < alpha_min:
        raise ValueError("alpha_max must be >= alpha_min.")
    if n_alpha < 10:
        raise ValueError("n_alpha must be >= 10.")


def run_ridge_alpha_sweep(
    alpha_min: float,
    alpha_max: float,
    n_alpha: int = 50,
    benchmarks: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate one Ridge alpha grid on the current research frame."""
    _validate_alpha_inputs(alpha_min, alpha_max, n_alpha)
    selected_benchmarks = list(benchmarks or DEFAULT_BENCHMARKS)
    alpha_grid = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)

    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rows: list[dict[str, object]] = []
        ridge_features = list(MODEL_FEATURE_OVERRIDES["ridge"])

        for benchmark in selected_benchmarks:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
            feature_cols = [col for col in ridge_features if col in x_df.columns]
            if not feature_cols:
                continue

            def pipeline_factory():
                return build_ridge_pipeline(alphas=alpha_grid, target_horizon_months=6)

            y_true, y_hat = custom_wfo(
                x_df[feature_cols].to_numpy(),
                y.to_numpy(),
                pipeline_factory,
            )
            metrics = compute_metrics(y_true, y_hat)
            rows.append(
                {
                    "benchmark": benchmark,
                    **metrics,
                    "_y_true": y_true,
                    "_y_hat": y_hat,
                }
            )

        if not rows:
            raise RuntimeError("Ridge alpha sweep produced no benchmark rows.")

        pooled = pool_metrics(rows)
        return {
            "pooled_oos_r2": float(pooled["r2"]),
            "pooled_ic": float(pooled["ic"]),
            "pooled_hit_rate": float(pooled["hit_rate"]),
            "pooled_sigma_ratio": float(pooled["sigma_ratio"]),
        }
    finally:
        conn.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Ridge alpha grids.")
    parser.add_argument("--alpha-min", type=float, default=config.RIDGE_ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=config.RIDGE_ALPHA_MAX)
    parser.add_argument("--n-alpha", type=int, default=config.RIDGE_ALPHA_N)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point."""
    args = _parse_args(argv)
    metrics = run_ridge_alpha_sweep(
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        n_alpha=args.n_alpha,
    )
    print(f"pooled_oos_r2={metrics['pooled_oos_r2']:.4f}")
    print(f"pooled_ic={metrics['pooled_ic']:.4f}")
    print(f"pooled_hit_rate={metrics['pooled_hit_rate']:.4f}")
    print(f"pooled_sigma_ratio={metrics['pooled_sigma_ratio']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
