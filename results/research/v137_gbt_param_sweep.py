"""v137 -- Standalone GBT hyperparameter sweep on the production research frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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
from src.models.regularized_models import build_gbt_pipeline

DEFAULT_BENCHMARKS = list(PRIMARY_FORECAST_UNIVERSE)
DEFAULT_PARAMS_PATH = (
    PROJECT_ROOT / "results" / "research" / "v137_gbt_params_candidate.json"
)


def _validate_params(
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    subsample: float,
) -> None:
    """Validate the GBT candidate parameters."""
    if max_depth not in {1, 2, 3, 4}:
        raise ValueError("max_depth must be one of {1, 2, 3, 4}.")
    if not 10 <= n_estimators <= 200:
        raise ValueError("n_estimators must be in [10, 200].")
    if not 0.01 <= learning_rate <= 0.50:
        raise ValueError("learning_rate must be in [0.01, 0.50].")
    if not 0.50 <= subsample <= 1.00:
        raise ValueError("subsample must be in [0.50, 1.00].")


def run_gbt_sweep(
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    subsample: float,
    benchmarks: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate one standalone GBT configuration."""
    _validate_params(max_depth, n_estimators, learning_rate, subsample)
    selected_benchmarks = list(benchmarks or DEFAULT_BENCHMARKS)

    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rows: list[dict[str, object]] = []
        gbt_features = list(MODEL_FEATURE_OVERRIDES["gbt"])

        for benchmark in selected_benchmarks:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
            feature_cols = [col for col in gbt_features if col in x_df.columns]
            if not feature_cols:
                continue

            def pipeline_factory():
                return build_gbt_pipeline(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    subsample=subsample,
                )

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
            raise RuntimeError("GBT sweep produced no benchmark rows.")

        pooled = pool_metrics(rows)
        return {
            "pooled_oos_r2": float(pooled["r2"]),
            "pooled_ic": float(pooled["ic"]),
            "pooled_hit_rate": float(pooled["hit_rate"]),
        }
    finally:
        conn.close()


def _load_params_from_file(path: str) -> dict[str, float | int]:
    """Load one params JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("params file must decode to a JSON object.")
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate GBT hyperparameters.")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--params-file", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point."""
    args = _parse_args(argv)
    params = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
    }
    if args.params_file:
        params = _load_params_from_file(args.params_file)
    metrics = run_gbt_sweep(
        max_depth=int(params["max_depth"]),
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
    )
    print(f"pooled_oos_r2={metrics['pooled_oos_r2']:.4f}")
    print(f"pooled_ic={metrics['pooled_ic']:.4f}")
    print(f"pooled_hit_rate={metrics['pooled_hit_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
