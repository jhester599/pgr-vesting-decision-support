"""v9.5 regime-slice backtests using production WFO paths."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from src.database import db_client
from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
from src.processing.feature_engineering import build_feature_matrix_from_db, get_X_y_relative
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import evaluate_wfo_model, reconstruct_ensemble_oos_predictions, summarize_predictions


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")


def _build_slice_labels(detail_df: pd.DataFrame, vix_series: pd.Series | None = None) -> pd.DataFrame:
    """Attach named regime slices to the per-date prediction rows."""
    detail_df = detail_df.copy()
    detail_df["date"] = pd.to_datetime(detail_df["date"])
    max_date = detail_df["date"].max()
    trailing_cutoff = max_date - pd.DateOffset(months=36)

    detail_df["slice_pre_2020"] = detail_df["date"] < pd.Timestamp("2020-01-01")
    detail_df["slice_2020_2021"] = (detail_df["date"] >= pd.Timestamp("2020-01-01")) & (
        detail_df["date"] < pd.Timestamp("2022-01-01")
    )
    detail_df["slice_post_2022"] = detail_df["date"] >= pd.Timestamp("2022-01-01")
    detail_df["slice_trailing_36m"] = detail_df["date"] >= trailing_cutoff

    if vix_series is not None and not vix_series.empty:
        vix_join = vix_series.rename("vix_regime").reindex(detail_df["date"], method="ffill").values
        detail_df["vix"] = vix_join
        detail_df["slice_low_vix"] = detail_df["vix"] <= 20.0
        detail_df["slice_high_vix"] = detail_df["vix"] > 20.0
    else:
        detail_df["vix"] = np.nan
        detail_df["slice_low_vix"] = False
        detail_df["slice_high_vix"] = False

    return detail_df


def _summarize_slice_rows(group: pd.DataFrame) -> dict[str, float | int]:
    """Compute summary metrics for a slice."""
    pred = pd.Series(group["predicted"].values, index=pd.DatetimeIndex(group["date"]))
    realized = pd.Series(group["realized"].values, index=pd.DatetimeIndex(group["date"]))
    summary = summarize_predictions(pred, realized, target_horizon_months=6)
    return {
        "n_obs": summary.n_obs,
        "mean_ic": summary.ic,
        "mean_hit_rate": float((np.sign(group["predicted"]) == np.sign(group["realized"])).mean()),
        "oos_r2": summary.oos_r2,
        "mae": summary.mae,
    }


def run_regime_slice_backtest(
    conn: Any,
    benchmarks: list[str],
    model_types: list[str],
    output_dir: str,
    target_horizon_months: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run regime-slice summaries for each model and the ensemble."""
    df = build_feature_matrix_from_db(conn)
    vix_series = df["vix"] if "vix" in df.columns else pd.Series(dtype=float)

    detail_rows: list[dict[str, Any]] = []
    rel_matrix = pd.DataFrame(
        {benchmark: load_relative_return_matrix(conn, benchmark, target_horizon_months) for benchmark in benchmarks}
    ).dropna(axis=1, how="all")
    ensemble_results = run_ensemble_benchmarks(
        df,
        rel_matrix,
        target_horizon_months=target_horizon_months,
    )

    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, target_horizon_months)
        if rel_series.empty:
            continue
        try:
            X_aligned, y_aligned = get_X_y_relative(df, rel_series, drop_na_target=True)
        except ValueError:
            continue

        for model_type in model_types:
            result, _ = evaluate_wfo_model(
                X_aligned,
                y_aligned,
                model_type=model_type,
                benchmark=benchmark,
                target_horizon_months=target_horizon_months,
            )
            dates = list(result.test_dates_all)
            for date_value, pred_value, realized_value in zip(dates, result.y_hat_all, result.y_true_all, strict=False):
                detail_rows.append(
                    {
                        "date": pd.Timestamp(date_value),
                        "benchmark": benchmark,
                        "model_type": model_type,
                        "predicted": float(pred_value),
                        "realized": float(realized_value),
                    }
                )

        ens_result = ensemble_results.get(benchmark)
        if ens_result is not None:
            y_hat, y_true = reconstruct_ensemble_oos_predictions(ens_result)
            for date_value, pred_value, realized_value in zip(y_hat.index, y_hat.values, y_true.values, strict=False):
                detail_rows.append(
                    {
                        "date": pd.Timestamp(date_value),
                        "benchmark": benchmark,
                        "model_type": "ensemble",
                        "predicted": float(pred_value),
                        "realized": float(realized_value),
                    }
                )

    detail_df = _build_slice_labels(pd.DataFrame(detail_rows), vix_series=vix_series)

    summary_rows: list[dict[str, Any]] = []
    slice_columns = [
        "slice_pre_2020",
        "slice_2020_2021",
        "slice_post_2022",
        "slice_trailing_36m",
        "slice_low_vix",
        "slice_high_vix",
    ]
    for model_type, model_df in detail_df.groupby("model_type", dropna=False):
        for slice_col in slice_columns:
            slice_df = model_df[model_df[slice_col]]
            if slice_df.empty:
                continue
            metrics = _summarize_slice_rows(slice_df)
            summary_rows.append(
                {
                    "model_type": model_type,
                    "slice_name": slice_col.replace("slice_", ""),
                    **metrics,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["model_type", "slice_name"],
        ascending=[True, True],
    )

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    detail_path = os.path.join(output_dir, f"regime_slice_detail_{stamp}.csv")
    summary_path = os.path.join(output_dir, f"regime_slice_summary_{stamp}.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote regime detail to {detail_path}")
    print(f"Wrote regime summary to {summary_path}")
    return detail_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 regime-slice backtests.")
    parser.add_argument(
        "--benchmarks",
        default=",".join(config.ETF_BENCHMARK_UNIVERSE),
        help="Comma-separated benchmark tickers.",
    )
    parser.add_argument(
        "--model-types",
        default=",".join(config.ENSEMBLE_MODELS),
        help="Comma-separated model types to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    benchmarks = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
    model_types = [value.strip() for value in args.model_types.split(",") if value.strip()]
    conn = db_client.get_connection(config.DB_PATH)
    run_regime_slice_backtest(
        conn=conn,
        benchmarks=benchmarks,
        model_types=model_types,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
