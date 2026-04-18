"""v162 broad technical-analysis screen helpers and harness.

The harness is intentionally research-only. It defines a broad Alpha Vantage
indicator inventory, expands candidate feature sets as add/replace experiments,
and records baseline deltas for regression and classification metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from scripts.confirmatory_classifier_experiments import evaluate_confirmatory_classifier
from src.database import db_client
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_X_y_relative,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.evaluation import evaluate_wfo_model
from src.research.v160_ta_features import build_ta_feature_matrix

PRIMARY_BENCHMARKS: tuple[str, ...] = (
    "VOO",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
)
REDUNDANT_BASELINE_FEATURES: tuple[str, ...] = (
    "mom_3m",
    "mom_6m",
    "mom_12m",
    "vol_63d",
    "vix",
)
OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "v162_ta_broad_screen_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "v162_ta_broad_screen_summary.csv"


def build_candidate_inventory(
    benchmarks: list[str] | tuple[str, ...] = PRIMARY_BENCHMARKS,
) -> pd.DataFrame:
    """Return the pre-registered TA candidate inventory."""
    rows: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        suffix = benchmark.lower()
        rows.extend(
            [
                {
                    "feature": f"ta_ratio_roc_6m_{suffix}",
                    "feature_group": "ratio_momentum",
                    "family": "momentum",
                    "include_in_model": True,
                },
                {
                    "feature": f"ta_ratio_ema_gap_12m_{suffix}",
                    "feature_group": "ratio_trend",
                    "family": "trend",
                    "include_in_model": True,
                },
                {
                    "feature": f"ta_ratio_rsi_6m_{suffix}",
                    "feature_group": "ratio_oscillator",
                    "family": "momentum",
                    "include_in_model": True,
                },
                {
                    "feature": f"ta_ratio_bb_pct_b_6m_{suffix}",
                    "feature_group": "ratio_bollinger",
                    "family": "volatility",
                    "include_in_model": True,
                },
                {
                    "feature": f"ta_ratio_bb_width_6m_{suffix}",
                    "feature_group": "ratio_bollinger",
                    "family": "volatility",
                    "include_in_model": True,
                },
                {
                    "feature": f"ta_ratio_roc_6m_{suffix}__x__ta_pgr_natr_63d",
                    "feature_group": "volatility_interaction",
                    "family": "regime",
                    "include_in_model": True,
                },
                {
                    "feature": f"ta_ratio_roc_6m_{suffix}__x__ta_pgr_adx_63d",
                    "feature_group": "trend_strength_interaction",
                    "family": "regime",
                    "include_in_model": True,
                },
            ]
        )

    rows.extend(
        [
            {
                "feature": "ta_pgr_natr_63d",
                "feature_group": "pgr_volatility",
                "family": "volatility",
                "include_in_model": True,
            },
            {
                "feature": "ta_pgr_adx_63d",
                "feature_group": "pgr_trend_strength",
                "family": "regime",
                "include_in_model": True,
            },
            {
                "feature": "ta_pgr_macd_hist_norm",
                "feature_group": "pgr_momentum",
                "family": "momentum",
                "include_in_model": True,
            },
            {
                "feature": "ta_pgr_obv_detrended",
                "feature_group": "pgr_volume",
                "family": "volume",
                "include_in_model": True,
            },
            {
                "feature": "ta_voo_pc_tech",
                "feature_group": "benchmark_regime",
                "family": "regime",
                "include_in_model": True,
            },
            {
                "feature": "ta_bnd_macd_hist_norm",
                "feature_group": "benchmark_regime",
                "family": "momentum",
                "include_in_model": True,
            },
            {
                "feature": "ta_peer_ratio_roc_6m",
                "feature_group": "peer_relative",
                "family": "momentum",
                "include_in_model": True,
            },
            {
                "feature": "ta_peer_ratio_rsi_6m",
                "feature_group": "peer_relative",
                "family": "momentum",
                "include_in_model": True,
            },
            {
                "feature": "ta_peer_ratio_ema_gap_12m",
                "feature_group": "peer_relative",
                "family": "trend",
                "include_in_model": True,
            },
        ]
    )
    for ticker in ("GLD", "DBC", "VDE"):
        rows.append(
            {
                "feature": f"ta_{ticker.lower()}_roc_6m",
                "feature_group": "benchmark_regime",
                "family": "momentum",
                "include_in_model": True,
            }
        )

    rows.extend(
        [
            {
                "feature": "candlestick_pattern_count",
                "feature_group": "diagnostic_excluded",
                "family": "pattern",
                "include_in_model": False,
            },
            {
                "feature": "ht_dcphase_sin_cos",
                "feature_group": "diagnostic_excluded",
                "family": "hilbert",
                "include_in_model": False,
            },
            {
                "feature": "typical_price_transform",
                "feature_group": "diagnostic_excluded",
                "family": "price_transform",
                "include_in_model": False,
            },
        ]
    )
    return pd.DataFrame(rows).drop_duplicates("feature").reset_index(drop=True)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_feature_set_specs(
    baseline_features: list[str],
    candidate_feature: str,
    redundant_features: list[str] | tuple[str, ...] = REDUNDANT_BASELINE_FEATURES,
) -> list[dict[str, Any]]:
    """Build baseline, add-one, and replacement feature-set specs."""
    baseline = _dedupe(list(baseline_features))
    specs: list[dict[str, Any]] = [
        {
            "experiment_mode": "baseline",
            "feature_columns": baseline,
            "replaced_feature": "",
        },
        {
            "experiment_mode": "add_feature",
            "feature_columns": _dedupe(baseline + [candidate_feature]),
            "replaced_feature": "",
        },
    ]
    for redundant in redundant_features:
        if redundant not in baseline:
            continue
        specs.append(
            {
                "experiment_mode": f"replace_{redundant}",
                "feature_columns": _dedupe(
                    [feature for feature in baseline if feature != redundant]
                    + [candidate_feature]
                ),
                "replaced_feature": redundant,
            }
        )
    return specs


def attach_baseline_deltas(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Attach metric deltas versus matching baseline rows."""
    if detail_df.empty:
        return detail_df.copy()
    key_cols = ["model_family", "benchmark", "model_type"]
    metric_cols = [
        "ic",
        "oos_r2",
        "hit_rate",
        "balanced_accuracy",
        "brier_score",
        "ba_covered",
    ]
    present_metrics = [col for col in metric_cols if col in detail_df.columns]
    baseline = detail_df.loc[detail_df["experiment_mode"] == "baseline", key_cols + present_metrics].copy()
    baseline = baseline.rename(columns={col: f"baseline_{col}" for col in present_metrics})
    merged = detail_df.merge(baseline, on=key_cols, how="left")
    for col in present_metrics:
        merged[f"delta_{col}"] = merged[col] - merged[f"baseline_{col}"]
    return merged


def summarize_screen(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize broad-screen rows by feature and model family."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in detail_df.groupby(
        ["model_family", "model_type", "feature", "experiment_mode"],
        dropna=False,
    ):
        model_family, model_type, feature, experiment_mode = keys
        rows.append(
            {
                "model_family": model_family,
                "model_type": model_type,
                "feature": feature,
                "experiment_mode": experiment_mode,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_delta_ic": float(group.get("delta_ic", pd.Series(dtype=float)).mean()),
                "mean_delta_oos_r2": float(group.get("delta_oos_r2", pd.Series(dtype=float)).mean()),
                "mean_delta_balanced_accuracy": float(
                    group.get("delta_balanced_accuracy", pd.Series(dtype=float)).mean()
                ),
                "mean_delta_brier_score": float(
                    group.get("delta_brier_score", pd.Series(dtype=float)).mean()
                ),
                "positive_benchmark_count": int(
                    (
                        group.get("delta_ic", pd.Series(dtype=float)).fillna(0.0) > 0.0
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_price_map(conn: Any, tickers: list[str]) -> dict[str, pd.DataFrame]:
    return {
        ticker: db_client.get_prices(conn, ticker)
        for ticker in tickers
        if not db_client.get_prices(conn, ticker).empty
    }


def run_broad_screen(
    output_dir: Path = OUTPUT_DIR,
    benchmarks: tuple[str, ...] = PRIMARY_BENCHMARKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the broad TA screen and write deterministic research artifacts."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        base_features = build_feature_matrix_from_db(conn)
        tickers = sorted(set(("PGR", *benchmarks, "ALL", "TRV", "CB", "HIG")))
        price_map = _load_price_map(conn, tickers)
        ta_features = build_ta_feature_matrix(price_map, benchmarks=benchmarks)
        feature_df = base_features.join(ta_features, how="left")
        inventory = build_candidate_inventory(benchmarks)
        candidate_features = [
            feature
            for feature in inventory.loc[inventory["include_in_model"], "feature"].tolist()
            if feature in feature_df.columns
        ]
        baseline_features = [
            feature
            for feature in config.MODEL_FEATURE_OVERRIDES["ridge"]
            if feature in feature_df.columns
        ]
        records: list[dict[str, Any]] = []
        for benchmark in benchmarks:
            rel_series = load_relative_return_matrix(conn, benchmark, 6)
            if rel_series.empty:
                continue
            try:
                X_aligned, y_reg = get_X_y_relative(feature_df, rel_series)
            except ValueError:
                continue
            for candidate in candidate_features:
                for spec in build_feature_set_specs(baseline_features, candidate):
                    selected = [
                        col for col in spec["feature_columns"] if col in X_aligned.columns
                    ]
                    for model_type in ("ridge", "gbt"):
                        _, metrics = evaluate_wfo_model(
                            X_aligned,
                            y_reg,
                            model_type=model_type,
                            benchmark=benchmark,
                            target_horizon_months=6,
                            feature_columns=selected,
                        )
                        records.append(
                            {
                                "model_family": "regression",
                                "benchmark": benchmark,
                                "model_type": model_type,
                                "experiment_mode": spec["experiment_mode"],
                                "feature": candidate if spec["experiment_mode"] != "baseline" else "__baseline__",
                                "replaced_feature": spec["replaced_feature"],
                                "n_features": len(selected),
                                **metrics,
                            }
                        )
                    y_binary = (y_reg > 0.0).astype(int).rename(f"{benchmark}_outperform")
                    _, _, cls_metrics = evaluate_confirmatory_classifier(
                        X_aligned,
                        y_binary,
                        model_type="ridge",
                        feature_columns=selected,
                        target_horizon_months=6,
                    )
                    records.append(
                        {
                            "model_family": "classification",
                            "benchmark": benchmark,
                            "model_type": "ridge",
                            "experiment_mode": spec["experiment_mode"],
                            "feature": candidate if spec["experiment_mode"] != "baseline" else "__baseline__",
                            "replaced_feature": spec["replaced_feature"],
                            "n_features": len(selected),
                            **cls_metrics,
                        }
                    )
    finally:
        conn.close()

    detail = attach_baseline_deltas(pd.DataFrame(records))
    summary = summarize_screen(detail)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / DETAIL_PATH.name, index=False)
    summary.to_csv(output_dir / SUMMARY_PATH.name, index=False)
    metadata_path = output_dir / "v162_ta_broad_screen_inventory.json"
    metadata_path.write_text(
        json.dumps(build_candidate_inventory(benchmarks).to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v162 broad TA feature screen.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run_broad_screen(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
