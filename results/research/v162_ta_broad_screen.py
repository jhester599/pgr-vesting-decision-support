"""v162 broad technical-analysis screen helpers and harness.

The harness is intentionally research-only. It defines a broad Alpha Vantage
indicator inventory, expands candidate feature sets as add/replace experiments,
and records baseline deltas for regression and classification metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
DELTA_PREFIXES: tuple[str, ...] = ("baseline_", "delta_")


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
    baseline = detail_df.loc[
        detail_df["experiment_mode"] == "baseline",
        key_cols + present_metrics,
    ].copy()
    baseline = baseline.drop_duplicates(key_cols, keep="first")
    baseline = baseline.rename(columns={col: f"baseline_{col}" for col in present_metrics})
    merged = detail_df.merge(baseline, on=key_cols, how="left")
    for col in present_metrics:
        merged[f"delta_{col}"] = merged[col] - merged[f"baseline_{col}"]
    return merged


def _positive_benchmark_mask(group: pd.DataFrame, model_family: str) -> pd.Series:
    """Return per-row gate-improvement flags for the model family."""
    index = group.index
    if model_family == "classification":
        ba_delta = group.get("delta_balanced_accuracy", pd.Series(0.0, index=index))
        brier_delta = group.get("delta_brier_score", pd.Series(0.0, index=index))
        return ba_delta.fillna(0.0).gt(0.0) | brier_delta.fillna(0.0).lt(0.0)

    ic_delta = group.get("delta_ic", pd.Series(0.0, index=index))
    r2_delta = group.get("delta_oos_r2", pd.Series(0.0, index=index))
    return ic_delta.fillna(0.0).gt(0.0) | r2_delta.fillna(0.0).gt(0.0)


def _mean_delta_score(group: pd.DataFrame, model_family: str) -> float:
    """Build a directionally consistent mean improvement score."""
    if model_family == "classification":
        columns = []
        if "delta_balanced_accuracy" in group:
            columns.append(group["delta_balanced_accuracy"])
        if "delta_brier_score" in group:
            columns.append(-group["delta_brier_score"])
    else:
        columns = []
        if "delta_ic" in group:
            columns.append(group["delta_ic"])
        if "delta_oos_r2" in group:
            columns.append(group["delta_oos_r2"])
    if not columns:
        return float("nan")
    return float(pd.concat(columns, axis=1).mean(axis=1).mean())


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
                "feature_group": str(group["feature_group"].dropna().iloc[0])
                if "feature_group" in group and group["feature_group"].notna().any()
                else "",
                "family": str(group["family"].dropna().iloc[0])
                if "family" in group and group["family"].notna().any()
                else "",
                "n_benchmarks": int(group["benchmark"].nunique()),
                "mean_delta_ic": float(group.get("delta_ic", pd.Series(dtype=float)).mean()),
                "mean_delta_oos_r2": float(group.get("delta_oos_r2", pd.Series(dtype=float)).mean()),
                "mean_delta_balanced_accuracy": float(
                    group.get("delta_balanced_accuracy", pd.Series(dtype=float)).mean()
                ),
                "mean_delta_brier_score": float(
                    group.get("delta_brier_score", pd.Series(dtype=float)).mean()
                ),
                "mean_delta_score": _mean_delta_score(group, str(model_family)),
                "positive_benchmark_count": int(
                    group.loc[_positive_benchmark_mask(group, str(model_family)), "benchmark"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def compact_detail_records(
    detail_df: pd.DataFrame,
    benchmarks: tuple[str, ...] = PRIMARY_BENCHMARKS,
) -> pd.DataFrame:
    """Deduplicate persisted detail rows and recompute corrected deltas."""
    if detail_df.empty:
        return detail_df.copy()

    compact = detail_df.copy()
    stale_delta_cols = [
        col for col in compact.columns if col.startswith(DELTA_PREFIXES)
    ]
    compact = compact.drop(columns=stale_delta_cols)
    compact = compact.drop_duplicates().reset_index(drop=True)

    inventory = build_candidate_inventory(benchmarks).set_index("feature")
    if "feature_group" not in compact.columns:
        compact["feature_group"] = compact["feature"].map(inventory["feature_group"]).fillna("")
    else:
        compact["feature_group"] = compact["feature_group"].fillna("")
        missing_group = compact["feature_group"].eq("") & compact["feature"].isin(inventory.index)
        compact.loc[missing_group, "feature_group"] = compact.loc[
            missing_group,
            "feature",
        ].map(inventory["feature_group"])
    if "family" not in compact.columns:
        compact["family"] = compact["feature"].map(inventory["family"]).fillna("")
    else:
        compact["family"] = compact["family"].fillna("")
        missing_family = compact["family"].eq("") & compact["feature"].isin(inventory.index)
        compact.loc[missing_family, "family"] = compact.loc[
            missing_family,
            "feature",
        ].map(inventory["family"])

    compact.loc[compact["feature"].eq("__baseline__"), ["feature_group", "family"]] = ""
    return attach_baseline_deltas(compact)


def write_screen_artifacts(
    detail: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    benchmarks: tuple[str, ...] = PRIMARY_BENCHMARKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write broad-screen detail, summary, and inventory artifacts."""
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


def _load_price_map(conn: Any, tickers: list[str]) -> dict[str, pd.DataFrame]:
    price_map: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        prices = db_client.get_prices(conn, ticker)
        if not prices.empty:
            price_map[ticker] = prices
    return price_map


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
        inventory_by_feature = inventory.set_index("feature").to_dict(orient="index")
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
            baseline_selected = [
                col for col in baseline_features if col in X_aligned.columns
            ]
            for model_type in ("ridge", "gbt"):
                _, metrics = evaluate_wfo_model(
                    X_aligned,
                    y_reg,
                    model_type=model_type,
                    benchmark=benchmark,
                    target_horizon_months=6,
                    feature_columns=baseline_selected,
                )
                records.append(
                    {
                        "model_family": "regression",
                        "benchmark": benchmark,
                        "model_type": model_type,
                        "experiment_mode": "baseline",
                        "feature": "__baseline__",
                        "feature_group": "",
                        "family": "",
                        "replaced_feature": "",
                        "n_features": len(baseline_selected),
                        **metrics,
                    }
                )
            y_binary = (y_reg > 0.0).astype(int).rename(f"{benchmark}_outperform")
            _, _, cls_metrics = evaluate_confirmatory_classifier(
                X_aligned,
                y_binary,
                model_type="ridge",
                feature_columns=baseline_selected,
                target_horizon_months=6,
            )
            records.append(
                {
                    "model_family": "classification",
                    "benchmark": benchmark,
                    "model_type": "ridge",
                    "experiment_mode": "baseline",
                    "feature": "__baseline__",
                    "feature_group": "",
                    "family": "",
                    "replaced_feature": "",
                    "n_features": len(baseline_selected),
                    **cls_metrics,
                }
            )
            for candidate in candidate_features:
                for spec in build_feature_set_specs(baseline_features, candidate):
                    if spec["experiment_mode"] == "baseline":
                        continue
                    selected = [
                        col for col in spec["feature_columns"] if col in X_aligned.columns
                    ]
                    metadata = inventory_by_feature.get(candidate, {})
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
                                "feature_group": metadata.get("feature_group", ""),
                                "family": metadata.get("family", ""),
                                "replaced_feature": spec["replaced_feature"],
                                "n_features": len(selected),
                                **metrics,
                            }
                        )
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
                            "feature_group": metadata.get("feature_group", ""),
                            "family": metadata.get("family", ""),
                            "replaced_feature": spec["replaced_feature"],
                            "n_features": len(selected),
                            **cls_metrics,
                        }
                    )
    finally:
        conn.close()

    detail = compact_detail_records(pd.DataFrame(records), benchmarks=benchmarks)
    return write_screen_artifacts(detail, output_dir=output_dir, benchmarks=benchmarks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v162 broad TA feature screen.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--reuse-detail",
        default="",
        help="Deduplicate and resummarize an existing detail CSV instead of refitting.",
    )
    args = parser.parse_args()
    if args.reuse_detail:
        existing_detail = pd.read_csv(args.reuse_detail)
        detail = compact_detail_records(existing_detail)
        write_screen_artifacts(detail, output_dir=Path(args.output_dir))
    else:
        run_broad_screen(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
