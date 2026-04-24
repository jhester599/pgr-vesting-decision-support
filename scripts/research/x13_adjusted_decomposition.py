"""Run x13 adjusted decomposition comparisons."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import db_client
from src.processing.feature_engineering import get_feature_columns
from src.research.x4_bvps_forecasting import normalize_bvps_monthly
from src.research.x5_pb_decomposition import evaluate_pb_baseline
from src.research.x9_bvps_bridge import (
    build_bvps_bridge_features,
    build_bvps_interactions,
    evaluate_x9_bvps_baseline,
    evaluate_x9_bvps_model,
)
from src.research.x12_bvps_target_audit import (
    build_adjusted_bvps_targets,
    build_monthly_dividend_series,
)
from src.research.x13_adjusted_decomposition import (
    X13_HORIZONS,
    combine_adjusted_decomposition_predictions,
    json_records,
    summarize_x13_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x13_adjusted_decomposition_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x13_adjusted_decomposition_summary.json"
MEMO_PATH = OUTPUT_DIR / "x13_research_memo.md"
BVPS_SPECS: tuple[tuple[str, str, list[str]], ...] = (
    ("drift_bvps_growth", "baseline", []),
    ("ridge_bridge", "model", [
        "current_bvps",
        "bvps_growth_1m",
        "bvps_growth_3m",
        "bvps_growth_6m",
        "bvps_growth_ytd",
        "bvps_yoy_dollar_change",
        "month_of_year",
        "q4_flag",
        "dividend_season_flag",
    ]),
    ("elastic_net_bridge", "model", [
        "premium_growth_x_underwriting_margin",
        "premium_to_surplus_x_cr_delta",
        "buyback_yield_x_pb_ratio",
        "buyback_yield_x_bvps_growth_3m",
        "unrealized_gain_x_real_rate",
        "investment_yield_x_bvps",
    ]),
)


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x13 reads this cache "
            "without refreshing it to preserve the research-only boundary."
        )
    return pd.read_parquet(path)


def _month_end_close(prices: pd.DataFrame) -> pd.Series:
    close = prices["close"].copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    result = close.resample("BME").last()
    result.name = "close_price"
    return result


def _build_feature_matrix(feature_df: pd.DataFrame, current_bvps: pd.Series) -> pd.DataFrame:
    base = feature_df[get_feature_columns(feature_df)].copy()
    bridge = build_bvps_bridge_features(base, current_bvps)
    interactions = build_bvps_interactions(bridge)
    return bridge.join(interactions, how="left")


def run_x13_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run raw-vs-adjusted BVPS decomposition comparisons."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        prices = db_client.get_prices(conn, "PGR")
        dividends = db_client.get_dividends(conn, "PGR")
        pgr_monthly = db_client.get_pgr_edgar_monthly(conn)
    finally:
        conn.close()

    current_bvps = normalize_bvps_monthly(
        pgr_monthly,
        filing_lag_months=config.EDGAR_FILING_LAG_MONTHS,
    )
    close_price = _month_end_close(prices)
    current_pb = (close_price / current_bvps).rename("current_pb")
    monthly_dividends = build_monthly_dividend_series(dividends, current_bvps.index)
    targets = build_adjusted_bvps_targets(
        current_bvps,
        monthly_dividends,
        horizons=X13_HORIZONS,
    )
    X = _build_feature_matrix(feature_df, current_bvps)

    rows: list[dict[str, Any]] = []
    for horizon in X13_HORIZONS:
        pb_target = targets[f"target_{horizon}m_bvps"] * 0 + current_pb.shift(-horizon)
        pb_target = pb_target.rename(f"target_{horizon}m_pb_proxy")
        pb_predictions, _ = evaluate_pb_baseline(
            X,
            pb_target,
            current_pb=current_pb,
            baseline_name="no_change_pb",
            target_kind="pb",
            target_horizon_months=horizon,
        )
        for target_variant, target_column in {
            "raw": f"target_{horizon}m_bvps_growth",
            "adjusted": f"target_{horizon}m_adjusted_bvps_growth",
        }.items():
            y = targets[target_column].rename(target_column)
            for model_name, spec_kind, feature_columns in BVPS_SPECS:
                if spec_kind == "baseline":
                    bvps_predictions, _ = evaluate_x9_bvps_baseline(
                        X,
                        y,
                        current_bvps=current_bvps,
                        baseline_name=model_name,
                        target_kind="growth",
                        target_horizon_months=horizon,
                    )
                else:
                    selected = [column for column in feature_columns if column in X.columns]
                    bvps_predictions, _, _ = evaluate_x9_bvps_model(
                        X,
                        y,
                        current_bvps=current_bvps,
                        model_name=model_name,
                        feature_columns=selected,
                        target_kind="growth",
                        target_horizon_months=horizon,
                    )
                _, metrics = combine_adjusted_decomposition_predictions(
                    bvps_predictions,
                    pb_predictions,
                    horizon_months=horizon,
                    bvps_model_name=model_name,
                    pb_model_name="no_change_pb",
                    target_variant=target_variant,
                )
                metrics["feature_block"] = (
                    "baseline" if spec_kind == "baseline" else
                    ("bvps_lags" if model_name == "ridge_bridge" else "logical_interactions")
                )
                rows.append(metrics)

    detail = pd.DataFrame(rows)
    summary = summarize_x13_results(detail)
    return detail, summary


def write_artifacts(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x13 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x13",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "horizons": list(X13_HORIZONS),
        "pb_anchor": "no_change_pb",
        "ranked_rows": json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(summary)


def write_memo(summary: pd.DataFrame) -> None:
    """Write compact x13 memo."""
    lines = [
        "# x13 Research Memo",
        "",
        "## Scope",
        "",
        "x13 compares raw and dividend-adjusted BVPS decomposition paths at",
        "the 3m and 6m horizons, keeping `no_change_pb` as the P/B anchor.",
        "",
        "## Results",
        "",
    ]
    for horizon in X13_HORIZONS:
        rows = summary[summary["horizon_months"] == horizon]
        if rows.empty:
            continue
        raw_best = rows[rows["target_variant"] == "raw"].iloc[0]
        adj_best = rows[rows["target_variant"] == "adjusted"].iloc[0]
        lines.append(
            f"- {horizon}m raw best `{raw_best['model_name']}` "
            f"({raw_best['feature_block']}, price MAE {raw_best['implied_price_mae']:.3f}) "
            f"vs adjusted best `{adj_best['model_name']}` "
            f"({adj_best['feature_block']}, price MAE {adj_best['implied_price_mae']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If adjusted decomposition improves the same horizons that x12",
            "  improved, it becomes the best current candidate for a structural",
            "  x-series indicator.",
            "- x14 should only nominate an indicator if this adjusted path is",
            "  directionally consistent with the broader x-series evidence.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, summary = run_x13_experiments()
    write_artifacts(detail, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
