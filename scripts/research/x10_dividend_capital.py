"""Run x10 capital-enhanced special-dividend experiments."""

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
from src.research.x1_targets import build_special_dividend_targets
from src.research.x4_bvps_forecasting import normalize_bvps_monthly
from src.research.x6_special_dividend import (
    evaluate_special_dividend_two_stage,
    summarize_special_dividend_results,
)
from src.research.x9_bvps_bridge import (
    build_bvps_bridge_features,
    build_bvps_interactions,
)
from src.research.x10_dividend_capital import (
    build_dividend_capital_features,
    build_x10_feature_sets,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x10_dividend_capital_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x10_dividend_capital_summary.json"
MEMO_PATH = OUTPUT_DIR / "x10_research_memo.md"
MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("historical_rate", "historical_positive_mean"),
    ("historical_rate", "ridge_positive_excess"),
    ("logistic_l2_balanced", "historical_positive_mean"),
    ("logistic_l2_balanced", "ridge_positive_excess"),
)


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x10 reads this cache "
            "without refreshing it to preserve the research-only boundary."
        )
    return pd.read_parquet(path)


def _month_end_close(prices: pd.DataFrame) -> pd.Series:
    close = prices["close"].copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    result = close.resample("BME").last()
    result.name = "close_price"
    return result


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def _build_monthly_capital_frame(
    feature_df: pd.DataFrame,
    current_bvps: pd.Series,
    close_price: pd.Series,
) -> pd.DataFrame:
    base = feature_df[get_feature_columns(feature_df)].copy()
    bridge = build_bvps_bridge_features(base, current_bvps)
    interactions = build_bvps_interactions(bridge)
    monthly = bridge.join(interactions, how="left")
    monthly["close_price"] = close_price.reindex(monthly.index)
    monthly["book_value_per_share"] = current_bvps.reindex(monthly.index)
    return build_dividend_capital_features(monthly)


def _build_annual_frame() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        prices = db_client.get_prices(conn, "PGR")
        dividends = db_client.get_dividends(conn, "PGR")
        pgr_monthly = db_client.get_pgr_edgar_monthly(conn)
    finally:
        conn.close()

    close_price = _month_end_close(prices)
    current_bvps = normalize_bvps_monthly(
        pgr_monthly,
        filing_lag_months=config.EDGAR_FILING_LAG_MONTHS,
    )
    monthly = _build_monthly_capital_frame(feature_df, current_bvps, close_price)
    targets = build_special_dividend_targets(monthly, dividends)
    annual = monthly[monthly.index.month == 11].join(
        targets[
            [
                "special_dividend_occurred",
                "special_dividend_excess",
                "special_dividend_excess_to_bvps",
                "special_dividend_excess_to_price",
                "regular_baseline_dividend",
                "q1_dividend_total",
            ]
        ],
        how="inner",
    )
    return annual, build_x10_feature_sets(annual)


def run_x10_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run x10 annual special-dividend feature-set comparison."""
    annual, feature_sets = _build_annual_frame()
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for feature_set, feature_columns in feature_sets.items():
        if not feature_columns:
            continue
        for stage1_name, stage2_name in MODEL_SPECS:
            predictions, metrics = evaluate_special_dividend_two_stage(
                annual,
                feature_columns=feature_columns,
                stage1_model_name=stage1_name,
                stage2_model_name=stage2_name,
                min_train_years=8,
            )
            model_name = f"{stage1_name}__{stage2_name}"
            predictions = predictions.copy()
            predictions["feature_set"] = feature_set
            predictions["model_name"] = model_name
            prediction_rows.append(predictions)
            metrics["feature_set"] = feature_set
            metric_rows.append(metrics)

    detail = pd.concat(prediction_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    summary = summarize_special_dividend_results(
        metrics_df.sort_values(
            ["expected_value_mae", "stage1_brier", "feature_set"],
            kind="mergesort",
        )
    )
    return detail, metrics_df, summary


def write_artifacts(
    detail: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write x10 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x10",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "prediction_timestamp": "November business month-end",
        "ranking_basis": "expected_value_mae ascending, then stage1_brier",
        "metrics": _json_records(metrics_df),
        "ranked_rows": _json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(detail, summary)


def write_memo(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write compact x10 memo."""
    rows_per_model_feature = int(
        detail.groupby(["feature_set", "model_name"]).size().iloc[0]
    )
    best = summary.iloc[0]
    lines = [
        "# x10 Research Memo",
        "",
        "## Scope",
        "",
        "x10 re-tests the annual Q1 special-dividend sidecar with x9",
        "capital-generation features. It remains research-only and uses",
        "November business-month-end snapshots only.",
        "",
        "## Sample",
        "",
        f"- OOS annual predictions per model-feature set: {rows_per_model_feature}.",
        "- Validation remains expanding annual train/test splits.",
        "",
        "## Results",
        "",
        (
            f"- Best row: `{best['feature_set']}` / `{best['model_name']}` "
            f"(EV MAE {best['expected_value_mae']:.3f}, stage-1 Brier "
            f"{best['stage1_brier']:.3f}, stage-1 balanced accuracy "
            f"{best['stage1_balanced_accuracy']:.3f})."
        ),
        "",
        "## Interpretation",
        "",
        "- x10 should be read as feature-set diagnostics, not as a dividend",
        "  deployment model.",
        "- Annual sample size remains the main confidence limiter.",
        "- x11 should compare x10 against x6 and document whether x9 capital",
        "  features improved the dividend sidecar enough to continue.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, metrics_df, summary = run_x10_experiments()
    write_artifacts(detail, metrics_df, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
