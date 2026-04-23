"""Run x6 special-dividend two-stage annual sidecar."""

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

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x6_special_dividend_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x6_special_dividend_summary.json"
MEMO_PATH = OUTPUT_DIR / "x6_research_memo.md"
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
            f"Missing feature matrix cache at {path}. x6 reads this cache "
            "without refreshing it to preserve the research-only boundary."
        )
    return pd.read_parquet(path)


def _month_end_close(prices: pd.DataFrame) -> pd.Series:
    close = prices["close"].copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    result = close.resample("BME").last()
    result.name = "close_price"
    return result


def _feature_subset(feature_df: pd.DataFrame) -> list[str]:
    preferred = [
        "book_value_per_share_growth_yoy",
        "pb_ratio",
        "pgr_premium_to_surplus",
        "pgr_price_to_book_relative",
        "combined_ratio_ttm",
        "underwriting_margin_ttm",
        "underwriting_income_growth_yoy",
        "roe_net_income_ttm",
        "npw_growth_yoy",
        "pif_growth_yoy",
        "gainshare_est",
        "investment_income_growth_yoy",
        "investment_book_yield",
        "unrealized_gain_pct_equity",
        "realized_gain_to_net_income_ratio",
        "buyback_yield",
        "buyback_acceleration",
        "debt_to_total_capital",
        "real_rate_10y",
        "credit_spread_hy",
        "vix",
    ]
    available = set(get_feature_columns(feature_df))
    return [feature for feature in preferred if feature in available]


def _build_annual_frame() -> tuple[pd.DataFrame, list[str]]:
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        prices = db_client.get_prices(conn, "PGR")
        dividends = db_client.get_dividends(conn, "PGR")
        pgr_monthly = db_client.get_pgr_edgar_monthly(conn)
    finally:
        conn.close()

    monthly_features = feature_df.copy()
    monthly_features["close_price"] = _month_end_close(prices).reindex(
        monthly_features.index
    )
    monthly_features["book_value_per_share"] = normalize_bvps_monthly(
        pgr_monthly,
        filing_lag_months=config.EDGAR_FILING_LAG_MONTHS,
    ).reindex(monthly_features.index)

    targets = build_special_dividend_targets(monthly_features, dividends)
    november_features = monthly_features[monthly_features.index.month == 11]
    annual = november_features.join(
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
    feature_columns = _feature_subset(feature_df)
    return annual, feature_columns


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def run_x6_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run x6 annual two-stage model rows."""
    annual, feature_columns = _build_annual_frame()
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for stage1_name, stage2_name in MODEL_SPECS:
        predictions, metrics = evaluate_special_dividend_two_stage(
            annual,
            feature_columns=feature_columns,
            stage1_model_name=stage1_name,
            stage2_model_name=stage2_name,
            min_train_years=8,
        )
        predictions = predictions.copy()
        predictions["stage1_model_name"] = stage1_name
        predictions["stage2_model_name"] = stage2_name
        predictions["model_name"] = f"{stage1_name}__{stage2_name}"
        prediction_rows.append(predictions)
        metric_rows.append(metrics)
    detail = pd.concat(prediction_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    summary = summarize_special_dividend_results(metrics_df)
    return detail, metrics_df, summary


def write_artifacts(
    detail: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write x6 CSV, JSON, and memo artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x6",
        "artifact_classification": "research",
        "production_changes": False,
        "prediction_timestamp": "November business month-end",
        "ranking_basis": "expected_value_mae ascending, then stage1_brier",
        "model_specs": [
            {"stage1": stage1, "stage2": stage2}
            for stage1, stage2 in MODEL_SPECS
        ],
        "metrics": _json_records(metrics_df),
        "ranked_rows": _json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(detail, summary)


def write_memo(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write a compact x6 memo."""
    rows_per_model = int(detail.groupby("model_name").size().iloc[0])
    best = summary.iloc[0]
    lines = [
        "# x6 Research Memo",
        "",
        "## Scope",
        "",
        "x6 runs a research-only two-stage Q1 special-dividend sidecar using",
        "November business-month-end snapshots only. It does not alter",
        "production or monthly shadow artifacts.",
        "",
        "## Sample",
        "",
        f"- OOS annual predictions per model: {rows_per_model}.",
        "- The normal quarterly dividend baseline is inferred from repo",
        "  dividend history by x1 target utilities, not hardcoded.",
        "",
        "## Results",
        "",
        (
            f"- Best expected-value row: `{best['model_name']}` "
            f"(EV MAE {best['expected_value_mae']:.3f}, "
            f"stage-1 Brier {best['stage1_brier']:.3f}, "
            f"stage-2 positive MAE {best['stage2_positive_mae']:.3f})."
        ),
        "",
        "## Interpretation",
        "",
        "- This annual sample is very small; treat all apparent edges as",
        "  fragile and hypothesis-generating.",
        "- Ridge conditional-size predictions are capped to the prior",
        "  training-fold positive excess range to avoid false precision from",
        "  the tiny positive-only sample.",
        "- x6 should remain complementary to the BVPS/capital-generation",
        "  research lane until a later synthesis step.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, metrics_df, summary = run_x6_experiments()
    write_artifacts(detail, metrics_df, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
