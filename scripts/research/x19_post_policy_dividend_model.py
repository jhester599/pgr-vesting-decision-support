"""Run x19 post-policy dividend experiments using persistent BVPS features."""

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
from src.research.x6_special_dividend import (
    evaluate_special_dividend_two_stage,
    summarize_special_dividend_results,
)
from src.research.x9_bvps_bridge import build_bvps_bridge_features, build_bvps_interactions
from src.research.x10_dividend_capital import build_dividend_capital_features
from src.research.x17_persistent_bvps import build_persistent_bvps_series
from src.research.x18_dividend_policy_regime import build_regime_aware_dividend_targets
from src.research.x19_post_policy_dividend_model import (
    build_x19_feature_sets,
    filter_post_policy_annual_frame,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x19_post_policy_dividend_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x19_post_policy_dividend_summary.json"
MEMO_PATH = OUTPUT_DIR / "x19_research_memo.md"
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
            f"Missing feature matrix cache at {path}. x19 reads this cache "
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


def _build_monthly_frame(
    feature_df: pd.DataFrame,
    current_bvps: pd.Series,
    persistent_bvps: pd.Series,
    close_price: pd.Series,
) -> pd.DataFrame:
    base = feature_df[get_feature_columns(feature_df)].copy()
    bridge = build_bvps_bridge_features(base, current_bvps)
    interactions = build_bvps_interactions(bridge)
    monthly = bridge.join(interactions, how="left")
    monthly["close_price"] = close_price.reindex(monthly.index)
    monthly["persistent_bvps"] = persistent_bvps.reindex(monthly.index)
    first_persistent_by_year = monthly.groupby(monthly.index.year)["persistent_bvps"].transform("first")
    monthly["persistent_bvps_growth_ytd"] = monthly["persistent_bvps"] / first_persistent_by_year - 1.0
    monthly["persistent_bvps_growth_3m"] = monthly["persistent_bvps"].pct_change(3, fill_method=None)
    monthly["persistent_bvps_yoy_dollar_change"] = monthly["persistent_bvps"].diff(12)
    monthly["persistent_bvps_to_price"] = monthly["persistent_bvps"] / monthly["close_price"].where(
        monthly["close_price"].abs() > 1e-12
    )
    monthly = build_dividend_capital_features(monthly)
    monthly["book_value_creation_proxy"] = (
        monthly["persistent_bvps_growth_ytd"] + monthly["capital_generation_proxy"].fillna(0.0)
    )
    return monthly


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
    from src.research.x12_bvps_target_audit import build_monthly_dividend_series

    monthly_dividends = build_monthly_dividend_series(dividends, current_bvps.index)
    persistent_bvps = build_persistent_bvps_series(current_bvps, monthly_dividends)
    monthly = _build_monthly_frame(feature_df, current_bvps, persistent_bvps, close_price)
    targets = build_regime_aware_dividend_targets(monthly, dividends)
    annual = monthly[monthly.index.month == 11].join(targets, how="inner")
    annual = annual.rename(
        columns={
            "special_dividend_occurred_regime": "special_dividend_occurred",
            "special_dividend_excess_regime": "special_dividend_excess",
        }
    )
    annual = filter_post_policy_annual_frame(annual)
    annual = annual.dropna(subset=["special_dividend_occurred", "special_dividend_excess"])
    return annual, build_x19_feature_sets(annual)


def run_x19_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run x19 post-policy dividend experiments."""
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
                min_train_years=4,
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


def write_artifacts(detail: pd.DataFrame, metrics_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x19 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x19",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "sample_scope": "post_policy_only",
        "prediction_timestamp": "November business month-end",
        "metrics": _json_records(metrics_df),
        "ranked_rows": _json_records(summary),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_memo(detail, summary)


def write_memo(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write a compact x19 memo."""
    rows_per_model_feature = int(detail.groupby(["feature_set", "model_name"]).size().iloc[0])
    best = summary.iloc[0]
    lines = [
        "# x19 Research Memo",
        "",
        "## Scope",
        "",
        "x19 rebuilds the annual dividend sidecar on post-policy snapshots only,",
        "using the x18 regime-aware label and persistent-BVPS-capital features.",
        "",
        "## Sample",
        "",
        f"- OOS annual predictions per model-feature set: {rows_per_model_feature}.",
        "- Validation remains expanding annual train/test splits with post-policy-only data.",
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
        "- x19 is intentionally low-sample and low-confidence.",
        "- The point is to test whether cleaner labels plus persistent-BVPS",
        "  state improve the dividend lane enough to keep pursuing.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, metrics_df, summary = run_x19_experiments()
    write_artifacts(detail, metrics_df, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
