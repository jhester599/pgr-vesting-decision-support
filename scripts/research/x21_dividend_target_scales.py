"""Run x21 post-policy dividend size target-scale experiments."""

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
from src.research.x9_bvps_bridge import build_bvps_bridge_features, build_bvps_interactions
from src.research.x10_dividend_capital import build_dividend_capital_features
from src.research.x17_persistent_bvps import build_persistent_bvps_series
from src.research.x18_dividend_policy_regime import build_regime_aware_dividend_targets
from src.research.x19_post_policy_dividend_model import (
    build_x19_feature_sets,
    filter_post_policy_annual_frame,
)
from src.research.x21_dividend_target_scales import (
    build_scaled_size_targets,
    evaluate_scaled_size_model,
    summarize_scaled_size_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x21_dividend_target_scales_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x21_dividend_target_scales_summary.json"
MEMO_PATH = OUTPUT_DIR / "x21_research_memo.md"
MODEL_NAMES: tuple[str, ...] = ("historical_scaled_mean", "ridge_scaled")
TARGET_SCALES: tuple[tuple[str, str, str], ...] = (
    ("raw_dollars", "target_raw_dollars", "unit_scale"),
    ("to_current_bvps", "target_to_current_bvps", "current_bvps"),
    ("to_persistent_bvps", "target_to_persistent_bvps", "persistent_bvps"),
    ("to_price", "target_to_price", "close_price"),
)


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x21 reads this cache "
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
    annual["unit_scale"] = 1.0
    annual = build_scaled_size_targets(annual)
    return annual, build_x19_feature_sets(annual)


def run_x21_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run x21 post-policy size target-scale experiments."""
    annual, feature_sets = _build_annual_frame()
    detail_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for feature_set, feature_columns in feature_sets.items():
        if not feature_columns:
            continue
        for target_scale_name, target_column, scale_column in TARGET_SCALES:
            for model_name in MODEL_NAMES:
                detail, metrics = evaluate_scaled_size_model(
                    annual,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    scale_column=scale_column,
                    target_scale_name=target_scale_name,
                    model_name=model_name,
                    min_train_years=3,
                )
                detail = detail.copy()
                detail["feature_set"] = feature_set
                detail_rows.append(detail)
                metrics["feature_set"] = feature_set
                metric_rows.append(metrics)
    detail_df = pd.concat(detail_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    summary = summarize_scaled_size_results(metrics_df)
    return detail_df, metrics_df, summary


def write_artifacts(detail: pd.DataFrame, metrics_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x21 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x21",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "sample_scope": "post_policy_positive_only",
        "ranking_basis": "dollar_mae ascending, then scaled_mae",
        "decision_point": {
            "question": "How many initial positive post-policy years should x21 require?",
            "criterion": (
                "Leave enough expanding folds to compare target scales while keeping "
                "the split annual and chronological."
            ),
            "choice": "min_train_years=3 for x21 size-only diagnostics",
        },
        "metrics": _json_records(metrics_df),
        "ranked_rows": _json_records(summary),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_memo(summary)


def write_memo(summary: pd.DataFrame) -> None:
    """Write a compact x21 memo."""
    best = summary.iloc[0]
    lines = [
        "# x21 Research Memo",
        "",
        "## Scope",
        "",
        "x21 compares post-policy dividend size targets after x20 concluded",
        "that occurrence is not identifiable on the current overlap sample.",
        "",
        "## Results",
        "",
        (
            f"- Best row: `{best['feature_set']}` / `{best['target_scale']}` / "
            f"`{best['model_name']}` (dollar MAE {best['dollar_mae']:.3f}, "
            f"scaled MAE {best['scaled_mae']:.3f}, OOS folds {int(best['n_obs'])})."
        ),
        "",
        "## Interpretation",
        "",
        "- x21 is a size-only diagnostic, not a deployable dividend model.",
        "- Rankings are based on dollar error after back-transforming any",
        "  normalized target, so normalized elegance does not get a free pass.",
        "- A surviving normalized target must beat raw dollars on dollar MAE.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, metrics_df, summary = run_x21_experiments()
    write_artifacts(detail, metrics_df, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
