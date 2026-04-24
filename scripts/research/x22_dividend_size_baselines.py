"""Run x22 dividend size baseline challengers."""

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
from src.research.x19_post_policy_dividend_model import filter_post_policy_annual_frame
from src.research.x22_dividend_size_baselines import (
    candidate_target_scales,
    evaluate_x22_baseline,
    summarize_x22_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x22_dividend_size_baselines_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x22_dividend_size_baselines_summary.json"
MEMO_PATH = OUTPUT_DIR / "x22_research_memo.md"
BASELINE_MODES: tuple[str, ...] = (
    "historical_mean",
    "prior_positive_year",
    "trailing_2_mean",
    "trailing_2_median",
)


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x22 reads this cache "
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


def _build_annual_frame() -> pd.DataFrame:
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
    from src.research.x21_dividend_target_scales import build_scaled_size_targets

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
    return build_scaled_size_targets(annual)


def _x21_candidate_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = json.loads((OUTPUT_DIR / "x21_dividend_target_scales_summary.json").read_text(encoding="utf-8"))
    detail = pd.read_csv(OUTPUT_DIR / "x21_dividend_target_scales_detail.csv")
    candidate_names = set(candidate_target_scales().keys())
    metrics = pd.DataFrame(payload["metrics"])
    metrics = metrics[metrics["target_scale"].isin(candidate_names)].copy()
    detail = detail[detail["target_scale"].isin(candidate_names)].copy()
    return detail, metrics


def run_x22_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run x22 baseline challengers against x21 candidate rows."""
    annual = _build_annual_frame()
    x21_detail, x21_metrics = _x21_candidate_metrics()
    detail_rows: list[pd.DataFrame] = [x21_detail]
    metric_rows: list[pd.DataFrame] = [x21_metrics]
    for target_scale_name, (target_column, scale_column) in candidate_target_scales().items():
        for mode in BASELINE_MODES:
            detail, metrics = evaluate_x22_baseline(
                annual,
                target_scale_name=target_scale_name,
                target_column=target_column,
                scale_column=scale_column,
                mode=mode,
                min_train_years=3,
            )
            detail = detail.copy()
            detail["feature_set"] = "baseline_only"
            detail_rows.append(detail)
            metric_rows.append(pd.DataFrame([{**metrics, "feature_set": "baseline_only"}]))
    detail_df = pd.concat(detail_rows, ignore_index=True)
    metrics_df = pd.concat(metric_rows, ignore_index=True)
    summary = summarize_x22_results(metrics_df)
    return detail_df, metrics_df, summary


def write_artifacts(detail: pd.DataFrame, metrics_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x22 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x22",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "candidate_target_scales": list(candidate_target_scales().keys()),
        "metrics": _json_records(metrics_df),
        "ranked_rows": _json_records(summary),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_memo(summary)


def write_memo(summary: pd.DataFrame) -> None:
    """Write a compact x22 memo."""
    best = summary.iloc[0]
    lines = [
        "# x22 Research Memo",
        "",
        "## Scope",
        "",
        "x22 challenges the x21 target-scale result with harder annual",
        "baselines so we can tell whether the gain is mostly from scaling or",
        "from the feature-driven size model.",
        "",
        "## Results",
        "",
        (
            f"- Best row: `{best['feature_set']}` / `{best['target_scale']}` / "
            f"`{best['model_name']}` (dollar MAE {best['dollar_mae']:.3f}, "
            f"scaled MAE {best['scaled_mae']:.3f})."
        ),
        "",
        "## Interpretation",
        "",
        "- If a baseline wins, target scaling matters more than feature depth.",
        "- If an x21 ridge row still wins, the current-BVPS normalization looks",
        "  like a real modeling improvement rather than a cosmetic transform.",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, metrics_df, summary = run_x22_experiments()
    write_artifacts(detail, metrics_df, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
