"""Run x4 BVPS forecasting benchmarks."""

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
from src.research.x1_targets import build_decomposition_targets
from src.research.x4_bvps_forecasting import (
    evaluate_bvps_baseline,
    evaluate_bvps_regressor,
    normalize_bvps_monthly,
    summarize_bvps_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x4_bvps_forecasting_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x4_bvps_forecasting_summary.json"
MEMO_PATH = OUTPUT_DIR / "x4_research_memo.md"
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)
BASELINE_NAMES: tuple[str, ...] = ("no_change_bvps", "drift_bvps_growth")
MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("ridge_bvps_growth", "growth"),
    ("ridge_log_bvps_growth", "log_growth"),
    ("hist_gbt_bvps_growth", "growth"),
    ("hist_gbt_log_bvps_growth", "log_growth"),
)


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x4 reads this cache "
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
    """Return an insurer-specific starting feature set for x4 BVPS research."""
    preferred = [
        "book_value_per_share_growth_yoy",
        "pb_ratio",
        "roe_net_income_ttm",
        "roe_trend",
        "pgr_premium_to_surplus",
        "reserve_to_npe_ratio",
        "combined_ratio_ttm",
        "underwriting_margin_ttm",
        "underwriting_income_growth_yoy",
        "npw_growth_yoy",
        "unearned_premium_growth_yoy",
        "pif_growth_yoy",
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


def _target_column(horizon: int, target_kind: str) -> str:
    if target_kind == "growth":
        return f"target_{horizon}m_bvps_growth"
    if target_kind == "log_growth":
        return f"target_{horizon}m_log_bvps_growth"
    raise ValueError(f"Unsupported target_kind '{target_kind}'.")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def run_x4_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run x4 BVPS model and baseline rows for all horizons."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        prices = db_client.get_prices(conn, "PGR")
        pgr_monthly = db_client.get_pgr_edgar_monthly(conn)
    finally:
        conn.close()

    monthly_close = _month_end_close(prices)
    monthly_bvps = normalize_bvps_monthly(
        pgr_monthly,
        filing_lag_months=config.EDGAR_FILING_LAG_MONTHS,
    )
    targets = build_decomposition_targets(
        monthly_close,
        monthly_bvps,
        horizons=HORIZONS,
    )
    feature_columns = _feature_subset(feature_df)
    X_features = feature_df[get_feature_columns(feature_df)]

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        current_bvps = targets["current_bvps"]
        for target_kind in ("growth", "log_growth"):
            y = targets[_target_column(horizon, target_kind)].rename(
                _target_column(horizon, target_kind)
            )
            for baseline_name in BASELINE_NAMES:
                _, metrics = evaluate_bvps_baseline(
                    X_features,
                    y,
                    current_bvps=current_bvps,
                    baseline_name=baseline_name,
                    target_kind=target_kind,
                    target_horizon_months=horizon,
                )
                rows.append(metrics)
        for model_name, target_kind in MODEL_SPECS:
            y = targets[_target_column(horizon, target_kind)].rename(
                _target_column(horizon, target_kind)
            )
            _, metrics = evaluate_bvps_regressor(
                X_features,
                y,
                current_bvps=current_bvps,
                model_name=model_name,
                feature_columns=feature_columns,
                target_kind=target_kind,
                target_horizon_months=horizon,
            )
            rows.append(metrics)

    detail = pd.DataFrame(rows)
    summary = summarize_bvps_results(detail)
    return detail, summary


def write_artifacts(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x4 CSV, JSON, and memo artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x4",
        "artifact_classification": "research",
        "production_changes": False,
        "ranking_basis": (
            "future_bvps_mae ascending, then growth_rmse ascending, then "
            "directional_hit_rate descending; beats_no_change_bvps must be "
            "true before treating a model as an edge"
        ),
        "models": [model_name for model_name, _ in MODEL_SPECS],
        "baselines": list(BASELINE_NAMES),
        "horizons": list(HORIZONS),
        "ranked_rows": _json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(detail, summary)


def write_memo(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write a compact human-readable x4 memo."""
    lines = [
        "# x4 Research Memo",
        "",
        "## Scope",
        "",
        "x4 runs the research-only BVPS forecasting leg for the future",
        "BVPS x P/B decomposition benchmark. It does not forecast P/B,",
        "recombine implied price, or alter production artifacts.",
        "",
        "## Results By Horizon",
        "",
    ]
    for horizon in HORIZONS:
        horizon_rows = summary[summary["horizon_months"] == horizon]
        if horizon_rows.empty:
            continue
        best = horizon_rows.iloc[0]
        no_change = detail[
            (detail["horizon_months"] == horizon)
            & (detail["model_name"] == "no_change_bvps")
            & (detail["target_kind"] == "growth")
        ].iloc[0]
        gate_label = (
            "cleared no-change BVPS gate"
            if bool(best["beats_no_change_bvps"])
            else "did not clear no-change BVPS gate"
        )
        lines.append(
            f"- {horizon}m best BVPS-MAE row: `{best['model_name']}` "
            f"({best['target_kind']}, BVPS MAE "
            f"{best['future_bvps_mae']:.3f}, growth RMSE "
            f"{best['growth_rmse']:.4f}, hit rate "
            f"{best['directional_hit_rate']:.3f}); no-change BVPS MAE "
            f"{no_change['future_bvps_mae']:.3f}, growth RMSE "
            f"{no_change['growth_rmse']:.4f}; {gate_label}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- x4 isolates the BVPS leg; x5 must test whether P/B forecasting",
            "  and recombination improve implied-price accuracy.",
            "- BVPS targets use PGR monthly EDGAR BVPS normalized to the",
            "  feature matrix's lagged business-month-end availability",
            "  calendar.",
            "- Treat BVPS model edges as structural inputs, not trading signals,",
            "  until the P/B leg and recombined price benchmark exist.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, summary = run_x4_experiments()
    write_artifacts(detail, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
