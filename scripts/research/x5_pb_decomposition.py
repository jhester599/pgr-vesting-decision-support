"""Run x5 P/B leg and recombined decomposition benchmarks."""

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
)
from src.research.x5_pb_decomposition import (
    combine_decomposition_predictions,
    evaluate_pb_baseline,
    evaluate_pb_regressor,
    summarize_decomposition_results,
)

OUTPUT_DIR = Path("results") / "research"
PB_DETAIL_PATH = OUTPUT_DIR / "x5_pb_leg_detail.csv"
DECOMP_DETAIL_PATH = OUTPUT_DIR / "x5_decomposition_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x5_decomposition_summary.json"
MEMO_PATH = OUTPUT_DIR / "x5_research_memo.md"
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)
BVPS_SPECS: tuple[tuple[str, str, str], ...] = (
    ("no_change_bvps", "baseline", "growth"),
    ("drift_bvps_growth", "baseline", "growth"),
    ("ridge_bvps_growth", "model", "growth"),
    ("hist_gbt_bvps_growth", "model", "growth"),
)
PB_SPECS: tuple[tuple[str, str, str], ...] = (
    ("no_change_pb", "baseline", "pb"),
    ("drift_pb", "baseline", "pb"),
    ("ridge_log_pb", "model", "log_pb"),
    ("hist_gbt_log_pb", "model", "log_pb"),
)


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x5 reads this cache "
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
        "pb_ratio",
        "pgr_price_to_book_relative",
        "book_value_per_share_growth_yoy",
        "roe_net_income_ttm",
        "pgr_premium_to_surplus",
        "combined_ratio_ttm",
        "underwriting_margin_ttm",
        "npw_growth_yoy",
        "investment_income_growth_yoy",
        "unrealized_gain_pct_equity",
        "buyback_yield",
        "real_rate_10y",
        "credit_spread_hy",
        "vix",
        "pgr_vs_peers_6m",
        "pgr_vs_vfh_6m",
    ]
    available = set(get_feature_columns(feature_df))
    return [feature for feature in preferred if feature in available]


def _target_column(horizon: int, target_kind: str) -> str:
    if target_kind == "pb":
        return f"target_{horizon}m_pb"
    if target_kind == "log_pb":
        return f"target_{horizon}m_log_pb"
    raise ValueError(f"Unsupported target_kind '{target_kind}'.")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def _load_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return feature_df, targets


def run_x5_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run x5 P/B leg and recombined decomposition rows."""
    feature_df, targets = _load_targets()
    X_features = feature_df[get_feature_columns(feature_df)]
    feature_columns = _feature_subset(feature_df)

    pb_rows: list[dict[str, Any]] = []
    decomp_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        current_bvps = targets["current_bvps"]
        current_pb = targets["current_pb"]
        bvps_predictions: dict[str, pd.DataFrame] = {}
        pb_predictions: dict[str, pd.DataFrame] = {}

        for model_name, spec_kind, target_kind in BVPS_SPECS:
            y = targets[f"target_{horizon}m_bvps_growth"].rename(
                f"target_{horizon}m_bvps_growth"
            )
            if spec_kind == "baseline":
                predictions, _ = evaluate_bvps_baseline(
                    X_features,
                    y,
                    current_bvps=current_bvps,
                    baseline_name=model_name,
                    target_kind=target_kind,
                    target_horizon_months=horizon,
                )
            else:
                predictions, _ = evaluate_bvps_regressor(
                    X_features,
                    y,
                    current_bvps=current_bvps,
                    model_name=model_name,
                    feature_columns=feature_columns,
                    target_kind=target_kind,
                    target_horizon_months=horizon,
                )
            bvps_predictions[model_name] = predictions

        for model_name, spec_kind, target_kind in PB_SPECS:
            y = targets[_target_column(horizon, target_kind)].rename(
                _target_column(horizon, target_kind)
            )
            if spec_kind == "baseline":
                predictions, metrics = evaluate_pb_baseline(
                    X_features,
                    y,
                    current_pb=current_pb,
                    baseline_name=model_name,
                    target_kind=target_kind,
                    target_horizon_months=horizon,
                )
            else:
                predictions, metrics = evaluate_pb_regressor(
                    X_features,
                    y,
                    current_pb=current_pb,
                    model_name=model_name,
                    feature_columns=feature_columns,
                    target_kind=target_kind,
                    target_horizon_months=horizon,
                )
            pb_rows.append(metrics)
            pb_predictions[model_name] = predictions

        for bvps_name, bvps_pred in bvps_predictions.items():
            for pb_name, pb_pred in pb_predictions.items():
                _, metrics = combine_decomposition_predictions(
                    bvps_pred,
                    pb_pred,
                    horizon_months=horizon,
                    bvps_model_name=bvps_name,
                    pb_model_name=pb_name,
                )
                decomp_rows.append(metrics)

    pb_detail = pd.DataFrame(pb_rows)
    decomp_detail = pd.DataFrame(decomp_rows)
    summary = summarize_decomposition_results(decomp_detail)
    return pb_detail, decomp_detail, summary


def write_artifacts(
    pb_detail: pd.DataFrame,
    decomp_detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write x5 CSV, JSON, and memo artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pb_detail.to_csv(PB_DETAIL_PATH, index=False)
    decomp_detail.to_csv(DECOMP_DETAIL_PATH, index=False)
    payload = {
        "version": "x5",
        "artifact_classification": "research",
        "production_changes": False,
        "ranking_basis": (
            "implied_price_mae ascending, implied_price_rmse ascending, "
            "directional_hit_rate descending"
        ),
        "bvps_specs": [spec[0] for spec in BVPS_SPECS],
        "pb_specs": [spec[0] for spec in PB_SPECS],
        "horizons": list(HORIZONS),
        "ranked_rows": _json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(pb_detail, decomp_detail, summary)


def write_memo(
    pb_detail: pd.DataFrame,
    decomp_detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Write a compact human-readable x5 memo."""
    lines = [
        "# x5 Research Memo",
        "",
        "## Scope",
        "",
        "x5 runs the research-only future P/B leg and recombined BVPS x",
        "P/B implied-price benchmark. It does not alter production artifacts.",
        "",
        "## Recombined Results By Horizon",
        "",
    ]
    for horizon in HORIZONS:
        horizon_rows = summary[summary["horizon_months"] == horizon]
        if horizon_rows.empty:
            continue
        best = horizon_rows.iloc[0]
        lines.append(
            f"- {horizon}m best decomposition row: `{best['model_name']}` "
            f"(price MAE {best['implied_price_mae']:.3f}, RMSE "
            f"{best['implied_price_rmse']:.3f}, hit rate "
            f"{best['directional_hit_rate']:.3f})."
        )
    lines.extend(
        [
            "",
            "## P/B Leg Takeaway",
            "",
        ]
    )
    for horizon in HORIZONS:
        pb_rows = pb_detail[pb_detail["horizon_months"] == horizon]
        if pb_rows.empty:
            continue
        best_pb = pb_rows.sort_values(
            ["pb_mae", "pb_rmse", "model_name"],
            ascending=[True, True, True],
        ).iloc[0]
        lines.append(
            f"- {horizon}m best P/B row: `{best_pb['model_name']}` "
            f"({best_pb['target_kind']}, P/B MAE {best_pb['pb_mae']:.3f}, "
            f"RMSE {best_pb['pb_rmse']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- x5 is still research-only and stacked on x4 while x4 is open.",
            "- Compare x5 against x3 direct return after both PRs are merged",
            "  and the x-series artifacts share one base branch.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    pb_detail, decomp_detail, summary = run_x5_experiments()
    write_artifacts(pb_detail, decomp_detail, summary)
    print(f"Wrote {PB_DETAIL_PATH}")
    print(f"Wrote {DECOMP_DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
