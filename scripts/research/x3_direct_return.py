"""Run x3 direct PGR forward-return regression benchmarks."""

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
from src.research.x1_targets import build_forward_return_targets
from src.research.x3_direct_return import (
    evaluate_direct_return_baseline,
    evaluate_direct_return_regressor,
    summarize_direct_return_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x3_direct_return_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x3_direct_return_summary.json"
MEMO_PATH = OUTPUT_DIR / "x3_research_memo.md"
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)
BASELINE_NAMES: tuple[str, ...] = ("no_change", "drift")
MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("ridge_return", "return"),
    ("ridge_log_return", "log_return"),
    ("hist_gbt_return", "return"),
    ("hist_gbt_log_return", "log_return"),
)


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x3 reads this cache "
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
    """Return the same conservative starting feature set used by x2."""
    preferred = [
        "mom_3m",
        "mom_6m",
        "mom_12m",
        "vol_63d",
        "high_52w",
        "pb_ratio",
        "roe_net_income_ttm",
        "combined_ratio_ttm",
        "monthly_combined_ratio_delta",
        "pif_growth_yoy",
        "gainshare_est",
        "underwriting_income_growth_yoy",
        "book_value_per_share_growth_yoy",
        "buyback_yield",
        "unrealized_gain_pct_equity",
        "yield_slope",
        "real_rate_10y",
        "credit_spread_hy",
        "vix",
        "pgr_vs_peers_6m",
        "pgr_vs_vfh_6m",
    ]
    available = set(get_feature_columns(feature_df))
    return [feature for feature in preferred if feature in available]


def _target_column(horizon: int, target_kind: str) -> str:
    if target_kind == "return":
        return f"target_{horizon}m_return"
    if target_kind == "log_return":
        return f"target_{horizon}m_log_return"
    raise ValueError(f"Unsupported target_kind '{target_kind}'.")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe records with non-finite floats converted to None."""
    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def run_x3_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run x3 model and baseline rows for all horizons."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        prices = db_client.get_prices(conn, "PGR")
    finally:
        conn.close()

    monthly_close = _month_end_close(prices)
    targets = build_forward_return_targets(monthly_close, horizons=HORIZONS)
    feature_columns = _feature_subset(feature_df)

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        X_features = feature_df[get_feature_columns(feature_df)]
        current_price = targets["current_price"]
        for target_kind in ("return", "log_return"):
            y = targets[_target_column(horizon, target_kind)].rename(
                _target_column(horizon, target_kind)
            )
            for baseline_name in BASELINE_NAMES:
                _, metrics = evaluate_direct_return_baseline(
                    X_features,
                    y,
                    current_price=current_price,
                    baseline_name=baseline_name,
                    target_kind=target_kind,
                    target_horizon_months=horizon,
                )
                rows.append(metrics)
        for model_name, target_kind in MODEL_SPECS:
            y = targets[_target_column(horizon, target_kind)].rename(
                _target_column(horizon, target_kind)
            )
            _, metrics = evaluate_direct_return_regressor(
                X_features,
                y,
                current_price=current_price,
                model_name=model_name,
                feature_columns=feature_columns,
                target_kind=target_kind,
                target_horizon_months=horizon,
            )
            rows.append(metrics)

    detail = pd.DataFrame(rows)
    summary = summarize_direct_return_results(detail)
    return detail, summary


def write_artifacts(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x3 CSV, JSON, and memo artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x3",
        "artifact_classification": "research",
        "production_changes": False,
        "ranking_basis": (
            "implied_price_mae ascending, then return_rmse ascending, then "
            "directional_hit_rate descending; beats_no_change must be true "
            "before treating a model as an edge"
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
    """Write a compact human-readable x3 memo."""
    lines = [
        "# x3 Research Memo",
        "",
        "## Scope",
        "",
        "x3 runs the research-only direct forward-return and log-return",
        "regression benchmark for absolute PGR forecasting. It does not alter",
        "production or monthly shadow artifacts.",
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
            & (detail["model_name"] == "no_change")
            & (detail["target_kind"] == "return")
        ].iloc[0]
        gate_label = (
            "cleared no-change gate"
            if bool(best["beats_no_change"])
            else "did not clear no-change gate"
        )
        lines.append(
            f"- {horizon}m best price-MAE row: `{best['model_name']}` "
            f"({best['target_kind']}, price MAE "
            f"{best['implied_price_mae']:.3f}, return RMSE "
            f"{best['return_rmse']:.3f}, hit rate "
            f"{best['directional_hit_rate']:.3f}); no-change price MAE "
            f"{no_change['implied_price_mae']:.3f}, return RMSE "
            f"{no_change['return_rmse']:.3f}; {gate_label}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Treat any apparent direct-return edge as preliminary until x4/x5",
            "  decomposition benchmarks exist.",
            "- x3 uses horizon-specific WFO gaps and fold-local preprocessing.",
            "- The direct-return feature subset excludes `pe_ratio` because the",
            "  cached feature has extreme historical values; valuation exposure",
            "  remains represented by `pb_ratio`.",
            "- Raw future price-level regression remains deferred; x3 derives",
            "  implied future price from return predictions.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, summary = run_x3_experiments()
    write_artifacts(detail, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
