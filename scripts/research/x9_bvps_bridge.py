"""Run x9 BVPS bridge feature and baseline experiments."""

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
from src.research.x4_bvps_forecasting import normalize_bvps_monthly
from src.research.x9_bvps_bridge import (
    build_bvps_bridge_features,
    build_bvps_interactions,
    build_x9_feature_blocks,
    evaluate_x9_bvps_baseline,
    evaluate_x9_bvps_model,
    summarize_x9_bvps_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x9_bvps_bridge_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x9_bvps_bridge_summary.json"
STABILITY_PATH = OUTPUT_DIR / "x9_bvps_feature_stability.csv"
MEMO_PATH = OUTPUT_DIR / "x9_research_memo.md"
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)
BASELINES: tuple[str, ...] = (
    "no_change_bvps",
    "drift_bvps_growth",
    "trailing_3m_growth",
    "seasonal_month_drift",
)
MODELS: tuple[str, ...] = ("ridge_bridge", "elastic_net_bridge")


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x9 reads this cache "
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


def _target_column(horizon: int) -> str:
    return f"target_{horizon}m_bvps_growth"


def _build_x9_feature_matrix(
    feature_df: pd.DataFrame,
    current_bvps: pd.Series,
) -> pd.DataFrame:
    base = feature_df[get_feature_columns(feature_df)].copy()
    bridge = build_bvps_bridge_features(base, current_bvps)
    interactions = build_bvps_interactions(bridge)
    return bridge.join(interactions, how="left")


def run_x9_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run x9 BVPS bridge experiments."""
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
    current_bvps = targets["current_bvps"]
    X = _build_x9_feature_matrix(feature_df, current_bvps)
    blocks = build_x9_feature_blocks(X)

    rows: list[dict[str, Any]] = []
    stability_rows: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        y = targets[_target_column(horizon)].rename(_target_column(horizon))
        for baseline in BASELINES:
            _, metrics = evaluate_x9_bvps_baseline(
                X,
                y,
                current_bvps=current_bvps,
                baseline_name=baseline,
                target_kind="growth",
                target_horizon_months=horizon,
            )
            metrics["feature_block"] = "baseline"
            rows.append(metrics)
        for block in blocks:
            if not block.feature_columns:
                continue
            for model_name in MODELS:
                _, metrics, stability = evaluate_x9_bvps_model(
                    X,
                    y,
                    current_bvps=current_bvps,
                    model_name=model_name,
                    feature_columns=block.feature_columns,
                    target_kind="growth",
                    target_horizon_months=horizon,
                )
                metrics["feature_block"] = block.block_name
                metrics["feature_block_notes"] = block.notes
                rows.append(metrics)
                if not stability.empty:
                    stability = stability.copy()
                    stability["horizon_months"] = horizon
                    stability["model_name"] = model_name
                    stability["feature_block"] = block.block_name
                    stability_rows.append(stability)

    detail = pd.DataFrame(rows)
    summary = summarize_x9_bvps_results(detail)
    stability = (
        pd.concat(stability_rows, ignore_index=True)
        if stability_rows
        else pd.DataFrame()
    )
    return detail, summary, stability


def write_artifacts(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    """Write x9 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    stability.to_csv(STABILITY_PATH, index=False)
    payload = {
        "version": "x9",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "horizons": list(HORIZONS),
        "baselines": list(BASELINES),
        "models": list(MODELS),
        "ranking_basis": (
            "future_bvps_mae ascending, growth_rmse ascending, "
            "directional_hit_rate descending"
        ),
        "ranked_rows": _json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(summary, stability)


def write_memo(summary: pd.DataFrame, stability: pd.DataFrame) -> None:
    """Write compact x9 memo."""
    lines = [
        "# x9 Research Memo",
        "",
        "## Scope",
        "",
        "x9 tests BVPS bridge baselines, lagged BVPS features, accounting",
        "features, and pre-registered logical interactions. It is",
        "research-only and does not touch production or shadow artifacts.",
        "",
        "## Results By Horizon",
        "",
    ]
    for horizon in HORIZONS:
        rows = summary[summary["horizon_months"] == horizon]
        if rows.empty:
            continue
        best = rows.iloc[0]
        lines.append(
            f"- {horizon}m best: `{best['model_name']}` "
            f"({best['feature_block']}, BVPS MAE "
            f"{best['future_bvps_mae']:.3f}, growth RMSE "
            f"{best['growth_rmse']:.4f}, hit rate "
            f"{best['directional_hit_rate']:.3f})."
        )
    lines.extend(["", "## Feature Stability", ""])
    if stability.empty:
        lines.append("- No coefficient stability rows were produced.")
    else:
        top = stability.sort_values(
            ["selection_rate", "mean_abs_coefficient", "feature"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['feature']}` selected in "
                f"{row['selection_rate']:.0%} of folds for "
                f"{int(row['horizon_months'])}m "
                f"`{row['feature_block']}` / `{row['model_name']}`."
            )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Interactions are bounded and economically pre-registered.",
            "- Feature-count discipline is enforced by reporting stability,",
            "  not by promoting the full candidate set.",
            "- x10 should reuse the most interpretable capital-generation",
            "  features for annual special-dividend testing.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, summary, stability = run_x9_experiments()
    write_artifacts(detail, summary, stability)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {STABILITY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
