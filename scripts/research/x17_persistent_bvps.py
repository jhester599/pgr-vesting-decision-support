"""Run x17 persistent-BVPS bridge experiments."""

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
from src.research.x9_bvps_bridge import (
    build_bvps_bridge_features,
    build_bvps_interactions,
    build_x9_feature_blocks,
    evaluate_x9_bvps_baseline,
    evaluate_x9_bvps_model,
)
from src.research.x12_bvps_target_audit import build_monthly_dividend_series
from src.research.x17_persistent_bvps import (
    build_persistent_bvps_series,
    build_persistent_bvps_targets,
    json_records,
    summarize_x17_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x17_persistent_bvps_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x17_persistent_bvps_summary.json"
MEMO_PATH = OUTPUT_DIR / "x17_research_memo.md"
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)
MODEL_NAMES: tuple[str, ...] = ("ridge_bridge", "elastic_net_bridge")
BASELINES: tuple[str, ...] = (
    "no_change_bvps",
    "drift_bvps_growth",
    "trailing_3m_growth",
    "seasonal_month_drift",
)


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x17 reads this cache "
            "without refreshing it to preserve the research-only boundary."
        )
    return pd.read_parquet(path)


def _build_feature_matrix(feature_df: pd.DataFrame, current_bvps: pd.Series) -> pd.DataFrame:
    base = feature_df[get_feature_columns(feature_df)].copy()
    bridge = build_bvps_bridge_features(base, current_bvps)
    interactions = build_bvps_interactions(bridge)
    return bridge.join(interactions, how="left")


def run_x17_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run raw-vs-persistent BVPS bridge experiments."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        dividends = db_client.get_dividends(conn, "PGR")
        pgr_monthly = db_client.get_pgr_edgar_monthly(conn)
    finally:
        conn.close()

    raw_bvps = normalize_bvps_monthly(
        pgr_monthly,
        filing_lag_months=config.EDGAR_FILING_LAG_MONTHS,
    )
    monthly_dividends = build_monthly_dividend_series(dividends, raw_bvps.index)
    persistent_bvps = build_persistent_bvps_series(raw_bvps, monthly_dividends)
    raw_targets = build_persistent_bvps_targets(raw_bvps.rename("raw_bvps"), horizons=HORIZONS)
    persistent_targets = build_persistent_bvps_targets(persistent_bvps, horizons=HORIZONS)

    variants = {
        "raw": (raw_bvps, raw_targets, "target_{horizon}m_persistent_bvps_growth"),
        "persistent": (
            persistent_bvps,
            persistent_targets,
            "target_{horizon}m_persistent_bvps_growth",
        ),
    }

    rows: list[dict[str, Any]] = []
    for target_variant, (current_series, targets, target_template) in variants.items():
        X = _build_feature_matrix(feature_df, current_series)
        feature_blocks = build_x9_feature_blocks(X)
        for horizon in HORIZONS:
            y = targets[target_template.format(horizon=horizon)].rename(
                f"{target_variant}_{horizon}m_growth"
            )
            for baseline_name in BASELINES:
                _, metrics = evaluate_x9_bvps_baseline(
                    X,
                    y,
                    current_bvps=current_series,
                    baseline_name=baseline_name,
                    target_kind="growth",
                    target_horizon_months=horizon,
                )
                metrics["target_variant"] = target_variant
                metrics["feature_block"] = "baseline"
                rows.append(metrics)
            for block in feature_blocks:
                for model_name in MODEL_NAMES:
                    _, metrics, _ = evaluate_x9_bvps_model(
                        X,
                        y,
                        current_bvps=current_series,
                        model_name=model_name,
                        feature_columns=block.feature_columns,
                        target_kind="growth",
                        target_horizon_months=horizon,
                    )
                    metrics["target_variant"] = target_variant
                    metrics["feature_block"] = block.block_name
                    rows.append(metrics)

    detail = pd.DataFrame(rows)
    summary = summarize_x17_results(detail)
    return detail, summary


def write_artifacts(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x17 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x17",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "horizons": list(HORIZONS),
        "target_variants": ["raw", "persistent"],
        "ranked_rows": json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(summary)


def write_memo(summary: pd.DataFrame) -> None:
    """Write a compact x17 memo."""
    lines = [
        "# x17 Research Memo",
        "",
        "## Scope",
        "",
        "x17 compares raw BVPS against a dividend-persistent synthetic BVPS",
        "history, asking whether persistent book-value creation is easier",
        "to forecast than the raw book-value level.",
        "",
        "## Results",
        "",
    ]
    for horizon in HORIZONS:
        rows = summary[summary["horizon_months"] == horizon]
        if rows.empty:
            continue
        raw_best = rows[rows["target_variant"] == "raw"].iloc[0]
        persistent_best = rows[rows["target_variant"] == "persistent"].iloc[0]
        lines.append(
            f"- {horizon}m raw best `{raw_best['model_name']}` "
            f"({raw_best['feature_block']}, MAE {raw_best['future_bvps_mae']:.3f}) "
            f"vs persistent best `{persistent_best['model_name']}` "
            f"({persistent_best['feature_block']}, MAE {persistent_best['future_bvps_mae']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If persistent BVPS helps, it supports separating capital",
            "  creation from dividend policy in later dividend work.",
            "- If it does not help, the remaining challenge is more likely",
            "  feature specification or regime change than dividend noise",
            "  alone.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, summary = run_x17_experiments()
    write_artifacts(detail, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
