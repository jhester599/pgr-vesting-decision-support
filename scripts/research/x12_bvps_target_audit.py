"""Run x12 BVPS raw-vs-adjusted target audit."""

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
from src.research.x12_bvps_target_audit import (
    build_adjusted_bvps_targets,
    build_monthly_dividend_series,
    identify_bvps_discontinuities,
    json_records,
    summarize_x12_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x12_bvps_target_audit_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x12_bvps_target_audit_summary.json"
DISCONTINUITY_PATH = OUTPUT_DIR / "x12_bvps_discontinuities.csv"
MEMO_PATH = OUTPUT_DIR / "x12_research_memo.md"
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
            f"Missing feature matrix cache at {path}. x12 reads this cache "
            "without refreshing it to preserve the research-only boundary."
        )
    return pd.read_parquet(path)


def _build_x12_feature_matrix(
    feature_df: pd.DataFrame,
    current_bvps: pd.Series,
) -> pd.DataFrame:
    base = feature_df[get_feature_columns(feature_df)].copy()
    bridge = build_bvps_bridge_features(base, current_bvps)
    interactions = build_bvps_interactions(bridge)
    return bridge.join(interactions, how="left")


def run_x12_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run raw vs adjusted BVPS target audit experiments."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        dividends = db_client.get_dividends(conn, "PGR")
        pgr_monthly = db_client.get_pgr_edgar_monthly(conn)
    finally:
        conn.close()

    current_bvps = normalize_bvps_monthly(
        pgr_monthly,
        filing_lag_months=config.EDGAR_FILING_LAG_MONTHS,
    )
    monthly_dividends = build_monthly_dividend_series(dividends, current_bvps.index)
    targets = build_adjusted_bvps_targets(
        current_bvps,
        monthly_dividends,
        horizons=HORIZONS,
    )
    discontinuities = identify_bvps_discontinuities(
        current_bvps,
        monthly_dividends,
    )
    X = _build_x12_feature_matrix(feature_df, current_bvps)
    blocks = build_x9_feature_blocks(X)

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        target_map = {
            "raw": f"target_{horizon}m_bvps_growth",
            "adjusted": f"target_{horizon}m_adjusted_bvps_growth",
        }
        for target_variant, target_column in target_map.items():
            y = targets[target_column].rename(target_column)
            for baseline in BASELINES:
                _, metrics = evaluate_x9_bvps_baseline(
                    X,
                    y,
                    current_bvps=current_bvps,
                    baseline_name=baseline,
                    target_kind="growth",
                    target_horizon_months=horizon,
                )
                metrics["target_variant"] = target_variant
                metrics["feature_block"] = "baseline"
                rows.append(metrics)
            for block in blocks:
                if not block.feature_columns:
                    continue
                for model_name in MODELS:
                    _, metrics, _ = evaluate_x9_bvps_model(
                        X,
                        y,
                        current_bvps=current_bvps,
                        model_name=model_name,
                        feature_columns=block.feature_columns,
                        target_kind="growth",
                        target_horizon_months=horizon,
                    )
                    metrics["target_variant"] = target_variant
                    metrics["feature_block"] = block.block_name
                    rows.append(metrics)

    detail = pd.DataFrame(rows)
    summary = summarize_x12_results(detail)
    return detail, summary, discontinuities


def write_artifacts(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    discontinuities: pd.DataFrame,
) -> None:
    """Write x12 artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    discontinuities.to_csv(DISCONTINUITY_PATH, index=False)
    payload = {
        "version": "x12",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "horizons": list(HORIZONS),
        "baselines": list(BASELINES),
        "models": list(MODELS),
        "discontinuity_count": int(len(discontinuities)),
        "ranked_rows": json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(summary, discontinuities)


def write_memo(summary: pd.DataFrame, discontinuities: pd.DataFrame) -> None:
    """Write compact x12 memo."""
    lines = [
        "# x12 Research Memo",
        "",
        "## Scope",
        "",
        "x12 audits whether raw BVPS targets are being distorted by capital",
        "return events. It compares raw vs dividend-adjusted BVPS targets",
        "using bounded x9-style baselines and regularized models.",
        "",
        "## Discontinuities",
        "",
        f"- Flagged discontinuity months: {len(discontinuities)}.",
    ]
    if not discontinuities.empty:
        dec_jan_share = float(
            discontinuities["date"].apply(pd.Timestamp).dt.month.isin([12, 1]).mean()
        )
        lines.append(
            f"- Share of discontinuities in December/January: {dec_jan_share:.0%}."
        )
    lines.extend(["", "## Raw vs Adjusted Leaders", ""])
    for horizon in HORIZONS:
        raw_rows = summary[
            (summary["horizon_months"] == horizon)
            & (summary["target_variant"] == "raw")
        ]
        adj_rows = summary[
            (summary["horizon_months"] == horizon)
            & (summary["target_variant"] == "adjusted")
        ]
        if raw_rows.empty or adj_rows.empty:
            continue
        raw_best = raw_rows.iloc[0]
        adj_best = adj_rows.iloc[0]
        lines.append(
            f"- {horizon}m raw best `{raw_best['model_name']}` "
            f"({raw_best['feature_block']}, MAE {raw_best['future_bvps_mae']:.3f}) "
            f"vs adjusted best `{adj_best['model_name']}` "
            f"({adj_best['feature_block']}, MAE {adj_best['future_bvps_mae']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If adjusted targets help most at longer horizons, the next",
            "  logical step is an adjusted BVPS x P/B recombination pass.",
            "- If adjusted targets do not help much, the remaining weakness is",
            "  more likely target-regime or model-specification driven than",
            "  pure dividend discontinuity noise.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, summary, discontinuities = run_x12_experiments()
    write_artifacts(detail, summary, discontinuities)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {DISCONTINUITY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
