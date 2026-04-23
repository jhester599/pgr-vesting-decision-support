"""Run x7 targeted TA replacement experiments for x-series classification."""

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
from src.research.v160_ta_features import build_ta_feature_matrix
from src.research.x1_targets import build_forward_return_targets
from src.research.x2_absolute_classification import evaluate_absolute_classifier
from src.research.x7_targeted_ta import (
    apply_feature_swaps,
    attach_baseline_deltas,
    build_x7_ta_variants,
    summarize_ta_variants,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x7_targeted_ta_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x7_targeted_ta_summary.json"
MEMO_PATH = OUTPUT_DIR / "x7_research_memo.md"
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)
TA_TICKERS: tuple[str, ...] = ("PGR", "VWO", "VOO")


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x7 reads this cache "
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


def _price_map(conn: object) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for ticker in TA_TICKERS:
        prices = db_client.get_prices(conn, ticker)
        if not prices.empty:
            result[ticker] = prices
    return result


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    return cleaned.to_dict(orient="records")


def run_x7_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run targeted TA replacement experiments for x2-style classifiers."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        prices = db_client.get_prices(conn, "PGR")
        ta_features = build_ta_feature_matrix(
            _price_map(conn),
            benchmarks=("VWO", "VOO"),
            peer_tickers=[],
        )
    finally:
        conn.close()

    feature_df = feature_df.join(ta_features, how="left")
    monthly_close = _month_end_close(prices)
    targets = build_forward_return_targets(monthly_close, horizons=HORIZONS)
    baseline_features = _feature_subset(feature_df)
    variants = build_x7_ta_variants()

    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        y = targets[f"target_{horizon}m_up"].rename(f"target_{horizon}m_up")
        X = feature_df.join(y, how="inner")
        X_features = X[get_feature_columns(feature_df)]
        y_aligned = X[y.name]
        for variant in variants:
            selected = apply_feature_swaps(
                baseline_features,
                variant.feature_swaps,
            )
            selected = [
                feature for feature in selected if feature in X_features.columns
            ]
            _, metrics = evaluate_absolute_classifier(
                X_features,
                y_aligned,
                model_name="logistic_l2_balanced",
                feature_columns=selected,
                target_horizon_months=horizon,
            )
            rows.append(
                {
                    "variant": variant.variant,
                    "experiment_mode": variant.experiment_mode,
                    "feature_swaps": json.dumps(
                        variant.feature_swaps,
                        sort_keys=True,
                    ),
                    "feature_columns": "|".join(selected),
                    "notes": variant.notes,
                    **metrics,
                }
            )

    detail = attach_baseline_deltas(pd.DataFrame(rows))
    summary = summarize_ta_variants(detail)
    return detail, summary


def write_artifacts(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x7 CSV, JSON, and memo artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x7",
        "artifact_classification": "research",
        "production_changes": False,
        "shadow_changes": False,
        "recommendation_rule": (
            "A TA variant must improve balanced accuracy and Brier score "
            "versus the x2 core baseline for a horizon to clear the x7 gate."
        ),
        "ranked_rows": _json_records(summary),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(summary)


def write_memo(summary: pd.DataFrame) -> None:
    """Write a compact x7 memo."""
    lines = [
        "# x7 Research Memo",
        "",
        "## Scope",
        "",
        "x7 runs targeted, replacement-only TA experiments for x-series",
        "absolute-direction classification. It does not add broad TA features",
        "or alter production/monthly/shadow artifacts.",
        "",
        "## Results",
        "",
    ]
    if summary.empty:
        lines.append("- No x7 variants produced evaluable rows.")
    else:
        for _, row in summary.iterrows():
            lines.append(
                f"- `{row['variant']}` cleared "
                f"{int(row['cleared_horizon_count'])}/"
                f"{int(row['tested_horizon_count'])} horizons "
                f"(mean delta BA "
                f"{row['mean_delta_balanced_accuracy']:.3f}, "
                f"mean delta Brier {row['mean_delta_brier_score']:.3f})."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Treat TA as a bounded replacement experiment, not as additive",
            "  indicator expansion.",
            "- x8 should compare this x7 evidence against x2/x3/x4/x5/x6",
            "  before any shadow-readiness recommendation.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, summary = run_x7_experiments()
    write_artifacts(detail, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
