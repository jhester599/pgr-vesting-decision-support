"""Run x2 absolute PGR direction classification baselines."""

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
from src.research.x2_absolute_classification import (
    evaluate_absolute_baseline,
    evaluate_absolute_classifier,
    summarize_absolute_classification_results,
)

OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "x2_absolute_classification_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "x2_absolute_classification_summary.json"
MEMO_PATH = OUTPUT_DIR / "x2_research_memo.md"
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)
MODEL_NAMES: tuple[str, ...] = ("logistic_l2_balanced", "hist_gbt_depth2")
BASELINE_NAMES: tuple[str, ...] = ("base_rate", "always_up")


def _load_feature_matrix_read_only() -> pd.DataFrame:
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature matrix cache at {path}. x2 reads this cache "
            "without refreshing it to preserve the research-only boundary."
        )
    return pd.read_parquet(path)


def _month_end_close(prices: pd.DataFrame) -> pd.Series:
    close = prices["close"].copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    result = close.resample("ME").last()
    result.name = "close_price"
    return result


def _feature_subset(feature_df: pd.DataFrame) -> list[str]:
    """Return a conservative starting feature set for x2."""
    preferred = [
        "mom_3m",
        "mom_6m",
        "mom_12m",
        "vol_63d",
        "high_52w",
        "pb_ratio",
        "pe_ratio",
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


def run_x2_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run x2 model and baseline rows for all horizons."""
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
        y = targets[f"target_{horizon}m_up"].rename(f"target_{horizon}m_up")
        X = feature_df.join(y, how="inner")
        X_features = X[get_feature_columns(feature_df)]
        y_aligned = X[y.name]
        for baseline_name in BASELINE_NAMES:
            _, metrics = evaluate_absolute_baseline(
                X_features,
                y_aligned,
                baseline_name=baseline_name,
                target_horizon_months=horizon,
            )
            rows.append(metrics)
        for model_name in MODEL_NAMES:
            _, metrics = evaluate_absolute_classifier(
                X_features,
                y_aligned,
                model_name=model_name,
                feature_columns=feature_columns,
                target_horizon_months=horizon,
            )
            rows.append(metrics)

    detail = pd.DataFrame(rows)
    summary = summarize_absolute_classification_results(detail)
    return detail, summary


def write_artifacts(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write x2 CSV, JSON, and memo artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False)
    payload = {
        "version": "x2",
        "artifact_classification": "research",
        "production_changes": False,
        "ranking_basis": (
            "balanced_accuracy descending, then brier_score/log_loss ascending; "
            "beats_base_rate must be true before treating a model as an edge"
        ),
        "models": list(MODEL_NAMES),
        "baselines": list(BASELINE_NAMES),
        "horizons": list(HORIZONS),
        "ranked_rows": summary.to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_memo(detail, summary)


def write_memo(detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write a compact human-readable x2 memo."""
    lines = [
        "# x2 Research Memo",
        "",
        "## Scope",
        "",
        "x2 runs the first research-only absolute PGR direction classification",
        "baseline. It does not alter production or monthly shadow artifacts.",
        "",
        "## Results By Horizon",
        "",
    ]
    for horizon in HORIZONS:
        horizon_rows = summary[summary["horizon_months"] == horizon]
        if horizon_rows.empty:
            continue
        best = horizon_rows.iloc[0]
        base_rate = detail[
            (detail["horizon_months"] == horizon)
            & (detail["model_name"] == "base_rate")
        ].iloc[0]
        gate_label = (
            "cleared base-rate gate"
            if bool(best["beats_base_rate"])
            else "did not clear base-rate gate"
        )
        lines.extend(
            [
                (
                    f"- {horizon}m best balanced-accuracy row: "
                    f"`{best['model_name']}` "
                    f"(BA {best['balanced_accuracy']:.3f}, "
                    f"Brier {best['brier_score']:.3f}); "
                    f"base-rate BA {base_rate['balanced_accuracy']:.3f}, "
                    f"Brier {base_rate['brier_score']:.3f}; "
                    f"{gate_label}."
                )
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Treat any apparent edge as preliminary until x3 return-regression",
            "  and x4/x5 decomposition benchmarks exist.",
            "- x2 uses horizon-specific WFO gaps and fold-local preprocessing.",
            "- x7 should revisit TA only as a bounded follow-up, not as broad",
            "  indicator dumping.",
        ]
    )
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail, summary = run_x2_experiments()
    write_artifacts(detail, summary)
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
