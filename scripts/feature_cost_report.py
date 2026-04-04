"""v9.1 feature cost, missingness, and coverage audit."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from src.database import db_client
from src.processing.feature_engineering import build_feature_matrix_from_db, get_feature_columns


DEFAULT_OUTPUT_DIR = os.path.join("results", "v9")


def _infer_feature_family(feature_name: str) -> str:
    """Best-effort feature family classification for audit output."""
    if feature_name.startswith("mom_") or feature_name.startswith("vol_") or feature_name in {
        "high_52w",
        "pgr_vs_kie_6m",
        "pgr_vs_peers_6m",
        "pgr_vs_vfh_6m",
    }:
        return "price_relative"
    if feature_name in {"pe_ratio", "pb_ratio", "roe"}:
        return "fundamental"
    if any(
        token in feature_name
        for token in (
            "combined_ratio",
            "pif_",
            "gainshare",
            "channel_mix",
            "npw_",
            "underwriting",
            "unearned",
            "investment_",
            "buyback",
            "roe_",
            "cr_acceleration",
        )
    ):
        return "edgar_company"
    if feature_name in {
        "yield_slope",
        "yield_curvature",
        "real_rate_10y",
        "credit_spread_hy",
        "nfci",
        "vix",
        "vmt_yoy",
        "used_car_cpi_yoy",
        "medical_cpi_yoy",
        "ppi_auto_ins_yoy",
    }:
        return "macro"
    return "other"


def build_feature_cost_report(df: pd.DataFrame) -> pd.DataFrame:
    """Create the per-feature coverage and stability audit table."""
    if df.empty:
        return pd.DataFrame()

    n_rows = len(df)
    trailing_window = min(60, n_rows)
    feature_rows: list[dict[str, Any]] = []
    for feature in get_feature_columns(df):
        series = df[feature]
        non_null = series.dropna()
        production_models = sorted(
            [
                model_type
                for model_type, feature_list in config.MODEL_FEATURE_OVERRIDES.items()
                if feature in feature_list
            ]
        )
        feature_rows.append(
            {
                "feature": feature,
                "family": _infer_feature_family(feature),
                "n_total_rows": n_rows,
                "n_non_null": int(non_null.shape[0]),
                "completeness_pct": float(non_null.shape[0] / n_rows),
                "trailing_60m_completeness_pct": float(series.tail(trailing_window).notna().mean()),
                "first_available": non_null.index.min().date().isoformat() if not non_null.empty else "",
                "last_available": non_null.index.max().date().isoformat() if not non_null.empty else "",
                "n_unique": int(non_null.nunique()),
                "std": float(non_null.std()) if not non_null.empty else float("nan"),
                "mean_abs": float(non_null.abs().mean()) if not non_null.empty else float("nan"),
                "in_production_model": bool(production_models),
                "production_models": ",".join(production_models),
            }
        )
    return pd.DataFrame(feature_rows).sort_values(
        by=["in_production_model", "completeness_pct", "feature"],
        ascending=[False, False, True],
    )


def build_feature_set_health(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage and dimensionality for each production feature set."""
    rows: list[dict[str, Any]] = []
    for model_type, feature_list in config.MODEL_FEATURE_OVERRIDES.items():
        present = [feature for feature in feature_list if feature in df.columns]
        if present:
            subset = df[present]
            fully_observed = int(subset.dropna(how="any").shape[0])
            completeness = float(subset.notna().all(axis=1).mean())
            earliest_full = ""
            full_rows = subset.dropna(how="any")
            if not full_rows.empty:
                earliest_full = full_rows.index.min().date().isoformat()
        else:
            fully_observed = 0
            completeness = float("nan")
            earliest_full = ""
        rows.append(
            {
                "model_type": model_type,
                "n_features": len(present),
                "n_fully_observed_rows": fully_observed,
                "full_row_completeness_pct": completeness,
                "earliest_full_observation": earliest_full,
                "features": ",".join(present),
            }
        )
    return pd.DataFrame(rows).sort_values(by="model_type")


def run_feature_cost_report(
    conn: Any,
    output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and persist the v9.1 feature audit outputs."""
    df = build_feature_matrix_from_db(conn)
    feature_df = build_feature_cost_report(df)
    feature_set_df = build_feature_set_health(df)

    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    feature_path = os.path.join(output_dir, f"feature_cost_report_{stamp}.csv")
    feature_set_path = os.path.join(output_dir, f"feature_set_health_{stamp}.csv")
    feature_df.to_csv(feature_path, index=False)
    feature_set_df.to_csv(feature_set_path, index=False)
    print(f"Wrote feature audit to {feature_path}")
    print(f"Wrote feature-set health to {feature_set_path}")
    return feature_df, feature_set_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v9 feature cost audit.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    conn = db_client.get_connection(config.DB_PATH)
    run_feature_cost_report(
        conn=conn,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
