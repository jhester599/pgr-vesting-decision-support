"""v58 - Domain-specific FRED features from long-format macro series."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    build_results_df,
    compute_metrics,
    custom_wfo,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    load_research_baseline_results,
    pool_metrics,
    print_delta,
    print_footer,
    print_header,
    print_per_benchmark,
    print_pooled,
    save_results,
)

EXTENDED_ALPHAS = np.logspace(0, 6, 100)
FRED_CANDIDATES: dict[str, str] = {
    "auto_ins_ppi_mom3m": "PCU5241265241261",
    "medical_cpi_mom3m": "CUSR0000SAM2",
    "motor_parts_cpi_mom3m": "CUSR0000SETA02",
    "mortgage_rate_delta_3m": "MORTGAGE30US",
    "term_premium_10y_raw": "THREEFYTP10",
}
FEATURE_MATRIX_FALLBACKS: list[str] = [
    "ppi_auto_ins_yoy",
    "medical_cpi_yoy",
    "motor_vehicle_ins_cpi_yoy",
    "used_car_cpi_yoy",
    "term_premium_10y",
]


def load_candidate_fred_features(conn) -> pd.DataFrame:
    """Pivot long-format FRED rows into derived monthly features."""
    series_ids = list(FRED_CANDIDATES.values())
    quoted = ",".join(f"'{series_id}'" for series_id in series_ids)
    query = (
        "SELECT month_end, series_id, value "
        "FROM fred_macro_monthly "
        f"WHERE series_id IN ({quoted}) "
        "ORDER BY month_end ASC"
    )
    fred_long = pd.read_sql_query(query, conn, parse_dates=["month_end"])
    if fred_long.empty:
        return pd.DataFrame()

    fred_wide = fred_long.pivot(index="month_end", columns="series_id", values="value").sort_index()
    derived = pd.DataFrame(index=fred_wide.index)

    if "PCU5241265241261" in fred_wide.columns:
        derived["auto_ins_ppi_mom3m"] = fred_wide["PCU5241265241261"].pct_change(3, fill_method=None)
    if "CUSR0000SAM2" in fred_wide.columns:
        derived["medical_cpi_mom3m"] = fred_wide["CUSR0000SAM2"].pct_change(3, fill_method=None)
    if "CUSR0000SETA02" in fred_wide.columns:
        derived["motor_parts_cpi_mom3m"] = fred_wide["CUSR0000SETA02"].pct_change(3, fill_method=None)
    if "MORTGAGE30US" in fred_wide.columns:
        derived["mortgage_rate_delta_3m"] = fred_wide["MORTGAGE30US"].diff(3)
    if "THREEFYTP10" in fred_wide.columns:
        derived["term_premium_10y_raw"] = fred_wide["THREEFYTP10"]

    return derived.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    """Run domain-specific FRED feature experiments against the v38 baseline."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        baseline = load_research_baseline_results()

        fred_features = load_candidate_fred_features(conn)
        available_raw = list(fred_features.columns)
        print(f"Derived long-format FRED candidates: {available_raw if available_raw else 'none'}")

        feature_df_aug = feature_df.join(fred_features, how="left")
        fallback_features = [col for col in FEATURE_MATRIX_FALLBACKS if col in feature_df_aug.columns]
        print(f"Feature-matrix fallback candidates: {fallback_features if fallback_features else 'none'}")

        extra_features: list[str] = []
        for feature_name in list(FRED_CANDIDATES.keys()) + FEATURE_MATRIX_FALLBACKS:
            if feature_name in feature_df_aug.columns and feature_name not in extra_features:
                extra_features.append(feature_name)
            if len(extra_features) >= 3:
                break

        if not extra_features and "rate_adequacy_gap_yoy" in feature_df_aug.columns:
            extra_features = ["rate_adequacy_gap_yoy"]

        selected_features = list(dict.fromkeys(RIDGE_FEATURES_12 + extra_features))
        print(f"Selected extra features: {extra_features}")

        rows: list[dict[str, object]] = []
        for benchmark in BENCHMARKS:
            rel_series = load_relative_series(conn, benchmark, horizon=6)
            if rel_series.empty:
                continue
            x_df, y = get_X_y_relative(feature_df_aug, rel_series, drop_na_target=True)
            feature_cols = [col for col in selected_features if col in x_df.columns]

            def ridge_factory() -> Pipeline:
                return Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", RidgeCV(alphas=EXTENDED_ALPHAS, cv=None)),
                    ]
                )

            y_true, y_hat = custom_wfo(
                x_df[feature_cols].to_numpy(),
                y.to_numpy(),
                ridge_factory,
                MAX_TRAIN_MONTHS,
                TEST_SIZE_MONTHS,
                GAP_MONTHS,
            )
            metrics = compute_metrics(y_true, y_hat)
            rows.append(
                {
                    "benchmark": benchmark,
                    **metrics,
                    "_y_true": y_true,
                    "_y_hat": y_hat,
                }
            )

        pooled = pool_metrics(rows)
        print_header("v58", f"Domain-Specific FRED Features (+{len(extra_features)} extras)")
        print_per_benchmark(rows)
        print_pooled(pooled)
        print_delta(pooled, baseline)
        print_footer()
        save_results(
            build_results_df(
                rows,
                pooled,
                extra_cols={
                    "variant": "A_domain_fred_features",
                    "features_used": "|".join(selected_features),
                    "version": "v58",
                },
            ),
            "v58_fred_results.csv",
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
