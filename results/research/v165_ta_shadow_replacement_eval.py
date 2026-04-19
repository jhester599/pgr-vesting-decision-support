"""v165 TA classification replacement shadow evaluation.

This harness is research-only. It tests the v164 replacement-candidate
recommendation in the existing shadow-classifier framing without changing live
monthly decision behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from scripts.confirmatory_classifier_experiments import evaluate_confirmatory_classifier
from src.database import db_client
from src.models.classification_shadow import (
    _fit_current_probability,
    _fit_oos_calibrator,
    classification_confidence_tier,
    classification_stance,
)
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_X_y_relative,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.v160_ta_features import build_ta_feature_matrix
from src.research.v87_utils import build_target_series

PRIMARY_BENCHMARKS: tuple[str, ...] = (
    "VOO",
    "VXUS",
    "VWO",
    "VMBS",
    "BND",
    "GLD",
    "DBC",
    "VDE",
)
ACTIONABLE_TARGET = "actionable_sell_3pct"
OUTPUT_DIR = Path("results") / "research"
DETAIL_PATH = OUTPUT_DIR / "v165_ta_shadow_replacement_detail.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "v165_ta_shadow_replacement_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "v165_ta_shadow_replacement_summary.csv"
REGIME_PATH = OUTPUT_DIR / "v165_ta_shadow_replacement_regime_slices.csv"
CURRENT_PATH = OUTPUT_DIR / "v165_ta_shadow_current.csv"
CURRENT_SUMMARY_PATH = OUTPUT_DIR / "v165_ta_shadow_current_summary.csv"
CANDIDATE_PATH = OUTPUT_DIR / "v165_ta_shadow_candidate.json"


def apply_feature_swaps(
    baseline_features: list[str],
    feature_swaps: dict[str, str],
) -> list[str]:
    """Apply replacement-only feature swaps while preserving baseline order."""
    result: list[str] = []
    seen: set[str] = set()
    for feature in baseline_features:
        replacement = feature_swaps.get(feature, feature)
        if replacement in seen:
            continue
        result.append(replacement)
        seen.add(replacement)
    return result


def build_candidate_variants() -> list[dict[str, Any]]:
    """Return the v165 pre-registered replacement-only candidate variants."""
    return [
        {
            "variant": "lean_baseline",
            "feature_swaps": {},
            "notes": "Existing lean shadow-classifier baseline.",
        },
        {
            "variant": "ta_obv_replaces_mom12",
            "feature_swaps": {"mom_12m": "ta_pgr_obv_detrended"},
            "notes": "Single-feature replacement from the strongest v164 survivor.",
        },
        {
            "variant": "ta_natr_replaces_vol63",
            "feature_swaps": {"vol_63d": "ta_pgr_natr_63d"},
            "notes": "Single volatility replacement from the v164 survivor set.",
        },
        {
            "variant": "ta_minimal_replacement",
            "feature_swaps": {
                "mom_12m": "ta_pgr_obv_detrended",
                "vol_63d": "ta_pgr_natr_63d",
            },
            "notes": "Minimal two-swap TA replacement candidate.",
        },
        {
            "variant": "ta_minimal_plus_vwo_pct_b",
            "feature_swaps": {
                "mom_12m": "ta_pgr_obv_detrended",
                "vol_63d": "ta_pgr_natr_63d",
                "vix": "ta_ratio_bb_pct_b_6m_vwo",
            },
            "notes": "Minimal candidate plus one representative ratio Bollinger feature.",
        },
    ]


def attach_variant_deltas(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-benchmark deltas versus the lean baseline variant."""
    if detail_df.empty:
        return detail_df.copy()
    key_cols = ["benchmark"]
    metric_cols = [
        "balanced_accuracy",
        "brier_score",
        "accuracy",
        "precision",
        "recall",
    ]
    present_metrics = [col for col in metric_cols if col in detail_df.columns]
    baseline = detail_df.loc[
        detail_df["variant"].eq("lean_baseline"),
        key_cols + present_metrics,
    ].drop_duplicates(key_cols)
    baseline = baseline.rename(
        columns={col: f"baseline_{col}" for col in present_metrics}
    )
    merged = detail_df.merge(baseline, on=key_cols, how="left")
    for col in present_metrics:
        merged[f"delta_{col}"] = merged[col] - merged[f"baseline_{col}"]
    merged["improved_vs_baseline"] = (
        merged.get("delta_balanced_accuracy", pd.Series(0.0, index=merged.index))
        .fillna(0.0)
        .gt(0.0)
        | merged.get("delta_brier_score", pd.Series(0.0, index=merged.index))
        .fillna(0.0)
        .lt(0.0)
    )
    return merged


def summarize_variants(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize v165 variants across benchmarks."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variant, group in detail_df.groupby("variant", dropna=False):
        rows.append(
            {
                "variant": variant,
                "n_benchmarks": int(group["benchmark"].nunique()),
                "n_features": int(group["n_features"].iloc[0])
                if "n_features" in group
                else 0,
                "mean_balanced_accuracy": float(
                    group.get("balanced_accuracy", pd.Series(dtype=float)).mean()
                ),
                "mean_brier_score": float(
                    group.get("brier_score", pd.Series(dtype=float)).mean()
                ),
                "mean_delta_balanced_accuracy": float(
                    group.get("delta_balanced_accuracy", pd.Series(dtype=float)).mean()
                ),
                "mean_delta_brier_score": float(
                    group.get("delta_brier_score", pd.Series(dtype=float)).mean()
                ),
                "positive_benchmark_count": int(
                    group.loc[
                        group.get(
                            "improved_vs_baseline",
                            pd.Series(False, index=group.index),
                        ).fillna(False),
                        "benchmark",
                    ].nunique()
                ),
                "degraded_benchmark_count": int(
                    group.loc[
                        ~group.get(
                            "improved_vs_baseline",
                            pd.Series(False, index=group.index),
                        ).fillna(False),
                        "benchmark",
                    ].nunique()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["positive_benchmark_count", "mean_delta_balanced_accuracy"],
        ascending=[False, False],
    )


def assign_regime_slice(date_like: object) -> str:
    """Map a timestamp to the v163/v165 regime label set."""
    timestamp = pd.Timestamp(date_like)
    if timestamp < pd.Timestamp("2020-01-01"):
        return "pre_2020"
    if timestamp < pd.Timestamp("2022-01-01"):
        return "covid_2020_2021"
    return "post_2022"


def _load_price_map(conn: Any, tickers: list[str]) -> dict[str, pd.DataFrame]:
    price_map: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        prices = db_client.get_prices(conn, ticker)
        if not prices.empty:
            price_map[ticker] = prices
    return price_map


def _build_feature_frame(conn: Any) -> pd.DataFrame:
    base_features = build_feature_matrix_from_db(conn)
    tickers = sorted(set(("PGR", *PRIMARY_BENCHMARKS, "ALL", "TRV", "CB", "HIG")))
    ta_features = build_ta_feature_matrix(
        _load_price_map(conn, tickers),
        benchmarks=PRIMARY_BENCHMARKS,
    )
    return base_features.join(ta_features, how="left")


def _latest_price_date(conn: Any) -> str | None:
    latest = pd.read_sql_query(
        "SELECT MAX(date) AS max_date FROM daily_prices",
        conn,
    )["max_date"].iloc[0]
    if pd.isna(latest):
        return None
    return pd.Timestamp(latest).date().isoformat()


def _evaluate_current_probability(
    x_aligned: pd.DataFrame,
    y_binary: pd.Series,
    current_features: pd.DataFrame,
    selected: list[str],
) -> tuple[float | None, float | None, int]:
    x_train = x_aligned[selected].copy()
    y_train = y_binary.reindex(x_train.index).dropna().astype(int)
    x_train = x_train.loc[y_train.index]
    if x_train.empty:
        return None, None, 0
    x_current = current_features[selected].copy()
    raw_probability = _fit_current_probability(x_train, y_train, x_current)
    if raw_probability is None:
        return None, None, 0
    prob_series, realized_series, _ = evaluate_confirmatory_classifier(
        x_aligned,
        y_binary,
        model_type="ridge",
        feature_columns=selected,
        target_horizon_months=6,
    )
    history = pd.DataFrame(
        {
            "y_prob": prob_series,
            "y_true": realized_series.astype(int),
        }
    ).dropna()
    calibrator = _fit_oos_calibrator(history)
    calibrated = raw_probability
    if calibrator is not None:
        calibrated = float(
            calibrator.predict_proba(
                np.array([[np.clip(raw_probability, 1e-6, 1.0 - 1e-6)]], dtype=float)
            )[0, 1]
        )
    return float(raw_probability), float(calibrated), int(len(history))


def build_candidate_payload(summary: pd.DataFrame) -> dict[str, Any]:
    """Build deterministic JSON handoff for the v165 result."""
    if summary.empty:
        recommendation = "abandon_ta_shadow"
        rows: list[dict[str, Any]] = []
    else:
        candidates = summary.loc[summary["variant"].ne("lean_baseline")].copy()
        winners = candidates.loc[
            candidates["positive_benchmark_count"].fillna(0).ge(4)
            & candidates["degraded_benchmark_count"].fillna(0).le(4)
        ]
        recommendation = "shadow_monitor" if not winners.empty else "abandon_ta_shadow"
        rows = candidates.to_dict(orient="records")
    return {
        "version": "v165",
        "recommendation": recommendation,
        "production_changes": False,
        "shadow_monthly_changes": False,
        "rows": rows,
    }


def run_shadow_replacement_eval(
    output_dir: Path = OUTPUT_DIR,
    benchmarks: tuple[str, ...] = PRIMARY_BENCHMARKS,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Run the v165 TA shadow replacement evaluation and write artifacts."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _build_feature_frame(conn)
        current_features = feature_df.iloc[[-1]].copy()
        feature_anchor_date = pd.Timestamp(current_features.index[0]).date().isoformat()
        data_as_of = _latest_price_date(conn) or feature_anchor_date
        baseline_features = [
            feature
            for feature in config.MODEL_FEATURE_OVERRIDES["ridge"]
            if feature in feature_df.columns
        ]
        variants = build_candidate_variants()
        detail_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        current_rows: list[dict[str, Any]] = []

        for benchmark in benchmarks:
            rel_series = load_relative_return_matrix(conn, benchmark, 6)
            if rel_series.empty:
                continue
            try:
                x_aligned, _ = get_X_y_relative(
                    feature_df,
                    rel_series,
                    drop_na_target=True,
                )
            except ValueError:
                continue
            y_binary = build_target_series(rel_series, ACTIONABLE_TARGET).reindex(
                x_aligned.index
            )
            y_binary = y_binary.dropna().astype(int)
            x_aligned = x_aligned.loc[y_binary.index]
            if x_aligned.empty:
                continue

            for variant in variants:
                feature_swaps = dict(variant["feature_swaps"])
                candidate_features = apply_feature_swaps(baseline_features, feature_swaps)
                selected = [
                    feature for feature in candidate_features if feature in x_aligned.columns
                ]
                if not selected:
                    continue
                prob_series, realized_series, metrics = evaluate_confirmatory_classifier(
                    x_aligned,
                    y_binary,
                    model_type="ridge",
                    feature_columns=selected,
                    target_horizon_months=6,
                )
                detail_rows.append(
                    {
                        "benchmark": benchmark,
                        "variant": variant["variant"],
                        "feature_columns": "|".join(selected),
                        "feature_swaps": json.dumps(feature_swaps, sort_keys=True),
                        "n_features": len(selected),
                        "notes": variant["notes"],
                        **metrics,
                    }
                )
                aligned_predictions = pd.concat(
                    [prob_series.rename("probability"), realized_series.rename("realized")],
                    axis=1,
                ).dropna()
                for date_index, row in aligned_predictions.iterrows():
                    prediction_rows.append(
                        {
                            "date": pd.Timestamp(date_index).date().isoformat(),
                            "benchmark": benchmark,
                            "variant": variant["variant"],
                            "probability": float(row["probability"]),
                            "realized": int(row["realized"]),
                            "regime_slice": assign_regime_slice(date_index),
                        }
                    )

                raw_current, calibrated_current, history_obs = _evaluate_current_probability(
                    x_aligned,
                    y_binary,
                    current_features,
                    selected,
                )
                current_rows.append(
                    {
                        "as_of": data_as_of,
                        "feature_anchor_date": feature_anchor_date,
                        "benchmark": benchmark,
                        "variant": variant["variant"],
                        "raw_probability_actionable_sell": raw_current,
                        "probability_actionable_sell": calibrated_current,
                        "history_obs": history_obs,
                        "confidence_tier": (
                            classification_confidence_tier(calibrated_current)
                            if calibrated_current is not None
                            else None
                        ),
                        "stance": (
                            classification_stance(calibrated_current)
                            if calibrated_current is not None
                            else None
                        ),
                    }
                )
    finally:
        conn.close()

    detail = attach_variant_deltas(pd.DataFrame(detail_rows))
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_variants(detail)
    regime = summarize_regime_slices(predictions)
    current = pd.DataFrame(current_rows)
    current_summary = summarize_current_shadow(current)
    payload = build_candidate_payload(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / DETAIL_PATH.name, index=False)
    predictions.to_csv(output_dir / PREDICTIONS_PATH.name, index=False)
    summary.to_csv(output_dir / SUMMARY_PATH.name, index=False)
    regime.to_csv(output_dir / REGIME_PATH.name, index=False)
    current.to_csv(output_dir / CURRENT_PATH.name, index=False)
    current_summary.to_csv(output_dir / CURRENT_SUMMARY_PATH.name, index=False)
    (output_dir / CANDIDATE_PATH.name).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "detail": detail,
        "predictions": predictions,
        "summary": summary,
        "regime": regime,
        "current": current,
        "current_summary": current_summary,
        "payload": payload,
    }


def summarize_regime_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize prediction-level probabilities by regime slice."""
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["variant", "regime_slice"], dropna=False):
        variant, regime_slice = keys
        realized = group["realized"].astype(int)
        probability = group["probability"].astype(float)
        predicted = probability.ge(0.5).astype(int)
        rows.append(
            {
                "variant": variant,
                "regime_slice": regime_slice,
                "n_obs": int(len(group)),
                "base_rate": float(realized.mean()),
                "mean_probability": float(probability.mean()),
                "accuracy": float((predicted == realized).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "regime_slice"])


def summarize_current_shadow(current: pd.DataFrame) -> pd.DataFrame:
    """Summarize current per-benchmark shadow probabilities by variant."""
    if current.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variant, group in current.groupby("variant", dropna=False):
        probabilities = group["probability_actionable_sell"].dropna().astype(float)
        if probabilities.empty:
            probability = float("nan")
            tier = None
            stance = None
        else:
            probability = float(probabilities.mean())
            tier = classification_confidence_tier(probability)
            stance = classification_stance(probability)
        rows.append(
            {
                "as_of": str(group["as_of"].dropna().iloc[0])
                if "as_of" in group and group["as_of"].notna().any()
                else "",
                "feature_anchor_date": str(group["feature_anchor_date"].dropna().iloc[0])
                if "feature_anchor_date" in group
                and group["feature_anchor_date"].notna().any()
                else "",
                "variant": variant,
                "benchmark_count": int(group["benchmark"].nunique()),
                "probability_actionable_sell": probability,
                "probability_actionable_sell_label": (
                    f"{probability * 100:.1f}%" if not np.isnan(probability) else None
                ),
                "confidence_tier": tier,
                "stance": stance,
            }
        )
    return pd.DataFrame(rows).sort_values("variant")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v165 TA classification replacement shadow evaluation."
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run_shadow_replacement_eval(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
