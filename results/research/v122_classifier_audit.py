"""v122 - Current classifier audit note and coefficient snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.database import db_client
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_X_y_relative,
    truncate_relative_target_for_asof,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.v37_utils import RESULTS_DIR, RIDGE_FEATURES_12
from src.research.v87_utils import build_target_series, logistic_factory


MONTHLY_DIR = Path("results/monthly_decisions/2026-04")


def _load_shadow_detail() -> tuple[dict[str, object], pd.DataFrame]:
    summary = json.loads((MONTHLY_DIR / "monthly_summary.json").read_text(encoding="utf-8"))
    shadow_df = pd.read_csv(MONTHLY_DIR / "classification_shadow.csv")
    return summary, shadow_df


def _fit_benchmark_coefficients(as_of: pd.Timestamp, benchmarks: list[str]) -> pd.DataFrame:
    conn = db_client.get_connection()
    try:
        feature_df = build_feature_matrix_from_db(conn, force_refresh=True)
        feature_df = feature_df.loc[feature_df.index <= as_of].sort_index()
        rows: list[dict[str, object]] = []
        for benchmark in benchmarks:
            rel_series = load_relative_return_matrix(conn, benchmark, 6)
            if rel_series.empty:
                continue
            rel_series = truncate_relative_target_for_asof(
                rel_series,
                as_of=as_of,
                horizon_months=6,
            )
            x_base, _ = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
            target = build_target_series(rel_series, "actionable_sell_3pct")
            y_train = target.reindex(x_base.index).dropna().astype(int)
            x_train = x_base.loc[y_train.index, RIDGE_FEATURES_12].copy()
            if x_train.empty or len(np.unique(y_train.to_numpy(dtype=int))) < 2:
                continue

            x_values = x_train.to_numpy(dtype=float).copy()  # .copy() ensures writable (pandas 3.0+)
            medians = np.nanmedian(x_values, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            for idx in range(x_values.shape[1]):
                x_values[np.isnan(x_values[:, idx]), idx] = medians[idx]

            model = logistic_factory(class_weight="balanced", c_value=0.5)()
            model.fit(x_values, y_train.to_numpy(dtype=int))
            coef = model.coef_[0]
            std = np.nanstd(x_values, axis=0)
            std = np.where(np.isnan(std) | (std <= 1e-12), 1.0, std)
            for feature, coefficient, feature_std in zip(
                RIDGE_FEATURES_12,
                coef,
                std,
                strict=False,
            ):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "feature": feature,
                        "coefficient": float(coefficient),
                        "feature_std": float(feature_std),
                        "standardized_abs_coef": float(abs(coefficient * feature_std)),
                        "train_n": int(len(x_train)),
                    }
                )
        return pd.DataFrame(rows)
    finally:
        conn.close()


def main() -> None:
    summary, shadow_df = _load_shadow_detail()
    as_of = pd.Timestamp(summary["as_of_date"])
    coeff_df = _fit_benchmark_coefficients(as_of, shadow_df["benchmark"].astype(str).tolist())
    coeff_df = coeff_df.merge(
        shadow_df[["benchmark", "classifier_weight"]],
        on="benchmark",
        how="left",
    )
    coeff_df["weighted_importance"] = (
        coeff_df["standardized_abs_coef"] * coeff_df["classifier_weight"]
    )
    coeff_df = coeff_df.sort_values(
        ["benchmark", "standardized_abs_coef"],
        ascending=[True, False],
    ).reset_index(drop=True)
    coeff_path = RESULTS_DIR / "v122_classifier_audit_coefficients.csv"
    coeff_df.to_csv(coeff_path, index=False)

    feature_totals = (
        coeff_df.groupby("feature", as_index=False)["weighted_importance"]
        .sum()
        .sort_values("weighted_importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_totals_path = RESULTS_DIR / "v122_classifier_audit_feature_totals.csv"
    feature_totals.to_csv(feature_totals_path, index=False)

    shadow_summary = summary["classification_shadow"]
    pooled_v90 = pd.read_csv(RESULTS_DIR / "v90_pooled_panel_classifiers_results.csv")
    pooled_best = pooled_v90.loc[
        pooled_v90["selected_next"].fillna(False).astype(bool)
    ].iloc[0]
    v92 = pd.read_csv(RESULTS_DIR / "v92_calibration_and_abstention_results.csv")
    v92_best = v92.loc[v92["selected_next"].fillna(False).astype(bool)].iloc[0]

    probability_table = shadow_df[
        [
            "benchmark",
            "classifier_prob_actionable_sell",
            "classifier_weight",
            "classifier_weighted_contribution",
            "classifier_shadow_tier",
        ]
    ].copy()
    probability_table["classifier_prob_actionable_sell"] = (
        probability_table["classifier_prob_actionable_sell"] * 100.0
    ).round(1)
    probability_table["classifier_weight"] = (
        probability_table["classifier_weight"] * 100.0
    ).round(1)
    probability_table["classifier_weighted_contribution"] = (
        probability_table["classifier_weighted_contribution"] * 100.0
    ).round(1)

    top_feature_lines = [
        f"- `{row.feature}`: weighted importance {row.weighted_importance:.3f}"
        for row in feature_totals.head(6).itertuples(index=False)
    ]

    top_benchmark_lines: list[str] = []
    for benchmark, frame in coeff_df.groupby("benchmark"):
        top_rows = frame.head(3)
        bits = ", ".join(
            f"`{row.feature}` ({row.standardized_abs_coef:.3f})"
            for row in top_rows.itertuples(index=False)
        )
        train_n = int(top_rows.iloc[0]["train_n"])
        top_benchmark_lines.append(
            f"- `{benchmark}` (`train_n={train_n}`): {bits}"
        )

    summary_lines = [
        "# v122 Classifier Audit",
        "",
        f"As-of monthly run: `{summary['as_of_date']}`.",
        f"Feature anchor date: `{shadow_summary['feature_anchor_date']}`.",
        "",
        "## Implemented Shadow Classifier",
        "",
        "- monthly shadow model family: `separate_benchmark_logistic_balanced`",
        "- target: `actionable_sell_3pct`",
        "- feature set: `lean_baseline`",
        "- calibration: `oos_logistic_calibration`",
        "- aggregation: benchmark-quality weighted probability pool",
        "",
        "Lean baseline features:",
        "",
        *[f"- `{feature}`" for feature in RIDGE_FEATURES_12],
        "",
        "## Training Scope",
        "",
        f"- benchmark coverage: {', '.join(shadow_df['benchmark'].astype(str).tolist())}",
        f"- benchmarks covered: `{len(shadow_df)}`",
        f"- pooled benchmark-month sample used in the selected `v90` pooled reference: `{int(pooled_best['n_obs'])}`",
        "- current monthly shadow implementation fits one logistic model per benchmark on all history available up to the as-of month, with explicit target truncation to prevent backdated leakage",
        "",
        "Per-benchmark training rows used in the current audit fit:",
        "",
        *top_benchmark_lines,
        "",
        "## Accuracy And Calibration",
        "",
        f"- best pooled classifier family from `v90`: `pooled_shared_logistic_balanced`",
        f"- pooled accuracy: `{pooled_best['accuracy']:.4f}`",
        f"- pooled balanced accuracy: `{pooled_best['balanced_accuracy']:.4f}`",
        f"- pooled Brier score: `{pooled_best['brier_score']:.4f}`",
        f"- pooled log loss: `{pooled_best['log_loss']:.4f}`",
        "",
        f"- best calibrated shadow-style path from `v92`: `{v92_best['candidate_name']}__{v92_best['calibration']}__{v92_best['lower_threshold']:.2f}_{v92_best['upper_threshold']:.2f}`",
        f"- calibrated accuracy: `{v92_best['accuracy']:.4f}`",
        f"- calibrated balanced accuracy: `{v92_best['balanced_accuracy']:.4f}`",
        f"- calibrated Brier score: `{v92_best['brier_score']:.4f}`",
        f"- calibrated log loss: `{v92_best['log_loss']:.4f}`",
        f"- calibrated ECE: `{v92_best['ece_10']:.4f}`",
        "",
        "## Current-Month Probability Snapshot",
        "",
        probability_table.to_markdown(index=False),
        "",
        f"Aggregated `P(Actionable Sell)`: `{shadow_summary['probability_actionable_sell_label']}`.",
        f"Confidence tier: `{shadow_summary['confidence_tier']}`.",
        f"Stance: `{shadow_summary['stance']}`.",
        f"Top supporting benchmark: `{shadow_summary['top_supporting_benchmark']}`.",
        "",
        "## Approximate Feature Importance",
        "",
        "This audit uses standardized absolute logistic coefficients within each benchmark as an approximate importance measure. That is directionally useful, but it is not a causal importance ranking.",
        "",
        "Top weighted features across the eight-benchmark monthly pool:",
        "",
        *top_feature_lines,
        "",
        "Interpretation:",
        "",
        "- `combined_ratio_ttm` dominates the current audit, which means underwriting quality is the strongest repeated driver of the actionable-sell classifier",
        "- `credit_spread_hy`, `real_yield_change_6m`, `yield_slope`, and `real_rate_10y` show that macro / rates / credit conditions are the next most important layer",
        "- `mom_12m` matters, but it is not the leading driver in the current audit",
        "- the current classifier is therefore not just a momentum rule; it is leaning heavily on insurance fundamentals plus macro stress context",
        "",
        f"Detailed coefficients: `{coeff_path.as_posix()}`",
        f"Aggregated feature totals: `{feature_totals_path.as_posix()}`",
        "",
    ]
    summary_path = RESULTS_DIR / "v122_classifier_audit_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
