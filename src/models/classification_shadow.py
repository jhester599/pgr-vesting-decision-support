"""Shadow-only monthly classifier summary for recommendation interpretation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_X_y_relative,
    truncate_relative_target_for_asof,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.v66_utils import benchmark_quality_weights
from src.research.v87_utils import (
    build_target_series,
    evaluate_binary_time_series,
    feature_set_from_name,
    logistic_factory,
)


ACTIONABLE_TARGET = "actionable_sell_3pct"
FEATURE_SET_NAME = "lean_baseline"
MODEL_FAMILY = "separate_benchmark_logistic_balanced"
CALIBRATION_LABEL = "oos_logistic_calibration"
MIN_CALIBRATION_HISTORY = 24


@dataclass(frozen=True)
class ClassificationShadowSummary:
    """Compact shadow-classifier summary for monthly reporting surfaces."""

    enabled: bool
    target_label: str
    feature_set: str
    model_family: str
    calibration: str
    probability_actionable_sell: float | None
    probability_actionable_sell_label: str | None
    probability_non_actionable: float | None
    probability_non_actionable_label: str | None
    confidence_tier: str | None
    stance: str | None
    agreement_with_live: bool | None
    agreement_label: str | None
    interpretation: str | None
    benchmark_count: int
    feature_anchor_date: str | None
    top_supporting_benchmark: str | None
    top_supporting_contribution: float | None
    top_supporting_contribution_label: str | None

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable payload for monthly summary artifacts."""
        return asdict(self)


def _format_pct(value: float | None, decimals: int = 1) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.{decimals}f}%"


def classification_confidence_tier(probability_actionable_sell: float) -> str:
    """Map actionable-sell probability into a simple confidence tier."""
    if probability_actionable_sell >= 0.70 or probability_actionable_sell <= 0.30:
        return "HIGH"
    if probability_actionable_sell >= 0.60 or probability_actionable_sell <= 0.40:
        return "MODERATE"
    return "LOW"


def classification_stance(probability_actionable_sell: float) -> str:
    """Return the shadow classifier stance from the actionable-sell probability."""
    if probability_actionable_sell >= 0.70:
        return "ACTIONABLE-SELL"
    if probability_actionable_sell <= 0.30:
        return "NON-ACTIONABLE"
    return "NEUTRAL"


def classification_interpretation(
    probability_actionable_sell: float,
    stance: str,
    confidence_tier: str,
) -> str:
    """Return a short human-readable interpretation for reports."""
    probability_label = _format_pct(probability_actionable_sell, decimals=1) or "n/a"
    if stance == "ACTIONABLE-SELL":
        if confidence_tier == "HIGH":
            return (
                f"Shadow classifier sees a strong actionable-sell regime ({probability_label}). "
                "Treat the live recommendation as lower-confidence if it remains non-actionable."
            )
        return (
            f"Shadow classifier leans actionable-sell ({probability_label}), "
            "but the signal is not yet decisive."
        )
    if stance == "NON-ACTIONABLE":
        if confidence_tier == "HIGH":
            return (
                f"Shadow classifier sees low actionability ({probability_label}), "
                "which supports a hold/defer interpretation this month."
            )
        return (
            f"Shadow classifier leans non-actionable ({probability_label}); "
            "evidence for a sell regime is limited."
        )
    return (
        f"Shadow classifier is near its neutral band ({probability_label}); "
        "use it as a low-confidence interpretation layer rather than a decision override."
    )


def agreement_with_live_recommendation(
    stance: str,
    live_recommendation_mode: str,
) -> bool:
    """Return whether the shadow stance agrees with the live recommendation mode."""
    live_is_actionable = str(live_recommendation_mode).upper() == "ACTIONABLE"
    classifier_is_actionable = stance == "ACTIONABLE-SELL"
    return live_is_actionable == classifier_is_actionable


def _fit_current_probability(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_current: pd.DataFrame,
) -> float | None:
    """Fit the benchmark-specific classifier on all history and score the current row."""
    if x_train.empty or x_current.empty or len(np.unique(y_train.to_numpy(dtype=int))) < 2:
        return None

    x_train_values = x_train.to_numpy(dtype=float)
    x_current_values = x_current.to_numpy(dtype=float)
    medians = np.nanmedian(x_train_values, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    for col_idx in range(x_train_values.shape[1]):
        x_train_values[np.isnan(x_train_values[:, col_idx]), col_idx] = medians[col_idx]
        x_current_values[np.isnan(x_current_values[:, col_idx]), col_idx] = medians[col_idx]

    model = logistic_factory(class_weight="balanced", c_value=0.5)()
    model.fit(x_train_values, y_train.to_numpy(dtype=int))
    return float(model.predict_proba(x_current_values)[0, 1])


def _fit_oos_calibrator(probability_history: pd.DataFrame) -> LogisticRegression | None:
    """Fit a lightweight logistic calibrator on prior OOS probabilities."""
    if probability_history.empty or len(probability_history) < MIN_CALIBRATION_HISTORY:
        return None

    y_hist = probability_history["y_true"].to_numpy(dtype=int)
    if len(np.unique(y_hist)) < 2:
        return None

    x_hist = np.clip(
        probability_history["y_prob"].to_numpy(dtype=float),
        1e-6,
        1.0 - 1e-6,
    ).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=2000, random_state=42)
    calibrator.fit(x_hist, y_hist)
    return calibrator


def _empty_summary() -> ClassificationShadowSummary:
    return ClassificationShadowSummary(
        enabled=False,
        target_label=ACTIONABLE_TARGET,
        feature_set=FEATURE_SET_NAME,
        model_family=MODEL_FAMILY,
        calibration=CALIBRATION_LABEL,
        probability_actionable_sell=None,
        probability_actionable_sell_label=None,
        probability_non_actionable=None,
        probability_non_actionable_label=None,
        confidence_tier=None,
        stance=None,
        agreement_with_live=None,
        agreement_label=None,
        interpretation=None,
        benchmark_count=0,
        feature_anchor_date=None,
        top_supporting_benchmark=None,
        top_supporting_contribution=None,
        top_supporting_contribution_label=None,
    )


def build_classification_shadow_summary(
    conn,
    as_of: date,
    *,
    live_recommendation_mode: str,
    benchmark_quality_df: pd.DataFrame | None,
) -> tuple[ClassificationShadowSummary, pd.DataFrame]:
    """Build the shadow classifier summary and per-benchmark detail table."""
    feature_df = build_feature_matrix_from_db(conn, force_refresh=True)
    feature_df = feature_df.loc[feature_df.index <= pd.Timestamp(as_of)].sort_index()
    if feature_df.empty:
        return _empty_summary(), pd.DataFrame()

    current_features = feature_df.iloc[[-1]].copy()
    feature_anchor_date = str(pd.Timestamp(current_features.index[0]).date())
    feature_columns = feature_set_from_name(feature_df, FEATURE_SET_NAME)
    benchmarks = list(
        benchmark_quality_df["benchmark"].astype(str).tolist()
        if benchmark_quality_df is not None and not benchmark_quality_df.empty
        else config.PRIMARY_FORECAST_UNIVERSE
    )
    model_factory = logistic_factory(class_weight="balanced", c_value=0.5)

    rows: list[dict[str, object]] = []
    for benchmark in benchmarks:
        rel_series = load_relative_return_matrix(conn, benchmark, 6)
        if rel_series.empty:
            continue
        rel_series = truncate_relative_target_for_asof(
            rel_series,
            as_of=pd.Timestamp(as_of),
            horizon_months=6,
        )

        x_base, _ = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
        if x_base.empty:
            continue

        usable_features = [feature for feature in feature_columns if feature in x_base.columns]
        if not usable_features:
            continue

        x_train = x_base[usable_features].copy()
        target = build_target_series(rel_series, ACTIONABLE_TARGET)
        y_train = target.reindex(x_train.index).dropna().astype(int)
        x_train = x_train.loc[y_train.index]
        if x_train.empty:
            continue

        x_current = current_features[usable_features].copy()
        raw_probability = _fit_current_probability(x_train, y_train, x_current)
        if raw_probability is None:
            continue

        probability_history = evaluate_binary_time_series(x_train, y_train, model_factory)
        calibrator = _fit_oos_calibrator(probability_history)
        calibrated_probability = raw_probability
        if calibrator is not None:
            calibrated_probability = float(
                calibrator.predict_proba(
                    np.array([[np.clip(raw_probability, 1e-6, 1.0 - 1e-6)]], dtype=float)
                )[0, 1]
            )

        rows.append(
            {
                "benchmark": benchmark,
                "classifier_raw_prob_actionable_sell": float(raw_probability),
                "classifier_prob_actionable_sell": float(calibrated_probability),
                "classifier_shadow_tier": classification_confidence_tier(
                    float(calibrated_probability)
                ),
                "classifier_history_obs": int(len(probability_history)),
            }
        )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        return _empty_summary(), detail_df

    if benchmark_quality_df is not None and not benchmark_quality_df.empty:
        quality_subset = benchmark_quality_df[
            benchmark_quality_df["benchmark"].astype(str).isin(detail_df["benchmark"])
        ].copy()
    else:
        quality_subset = pd.DataFrame({"benchmark": detail_df["benchmark"], "nw_ic": 0.0})
    if "nw_ic" not in quality_subset.columns:
        quality_subset["nw_ic"] = 0.0

    weights = benchmark_quality_weights(quality_subset, score_col="nw_ic", lambda_mix=0.25)
    detail_df["classifier_weight"] = detail_df["benchmark"].map(
        lambda benchmark: float(weights.get(str(benchmark), 0.0))
    )
    weight_sum = float(detail_df["classifier_weight"].sum())
    if weight_sum <= 1e-12:
        detail_df["classifier_weight"] = 1.0 / len(detail_df)
    else:
        detail_df["classifier_weight"] = detail_df["classifier_weight"] / weight_sum

    detail_df["classifier_weighted_contribution"] = (
        detail_df["classifier_prob_actionable_sell"] * detail_df["classifier_weight"]
    )
    aggregated_probability = float(detail_df["classifier_weighted_contribution"].sum())
    tier = classification_confidence_tier(aggregated_probability)
    stance = classification_stance(aggregated_probability)
    agreement = agreement_with_live_recommendation(stance, live_recommendation_mode)
    top_row = detail_df.sort_values(
        "classifier_weighted_contribution",
        ascending=False,
    ).iloc[0]

    summary = ClassificationShadowSummary(
        enabled=True,
        target_label=ACTIONABLE_TARGET,
        feature_set=FEATURE_SET_NAME,
        model_family=MODEL_FAMILY,
        calibration=CALIBRATION_LABEL,
        probability_actionable_sell=aggregated_probability,
        probability_actionable_sell_label=_format_pct(aggregated_probability, decimals=1),
        probability_non_actionable=1.0 - aggregated_probability,
        probability_non_actionable_label=_format_pct(1.0 - aggregated_probability, decimals=1),
        confidence_tier=tier,
        stance=stance,
        agreement_with_live=agreement,
        agreement_label="Aligned" if agreement else "Mixed",
        interpretation=classification_interpretation(aggregated_probability, stance, tier),
        benchmark_count=int(len(detail_df)),
        feature_anchor_date=feature_anchor_date,
        top_supporting_benchmark=str(top_row["benchmark"]),
        top_supporting_contribution=float(top_row["classifier_weighted_contribution"]),
        top_supporting_contribution_label=_format_pct(
            float(top_row["classifier_weighted_contribution"]),
            decimals=1,
        ),
    )
    return summary, detail_df.sort_values("benchmark").reset_index(drop=True)
