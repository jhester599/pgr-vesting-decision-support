"""Shadow-only monthly classifier summary for recommendation interpretation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config
from config.features import (
    INVESTABLE_CLASSIFIER_BENCHMARKS,
    INVESTABLE_CLASSIFIER_BASE_WEIGHTS,
    PRIMARY_FORECAST_UNIVERSE,
    V128_BENCHMARK_FEATURE_MAP_PATH,
)
from src.models.v129_feature_map import (
    DualTrackFeatureMapError,
    load_v128_feature_map,
    resolve_benchmark_features,
)
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_X_y_relative,
    truncate_relative_target_for_asof,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.v66_utils import benchmark_quality_weights
from src.models.path_b_classifier import (
    apply_prequential_temperature_scaling,
    build_composite_return_series,
    fit_path_b_classifier,
    PATH_B_THRESHOLD,
    _apply_temperature as _path_b_apply_temperature,
    _fit_temperature_grid as _path_b_fit_temperature_grid,
)
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
    # v123: investable portfolio-weighted aggregate
    probability_investable_pool: float | None = None
    probability_investable_pool_label: str | None = None
    confidence_tier_investable_pool: str | None = None
    stance_investable_pool: str | None = None
    investable_benchmark_count: int = 0
    # v129: dual-track delta between benchmark-specific and lean-baseline probs
    dual_track_delta: dict[str, float] | None = None
    # v131: Path B composite portfolio-target classifier (temperature-scaled)
    probability_path_b_temp_scaled: float | None = None
    probability_path_b_temp_scaled_label: str | None = None
    confidence_tier_path_b: str | None = None
    stance_path_b: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-serializable payload for monthly summary artifacts."""
        return asdict(self)


def _format_pct(value: float | None, decimals: int = 1) -> str | None:
    if value is None:
        return None
    return f"{value * 100:.{decimals}f}%"


def classification_confidence_tier(probability_actionable_sell: float) -> str:
    """Map actionable-sell probability into a simple confidence tier."""
    if (
        probability_actionable_sell >= config.SHADOW_CLASSIFIER_HIGH_THRESH
        or probability_actionable_sell <= config.SHADOW_CLASSIFIER_LOW_THRESH
    ):
        return "HIGH"
    if (
        probability_actionable_sell >= config.SHADOW_CLASSIFIER_MODERATE_HIGH_THRESH
        or probability_actionable_sell <= config.SHADOW_CLASSIFIER_MODERATE_LOW_THRESH
    ):
        return "MODERATE"
    return "LOW"


def classification_stance(probability_actionable_sell: float) -> str:
    """Return the shadow classifier stance from the actionable-sell probability."""
    if probability_actionable_sell >= config.SHADOW_CLASSIFIER_HIGH_THRESH:
        return "ACTIONABLE-SELL"
    if probability_actionable_sell <= config.SHADOW_CLASSIFIER_LOW_THRESH:
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


def _portfolio_weighted_aggregate(
    detail_df: pd.DataFrame,
    investable_benchmarks: list[str],
    base_weights: dict[str, float],
) -> float | None:
    """Compute portfolio-weight-aligned aggregate P(Actionable Sell).

    Filters detail_df to investable benchmarks with valid calibrated
    probabilities, renormalizes base_weights over the available subset,
    and returns a weighted-average probability.

    Returns None if no investable benchmarks have valid probabilities.
    """
    mask = (
        detail_df["benchmark"].isin(investable_benchmarks)
        & detail_df["classifier_prob_actionable_sell"].notna()
    )
    available = detail_df.loc[mask].copy()
    if available.empty:
        return None

    available["_base_weight"] = available["benchmark"].map(base_weights).fillna(0.0)
    total_weight = available["_base_weight"].sum()
    if total_weight <= 0.0:
        return None

    available["_norm_weight"] = available["_base_weight"] / total_weight
    aggregate = float(
        (available["classifier_prob_actionable_sell"] * available["_norm_weight"]).sum()
    )
    return aggregate


def _fit_current_probability(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_current: pd.DataFrame,
) -> float | None:
    """Fit the benchmark-specific classifier on all history and score the current row."""
    if x_train.empty or x_current.empty or len(np.unique(y_train.to_numpy(dtype=int))) < 2:
        return None

    x_train_values = x_train.to_numpy(dtype=float).copy()  # .copy() ensures writable (pandas 3.0+)
    x_current_values = x_current.to_numpy(dtype=float).copy()
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


def _run_dual_track_pass(
    *,
    detail_df: pd.DataFrame,
    conn: object | None,
    as_of: date | None,
    feature_df: pd.DataFrame,
    current_features: pd.DataFrame,
    lean_baseline: list[str],
    feature_map: dict[str, list[str]],
) -> None:
    """Append benchmark-specific dual-track columns to detail_df in-place (v129).

    For benchmarks where the v128 feature map provides a distinct feature set
    (currently BND, DBC, VIG), trains a second classifier on that set and records
    its current-month probability. For all other benchmarks (including VGT due
    to the robustness audit), copies the existing lean-baseline probability.

    Does NOT modify classifier_prob_actionable_sell or any existing column.

    Args:
        detail_df: The per-benchmark detail frame already populated by the main loop.
        conn: Database connection (needed to load relative return matrices for
            re-training on benchmark-specific features). May be None for
            copy-through-only paths.
        as_of: As-of date for truncation. May be None for copy-through-only paths.
        feature_df: Full feature DataFrame used for training.
        current_features: Single-row DataFrame of current-month features.
        lean_baseline: List of feature names in the lean_baseline set.
        feature_map: Dict of benchmark -> feature list, containing ONLY
            benchmarks where switched_from_baseline is True. Obtained from
            load_v128_feature_map().
    """
    bs_probs: list[float | None] = []
    bs_tiers: list[str | None] = []
    bs_features: list[str | None] = []

    for _, row in detail_df.iterrows():
        benchmark = str(row["benchmark"])
        bm_features = resolve_benchmark_features(
            benchmark,
            feature_map,
            lean_baseline=lean_baseline,
        )
        is_same_as_lean = bm_features == lean_baseline

        if is_same_as_lean:
            # Copy-through: no re-training needed
            bs_probs.append(row.get("classifier_prob_actionable_sell"))
            bs_tiers.append(row.get("classifier_shadow_tier"))
            bs_features.append("|".join(lean_baseline))
        else:
            # Train a second classifier on the benchmark-specific feature set
            try:
                if conn is None or as_of is None or feature_df.empty:
                    raise ValueError("Cannot train without conn/as_of/feature_df")

                rel_series = load_relative_return_matrix(conn, benchmark, 6)
                if rel_series.empty:
                    raise ValueError(f"No relative return data for {benchmark}")
                rel_series = truncate_relative_target_for_asof(
                    rel_series,
                    as_of=pd.Timestamp(as_of),
                    horizon_months=6,
                )
                x_base, _ = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
                if x_base.empty:
                    raise ValueError(f"Empty x_base for {benchmark}")

                usable_features = [f for f in bm_features if f in x_base.columns]
                if not usable_features:
                    raise ValueError(f"No usable features for {benchmark}")

                x_train = x_base[usable_features].copy()
                target = build_target_series(rel_series, ACTIONABLE_TARGET)
                y_train = target.reindex(x_train.index).dropna().astype(int)
                x_train = x_train.loc[y_train.index]
                if x_train.empty:
                    raise ValueError(f"Empty training set for {benchmark}")

                # Build current features for the benchmark-specific feature set
                x_current_cols = [f for f in usable_features if f in current_features.columns]
                if not x_current_cols:
                    raise ValueError(f"No current features for {benchmark}")
                x_current = current_features[x_current_cols].copy()

                # Align training features to match current features
                x_train = x_train[x_current_cols]

                specific_prob = _fit_current_probability(x_train, y_train, x_current)
                bs_probs.append(specific_prob)
                bs_tiers.append(
                    classification_confidence_tier(specific_prob) if specific_prob is not None else None
                )
                bs_features.append("|".join(bm_features))
            except Exception:
                bs_probs.append(None)
                bs_tiers.append(None)
                bs_features.append("|".join(bm_features))

    detail_df["benchmark_specific_features"] = bs_features
    detail_df["benchmark_specific_prob_actionable_sell"] = bs_probs
    detail_df["benchmark_specific_tier"] = bs_tiers


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
    if benchmark_quality_df is not None and not benchmark_quality_df.empty:
        benchmarks_to_model = list(benchmark_quality_df["benchmark"].unique())
        # Ensure investable benchmarks are always included even if absent from quality_df
        for b in INVESTABLE_CLASSIFIER_BENCHMARKS:
            if b not in benchmarks_to_model:
                benchmarks_to_model.append(b)
    else:
        benchmarks_to_model = list(
            dict.fromkeys(PRIMARY_FORECAST_UNIVERSE + INVESTABLE_CLASSIFIER_BENCHMARKS)
        )
    benchmarks = benchmarks_to_model
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

    # --- Investable-pool aggregate (v123) ---
    investable_prob_raw = _portfolio_weighted_aggregate(
        detail_df,
        investable_benchmarks=INVESTABLE_CLASSIFIER_BENCHMARKS,
        base_weights=INVESTABLE_CLASSIFIER_BASE_WEIGHTS,
    )
    investable_benchmark_count = int(
        detail_df["benchmark"].isin(INVESTABLE_CLASSIFIER_BENCHMARKS).sum()
    )
    investable_prob_label: str | None = None
    investable_tier: str | None = None
    investable_stance: str | None = None
    if investable_prob_raw is not None:
        investable_prob_label = f"{investable_prob_raw * 100:.1f}%"
        investable_tier = classification_confidence_tier(investable_prob_raw)
        investable_stance = classification_stance(investable_prob_raw)

    # --- v129: dual-track benchmark-specific classifier pass ---
    try:
        _feature_map = load_v128_feature_map(V128_BENCHMARK_FEATURE_MAP_PATH)
    except DualTrackFeatureMapError:
        _feature_map = {}
    _run_dual_track_pass(
        detail_df=detail_df,
        conn=conn,
        as_of=as_of,
        feature_df=feature_df,
        current_features=current_features,
        lean_baseline=list(feature_columns),
        feature_map=_feature_map,
    )

    _dual_track_delta: dict[str, float] | None = None
    if "benchmark_specific_prob_actionable_sell" in detail_df.columns:
        _mask = detail_df["benchmark"].isin(INVESTABLE_CLASSIFIER_BENCHMARKS)
        _delta_rows = detail_df.loc[_mask].dropna(
            subset=["benchmark_specific_prob_actionable_sell", "classifier_prob_actionable_sell"]
        )
        if not _delta_rows.empty:
            _dual_track_delta = {
                str(r["benchmark"]): (
                    float(r["benchmark_specific_prob_actionable_sell"])
                    - float(r["classifier_prob_actionable_sell"])
                )
                for _, r in _delta_rows.iterrows()
            }

    # --- v131: Path B composite portfolio-target classifier (temperature-scaled) ---
    _path_b_prob_scaled: float | None = None
    _path_b_label: str | None = None
    _path_b_tier: str | None = None
    _path_b_stance: str | None = None

    try:
        # Build composite return series and binary target
        _composite_rel = build_composite_return_series(
            conn, weights=INVESTABLE_CLASSIFIER_BASE_WEIGHTS, as_of=as_of
        )
        _y_composite = (_composite_rel < -PATH_B_THRESHOLD).astype(int)
        # Align with feature matrix using lean baseline features
        _lean_cols = [c for c in feature_columns if c in feature_df.columns]
        _aligned = (
            feature_df[_lean_cols]
            .join(_y_composite, how="inner")
            .dropna()
        )
        _composite_col = _y_composite.name
        if len(_aligned) >= 30 and _aligned[_composite_col].nunique() >= 2:
            _X_b = _aligned[_lean_cols]
            _y_b = _aligned[_composite_col].astype(int)
            # WFO OOS probabilities for prequential temperature calibration
            from sklearn.model_selection import TimeSeriesSplit
            _tscv = TimeSeriesSplit(n_splits=10, test_size=3, gap=8, max_train_size=60)
            _oos_probs_list: list[float] = []
            _oos_labels_list: list[int] = []
            for _tr, _te in _tscv.split(_X_b):
                if len(_tr) < 30:
                    continue
                _m = LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                )
                try:
                    _m.fit(_X_b.iloc[_tr], _y_b.iloc[_tr])
                    _oos_probs_list.extend(
                        _m.predict_proba(_X_b.iloc[_te])[:, 1].tolist()
                    )
                    _oos_labels_list.extend(_y_b.iloc[_te].tolist())
                except Exception:
                    continue
            if len(_oos_probs_list) >= 10:
                _oos_probs_arr = np.array(_oos_probs_list, dtype=float)
                _oos_labels_arr = np.array(_oos_labels_list, dtype=int)
                # Train on all data, get current-month raw probability
                _path_b_prob_raw = fit_path_b_classifier(
                    _X_b, _y_b, feature_cols=_lean_cols
                )
                if _path_b_prob_raw is not None:
                    # Fit temperature using all available OOS history
                    if len(np.unique(_oos_labels_arr)) >= 2:
                        _avg_temp = _path_b_fit_temperature_grid(
                            _oos_labels_arr, _oos_probs_arr
                        )
                        _path_b_prob_scaled = _path_b_apply_temperature(
                            _path_b_prob_raw, _avg_temp
                        )
                        _path_b_label = _format_pct(_path_b_prob_scaled, decimals=1)
                        _path_b_tier = classification_confidence_tier(_path_b_prob_scaled)
                        _path_b_stance = classification_stance(_path_b_prob_scaled)
    except Exception:
        pass  # Path B failure is non-fatal; fields remain None

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
        probability_investable_pool=investable_prob_raw,
        probability_investable_pool_label=investable_prob_label,
        confidence_tier_investable_pool=investable_tier,
        stance_investable_pool=investable_stance,
        investable_benchmark_count=investable_benchmark_count,
        dual_track_delta=_dual_track_delta,
        probability_path_b_temp_scaled=_path_b_prob_scaled,
        probability_path_b_temp_scaled_label=_path_b_label,
        confidence_tier_path_b=_path_b_tier,
        stance_path_b=_path_b_stance,
    )
    return summary, detail_df.sort_values("benchmark").reset_index(drop=True)
