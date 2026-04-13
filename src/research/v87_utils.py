"""Shared helpers for the v87-v96 classification and hybrid research cycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from config.features import MODEL_FEATURE_OVERRIDES
from src.models.calibration import compute_ece
from src.models.evaluation import summarize_binary_predictions
from src.models.forecast_diagnostics import summarize_prediction_diagnostics
from src.models.policy_metrics import PolicySummary, evaluate_hold_fraction_series, evaluate_policy_series
from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    BENCHMARKS,
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RESULTS_DIR,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    compute_metrics,
    get_connection,
    load_feature_matrix,
    load_relative_series,
)
from src.research.v66_utils import (
    benchmark_quality_weights,
    build_consensus_frame,
    load_ensemble_oos_sequences,
)


PRIMARY_BINARY_TARGET = "benchmark_underperform_0pct"
ACTION_THRESHOLD = 0.03
CALIBRATION_MIN_HISTORY = 24


@dataclass(frozen=True)
class BinaryMetricBundle:
    """Binary-classification metrics plus calibration diagnostics."""

    n_obs: int
    accuracy: float
    balanced_accuracy: float
    brier_score: float
    log_loss: float
    precision: float
    recall: float
    base_rate: float
    predicted_positive_rate: float
    ece_10: float


def available_feature_families(feature_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return the curated feature-family registry filtered to available columns."""
    registry = {
        "lean_baseline": list(RIDGE_FEATURES_12),
        "price_momentum": [
            "mom_3m",
            "mom_6m",
            "mom_12m",
            "high_52w",
            "vol_63d",
        ],
        "valuation": [
            "pe_ratio",
            "pb_ratio",
            "roe",
            "roe_net_income_ttm",
            "roe_trend",
            "pgr_pe_vs_market_pe",
            "pgr_price_to_book_relative",
            "equity_risk_premium",
        ],
        "extended_pgr_operations": [
            "combined_ratio_ttm",
            "monthly_combined_ratio_delta",
            "pif_growth_yoy",
            "gainshare_est",
            "cr_acceleration",
            "pif_growth_acceleration",
            "channel_mix_agency_pct",
            "npw_growth_yoy",
            "npw_per_pif_yoy",
            "npw_vs_npe_spread_pct",
            "underwriting_income",
            "underwriting_income_3m",
            "underwriting_income_growth_yoy",
            "underwriting_margin_ttm",
            "unearned_premium_growth_yoy",
            "unearned_premium_to_npw_ratio",
            "book_value_per_share_growth_yoy",
            "pgr_premium_to_surplus",
            "direct_channel_pif_share_ttm",
            "channel_mix_direct_pct_yoy",
        ],
        "investment_capital": [
            "investment_income_growth_yoy",
            "investment_book_yield",
            "duration_rate_shock_3m",
            "buyback_yield",
            "buyback_acceleration",
            "unrealized_gain_pct_equity",
        ],
        "macro_rates_spreads": [
            "yield_slope",
            "yield_curvature",
            "real_rate_10y",
            "real_yield_change_6m",
            "breakeven_inflation_10y",
            "breakeven_momentum_3m",
            "baa10y_spread",
            "credit_spread_hy",
            "credit_spread_ratio",
            "excess_bond_premium_proxy",
            "mortgage_spread_30y_10y",
            "term_premium_10y",
            "nfci",
            "vix",
            "usd_broad_return_3m",
            "usd_momentum_6m",
            "wti_return_3m",
            "gold_vs_treasury_6m",
            "commodity_equity_momentum",
        ],
        "inflation_insurance": [
            "vmt_yoy",
            "legal_services_ppi_relative",
            "gasoline_retail_sales_delta",
            "used_car_cpi_yoy",
            "medical_cpi_yoy",
            "severity_index_yoy",
            "ppi_auto_ins_yoy",
            "motor_vehicle_ins_cpi_yoy",
            "rate_adequacy_gap_yoy",
            "auto_pricing_power_spread",
        ],
        "benchmark_context": [
            "pgr_vs_kie_6m",
            "pgr_vs_peers_6m",
            "pgr_vs_vfh_6m",
            "vwo_vxus_spread_6m",
        ],
        "regime_indicators": [
            "combined_ratio_ttm",
            "vix",
            "yield_slope",
            "credit_spread_hy",
        ],
    }
    available = set(feature_df.columns)
    return {
        family: [feature for feature in features if feature in available]
        for family, features in registry.items()
    }


def unique_features(*feature_lists: list[str]) -> list[str]:
    """Return unique features preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for features in feature_lists:
        for feature in features:
            if feature not in seen:
                seen.add(feature)
                ordered.append(feature)
    return ordered


def load_research_inputs() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load the pre-holdout feature matrix and benchmark relative returns."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rel_map = {
            benchmark: load_relative_series(conn, benchmark, horizon=6)
            for benchmark in BENCHMARKS
        }
    finally:
        conn.close()
    return feature_df, rel_map


def build_target_series(
    rel_series: pd.Series,
    target_name: str,
) -> pd.Series:
    """Return a binary or ternary target series from one benchmark path."""
    if target_name == "benchmark_underperform_0pct":
        target = (rel_series < 0.0).astype(int)
    elif target_name == "actionable_sell_3pct":
        target = (rel_series < -ACTION_THRESHOLD).astype(int)
    elif target_name == "ternary_neutral_3pct":
        target = pd.Series(
            np.select(
                [rel_series < -ACTION_THRESHOLD, rel_series > ACTION_THRESHOLD],
                [2, 0],
                default=1,
            ),
            index=rel_series.index,
            name=target_name,
        )
    else:
        raise ValueError(f"Unsupported benchmark target '{target_name}'.")
    target.name = target_name
    return target


def build_basket_target_series(
    rel_map: dict[str, pd.Series],
    target_name: str,
) -> pd.Series:
    """Return a basket-level target series indexed by month."""
    rel_df = pd.DataFrame(rel_map).sort_index()
    if rel_df.empty:
        return pd.Series(dtype=float, name=target_name)
    rel_df = rel_df.dropna(how="all")

    if target_name == "basket_underperform_0pct":
        series = (rel_df.mean(axis=1) < 0.0).astype(int)
    elif target_name == "basket_actionable_sell_3pct":
        series = (rel_df.mean(axis=1) < -ACTION_THRESHOLD).astype(int)
    elif target_name == "breadth_underperform_majority":
        threshold = max(1, int(np.floor(rel_df.shape[1] / 2.0)) + 1)
        series = (rel_df.lt(0.0).sum(axis=1) >= threshold).astype(int)
    else:
        raise ValueError(f"Unsupported basket target '{target_name}'.")
    series.name = target_name
    return series


def describe_target_series(
    target: pd.Series,
    target_name: str,
) -> dict[str, float | int | str]:
    """Describe the class balance of one target formulation."""
    clean = target.dropna()
    value_counts = clean.value_counts(normalize=True).sort_index()
    row: dict[str, float | int | str] = {
        "target": target_name,
        "n_obs": int(len(clean)),
        "n_classes": int(clean.nunique()),
    }
    for label, share in value_counts.items():
        row[f"class_{label}_share"] = float(share)
    if clean.nunique() == 2:
        row["positive_rate"] = float(clean.mean())
    return row


def _outer_time_series_splitter(n_obs: int) -> TimeSeriesSplit:
    available = n_obs - MAX_TRAIN_MONTHS - GAP_MONTHS
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    return TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=MAX_TRAIN_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_MONTHS,
    )


def _impute_fold(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = x_train.copy()
    test = x_test.copy()
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for col_idx in range(train.shape[1]):
        train[np.isnan(train[:, col_idx]), col_idx] = medians[col_idx]
        test[np.isnan(test[:, col_idx]), col_idx] = medians[col_idx]
    return train, test


def logistic_factory(
    *,
    class_weight: str | None = None,
    penalty: str = "l2",
    c_value: float = 1.0,
) -> Callable[[], LogisticRegression]:
    """Return a small-sample logistic model factory."""
    # sklearn 1.8+: penalty= is deprecated; use l1_ratio to select regularisation type.
    # l1_ratio=0  → L2 (equivalent to former penalty='l2', the default)
    # l1_ratio=1  → L1 (equivalent to former penalty='l1'); saga supports l1_ratio
    if penalty == "l1":
        l1_ratio: float = 1.0
        solver = "saga"
    else:
        l1_ratio = 0.0
        solver = "lbfgs"
    return lambda: LogisticRegression(
        l1_ratio=l1_ratio,
        C=c_value,
        class_weight=class_weight,
        max_iter=5000,
        solver=solver,
        random_state=42,
    )


def hist_gbt_factory(
    *,
    max_depth: int = 2,
    max_iter: int = 150,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 10,
) -> Callable[[], HistGradientBoostingClassifier]:
    """Return a conservative nonlinear classifier factory."""
    return lambda: HistGradientBoostingClassifier(
        max_depth=max_depth,
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=1.0,
        random_state=42,
    )


def binary_metric_bundle(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> BinaryMetricBundle:
    """Compute pooled binary metrics from probabilities."""
    y_true_series = pd.Series(y_true, name="y_true")
    y_prob_series = pd.Series(y_prob, name="y_prob")
    summary = summarize_binary_predictions(y_prob_series, y_true_series)
    clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    return BinaryMetricBundle(
        n_obs=summary.n_obs,
        accuracy=float(summary.accuracy),
        balanced_accuracy=float(summary.balanced_accuracy),
        brier_score=float(summary.brier_score),
        log_loss=float(log_loss(y_true, clipped, labels=[0, 1])),
        precision=float(summary.precision),
        recall=float(summary.recall),
        base_rate=float(summary.base_rate),
        predicted_positive_rate=float(summary.predicted_positive_rate),
        ece_10=float(compute_ece(clipped, y_true, n_bins=10)),
    )


def evaluate_binary_time_series(
    x_df: pd.DataFrame,
    y_series: pd.Series,
    model_factory: Callable[[], Any],
) -> pd.DataFrame:
    """Return OOS binary predictions for one time-indexed dataset."""
    aligned = x_df.join(y_series, how="inner")
    aligned = aligned.dropna(subset=[y_series.name])
    if aligned.empty:
        return pd.DataFrame(columns=["date", "y_true", "y_prob"])

    x_values = aligned[x_df.columns].to_numpy(dtype=float)
    y_values = aligned[y_series.name].to_numpy(dtype=int)
    dates = pd.DatetimeIndex(aligned.index)

    rows: list[dict[str, Any]] = []
    for train_idx, test_idx in _outer_time_series_splitter(len(aligned)).split(x_values):
        y_train = y_values[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        x_train, x_test = _impute_fold(x_values[train_idx], x_values[test_idx])
        model = model_factory()
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test)[:, 1]
        else:
            decision = model.decision_function(x_test)
            y_prob = 1.0 / (1.0 + np.exp(-decision))
        for offset, row_idx in enumerate(test_idx):
            rows.append(
                {
                    "date": dates[row_idx],
                    "y_true": int(y_values[row_idx]),
                    "y_prob": float(y_prob[offset]),
                }
            )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def evaluate_separate_benchmark_binary(
    feature_df: pd.DataFrame,
    rel_map: dict[str, pd.Series],
    target_name: str,
    feature_columns: list[str],
    model_factory: Callable[[], Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate one binary model separately for each benchmark."""
    rows: list[dict[str, Any]] = []
    sequence_map: dict[str, pd.DataFrame] = {}
    pooled_true: list[np.ndarray] = []
    pooled_prob: list[np.ndarray] = []
    features = [feature for feature in feature_columns if feature in feature_df.columns]

    for benchmark in BENCHMARKS:
        rel_series = rel_map.get(benchmark)
        if rel_series is None or rel_series.empty:
            continue
        x_base, _ = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
        x_df = x_base[features].copy()
        target = build_target_series(rel_series, target_name)
        pred_df = evaluate_binary_time_series(x_df, target, model_factory)
        if pred_df.empty:
            continue
        metrics = binary_metric_bundle(
            pred_df["y_true"].to_numpy(dtype=int),
            pred_df["y_prob"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "benchmark": benchmark,
                "target": target_name,
                "n_features": len(features),
                **metrics.__dict__,
            }
        )
        dated = pred_df.copy()
        dated["date"] = pd.to_datetime(dated["date"])
        sequence_map[benchmark] = dated.set_index("date").sort_index()
        pooled_true.append(pred_df["y_true"].to_numpy(dtype=int))
        pooled_prob.append(pred_df["y_prob"].to_numpy(dtype=float))

    if pooled_true:
        pooled_metrics = binary_metric_bundle(
            np.concatenate(pooled_true),
            np.concatenate(pooled_prob),
        )
        rows.append(
            {
                "benchmark": "POOLED",
                "target": target_name,
                "n_features": len(features),
                **pooled_metrics.__dict__,
            }
        )

    return pd.DataFrame(rows), sequence_map


def build_panel_dataset(
    feature_df: pd.DataFrame,
    rel_map: dict[str, pd.Series],
    target_name: str,
    feature_columns: list[str],
    include_benchmark_dummies: bool,
) -> pd.DataFrame:
    """Build a benchmark-month panel for pooled classification."""
    features = [feature for feature in feature_columns if feature in feature_df.columns]
    rows: list[pd.DataFrame] = []
    for benchmark in BENCHMARKS:
        rel_series = rel_map.get(benchmark)
        if rel_series is None or rel_series.empty:
            continue
        x_base, _ = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
        target = build_target_series(rel_series, target_name)
        aligned = x_base[features].join(target, how="inner").dropna(subset=[target_name]).copy()
        if aligned.empty:
            continue
        aligned["benchmark"] = benchmark
        aligned["date"] = pd.DatetimeIndex(aligned.index)
        rows.append(aligned.reset_index(drop=True))

    if not rows:
        return pd.DataFrame(columns=["date", "benchmark", target_name, *features])

    panel = pd.concat(rows, ignore_index=True).sort_values(["date", "benchmark"]).reset_index(drop=True)
    if include_benchmark_dummies:
        for benchmark in BENCHMARKS:
            panel[f"bm_{benchmark}"] = (panel["benchmark"] == benchmark).astype(int)
    return panel


def evaluate_panel_binary(
    panel_df: pd.DataFrame,
    target_name: str,
    feature_columns: list[str],
    model_factory: Callable[[], Any],
) -> pd.DataFrame:
    """Evaluate a pooled-panel binary model with month-grouped WFO splits."""
    if panel_df.empty:
        return pd.DataFrame(columns=["date", "benchmark", "y_true", "y_prob"])

    x_cols = [feature for feature in feature_columns if feature in panel_df.columns]
    unique_dates = pd.Index(sorted(pd.unique(panel_df["date"])))
    splitter = _outer_time_series_splitter(len(unique_dates))

    rows: list[dict[str, Any]] = []
    for train_month_idx, test_month_idx in splitter.split(np.arange(len(unique_dates))):
        train_dates = set(unique_dates[train_month_idx])
        test_dates = set(unique_dates[test_month_idx])
        train_df = panel_df[panel_df["date"].isin(train_dates)].copy()
        test_df = panel_df[panel_df["date"].isin(test_dates)].copy()
        y_train = train_df[target_name].to_numpy(dtype=int)
        if len(np.unique(y_train)) < 2:
            continue

        x_train, x_test = _impute_fold(
            train_df[x_cols].to_numpy(dtype=float),
            test_df[x_cols].to_numpy(dtype=float),
        )
        model = model_factory()
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test)[:, 1]
        else:
            decision = model.decision_function(x_test)
            y_prob = 1.0 / (1.0 + np.exp(-decision))

        for idx in range(len(test_df)):
            rows.append(
                {
                    "date": test_df.iloc[idx]["date"],
                    "benchmark": str(test_df.iloc[idx]["benchmark"]),
                    "y_true": int(test_df.iloc[idx][target_name]),
                    "y_prob": float(y_prob[idx]),
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "benchmark"]).reset_index(drop=True)


def summarize_prediction_rows(
    prediction_df: pd.DataFrame,
    *,
    group_label: str,
    target_name: str,
    feature_set_name: str,
    model_name: str,
) -> pd.DataFrame:
    """Summarize panel or per-benchmark binary predictions into metric rows."""
    if prediction_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = {"POOLED": prediction_df}
    if "benchmark" in prediction_df.columns:
        grouped.update(dict(tuple(prediction_df.groupby("benchmark"))))

    for benchmark, frame in grouped.items():
        metrics = binary_metric_bundle(
            frame["y_true"].to_numpy(dtype=int),
            frame["y_prob"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "group_label": group_label,
                "benchmark": benchmark,
                "target": target_name,
                "feature_set": feature_set_name,
                "model_name": model_name,
                **metrics.__dict__,
            }
        )
    return pd.DataFrame(rows)


def choose_primary_target(results_df: pd.DataFrame) -> str:
    """Choose the forward binary target from the v87 taxonomy results."""
    pooled = results_df[results_df["benchmark"] == "POOLED"].copy()
    if pooled.empty:
        return PRIMARY_BINARY_TARGET
    if "positive_rate" in pooled.columns:
        pooled = pooled[pooled["positive_rate"].between(0.15, 0.85, inclusive="both")]
    if pooled.empty:
        return PRIMARY_BINARY_TARGET
    ranked = pooled.sort_values(
        ["balanced_accuracy", "ece_10", "brier_score"],
        ascending=[False, True, True],
    )
    return str(ranked.iloc[0]["target"])


def choose_best_feature_set(results_df: pd.DataFrame) -> str:
    """Choose the best feature set from the v88 sweep."""
    pooled = results_df[results_df["benchmark"] == "POOLED"].copy()
    if pooled.empty:
        return "lean_baseline"
    ranked = pooled.sort_values(
        ["balanced_accuracy", "ece_10", "brier_score"],
        ascending=[False, True, True],
    )
    return str(ranked.iloc[0]["feature_set"])


def load_v87_results() -> pd.DataFrame | None:
    """Load the v87 results CSV if present."""
    path = RESULTS_DIR / "v87_target_taxonomy_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_v88_results() -> pd.DataFrame | None:
    """Load the v88 feature sweep CSV if present."""
    path = RESULTS_DIR / "v88_feature_sweep_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def resolve_primary_target() -> str:
    """Return the chosen forward target if v87 has already been run."""
    v87_df = load_v87_results()
    if v87_df is None:
        return PRIMARY_BINARY_TARGET
    return choose_primary_target(v87_df)


def resolve_best_feature_set() -> str:
    """Return the chosen forward feature-set name if v88 has already run."""
    v88_df = load_v88_results()
    if v88_df is None:
        return "lean_baseline"
    return choose_best_feature_set(v88_df)


def build_feature_sets(feature_df: pd.DataFrame) -> dict[str, list[str]]:
    """Build named feature sets used across v88-v96."""
    families = available_feature_families(feature_df)
    lean = families["lean_baseline"]
    inflation = families["inflation_insurance"]
    benchmark_context = families["benchmark_context"]
    extended_ops = [
        feature for feature in families["extended_pgr_operations"] if feature not in lean
    ]
    valuation = families["valuation"]
    investment = families["investment_capital"]

    feature_sets = {
        "lean_baseline": lean,
        "lean_plus_extended_ops": unique_features(lean, extended_ops),
        "lean_plus_valuation": unique_features(lean, valuation),
        "lean_plus_inflation": unique_features(lean, inflation),
        "lean_plus_investment": unique_features(lean, investment),
        "lean_plus_benchmark_context": unique_features(lean, benchmark_context),
        "lean_plus_ops_and_context": unique_features(lean, extended_ops, benchmark_context),
        "lean_plus_inflation_and_context": unique_features(lean, inflation, benchmark_context),
        "lean_plus_all_curated": unique_features(
            lean,
            extended_ops,
            valuation,
            inflation,
            investment,
            benchmark_context,
        ),
    }
    return {name: [feature for feature in features if feature in feature_df.columns] for name, features in feature_sets.items()}


def prequential_logistic_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_history: int = CALIBRATION_MIN_HISTORY,
) -> np.ndarray:
    """Apply expanding-history logistic calibration to model probabilities."""
    calibrated = np.asarray(y_prob, dtype=float).copy()
    clipped = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    for idx in range(len(clipped)):
        if idx < min_history:
            continue
        history_y = np.asarray(y_true[:idx], dtype=int)
        if len(np.unique(history_y)) < 2:
            continue
        history_x = clipped[:idx].reshape(-1, 1)
        model = LogisticRegression(C=1e6, max_iter=2000, random_state=42)
        model.fit(history_x, history_y)
        calibrated[idx] = float(model.predict_proba(np.array([[clipped[idx]]], dtype=float))[0, 1])
    return np.clip(calibrated, 1e-6, 1.0 - 1e-6)


def classifier_hold_fraction(
    prob_sell: pd.Series,
    lower: float,
    upper: float,
) -> pd.Series:
    """Map sell probabilities to a hold fraction."""
    hold = np.where(
        prob_sell >= upper,
        0.0,
        np.where(prob_sell <= lower, 1.0, 0.5),
    )
    return pd.Series(hold, index=prob_sell.index, name="hold_fraction")


def hybrid_hold_fraction(
    predicted_relative_return: pd.Series,
    prob_sell: pd.Series,
    lower: float,
    upper: float,
    action_threshold: float = ACTION_THRESHOLD,
) -> pd.Series:
    """Map regression + classifier signals into a conservative hold fraction."""
    aligned = pd.concat(
        [predicted_relative_return.rename("predicted"), prob_sell.rename("prob_sell")],
        axis=1,
    ).dropna()
    hold = np.where(
        aligned["prob_sell"] >= upper,
        np.where(aligned["predicted"] < -action_threshold, 0.0, 0.5),
        np.where(
            aligned["prob_sell"] <= lower,
            np.where(aligned["predicted"] > action_threshold, 1.0, 0.5),
            0.5,
        ),
    )
    return pd.Series(hold, index=aligned.index, name="hold_fraction")


def summarize_hold_series(
    variant: str,
    hold_fraction: pd.Series,
    realized_relative_return: pd.Series,
) -> dict[str, float | int | str]:
    """Summarize one explicit hold-fraction policy path."""
    summary = evaluate_hold_fraction_series(hold_fraction, realized_relative_return)
    changes = hold_fraction.ne(hold_fraction.shift(1)).sum()
    return {
        "variant": variant,
        "n_obs": int(summary.n_obs),
        "avg_hold_fraction": float(summary.avg_hold_fraction),
        "mean_policy_return": float(summary.mean_policy_return),
        "median_policy_return": float(summary.median_policy_return),
        "cumulative_policy_return": float(summary.cumulative_policy_return),
        "positive_utility_rate": float(summary.positive_utility_rate),
        "regret_vs_oracle": float(summary.regret_vs_oracle),
        "uplift_vs_sell_all": float(summary.uplift_vs_sell_all),
        "uplift_vs_sell_50": float(summary.uplift_vs_sell_50),
        "uplift_vs_hold_all": float(summary.uplift_vs_hold_all),
        "capture_ratio": float(summary.capture_ratio),
        "hold_fraction_changes": int(changes),
    }


def build_quality_weighted_regression_consensus(
    feature_df: pd.DataFrame,
    rel_map: dict[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, float], dict[str, pd.DataFrame]]:
    """Rebuild the current regression baseline consensus used for production."""
    sequences = load_ensemble_oos_sequences(
        feature_df,
        pd.DataFrame(rel_map),
        shrinkage_alpha=0.50,
    )
    quality_rows: list[dict[str, float | int | str]] = []
    for benchmark, frame in sequences.items():
        summary = summarize_prediction_diagnostics(frame["y_hat"], frame["y_true"])
        quality_rows.append({"benchmark": benchmark, **summary})
    quality_df = pd.DataFrame(quality_rows)
    weights = benchmark_quality_weights(quality_df, score_col="nw_ic", lambda_mix=0.25)
    consensus = build_consensus_frame(sequences, weights=weights)
    return consensus, weights, sequences


def aggregate_probability_sequences(
    sequence_map: dict[str, pd.DataFrame],
    *,
    probability_column: str = "y_prob",
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Aggregate benchmark probabilities into one weighted sell-probability series."""
    if not sequence_map:
        return pd.Series(dtype=float, name="prob_sell")

    prob_frames: list[pd.Series] = []
    for benchmark, frame in sequence_map.items():
        if probability_column not in frame.columns:
            continue
        prob_frames.append(frame[probability_column].rename(benchmark))
    if not prob_frames:
        return pd.Series(dtype=float, name="prob_sell")

    prob_df = pd.concat(prob_frames, axis=1).sort_index()
    if weights is None:
        return prob_df.mean(axis=1).rename("prob_sell")

    aligned_weights = pd.Series(
        {benchmark: float(weights.get(benchmark, 0.0)) for benchmark in prob_df.columns},
        dtype=float,
    )
    weight_sum = float(aligned_weights.sum())
    if weight_sum <= 1e-12:
        aligned_weights[:] = 1.0 / len(aligned_weights)
    else:
        aligned_weights = aligned_weights / weight_sum
    return prob_df.mul(aligned_weights, axis=1).sum(axis=1).rename("prob_sell")


def summarize_monthly_policy_path(
    variant: str,
    hold_fraction: pd.Series,
    realized_relative_return: pd.Series,
) -> dict[str, float | int | str]:
    """Summarize month-to-month stability for one decision path."""
    aligned = pd.concat([hold_fraction.rename("hold_fraction"), realized_relative_return.rename("realized")], axis=1).dropna()
    if aligned.empty:
        return {
            "variant": variant,
            "review_months": 0,
            "mean_hold_fraction": float("nan"),
            "hold_fraction_changes": 0,
            "defer_rate": float("nan"),
            "sell_all_rate": float("nan"),
            "hold_all_rate": float("nan"),
            "mean_policy_return": float("nan"),
            "capture_ratio": float("nan"),
        }
    hold = aligned["hold_fraction"]
    policy = evaluate_hold_fraction_series(hold, aligned["realized"])
    return {
        "variant": variant,
        "review_months": int(len(aligned)),
        "mean_hold_fraction": float(hold.mean()),
        "hold_fraction_changes": int(hold.ne(hold.shift(1)).sum()),
        "defer_rate": float((hold == 0.5).mean()),
        "sell_all_rate": float((hold == 0.0).mean()),
        "hold_all_rate": float((hold == 1.0).mean()),
        "mean_policy_return": float(policy.mean_policy_return),
        "capture_ratio": float(policy.capture_ratio),
    }


def write_markdown_summary(path: Path, title: str, body_lines: list[str]) -> None:
    """Write a small markdown summary artifact."""
    text = "\n".join([f"# {title}", "", *body_lines, ""])
    path.write_text(text, encoding="utf-8")


def feature_set_from_name(feature_df: pd.DataFrame, feature_set_name: str) -> list[str]:
    """Resolve a named feature set from the curated registry."""
    feature_sets = build_feature_sets(feature_df)
    return feature_sets.get(feature_set_name, feature_sets["lean_baseline"])


def probability_candidate_sequences(
    feature_df: pd.DataFrame,
    rel_map: dict[str, pd.Series],
    target_name: str,
    feature_set_name: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Rebuild the main benchmark-level classifier candidates used in v92-v95."""
    feature_columns = feature_set_from_name(feature_df, feature_set_name)

    separate_metrics, separate_sequences = evaluate_separate_benchmark_binary(
        feature_df,
        rel_map,
        target_name=target_name,
        feature_columns=feature_columns,
        model_factory=logistic_factory(class_weight="balanced", c_value=0.5),
    )

    panel_fe = build_panel_dataset(
        feature_df,
        rel_map,
        target_name=target_name,
        feature_columns=feature_columns,
        include_benchmark_dummies=True,
    )
    fe_features = feature_columns + [column for column in panel_fe.columns if column.startswith("bm_")]

    pooled_logistic_preds = evaluate_panel_binary(
        panel_fe,
        target_name=target_name,
        feature_columns=fe_features,
        model_factory=logistic_factory(class_weight="balanced", c_value=0.5),
    )
    pooled_histgb_preds = evaluate_panel_binary(
        panel_fe,
        target_name=target_name,
        feature_columns=fe_features,
        model_factory=hist_gbt_factory(max_depth=2, max_iter=120, learning_rate=0.05, min_samples_leaf=10),
    )

    def _panel_to_sequence_map(pred_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if pred_df.empty:
            return {}
        return {
            str(benchmark): frame.assign(date=pd.to_datetime(frame["date"]))
            .set_index("date")
            .sort_index()
            for benchmark, frame in pred_df.groupby("benchmark")
        }

    _ = separate_metrics  # keeps the evaluation side-effect explicit for readability
    return {
        "separate_logistic_balanced": separate_sequences,
        "pooled_fixed_effects_logistic_balanced": _panel_to_sequence_map(pooled_logistic_preds),
        "pooled_fixed_effects_histgb_depth2": _panel_to_sequence_map(pooled_histgb_preds),
    }


def basket_probability_candidates(
    feature_df: pd.DataFrame,
    rel_map: dict[str, pd.Series],
    feature_set_name: str,
) -> dict[str, pd.DataFrame]:
    """Return basket-level classifier candidates for v93-v95."""
    feature_columns = feature_set_from_name(feature_df, feature_set_name)
    x_df = feature_df[feature_columns].copy()
    candidates: dict[str, pd.DataFrame] = {}
    for target_name in (
        "basket_underperform_0pct",
        "basket_actionable_sell_3pct",
        "breadth_underperform_majority",
    ):
        target = build_basket_target_series(rel_map, target_name)
        pred_df = evaluate_binary_time_series(
            x_df,
            target,
            logistic_factory(class_weight="balanced", c_value=0.5),
        )
        if not pred_df.empty:
            candidates[target_name] = pred_df
    return candidates


def current_production_policy_summary(
    consensus_frame: pd.DataFrame,
) -> PolicySummary:
    """Summarize the current regression-only production decision baseline."""
    return evaluate_policy_series(
        predicted=consensus_frame["predicted"],
        realized_relative_return=consensus_frame["realized"],
        policy_name="neutral_band_3pct",
    )


def summarize_regression_baseline(
    consensus_frame: pd.DataFrame,
) -> dict[str, float | int | str]:
    """Return current production-style regression-only policy metrics."""
    forecast = compute_metrics(
        consensus_frame["realized"].to_numpy(dtype=float),
        consensus_frame["predicted"].to_numpy(dtype=float),
    )
    hold_fraction = pd.Series(
        np.where(
            consensus_frame["predicted"] > ACTION_THRESHOLD,
            1.0,
            np.where(consensus_frame["predicted"] < -ACTION_THRESHOLD, 0.0, 0.5),
        ),
        index=consensus_frame.index,
        name="hold_fraction",
    )
    policy_summary = summarize_hold_series(
        "regression_only_quality_weighted",
        hold_fraction,
        consensus_frame["realized"],
    )
    return {
        **policy_summary,
        "oos_r2": float(forecast["r2"]),
        "ic": float(forecast["ic"]),
        "hit_rate": float(forecast["hit_rate"]),
        "mae": float(forecast["mae"]),
    }


def current_production_feature_sets() -> dict[str, list[str]]:
    """Expose the model-specific feature overrides used in production."""
    return {name: list(features) for name, features in MODEL_FEATURE_OVERRIDES.items()}
