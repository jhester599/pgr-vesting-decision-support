"""v128 -- Benchmark-specific full feature search for classification models.

This research harness explores the full point-in-time-safe feature universe for
the benchmark-specific classification stack while preserving the current model
family used by the shadow classifier:

- target: ``actionable_sell_3pct``
- model: separate benchmark-specific balanced logistic
- validation: strict rolling WFO with purge / embargo
- calibration: prequential logistic calibration

The search is benchmark-specific across the 10 current classifier benchmarks:
``VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE, VGT, VIG``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.features import INVESTABLE_CLASSIFIER_BENCHMARKS, PRIMARY_FORECAST_UNIVERSE
from src.processing.feature_engineering import get_X_y_relative, get_feature_columns
from src.research.v37_utils import (
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RESULTS_DIR,
    TEST_SIZE_MONTHS,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    print_footer,
    print_header,
    save_results,
)
from src.research.v87_utils import (
    available_feature_families,
    binary_metric_bundle,
    build_target_series,
    feature_set_from_name,
    prequential_logistic_calibration,
    write_markdown_summary,
)


ACTIONABLE_TARGET = "actionable_sell_3pct"
INCUMBENT_FEATURE_SET = "lean_baseline"
MIN_FEATURE_OBS = 60
LOWER_THRESHOLD = 0.30
UPPER_THRESHOLD = 0.70
MAX_FEATURES = 12
FORWARD_MIN_BA_IMPROVEMENT = 0.005
FORWARD_MAX_ECE_WORSENING = 0.01
FORWARD_MAX_BRIER_WORSENING = 0.005
WINNER_MAX_ECE_WORSENING = 0.01
WINNER_MAX_BRIER_WORSENING = 0.005
REGULARIZED_SELECTION_EPS = 1e-8

L1_C_GRID = (0.05, 0.10, 0.25, 0.50, 1.00)
ELASTIC_NET_C_GRID = (0.05, 0.10, 0.25, 0.50)
ELASTIC_NET_L1_GRID = (0.20, 0.50, 0.80)
RIDGE_C_GRID = (0.05, 0.10, 0.25, 0.50, 1.00)

FEATURE_INVENTORY_PATH = RESULTS_DIR / "v128_feature_inventory.csv"
BASELINE_PATH = RESULTS_DIR / "v128_baseline_metrics.csv"
SINGLE_FEATURE_PATH = RESULTS_DIR / "v128_single_feature_results.csv"
FORWARD_TRACE_PATH = RESULTS_DIR / "v128_forward_stepwise_trace.csv"
REGULARIZED_SELECTION_PATH = RESULTS_DIR / "v128_regularized_selection_detail.csv"
REGULARIZED_COMPARISON_PATH = RESULTS_DIR / "v128_regularized_comparison.csv"
COMPARISON_PATH = RESULTS_DIR / "v128_benchmark_feature_search_comparison.csv"
FEATURE_MAP_PATH = RESULTS_DIR / "v128_benchmark_feature_map.csv"
SUMMARY_PATH = RESULTS_DIR / "v128_benchmark_feature_search_summary.md"


@dataclass(frozen=True)
class BenchmarkDataset:
    """Aligned benchmark-specific feature / target dataset."""

    benchmark: str
    x_df: pd.DataFrame
    y_series: pd.Series
    eligible_features: list[str]


def benchmark_universe() -> list[str]:
    """Return the 10-benchmark v128 classifier universe."""
    return list(dict.fromkeys(PRIMARY_FORECAST_UNIVERSE + INVESTABLE_CLASSIFIER_BENCHMARKS))


def candidate_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return the full non-target feature universe available to v128."""
    return list(get_feature_columns(feature_df))


def _feature_family_lookup(feature_df: pd.DataFrame) -> dict[str, str]:
    """Map each feature to its curated family labels when available."""
    families = available_feature_families(feature_df)
    inverse: dict[str, list[str]] = {}
    for family, features in families.items():
        for feature in features:
            inverse.setdefault(feature, []).append(family)
    return {
        feature: ",".join(sorted(inverse.get(feature, ["unassigned"])))
        for feature in candidate_feature_columns(feature_df)
    }


def load_v128_inputs(
    *,
    benchmarks: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load the full research feature matrix and relative-return series."""
    selected = benchmarks or benchmark_universe()
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rel_map = {
            benchmark: load_relative_series(conn, benchmark, horizon=6)
            for benchmark in selected
        }
    finally:
        conn.close()
    return feature_df, rel_map


def build_benchmark_datasets(
    feature_df: pd.DataFrame,
    rel_map: dict[str, pd.Series],
    *,
    min_feature_obs: int = MIN_FEATURE_OBS,
) -> dict[str, BenchmarkDataset]:
    """Build one aligned benchmark dataset per classification target."""
    datasets: dict[str, BenchmarkDataset] = {}
    for benchmark in benchmark_universe():
        rel_series = rel_map.get(benchmark)
        if rel_series is None or rel_series.empty:
            continue
        x_base, _ = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
        target = build_target_series(rel_series, ACTIONABLE_TARGET)
        aligned = x_base.join(target, how="inner").dropna(subset=[ACTIONABLE_TARGET]).copy()
        if aligned.empty:
            continue
        x_df = aligned[x_base.columns].copy()
        y_series = aligned[ACTIONABLE_TARGET].astype(int).copy()
        eligible_features = [
            feature
            for feature in x_df.columns
            if int(x_df[feature].notna().sum()) >= min_feature_obs
        ]
        datasets[benchmark] = BenchmarkDataset(
            benchmark=benchmark,
            x_df=x_df,
            y_series=y_series,
            eligible_features=eligible_features,
        )
    return datasets


def build_feature_inventory(
    datasets: dict[str, BenchmarkDataset],
    feature_family_lookup: dict[str, str],
) -> pd.DataFrame:
    """Return the benchmark-by-feature availability inventory."""
    rows: list[dict[str, object]] = []
    for benchmark, dataset in datasets.items():
        total_obs = len(dataset.y_series)
        positive_rate = float(dataset.y_series.mean())
        for feature in dataset.x_df.columns:
            non_null = int(dataset.x_df[feature].notna().sum())
            rows.append(
                {
                    "benchmark": benchmark,
                    "feature": feature,
                    "feature_family": feature_family_lookup.get(feature, "unassigned"),
                    "n_obs_total": total_obs,
                    "n_obs_non_null": non_null,
                    "null_share": float(1.0 - (non_null / max(total_obs, 1))),
                    "eligible": bool(feature in dataset.eligible_features),
                    "positive_rate": positive_rate,
                }
            )
    return pd.DataFrame(rows).sort_values(["benchmark", "feature"]).reset_index(drop=True)


def _resolve_time_series_splitter(
    n_obs: int,
    *,
    allow_short_history: bool = False,
) -> TimeSeriesSplit | None:
    """Return the repo-standard rolling splitter, with a short-history fallback."""
    available = n_obs - MAX_TRAIN_MONTHS - GAP_MONTHS
    if available >= TEST_SIZE_MONTHS:
        n_splits = max(1, available // TEST_SIZE_MONTHS)
        return TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=MAX_TRAIN_MONTHS,
            test_size=TEST_SIZE_MONTHS,
            gap=GAP_MONTHS,
        )

    if not allow_short_history:
        return None

    return None


def _impute_fold_arrays(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaNs using training-fold medians."""
    train = x_train.copy()
    test = x_test.copy()
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for col_idx in range(train.shape[1]):
        train[np.isnan(train[:, col_idx]), col_idx] = medians[col_idx]
        test[np.isnan(test[:, col_idx]), col_idx] = medians[col_idx]
    return train, test


def evaluate_probability_time_series(
    x_df: pd.DataFrame,
    y_series: pd.Series,
    model_builder: Any,
    *,
    allow_short_history: bool = False,
) -> pd.DataFrame:
    """Return OOS binary predictions for one time-indexed dataset."""
    aligned = x_df.join(y_series, how="inner").dropna(subset=[y_series.name]).copy()
    if aligned.empty:
        return pd.DataFrame(columns=["date", "y_true", "y_prob"])

    splitter = _resolve_time_series_splitter(len(aligned), allow_short_history=allow_short_history)
    if splitter is None:
        return pd.DataFrame(columns=["date", "y_true", "y_prob"])

    x_values = aligned[x_df.columns].to_numpy(dtype=float)
    y_values = aligned[y_series.name].to_numpy(dtype=int)
    dates = pd.DatetimeIndex(aligned.index)

    rows: list[dict[str, object]] = []
    for train_idx, test_idx in splitter.split(x_values):
        y_train = y_values[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        x_train, x_test = _impute_fold_arrays(x_values[train_idx], x_values[test_idx])
        model = model_builder()
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


def _balanced_accuracy_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return balanced accuracy when both classes are present, else NaN."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def _clip_probs(values: np.ndarray) -> np.ndarray:
    """Clip probabilities to a numerically stable open interval."""
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def _format_feature_list(features: list[str]) -> str:
    """Serialize a feature list for artifact tables."""
    return "|".join(features)


def _candidate_metric_row(
    *,
    benchmark: str,
    method: str,
    feature_list: list[str],
    pred_df: pd.DataFrame,
    selector_penalty: str | None = None,
    production_eligible: bool = True,
) -> dict[str, object]:
    """Summarize calibrated prediction rows into the v128 comparison schema."""
    calibrated = pred_df.copy()
    calibrated["y_prob_cal"] = prequential_logistic_calibration(
        calibrated["y_true"].to_numpy(dtype=int),
        calibrated["y_prob"].to_numpy(dtype=float),
    )

    y_true = calibrated["y_true"].to_numpy(dtype=int)
    y_prob = _clip_probs(calibrated["y_prob_cal"].to_numpy(dtype=float))
    bundle = binary_metric_bundle(y_true, y_prob).__dict__
    covered_mask = (y_prob <= LOWER_THRESHOLD) | (y_prob >= UPPER_THRESHOLD)
    n_covered = int(covered_mask.sum())
    if n_covered > 0:
        covered_pred = (y_prob[covered_mask] >= 0.5).astype(int)
        ba_covered = _balanced_accuracy_safe(y_true[covered_mask], covered_pred)
    else:
        ba_covered = float("nan")

    return {
        "benchmark": benchmark,
        "method": method,
        "selector_penalty": selector_penalty,
        "n_features": int(len(feature_list)),
        "features": _format_feature_list(feature_list),
        "n_obs": int(bundle["n_obs"]),
        "n_covered": n_covered,
        "coverage": float(n_covered / max(bundle["n_obs"], 1)),
        "accuracy": float(bundle["accuracy"]),
        "balanced_accuracy": float(bundle["balanced_accuracy"]),
        "balanced_accuracy_covered": float(ba_covered),
        "brier_score": float(bundle["brier_score"]),
        "log_loss": float(bundle["log_loss"]),
        "precision": float(bundle["precision"]),
        "recall": float(bundle["recall"]),
        "base_rate": float(bundle["base_rate"]),
        "predicted_positive_rate": float(bundle["predicted_positive_rate"]),
        "ece_10": float(bundle["ece_10"]),
        "production_eligible": bool(production_eligible),
    }


class CachedEvaluator:
    """Cache benchmark-subset evaluations to keep the stepwise search tractable."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[str, ...]], tuple[dict[str, object], pd.DataFrame]] = {}

    def evaluate(
        self,
        dataset: BenchmarkDataset,
        feature_list: list[str],
        *,
        method: str,
        selector_penalty: str | None = None,
        production_eligible: bool = True,
    ) -> tuple[dict[str, object], pd.DataFrame]:
        """Evaluate one benchmark subset under the current shadow model family."""
        filtered = [feature for feature in feature_list if feature in dataset.x_df.columns]
        key = (dataset.benchmark, tuple(filtered))
        cached = self._cache.get(key)
        if cached is not None:
            row, pred_df = cached
            copied = dict(row)
            copied["method"] = method
            copied["selector_penalty"] = selector_penalty
            copied["production_eligible"] = production_eligible
            return copied, pred_df.copy()

        def _model_builder() -> LogisticRegression:
            return LogisticRegression(
                C=0.5,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=5000,
                random_state=42,
            )

        pred_df = evaluate_probability_time_series(
            dataset.x_df[filtered].copy(),
            dataset.y_series,
            _model_builder,
        )
        if pred_df.empty:
            row = {
                "benchmark": dataset.benchmark,
                "method": method,
                "selector_penalty": selector_penalty,
                "n_features": int(len(filtered)),
                "features": _format_feature_list(filtered),
                "n_obs": 0,
                "n_covered": 0,
                "coverage": float("nan"),
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "balanced_accuracy_covered": float("nan"),
                "brier_score": float("nan"),
                "log_loss": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "base_rate": float("nan"),
                "predicted_positive_rate": float("nan"),
                "ece_10": float("nan"),
                "production_eligible": bool(production_eligible),
            }
        else:
            row = _candidate_metric_row(
                benchmark=dataset.benchmark,
                method=method,
                feature_list=filtered,
                pred_df=pred_df,
                selector_penalty=selector_penalty,
                production_eligible=production_eligible,
            )
        self._cache[key] = (dict(row), pred_df.copy())
        return dict(row), pred_df.copy()


def _sort_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Sort candidate rows by the v128 objective and deterministic tie-breakers."""
    ranked = df.copy()
    ranked["_rank_ba_cov"] = ranked["balanced_accuracy_covered"].fillna(-np.inf)
    ranked["_rank_log_loss"] = ranked["log_loss"].fillna(np.inf)
    ranked["_rank_ece"] = ranked["ece_10"].fillna(np.inf)
    ranked["_rank_brier"] = ranked["brier_score"].fillna(np.inf)
    ranked["_rank_features"] = ranked["n_features"].fillna(np.inf)
    ranked = ranked.sort_values(
        [
            "_rank_ba_cov",
            "_rank_log_loss",
            "_rank_ece",
            "_rank_brier",
            "_rank_features",
            "features",
        ],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)
    return ranked.drop(
        columns=[
            "_rank_ba_cov",
            "_rank_log_loss",
            "_rank_ece",
            "_rank_brier",
            "_rank_features",
        ]
    )


def _passes_forward_gate(candidate: pd.Series, reference: pd.Series) -> bool:
    """Return whether a stepwise addition improves enough to be accepted."""
    ba_candidate = float(candidate["balanced_accuracy_covered"])
    ba_reference = float(reference["balanced_accuracy_covered"])
    if np.isnan(ba_candidate) or np.isnan(ba_reference):
        return False
    return bool(
        ba_candidate >= ba_reference + FORWARD_MIN_BA_IMPROVEMENT
        and float(candidate["ece_10"]) <= float(reference["ece_10"]) + FORWARD_MAX_ECE_WORSENING
        and float(candidate["brier_score"]) <= float(reference["brier_score"]) + FORWARD_MAX_BRIER_WORSENING
    )


def _passes_winner_guardrails(candidate: pd.Series, baseline: pd.Series) -> bool:
    """Return whether a candidate can replace the incumbent baseline."""
    ba_candidate = float(candidate["balanced_accuracy_covered"])
    ba_baseline = float(baseline["balanced_accuracy_covered"])
    if np.isnan(ba_candidate) or np.isnan(ba_baseline):
        return False
    return bool(
        ba_candidate >= ba_baseline
        and float(candidate["ece_10"]) <= float(baseline["ece_10"]) + WINNER_MAX_ECE_WORSENING
        and float(candidate["brier_score"]) <= float(baseline["brier_score"]) + WINNER_MAX_BRIER_WORSENING
    )


def aggregate_selection_detail(
    selection_df: pd.DataFrame,
    *,
    max_features: int = MAX_FEATURES,
) -> pd.DataFrame:
    """Aggregate fold-level regularized selections into one consensus ranking."""
    if selection_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "selection_count",
                "fold_count",
                "selection_frequency",
                "mean_abs_coef",
                "rank",
                "in_consensus_subset",
            ]
        )

    selected_only = selection_df[selection_df["selected"].fillna(False).astype(bool)].copy()
    fold_count = int(selection_df["fold"].nunique())
    if selected_only.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "selection_count",
                "fold_count",
                "selection_frequency",
                "mean_abs_coef",
                "rank",
                "in_consensus_subset",
            ]
        )

    grouped = (
        selected_only.groupby("feature", as_index=False)
        .agg(
            selection_count=("selected", "sum"),
            mean_abs_coef=("abs_coef", "mean"),
        )
        .sort_values(
            ["selection_count", "mean_abs_coef", "feature"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    grouped["fold_count"] = fold_count
    grouped["selection_frequency"] = grouped["selection_count"] / max(fold_count, 1)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped["in_consensus_subset"] = grouped["rank"] <= max_features
    return grouped


def build_consensus_subset(
    selection_df: pd.DataFrame,
    *,
    max_features: int = MAX_FEATURES,
) -> list[str]:
    """Return the capped regularized consensus subset."""
    aggregated = aggregate_selection_detail(selection_df, max_features=max_features)
    chosen = aggregated[aggregated["in_consensus_subset"].fillna(False).astype(bool)]
    return chosen["feature"].astype(str).tolist()


def _regularized_param_grid(selector: str) -> list[dict[str, object]]:
    """Return the hyperparameter grid for one regularized selector.

    Uses sklearn 1.8+ forward-compatible form: l1_ratio selects regularisation
    type instead of the deprecated penalty= keyword.
      l1_ratio=1  → L1  (saga solver required)
      l1_ratio=0  → L2  (lbfgs solver)
      0 < l1_ratio < 1  → elastic net  (saga solver)
    """
    if selector == "l1":
        return [
            {"l1_ratio": 1.0, "solver": "saga", "C": c_value}
            for c_value in L1_C_GRID
        ]
    if selector == "elastic_net":
        # sklearn 1.8+: elastic net is expressed via a fractional l1_ratio (0 < l1_ratio < 1)
        # without setting penalty=.  saga is the only solver supporting this.
        return [
            {
                "solver": "saga",
                "C": c_value,
                "l1_ratio": l1_ratio,
            }
            for c_value in ELASTIC_NET_C_GRID
            for l1_ratio in ELASTIC_NET_L1_GRID
        ]
    if selector == "ridge":
        return [
            {"l1_ratio": 0.0, "solver": "lbfgs", "C": c_value}
            for c_value in RIDGE_C_GRID
        ]
    raise ValueError(f"Unsupported selector '{selector}'.")


def _regularized_model_builder(spec: dict[str, object]) -> Any:
    """Return a fold-safe scaled logistic pipeline for regularized models.

    Accepts spec dicts produced by _regularized_param_grid.  Regularisation
    type is conveyed entirely via l1_ratio (sklearn 1.8+ forward-compatible
    form); the deprecated penalty= keyword is never passed.
      l1_ratio=1            → L1  (saga solver)
      l1_ratio=0            → L2  (lbfgs solver)
      0 < l1_ratio < 1      → elastic net  (saga solver)
    """
    logistic_kwargs: dict[str, object] = {
        "C": float(spec["C"]),
        "solver": str(spec["solver"]),
        "class_weight": "balanced",
        "max_iter": 10000,
        "random_state": 42,
    }
    if "l1_ratio" in spec and spec["l1_ratio"] is not None:
        logistic_kwargs["l1_ratio"] = float(spec["l1_ratio"])
    logistic = LogisticRegression(**logistic_kwargs)
    return lambda: Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", logistic),
        ]
    )


def _tune_regularized_params(
    x_df: pd.DataFrame,
    y_series: pd.Series,
    *,
    selector: str,
) -> dict[str, object]:
    """Tune one regularized selector on the training history only."""
    best_spec: dict[str, object] | None = None
    best_score = -np.inf
    best_log_loss = np.inf
    grids = _regularized_param_grid(selector)
    for spec in grids:
        pred_df = evaluate_probability_time_series(
            x_df,
            y_series,
            _regularized_model_builder(spec),
            allow_short_history=True,
        )
        if pred_df.empty:
            continue
        y_true = pred_df["y_true"].to_numpy(dtype=int)
        y_prob = _clip_probs(pred_df["y_prob"].to_numpy(dtype=float))
        score = _balanced_accuracy_safe(y_true, (y_prob >= 0.5).astype(int))
        loss = float(log_loss(y_true, y_prob, labels=[0, 1]))
        if score > best_score or (np.isclose(score, best_score) and loss < best_log_loss):
            best_score = score
            best_log_loss = loss
            best_spec = dict(spec)
    if best_spec is None:
        best_spec = dict(grids[0])
    return best_spec


def _fit_regularized_coefficients(
    x_df: pd.DataFrame,
    y_series: pd.Series,
    spec: dict[str, object],
) -> np.ndarray:
    """Fit one regularized model and return scaled-space coefficients."""
    x_values = x_df.to_numpy(dtype=float)
    train_imputed, _ = _impute_fold_arrays(x_values, x_values[:1].copy())
    pipeline = _regularized_model_builder(spec)()
    pipeline.fit(train_imputed, y_series.to_numpy(dtype=int))
    logistic = pipeline.named_steps["logistic"]
    return np.asarray(logistic.coef_[0], dtype=float)


def build_regularized_selection_detail(
    dataset: BenchmarkDataset,
    *,
    selector: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Build fold-level regularized feature selections and the consensus subset."""
    splitter = _resolve_time_series_splitter(len(dataset.y_series), allow_short_history=False)
    if splitter is None:
        return pd.DataFrame(), []

    x_df = dataset.x_df[dataset.eligible_features].copy()
    y_series = dataset.y_series.copy()
    rows: list[dict[str, object]] = []
    for fold_idx, (train_idx, _) in enumerate(splitter.split(x_df.to_numpy(dtype=float))):
        x_train = x_df.iloc[train_idx].copy()
        y_train = y_series.iloc[train_idx].copy()
        if len(np.unique(y_train.to_numpy(dtype=int))) < 2:
            continue
        spec = _tune_regularized_params(x_train, y_train, selector=selector)
        coef = _fit_regularized_coefficients(x_train, y_train, spec)
        for feature, coef_value in zip(dataset.eligible_features, coef):
            rows.append(
                {
                    "benchmark": dataset.benchmark,
                    "selector": selector,
                    "fold": int(fold_idx),
                    "feature": feature,
                    "coef": float(coef_value),
                    "abs_coef": float(abs(coef_value)),
                    "selected": bool(abs(coef_value) > REGULARIZED_SELECTION_EPS),
                    "C": float(spec["C"]),
                    "l1_ratio": (
                        float(spec["l1_ratio"])
                        if "l1_ratio" in spec and spec["l1_ratio"] is not None
                        else np.nan
                    ),
                }
            )
    selection_df = pd.DataFrame(rows)
    subset = build_consensus_subset(selection_df, max_features=MAX_FEATURES)
    if not selection_df.empty:
        selection_df["in_consensus_subset"] = selection_df["feature"].isin(subset)
    return selection_df, subset


def evaluate_regularized_control(
    dataset: BenchmarkDataset,
    *,
    selector: str = "ridge",
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate a nested-tuned regularized control on the full eligible pool."""
    splitter = _resolve_time_series_splitter(len(dataset.y_series), allow_short_history=False)
    if splitter is None:
        empty_row = {
            "benchmark": dataset.benchmark,
            "method": "ridge_full_pool_control",
            "selector_penalty": selector,
            "n_features": int(len(dataset.eligible_features)),
            "features": _format_feature_list(dataset.eligible_features),
            "n_obs": 0,
            "n_covered": 0,
            "coverage": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "balanced_accuracy_covered": float("nan"),
            "brier_score": float("nan"),
            "log_loss": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "base_rate": float("nan"),
            "predicted_positive_rate": float("nan"),
            "ece_10": float("nan"),
            "production_eligible": False,
        }
        return empty_row, pd.DataFrame(columns=["date", "y_true", "y_prob"])

    x_df = dataset.x_df[dataset.eligible_features].copy()
    y_series = dataset.y_series.copy()
    x_values = x_df.to_numpy(dtype=float)
    y_values = y_series.to_numpy(dtype=int)
    dates = pd.DatetimeIndex(x_df.index)

    rows: list[dict[str, object]] = []
    for train_idx, test_idx in splitter.split(x_values):
        y_train = y_values[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        x_train_df = x_df.iloc[train_idx].copy()
        y_train_series = y_series.iloc[train_idx].copy()
        spec = _tune_regularized_params(x_train_df, y_train_series, selector=selector)
        model = _regularized_model_builder(spec)()
        x_train, x_test = _impute_fold_arrays(x_values[train_idx], x_values[test_idx])
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        for offset, row_idx in enumerate(test_idx):
            rows.append(
                {
                    "date": dates[row_idx],
                    "y_true": int(y_values[row_idx]),
                    "y_prob": float(y_prob[offset]),
                }
            )
    pred_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if pred_df.empty:
        return {
            "benchmark": dataset.benchmark,
            "method": "ridge_full_pool_control",
            "selector_penalty": selector,
            "n_features": int(len(dataset.eligible_features)),
            "features": _format_feature_list(dataset.eligible_features),
            "n_obs": 0,
            "n_covered": 0,
            "coverage": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "balanced_accuracy_covered": float("nan"),
            "brier_score": float("nan"),
            "log_loss": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "base_rate": float("nan"),
            "predicted_positive_rate": float("nan"),
            "ece_10": float("nan"),
            "production_eligible": False,
        }, pred_df

    row = _candidate_metric_row(
        benchmark=dataset.benchmark,
        method="ridge_full_pool_control",
        feature_list=dataset.eligible_features,
        pred_df=pred_df,
        selector_penalty=selector,
        production_eligible=False,
    )
    return row, pred_df


def run_single_feature_screen(
    datasets: dict[str, BenchmarkDataset],
    *,
    feature_family_lookup: dict[str, str],
    evaluator: CachedEvaluator,
    candidate_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate every eligible single feature for every benchmark."""
    rows: list[dict[str, object]] = []
    sequence_map: dict[str, pd.DataFrame] = {}
    allowed = set(candidate_features) if candidate_features is not None else None
    for benchmark, dataset in datasets.items():
        for feature in dataset.eligible_features:
            if allowed is not None and feature not in allowed:
                continue
            row, pred_df = evaluator.evaluate(
                dataset,
                [feature],
                method="single_feature",
            )
            row["feature"] = feature
            row["feature_family"] = feature_family_lookup.get(feature, "unassigned")
            rows.append(row)
            sequence_map[f"{benchmark}::{feature}"] = pred_df
    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df, sequence_map

    ranked_frames: list[pd.DataFrame] = []
    for _, frame in results_df.groupby("benchmark"):
        ranked = _sort_metric_rows(frame.copy()).reset_index(drop=True)
        ranked["rank_within_benchmark"] = np.arange(1, len(ranked) + 1)
        ranked["best_single_feature"] = ranked["rank_within_benchmark"] == 1
        ranked_frames.append(ranked)
    return pd.concat(ranked_frames, ignore_index=True), sequence_map


def forward_stepwise_search(
    dataset: BenchmarkDataset,
    single_feature_df: pd.DataFrame,
    *,
    evaluator: CachedEvaluator,
    max_features: int = MAX_FEATURES,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Run forward stepwise search from the benchmark's best single feature."""
    benchmark_single = single_feature_df[single_feature_df["benchmark"] == dataset.benchmark].copy()
    ranked_single = _sort_metric_rows(benchmark_single)
    best_start = ranked_single.iloc[0]
    current_features = [str(best_start["feature"])]
    current_row, current_pred = evaluator.evaluate(
        dataset,
        current_features,
        method="forward_stepwise",
    )

    trace_rows: list[dict[str, object]] = [
        {
            **current_row,
            "step": 1,
            "candidate_feature": current_features[0],
            "accepted": True,
            "selected_this_step": True,
            "stage": "start",
        }
    ]

    while len(current_features) < max_features:
        step = len(current_features) + 1
        candidate_rows: list[dict[str, object]] = []
        for feature in dataset.eligible_features:
            if feature in current_features:
                continue
            trial_features = [*current_features, feature]
            row, _ = evaluator.evaluate(
                dataset,
                trial_features,
                method="forward_stepwise",
            )
            accepted = _passes_forward_gate(pd.Series(row), pd.Series(current_row))
            candidate_rows.append(
                {
                    **row,
                    "step": step,
                    "candidate_feature": feature,
                    "accepted": bool(accepted),
                    "selected_this_step": False,
                    "stage": "considered",
                }
            )

        if not candidate_rows:
            break

        candidate_df = pd.DataFrame(candidate_rows)
        admissible = candidate_df[candidate_df["accepted"].fillna(False).astype(bool)].copy()
        if admissible.empty:
            trace_rows.extend(candidate_rows)
            break

        chosen = _sort_metric_rows(admissible).iloc[0]
        chosen_feature = str(chosen["candidate_feature"])
        current_features.append(chosen_feature)
        current_row, current_pred = evaluator.evaluate(
            dataset,
            current_features,
            method="forward_stepwise",
        )
        for row in candidate_rows:
            if str(row["candidate_feature"]) == chosen_feature:
                row["selected_this_step"] = True
        trace_rows.extend(candidate_rows)

    final_row = dict(current_row)
    final_row["method"] = "forward_stepwise"
    return pd.DataFrame(trace_rows), final_row, current_pred


def select_benchmark_winner(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Select the production-eligible winner for each benchmark."""
    winners: list[dict[str, object]] = []
    for _, frame in comparison_df.groupby("benchmark"):
        baseline = frame[frame["method"] == "lean_baseline"].iloc[0]
        frame = frame.copy()
        frame["passes_guardrails"] = False
        for idx, row in frame.iterrows():
            if not bool(row["production_eligible"]):
                continue
            frame.loc[idx, "passes_guardrails"] = _passes_winner_guardrails(
                row,
                baseline,
            )
        eligible = frame[frame["passes_guardrails"].fillna(False).astype(bool)].copy()
        if eligible.empty:
            winner = baseline.copy()
            winner["passes_guardrails"] = True
        else:
            winner = _sort_metric_rows(eligible).iloc[0].copy()
        winner["selected_next"] = True
        winners.append(dict(winner))
    return pd.DataFrame(winners).sort_values("benchmark").reset_index(drop=True)


def _pooled_metric_row(
    *,
    label: str,
    sequence_map: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Aggregate benchmark-level sequences into one pooled metric row."""
    frames = [frame.copy() for frame in sequence_map.values() if not frame.empty]
    if not frames:
        return {
            "benchmark": "POOLED",
            "method": label,
            "n_features": np.nan,
            "features": "",
            "n_obs": 0,
            "n_covered": 0,
            "coverage": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "balanced_accuracy_covered": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "base_rate": np.nan,
            "predicted_positive_rate": np.nan,
            "ece_10": np.nan,
        }
    pooled = pd.concat(frames, ignore_index=True)
    row = _candidate_metric_row(
        benchmark="POOLED",
        method=label,
        feature_list=[],
        pred_df=pooled[["date", "y_true", "y_prob"]].copy(),
    )
    row["benchmark_count"] = len(frames)
    return row


def _write_outputs(
    artifacts: dict[str, pd.DataFrame],
    *,
    pooled_baseline_row: dict[str, object],
    pooled_final_row: dict[str, object],
    ridge_pooled_row: dict[str, object] | None,
) -> None:
    """Persist the v128 CSV outputs and markdown summary."""
    save_results(artifacts["feature_inventory"], FEATURE_INVENTORY_PATH.name)
    save_results(artifacts["baseline"], BASELINE_PATH.name)
    save_results(artifacts["single_feature"], SINGLE_FEATURE_PATH.name)
    save_results(artifacts["forward_trace"], FORWARD_TRACE_PATH.name)
    save_results(artifacts["regularized_selection"], REGULARIZED_SELECTION_PATH.name)
    save_results(artifacts["regularized_comparison"], REGULARIZED_COMPARISON_PATH.name)
    save_results(artifacts["comparison"], COMPARISON_PATH.name)
    save_results(artifacts["feature_map"], FEATURE_MAP_PATH.name)

    feature_map = artifacts["feature_map"].copy()
    switch_count = int(feature_map["switched_from_baseline"].fillna(False).sum())
    body_lines = [
        "v128 performs a full benchmark-specific feature search across the 72-feature "
        "non-target research matrix while preserving the current benchmark-specific "
        "balanced-logistic classifier family, rolling WFO geometry, and prequential "
        "logistic calibration path.",
        "",
        "## Pooled Comparison",
        "",
        pd.DataFrame([pooled_baseline_row, pooled_final_row]).to_markdown(index=False),
        "",
        "## Benchmark Winners",
        "",
        feature_map[
            [
                "benchmark",
                "selected_method",
                "n_features",
                "balanced_accuracy_covered",
                "delta_balanced_accuracy_covered",
                "delta_ece_10",
                "delta_brier_score",
                "switched_from_baseline",
            ]
        ].to_markdown(index=False),
        "",
        f"Benchmarks switching away from the incumbent baseline: `{switch_count}`.",
    ]
    if ridge_pooled_row is not None:
        body_lines.extend(
            [
                "",
                "## Ridge Diagnostic Control",
                "",
                pd.DataFrame([ridge_pooled_row]).to_markdown(index=False),
            ]
        )
    body_lines.extend(
        [
            "",
            "## Artifact Notes",
            "",
            "- `v128_feature_inventory.csv`: benchmark-by-feature availability and eligibility",
            "- `v128_single_feature_results.csv`: full single-feature leaderboard",
            "- `v128_forward_stepwise_trace.csv`: every considered forward-stepwise addition",
            "- `v128_regularized_selection_detail.csv`: fold-level L1 / elastic-net feature selections",
            "- `v128_regularized_comparison.csv`: evaluated L1, elastic-net, and ridge candidates",
            "- `v128_benchmark_feature_map.csv`: final benchmark-specific recommendation map",
        ]
    )
    write_markdown_summary(
        SUMMARY_PATH,
        "v128 Benchmark-Specific Feature Search Summary",
        body_lines,
    )


def run_feature_search(
    *,
    benchmarks: list[str] | None = None,
    candidate_features: list[str] | None = None,
    max_features: int = MAX_FEATURES,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full v128 feature search and optionally write artifacts."""
    feature_df, rel_map = load_v128_inputs(benchmarks=benchmarks)
    datasets = build_benchmark_datasets(feature_df, rel_map)
    family_lookup = _feature_family_lookup(feature_df)
    inventory_df = build_feature_inventory(datasets, family_lookup)

    evaluator = CachedEvaluator()
    baseline_features = feature_set_from_name(feature_df, INCUMBENT_FEATURE_SET)

    baseline_rows: list[dict[str, object]] = []
    baseline_sequences: dict[str, pd.DataFrame] = {}
    for benchmark, dataset in datasets.items():
        feature_list = [feature for feature in baseline_features if feature in dataset.x_df.columns]
        row, pred_df = evaluator.evaluate(dataset, feature_list, method="lean_baseline")
        baseline_rows.append(row)
        baseline_sequences[benchmark] = pred_df
    baseline_df = pd.DataFrame(baseline_rows).sort_values("benchmark").reset_index(drop=True)

    single_feature_df, _ = run_single_feature_screen(
        datasets,
        feature_family_lookup=family_lookup,
        evaluator=evaluator,
        candidate_features=candidate_features,
    )

    forward_rows: list[dict[str, object]] = []
    forward_traces: list[pd.DataFrame] = []
    regularized_rows: list[dict[str, object]] = []
    regularized_selection_frames: list[pd.DataFrame] = []
    forward_sequences: dict[str, pd.DataFrame] = {}
    l1_sequences: dict[str, pd.DataFrame] = {}
    elastic_sequences: dict[str, pd.DataFrame] = {}
    ridge_sequences: dict[str, pd.DataFrame] = {}

    for benchmark, base_dataset in datasets.items():
        dataset = base_dataset
        if candidate_features is not None:
            allowed = [feature for feature in dataset.eligible_features if feature in set(candidate_features)]
            dataset = BenchmarkDataset(
                benchmark=dataset.benchmark,
                x_df=dataset.x_df.copy(),
                y_series=dataset.y_series.copy(),
                eligible_features=allowed,
            )

        trace_df, forward_row, forward_pred = forward_stepwise_search(
            dataset,
            single_feature_df,
            evaluator=evaluator,
            max_features=max_features,
        )
        forward_rows.append(forward_row)
        forward_traces.append(trace_df)
        forward_sequences[benchmark] = forward_pred

        for selector, method_name in (("l1", "l1_consensus"), ("elastic_net", "elastic_net_consensus")):
            selection_df, subset = build_regularized_selection_detail(dataset, selector=selector)
            if not selection_df.empty:
                selection_df["selected_method"] = method_name
            regularized_selection_frames.append(selection_df)
            if subset:
                row, pred_df = evaluator.evaluate(
                    dataset,
                    subset,
                    method=method_name,
                    selector_penalty=selector,
                )
            else:
                row = {
                    "benchmark": benchmark,
                    "method": method_name,
                    "selector_penalty": selector,
                    "n_features": 0,
                    "features": "",
                    "n_obs": 0,
                    "n_covered": 0,
                    "coverage": np.nan,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "balanced_accuracy_covered": np.nan,
                    "brier_score": np.nan,
                    "log_loss": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "base_rate": np.nan,
                    "predicted_positive_rate": np.nan,
                    "ece_10": np.nan,
                    "production_eligible": True,
                }
                pred_df = pd.DataFrame(columns=["date", "y_true", "y_prob"])
            regularized_rows.append(row)
            if method_name == "l1_consensus":
                l1_sequences[benchmark] = pred_df
            else:
                elastic_sequences[benchmark] = pred_df

        ridge_row, ridge_pred = evaluate_regularized_control(dataset, selector="ridge")
        regularized_rows.append(ridge_row)
        ridge_sequences[benchmark] = ridge_pred

    forward_df = pd.DataFrame(forward_rows).sort_values("benchmark").reset_index(drop=True)
    forward_trace_df = (
        pd.concat(forward_traces, ignore_index=True)
        if forward_traces
        else pd.DataFrame()
    )
    regularized_selection_df = (
        pd.concat(regularized_selection_frames, ignore_index=True)
        if regularized_selection_frames
        else pd.DataFrame()
    )
    regularized_comparison_df = (
        pd.DataFrame(regularized_rows)
        .sort_values(["benchmark", "method"])
        .reset_index(drop=True)
    )

    comparison_df = pd.concat(
        [
            baseline_df.assign(production_eligible=True),
            forward_df.assign(production_eligible=True),
            regularized_comparison_df,
        ],
        ignore_index=True,
    ).sort_values(["benchmark", "method"]).reset_index(drop=True)

    winners_df = select_benchmark_winner(comparison_df)
    selected_sequences: dict[str, pd.DataFrame] = {}
    feature_map_rows: list[dict[str, object]] = []
    for _, winner in winners_df.iterrows():
        benchmark = str(winner["benchmark"])
        baseline_row = baseline_df[baseline_df["benchmark"] == benchmark].iloc[0]
        selected_method = str(winner["method"])
        if selected_method == "lean_baseline":
            selected_sequences[benchmark] = baseline_sequences[benchmark]
        elif selected_method == "forward_stepwise":
            selected_sequences[benchmark] = forward_sequences[benchmark]
        elif selected_method == "l1_consensus":
            selected_sequences[benchmark] = l1_sequences[benchmark]
        elif selected_method == "elastic_net_consensus":
            selected_sequences[benchmark] = elastic_sequences[benchmark]
        else:
            selected_sequences[benchmark] = baseline_sequences[benchmark]
        feature_map_rows.append(
            {
                "benchmark": benchmark,
                "selected_method": selected_method,
                "n_features": int(winner["n_features"]),
                "selected_features": str(winner["features"]),
                "balanced_accuracy_covered": float(winner["balanced_accuracy_covered"]),
                "baseline_balanced_accuracy_covered": float(baseline_row["balanced_accuracy_covered"]),
                "delta_balanced_accuracy_covered": float(
                    winner["balanced_accuracy_covered"] - baseline_row["balanced_accuracy_covered"]
                ),
                "ece_10": float(winner["ece_10"]),
                "baseline_ece_10": float(baseline_row["ece_10"]),
                "delta_ece_10": float(winner["ece_10"] - baseline_row["ece_10"]),
                "brier_score": float(winner["brier_score"]),
                "baseline_brier_score": float(baseline_row["brier_score"]),
                "delta_brier_score": float(winner["brier_score"] - baseline_row["brier_score"]),
                "switched_from_baseline": bool(selected_method != "lean_baseline"),
            }
        )
    feature_map_df = pd.DataFrame(feature_map_rows).sort_values("benchmark").reset_index(drop=True)

    pooled_baseline_row = _pooled_metric_row(label="lean_baseline", sequence_map=baseline_sequences)
    pooled_final_row = _pooled_metric_row(label="final_feature_map", sequence_map=selected_sequences)
    ridge_pooled_row = _pooled_metric_row(label="ridge_full_pool_control", sequence_map=ridge_sequences)

    comparison_with_pooled = pd.concat(
        [
            comparison_df,
            pd.DataFrame([pooled_baseline_row, pooled_final_row, ridge_pooled_row]),
        ],
        ignore_index=True,
    ).reset_index(drop=True)

    artifacts = {
        "feature_inventory": inventory_df,
        "baseline": baseline_df,
        "single_feature": single_feature_df,
        "forward_trace": forward_trace_df,
        "regularized_selection": regularized_selection_df,
        "regularized_comparison": regularized_comparison_df,
        "comparison": comparison_with_pooled,
        "feature_map": feature_map_df,
    }
    if write_outputs:
        _write_outputs(
            artifacts,
            pooled_baseline_row=pooled_baseline_row,
            pooled_final_row=pooled_final_row,
            ridge_pooled_row=ridge_pooled_row,
        )
    return artifacts


def main() -> None:
    """Run the full v128 feature-search artifact generation."""
    parser = argparse.ArgumentParser(description="Run the v128 benchmark feature search.")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="",
        help="Optional comma-separated benchmark subset for a smaller run.",
    )
    parser.add_argument(
        "--candidate-features",
        type=str,
        default="",
        help="Optional comma-separated feature subset for smoke runs.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=MAX_FEATURES,
        help="Maximum subset size for forward search and consensus subsets.",
    )
    args = parser.parse_args()

    selected_benchmarks = (
        [part.strip() for part in args.benchmarks.split(",") if part.strip()]
        if args.benchmarks
        else None
    )
    candidate_features = (
        [part.strip() for part in args.candidate_features.split(",") if part.strip()]
        if args.candidate_features
        else None
    )

    print_header("v128", "Benchmark-Specific Full Feature Search")
    artifacts = run_feature_search(
        benchmarks=selected_benchmarks,
        candidate_features=candidate_features,
        max_features=int(args.max_features),
        write_outputs=True,
    )
    winners = artifacts["feature_map"][
        [
            "benchmark",
            "selected_method",
            "n_features",
            "delta_balanced_accuracy_covered",
            "delta_ece_10",
            "delta_brier_score",
        ]
    ]
    pooled = artifacts["comparison"][
        artifacts["comparison"]["benchmark"].astype(str).eq("POOLED")
    ][
        [
            "method",
            "balanced_accuracy_covered",
            "ece_10",
            "brier_score",
            "coverage",
        ]
    ]
    print("\nBenchmark Winners:")
    print(winners.to_string(index=False, float_format="{:.4f}".format))
    print("\nPooled Comparison:")
    print(pooled.to_string(index=False, float_format="{:.4f}".format))
    print_footer()


if __name__ == "__main__":
    main()
