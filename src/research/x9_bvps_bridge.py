"""Research-only x9 BVPS bridge features and WFO utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.research.x2_absolute_classification import iter_absolute_wfo_splits
from src.research.x4_bvps_forecasting import (
    _growth_to_target,
    _impute_fold,
    _metrics_from_predictions,
    _rows_for_fold,
    _target_to_growth,
)


@dataclass(frozen=True)
class X9FeatureBlock:
    """One pre-registered x9 feature block."""

    block_name: str
    feature_columns: list[str]
    notes: str


def _align_x9_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    current_bvps: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Align x9 inputs while allowing current_bvps as a feature."""
    selected = [feature for feature in feature_columns if feature in X.columns]
    if not selected:
        raise ValueError("No requested feature columns are present in X.")
    target = pd.to_numeric(y.copy(), errors="coerce")
    target.name = y.name or "target"
    current = pd.to_numeric(current_bvps.copy(), errors="coerce")
    current.name = "__current_bvps_reference"
    aligned = X[selected].join([target, current], how="inner")
    aligned = aligned.dropna(subset=[target.name, "__current_bvps_reference"])
    if aligned.empty:
        raise ValueError("No aligned non-null BVPS target observations.")
    return (
        aligned[selected].copy(),
        aligned[target.name].astype(float).copy(),
        aligned["__current_bvps_reference"].astype(float).copy(),
    )


def build_bvps_bridge_features(
    feature_df: pd.DataFrame,
    current_bvps: pd.Series,
) -> pd.DataFrame:
    """Build lagged BVPS and capital-generation bridge features."""
    result = feature_df.copy()
    bvps = pd.to_numeric(current_bvps.copy(), errors="coerce")
    bvps.index = pd.DatetimeIndex(pd.to_datetime(bvps.index))
    bvps = bvps.sort_index()
    result = result.reindex(result.index.union(bvps.index)).sort_index()
    result["current_bvps"] = bvps.reindex(result.index)
    result["bvps_growth_1m"] = result["current_bvps"].pct_change(
        1,
        fill_method=None,
    )
    result["bvps_growth_3m"] = result["current_bvps"].pct_change(
        3,
        fill_method=None,
    )
    result["bvps_growth_6m"] = result["current_bvps"].pct_change(
        6,
        fill_method=None,
    )
    result["bvps_yoy_dollar_change"] = result["current_bvps"].diff(12)
    result["bvps_growth_yoy_direct"] = result["current_bvps"].pct_change(
        12,
        fill_method=None,
    )
    result["month_of_year"] = result.index.month.astype(int)
    result["quarter_of_year"] = result.index.quarter.astype(int)
    result["q4_flag"] = (result.index.quarter == 4).astype(int)
    result["november_snapshot_flag"] = (result.index.month == 11).astype(int)
    result["dividend_season_flag"] = result.index.month.isin([11, 12, 1]).astype(
        int
    )
    first_bvps_by_year = result.groupby(result.index.year)["current_bvps"].transform(
        "first"
    )
    result["bvps_growth_ytd"] = result["current_bvps"] / first_bvps_by_year - 1.0
    return result.reindex(feature_df.index)


def _interaction(
    frame: pd.DataFrame,
    left: str,
    right: str,
) -> pd.Series:
    if left not in frame.columns or right not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[left], errors="coerce") * pd.to_numeric(
        frame[right],
        errors="coerce",
    )


def build_bvps_interactions(features: pd.DataFrame) -> pd.DataFrame:
    """Build economically pre-registered x9 interaction features."""
    result = pd.DataFrame(index=features.index)
    result["premium_growth_x_underwriting_margin"] = _interaction(
        features,
        "npw_growth_yoy",
        "underwriting_margin_ttm",
    )
    result["premium_to_surplus_x_cr_delta"] = _interaction(
        features,
        "pgr_premium_to_surplus",
        "monthly_combined_ratio_delta",
    )
    result["buyback_yield_x_pb_ratio"] = _interaction(
        features,
        "buyback_yield",
        "pb_ratio",
    )
    result["buyback_yield_x_bvps_growth_3m"] = _interaction(
        features,
        "buyback_yield",
        "bvps_growth_3m",
    )
    result["unrealized_gain_x_real_rate"] = _interaction(
        features,
        "unrealized_gain_pct_equity",
        "real_rate_10y",
    )
    result["investment_yield_x_bvps"] = _interaction(
        features,
        "investment_book_yield",
        "current_bvps",
    )
    return result


def build_x9_feature_blocks(features: pd.DataFrame) -> list[X9FeatureBlock]:
    """Return bounded, pre-registered x9 feature blocks."""
    blocks = {
        "bvps_lags": [
            "current_bvps",
            "bvps_growth_1m",
            "bvps_growth_3m",
            "bvps_growth_6m",
            "bvps_growth_ytd",
            "bvps_yoy_dollar_change",
            "month_of_year",
            "q4_flag",
            "dividend_season_flag",
        ],
        "accounting_core": [
            "combined_ratio_ttm",
            "monthly_combined_ratio_delta",
            "underwriting_margin_ttm",
            "underwriting_income_growth_yoy",
            "npw_growth_yoy",
            "pif_growth_yoy",
            "pgr_premium_to_surplus",
            "unrealized_gain_pct_equity",
            "realized_gain_to_net_income_ratio",
            "buyback_yield",
            "investment_book_yield",
        ],
        "logical_interactions": list(build_bvps_interactions(features).columns),
    }
    result: list[X9FeatureBlock] = []
    for block_name, columns in blocks.items():
        available = [column for column in columns if column in features.columns]
        result.append(
            X9FeatureBlock(
                block_name=block_name,
                feature_columns=available,
                notes=f"Pre-registered x9 {block_name} feature block.",
            )
        )
    combined: list[str] = []
    for columns in blocks.values():
        for column in columns:
            if column in features.columns and column not in combined:
                combined.append(column)
    result.append(
        X9FeatureBlock(
            block_name="bridge_combined",
            feature_columns=combined,
            notes="Union of BVPS lags, accounting core, and logical interactions.",
        )
    )
    return result


def build_x9_regressor(model_name: str) -> Pipeline:
    """Build one x9 regularized regressor."""
    if model_name == "ridge_bridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1000.0)),
            ]
        )
    if model_name == "elastic_net_bridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    ElasticNet(
                        alpha=0.002,
                        l1_ratio=0.65,
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported x9 regressor '{model_name}'.")


def _coef_stability_rows(
    model: Pipeline,
    feature_columns: list[str],
    fold_idx: int,
) -> list[dict[str, Any]]:
    regressor = model.named_steps["regressor"]
    coef = getattr(regressor, "coef_", np.zeros(len(feature_columns)))
    return [
        {
            "fold_idx": int(fold_idx),
            "feature": feature,
            "coefficient": float(coef[idx]),
            "selected": bool(abs(float(coef[idx])) > 1e-12),
        }
        for idx, feature in enumerate(feature_columns)
    ]


def summarize_feature_stability(stability_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize coefficient non-zero frequency by feature."""
    if stability_rows.empty:
        return pd.DataFrame()
    grouped = stability_rows.groupby("feature", dropna=False)
    result = grouped.agg(
        fold_count=("fold_idx", "nunique"),
        selected_count=("selected", "sum"),
        mean_abs_coefficient=("coefficient", lambda x: float(np.mean(np.abs(x)))),
    ).reset_index()
    result["selection_rate"] = (
        result["selected_count"] / result["fold_count"].where(
            result["fold_count"] > 0
        )
    )
    return result.sort_values(
        ["selection_rate", "mean_abs_coefficient", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def evaluate_x9_bvps_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    current_bvps: pd.Series,
    model_name: str,
    feature_columns: list[str],
    target_kind: str,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Evaluate one x9 regularized BVPS bridge model."""
    X_aligned, y_aligned, current_aligned = _align_x9_inputs(
        X,
        y,
        current_bvps,
        feature_columns,
    )
    _, splitter = iter_absolute_wfo_splits(
        len(X_aligned),
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
    )
    rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    fold_count = 0
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_aligned)):
        x_train, x_test = _impute_fold(
            X_aligned.iloc[train_idx].to_numpy(dtype=float),
            X_aligned.iloc[test_idx].to_numpy(dtype=float),
        )
        model = build_x9_regressor(model_name)
        model.fit(x_train, y_aligned.iloc[train_idx].to_numpy(dtype=float))
        stability_rows.extend(
            _coef_stability_rows(model, list(X_aligned.columns), fold_idx)
        )
        rows.extend(
            _rows_for_fold(
                X_aligned=X_aligned,
                y_aligned=y_aligned,
                current_aligned=current_aligned,
                target_kind=target_kind,
                fold_idx=fold_idx,
                test_idx=test_idx,
                train_start_date=X_aligned.index[train_idx[0]],
                train_end_date=X_aligned.index[train_idx[-1]],
                y_pred_target=model.predict(x_test),
            )
        )
        fold_count += 1
    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    metrics = _metrics_from_predictions(
        predictions,
        model_name=model_name,
        target_kind=target_kind,
        horizon_months=target_horizon_months,
        fold_count=fold_count,
        n_features=len(X_aligned.columns),
    )
    return predictions, metrics, summarize_feature_stability(
        pd.DataFrame(stability_rows)
    )


def _fold_growth_mean(y_train: np.ndarray, target_kind: str) -> float:
    growth_train = _target_to_growth(y_train, target_kind)
    return float(np.nanmean(growth_train))


def evaluate_x9_bvps_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    current_bvps: pd.Series,
    baseline_name: str,
    target_kind: str,
    target_horizon_months: int,
    purge_buffer: int | None = None,
    train_window_months: int | None = None,
    test_window_months: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate fold-local x9 BVPS bridge baselines."""
    X_aligned, y_aligned, current_aligned = _align_x9_inputs(
        X,
        y,
        current_bvps,
        list(X.columns),
    )
    _, splitter = iter_absolute_wfo_splits(
        len(X_aligned),
        target_horizon_months=target_horizon_months,
        purge_buffer=purge_buffer,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
    )
    rows: list[dict[str, Any]] = []
    fold_count = 0
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_aligned)):
        y_train = y_aligned.iloc[train_idx].to_numpy(dtype=float)
        if baseline_name == "no_change_bvps":
            pred_target = 0.0
        elif baseline_name == "drift_bvps_growth":
            pred_target = _growth_to_target(
                _fold_growth_mean(y_train, target_kind),
                target_kind,
            )
        elif baseline_name == "trailing_3m_growth":
            train_growth = _target_to_growth(y_train, target_kind)
            pred_target = _growth_to_target(
                float(np.nanmean(train_growth[-3:])),
                target_kind,
            )
        elif baseline_name == "seasonal_month_drift":
            train_months = X_aligned.index[train_idx].month
            test_month = int(X_aligned.index[test_idx[0]].month)
            train_growth = pd.Series(
                _target_to_growth(y_train, target_kind),
                index=train_months,
            )
            seasonal = train_growth.loc[train_growth.index == test_month]
            pred_growth = (
                float(seasonal.mean())
                if not seasonal.empty
                else _fold_growth_mean(y_train, target_kind)
            )
            pred_target = _growth_to_target(pred_growth, target_kind)
        else:
            raise ValueError(f"Unsupported x9 baseline '{baseline_name}'.")
        rows.extend(
            _rows_for_fold(
                X_aligned=X_aligned,
                y_aligned=y_aligned,
                current_aligned=current_aligned,
                target_kind=target_kind,
                fold_idx=fold_idx,
                test_idx=test_idx,
                train_start_date=X_aligned.index[train_idx[0]],
                train_end_date=X_aligned.index[train_idx[-1]],
                y_pred_target=np.repeat(pred_target, len(test_idx)),
            )
        )
        fold_count += 1
    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    metrics = _metrics_from_predictions(
        predictions,
        model_name=baseline_name,
        target_kind=target_kind,
        horizon_months=target_horizon_months,
        fold_count=fold_count,
        n_features=0,
    )
    return predictions, metrics


def summarize_x9_bvps_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Rank x9 BVPS bridge rows by horizon."""
    if detail_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in detail_df.groupby("horizon_months", dropna=False):
        ranked = group.sort_values(
            [
                "future_bvps_mae",
                "growth_rmse",
                "directional_hit_rate",
                "model_name",
                "feature_block",
            ],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        no_change = group[group["model_name"] == "no_change_bvps"]
        baseline_mae = (
            float(no_change["future_bvps_mae"].iloc[0])
            if not no_change.empty
            else float("nan")
        )
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            out = row.copy()
            out["rank"] = int(rank)
            out["beats_no_change_bvps"] = bool(
                row["future_bvps_mae"] < baseline_mae
            )
            rows.append(out)
    return pd.DataFrame(rows).reset_index(drop=True)
