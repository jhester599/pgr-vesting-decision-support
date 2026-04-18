"""Shared helpers for the v153-v158 classification and feature research cycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score

from src.research.v37_utils import (
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RESULTS_DIR,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    get_connection,
    load_feature_matrix,
    load_relative_series,
)
from src.research.v87_utils import (
    BENCHMARKS,
    _impute_fold,
    _outer_time_series_splitter,
    build_target_series,
    logistic_factory,
)

FIRTH_THIN_THRESHOLD: int = 30
HIGH_THRESHOLD: float = 0.70
LOW_THRESHOLD: float = 0.30
ACTIONABLE_TARGET: str = "actionable_sell_3pct"


def _hat_diag(XW_half: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix H = sqrt(W) X (X'WX)^{-1} X' sqrt(W).

    Uses thin SVD for numerical stability on near-rank-deficient designs.
    """
    try:
        _, s, Vt = np.linalg.svd(XW_half, full_matrices=False)
        threshold = max(1e-12, 1e-10 * float(s[0])) if len(s) > 0 else 1e-12
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        U = XW_half @ (Vt.T * s_inv)
        return np.sum(U ** 2, axis=1)
    except np.linalg.LinAlgError:
        return np.zeros(XW_half.shape[0])


def fit_firth_logistic(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> np.ndarray:
    """Fit Firth-penalized logistic regression via IRLS with Jeffreys-prior correction.

    X must already include a leading intercept column.
    Returns coefficient vector beta of shape (X.shape[1],).
    """
    _, p = X.shape
    beta = np.zeros(p)

    for _ in range(max_iter):
        mu = expit(X @ beta)
        W_diag = mu * (1.0 - mu)

        XtWX = X.T @ (X * W_diag[:, None])
        try:
            XtWX_inv = np.linalg.solve(XtWX + 1e-8 * np.eye(p), np.eye(p))
        except np.linalg.LinAlgError:
            break

        XW_half = X * np.sqrt(np.maximum(W_diag, 0.0))[:, None]
        h = _hat_diag(XW_half)

        score = X.T @ (y - mu + h * (0.5 - mu))
        delta = XtWX_inv @ score
        beta = beta + delta
        if float(np.linalg.norm(delta)) < tol:
            break

    return beta


def predict_firth_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Return class-1 probabilities from fitted Firth coefficients."""
    return expit(X @ beta)


def count_training_positives(y: np.ndarray) -> int:
    """Return number of positive-class examples."""
    return int(np.sum(y == 1))


def load_research_inputs_for_classification() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load pre-holdout feature matrix and benchmark relative-return series."""
    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rel_map = {
            bm: load_relative_series(conn, bm, horizon=6) for bm in BENCHMARKS
        }
    finally:
        conn.close()
    return feature_df, rel_map


def evaluate_classifier_wfo(
    feature_df: pd.DataFrame,
    rel_series: pd.Series,
    feature_cols: list[str],
    benchmark: str,
    use_firth: bool = False,
) -> dict[str, Any]:
    """Run WFO binary-classifier evaluation for one benchmark.

    Returns a dict with keys: benchmark, n_obs, avg_train_positives,
    ba_all, ba_covered, coverage, use_firth, skipped.
    """
    avail_cols = [c for c in feature_cols if c in feature_df.columns]
    if not avail_cols:
        return {"benchmark": benchmark, "n_obs": 0, "skipped": True, "use_firth": use_firth}

    target = build_target_series(rel_series, ACTIONABLE_TARGET)
    aligned = feature_df[avail_cols].join(target).dropna()
    min_obs = MAX_TRAIN_MONTHS + GAP_MONTHS + TEST_SIZE_MONTHS + 1
    if len(aligned) < min_obs:
        return {"benchmark": benchmark, "n_obs": len(aligned), "skipped": True, "use_firth": use_firth}

    X_all = aligned[avail_cols].to_numpy(dtype=float)
    y_all = aligned[ACTIONABLE_TARGET].to_numpy(dtype=float)
    splitter = _outer_time_series_splitter(len(X_all))

    y_pred_all: list[float] = []
    y_true_all: list[float] = []
    n_positives_per_fold: list[int] = []

    for train_idx, test_idx in splitter.split(X_all):
        X_train, X_test = _impute_fold(X_all[train_idx], X_all[test_idx])
        y_train = y_all[train_idx]
        n_positives_per_fold.append(count_training_positives(y_train))

        if use_firth:
            ones_tr = np.ones((len(X_train), 1))
            ones_te = np.ones((len(X_test), 1))
            X_tr_aug = np.column_stack([ones_tr, X_train])
            X_te_aug = np.column_stack([ones_te, X_test])
            beta = fit_firth_logistic(X_tr_aug, y_train)
            proba = predict_firth_proba(X_te_aug, beta)
        else:
            model = logistic_factory(class_weight="balanced")()
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]

        y_pred_all.extend(proba.tolist())
        y_true_all.extend(y_all[test_idx].tolist())

    y_pred_arr = np.asarray(y_pred_all)
    y_true_arr = np.asarray(y_true_all)

    covered_mask = (y_pred_arr >= HIGH_THRESHOLD) | (y_pred_arr <= LOW_THRESHOLD)
    ba_all = float(balanced_accuracy_score(
        y_true_arr, (y_pred_arr >= 0.5).astype(int)
    ))
    coverage = float(covered_mask.mean())

    if covered_mask.sum() >= 2:
        ba_covered = float(balanced_accuracy_score(
            y_true_arr[covered_mask],
            (y_pred_arr[covered_mask] >= 0.5).astype(int),
        ))
    else:
        ba_covered = float("nan")

    return {
        "benchmark": benchmark,
        "n_obs": len(y_true_arr),
        "avg_train_positives": float(np.mean(n_positives_per_fold)),
        "ba_all": ba_all,
        "ba_covered": ba_covered,
        "coverage": coverage,
        "use_firth": use_firth,
        "skipped": False,
    }


def compare_logistic_vs_firth(
    feature_df: pd.DataFrame,
    rel_map: dict[str, pd.Series],
    benchmarks: list[str],
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    """Evaluate both standard logistic and Firth logistic for each benchmark.

    Returns list of row dicts with delta_ba_covered = firth_ba_covered - logistic_ba_covered.
    """
    rows: list[dict[str, Any]] = []
    for bm in benchmarks:
        if bm not in rel_map or rel_map[bm].empty:
            continue
        std_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=feature_cols,
            benchmark=bm,
            use_firth=False,
        )
        firth_result = evaluate_classifier_wfo(
            feature_df=feature_df,
            rel_series=rel_map[bm],
            feature_cols=feature_cols,
            benchmark=bm,
            use_firth=True,
        )
        delta = float("nan")
        if not std_result.get("skipped") and not firth_result.get("skipped"):
            std_ba = std_result.get("ba_covered", float("nan"))
            firth_ba = firth_result.get("ba_covered", float("nan"))
            if not (np.isnan(std_ba) or np.isnan(firth_ba)):
                delta = firth_ba - std_ba
        rows.append({
            "benchmark": bm,
            "avg_train_positives": std_result.get("avg_train_positives", float("nan")),
            "is_thin": std_result.get("avg_train_positives", 999) < FIRTH_THIN_THRESHOLD,
            "std_ba_covered": std_result.get("ba_covered", float("nan")),
            "firth_ba_covered": firth_result.get("ba_covered", float("nan")),
            "delta_ba_covered": delta,
            "std_coverage": std_result.get("coverage", float("nan")),
            "firth_coverage": firth_result.get("coverage", float("nan")),
        })
    return rows
