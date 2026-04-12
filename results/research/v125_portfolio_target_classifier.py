"""
v125 -- Path B: Composite portfolio-target classifier.

Trains a single logistic classifier on the composite balanced_pref_95_5
portfolio-return target and compares head-to-head with the v123 Path A
investable-pool aggregate on matched WFO folds.

Usage:
    python results/research/v125_portfolio_target_classifier.py
    python results/research/v125_portfolio_target_classifier.py --as-of 2024-03-31

Outputs (written to results/research/):
    v125_portfolio_target_results.csv
    v125_portfolio_target_summary.md
    v125_portfolio_target_fold_detail.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.features import INVESTABLE_CLASSIFIER_BASE_WEIGHTS
from config.model import (
    WFO_EMBARGO_MONTHS_6M,
    WFO_PURGE_BUFFER_6M,
    WFO_TRAIN_WINDOW_MONTHS,
)
from src.database import db_client
from src.processing.feature_engineering import (
    build_feature_matrix_from_db,
    get_X_y_relative,
    truncate_relative_target_for_asof,
)
from src.processing.multi_total_return import load_relative_return_matrix
from src.research.v87_utils import (
    build_target_series,
    feature_set_from_name,
    logistic_factory,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_SET = "lean_baseline"
TARGET_LABEL = "actionable_sell_3pct"
THRESHOLD = 0.03
HORIZON_MONTHS = 6
GAP = WFO_EMBARGO_MONTHS_6M + WFO_PURGE_BUFFER_6M  # embargo + purge
TEST_SIZE = 6
N_SPLITS = 15
ABSTAIN_LOW = 0.30
ABSTAIN_HIGH = 0.70
MIN_TRAIN_OBS = 60

COMPOSITE_WEIGHTS: dict[str, float] = INVESTABLE_CLASSIFIER_BASE_WEIGHTS.copy()
OUTPUT_DIR = PROJECT_ROOT / "results" / "research"

# Path A v92 reference baseline (hardcoded from prior evaluation)
PATH_A_REFERENCE: dict[str, object] = {
    "model": "path_a_separate_logistic_calibrated_v92_reference",
    "n_obs": None,
    "n_covered": None,
    "coverage": None,
    "balanced_accuracy_all": 0.7538,
    "balanced_accuracy_covered": 0.5132,
    "brier_score": 0.1852,
    "log_loss": 0.5985,
    "ece": 0.0813,
    "base_rate_positive": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (equal-width bins)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(
            float(y_true[mask].mean()) - float(y_prob[mask].mean())
        )
    return ece / max(len(y_true), 1)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def build_composite_return_series(
    conn: object,
    weights: dict[str, float] | None = None,
    as_of: date | None = None,
) -> pd.Series:
    """Build a weighted-composite relative-return series across benchmarks."""
    if weights is None:
        weights = COMPOSITE_WEIGHTS

    rel_series_list: list[pd.Series] = []
    weight_list: list[float] = []

    for ticker, weight in weights.items():
        try:
            rel = load_relative_return_matrix(conn, ticker, HORIZON_MONTHS)
            if rel is not None and not rel.empty:
                if as_of is not None:
                    rel = truncate_relative_target_for_asof(
                        rel,
                        as_of=pd.Timestamp(as_of),
                        horizon_months=HORIZON_MONTHS,
                    )
                rel_series_list.append(rel.rename(ticker))
                weight_list.append(weight)
        except Exception:
            continue

    if not rel_series_list:
        raise ValueError(
            "No relative return series could be loaded for composite target."
        )

    total_w = sum(weight_list)
    norm_weights = [w / total_w for w in weight_list]

    combined = pd.concat(rel_series_list, axis=1).dropna(how="any")
    composite: pd.Series = sum(  # type: ignore[assignment]
        combined[s.name] * w
        for s, w in zip(rel_series_list, norm_weights)
        if s.name in combined.columns
    )
    composite.name = "composite_relative_return"
    return composite


def run_wfo(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    gap: int = GAP,
    test_size: int = TEST_SIZE,
    min_train_obs: int = MIN_TRAIN_OBS,
) -> pd.DataFrame:
    """Walk-forward optimisation via TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)

    records: list[dict[str, object]] = []
    model_factory = logistic_factory(class_weight="balanced", c_value=0.5)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()

        if len(X_train) < min_train_obs:
            continue

        unique_classes = np.unique(y_train.to_numpy(dtype=int))
        if len(unique_classes) < 2:
            continue

        # Impute NaN with training column medians
        x_train_vals = X_train.to_numpy(dtype=float)
        medians = np.nanmedian(x_train_vals, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        for col_idx in range(x_train_vals.shape[1]):
            nan_mask = np.isnan(x_train_vals[:, col_idx])
            x_train_vals[nan_mask, col_idx] = medians[col_idx]

        x_test_vals = X_test.to_numpy(dtype=float)
        for col_idx in range(x_test_vals.shape[1]):
            nan_mask = np.isnan(x_test_vals[:, col_idx])
            x_test_vals[nan_mask, col_idx] = medians[col_idx]

        model = model_factory()
        model.fit(x_train_vals, y_train.to_numpy(dtype=int))
        y_prob = model.predict_proba(x_test_vals)[:, 1]

        for i, test_pos in enumerate(test_idx):
            records.append(
                {
                    "fold": fold_idx,
                    "train_start": str(X.index[train_idx[0]].date()),
                    "train_end": str(X.index[train_idx[-1]].date()),
                    "test_date": str(X.index[test_pos].date()),
                    "y_true": int(y_test.iloc[i]),
                    "y_prob": float(y_prob[i]),
                }
            )

    return pd.DataFrame(records)


def compute_metrics(results_df: pd.DataFrame, label: str) -> dict[str, object]:
    """Pool all WFO observations and compute classification metrics."""
    y_true = results_df["y_true"].to_numpy()
    y_prob = results_df["y_prob"].to_numpy()
    n_obs = len(y_true)

    # Coverage: observations where model is confident (abs(prob-0.5) > 0.2)
    covered_mask = np.abs(y_prob - 0.5) > 0.2
    n_covered = int(covered_mask.sum())
    coverage = n_covered / max(n_obs, 1)

    y_pred_all = (y_prob >= 0.5).astype(int)
    ba_all = balanced_accuracy_score(y_true, y_pred_all)

    if n_covered > 0:
        y_pred_cov = (y_prob[covered_mask] >= 0.5).astype(int)
        ba_covered = balanced_accuracy_score(y_true[covered_mask], y_pred_cov)
    else:
        ba_covered = float("nan")

    bs = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob, labels=[0, 1])
    ece = _compute_ece(y_true, y_prob)
    base_rate = float(y_true.mean())

    return {
        "model": label,
        "n_obs": n_obs,
        "n_covered": n_covered,
        "coverage": round(coverage, 4),
        "balanced_accuracy_all": round(ba_all, 4),
        "balanced_accuracy_covered": round(ba_covered, 4),
        "brier_score": round(bs, 4),
        "log_loss": round(ll, 4),
        "ece": round(ece, 4),
        "base_rate_positive": round(base_rate, 4),
    }


def _write_summary(
    path_b_metrics: dict[str, object],
    composite_info: dict[str, object],
    weights: dict[str, float],
    as_of: date | None,
) -> None:
    """Write v125 summary markdown."""
    ba_covered = path_b_metrics["balanced_accuracy_covered"]
    ref_ba = PATH_A_REFERENCE["balanced_accuracy_covered"]

    if isinstance(ba_covered, float) and not np.isnan(ba_covered) and isinstance(ref_ba, float):
        delta_ba = ba_covered - ref_ba
    else:
        delta_ba = float("nan")

    if not np.isnan(delta_ba) and delta_ba >= 0.03:
        verdict = (
            "Path B shows >= 3% balanced accuracy improvement over Path A reference. "
            "Elevate Path B to co-primary research track."
        )
    elif not np.isnan(delta_ba) and 0 < delta_ba < 0.03:
        verdict = (
            "Path B shows marginal improvement. Continue parallel development "
            "but retain Path A as primary."
        )
    else:
        verdict = (
            "Path B does not improve on Path A reference. Retain Path A as "
            "primary architecture. Archive Path B as secondary diagnostic."
        )

    # Weights table
    weights_rows = "\n".join(
        f"| {t} | {w:.2f} |" for t, w in weights.items()
    )

    # Metrics comparison table
    def _fmt(v: object) -> str:
        if v is None:
            return "--"
        if isinstance(v, float):
            if np.isnan(v):
                return "--"
            return f"{v:.4f}"
        return str(v)

    ref = PATH_A_REFERENCE
    pb = path_b_metrics

    metric_names = [
        "n_obs",
        "n_covered",
        "coverage",
        "balanced_accuracy_all",
        "balanced_accuracy_covered",
        "brier_score",
        "log_loss",
        "ece",
        "base_rate_positive",
    ]

    metric_rows = ""
    for m in metric_names:
        ref_val = ref.get(m)
        pb_val = pb.get(m)
        if isinstance(ref_val, (int, float)) and isinstance(pb_val, (int, float)):
            if isinstance(ref_val, float) and np.isnan(ref_val):
                delta_str = "--"
            elif isinstance(pb_val, float) and np.isnan(pb_val):
                delta_str = "--"
            else:
                delta_str = f"{pb_val - ref_val:+.4f}"
        else:
            delta_str = "--"
        metric_rows += f"| {m} | {_fmt(ref_val)} | {_fmt(pb_val)} | {delta_str} |\n"

    run_date = date.today().isoformat()
    as_of_str = str(as_of) if as_of else "latest available"

    md = f"""# v125 -- Path B: Composite Portfolio-Target Classifier

**Run date:** {run_date}
**As-of cutoff:** {as_of_str}

## Composite Return Target

- Observations: {composite_info['n_obs']}
- Date range: {composite_info['date_min']} to {composite_info['date_max']}
- Positive rate (sell signal): {composite_info['positive_rate']:.4f}

### Weights

| Benchmark | Weight |
|-----------|--------|
{weights_rows}

## WFO Metrics Comparison

| Metric | Path A (v92 ref) | Path B (v125) | Delta |
|--------|-------------------|---------------|-------|
{metric_rows}

## Architecture Verdict

{verdict}

## Notes

- Path B trains a single logistic classifier on the composite weighted relative
  return rather than separate per-benchmark classifiers.
- Composite target: sell signal when weighted relative return < -{THRESHOLD:.0%}.
- WFO parameters: {N_SPLITS} splits, gap={GAP}, test_size={TEST_SIZE},
  min_train_obs={MIN_TRAIN_OBS}.
- Feature set: {FEATURE_SET}.
- Abstain band: [{ABSTAIN_LOW:.2f}, {ABSTAIN_HIGH:.2f}] (coverage filter
  uses |prob - 0.5| > 0.2).
"""
    out_path = OUTPUT_DIR / "v125_portfolio_target_summary.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Summary written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="v125 composite portfolio-target classifier")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="As-of date for truncation (YYYY-MM-DD). Default: use all data.",
    )
    args = parser.parse_args()
    as_of: date | None = date.fromisoformat(args.as_of) if args.as_of else None

    # 1. Connect to DB
    conn = db_client.get_connection()

    # 2. Build feature matrix
    print("Building feature matrix...")
    feature_df = build_feature_matrix_from_db(conn, force_refresh=True)
    if as_of is not None:
        feature_df = feature_df.loc[feature_df.index <= pd.Timestamp(as_of)].sort_index()

    # 3. Get feature columns
    feature_cols = feature_set_from_name(feature_df, FEATURE_SET)
    print(f"Feature set '{FEATURE_SET}': {len(feature_cols)} features")

    # 4. Build composite return series
    print("Building composite return series...")
    composite_rel = build_composite_return_series(conn, as_of=as_of)
    print(
        f"Composite series: {len(composite_rel)} obs, "
        f"{composite_rel.index.min().date()} to {composite_rel.index.max().date()}"
    )

    # 5. Align features + target (binarize after alignment to avoid silent date shrinkage)
    X_aligned, y_cont = get_X_y_relative(feature_df, composite_rel, drop_na_target=True)
    X_aligned = X_aligned[feature_cols].copy()
    y_aligned = (y_cont < -THRESHOLD).astype(int)
    y_aligned.name = TARGET_LABEL

    positive_rate = float(y_aligned.mean())
    print(f"Positive rate (sell signal): {positive_rate:.4f}")
    print(f"Aligned dataset: {len(X_aligned)} observations")

    if len(X_aligned) < MIN_TRAIN_OBS:
        print(f"ERROR: Only {len(X_aligned)} obs, need at least {MIN_TRAIN_OBS}. Aborting.")
        sys.exit(1)

    # 7. Run WFO
    print(f"Running WFO ({N_SPLITS} splits, gap={GAP}, test={TEST_SIZE})...")
    fold_detail = run_wfo(X_aligned, y_aligned)
    if fold_detail.empty:
        print("ERROR: No valid WFO folds produced. Aborting.")
        sys.exit(1)
    print(f"WFO complete: {len(fold_detail)} test observations across folds")

    # 8. Compute metrics
    path_b_metrics = compute_metrics(fold_detail, "path_b_composite_logistic_v125")

    # 9. Save results CSV
    results_rows = [PATH_A_REFERENCE, path_b_metrics]
    results_df = pd.DataFrame(results_rows)
    results_path = OUTPUT_DIR / "v125_portfolio_target_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results CSV written to {results_path}")

    # 10. Save fold-detail CSV
    fold_path = OUTPUT_DIR / "v125_portfolio_target_fold_detail.csv"
    fold_detail.to_csv(fold_path, index=False)
    print(f"Fold detail CSV written to {fold_path}")

    # 11. Write summary
    composite_info = {
        "n_obs": len(composite_rel),
        "date_min": str(composite_rel.index.min().date()),
        "date_max": str(composite_rel.index.max().date()),
        "positive_rate": positive_rate,
    }
    _write_summary(path_b_metrics, composite_info, COMPOSITE_WEIGHTS, as_of)

    # Print key result
    ba_cov = path_b_metrics["balanced_accuracy_covered"]
    ref_ba = PATH_A_REFERENCE["balanced_accuracy_covered"]
    if isinstance(ba_cov, float) and not np.isnan(ba_cov) and isinstance(ref_ba, float):
        delta = ba_cov - ref_ba
        print(f"\nPath B balanced accuracy (covered): {ba_cov:.4f}")
        print(f"Path A reference (v92):              {ref_ba:.4f}")
        print(f"Delta:                               {delta:+.4f}")
    else:
        print(f"\nPath B balanced accuracy (covered): {ba_cov}")
    print("Done.")


if __name__ == "__main__":
    main()
