"""
v125 / v126 -- Path B: Composite portfolio-target classifier.

v125 introduced Path B: a single logistic classifier trained on the composite
balanced_pref_95_5 portfolio-return target.

v126 hardens that work so the comparison is methodologically comparable:
1. Path A is recomputed on the same matched dates rather than referenced from
   a hardcoded legacy result.
2. Both paths use the configured rolling WFO window (`WFO_TRAIN_WINDOW_MONTHS`)
   with the same purge/embargo gap and test size.
3. The saved fold-detail artifact includes both Path A and Path B probabilities
   on the exact matched evaluation dates.

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
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.features import (
    INVESTABLE_CLASSIFIER_BASE_WEIGHTS,
    INVESTABLE_CLASSIFIER_BENCHMARKS,
)
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
    CALIBRATION_MIN_HISTORY,
    build_target_series,
    feature_set_from_name,
    logistic_factory,
    prequential_logistic_calibration,
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


def _resolve_n_splits(
    n_obs: int,
    *,
    requested_n_splits: int,
    train_window_months: int,
    gap: int,
    test_size: int,
) -> int:
    """Return the feasible number of rolling WFO splits for one dataset."""
    available = n_obs - train_window_months - gap
    if available < test_size:
        return 0
    max_splits = available // test_size
    return min(requested_n_splits, max_splits)


def _impute_train_test(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute train/test folds using training medians only."""
    x_train_vals = x_train.to_numpy(dtype=float).copy()  # .copy() ensures writable (pandas 3.0+)
    medians = np.nanmedian(x_train_vals, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for col_idx in range(x_train_vals.shape[1]):
        nan_mask = np.isnan(x_train_vals[:, col_idx])
        x_train_vals[nan_mask, col_idx] = medians[col_idx]

    x_test_vals = x_test.to_numpy(dtype=float).copy()  # writable
    for col_idx in range(x_test_vals.shape[1]):
        nan_mask = np.isnan(x_test_vals[:, col_idx])
        x_test_vals[nan_mask, col_idx] = medians[col_idx]
    return x_train_vals, x_test_vals


def _aggregate_probability_panel(
    probability_df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    """Aggregate benchmark probabilities with row-wise weight renormalization."""
    if probability_df.empty:
        return pd.DataFrame(
            columns=["test_date", "path_a_prob", "path_a_available_benchmarks", "path_a_weight_sum"]
        )

    weight_series = pd.Series(weights, dtype=float)
    rows: list[dict[str, object]] = []
    for test_date, row in probability_df.sort_index().iterrows():
        available = row.dropna()
        if available.empty:
            continue
        row_weights = weight_series.reindex(available.index).fillna(0.0)
        total_weight = float(row_weights.sum())
        if total_weight <= 0.0:
            continue
        normalized = row_weights / total_weight
        rows.append(
            {
                "test_date": pd.Timestamp(test_date),
                "path_a_prob": float((available * normalized).sum()),
                "path_a_available_benchmarks": int(len(available)),
                "path_a_weight_sum": total_weight,
            }
        )
    return pd.DataFrame(rows)


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
    train_window_months: int = WFO_TRAIN_WINDOW_MONTHS,
) -> pd.DataFrame:
    """Walk-forward optimisation via TimeSeriesSplit."""
    resolved_n_splits = _resolve_n_splits(
        len(X),
        requested_n_splits=n_splits,
        train_window_months=train_window_months,
        gap=gap,
        test_size=test_size,
    )
    if resolved_n_splits == 0:
        return pd.DataFrame(
            columns=["fold", "train_start", "train_end", "train_obs", "test_date", "y_true", "y_prob"]
        )

    tscv = TimeSeriesSplit(
        n_splits=resolved_n_splits,
        gap=gap,
        test_size=test_size,
        max_train_size=train_window_months,
    )

    records: list[dict[str, object]] = []
    model_factory = logistic_factory(class_weight="balanced", c_value=0.5)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()

        if len(X_train) < min_train_obs:
            continue

        unique_classes = np.unique(y_train.to_numpy(dtype=int))
        if len(unique_classes) < 2:
            continue

        x_train_vals, x_test_vals = _impute_train_test(X_train, X_test)

        model = model_factory()
        model.fit(x_train_vals, y_train.to_numpy(dtype=int))
        y_prob = model.predict_proba(x_test_vals)[:, 1]

        for i, test_pos in enumerate(test_idx):
            records.append(
                {
                    "fold": fold_idx,
                    "train_start": str(X.index[train_idx[0]].date()),
                    "train_end": str(X.index[train_idx[-1]].date()),
                    "train_obs": int(len(train_idx)),
                    "test_date": str(X.index[test_pos].date()),
                    "y_true": int(y_test.iloc[i]),
                    "y_prob": float(y_prob[i]),
                }
            )

    return pd.DataFrame(records)


def build_path_a_matched_probability_frame(
    conn: object,
    feature_df: pd.DataFrame,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Recompute Path A on the same rolling WFO geometry used by Path B."""
    feature_cols = feature_set_from_name(feature_df, FEATURE_SET)
    benchmark_prob_series: list[pd.Series] = []

    for benchmark in INVESTABLE_CLASSIFIER_BENCHMARKS:
        if benchmark not in COMPOSITE_WEIGHTS:
            continue
        rel_series = load_relative_return_matrix(conn, benchmark, HORIZON_MONTHS)
        if rel_series is None or rel_series.empty:
            continue
        if as_of is not None:
            rel_series = truncate_relative_target_for_asof(
                rel_series,
                as_of=pd.Timestamp(as_of),
                horizon_months=HORIZON_MONTHS,
            )
        x_base, _ = get_X_y_relative(feature_df, rel_series, drop_na_target=True)
        usable_features = [feature for feature in feature_cols if feature in x_base.columns]
        if not usable_features:
            continue

        target = build_target_series(rel_series, TARGET_LABEL)
        y_benchmark = target.reindex(x_base.index).dropna().astype(int)
        X_benchmark = x_base.loc[y_benchmark.index, usable_features].copy()
        if len(X_benchmark) < MIN_TRAIN_OBS:
            continue

        benchmark_fold_df = run_wfo(
            X_benchmark,
            y_benchmark,
            n_splits=N_SPLITS,
            gap=GAP,
            test_size=TEST_SIZE,
            min_train_obs=MIN_TRAIN_OBS,
            train_window_months=WFO_TRAIN_WINDOW_MONTHS,
        )
        if benchmark_fold_df.empty:
            continue

        calibrated = prequential_logistic_calibration(
            benchmark_fold_df["y_true"].to_numpy(dtype=int),
            benchmark_fold_df["y_prob"].to_numpy(dtype=float),
            min_history=CALIBRATION_MIN_HISTORY,
        )
        benchmark_series = pd.Series(
            calibrated,
            index=pd.to_datetime(benchmark_fold_df["test_date"]),
            name=benchmark,
            dtype=float,
        )
        benchmark_prob_series.append(benchmark_series)

    if not benchmark_prob_series:
        return pd.DataFrame(
            columns=["test_date", "path_a_prob", "path_a_available_benchmarks", "path_a_weight_sum"]
        )

    probability_df = pd.concat(benchmark_prob_series, axis=1).sort_index()
    return _aggregate_probability_panel(probability_df, COMPOSITE_WEIGHTS)


def compute_metrics(
    results_df: pd.DataFrame,
    label: str,
    *,
    probability_col: str = "y_prob",
) -> dict[str, object]:
    """Pool all WFO observations and compute classification metrics."""
    y_true = results_df["y_true"].to_numpy()
    y_prob = results_df[probability_col].to_numpy()
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


def _metric_delta(
    path_a_metrics: dict[str, object],
    path_b_metrics: dict[str, object],
    key: str,
) -> float | None:
    """Return one numeric delta when both metrics are present."""
    path_a_value = path_a_metrics.get(key)
    path_b_value = path_b_metrics.get(key)
    if isinstance(path_a_value, (int, float)) and isinstance(path_b_value, (int, float)):
        if isinstance(path_a_value, float) and np.isnan(path_a_value):
            return None
        if isinstance(path_b_value, float) and np.isnan(path_b_value):
            return None
        return float(path_b_value - path_a_value)
    return None


def _verdict_text(
    path_a_metrics: dict[str, object],
    path_b_metrics: dict[str, object],
) -> str:
    """Return a conservative verdict from the matched Path A vs Path B comparison."""
    delta_ba = _metric_delta(path_a_metrics, path_b_metrics, "balanced_accuracy_covered")
    delta_brier = _metric_delta(path_a_metrics, path_b_metrics, "brier_score")
    delta_ece = _metric_delta(path_a_metrics, path_b_metrics, "ece")

    if (
        delta_ba is not None
        and delta_ba >= 0.03
        and delta_brier is not None
        and delta_brier <= 0.0
        and delta_ece is not None
        and delta_ece <= 0.0
    ):
        return (
            "Path B improves covered balanced accuracy on the matched v126 comparison "
            "without degrading calibration. Elevate Path B to a co-primary research track."
        )
    if delta_ba is not None and delta_ba >= 0.03:
        return (
            "Path B improves covered balanced accuracy on the matched v126 comparison, "
            "but calibration worsens versus Path A. Keep Path B as a secondary research "
            "track until calibration work closes that gap."
        )
    if delta_ba is not None and delta_ba > 0.0:
        return (
            "Path B shows only marginal covered balanced-accuracy improvement on the "
            "matched v126 comparison. Retain Path A as primary and continue Path B in parallel."
        )
    return (
        "Path B does not outperform Path A on the matched v126 comparison. Retain "
        "Path A as the primary architecture and keep Path B diagnostic-only."
    )


def _write_summary(
    path_a_metrics: dict[str, object],
    path_b_metrics: dict[str, object],
    composite_info: dict[str, object],
    weights: dict[str, float],
    as_of: date | None,
    *,
    realized_folds: int,
    path_a_benchmark_range: str,
) -> None:
    """Write v125 summary markdown."""
    verdict = _verdict_text(path_a_metrics, path_b_metrics)

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

    ref = path_a_metrics
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

    md = f"""# v126 -- Path B Methodology Hardening (v125 Remediation)

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

| Metric | Path A (matched) | Path B (matched) | Delta |
|--------|-------------------|---------------|-------|
{metric_rows}

## Architecture Verdict

{verdict}

## Notes

- Path A is recomputed here as the matched benchmark-specific baseline:
  separate per-benchmark logistic classifiers, prequential calibration,
  and row-wise renormalized fixed investable weights.
- Path B trains a single logistic classifier on the composite weighted relative
  return rather than separate per-benchmark classifiers.
- Composite target: sell signal when weighted relative return < -{THRESHOLD:.0%}.
- Rolling WFO parameters: requested_splits={N_SPLITS}, realized_splits={realized_folds},
  train_window={WFO_TRAIN_WINDOW_MONTHS}, gap={GAP}, test_size={TEST_SIZE},
  min_train_obs={MIN_TRAIN_OBS}.
- Feature set: {FEATURE_SET}.
- Abstain band: [{ABSTAIN_LOW:.2f}, {ABSTAIN_HIGH:.2f}] (coverage filter
  uses |prob - 0.5| > 0.2).
- Path A available benchmark count per matched month: {path_a_benchmark_range}.
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

    conn = db_client.get_connection()
    try:
        # 1. Build feature matrix
        print("Building feature matrix...")
        feature_df = build_feature_matrix_from_db(conn, force_refresh=True)
        if as_of is not None:
            feature_df = feature_df.loc[feature_df.index <= pd.Timestamp(as_of)].sort_index()

        # 2. Get feature columns
        feature_cols = feature_set_from_name(feature_df, FEATURE_SET)
        print(f"Feature set '{FEATURE_SET}': {len(feature_cols)} features")

        # 3. Build composite return series
        print("Building composite return series...")
        composite_rel = build_composite_return_series(conn, as_of=as_of)
        print(
            f"Composite series: {len(composite_rel)} obs, "
            f"{composite_rel.index.min().date()} to {composite_rel.index.max().date()}"
        )

        # 4. Align features + target (binarize after alignment to avoid silent date shrinkage)
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

        # 5. Run matched Path B rolling WFO
        print(
            f"Running Path B rolling WFO (requested={N_SPLITS}, "
            f"train_window={WFO_TRAIN_WINDOW_MONTHS}, gap={GAP}, test={TEST_SIZE})..."
        )
        path_b_fold_detail = run_wfo(
            X_aligned,
            y_aligned,
            n_splits=N_SPLITS,
            gap=GAP,
            test_size=TEST_SIZE,
            min_train_obs=MIN_TRAIN_OBS,
            train_window_months=WFO_TRAIN_WINDOW_MONTHS,
        )
        if path_b_fold_detail.empty:
            print("ERROR: No valid WFO folds produced. Aborting.")
            sys.exit(1)
        print(
            "Path B WFO complete: "
            f"{len(path_b_fold_detail)} test observations across "
            f"{path_b_fold_detail['fold'].nunique()} folds"
        )

        # 6. Rebuild matched Path A baseline on the same rolling geometry
        print("Recomputing matched Path A benchmark-aggregate baseline...")
        path_a_frame = build_path_a_matched_probability_frame(
            conn,
            feature_df,
            as_of=as_of,
        )
        if path_a_frame.empty:
            print("ERROR: Path A matched baseline could not be reconstructed. Aborting.")
            sys.exit(1)

        fold_detail = path_b_fold_detail.rename(columns={"y_prob": "path_b_prob"}).copy()
        fold_detail["test_date"] = pd.to_datetime(fold_detail["test_date"])
        fold_detail = fold_detail.merge(path_a_frame, on="test_date", how="inner")
        if fold_detail.empty:
            print("ERROR: No matched dates between Path A and Path B. Aborting.")
            sys.exit(1)
        fold_detail["test_date"] = fold_detail["test_date"].dt.date.astype(str)

        # 7. Compute matched metrics
        path_a_metrics = compute_metrics(
            fold_detail,
            "path_a_investable_pool_matched_v126",
            probability_col="path_a_prob",
        )
        path_b_metrics = compute_metrics(
            fold_detail,
            "path_b_composite_logistic_matched_v126",
            probability_col="path_b_prob",
        )

        # 8. Save results CSV
        results_rows = [path_a_metrics, path_b_metrics]
        results_df = pd.DataFrame(results_rows)
        results_path = OUTPUT_DIR / "v125_portfolio_target_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Results CSV written to {results_path}")

        # 9. Save fold-detail CSV
        fold_path = OUTPUT_DIR / "v125_portfolio_target_fold_detail.csv"
        fold_detail.to_csv(fold_path, index=False)
        print(f"Fold detail CSV written to {fold_path}")

        # 10. Write summary
        available_counts = fold_detail["path_a_available_benchmarks"].astype(int)
        composite_info = {
            "n_obs": len(composite_rel),
            "date_min": str(composite_rel.index.min().date()),
            "date_max": str(composite_rel.index.max().date()),
            "positive_rate": positive_rate,
        }
        _write_summary(
            path_a_metrics,
            path_b_metrics,
            composite_info,
            COMPOSITE_WEIGHTS,
            as_of,
            realized_folds=int(fold_detail["fold"].nunique()),
            path_a_benchmark_range=f"{available_counts.min()}-{available_counts.max()}",
        )

        # Print key result
        ba_cov = path_b_metrics["balanced_accuracy_covered"]
        ref_ba = path_a_metrics["balanced_accuracy_covered"]
        if (
            isinstance(ba_cov, float)
            and not np.isnan(ba_cov)
            and isinstance(ref_ba, float)
            and not np.isnan(ref_ba)
        ):
            delta = ba_cov - ref_ba
            print(f"\nPath B balanced accuracy (covered): {ba_cov:.4f}")
            print(f"Path A matched baseline:             {ref_ba:.4f}")
            print(f"Delta:                               {delta:+.4f}")
        else:
            print(f"\nPath B balanced accuracy (covered): {ba_cov}")
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
