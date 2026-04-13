"""v129 -- VGT robustness audit across multiple as-of dates.

Audit the v128 forward-stepwise VGT 2-feature finding
(rate_adequacy_gap_yoy, severity_index_yoy) for temporal stability.

The v128 search reported covered balanced accuracy of 0.947 for the VGT
benchmark using only 2 features. This script tests whether that result
holds at earlier as-of dates or is a one-off artefact.

Methodology:
  - Re-run the WFO evaluation at 3 as-of dates: 2022-03-31, 2023-03-31,
    2024-03-31 (the original v128 date).
  - Compare the 2-feature model against the lean_baseline 12-feature
    control at each date.
  - Assess stability: adopt only if BA advantage >= +0.05 at all dates
    AND n_covered >= 10 at every date.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    GAP_MONTHS,
    MAX_TRAIN_MONTHS,
    RESULTS_DIR,
    RIDGE_FEATURES_12,
    TEST_SIZE_MONTHS,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    print_footer,
    print_header,
)
from src.research.v87_utils import (
    build_target_series,
    prequential_logistic_calibration,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AS_OF_DATES: list[str] = ["2022-03-31", "2023-03-31", "2024-03-31"]
VGT_2_FEATURE: list[str] = ["rate_adequacy_gap_yoy", "severity_index_yoy"]
LEAN_BASELINE: list[str] = list(RIDGE_FEATURES_12)
ACTIONABLE_TARGET: str = "actionable_sell_3pct"
LOWER_THRESHOLD: float = 0.30
UPPER_THRESHOLD: float = 0.70
MIN_COVERED_OBS: int = 10
MIN_BA_ADVANTAGE: float = 0.05

RESULTS_CSV = RESULTS_DIR / "v129_vgt_robustness_audit_results.csv"
SUMMARY_MD = RESULTS_DIR / "v129_vgt_robustness_audit_summary.md"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def model_builder() -> LogisticRegression:
    """Construct the same balanced logistic model used in v128."""
    return LogisticRegression(
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
        l1_ratio=0.0,
        max_iter=5000,
        random_state=42,
    )


def resolve_splitter(n_obs: int) -> TimeSeriesSplit | None:
    """Return the repo-standard rolling WFO splitter."""
    available = n_obs - MAX_TRAIN_MONTHS - GAP_MONTHS
    if available < TEST_SIZE_MONTHS:
        return None
    n_splits = max(1, available // TEST_SIZE_MONTHS)
    return TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=MAX_TRAIN_MONTHS,
        test_size=TEST_SIZE_MONTHS,
        gap=GAP_MONTHS,
    )


def impute_fold_arrays(
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


def evaluate_wfo(
    x_df: pd.DataFrame,
    y_series: pd.Series,
    feature_list: list[str],
) -> pd.DataFrame:
    """Run WFO and return OOS prediction rows."""
    filtered = [f for f in feature_list if f in x_df.columns]
    aligned = x_df[filtered].join(y_series, how="inner").dropna(
        subset=[y_series.name]
    ).copy()
    if aligned.empty:
        return pd.DataFrame(columns=["date", "y_true", "y_prob"])

    splitter = resolve_splitter(len(aligned))
    if splitter is None:
        return pd.DataFrame(columns=["date", "y_true", "y_prob"])

    x_values = aligned[filtered].to_numpy(dtype=float)
    y_values = aligned[y_series.name].to_numpy(dtype=int)
    dates = pd.DatetimeIndex(aligned.index)

    rows: list[dict[str, object]] = []
    for train_idx, test_idx in splitter.split(x_values):
        y_train = y_values[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        x_train, x_test = impute_fold_arrays(
            x_values[train_idx], x_values[test_idx]
        )
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


def compute_covered_metrics(pred_df: pd.DataFrame) -> dict[str, object]:
    """Compute covered BA and brier score from calibrated predictions."""
    if pred_df.empty:
        return {
            "n_obs": 0,
            "n_covered": 0,
            "balanced_accuracy_covered": float("nan"),
            "brier_score": float("nan"),
        }

    calibrated = pred_df.copy()
    calibrated["y_prob_cal"] = prequential_logistic_calibration(
        calibrated["y_true"].to_numpy(dtype=int),
        calibrated["y_prob"].to_numpy(dtype=float),
    )

    y_true = calibrated["y_true"].to_numpy(dtype=int)
    y_prob = np.clip(calibrated["y_prob_cal"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)

    covered_mask = (y_prob <= LOWER_THRESHOLD) | (y_prob >= UPPER_THRESHOLD)
    n_covered = int(covered_mask.sum())

    if n_covered > 0 and len(np.unique(y_true[covered_mask])) >= 2:
        covered_pred = (y_prob[covered_mask] >= 0.5).astype(int)
        ba_covered = float(balanced_accuracy_score(y_true[covered_mask], covered_pred))
    else:
        ba_covered = float("nan")

    bs = float(brier_score_loss(y_true, y_prob))

    return {
        "n_obs": len(y_true),
        "n_covered": n_covered,
        "balanced_accuracy_covered": ba_covered,
        "brier_score": bs,
    }


def single_feature_ranking(
    x_df: pd.DataFrame,
    y_series: pd.Series,
    feature_pool: list[str],
) -> list[tuple[str, float]]:
    """Rank features by individual covered BA (for stability check)."""
    results: list[tuple[str, float]] = []
    for feature in feature_pool:
        if feature not in x_df.columns:
            continue
        pred_df = evaluate_wfo(x_df, y_series, [feature])
        metrics = compute_covered_metrics(pred_df)
        ba = metrics["balanced_accuracy_covered"]
        results.append((feature, ba if not np.isnan(ba) else -1.0))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def determine_verdict(results_df: pd.DataFrame) -> str:
    """Return STABLE or UNSTABLE based on cross-date audit results.

    Adopt (STABLE) only if:
      - BA advantage (2-feature minus lean_baseline) >= MIN_BA_ADVANTAGE at all dates
      - n_covered >= MIN_COVERED_OBS at all dates for the 2-feature model
    """
    two_feat = results_df[results_df["model"] == "vgt_2feature"].copy()
    baseline = results_df[results_df["model"] == "lean_baseline"].copy()

    if two_feat.empty or baseline.empty:
        return "UNSTABLE"

    merged = two_feat.merge(
        baseline[["as_of_date", "balanced_accuracy_covered"]],
        on="as_of_date",
        suffixes=("_2feat", "_baseline"),
    )

    for _, row in merged.iterrows():
        ba_2 = row["balanced_accuracy_covered_2feat"]
        ba_b = row["balanced_accuracy_covered_baseline"]
        n_cov = row["n_covered"]
        if np.isnan(ba_2) or np.isnan(ba_b):
            return "UNSTABLE"
        if (ba_2 - ba_b) < MIN_BA_ADVANTAGE:
            return "UNSTABLE"
        if n_cov < MIN_COVERED_OBS:
            return "UNSTABLE"

    return "STABLE"


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def run_audit() -> None:
    """Execute the VGT robustness audit."""
    print_header("v129", "VGT Robustness Audit")

    conn = get_connection()
    try:
        feature_df = load_feature_matrix(conn)
        rel_series_full = load_relative_series(conn, "VGT", horizon=6)
    finally:
        conn.close()

    if rel_series_full is None or rel_series_full.empty:
        print("ERROR: No VGT relative return data found.")
        return

    print(f"Full VGT relative return range: "
          f"{rel_series_full.index.min().date()} to {rel_series_full.index.max().date()}")
    print(f"Full feature matrix range: "
          f"{feature_df.index.min().date()} to {feature_df.index.max().date()}")
    print()

    all_rows: list[dict[str, object]] = []
    stability_notes: list[str] = []

    for as_of_date_str in AS_OF_DATES:
        as_of_date = pd.Timestamp(as_of_date_str)
        print(f"=== As-of date: {as_of_date.date()} ===")

        # Truncate data to as-of date
        feat_trunc = feature_df.loc[feature_df.index <= as_of_date].copy()
        rel_trunc = rel_series_full.loc[rel_series_full.index <= as_of_date].copy()

        if rel_trunc.empty:
            print(f"  No VGT data available up to {as_of_date.date()}")
            continue

        x_base, _ = get_X_y_relative(feat_trunc, rel_trunc, drop_na_target=True)
        target = build_target_series(rel_trunc, ACTIONABLE_TARGET)

        # Evaluate 2-feature model
        print(f"  Evaluating 2-feature model ...")
        pred_2feat = evaluate_wfo(x_base, target, VGT_2_FEATURE)
        metrics_2feat = compute_covered_metrics(pred_2feat)
        print(f"    n_obs={metrics_2feat['n_obs']}  "
              f"n_covered={metrics_2feat['n_covered']}  "
              f"BA_covered={metrics_2feat['balanced_accuracy_covered']:.4f}  "
              f"brier={metrics_2feat['brier_score']:.4f}")

        all_rows.append({
            "as_of_date": as_of_date_str,
            "model": "vgt_2feature",
            "features": "|".join(VGT_2_FEATURE),
            "n_features": len(VGT_2_FEATURE),
            **metrics_2feat,
        })

        # Evaluate lean_baseline control
        print(f"  Evaluating lean_baseline control ...")
        pred_baseline = evaluate_wfo(x_base, target, LEAN_BASELINE)
        metrics_baseline = compute_covered_metrics(pred_baseline)
        print(f"    n_obs={metrics_baseline['n_obs']}  "
              f"n_covered={metrics_baseline['n_covered']}  "
              f"BA_covered={metrics_baseline['balanced_accuracy_covered']:.4f}  "
              f"brier={metrics_baseline['brier_score']:.4f}")

        all_rows.append({
            "as_of_date": as_of_date_str,
            "model": "lean_baseline",
            "features": "|".join(LEAN_BASELINE),
            "n_features": len(LEAN_BASELINE),
            **metrics_baseline,
        })

        # Stability check: are the 2 features in top-3 individually?
        print(f"  Single-feature stability ranking ...")
        all_eligible = [
            f for f in x_base.columns
            if int(x_base[f].notna().sum()) >= 60
        ]
        ranking = single_feature_ranking(x_base, target, all_eligible)
        top3_features = [name for name, _ in ranking[:3]]
        feat_in_top3 = [f for f in VGT_2_FEATURE if f in top3_features]
        note = (
            f"  {as_of_date.date()}: top-3 = {top3_features}; "
            f"VGT 2-feature in top-3: {feat_in_top3}"
        )
        stability_notes.append(note)
        print(note)
        print()

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved to {RESULTS_CSV}")

    verdict = determine_verdict(results_df)
    print(f"\nVerdict: {verdict}")

    # Write summary
    _write_summary(results_df, stability_notes, verdict)
    print(f"Summary saved to {SUMMARY_MD}")
    print_footer()


def _write_summary(
    results_df: pd.DataFrame,
    stability_notes: list[str],
    verdict: str,
) -> None:
    """Write the markdown summary artifact."""
    lines: list[str] = [
        "# v129 VGT Robustness Audit Summary",
        "",
        "## Objective",
        "",
        "Audit the v128 VGT forward-stepwise 2-feature finding",
        "(`rate_adequacy_gap_yoy`, `severity_index_yoy`) for temporal stability.",
        "The v128 search reported covered balanced accuracy of 0.947 (n_covered=21)",
        "vs lean_baseline BA of 0.579 -- a +0.368 advantage that demands scrutiny.",
        "",
        "## Methodology",
        "",
        "- Re-run the WFO evaluation at 3 as-of dates: 2022-03-31, 2023-03-31, 2024-03-31",
        "- Same WFO geometry: max_train=60, test_size=6, gap=8 (TimeSeriesSplit)",
        "- Same model: LogisticRegression(C=0.5, class_weight='balanced', l1_ratio=0.0, solver='lbfgs')",
        "- Same calibration: prequential logistic calibration",
        "- Covered BA computed on observations where P(sell) <= 0.30 or >= 0.70",
        "",
        "## Results",
        "",
        "| As-of Date | Model | n_obs | n_covered | BA_covered | Brier |",
        "|------------|-------|-------|-----------|------------|-------|",
    ]

    for _, row in results_df.iterrows():
        ba = row["balanced_accuracy_covered"]
        bs = row["brier_score"]
        ba_str = f"{ba:.4f}" if not np.isnan(ba) else "N/A"
        bs_str = f"{bs:.4f}" if not np.isnan(bs) else "N/A"
        lines.append(
            f"| {row['as_of_date']} | {row['model']} | "
            f"{row['n_obs']} | {row['n_covered']} | "
            f"{ba_str} | {bs_str} |"
        )

    lines.extend([
        "",
        "## BA Advantage (2-feature minus lean_baseline)",
        "",
    ])
    two_feat = results_df[results_df["model"] == "vgt_2feature"].set_index("as_of_date")
    baseline = results_df[results_df["model"] == "lean_baseline"].set_index("as_of_date")
    for date in AS_OF_DATES:
        if date in two_feat.index and date in baseline.index:
            ba2 = two_feat.loc[date, "balanced_accuracy_covered"]
            bab = baseline.loc[date, "balanced_accuracy_covered"]
            delta = ba2 - bab if not (np.isnan(ba2) or np.isnan(bab)) else float("nan")
            n_cov = two_feat.loc[date, "n_covered"]
            delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
            lines.append(f"- **{date}**: delta_BA = {delta_str}, n_covered = {n_cov}")

    lines.extend([
        "",
        "## Single-Feature Stability Check",
        "",
    ])
    for note in stability_notes:
        lines.append(f"- {note.strip()}")

    lines.extend([
        "",
        "## Economic Plausibility",
        "",
        "PGR (Progressive Corp) is a P&C insurance company. VGT is a technology",
        "sector ETF. The two winning features are insurance-specific:",
        "",
        "- `rate_adequacy_gap_yoy`: measures how much PGR's pricing power is",
        "  changing year-over-year. When PGR raises rates faster than loss costs",
        "  grow, underwriting margins expand, boosting PGR relative to non-insurance",
        "  benchmarks like VGT.",
        "- `severity_index_yoy`: captures claims severity inflation. Rising severity",
        "  compresses margins unless offset by rate adequacy, creating a direct",
        "  headwind to PGR vs tech-sector returns.",
        "",
        "These features are economically sensible for predicting PGR-vs-VGT relative",
        "returns because VGT has essentially zero exposure to underwriting cycle",
        "dynamics, making insurance-specific features orthogonal to VGT's drivers.",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
    ])

    adopt_criteria = (
        "Adoption criteria: BA advantage >= +0.05 at all 3 as-of dates "
        f"AND n_covered >= {MIN_COVERED_OBS} at every date."
    )
    lines.append(adopt_criteria)

    if verdict == "STABLE":
        lines.append("")
        lines.append(
            "Recommendation: adopt the 2-feature VGT-specific model "
            "(`rate_adequacy_gap_yoy`, `severity_index_yoy`) for VGT classification."
        )
    else:
        lines.append("")
        lines.append(
            "Recommendation: do NOT adopt. The v128 VGT result does not satisfy "
            "the temporal stability criteria. Retain the lean_baseline for VGT."
        )

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_audit()
