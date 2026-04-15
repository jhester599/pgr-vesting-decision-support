"""v132 -- Temporal hold-out validation for asymmetric abstention thresholds.

Design
------
The v131 autoresearch sweep evaluated 41 threshold pairs against all 84 OOS
rows from v125_portfolio_target_fold_detail.csv.  This introduces selection
bias: the winning pair (0.15, 0.70) was chosen by seeing every row.

This script validates the finding using a strict temporal hold-out:

  Selection set  (n=63): 2016-10-31 -> 2021-12-31  (rows the v131 sweep saw)
  Hold-out set   (n=21): 2022-01-31 -> 2023-09-29  (never used in any prior sweep)

Procedure
---------
1. Apply prequential temperature scaling to the full 84-row sequence (correct
   -- each row is calibrated only by its predecessors).
2. Re-run the v131 threshold grid on the selection set only -> selection winner.
3. Evaluate three pairs on the hold-out:
     a. Baseline         (0.30, 0.70)
     b. A-priori v131    (0.15, 0.70)  -- the credible finding from v131
     c. Selection winner              -- data-driven pick from step 2
4. Adoption verdict: ADOPT if hold-out BA delta >= 0.03 AND coverage >= 0.20.

Outputs
-------
  results/research/v132_threshold_validation_results.csv
  results/research/v132_threshold_validation_summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
FOLD_DETAIL_PATH: Path = (
    PROJECT_ROOT / "results" / "research" / "v125_portfolio_target_fold_detail.csv"
)
RESULTS_PATH: Path = (
    PROJECT_ROOT / "results" / "research" / "v132_threshold_validation_results.csv"
)
SUMMARY_PATH: Path = (
    PROJECT_ROOT / "results" / "research" / "v132_threshold_validation_summary.md"
)

HOLDOUT_CUTOFF: str = "2021-12-31"
MIN_COVERAGE: float = 0.20
MIN_BA_DELTA_FOR_ADOPTION: float = 0.03
WARMUP: int = 24

# Candidate threshold pairs (same grid as v131 sweep)
_STEP = 0.05
_CANDIDATE_PAIRS: list[tuple[float, float]] = [
    (round(lo, 2), round(hi, 2))
    for lo in [round(x * _STEP + 0.10, 2) for x in range(8)]   # 0.10 -> 0.45
    for hi in [round(x * _STEP + 0.55, 2) for x in range(8)]   # 0.55 -> 0.90
    if round(hi, 2) > round(lo + 0.20, 2)
]

# ---------------------------------------------------------------------------
# Temperature scaling (verbatim from path_b_classifier.py for reproducibility)
# ---------------------------------------------------------------------------
_GRID_TEMPERATURES: np.ndarray = np.concatenate(
    [np.linspace(0.50, 0.95, 10), np.linspace(1.0, 3.0, 41)]
)


def _clip(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(probs: np.ndarray) -> np.ndarray:
    p = _clip(probs)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _fit_temp(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = _clip(y_prob)
    best_t, best_loss = 1.0, float("inf")
    for t in _GRID_TEMPERATURES:
        scaled = np.clip(_sigmoid(_logit(p) / float(t)), 1e-6, 1.0 - 1e-6)
        loss = -float(
            np.mean(y_true * np.log(scaled) + (1 - y_true) * np.log(1.0 - scaled))
        )
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


def apply_prequential_temperature_scaling(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    warmup: int = WARMUP,
) -> np.ndarray:
    """Apply prequential (walk-forward) temperature scaling.

    For each index t >= warmup, fits temperature on probs[:t]/labels[:t]
    and applies it to probs[t]. Pre-warmup observations are clipped but
    otherwise unchanged.

    Parameters
    ----------
    probs:
        Raw OOS probabilities in chronological order.
    labels:
        Integer binary labels (0/1) aligned with probs.
    warmup:
        Minimum prior observations required before calibration is applied.

    Returns
    -------
    np.ndarray
        Calibrated probabilities, same shape as probs, clipped to [0, 1].
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    calibrated = _clip(probs).copy()
    for idx in range(len(probs)):
        if idx < warmup or len(np.unique(labels[:idx])) < 2:
            continue
        t = _fit_temp(labels[:idx], probs[:idx])
        raw = float(probs[idx])
        calibrated[idx] = float(
            np.clip(
                _sigmoid(np.array([_logit(np.array([raw]))[0] / t]))[0],
                1e-6,
                1.0 - 1e-6,
            )
        )
    return np.clip(calibrated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low: float,
    high: float,
) -> dict[str, float]:
    """Return covered_ba and coverage for the given abstention band.

    Rows with ``low <= y_prob <= high`` are abstained (excluded from scoring).
    Covered rows are those with ``y_prob < low OR y_prob > high``.

    Parameters
    ----------
    y_true:
        Integer binary labels (0/1).
    y_prob:
        Temperature-scaled probabilities aligned with y_true.
    low:
        Lower abstention boundary.
    high:
        Upper abstention boundary.

    Returns
    -------
    dict with keys ``covered_ba`` and ``coverage``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = (y_prob < low) | (y_prob > high)
    n_covered = int(mask.sum())
    coverage = n_covered / len(y_true) if len(y_true) > 0 else 0.0
    if n_covered == 0 or len(np.unique(y_true[mask])) < 2:
        return {"covered_ba": 0.5, "coverage": coverage}
    covered_ba = float(
        balanced_accuracy_score(y_true[mask], (y_prob[mask] >= 0.5).astype(int))
    )
    return {"covered_ba": covered_ba, "coverage": coverage}


def run_threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pairs: list[tuple[float, float]],
) -> pd.DataFrame:
    """Evaluate every (low, high) pair and return a DataFrame of metrics.

    Parameters
    ----------
    y_true:
        Integer binary labels (0/1).
    y_prob:
        Calibrated probabilities aligned with y_true.
    pairs:
        List of (low, high) threshold pairs to evaluate.

    Returns
    -------
    pd.DataFrame with columns: low, high, covered_ba, coverage.
    """
    rows = []
    for low, high in pairs:
        m = evaluate_thresholds(y_true, y_prob, low=low, high=high)
        rows.append({"low": low, "high": high, **m})
    return pd.DataFrame(rows)


def derive_verdict(
    holdout_ba_candidate: float,
    holdout_ba_baseline: float,
    holdout_coverage_candidate: float,
) -> str:
    """Return an adoption verdict string based on hold-out metrics.

    Adoption requires BOTH:
      - BA delta >= MIN_BA_DELTA_FOR_ADOPTION (0.03)
      - coverage >= MIN_COVERAGE (0.20)

    Parameters
    ----------
    holdout_ba_candidate:
        Covered balanced accuracy of the candidate pair on the hold-out set.
    holdout_ba_baseline:
        Covered balanced accuracy of the baseline (0.30, 0.70) on hold-out.
    holdout_coverage_candidate:
        Coverage fraction of the candidate pair on the hold-out set.

    Returns
    -------
    str starting with "ADOPT" or "DO NOT ADOPT".
    """
    ba_delta = holdout_ba_candidate - holdout_ba_baseline
    if ba_delta >= MIN_BA_DELTA_FOR_ADOPTION and holdout_coverage_candidate >= MIN_COVERAGE:
        return (
            f"ADOPT: hold-out BA delta = {ba_delta:+.4f} >= {MIN_BA_DELTA_FOR_ADOPTION} "
            f"and coverage = {holdout_coverage_candidate:.4f} >= {MIN_COVERAGE}."
        )
    reasons = []
    if ba_delta < MIN_BA_DELTA_FOR_ADOPTION:
        reasons.append(f"BA delta {ba_delta:+.4f} < {MIN_BA_DELTA_FOR_ADOPTION}")
    if holdout_coverage_candidate < MIN_COVERAGE:
        reasons.append(f"coverage {holdout_coverage_candidate:.4f} < {MIN_COVERAGE}")
    return f"DO NOT ADOPT: {'; '.join(reasons)}."


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_summary(
    selection_grid: pd.DataFrame,
    holdout_rows: list[dict],
    selection_winner: tuple[float, float],
    verdict: str,
    n_selection: int,
    n_holdout: int,
) -> None:
    baseline_ba = next(
        (r["covered_ba"] for r in holdout_rows if r["label"] == "baseline"), None
    )
    lines = [
        "# v132 Threshold Validation Summary",
        "",
        f"**Holdout cutoff:** {HOLDOUT_CUTOFF}",
        f"**Selection set:** {n_selection} rows",
        f"**Hold-out set:** {n_holdout} rows",
        f"**Candidate pairs evaluated:** {len(selection_grid)}",
        "",
        "## Selection Set -- Top 5 Pairs",
        "",
        "| low | high | covered_ba | coverage |",
        "|-----|------|-----------|---------|",
    ]
    for _, row in selection_grid.nlargest(5, "covered_ba").iterrows():
        lines.append(
            f"| {row['low']:.2f} | {row['high']:.2f} | "
            f"{row['covered_ba']:.4f} | {row['coverage']:.4f} |"
        )
    lines += [
        "",
        f"**Selection winner:** low={selection_winner[0]:.2f}, high={selection_winner[1]:.2f}",
        "",
        "## Hold-out Set -- Candidate Evaluation",
        "",
        "| pair | low | high | covered_ba | coverage | delta_vs_baseline |",
        "|------|-----|------|-----------|---------|-----------------|",
    ]
    for r in holdout_rows:
        delta = (r["covered_ba"] - baseline_ba) if baseline_ba is not None else float("nan")
        lines.append(
            f"| {r['label']} | {r['low']:.2f} | {r['high']:.2f} | "
            f"{r['covered_ba']:.4f} | {r['coverage']:.4f} | {delta:+.4f} |"
        )
    lines += ["", "## Verdict", "", f"> {verdict}", ""]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run v132 temporal hold-out validation. Returns exit code."""
    df = pd.read_csv(FOLD_DETAIL_PATH)
    df["test_date"] = pd.to_datetime(df["test_date"])
    df = df.sort_values("test_date").reset_index(drop=True)

    y_true_all = df["y_true"].to_numpy(dtype=int)
    path_b_raw = df["path_b_prob"].to_numpy(dtype=float)

    # Apply prequential temperature scaling to full sequence.
    # Temporal integrity is preserved: row t is calibrated using only rows 0..t-1.
    y_prob_all = apply_prequential_temperature_scaling(
        path_b_raw, y_true_all, warmup=WARMUP
    )

    # Temporal split
    cutoff = pd.Timestamp(HOLDOUT_CUTOFF)
    sel_mask = (df["test_date"] <= cutoff).to_numpy()
    hld_mask = (df["test_date"] > cutoff).to_numpy()

    y_true_sel = y_true_all[sel_mask]
    y_prob_sel = y_prob_all[sel_mask]
    y_true_hld = y_true_all[hld_mask]
    y_prob_hld = y_prob_all[hld_mask]

    n_sel, n_hld = int(sel_mask.sum()), int(hld_mask.sum())

    print(f"\n=== v132 Temporal Hold-out Threshold Validation ===")
    print(
        f"Selection set: {n_sel} rows  "
        f"({df[sel_mask]['test_date'].min().date()} -> "
        f"{df[sel_mask]['test_date'].max().date()})"
    )
    print(
        f"Hold-out set:  {n_hld} rows  "
        f"({df[hld_mask]['test_date'].min().date()} -> "
        f"{df[hld_mask]['test_date'].max().date()})"
    )

    # --- Step 1: grid sweep on selection set only ---
    print(f"\nEvaluating {len(_CANDIDATE_PAIRS)} threshold pairs on selection set...")
    selection_grid = run_threshold_grid(y_true_sel, y_prob_sel, _CANDIDATE_PAIRS)

    valid = selection_grid[selection_grid["coverage"] >= MIN_COVERAGE]
    if valid.empty:
        print("ERROR: no threshold pair achieves MIN_COVERAGE on selection set.")
        return 1

    best_sel_row = valid.loc[valid["covered_ba"].idxmax()]
    selection_winner = (float(best_sel_row["low"]), float(best_sel_row["high"]))
    print(
        f"Selection winner: low={selection_winner[0]:.2f}, high={selection_winner[1]:.2f}  "
        f"(covered_ba={best_sel_row['covered_ba']:.4f}, "
        f"coverage={best_sel_row['coverage']:.4f})"
    )

    # --- Step 2: evaluate three pairs on hold-out ---
    eval_specs = [
        ("baseline",       0.30, 0.70),
        ("a_priori_v131",  0.15, 0.70),
        ("selection_winner", selection_winner[0], selection_winner[1]),
    ]
    # Deduplicate if selection winner matches an existing pair
    seen_pairs: set[tuple[float, float]] = set()
    unique_eval = []
    for label, lo, hi in eval_specs:
        key = (round(lo, 4), round(hi, 4))
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_eval.append((label, lo, hi))

    print(f"\nHold-out evaluation:")
    holdout_rows: list[dict] = []
    for label, lo, hi in unique_eval:
        m = evaluate_thresholds(y_true_hld, y_prob_hld, low=lo, high=hi)
        holdout_rows.append({"label": label, "low": lo, "high": hi, **m})
        print(
            f"  {label:<22}  low={lo:.2f}  high={hi:.2f}  "
            f"covered_ba={m['covered_ba']:.4f}  coverage={m['coverage']:.4f}"
        )

    # --- Step 3: adoption verdict on a-priori candidate (0.15, 0.70) ---
    baseline_m = evaluate_thresholds(y_true_hld, y_prob_hld, low=0.30, high=0.70)
    apriori_m = evaluate_thresholds(y_true_hld, y_prob_hld, low=0.15, high=0.70)
    verdict = derive_verdict(
        holdout_ba_candidate=apriori_m["covered_ba"],
        holdout_ba_baseline=baseline_m["covered_ba"],
        holdout_coverage_candidate=apriori_m["coverage"],
    )
    print(f"\nVERDICT (a-priori 0.15/0.70): {verdict}")

    # --- Write outputs ---
    results_rows = []
    for _, row in selection_grid.iterrows():
        results_rows.append(
            {
                "low": row["low"],
                "high": row["high"],
                "covered_ba": row["covered_ba"],
                "coverage": row["coverage"],
                "dataset": "selection",
                "label": "",
            }
        )
    for r in holdout_rows:
        results_rows.append(
            {
                "low": r["low"],
                "high": r["high"],
                "covered_ba": r["covered_ba"],
                "coverage": r["coverage"],
                "dataset": "holdout",
                "label": r["label"],
            }
        )
    pd.DataFrame(results_rows).to_csv(RESULTS_PATH, index=False)
    _write_summary(
        selection_grid, holdout_rows, selection_winner, verdict, n_sel, n_hld
    )

    print(f"\nResults CSV: {RESULTS_PATH}")
    print(f"Summary:     {SUMMARY_PATH}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
