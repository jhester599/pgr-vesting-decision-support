"""
v130 -- Path B temperature scaling revised adoption analysis.

Re-evaluates temperature scaling Path B against Path A matched (the correct
baseline) using three adoption criteria:

  A: BA_covered(temp_B) - BA_covered(path_A) >= 0.03
  B: Brier(temp_B) <= Brier(path_A) + 0.02
  C: ECE(temp_B) <= ECE(path_A) * 1.5

All three criteria must pass for full adoption.  If A and B pass but not C,
a CONDITIONAL ADOPT verdict is issued.  Otherwise DO NOT ADOPT.

The v127 adoption gate compared temperature scaling against raw Path B (not
Path A matched) and rejected it because BA fell by 0.0725 versus raw Path B.
That criterion was inverted -- temperature scaling trades BA against raw Path B
for materially better calibration.  The correct question is whether temp-B
beats Path A on a risk-adjusted basis.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.models.calibration import compute_ece

RESULTS_DIR = PROJECT_ROOT / "results" / "research"
FOLD_DETAIL_PATH = RESULTS_DIR / "v125_portfolio_target_fold_detail.csv"
RESULTS_PATH = RESULTS_DIR / "v130_path_b_temp_scaling_results.csv"
SUMMARY_PATH = RESULTS_DIR / "v130_path_b_temp_scaling_summary.md"

MIN_HISTORY: int = 24
ABSTAIN_LOW: float = 0.30
ABSTAIN_HIGH: float = 0.70
N_ECE_BINS: int = 10

# Adoption criterion thresholds
CRITERION_A_MIN_BA_DELTA: float = 0.03
CRITERION_B_MAX_BRIER_EXCESS: float = 0.02
CRITERION_C_MAX_ECE_RATIO: float = 1.50


# ---------------------------------------------------------------------------
# Pure utility functions (copied from v127 for reproducibility)
# ---------------------------------------------------------------------------


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    """Clip probabilities away from 0 and 1 for stable log transforms."""
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(probs: np.ndarray) -> np.ndarray:
    """Return the logit transform of probabilities."""
    clipped = _clip_probs(probs)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the logistic sigmoid of arbitrary values."""
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _apply_temperature(prob: float, temperature: float) -> float:
    """Apply temperature scaling to one binary probability."""
    temp = max(float(temperature), 1e-6)
    value = float(_sigmoid(np.array([_logit(np.array([prob]))[0] / temp]))[0])
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _fit_temperature_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidate_temperatures: np.ndarray | None = None,
) -> float:
    """Fit temperature on historical OOS points via simple log-loss grid search."""
    if candidate_temperatures is None:
        candidate_temperatures = np.concatenate(
            [
                np.linspace(0.50, 0.95, 10),
                np.linspace(1.0, 3.0, 41),
            ]
        )
    y_hist = np.asarray(y_true, dtype=int)
    p_hist = _clip_probs(y_prob)
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in candidate_temperatures:
        scaled = _sigmoid(_logit(p_hist) / float(temperature))
        loss = float(log_loss(y_hist, scaled, labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


def prequential_temperature_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_history: int = MIN_HISTORY,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply temperature scaling using only prior OOS observations (prequential)."""
    y_hist = np.asarray(y_true, dtype=int)
    p_hist = _clip_probs(y_prob)
    calibrated = p_hist.copy()
    temperatures = np.ones(len(p_hist), dtype=float)
    for idx in range(len(p_hist)):
        if idx < min_history or len(np.unique(y_hist[:idx])) < 2:
            continue
        temperature = _fit_temperature_grid(y_hist[:idx], p_hist[:idx])
        temperatures[idx] = temperature
        calibrated[idx] = _apply_temperature(float(p_hist[idx]), temperature)
    return calibrated, temperatures


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
    *,
    abstain_low: float = ABSTAIN_LOW,
    abstain_high: float = ABSTAIN_HIGH,
    n_bins: int = N_ECE_BINS,
) -> dict[str, object]:
    """Compute coverage-aware binary probability metrics for one model."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _clip_probs(y_prob)
    covered_mask = (y_prob <= abstain_low) | (y_prob >= abstain_high)
    n_obs = len(y_true)
    n_covered = int(covered_mask.sum())
    coverage = n_covered / max(n_obs, 1)

    pred_all = (y_prob >= 0.5).astype(int)
    ba_all = float(balanced_accuracy_score(y_true, pred_all))
    if n_covered > 0:
        pred_cov = (y_prob[covered_mask] >= 0.5).astype(int)
        ba_cov = float(balanced_accuracy_score(y_true[covered_mask], pred_cov))
    else:
        ba_cov = float("nan")

    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    ece = float(compute_ece(y_prob, y_true, n_bins=n_bins))

    return {
        "model": label,
        "n_obs": n_obs,
        "n_covered": n_covered,
        "coverage": round(coverage, 4),
        "balanced_accuracy_all": round(ba_all, 4),
        "balanced_accuracy_covered": round(ba_cov, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "ece_10": round(ece, 4),
    }


# ---------------------------------------------------------------------------
# Adoption gate (pure function, importable for tests)
# ---------------------------------------------------------------------------


def evaluate_adoption_criteria(
    *,
    ba_temp: float,
    ba_path_a: float,
    brier_temp: float,
    brier_path_a: float,
    ece_temp: float,
    ece_path_a: float,
) -> dict[str, bool | float]:
    """Evaluate the three adoption criteria for temperature-scaled Path B vs Path A.

    Parameters
    ----------
    ba_temp:
        Covered balanced accuracy of temperature-scaled Path B.
    ba_path_a:
        Covered balanced accuracy of Path A matched.
    brier_temp:
        Brier score of temperature-scaled Path B.
    brier_path_a:
        Brier score of Path A matched.
    ece_temp:
        ECE of temperature-scaled Path B.
    ece_path_a:
        ECE of Path A matched.

    Returns
    -------
    dict with keys: criterion_a, criterion_b, criterion_c, adopt, ba_delta,
    brier_excess, ece_ratio.
    """
    ba_delta = float(ba_temp) - float(ba_path_a)
    brier_excess = float(brier_temp) - float(brier_path_a)
    ece_ratio = float(ece_temp) / max(float(ece_path_a), 1e-9)

    criterion_a = ba_delta >= CRITERION_A_MIN_BA_DELTA
    criterion_b = brier_excess <= CRITERION_B_MAX_BRIER_EXCESS
    criterion_c = ece_ratio <= CRITERION_C_MAX_ECE_RATIO

    adopt = criterion_a and criterion_b and criterion_c

    return {
        "criterion_a": criterion_a,
        "criterion_b": criterion_b,
        "criterion_c": criterion_c,
        "adopt": adopt,
        "ba_delta": round(ba_delta, 6),
        "brier_excess": round(brier_excess, 6),
        "ece_ratio": round(ece_ratio, 6),
    }


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def _derive_verdict(
    gate: dict[str, bool | float],
) -> str:
    """Derive the textual adoption verdict from gate evaluation."""
    if gate["adopt"]:
        return (
            "ADOPT temperature scaling for Path B shadow signal. Replace raw Path B "
            "probability with temperature-scaled probability in the investable-pool "
            "aggregate computation."
        )
    if gate["criterion_a"] and gate["criterion_b"] and not gate["criterion_c"]:
        return (
            "CONDITIONAL ADOPT — acceptable for shadow reporting but calibration "
            "warrants further monitoring."
        )
    return (
        "DO NOT ADOPT. Path A matched remains the better risk-adjusted option."
    )


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------


def _write_summary(
    results_df: pd.DataFrame,
    gate: dict[str, bool | float],
    verdict: str,
    n_obs: int,
) -> None:
    """Write the v130 markdown summary artifact."""
    table = results_df[
        [
            "model",
            "balanced_accuracy_covered",
            "brier_score",
            "log_loss",
            "ece_10",
            "coverage",
        ]
    ].to_markdown(index=False)

    lines = [
        "# v130 Path B Temperature Scaling Revised Adoption Analysis",
        "",
        f"Run date: `{date.today().isoformat()}`",
        f"Input fold frame: `{FOLD_DETAIL_PATH.as_posix()}`",
        f"Matched OOS observations: `{n_obs}`",
        f"Warmup (prequential calibration): `{MIN_HISTORY}` observations",
        f"Coverage abstain window: `[{ABSTAIN_LOW}, {ABSTAIN_HIGH}]`",
        "",
        "## Background",
        "",
        "v127 rejected temperature scaling because it compared against raw Path B.",
        "The correct baseline is **Path A matched** (the incumbent signal).",
        "This script re-evaluates with the corrected adoption criterion.",
        "",
        "## Candidate Comparison",
        "",
        table,
        "",
        "## Adoption Criteria (vs Path A matched)",
        "",
        f"| Criterion | Threshold | Observed | Pass |",
        f"|---|---|---|---|",
        f"| A: BA delta | >= {CRITERION_A_MIN_BA_DELTA:.2f} "
        f"| {gate['ba_delta']:+.4f} "
        f"| {'YES' if gate['criterion_a'] else 'NO'} |",
        f"| B: Brier excess | <= {CRITERION_B_MAX_BRIER_EXCESS:.2f} "
        f"| {gate['brier_excess']:+.4f} "
        f"| {'YES' if gate['criterion_b'] else 'NO'} |",
        f"| C: ECE ratio | <= {CRITERION_C_MAX_ECE_RATIO:.1f}x "
        f"| {gate['ece_ratio']:.4f}x "
        f"| {'YES' if gate['criterion_c'] else 'NO'} |",
        "",
        "## Verdict",
        "",
        verdict,
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the v130 adoption analysis."""
    df = pd.read_csv(FOLD_DETAIL_PATH)
    df["test_date"] = pd.to_datetime(df["test_date"])
    df = df.sort_values("test_date").reset_index(drop=True)

    y_true = df["y_true"].to_numpy(dtype=int)
    path_a_raw = df["path_a_prob"].to_numpy(dtype=float)
    path_b_raw = df["path_b_prob"].to_numpy(dtype=float)

    path_b_temp, temperatures = prequential_temperature_calibration(
        y_true,
        path_b_raw,
        min_history=MIN_HISTORY,
    )

    path_a_metrics = compute_metrics(y_true, path_a_raw, label="path_a_matched")
    path_b_raw_metrics = compute_metrics(y_true, path_b_raw, label="path_b_raw")
    path_b_temp_metrics = compute_metrics(y_true, path_b_temp, label="path_b_temp_scaled")

    results_rows = [path_a_metrics, path_b_raw_metrics, path_b_temp_metrics]
    results_df = pd.DataFrame(results_rows)

    # Evaluate adoption gate
    gate = evaluate_adoption_criteria(
        ba_temp=float(path_b_temp_metrics["balanced_accuracy_covered"]),
        ba_path_a=float(path_a_metrics["balanced_accuracy_covered"]),
        brier_temp=float(path_b_temp_metrics["brier_score"]),
        brier_path_a=float(path_a_metrics["brier_score"]),
        ece_temp=float(path_b_temp_metrics["ece_10"]),
        ece_path_a=float(path_a_metrics["ece_10"]),
    )

    verdict = _derive_verdict(gate)

    # Tag results CSV with adopt flag
    results_df["adopt"] = results_df["model"].map(
        lambda m: bool(gate["adopt"]) if m == "path_b_temp_scaled" else None
    )

    results_df.to_csv(RESULTS_PATH, index=False)
    _write_summary(results_df, gate, verdict, n_obs=len(df))

    # Print summary to stdout
    print("\n=== v130 Path B Temperature Scaling Adoption Analysis ===\n")
    print(
        results_df[
            ["model", "balanced_accuracy_covered", "brier_score", "log_loss", "ece_10", "coverage"]
        ].to_string(index=False)
    )
    print(f"\nCriterion A (BA delta >= {CRITERION_A_MIN_BA_DELTA}): "
          f"{gate['ba_delta']:+.4f} -> {'PASS' if gate['criterion_a'] else 'FAIL'}")
    print(f"Criterion B (Brier excess <= {CRITERION_B_MAX_BRIER_EXCESS}): "
          f"{gate['brier_excess']:+.4f} -> {'PASS' if gate['criterion_b'] else 'FAIL'}")
    print(f"Criterion C (ECE ratio <= {CRITERION_C_MAX_ECE_RATIO}x): "
          f"{gate['ece_ratio']:.4f}x -> {'PASS' if gate['criterion_c'] else 'FAIL'}")
    print(f"\nVERDICT: {verdict}\n")
    print(f"Results CSV: {RESULTS_PATH}")
    print(f"Summary:     {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
