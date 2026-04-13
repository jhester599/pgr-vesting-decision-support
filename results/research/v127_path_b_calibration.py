"""
v127 -- Path B calibration sweep on the matched v126 fold frame.

Reads the matched Path A / Path B fold detail produced by v126 and evaluates
strictly prequential calibration candidates for Path B:

1. raw probabilities
2. prequential Platt scaling
3. prequential temperature scaling

The goal is to determine whether Path B's stronger discrimination can be kept
while materially improving calibration quality relative to the raw v126 output.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.models.calibration import compute_ece


RESULTS_DIR = PROJECT_ROOT / "results" / "research"
FOLD_DETAIL_PATH = RESULTS_DIR / "v125_portfolio_target_fold_detail.csv"
RESULTS_PATH = RESULTS_DIR / "v127_path_b_calibration_results.csv"
DETAIL_PATH = RESULTS_DIR / "v127_path_b_calibration_detail.csv"
SUMMARY_PATH = RESULTS_DIR / "v127_path_b_calibration_summary.md"
ABSTAIN_WIDTH = 0.20
MIN_HISTORY = 24


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    probability_column: str
    family: str


def _load_fold_detail(path: Path = FOLD_DETAIL_PATH) -> pd.DataFrame:
    """Load the matched v126 fold-detail frame."""
    df = pd.read_csv(path)
    df["test_date"] = pd.to_datetime(df["test_date"])
    return df.sort_values("test_date").reset_index(drop=True)


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
    """Apply temperature scaling using only prior OOS observations."""
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


def prequential_platt_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_history: int = MIN_HISTORY,
) -> np.ndarray:
    """Apply Platt scaling using only prior OOS observations."""
    y_hist = np.asarray(y_true, dtype=int)
    p_hist = _clip_probs(y_prob)
    calibrated = p_hist.copy()

    for idx in range(len(p_hist)):
        if idx < min_history or len(np.unique(y_hist[:idx])) < 2:
            continue
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
        x_train = _logit(p_hist[:idx]).reshape(-1, 1)
        model.fit(x_train, y_hist[:idx])
        calibrated[idx] = float(
            model.predict_proba(_logit(np.array([p_hist[idx]])).reshape(-1, 1))[0, 1]
        )
    return calibrated


def compute_candidate_metrics(
    detail_df: pd.DataFrame,
    *,
    probability_col: str,
    label: str,
) -> dict[str, object]:
    """Compute binary probability metrics for one candidate."""
    y_true = detail_df["y_true"].to_numpy(dtype=int)
    y_prob = detail_df[probability_col].to_numpy(dtype=float)
    covered_mask = np.abs(y_prob - 0.5) > ABSTAIN_WIDTH
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

    return {
        "model": label,
        "n_obs": int(n_obs),
        "n_covered": n_covered,
        "coverage": round(coverage, 4),
        "balanced_accuracy_all": round(ba_all, 4),
        "balanced_accuracy_covered": round(ba_cov, 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4),
        "ece": round(float(compute_ece(y_prob, y_true, n_bins=10)), 4),
        "base_rate_positive": round(float(y_true.mean()), 4),
    }


def _rank_candidates(results_df: pd.DataFrame) -> pd.DataFrame:
    """Rank Path B calibration candidates conservatively."""
    baseline = results_df.loc[results_df["model"] == "path_b_raw_v126"].iloc[0]
    eligible = results_df.copy()
    eligible["is_raw"] = eligible["model"] == "path_b_raw_v126"
    eligible["keeps_ba"] = (
        eligible["balanced_accuracy_covered"]
        >= float(baseline["balanced_accuracy_covered"]) - 0.03
    )
    eligible["improves_ece"] = eligible["ece"] <= float(baseline["ece"])
    eligible["improves_brier"] = eligible["brier_score"] <= float(baseline["brier_score"])
    eligible["improves_log_loss"] = eligible["log_loss"] <= float(baseline["log_loss"])
    eligible["best_calibrated_candidate"] = False
    eligible["selected_next"] = False

    calibrated_only = eligible[~eligible["is_raw"]].copy()
    ranked_calibrated = calibrated_only.sort_values(
        ["improves_ece", "improves_brier", "improves_log_loss", "ece", "brier_score", "log_loss"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    if not ranked_calibrated.empty:
        best_model = str(ranked_calibrated.iloc[0]["model"])
        eligible.loc[eligible["model"] == best_model, "best_calibrated_candidate"] = True
        adoption_gate = (
            bool(ranked_calibrated.iloc[0]["keeps_ba"])
            and bool(ranked_calibrated.iloc[0]["improves_ece"])
            and bool(ranked_calibrated.iloc[0]["improves_brier"])
            and bool(ranked_calibrated.iloc[0]["improves_log_loss"])
        )
        if adoption_gate:
            eligible.loc[eligible["model"] == best_model, "selected_next"] = True

    return eligible.sort_values(
        ["is_raw", "best_calibrated_candidate", "ece", "brier_score", "log_loss"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)


def _write_summary(
    ranked_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> None:
    """Write the v127 markdown summary artifact."""
    baseline = ranked_df.loc[ranked_df["model"] == "path_b_raw_v126"].iloc[0]
    path_a = ranked_df.loc[ranked_df["model"] == "path_a_matched_v126"].iloc[0]
    best_calibrated = ranked_df.loc[
        ranked_df["best_calibrated_candidate"].fillna(False).astype(bool)
    ].iloc[0]
    selected_rows = ranked_df.loc[ranked_df["selected_next"].fillna(False).astype(bool)]
    selected = selected_rows.iloc[0] if not selected_rows.empty else None

    delta_ba = float(best_calibrated["balanced_accuracy_covered"]) - float(
        baseline["balanced_accuracy_covered"]
    )
    delta_ece = float(best_calibrated["ece"]) - float(baseline["ece"])
    delta_brier = float(best_calibrated["brier_score"]) - float(baseline["brier_score"])
    delta_log_loss = float(best_calibrated["log_loss"]) - float(baseline["log_loss"])

    if selected is None:
        verdict = (
            "The best v127 calibration candidate improves Path B's reliability metrics, "
            "but no candidate clears the adoption gate because covered balanced accuracy "
            "falls too much versus raw v126. Continue calibration research before "
            "replacing the raw Path B probabilities."
        )
    elif (
        float(selected["ece"]) < float(path_a["ece"])
        and float(selected["brier_score"]) <= float(path_a["brier_score"])
    ):
        verdict = (
            "The selected v127 calibration candidate materially improves Path B and now "
            "matches or beats Path A on the primary calibration diagnostics. Path B can "
            "remain an active secondary architecture candidate."
        )
    else:
        verdict = (
            "The selected v127 calibration candidate improves Path B relative to raw v126, "
            "but it still does not close the full calibration gap to Path A. Keep Path B "
            "secondary and continue calibration research before any promotion discussion."
        )

    candidate_table = ranked_df[
        [
            "model",
            "balanced_accuracy_covered",
            "brier_score",
            "log_loss",
            "ece",
            "coverage",
            "best_calibrated_candidate",
            "selected_next",
        ]
    ].to_markdown(index=False)

    temperature_lines = []
    if "path_b_temp_temperature" in detail_df.columns:
        temp_values = detail_df["path_b_temp_temperature"].to_numpy(dtype=float)
        temperature_lines = [
            f"- average fitted temperature after warmup: `{np.nanmean(temp_values[MIN_HISTORY:]):.3f}`",
            f"- min / max fitted temperature after warmup: `{np.nanmin(temp_values[MIN_HISTORY:]):.3f}` / `{np.nanmax(temp_values[MIN_HISTORY:]):.3f}`",
        ]

    temperature_section = (
        temperature_lines if temperature_lines else ["- no temperature diagnostics available"]
    )

    lines = [
        "# v127 Path B Calibration Sweep",
        "",
        f"Run date: `{date.today().isoformat()}`",
        f"Input fold frame: `{FOLD_DETAIL_PATH.as_posix()}`",
        f"Matched OOS observations: `{len(detail_df)}`",
        f"Warmup before activating prequential calibrators: `{MIN_HISTORY}` observations",
        "",
        "## Candidate Comparison",
        "",
        candidate_table,
        "",
        "## Selected Candidate",
        "",
        f"- best calibrated candidate: `{best_calibrated['model']}`",
        f"- adoption candidate selected_next: `{selected['model'] if selected is not None else 'none'}`",
        f"- delta vs raw covered balanced accuracy: `{delta_ba:+.4f}`",
        f"- delta vs raw Brier score: `{delta_brier:+.4f}`",
        f"- delta vs raw log loss: `{delta_log_loss:+.4f}`",
        f"- delta vs raw ECE: `{delta_ece:+.4f}`",
        "",
        "## Temperature Notes",
        "",
        *temperature_section,
        "",
        "## Verdict",
        "",
        verdict,
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    detail_df = _load_fold_detail()
    y_true = detail_df["y_true"].to_numpy(dtype=int)
    path_b_raw = detail_df["path_b_prob"].to_numpy(dtype=float)

    path_b_platt = prequential_platt_calibration(y_true, path_b_raw, min_history=MIN_HISTORY)
    path_b_temp, temperatures = prequential_temperature_calibration(
        y_true,
        path_b_raw,
        min_history=MIN_HISTORY,
    )

    detail_out = detail_df.copy()
    detail_out["path_b_raw_v126"] = detail_out["path_b_prob"]
    detail_out["path_b_platt_prob"] = path_b_platt
    detail_out["path_b_temp_prob"] = path_b_temp
    detail_out["path_b_temp_temperature"] = temperatures
    detail_out.to_csv(DETAIL_PATH, index=False)

    candidate_specs = [
        CandidateSpec("path_a_matched_v126", "path_a_prob", "reference"),
        CandidateSpec("path_b_raw_v126", "path_b_raw_v126", "path_b"),
        CandidateSpec("path_b_platt_v127", "path_b_platt_prob", "path_b"),
        CandidateSpec("path_b_temp_v127", "path_b_temp_prob", "path_b"),
    ]

    rows: list[dict[str, object]] = []
    for spec in candidate_specs:
        row = compute_candidate_metrics(
            detail_out,
            probability_col=spec.probability_column,
            label=spec.name,
        )
        row["family"] = spec.family
        rows.append(row)

    results_df = pd.DataFrame(rows)
    path_b_ranked = _rank_candidates(results_df[results_df["family"] == "path_b"].copy())
    reference_df = results_df[results_df["family"] == "reference"].copy()
    reference_df["is_raw"] = False
    reference_df["keeps_ba"] = False
    reference_df["improves_ece"] = False
    reference_df["improves_brier"] = False
    reference_df["improves_log_loss"] = False
    reference_df["best_calibrated_candidate"] = False
    reference_df["selected_next"] = False
    final_results = pd.concat([reference_df, path_b_ranked], ignore_index=True)
    final_results.to_csv(RESULTS_PATH, index=False)

    _write_summary(final_results, detail_out)
    print(f"Results CSV written to {RESULTS_PATH}")
    print(f"Detail CSV written to {DETAIL_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
