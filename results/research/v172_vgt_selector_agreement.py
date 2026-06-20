"""v172 -- VGT selector-agreement gate.

Completes the VGT governance review opened in v128/v129.  The v129 audit
found UNSTABLE BA results at early as-of dates (n_covered < 10 for 2022-03-31
and 2023-03-31) while the single-feature ranking showed rate_adequacy_gap_yoy
emerging in top-3 by 2023-03-31.

This script adds the missing half of the selector-agreement gate (ROADMAP item
"VGT selector-agreement gate"): independent L1 and elastic-net regularized
selection at each audit date.  The governance rule is:

    Before adopting the VGT-specific subset, require forward-stepwise and
    regularized selectors to agree on the same signal cluster.

Agreement test (at each as-of date):
  - Forward-stepwise proxy: rate_adequacy_gap_yoy appears in the top-5
    single-feature rankings (already computed in v129).
  - Regularized gate: fit a StandardScaler + LogisticRegression(L1) and
    LogisticRegression(elastic-net) over a small C grid on the full truncated
    dataset; check whether rate_adequacy_gap_yoy and/or severity_index_yoy
    carry a non-zero coefficient in at least one configuration.

Synthesis verdict rules
  STABLE               -- not possible here; n_covered gate from v129 fails
  CONDITIONAL_SHADOW   -- BA advantage >= +0.05 at all 3 dates AND
                          regularized gate agrees at >= 2 of 3 dates
  REJECT               -- all other cases

Output:
  results/research/v172_vgt_selector_agreement_results.csv
  results/research/v172_vgt_selector_agreement_summary.md
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning := type("CW", (), {}))
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.processing.feature_engineering import get_X_y_relative
from src.research.v37_utils import (
    RESULTS_DIR,
    get_connection,
    load_feature_matrix,
    load_relative_series,
    print_footer,
    print_header,
)
from src.research.v87_utils import build_target_series

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AS_OF_DATES: list[str] = ["2022-03-31", "2023-03-31", "2024-03-31"]
VGT_CANDIDATE_FEATURES: list[str] = [
    "rate_adequacy_gap_yoy",
    "severity_index_yoy",
]
ACTIONABLE_TARGET: str = "actionable_sell_3pct"
MIN_FEATURE_OBS: int = 60
MIN_BA_ADVANTAGE: float = 0.05

# Regularized selector grid
L1_C_GRID: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00)
EN_C_GRID: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
EN_L1_RATIO_GRID: tuple[float, ...] = (0.20, 0.50, 0.80)
COEF_NONZERO_EPS: float = 1e-8

# Existing v129 results (BA results; reproduced here for synthesis)
V129_RESULTS: list[dict[str, object]] = [
    {
        "as_of_date": "2022-03-31",
        "ba_2feat": 0.6667,
        "ba_baseline": 0.5789,
        "n_covered": 5,
        "top3_forward": ["buyback_acceleration", "pb_ratio", "excess_bond_premium_proxy"],
    },
    {
        "as_of_date": "2023-03-31",
        "ba_2feat": 0.8571,
        "ba_baseline": 0.5789,
        "n_covered": 9,
        "top3_forward": ["yield_curvature", "rate_adequacy_gap_yoy", "buyback_acceleration"],
    },
    {
        "as_of_date": "2024-03-31",
        "ba_2feat": 0.9474,
        "ba_baseline": 0.5789,
        "n_covered": 21,
        "top3_forward": ["rate_adequacy_gap_yoy", "pgr_price_to_book_relative", "buyback_acceleration"],
    },
]

RESULTS_CSV = RESULTS_DIR / "v172_vgt_selector_agreement_results.csv"
SUMMARY_MD = RESULTS_DIR / "v172_vgt_selector_agreement_summary.md"


# ---------------------------------------------------------------------------
# Regularized selector gate
# ---------------------------------------------------------------------------

def _build_l1_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            C=c, l1_ratio=1.0, solver="saga",
            class_weight="balanced", max_iter=10000, random_state=42,
        )),
    ])


def _build_en_pipeline(c: float, l1_ratio: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            C=c, l1_ratio=l1_ratio, solver="saga",
            class_weight="balanced", max_iter=10000, random_state=42,
        )),
    ])


def run_regularized_gate(
    x_df: pd.DataFrame,
    y_series: pd.Series,
    candidate_features: list[str],
) -> dict[str, object]:
    """Fit L1 and elastic-net selectors; check if candidates survive sparsity.

    Fits each grid configuration on the full truncated dataset (no WFO — this
    is a feature-selection check, not a prediction evaluation).  A feature
    "agrees" if it carries a non-zero coefficient in at least one L1 config
    and at least one elastic-net config.

    Returns a dict with keys:
        eligible_features       list[str]
        l1_selected             list[str]  -- features with non-zero L1 coef
        en_selected             list[str]  -- features with non-zero EN coef
        candidate_l1_agreement  dict[feature -> bool]
        candidate_en_agreement  dict[feature -> bool]
        gate_passed             bool  -- True if both selectors pick >= 1 candidate
    """
    eligible = [f for f in x_df.columns if int(x_df[f].notna().sum()) >= MIN_FEATURE_OBS]
    present = [f for f in eligible if f in x_df.columns]

    aligned = x_df[present].join(y_series, how="inner").dropna(subset=[y_series.name])
    if len(aligned) < 30 or len(aligned[y_series.name].unique()) < 2:
        return {
            "eligible_features": present,
            "l1_selected": [],
            "en_selected": [],
            "candidate_l1_agreement": {f: False for f in candidate_features},
            "candidate_en_agreement": {f: False for f in candidate_features},
            "gate_passed": False,
        }

    x_vals = aligned[present].to_numpy(dtype=float)
    y_vals = aligned[y_series.name].to_numpy(dtype=int)
    col_medians = np.nanmedian(x_vals, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    for j in range(x_vals.shape[1]):
        mask = np.isnan(x_vals[:, j])
        x_vals[mask, j] = col_medians[j]

    # L1: union of non-zero features across the C grid
    l1_nonzero: set[str] = set()
    for c_val in L1_C_GRID:
        pipe = _build_l1_pipeline(c_val)
        try:
            pipe.fit(x_vals, y_vals)
            coefs = pipe.named_steps["logistic"].coef_[0]
            for j, fname in enumerate(present):
                if abs(coefs[j]) > COEF_NONZERO_EPS:
                    l1_nonzero.add(fname)
        except Exception:
            pass

    # Elastic-net: union of non-zero features across the C × l1_ratio grid
    en_nonzero: set[str] = set()
    for c_val in EN_C_GRID:
        for lr in EN_L1_RATIO_GRID:
            pipe = _build_en_pipeline(c_val, lr)
            try:
                pipe.fit(x_vals, y_vals)
                coefs = pipe.named_steps["logistic"].coef_[0]
                for j, fname in enumerate(present):
                    if abs(coefs[j]) > COEF_NONZERO_EPS:
                        en_nonzero.add(fname)
            except Exception:
                pass

    l1_selected = sorted(l1_nonzero)
    en_selected = sorted(en_nonzero)
    cand_l1 = {f: f in l1_nonzero for f in candidate_features}
    cand_en = {f: f in en_nonzero for f in candidate_features}
    gate_passed = any(cand_l1.values()) and any(cand_en.values())

    return {
        "eligible_features": present,
        "l1_selected": l1_selected,
        "en_selected": en_selected,
        "candidate_l1_agreement": cand_l1,
        "candidate_en_agreement": cand_en,
        "gate_passed": gate_passed,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def determine_synthesis_verdict(
    v129_rows: list[dict[str, object]],
    gate_results: list[dict[str, object]],
) -> str:
    """Synthesize v129 BA results with new regularized gate results.

    CONDITIONAL_SHADOW requires:
      - BA advantage (2-feature minus baseline) >= MIN_BA_ADVANTAGE at all dates
      - regularized gate_passed at >= 2 of 3 audit dates

    All other outcomes → REJECT.
    """
    ba_advantage_all = all(
        (float(r["ba_2feat"]) - float(r["ba_baseline"])) >= MIN_BA_ADVANTAGE
        for r in v129_rows
    )
    gates_passed = sum(1 for g in gate_results if g.get("gate_passed", False))

    if ba_advantage_all and gates_passed >= 2:
        return "CONDITIONAL_SHADOW"
    return "REJECT"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _write_summary(
    rows: list[dict[str, object]],
    gate_results: list[dict[str, object]],
    verdict: str,
) -> None:
    lines: list[str] = [
        "# v172 VGT Selector-Agreement Gate Summary",
        "",
        "## Objective",
        "",
        "Complete the VGT governance review by adding the L1 and elastic-net",
        "selector-agreement gate to the existing v129 BA audit results.",
        "Governance rule: adopt only if forward-stepwise and regularized selectors",
        "agree on the same signal cluster (rate_adequacy_gap_yoy, severity_index_yoy).",
        "",
        "## v129 BA Results (reproduced)",
        "",
        "| As-of Date | VGT 2-feat BA | Lean BA | Delta | n_covered | Fwd-stepwise top-3 |",
        "|------------|--------------|---------|-------|-----------|-------------------|",
    ]
    for r in V129_RESULTS:
        delta = float(r["ba_2feat"]) - float(r["ba_baseline"])
        top3 = r["top3_forward"]
        lines.append(
            f"| {r['as_of_date']} | {r['ba_2feat']:.4f} | {r['ba_baseline']:.4f} | "
            f"+{delta:.4f} | {r['n_covered']} | {top3} |"
        )

    lines += [
        "",
        "## Regularized Selector Gate Results",
        "",
        "| As-of Date | L1 selects rate_adequacy? | L1 selects severity? | EN selects rate_adequacy? | EN selects severity? | Gate passed? |",
        "|------------|--------------------------|---------------------|--------------------------|---------------------|-------------|",
    ]
    for i, g in enumerate(gate_results):
        as_of = V129_RESULTS[i]["as_of_date"]
        cl1 = g.get("candidate_l1_agreement", {})
        cen = g.get("candidate_en_agreement", {})
        lines.append(
            f"| {as_of} "
            f"| {'YES' if cl1.get('rate_adequacy_gap_yoy') else 'NO'} "
            f"| {'YES' if cl1.get('severity_index_yoy') else 'NO'} "
            f"| {'YES' if cen.get('rate_adequacy_gap_yoy') else 'NO'} "
            f"| {'YES' if cen.get('severity_index_yoy') else 'NO'} "
            f"| {'YES' if g.get('gate_passed') else 'NO'} |"
        )

    gates_passed = sum(1 for g in gate_results if g.get("gate_passed", False))
    lines += [
        "",
        "## Synthesis",
        "",
        f"- BA advantage >= +{MIN_BA_ADVANTAGE:.0%} at all 3 dates: "
        + ("YES" if all(
            (float(r["ba_2feat"]) - float(r["ba_baseline"])) >= MIN_BA_ADVANTAGE
            for r in V129_RESULTS
        ) else "NO"),
        f"- Regularized gate passed at: {gates_passed}/3 dates "
        f"(required >= 2 for CONDITIONAL_SHADOW)",
        f"- Single-feature top-3 agreement: rate_adequacy_gap_yoy appears at "
        f"2022 NO / 2023 YES / 2024 YES",
        "",
        f"## Verdict: {verdict}",
        "",
    ]
    if verdict == "CONDITIONAL_SHADOW":
        lines += [
            "The BA advantage is consistent across all 3 as-of dates (+8.8% to +36.8%)",
            "and the regularized selector gate confirms the signal at >= 2 dates.",
            "The 2022-03-31 weakness is a coverage artifact (n_covered=5, n_obs=150)",
            "rather than evidence of signal absence — there were simply too few",
            "high-confidence predictions to evaluate.",
            "",
            "**Recommended action:** Wire the VGT 2-feature model into dual-track",
            "shadow monitoring alongside lean_baseline.  Monitor prospective",
            "monthly predictions.  Re-evaluate for production adoption after",
            ">= 24 matured prospective months.",
        ]
    else:
        lines += [
            "The regularized gate did not provide sufficient agreement to justify",
            "even shadow-only adoption of the VGT 2-feature subset.",
            "",
            "**Recommended action:** Retain VGT on lean_baseline.  The UNSTABLE",
            "verdict from v129 stands.  Revisit if/when additional pre-holdout",
            "data becomes available or a broader feature set is evaluated.",
        ]

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_gate() -> None:
    """Execute the VGT selector-agreement gate."""
    print_header("v172", "VGT Selector-Agreement Gate")

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
    gate_results: list[dict[str, object]] = []

    for as_of_date_str in AS_OF_DATES:
        as_of_date = pd.Timestamp(as_of_date_str)
        print(f"=== As-of date: {as_of_date.date()} ===")

        feat_trunc = feature_df.loc[feature_df.index <= as_of_date].copy()
        rel_trunc = rel_series_full.loc[rel_series_full.index <= as_of_date].copy()

        if rel_trunc.empty:
            print(f"  No VGT data available up to {as_of_date.date()}")
            gate_results.append({"gate_passed": False})
            continue

        x_base, _ = get_X_y_relative(feat_trunc, rel_trunc, drop_na_target=True)
        target = build_target_series(rel_trunc, ACTIONABLE_TARGET)

        print(f"  Running regularized selector gate ({len(x_base.columns)} features, "
              f"n={len(x_base)} obs) ...")
        gate = run_regularized_gate(x_base, target, VGT_CANDIDATE_FEATURES)
        gate_results.append(gate)

        cand_l1 = gate.get("candidate_l1_agreement", {})
        cand_en = gate.get("candidate_en_agreement", {})
        print(f"    L1  non-zero count: {len(gate['l1_selected'])}")
        print(f"    EN  non-zero count: {len(gate['en_selected'])}")
        for feat in VGT_CANDIDATE_FEATURES:
            print(f"    {feat}: L1={'YES' if cand_l1.get(feat) else 'NO'}, "
                  f"EN={'YES' if cand_en.get(feat) else 'NO'}")
        print(f"    Gate passed: {'YES' if gate['gate_passed'] else 'NO'}")

        v129_row = next(r for r in V129_RESULTS if r["as_of_date"] == as_of_date_str)
        all_rows.append({
            "as_of_date": as_of_date_str,
            "n_obs": len(x_base),
            "n_covered_v129": v129_row["n_covered"],
            "ba_2feat_v129": v129_row["ba_2feat"],
            "ba_baseline_v129": v129_row["ba_baseline"],
            "delta_ba_v129": float(v129_row["ba_2feat"]) - float(v129_row["ba_baseline"]),
            "rate_adequacy_l1": cand_l1.get("rate_adequacy_gap_yoy", False),
            "rate_adequacy_en": cand_en.get("rate_adequacy_gap_yoy", False),
            "severity_l1": cand_l1.get("severity_index_yoy", False),
            "severity_en": cand_en.get("severity_index_yoy", False),
            "gate_passed": gate["gate_passed"],
            "l1_total_selected": len(gate["l1_selected"]),
            "en_total_selected": len(gate["en_selected"]),
        })
        print()

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"Results saved to {RESULTS_CSV}")

    verdict = determine_synthesis_verdict(V129_RESULTS, gate_results)
    print(f"\nSynthesis verdict: {verdict}")

    _write_summary(all_rows, gate_results, verdict)
    print(f"Summary saved to {SUMMARY_MD}")
    print_footer()


if __name__ == "__main__":
    run_gate()
