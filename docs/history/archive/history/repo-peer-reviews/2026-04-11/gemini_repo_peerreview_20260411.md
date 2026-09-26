<proposed_plan>
# PGR Vesting Decision Support: v97-v101 Implementation Plan
# Goal: Transition from Regression-only to Hybrid Classification-Gating

## Phase 1: v97 - Shadow Classification Integration (Production)
**Objective:** Integrate the winning classifier from the v87-v96 research cycle into the monthly production script in shadow mode. No changes to the primary recommendation logic.

* **Target Files:**
    * `src/models/wfo_engine.py`
    * `scripts/monthly_decision.py`
    * `src/reporting/snapshot_summary.py`
* **Tasks:**
    1.  Update `wfo_engine.py` (or create `classification_wfo.py`) to support `predict_proba` using strict expanding-window WFO.
    2.  Modify `_generate_signals()` in `monthly_decision.py` to instantiate the best v96 classifier (e.g., LogisticRegression with L1 penalty or GBT Classifier).
    3.  Compute `prob_actionable_sell` for each benchmark.
    4.  Append `prob_actionable_sell` as a new column to the `signals.csv` output.
* **Validation:** Run `python scripts/monthly_decision.py --dry-run`. Ensure `signals.csv` outputs successfully with the new column and values between 0.0 and 1.0. No data leakage allowed.

## Phase 2: v98 - Hybrid Policy Gating Backtest (Research)
**Objective:** Determine the optimal probability threshold required to veto a regression-based SELL signal.

* **Target Files:**
    * `scripts/v98_hybrid_policy_backtest.py` (NEW)
    * `src/research/policy_metrics.py`
* **Tasks:**
    1.  Create `v98_hybrid_policy_backtest.py`. Load historical WFO predictions for both `v38` (regression) and `v96` (classification).
    2.  Simulate a hybrid policy: `IF reg_signal == SELL AND class_prob > THRESHOLD THEN SELL ELSE HOLD`.
    3.  Sweep `THRESHOLD` from `0.50` to `0.75` in increments of `0.05`.
    4.  Output `results/research/v98_hybrid_policy_sweep.csv` tracking Win Rate, False Positive Rate (tax events triggered unnecessarily), and OOS Return.
* **Validation:** Confirm that increasing the threshold reduces turnover/false positives while preserving true positive sell events.

## Phase 3: v99 - UI and Reporting Surface Updates (Production)
**Objective:** Expose the shadow classification probability to the user cleanly in the monthly artifacts.

* **Target Files:**
    * `src/reporting/decision_rendering.py`
    * `src/reporting/dashboard_snapshot.py`
    * `dashboard/app.py` (if applicable)
* **Tasks:**
    1.  Update `build_vest_decision_lines()` to include a "Classification Confidence" sub-bullet.
    2.  Update the HTML/Markdown generation so that if `prob_actionable_sell` > 0.60, it highlights the risk visually.
    3.  Update `monthly_summary.json` schema to include `hybrid_classification_confidence`.
* **Validation:** Generate a dry-run report. Verify the markdown table and summary JSON contain the new confidence metrics. Ensure it does not look excessively noisy.

## Phase 4: v100 - Artifact and Governance Documentation (Docs)
**Objective:** Formalize the new hybrid architecture in the project docs.

* **Target Files:**
    * `docs/architecture.md`
    * `docs/decision-output-guide.md`
    * `docs/model-governance.md`
* **Tasks:**
    1.  Update `architecture.md` to document the dual-pipeline (Magnitude Regression + Probability Classifier).
    2.  Update `decision-output-guide.md` to define `prob_actionable_sell`.
    3.  Update `model-governance.md` to outline how classification drift will be monitored (e.g., tracking Brier Score or Log Loss alongside OOS R²).
* **Validation:** Review markdown files for clarity. Ensure the distinction between the "sizing" model and the "gating" model is clearly explained.

## Phase 5: v101 - Promote Hybrid Policy to Master (Production)
**Objective:** Flip the switch. Make the hybrid classifier the primary driver of the monthly recommendation.

* **Target Files:**
    * `scripts/monthly_decision.py`
    * `src/portfolio/redeploy_buckets.py`
* **Tasks:**
    1.  Modify the core policy logic in `monthly_decision.py`. Set the recommendation to SELL *only* if the consensus classification probability exceeds the threshold identified in `v98` (e.g., 0.60).
    2.  If the gate is passed, route the regression magnitude into `recommend_redeploy_buckets()` to size the reallocation.
    3.  If the gate fails, force a `HOLD` recommendation, explicitly logging: "Vetoed by classification gate (Confidence < 60%)".
* **Validation:** Run a backdated dry-run to a known volatile month. Verify the logic accurately suppresses weak regression signals. Commit and merge to `main`.
</proposed_plan>