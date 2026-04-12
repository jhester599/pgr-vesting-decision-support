# Deep Research Review of the pgr-vesting-decision-support Repository

## Executive Summary

The repository is already in a strong “production-consumable” state: it runs on scheduled entity["company","GitHub","code hosting platform"] Actions, persists an artifact bundle per month, and keeps a conservative decision posture (defaulting to diversification/tax discipline when model quality is weak). The live production path is a quality-weighted, shrinkage-stabilized regression consensus, while the v87–v96 program built a credible classification/hybrid research track and shipped a shadow-only classifier confidence block into monthly surfaces. fileciteturn3file0L1-L1 fileciteturn4file0L1-L1 fileciteturn48file0L1-L1 fileciteturn42file0L1-L1

The “why not promoted yet” is also correctly articulated in repo artifacts: the best classification-led policy variant improved a policy utility metric but diverged too often from the regression baseline, failing a conservative promotion gate that explicitly requires agreement and stability. That is a useful outcome: it narrows the next cycle to *alignment and decision-layer design* rather than more model-family sprawl. fileciteturn36file0L1-L1 fileciteturn68file0L1-L1 fileciteturn67file0L1-L1

Most important conclusions (implementation-oriented):

- The next “highest ROI” work is **not** another broad model sweep; it is **decision-layer alignment**: turn classification into a *conservative overlay / gate* that reduces large mistakes and improves “hold vs sell” clarity *without* causing large, frequent disagreements with the production regression baseline. fileciteturn67file0L1-L1 fileciteturn68file0L1-L1
- The repo already has the right scaffolding (artifact bundle, monthly_summary.json contract, drift trigger, CI smoke tests). The cleanest next promotion path is to add **one new shadow-only artifact + summary payload fields** for classification-gate replay, and let it run for months before any production override is allowed. fileciteturn48file0L1-L1 fileciteturn56file0L1-L1 fileciteturn54file0L1-L1
- A small but real methodology risk exists for **backdated “--as-of” runs**: the current alignment utilities can inadvertently include targets computed with future prices if the database contains later data. Fixing this is low-complexity and strengthens the repo’s strict “no leakage” posture. fileciteturn71file0L1-L1 fileciteturn70file0L1-L1
- The most user-relevant UX gain is to make the monthly artifacts explicitly answer: “**For the next vest tranche, sell X% / hold Y%**” and “**Is this month actionable?**” with a crisp explanation of what moved since last month, plus a compact “disagreement panel” (regression vs cross-check vs classifier shadow). This mostly requires wiring and rendering, not new modeling. fileciteturn69file0L1-L1 fileciteturn55file0L1-L1 fileciteturn57file0L1-L1

## Detailed Findings

**Modeling**

The production modeling stack is disciplined and conservative: it uses rolling walk-forward optimization with explicit gaps to avoid overlap leakage for forward return targets, and it explicitly prohibits K-Fold validation in favor of time-series split schemes. This is correctly enforced both in design docs and in the core WFO engine implementation (TimeSeriesSplit with gap = horizon + purge buffer, plus fold-local imputation). fileciteturn73file0L1-L1 fileciteturn63file0L1-L1 fileciteturn5file0L1-L1

The live “consensus layer” is intentionally not a fragile ensemble-of-ensembles: the repo promotes a quality-weighted cross-benchmark consensus (with shrinkage toward equal weights) and preserves an equal-weight shadow for diagnostics/cross-checks. This is a good fit for the stated constraint “avoid complexity unless it materially improves decision quality,” and it maps cleanly to artifact workflows. fileciteturn63file0L1-L1 fileciteturn62file0L1-L1 fileciteturn9file0L1-L1

The v87–v96 classification/hybrid program is well-structured and already answers several key “what to avoid” questions: the actionable-sell target became the primary label, the lean baseline feature set remained best, pooled/panel linear structure helped, and nonlinear sweeps did not materially beat pooled linear baselines. That is strong evidence *against* spending the next cycle on high-variance model families or broad feature expansion. fileciteturn67file0L1-L1 fileciteturn35file0L1-L1 fileciteturn31file0L1-L1 fileciteturn33file0L1-L1 fileciteturn32file0L1-L1

The core reason classification is not yet promotable is also explicit and measurable: the best policy candidate by mean policy return diverged too much from the regression baseline, failing the promotion gate that requires high agreement (≥0.70) and controlled recommendation instability. The gate logic is implemented in code, which is good governance (it prevents hand-wavy promotion decisions). fileciteturn68file0L1-L1 fileciteturn36file0L1-L1

A subtle but important methodology edge case: in backdated execution (using `--as-of`), the target series loaded from the full database can include forward returns that would not have been knowable as-of that earlier date, because forward returns are computed wherever future price data exists. The repo’s feature matrix architecture is careful about look-ahead (lags for FRED and EDGAR, and NaN targets near the end of history), but the alignment helper (`get_X_y_relative`) will happily join to any non-NaN target rows that exist in the DB. For strict “point-in-time as-of simulation,” the target series must be explicitly truncated. fileciteturn71file0L1-L1 fileciteturn70file0L1-L1

**Decision layer**

The live recommendation mode is driven primarily by aggregate regression diagnostics (OOS R², IC, hit rate, CPCV stability), with a conservative downgrade path to MONITORING-ONLY or DEFER-TO-TAX-DEFAULT when the model is weak. That is aligned with “hold vs sell” being the real decision, because the system declines to overtrade when it lacks evidence. fileciteturn69file0L1-L1 fileciteturn10file0L1-L1

However, the current policy mapping still mostly treats “actionability” as “is the regression model healthy enough to use,” rather than “is this month a sell regime.” Classification work strongly suggests the best next step is to separate those concerns: keep a regression health gate, but add a *decision-action gate* (classification) that answers “is this an actionable sell window.” That is precisely the hybrid framing documents endorse, even though v96 did not promote a classifier-led replacement. fileciteturn67file0L1-L1 fileciteturn36file0L1-L1 fileciteturn69file0L1-L1

The current classifier integration is intentionally shadow-only and interpretive, which matches the conservative posture and the v96 outcome. The classifier is built per benchmark, calibrated on an expanding-history logistic calibrator when there is enough OOS history, and then aggregated using benchmark quality weights into one “P(actionable sell)” number with a tier/stance. That is a sensible “small-sample first” implementation. fileciteturn60file0L1-L1 fileciteturn61file0L1-L1

**Reporting and UI surfaces**

The artifact bundle shape is clear and enforced in the workflow: monthly outputs include a markdown recommendation memo, diagnostics, CSVs for signals and benchmark quality, a consensus shadow CSV, plus a static HTML dashboard snapshot and machine-readable JSON artifacts (monthly_summary.json, run_manifest.json). This is exactly the kind of stable artifact contract that works well with GitHub Actions and avoids extra infrastructure. fileciteturn48file0L1-L1 fileciteturn56file0L1-L1 fileciteturn8file0L1-L1

A key strength is the move toward a structured summary payload (`monthly_summary.json`) and a manifest, which reduces the brittleness of downstream parsing. That said, both the email sender and the Streamlit dashboard still do some regex parsing of markdown for backwards compatibility. The next low-risk UX/maintainability win is to extend the summary payload enough that email/dashboard rely on it for *all* top-level fields and only embed markdown as a “full report” blob. fileciteturn56file0L1-L1 fileciteturn55file0L1-L1 fileciteturn59file0L1-L1

The static dashboard snapshot is a pragmatic choice: it gives you a linkable, committed HTML report without running entity["company","Streamlit","python app framework"] in production. It already includes classifier shadow fields and a per-benchmark signals table, but it explicitly drops the consensus shadow dataframe and does not show a compact “where did cross-check disagree” block. A small addition here can improve end-user clarity without adding noise: show a single cross-check row (live vs shadow mode/sell% agreement) and only expand to details if disagreement occurs. fileciteturn57file0L1-L1 fileciteturn56file0L1-L1

**Architecture and maintainability**

The codebase is sensibly modularized (processing, models, reporting, research), but `scripts/monthly_decision.py` remains very large and acts as an orchestration “god script.” That is not inherently wrong for a single-user ops repo, but it increases the cost of safe changes. The biggest practical refactor opportunity is to extract *pure functions* for “build report bundle” and “write artifacts” into a dedicated module, and to pass shared intermediate frames (feature_df, benchmark_quality_df) into both regression and classifier paths rather than rebuilding them. fileciteturn42file0L1-L1 fileciteturn60file0L1-L1 fileciteturn71file0L1-L1

There is also some “historical layering” debt: some modules still read like they support a 4-model ensemble including BayesianRidge, while current config defaults are Ridge + GBT with prediction shrinkage and quality-weighted consensus. This shows up as documentation mismatch and unused branches. This is a maintainability risk (readers will make incorrect change assumptions). A low-risk fix is to update docstrings and remove or clearly label legacy branches as “retained for historical reruns only.” fileciteturn63file0L1-L1 fileciteturn72file0L1-L1

The archived repo peer review readmes include absolute local file paths, which breaks portability. Since this repo is explicitly designed to run via GitHub Actions, even archival docs should remain repo-relative to preserve “clone-and-read” usability. fileciteturn25file0L1-L1 fileciteturn26file0L1-L1

**Testing and validation**

CI is robust for this kind of repo: you have lint, mypy on hardened modules, pytest, and smoke tests of production entrypoints including a dry-run monthly decision path. This is unusually strong and should be preserved as-is while you add small new artifacts and decision-layer logic. fileciteturn52file0L1-L1

What’s missing (and is high-value) is a tight, deterministic unit-test suite around the *decision layer mapping functions*—especially anything that converts probabilities and predicted returns into a sell percentage or mode. This is where small logic regressions can have outsize real-world impact, even if model accuracy doesn’t change. The repo already has clean entry points to test here (`decision_rendering`, `monthly_summary`, `classification_shadow`). fileciteturn69file0L1-L1 fileciteturn56file0L1-L1 fileciteturn60file0L1-L1

**Workflow and deployment fit**

The GitHub Actions workflows are aligned with your constraints: weekly data accumulation, peer weekly data fetch, monthly EDGAR 8‑K fetch, the monthly report run, plus an automated drift-based retrain dispatcher that triggers the monthly decision workflow out-of-cycle if drift persists. This is a practical, low-infrastructure orchestration model. fileciteturn49file0L1-L1 fileciteturn50file0L1-L1 fileciteturn51file0L1-L1 fileciteturn54file0L1-L1 fileciteturn48file0L1-L1

The workflow also enforces artifact presence, which is excellent for contract stability. If you add any new production artifacts (recommended below), you should update that assertion list immediately so missing outputs fail fast rather than silently degrading downstream email/dashboard behavior. fileciteturn48file0L1-L1

**Underused or unfinished work**

The local Streamlit dashboard is useful for interactive exploration, but the repo has already “productized” the safer piece: the committed static dashboard snapshot used for email linking and lightweight review. You can either (a) keep Streamlit as a local-only tool and explicitly label it as such in documentation, or (b) invest a small amount of effort to make Streamlit load only from `monthly_summary.json` + CSVs (no markdown parsing), so it is a stable debugging surface. The latter is usually the better option if the dashboard is still used monthly. fileciteturn57file0L1-L1 fileciteturn58file0L1-L1 fileciteturn59file0L1-L1

The April 8 peer review correctly emphasized shrinkage and classification as the “high-leverage” directions, and the repo already acted on those ideas (v38 shrinkage baseline, later v76 consensus changes, and the v87–v96 classification effort). That means the next enhancement should not re-litigate “should we do classification” but instead resolve the precise failure mode: disagreement with the baseline and insufficiently conservative integration into decision policy. fileciteturn77file0L1-L1 fileciteturn76file0L1-L1 fileciteturn67file0L1-L1 fileciteturn68file0L1-L1

## Recommended Next Steps

Immediate production-safe improvements (low complexity, high clarity payoff):

- Add a **classifier detail artifact** (e.g., `classification_shadow.csv`) to each monthly folder. The repo already computes per-benchmark classifier probabilities internally for reporting; persisting them makes diagnostics and future calibration monitoring much easier, and it fits the existing artifact workflow. Update artifact policy and the workflow’s expected output list accordingly. fileciteturn60file0L1-L1 fileciteturn48file0L1-L1 fileciteturn8file0L1-L1
- Promote **monthly_summary.json as the single source of truth** for top-level decision fields across email and dashboards by expanding the schema slightly (no major UI changes required). This reduces brittle regex parsing and keeps future surface changes cheap. fileciteturn56file0L1-L1 fileciteturn55file0L1-L1
- Fix portability of archived docs by removing absolute local links and replacing them with repo-relative paths. This has zero modeling risk and improves maintainability. fileciteturn25file0L1-L1 fileciteturn26file0L1-L1
- Add an explicit “**hold vs sell**” line at the very top of the recommendation memo (and mirror it in the dashboard snapshot), using existing fields (Recommendation Mode, sell_pct). This is a decision-support repo; the memo should answer the question in the first 5 seconds. fileciteturn69file0L1-L1 fileciteturn57file0L1-L1

Shadow-only production monitoring additions (no behavior change; increases evidence and promotion readiness):

- Implement a **shadow “classification gate overlay” recommendation** that computes “what sell_pct would have been” under a conservative classifier gate, but does *not* change the live recommendation. Persist as a small artifact (e.g., `decision_overlays.csv` with one row for live and one row for shadow gate) and add a compact section in diagnostic.md (not in the top executive summary). fileciteturn67file0L1-L1 fileciteturn68file0L1-L1
- Start logging classifier probabilities (at least the aggregated “P(actionable sell)” and the stance/tier) in a stable, queryable location (either DB table or a lightweight append-only CSV inside `results/monthly_decisions/decision_log.md`-style history). This enables later calibration drift monitoring when outcomes become known after the forecast horizon. fileciteturn60file0L1-L1 fileciteturn54file0L1-L1

Next research-only cycle (narrowly scoped; avoid unnecessary complexity):

- Re-run the “best candidate selection” logic as a constrained decision problem: maximize policy utility *subject to* agreement/stability constraints, rather than selecting only the highest mean policy return. The repo’s own v96 gate makes this explicit; the research selection step should incorporate the same constraints earlier so you shortlist promotable candidates. fileciteturn68file0L1-L1
- Reformulate classification targets to better map to the actual vest decision structure. The v87-v96 cycle focused heavily on “actionable sell” one-sided labeling; the next minimal extension is either (a) add a symmetric “actionable hold” label, or (b) redefine the policy objective around “deviate from default 50% sell” rather than “hold all vs sell all.” This directly targets the agreement failure mode. fileciteturn67file0L1-L1 fileciteturn69file0L1-L1

Medium-term candidates for promotion (only after shadow monitoring shows stability):

- Promote classification from “confidence block” to a **limited production gate** that only allows *more aggressive selling* when classifier confidence is high and regression diagnostics are above threshold. This is materially safer than making classification the primary engine immediately because it limits activation to “high-confidence sell regimes” and preserves the default behavior otherwise. fileciteturn60file0L1-L1 fileciteturn69file0L1-L1
- If (and only if) shadow overlay shows strong long-run agreement and incremental utility, consider making classification the primary “Recommendation Mode” selector, with regression retained for sizing/magnitude/context. fileciteturn67file0L1-L1

Ideas to explicitly defer or retire (based on repo evidence and constraints):

- Defer further nonlinear classifier family sweeps and broad feature-family expansion: v88 and v91 already indicate low payoff and higher overfitting risk in this small-sample setting. fileciteturn29file0L1-L1 fileciteturn32file0L1-L1
- Defer multi-infrastructure deployments (databricks, scheduled servers) because the repo is already well-fit to GitHub Actions and artifact commits; new infra adds operational risk without strong evidence of decision-quality gain. fileciteturn48file0L1-L1 fileciteturn7file0L1-L1
- Retire or clearly label legacy ensemble branches (if they are not used in production) to reduce mental overhead and “false affordances” for future changes. fileciteturn63file0L1-L1 fileciteturn72file0L1-L1

## Classification and Hybrid Research Recommendation

Classification-first vs regression-first vs hybrid

A classification-first decision *layer* is the right direction, but not a classification-only production engine. The v87–v96 evidence supports that classification can meaningfully describe “actionability regimes,” but it also shows you can accidentally get “high policy return” by doing something that is not operationally acceptable (e.g., diverging too often from the baseline). Therefore, the correct posture is: **hybrid architecture, with classification as the action gate and regression as the magnitude/explanation layer**—but integrated conservatively. fileciteturn67file0L1-L1 fileciteturn36file0L1-L1 fileciteturn68file0L1-L1

What should be researched next

The next cycle should not expand the model zoo. Instead, it should address the v96 failure mode directly:

- Reframe the policy so “classification controls *permission to deviate*” rather than “classification controls the whole hold_fraction.” This will mechanically increase agreement with the regression baseline and with the default vest rule, which is exactly what the promotion gate demands. fileciteturn68file0L1-L1 fileciteturn69file0L1-L1
- Use the existing best-performing ingredients (actionable_sell_3pct target, lean feature set, logistic + prequential calibration) and spend effort on a *small* threshold/policy sweep that optimizes a multi-objective score: utility uplift, agreement, and stability. The repo already codifies those criteria; the research selection stage should as well. fileciteturn35file0L1-L1 fileciteturn61file0L1-L1 fileciteturn68file0L1-L1

What should remain in production now

Keep the current regression + quality-weighted consensus as the only decision-changing engine. The repo’s governance and artifacts are designed for that conservative posture, and the classification branch has not earned promotion. The classifier should remain explicitly shadow-only in primary surfaces until a shadow overlay clears gates repeatedly. fileciteturn63file0L1-L1 fileciteturn36file0L1-L1 fileciteturn60file0L1-L1

Safest promotion path for classification in production

A phased progression that matches your constraints and current repo maturity:

- Phase A (now): classifier confidence block only (already implemented). fileciteturn60file0L1-L1
- Phase B: add a **shadow gate overlay artifact** (what would the sell% and mode have been). Keep it out of the top-of-memo “Executive Summary,” but show it in diagnostics and in machine-readable summary for longitudinal tracking. This enables stable monitoring and disagreement analysis without user-facing noise. fileciteturn56file0L1-L1
- Phase C: if the overlay repeatedly improves utility proxies *and* meets agreement/stability gates, promote classification to a **limited gate**: only allow “sell more than default” when classifier is high-confidence actionable-sell and regression health is strong. This limits blast radius and aligns with “hold vs sell” under concentrated equity risk. fileciteturn69file0L1-L1
- Phase D: only after Phase C demonstrates stability over enough months and any reserved holdout promotion checks are passed, allow classification to influence Recommendation Mode directly (still with regression as magnitude/context). fileciteturn10file0L1-L1

Additional classification research worth doing

Worth doing (high signal-to-effort):

- Two-sided actionability (add a separate “actionable_hold” label or a “deviate-from-default” target) because it directly addresses the divergence problem and better matches the actual vest action space. fileciteturn67file0L1-L1
- Constrained threshold selection and multi-objective selection criteria (utility uplift + agreement + stability) to produce promotable candidates rather than “interesting but unusable” winners. fileciteturn68file0L1-L1
- Calibration monitoring that is explicitly prequential and horizon-aware (evaluate once outcomes mature). fileciteturn61file0L1-L1

Avoid (unnecessary complexity given evidence):

- Any large hyperparameter sweep of nonlinear models or deep models; the repo’s own nonlinear sweep did not justify it. fileciteturn32file0L1-L1
- Broad feature expansion beyond curated families without a precommitted, stepwise plan; v88 indicates it mostly hurts. fileciteturn29file0L1-L1

## Implementation Plan

This plan assumes the version sequence continues after v96 and uses “v97+” as the next enhancement block. The goal is to produce immediate decision-quality clarity wins, add the missing shadow overlay scaffolding for classification promotion, and run a tightly bounded research cycle focused on alignment and gate-passability.

Documentation and archive work

- Update archived peer-review readmes to remove absolute local paths and replace with repo-relative links.
- Extend `docs/artifact-policy.md` to include any new artifacts added in v97/v98.
- Publish a short “Decision-layer schema” note in `docs/model-governance.md` describing promotion gates for classification overlays (mirroring the v96 code gate).

Production code changes (safe, incremental)

- Add `classification_shadow.csv` artifact writing (per-benchmark classifier details: raw prob, calibrated prob, history obs, classifier_weight, contribution).
- Expand `monthly_summary.json` schema to include:
  - top-line decision headline (string)
  - classifier artifact filename (if enabled)
  - shadow overlay (if present; see v98)
- Reduce markdown parsing in email/dashboard:
  - prefer summary payload fields wherever available, leave markdown as full-body embed only.

Shadow-only changes

- Implement a new “shadow classification gate overlay” that computes a hypothetical mode/sell% under a conservative hybrid rule, but does not change the live decision.
- Emit `decision_overlays.csv` or `shadow_gate.csv` with exactly two rows: live decision and shadow-gated decision, plus a “would_change” boolean and a short reason string.
- Add a small diagnostic section in `diagnostic.md` summarizing that comparison.

Research scripts (next bounded cycle)

- Add a research script to re-rank existing v94/v95 variants using the same constraints as the v96 gate (agreement/stability), so the top candidate is promotable rather than purely highest-utility.
- Add one narrow experiment: policy mapping that treats default 50% sell as baseline and only deviates when classification is high confidence and regression magnitude corroborates.

Tests and validation

- Add unit tests for:
  - the new shadow overlay mapping logic (input → deterministic mode/sell% output)
  - monthly summary schema increment (schema_version, required keys)
  - artifact bundle completeness (including new files) in a dry-run monthly_decision execution
- Add a regression test ensuring backdated `--as-of` runs do not include targets that require future data (truncate targets beyond as_of-horizon).

Promotion gates to encode (beyond research)

- Shadow overlay may be considered for promotion to “gating layer” only if:
  - it meets or exceeds regression baseline utility proxy *and*
  - agreement with baseline recommendation mode and sell bands is ≥ 0.70 over a sufficiently long replay window *and*
  - it does not materially increase month-to-month action volatility (hold/sell changes).

## Final Deliverable

```text
<proposed_plan>
v97 — Production-safe decision-surface hardening + classifier artifactization
  Goal
    - Improve decision clarity and monitoring readiness without changing live behavior.
    - Make classifier shadow output fully inspectable and contract-stable.

  Documentation / archive
    - Edit:
      - docs/archive/history/repo-peer-reviews/2026-04-05/README.md
      - docs/archive/history/repo-peer-reviews/2026-04-10/README.md
      - Replace absolute local file links with repo-relative links.
    - Edit:
      - docs/artifact-policy.md
        - Add new artifact: classification_shadow.csv
        - Define retention and “shadow-only” status.
    - Edit:
      - docs/model-governance.md
        - Add a short section: “Classifier shadow + promotion gates (v96+)”
        - Explicitly note: classifier is non-binding unless promoted.

  Production code changes (no decision behavior changes)
    - Add new artifact writer:
      - New file: src/reporting/classification_artifacts.py
        - Function: write_classification_shadow_csv(out_dir: Path, detail_df: pd.DataFrame) -> Path
        - Enforce stable columns:
          benchmark,
          classifier_raw_prob_actionable_sell,
          classifier_prob_actionable_sell,
          classifier_history_obs,
          classifier_weight,
          classifier_weighted_contribution,
          classifier_shadow_tier
    - Wire into monthly pipeline:
      - Edit: scripts/monthly_decision.py
        - After build_classification_shadow_summary(...):
          - Persist detail_df via write_classification_shadow_csv(...)
          - Add filename into monthly_summary payload (see below)
    - Extend summary payload:
      - Edit: src/reporting/monthly_summary.py
        - Bump schema_version from 2 -> 3
        - Add fields:
          artifacts.classification_shadow_csv = "classification_shadow.csv" (when enabled)
          classification_shadow.top_supporting_benchmark (already present in summary payload today)
          classification_shadow.top_supporting_contribution_label (already present today)
        - Ensure backward compatibility: if no classifier, artifact field omitted or null.
    - Reduce parsing brittleness:
      - Edit: src/reporting/email_sender.py
        - Prefer monthly_summary.json fields for:
          Signal, Recommendation Mode, Sell %, Predicted return, Confidence checks, Classification fields.
        - Keep markdown parsing only as fallback.
      - Edit: dashboard/app.py and dashboard/data.py
        - Prefer monthly_summary.json for top-level fields; keep regex parsing only as fallback.

  Workflow / artifact contract
    - Edit: .github/workflows/monthly_decision.yml
      - Add expected artifact:
        - results/monthly_decisions/<YYYY-MM>/classification_shadow.csv
    - Validation
      - Run CI smoke test monthly_decision.py --dry-run
      - Confirm workflow artifact verification passes.

  Tests
    - Add:
      - tests/test_monthly_summary_schema_v3.py
        - Validate schema_version==3 keys exist and types are stable.
      - tests/test_classification_shadow_artifact_columns.py
        - Build minimal synthetic detail_df; assert CSV columns and ordering.

v98 — Shadow-only classification gate overlay (no production override)
  Goal
    - Compute and display a conservative “what would we do if classification gated actionability?”
      without changing the live action.
    - Create a promotion-ready shadow monitoring surface that directly targets the v96 failure mode
      (low agreement with regression baseline).

  Shadow-only code
    - New module: src/models/classification_gate_overlay.py
      - Function: compute_shadow_gate_overlay(
          live_mode: str,
          live_sell_pct: float,
          consensus: str,
          mean_predicted: float,
          mean_ic: float,
          classifier_prob_actionable_sell: float | None,
          lower: float = 0.30,
          upper: float = 0.70,
        ) -> dict
      - Policy (conservative):
        - If classifier_prob_actionable_sell is None:
            overlay_mode = live_mode; overlay_sell_pct = live_sell_pct; reason = "no classifier"
        - Else if prob >= upper:
            overlay_mode = "ACTIONABLE" only if live diagnostics already pass actionable gate;
            overlay_sell_pct = max(live_sell_pct, 0.75) or 1.00 when consensus == "UNDERPERFORM"
            reason = "high confidence actionable-sell"
        - Else if prob <= lower:
            overlay does NOT force “hold all”; it defaults to live decision (protects agreement)
            reason = "classifier non-actionable (no override)"
        - Else:
            overlay = live decision; reason = "classifier neutral band"
      - Hard constraint: overlay may only *increase* sell_pct vs live; never decrease (conservative).

  Artifact + reporting integration
    - New artifact: decision_overlays.csv (or shadow_gate.csv)
      - Two rows: live, shadow_gate
      - Columns:
        variant, recommendation_mode, recommended_sell_pct, would_change, reason,
        classifier_prob_actionable_sell
    - Edit: scripts/monthly_decision.py
      - Compute overlay after live recommendation computed
      - Write decision_overlays.csv
      - Include overlay summary in monthly_summary.json under key "shadow_gate_overlay"
    - Edit: src/reporting/decision_rendering.py (diagnostic section only)
      - Add a short “Shadow gate overlay” block in diagnostic.md showing:
        - live vs shadow sell%, would_change, reason
      - Do NOT add to top Executive Summary to avoid noise.
    - Edit: src/reporting/dashboard_snapshot.py
      - If overlay present, show a compact “Overlay: would_change = yes/no” card.

  Workflow / artifact contract
    - Edit: .github/workflows/monthly_decision.yml
      - Add decision_overlays.csv to expected artifacts.

  Tests
    - Add: tests/test_classification_gate_overlay_policy.py
      - Parameterized inputs across:
        prob=None, prob=0.2, prob=0.5, prob=0.8
        consensus UNDERPERFORM / NEUTRAL / OUTPERFORM
        mean_ic below/above threshold
      - Assert:
        - overlay never decreases sell_pct
        - overlay never sets ACTIONABLE when live is DEFER-TO-TAX-DEFAULT
        - output dict keys stable

v99 — Research-only: promotable-candidate selection aligned with v96 gate
  Goal
    - Produce a shortlist that satisfies agreement + stability constraints, not just utility.

  Research scripts
    - Add: results/research/v99_constrained_policy_selection.py
      - Load v94_hybrid_gate_results.csv and v95_policy_replay_results.csv
      - Re-rank candidates by:
        1) mean_policy_return uplift vs regression baseline (must be > 0 to pass)
        2) agreement_with_regression_rate (must be >= 0.70)
        3) hold_fraction_changes <= baseline + 4
        4) tie-break: lower defer_rate (optional) or lower churn
      - Write:
        - results/research/v99_constrained_policy_selection_results.csv
        - results/research/v99_constrained_policy_selection_summary.md

  Tests
    - Add: tests/test_research_v99_outputs.py
      - Assert required output files exist and required columns present.

v100 — Research-only: target/policy reframing around “deviate from default 50%”
  Goal
    - Reduce baseline disagreement by construction: default = sell 50% unless high-confidence override.

  Research scripts
    - Add: results/research/v100_default_deviation_policy_replay.py
      - Policy candidates:
        - baseline: regression sell_pct mapping
        - overlay: only increase sell_pct when classifier high-confidence actionable-sell
        - (optional) two-sided: add actionable_hold and allow decrease sell_pct only when both
          regression and classifier strongly support it (very conservative thresholds)
      - Output:
        - results/research/v100_default_deviation_policy_replay_results.csv
        - results/research/v100_default_deviation_policy_replay_summary.md

  Promotion gate definition (document-only)
    - Add to docs/model-governance.md:
      - For any future production promotion:
        - must pass v96-style gate
        - must show stable behavior in shadow overlay for N months (configure N, e.g., 6+)

v101 — Production readiness hardening for strict as-of backdated runs (methodology hygiene)
  Goal
    - Ensure backdated --as-of runs cannot leak future target values when DB contains later prices.

  Production code changes
    - Edit: src/processing/feature_engineering.py
      - Add helper:
        truncate_relative_target_for_asof(relative_returns: pd.Series, as_of: pd.Timestamp, horizon_months: int) -> pd.Series
        - Set target to NaN for dates > (as_of - horizon_months offset) OR enforce end_date bound.
    - Edit: scripts/monthly_decision.py and src/models/classification_shadow.py
      - When loading relative return series for training/evaluation, apply truncation under backdated as-of.

  Tests
    - Add: tests/test_asof_target_truncation.py
      - Synthetic target series with future-known values; ensure truncation zeros them out for backdated as-of.

Execution order recommendation
  - Implement v97 fully (artifact + schema + docs + tests) and land it behind CI.
  - Implement v98 shadow overlay and observe it for several months without promotion.
  - Run v99–v100 research to identify the best promotable candidate consistent with agreement gates.
  - Only after shadow overlay shows acceptable agreement/stability, consider a gated production promotion proposal.

</proposed_plan>
```