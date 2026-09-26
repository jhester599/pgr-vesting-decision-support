# Deep Research Prompt: Full-Repo Review and Next Enhancement Plan

Use the prompt below with a deep research model to review the repository in detail and propose the next enhancement cycle.

---

## Copy-Paste Prompt

You are conducting a deep research review of the GitHub repository `jhester599/pgr-vesting-decision-support`.

Your job is to review the repository in detail and recommend the next enhancement steps. This review should be practical and implementation-oriented, not just descriptive. The final output must include a concrete implementation plan that can be handed directly to Codex or Claude Code.

### Project Context

This repository supports monthly and vest-event decision support for a user holding concentrated Progressive Corporation (`PGR`) equity. The most important real-world decision is `hold vs. sell`, not precise regression accuracy for relative return magnitude.

The project started as a relative-return forecasting system built mainly around regression. More recent research suggests that a hybrid approach may be more aligned with the user problem:

- classification for the primary decision question:
  - is this month/action window better framed as `actionable sell` vs `non-actionable`?
- regression for secondary context:
  - magnitude
  - explanation
  - sell sizing

The project has already added:

- production monthly reporting and email/dashboard surfaces
- a live quality-weighted consensus path
- cross-check and diagnostic artifacts
- a full `v87-v96` classification/hybrid research program
- a shadow-only monthly classification confidence section

The repo is intentionally conservative. We do not want to introduce unnecessary complexity. We prefer maintainable, transparent improvements that work well with GitHub Actions as the primary deployment and orchestration surface.

### Important Constraints

Please use these constraints when recommending next steps:

- The key user decision is `hold vs. sell`.
- Avoid complexity unless there is strong evidence it materially improves decision quality.
- Keep GitHub Actions workflow as the primary deployment/execution model.
- Prefer additions that preserve the current monthly artifact workflow rather than replacing it.
- Respect strict time-series methodology:
  - no K-Fold validation
  - no temporal leakage
  - no use of future information in model fitting, calibration, or thresholds
- Recommendations should distinguish clearly between:
  - research-only next steps
  - shadow-only production monitoring additions
  - true production-promotion candidates

### Materials To Review

Review the repository broadly, but pay particular attention to:

#### High-level docs and current-state docs

- `README.md`
- `ROADMAP.md`
- `docs/architecture.md`
- `docs/workflows.md`
- `docs/operations-runbook.md`
- `docs/artifact-policy.md`
- `docs/decision-output-guide.md`
- `docs/model-governance.md`
- `docs/troubleshooting.md`

#### Prior peer-review and repo-review materials

- `docs/archive/history/peer-reviews/2026-04-08/`
- `docs/archive/history/repo-peer-reviews/2026-04-05/`
- `docs/archive/history/repo-peer-reviews/2026-04-10/`

#### Planning documents and recent implementation plans

- all relevant files under `docs/superpowers/plans/`
- especially:
  - `2026-04-10-v37-v60-results-summary.md`
  - `2026-04-11-v81-v88-repo-review-and-adoption-plan.md`
  - `2026-04-11-v87-v96-classification-hybrid-research.md`

#### Recent research outputs

- `results/research/v87_target_taxonomy_summary.md`
- `results/research/v88_feature_sweep_summary.md`
- `results/research/v89_per_benchmark_linear_summary.md`
- `results/research/v90_pooled_panel_classifiers_summary.md`
- `results/research/v91_nonlinear_classifier_sweep_summary.md`
- `results/research/v92_calibration_and_abstention_summary.md`
- `results/research/v93_basket_targets_summary.md`
- `results/research/v94_hybrid_gate_summary.md`
- `results/research/v95_policy_replay_summary.md`
- `results/research/v96_decision_summary.md`

#### Current production/reporting surfaces

- `scripts/monthly_decision.py`
- `src/reporting/`
- `dashboard/`
- `.github/workflows/`

#### Current research and helper modules

- `src/research/`
- `src/models/`
- `src/processing/`

### Review Goals

Please review the repo and suggest next steps along several dimensions:

1. Model accuracy
   - What are the highest-value ways to improve decision-relevant predictive quality?
   - Should further work focus more on classification, regression, or hybrid approaches?
   - Is there a better way to define the target around the actual decision problem?

2. Model stability and governance
   - How should the repo better manage overfitting risk, model drift, calibration drift, and recommendation instability?
   - Are there simpler models or validation strategies that may be more robust?
   - Are there shadow-mode or holdout practices that should be strengthened?

3. Decision quality
   - How well does the current system support the real user decision of `hold vs. sell`?
   - Are current thresholds, neutral bands, or policy rules well aligned with that decision?
   - Should the recommendation layer become more classification-driven?

4. Recommendation clarity and UX
   - How can the monthly report, email, dashboard, and related artifacts better help a technically literate user understand:
     - what the recommendation is
     - how confident the system is
     - what changed from prior months
     - when the model disagrees with defaults or cross-checks
   - Suggest improvements that increase clarity without making the interface noisy.

5. Software architecture and maintainability
   - Identify architecture improvements that would reduce duplication or improve testability.
   - Review whether current reporting, research, and production code boundaries are clean.
   - Evaluate whether any parts are too brittle, too manual, or too tightly coupled.

6. Deployment and workflow fit
   - Recommendations should preserve GitHub Actions as the main operating model.
   - Avoid proposals that require heavy new infrastructure unless clearly justified.
   - Suggest improvements that fit the current artifact-based workflow and monthly cadence.

7. Underused or unfinished work
   - Identify prior enhancements that were built or explored but not fully adopted.
   - Consider whether they should be retired, promoted, or integrated more cleanly.
   - Examples may include dashboard adoption, email/reporting improvements, shadow artifacts, or partially exploited research findings.

### Specific Topics To Evaluate

Please explicitly address these questions:

1. Should the next major research cycle focus on classification-first decision support?
2. How should a hybrid classifier + regression system be structured if the primary goal is `hold vs. sell` rather than fine-grained return prediction?
3. What is the best way to use classification in production:
   - shadow-only confidence layer
   - gating layer
   - primary recommendation engine
   - some phased progression across those stages?
4. What additional classification research is worth doing, and what should be avoided as unnecessary complexity?
5. Are there promising model families, feature groups, calibration methods, or target formulations from the earlier peer reviews that deserve renewed attention?
6. Are there simpler policy-layer improvements that may matter more than squeezing more forecast accuracy from the core models?

### Preferences For Recommendations

Please bias toward recommendations that are:

- interpretable
- easy to monitor
- easy to validate with strict time-series discipline
- friendly to GitHub Actions and artifact-based workflows
- maintainable by one technically capable user with LLM coding assistance

Please avoid recommending large platform rewrites, overengineered orchestration, or model families that are unlikely to be stable in a small-sample time-series setting unless you make a very strong case.

### Required Output Format

Your response must include these sections:

#### 1. Executive Summary

- short summary of current repo status
- top 3-5 most important conclusions

#### 2. Detailed Findings

- findings organized by category:
  - modeling
  - decision layer
  - reporting / UI
  - architecture
  - testing / validation
  - workflow / deployment
- cite specific repo files and documents when relevant

#### 3. Recommended Next Steps

- identify:
  - immediate production-safe improvements
  - next research-only cycle
  - medium-term candidates for promotion
  - ideas that should be explicitly deferred or retired

#### 4. Classification / Hybrid Research Recommendation

- give a specific point of view on:
  - classification-only vs regression-only vs hybrid
  - which should be researched next
  - which should remain in production now
  - what promotion path would be safest

#### 5. Implementation Plan

Produce a concrete implementation plan that could be handed directly to Codex or Claude Code.

This plan should:

- assign version numbers consistent with the project’s existing versioning sequence
- propose a new version block after `v96`
- use the existing git/documentation structure where possible
- clearly separate:
  - documentation/archive work
  - production code changes
  - shadow-only changes
  - research scripts
  - tests
- identify likely files/modules to change
- include suggested validation/tests for each phase
- include promotion gates where relevant

The implementation plan should be detailed enough that a coding agent can begin executing it directly.

#### 6. Final Deliverable Block

End with a single structured block in this form:

```text
<proposed_plan>
[full detailed implementation plan here]
</proposed_plan>
```

The `<proposed_plan>` block should be the most actionable part of the response and should read like a working implementation roadmap for Codex or Claude Code.

### Additional Guidance

- Be thorough.
- Prefer repo-specific observations over generic advice.
- Distinguish signal from noise.
- If something was already tried and did not work, say so plainly and avoid re-proposing it unless there is a specific new angle.
- If a recommendation increases complexity, justify why the expected value exceeds that cost.
- Treat the user’s real problem as a decision-support problem first and a pure forecasting problem second.

