# PGR Vesting Decision Support - Roadmap

For completed work and release history, see [CHANGELOG.md](CHANGELOG.md).

## Current State

**Master baseline: v122** - the live monthly workflow uses the quality-weighted
consensus regression path (v76/v38 stack) with a shadow classification layer
(v122 classifier audit). A full `v87-v128` classification research arc has been
completed. Two external deep research peer reviews were commissioned on
2026-04-12 and synthesized into the `v123-v128` enhancement plan, which then
delivered `v126` methodology hardening, `v127` Path B calibration research, and
`v128` benchmark-specific full feature search across the 72-feature candidate
universe.

**Current operating posture**

- production recommendation path: quality-weighted regression consensus (v76/v38)
- shadow classification: per-benchmark separate logistic, lean 12-feature baseline,
  prequential calibration, quality-weighted aggregation
  (`separate_benchmark_logistic_balanced`)
- monthly artifacts include `benchmark_quality.csv`, `consensus_shadow.csv`,
  `classification_shadow.csv`, `dashboard.html`, and `monthly_summary.json`
- April 2026 classifier snapshot: P(Actionable Sell) = 35.2%, stance `NEUTRAL`
- classifier remains shadow-only; production promotion requires >= 24 matured
  prospective months meeting calibration, balanced accuracy, and precision gates
- v128 benchmark-specific selection switched 4 of 10 benchmarks away from the
  shared `lean_baseline` map: `BND`, `DBC`, `VGT`, and `VIG`

## Active Research Direction: v123-v129

The active plan is documented in:

- [`docs/superpowers/plans/2026-04-12-v123-v128-classification-enhancement-plan.md`](docs/superpowers/plans/2026-04-12-v123-v128-classification-enhancement-plan.md)
- [`docs/superpowers/plans/2026-04-12-v126-methodology-hardening.md`](docs/superpowers/plans/2026-04-12-v126-methodology-hardening.md)
- [`docs/superpowers/plans/2026-04-12-v127-path-b-calibration.md`](docs/superpowers/plans/2026-04-12-v127-path-b-calibration.md)
- [`docs/superpowers/plans/2026-04-12-v128-benchmark-specific-feature-search.md`](docs/superpowers/plans/2026-04-12-v128-benchmark-specific-feature-search.md)

Summary of the v123-v129 arc:

| Version | Theme | Type |
|---|---|---|
| v123 | Portfolio-aligned aggregation plumbing | Research + shadow |
| v124 | VGT classifier addition | Shadow expansion |
| v125 | Path B composite portfolio-target classifier | Parallel research |
| v126 | Methodology hardening: matched Path A parity, rolling WFO enforcement, artifact refresh | Research hygiene |
| v127 | Path B calibration sweep on matched v126 folds | Research |
| v128 | Full benchmark-specific feature search over the 72-feature universe with forward stepwise plus L1 / elastic-net cross-checks | Research |
| v129 | Optional benchmark-specific shadow integration, prospective monitoring, and promotion governance | Research + governance |
| v130+ | SCHD preparation and conditional per-benchmark inclusion | Future |

**The single highest-priority completed change (v123):** replace
quality-weighted aggregation with portfolio-weighted aggregation over the
investable redeploy universe `{VOO, VGT, VXUS, VWO, BND}`, using fixed
`balanced_pref_95_5` base weights. This closes the portfolio alignment gap
identified in the 2026-04-12 peer reviews.

**Key architectural decision resolved:** both Path A (per-benchmark classifiers ->
portfolio-weighted aggregation) and Path B (single classifier on composite
portfolio target) were developed in parallel from v125. v126 provides the
matched, rolling-window comparison frame required to compare those paths fairly.

**Current classifier research takeaway:** benchmark-specific subsetting helps a
minority of benchmarks materially, especially `VGT`, but the pooled v128 gain is
modest. Any adoption decision should therefore be framed as a shadow-monitoring
upgrade first, not an immediate promotion trigger.

## Strategic Backlog

| Item | Description |
|---|---|
| SCHD per-benchmark classifier | Add when history reaches >= 185 observations (~v135, late 2027) |
| Structured monthly schema follow-through | Expand `monthly_summary.json` into future automations |
| Promotion gate implementation | v129: implement gate check infrastructure; gate remains off until 24 matured prospective months |
| Composite target Path B evaluation | v127 completed calibration sweep; no candidate clears adoption gate, so Path B remains diagnostic-only |
| Benchmark-specific feature-map integration | v128 completed map selection; shadow integration remains a follow-on decision |
| VGT robustness audit | Re-run the VGT search across nearby as-of dates and require evidence that the winning signal family remains stable, not just the exact 2-feature pair |
| VGT selector-agreement gate | Before adopting the VGT-specific subset, require forward-stepwise and regularized selectors to agree on the same signal cluster or produce comparable prospective results |
| VGT dual-track shadow monitoring | If the feature map is wired into shadow mode, report both `lean_baseline` and the VGT-specific subset side by side until enough prospective months accumulate |
| VGT minimum robustness rule | Consider a governance rule that a benchmark-specific switch is not adopted unless it survives fold-stability checks or expands into a slightly broader, still-calibrated subset |

## Development Principles

- Never finalize a module without a passing pytest suite.
- No K-Fold cross-validation - `TimeSeriesSplit` with purge/embargo only.
- No `StandardScaler` across the full dataset prior to temporal splitting.
- No `yfinance` for fundamentals or historical ratios.
- Python 3.10+, strict PEP 8, standard type hints.

## Monthly Decision Log

See [`results/monthly_decisions/decision_log.md`](results/monthly_decisions/decision_log.md)
for the persistent record of monthly recommendations.
