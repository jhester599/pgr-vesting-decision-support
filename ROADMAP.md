# PGR Vesting Decision Support - Roadmap

For completed work and release history, see [CHANGELOG.md](CHANGELOG.md).

## Current State

**Master baseline: v138** - the live monthly workflow uses the quality-weighted
consensus regression path (v76/v38 stack) with the shadow classification and
decision-layer research artifacts completed through the 2026-04-14 bounded
sweep set. The local autoresearch stack now includes `v129`, `v131`,
`v133-v138`, and the runtime optimization pass.

**Current operating posture**

- production recommendation path: quality-weighted regression consensus (v76/v38)
- shadow classification: per-benchmark separate logistic, lean 12-feature baseline,
  prequential calibration, quality-weighted aggregation
  (`separate_benchmark_logistic_balanced`)
- Path B (temperature-scaled composite portfolio-target classifier) now runs alongside
  Path A in shadow mode; both signals appear in monthly artifacts
- monthly artifacts include `benchmark_quality.csv`, `consensus_shadow.csv`,
  `classification_shadow.csv`, `dashboard.html`, and `monthly_summary.json`
- April 2026 classifier snapshot: Path A P(Actionable Sell) = 36.0%, stance `NEUTRAL`;
  Path B temp-scaled P(Actionable Sell) = 53.4%, stance `NEUTRAL`, tier `LOW`
- classifier remains shadow-only; production promotion requires >= 24 matured
  prospective months meeting calibration, balanced accuracy, and precision gates
- v128 benchmark-specific selection switched 4 of 10 benchmarks away from the
  shared `lean_baseline` map: `BND`, `DBC`, `VGT`, and `VIG`

## Active Research Direction: v139-v152

The active follow-on plan is documented in:

- [`docs/superpowers/plans/2026-04-16-v139-v152-autoresearch-followon.md`](docs/superpowers/plans/2026-04-16-v139-v152-autoresearch-followon.md)
- [`docs/superpowers/plans/2026-04-13-autoresearch-execution-plan.md`](docs/superpowers/plans/2026-04-13-autoresearch-execution-plan.md)
- [`docs/superpowers/plans/2026-04-14-autoresearch-execution-log.md`](docs/superpowers/plans/2026-04-14-autoresearch-execution-log.md)

Summary of the v139-v152 follow-on arc:

| Version | Theme | Type |
|---|---|---|
| v139 | Re-baseline docs, archive the 2026-04-16 report, refresh backlog, publish restart procedure | Documentation + governance |
| v140 | Ensemble shrinkage alpha sweep on the current production research frame | Research harness |
| v141 | Fixed Ridge-vs-GBT blend-weight sweep on the current production research frame | Research harness |
| v142 | EDGAR filing-lag bounded review | Research harness |
| v143 | Correlation-pruned feature-set evaluation | Research harness |
| v144 | Conformal coverage and ACI gamma replay backtest | Research harness |
| v145 | WFO train/test window bounded sweep | Research harness |
| v146 | Path B threshold sweep on top of the tuned v135 temperature baseline | Research harness |
| v147 | Coverage-weighted Path A / Path B aggregation proxy | Research harness |
| v148 | Positive-class weighting replay proxy | Research harness |
| v149 | Kelly fraction / cap replay proxy | Research harness |
| v150 | Neutral-band replay proxy | Research harness |
| v151 | Reporting/documentation polish for restart-safe autonomous runs | Docs + UX |
| v152 | Final synthesis and next-session handoff | Documentation |

The working rule for the v139-v152 cycle remains the same as the original
autoresearch pass: research only, no automatic promotion into the live monthly
decision path, and mandatory closeout/update notes after each completed block.

Current execution progress on 2026-04-16:

- `v140` complete: shrinkage re-check was flat across the bounded search range,
  so the `0.50` candidate remains unchanged
- `v141` complete: fixed Ridge-vs-GBT blend-weight sweep found a research-only
  winner at `ridge_weight=0.60`
- `v142` complete: EDGAR filing-lag review did not beat the incumbent
  `lag=2` on the balanced pooled metric view
- `v143` complete: correlation-pruned feature sweep found a bounded winner at
  `rho=0.80`
- `v144` complete: conformal replay tuning now favors
  `{"coverage": 0.75, "aci_gamma": 0.03}` on the bounded search frame
- `v145` complete: WFO window review logged a tradeoff-heavy `(48, 6)` result,
  but retained the incumbent `{"train": 60, "test": 6}` candidate
- `v146` complete: threshold follow-through retained the incumbent
  `{"low": 0.15, "high": 0.70}` candidate
- `v147` complete: coverage-weighted Path A / Path B aggregation did not beat
  the baseline multiplier
- `v148` complete: class-weight replay proxy retained the incumbent
  `positive_weight=1.0`
- `v149` complete: Kelly replay-proxy sweep updated the candidate to
  `{"fraction": 0.50, "cap": 0.25}`
- `v150` complete: neutral-band review kept the incumbent `0.015` setting after
  the Kelly update
- Next restart point: `v151` reporting and artifact polish

## Prior Research Direction: v123-v129

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
| v129 | Dual-track benchmark-specific shadow integration and promotion governance | Research + governance |
| v130 | Temperature-scaled Path B adoption analysis vs Path A matched | Research |
| v131 | Temperature-scaled Path B wired into production shadow artifacts | Shadow integration |
| v132+ | SCHD preparation and conditional per-benchmark inclusion | Future |

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
| Composite target Path B evaluation | v127 completed calibration sweep; v130 re-evaluated against Path A matched and adopted; v131 wired into shadow artifacts |
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
