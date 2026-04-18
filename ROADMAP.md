# PGR Vesting Decision Support - Roadmap

For completed work and release history, see [CHANGELOG.md](CHANGELOG.md).

## Current State

**Master baseline: v151** - the repository now includes the merged
`v139-v150` follow-on research artifacts plus the `v151` side-by-side shadow
reporting lane. The live monthly workflow still uses the quality-weighted
consensus regression path (v76/v38 stack), while the follow-on winners remain
reporting-only in shadow artifacts.

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

## Active Research Direction: v153-v158

The active plan is documented in:

- [`docs/superpowers/plans/2026-04-17-v153-v158-classification-feature-research.md`](docs/superpowers/plans/2026-04-17-v153-v158-classification-feature-research.md)

Source peer review: [`docs/archive/history/repo-peer-reviews/2026-04-17/chatgpt_repo_peerreview_20260417.md`](docs/archive/history/repo-peer-reviews/2026-04-17/chatgpt_repo_peerreview_20260417.md)

Summary of the v153-v158 classification and feature research arc:

| Version | Theme | Type |
|---|---|---|
| v153 | Archive 2026-04-17 peer review; update backlog with FEAT-03; reorder priorities | Documentation |
| v154 | CLS-02: Firth-penalized logistic for short-history benchmarks | Classifier research |
| v155 | FEAT-02: WTI 3M momentum for DBC/VDE classification | Feature research |
| v156 | FEAT-01: USD index momentum (DTWEXBGS) for BND/VXUS/VWO | Feature research |
| v157 | FEAT-03: Term premium 3M differential signal | Feature research |
| v158 | Synthesis: compare all four experiments; update shadow lane if any winner qualifies | Research + shadow |

Working rule: research-only. No automatic promotion into the live monthly decision path. Mandatory closeout after each completed block.

Priority shift from v152 closeout: the 2026-04-17 peer review re-orders the queue to classification-first (CLS-02 before BL-01) on the basis that predictive signal improvements take precedence over decision-layer policy tuning. BL-01 remains open and should follow after v158 synthesis.

Current execution progress on 2026-04-17:

- `v153` complete: archived peer review, updated backlog and priorities
- `v154` complete: Firth logistic — **VMBS +0.0412, BND +0.0704 (winners)**
- `v155` complete: WTI momentum — no benefit (DBC +0.005, VDE +0.021)
- `v156` complete: USD momentum — no benefit (BND -0.077, VXUS flat, VWO +0.009)
- `v157` complete: term premium diff — no benefit (best VDE +0.017)
- `v158` complete: synthesis and handoff documented

Completed next queue after `v158`:

1. `v159` — Firth logistic shadow integration for VMBS and BND — **complete (2026-04-18)** (PR #83)
2. `BL-01` — Black-Litterman tau/view tuning — **complete (2026-04-18)**
3. `CLS-03` — Path A vs Path B (time-locked)

## Active Research Direction: Post-v158 (v159 + BL-01)

- `v159` complete: Firth shadow lane wired; see `docs/closeouts/V159_CLOSEOUT_AND_HANDOFF.md`
- `BL-01` complete: tau/risk_aversion sweep found incumbent optimal; see `docs/closeouts/BL01_CLOSEOUT_AND_HANDOFF.md`
- Next: `CLS-03` (time-locked on 24 matured months), `CLS-01` (depends on CLS-03)

## Prior Research Direction: v139-v152

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
- `v151` complete: the promoted `v139-v150` winners now appear in one
  reporting-only side-by-side shadow lane named `autoresearch_followon_v150`
- `v152` complete: the cycle is now closed with a final synthesis and ranked
  handoff note in `docs/closeouts/V152_CLOSEOUT_AND_HANDOFF.md`

Recommended next autonomous queue after `v152`:

1. `BL-01` - Black-Litterman tau/view tuning on the preserved replay-proxy
   frame, using the new follow-on winners only as shadow-side context
2. `CLS-02` - Firth logistic / short-history classifier stabilization research
   for thin benchmarks
3. `FEAT-01` - DTWEXBGS post-v128 feature search, starting with the benchmarks
   most likely to benefit from currency momentum
4. `FEAT-02` - WTI 3M momentum follow-through for the commodity / energy slice

Keep `CLS-03` blocked until the prospective-month gate matures, and keep
`REG-02` deferred until there is a stronger ensemble-level reason to reopen the
GBT line.

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
