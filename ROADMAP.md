# PGR Vesting Decision Support - Roadmap

For completed work and release history, see [CHANGELOG.md](CHANGELOG.md).

## Current State

**Master baseline: v122** — the live monthly workflow uses the quality-weighted
consensus regression path (v76/v38 stack) with a shadow classification layer
(v122 classifier audit). A full `v87–v122` classification research arc has been
completed. Two external deep research peer reviews were commissioned on 2026-04-12
and synthesized into the active `v123–v128` enhancement plan.

**Current operating posture**

- production recommendation path: quality-weighted regression consensus (v76/v38)
- shadow classification: per-benchmark separate logistic, lean 12-feature baseline,
  prequential calibration, quality-weighted aggregation (`separate_benchmark_logistic_balanced`)
- monthly artifacts include `benchmark_quality.csv`, `consensus_shadow.csv`,
  `classification_shadow.csv`, `dashboard.html`, and `monthly_summary.json`
- April 2026 classifier snapshot: P(Actionable Sell) = 35.2%, stance NEUTRAL
- classifier remains shadow-only; production promotion requires ≥ 24 matured
  prospective months meeting calibration, balanced accuracy, and precision gates

## Active Research Direction: v123–v128

The active plan is documented in:

- [`docs/superpowers/plans/2026-04-12-v123-v128-classification-enhancement-plan.md`](docs/superpowers/plans/2026-04-12-v123-v128-classification-enhancement-plan.md)

Summary of the v123–v128 arc:

| Version | Theme | Type |
|---|---|---|
| v123 | Portfolio-aligned aggregation plumbing | Research + shadow |
| v124 | VGT classifier addition | Shadow expansion |
| v125 | Path B composite portfolio-target classifier | Parallel research |
| v126 | Calibration overhaul + asymmetric threshold optimization | Research |
| v127 | Benchmark-specific feature subsetting (L1→L2) | Research |
| v128 | New feature candidates + promotion governance | Research + governance |
| v130+ | SCHD preparation and conditional per-benchmark inclusion | Future |

**The single highest-priority change (v123):** replace quality-weighted aggregation
with portfolio-weighted aggregation over the investable redeploy universe {VOO, VGT,
VXUS, VWO, BND}, using fixed `balanced_pref_95_5` base weights. This closes the
portfolio alignment gap identified in the 2026-04-12 peer reviews.

**Key architectural decision resolved:** both Path A (per-benchmark classifiers →
portfolio-weighted aggregation) and Path B (single classifier on composite portfolio
target) will be developed in parallel from v125. Architecture selection will be made
empirically after results are in hand.

## Strategic Backlog

| Item | Description |
|---|---|
| SCHD per-benchmark classifier | Add when history reaches ≥ 185 observations (~v135, late 2027) |
| Structured monthly schema follow-through | Expand `monthly_summary.json` into future automations |
| Promotion gate implementation | v128: implement gate check infrastructure; gate remains off until 24 matured prospective months |
| Composite target Path B evaluation | v125: empirical head-to-head vs Path A |

## Development Principles

- Never finalize a module without a passing pytest suite.
- No K-Fold cross-validation - `TimeSeriesSplit` with purge/embargo only.
- No `StandardScaler` across the full dataset prior to temporal splitting.
- No `yfinance` for fundamentals or historical ratios.
- Python 3.10+, strict PEP 8, standard type hints.

## Monthly Decision Log

See [`results/monthly_decisions/decision_log.md`](results/monthly_decisions/decision_log.md)
for the persistent record of monthly recommendations.
