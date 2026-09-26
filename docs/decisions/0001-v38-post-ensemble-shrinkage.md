# 0001 — Post-ensemble shrinkage (v38)

| | |
|---|---|
| **Status** | Accepted, live. Amended by [0006](0006-validation-gates-and-cpcv-diagnostic.md): alpha chosen prequentially since 2026-09-25 |
| **Date** | 2026-04-10 (v37–v60 cycle) |
| **Where it lives** | `config.model.ENSEMBLE_SHRINKAGE_ALPHA_GRID`; `src/models/prequential.py` |
| **Study** | `research/studies/v38_shrinkage/` (registry id `v38`, status `promoted`) |

## Context

The v37–v60 research cycle tried 24 variations on the Ridge + GBT ensemble
(regularisation, winsorising, expanding windows, PCA, Bayesian ridge,
classification sidecars, composite benchmarks, panel pooling, regime features,
GPR, rank targets and more), each under the same walk-forward validation.
Every variant still had a negative pooled OOS R², so the question was which
change made the forecasts least wrong without hurting IC or hit rate.

## Decision

Shrink the ensemble prediction towards zero after aggregation:
`y_hat_shrunk = alpha * y_hat`. v38 was adopted as the production and research
baseline; every later candidate had to beat it.

## Evidence

Pooled OOS R² of the top results (v37–v60 summary):

| Variant | Pooled OOS R² |
|---|---:|
| v38 shrinkage, alpha 0.50 | −0.1310 |
| v37 baseline | −0.2269 |
| v50 clip + shrink | −0.2300 |
| v40 constrained GBT | −0.2355 |

v38 was the only phase-1 winner, and the only variant that improved pooled
OOS R² materially without harming IC or hit rate. v140 (2026-04-16) found the
result flat across the bounded alpha range and kept 0.50.

## Consequences

- The fixed alpha 0.50 was chosen on the full sample. Review 2026-09-25 (F13)
  found that this inflated the reported health, so since step 5 production
  re-chooses alpha every month from the grid, using only realised OOS errors
  ([0006](0006-validation-gates-and-cpcv-diagnostic.md)). The fixed
  `ENSEMBLE_PREDICTION_SHRINKAGE_ALPHA = 0.50` is research-only.
- A re-run of v38 on the corrected data is pending (review WP11).

## Sources

- [v37–v60 results summary](../history/superpowers/plans/2026-04-10-v37-v60-results-summary.md)
- [v38 study README](../../research/studies/v38_shrinkage/README.md)
- [V152 closeout](../history/closeouts/V152_CLOSEOUT_AND_HANDOFF.md) (v140 shrinkage re-check)
