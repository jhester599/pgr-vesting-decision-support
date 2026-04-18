# BL-01 Closeout And Handoff

Created: 2026-04-18

## Completed Block

`BL-01` closes the Black-Litterman tau/view tuning research cycle. A 5×5 Monte Carlo
sweep (50 scenarios × 25 parameter combinations) evaluated whether updating the BL
prior trust parameter (tau) and risk aversion coefficient improves IC-rank-weighted
portfolio quality.

## Final Outcomes

- Harness: `results/research/bl01_tau_sweep_eval.py`
- Candidate: `results/research/bl01_tau_candidate.json`
- Recommendation: **keep_incumbent** — tau=0.05, risk_aversion=2.5 retained
- Incumbent rank_corr: 0.8643 (Spearman correlation between BL weights and IC values)
- Best alternative: tau=0.05, risk_aversion=1.5 → rank_corr=0.8733, delta=+0.009
- Delta +0.009 is below the 0.05 win threshold — no config change warranted
- quality_filter_bypassed: False — all candidates had fallback_rate < 50%
- `config/model.py` unchanged: BL_TAU=0.05, BL_RISK_AVERSION=2.5

## Key Finding

The current defaults are near-optimal under the IC-rank-correlation scoring metric.
Lower risk_aversion (1.5 vs 2.5) shows a marginal improvement (+0.009) by concentrating
more weight in high-IC benchmarks, but the gap is too small to justify a config change.
The BL model's signal-utilisation quality (rank_corr ≈ 0.86) is already high; further
tuning would require a more sensitive discriminating metric or a materially changed signal
environment.

## Side Effect: Production Fix

As part of BL-01, `build_bl_weights` in `src/portfolio/black_litterman.py` received a
new `risk_free_rate: float | None = None` parameter (default 0.04). This allows callers
to specify the risk-free rate in the same time units as their returns DataFrame, avoiding
a latent annual-vs-monthly unit mismatch that would cause 100% fallback in synthetic
sweep scenarios. Production callers that omit the argument continue to use 0.04.

## Promotion Boundaries

- Production: no change to live monthly recommendation path
- Config: BL_TAU and BL_RISK_AVERSION unchanged
- BL remains a shadow-only diagnostic; this research provides a baseline for future
  re-evaluation if signal quality changes materially

## Recommended Next Queue

1. `CLS-03` — Path A vs Path B production decision (time-locked on 24 matured months)
2. `CLS-01` — SCHD per-benchmark classifier addition (depends on CLS-03)

## Verification

```bash
python -m pytest tests/test_black_litterman.py tests/test_bl_fallback_monthly.py tests/test_bl01_sweep.py -q --tb=short
```
