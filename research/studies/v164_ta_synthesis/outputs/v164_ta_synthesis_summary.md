# v164 Technical Analysis Synthesis Summary

Created: 2026-04-18

## Status

`v160-v164` is complete as a research-only technical-analysis arc. It does
not change production recommendation logic, live monthly decision behavior, or
shadow reporting.

The available database snapshot already covered the required PGR, benchmark,
and peer price histories through April 2026. No additional Alpha Vantage
backfill workflow was needed for this empirical pass.

## Empirical Artifacts

- `results/research/v162_ta_broad_screen_detail.csv`
- `results/research/v162_ta_broad_screen_summary.csv`
- `results/research/v162_ta_broad_screen_inventory.json`
- `results/research/v163_ta_survivor_confirm_summary.csv`
- `results/research/v163_ta_survivor_candidate.json`

The first v162 run completed the model-evaluation grid, but exposed an
artifact-shaping bug: baseline rows were duplicated before baseline-delta
attachment. The harness now supports deterministic detail compaction and
resummary, producing a compact corrected detail artifact from the completed
evaluation grid.

## Outcome

Recommendation: `replacement_candidate`

Interpretation: TA should not be added wholesale. The only candidates worth
carrying forward are replacement-style candidates that may substitute for
existing momentum, volatility, or VIX-style baseline inputs in the
classification path.

Primary survivor set:

| Feature | Replacement Mode | Mean BA Delta | Mean Brier Delta | Positive Benchmarks |
| --- | --- | ---: | ---: | ---: |
| `ta_pgr_obv_detrended` | `replace_mom_12m` | 0.0377 | -0.0292 | 8 |
| `ta_pgr_natr_63d` | `replace_vol_63d` | 0.0263 | -0.0316 | 8 |
| `ta_ratio_bb_pct_b_6m_vwo` | `replace_vol_63d` | 0.0147 | -0.0292 | 8 |
| `ta_ratio_bb_width_6m_vde` | `replace_vix` | 0.0388 | -0.0133 | 7 |
| `ta_ratio_bb_width_6m_voo` | `replace_vix` | 0.0311 | -0.0117 | 7 |
| `ta_ratio_bb_width_6m_vxus` | `replace_vol_63d` | 0.0301 | -0.0110 | 7 |

Regression results were directionally interesting but should not drive the
next step alone. The most consistent regression gains were in shallow GBT
replacement tests, especially `ta_pgr_obv_detrended`,
`ta_ratio_bb_pct_b_6m_vwo`, `ta_ratio_bb_pct_b_6m_bnd`, and
`ta_ratio_rsi_6m_bnd`. Because the pre-registered primary target was
classification, v163 promotes the classification survivor set.

## Complexity Control

The empirical signal favors replacement over addition. The next plan should
avoid expanding the production feature count and should instead test a narrow
classification shadow candidate:

- Replace `mom_12m` with `ta_pgr_obv_detrended`.
- Replace `vol_63d` with `ta_pgr_natr_63d`.
- Test one representative Bollinger ratio feature before considering multiple
  benchmark-specific Bollinger width/%B variants.

## Next Direction

Recommended next project direction: create a separate shadow-only plan for a
small classification replacement candidate. Do not promote TA features directly
to production.

Suggested scope for that next arc:

1. Build a named classification candidate that swaps in `ta_pgr_obv_detrended`
   and `ta_pgr_natr_63d`.
2. Compare one VWO `%B` ratio feature against one width feature family rather
   than carrying all Bollinger variants forward.
3. Run prediction-level WFO diagnostics, regime slices, and monthly
   reporting-only shadow output.
4. Promote only if the replacement candidate preserves or improves robustness
   without increasing feature count.
