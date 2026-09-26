# V164 Closeout And Handoff

Created: 2026-04-18

## Completed Block

`v160-v164` completed the research-only technical-analysis feature arc. It
archived the source reports, pre-registered the broad-but-pruned Alpha Vantage
indicator universe, implemented pure pandas/numpy TA feature construction, ran
the broad screen, produced capped survivor confirmation artifacts, and wrote a
go/no-go synthesis.

## Final Outcome

Recommendation: `replacement_candidate`

No production or shadow behavior changed. The empirical result is strong enough
to justify a later shadow-only replacement plan, not a direct production
promotion.

## Key Artifacts

- Reports: `docs/archive/history/v160-ta-research-reports/`
- Plan:
  `docs/superpowers/plans/2026-04-18-v160-v164-technical-analysis-feature-research.md`
- Feature factory: `src/research/v160_ta_features.py`
- Broad screen: `results/research/v162_ta_broad_screen.py`
- Survivor confirmation: `results/research/v163_ta_survivor_confirm.py`
- Empirical outputs:
  - `results/research/v162_ta_broad_screen_detail.csv`
  - `results/research/v162_ta_broad_screen_summary.csv`
  - `results/research/v163_ta_survivor_confirm_summary.csv`
  - `results/research/v163_ta_survivor_candidate.json`
  - `results/research/v164_ta_synthesis_summary.md`
  - `results/research/v164_ta_candidate.json`

## Survivor Summary

The v163 confirmation step prioritized the pre-registered primary target:
binary PGR-vs-benchmark outperformance classification. The strongest
replacement candidates were:

- `ta_pgr_obv_detrended` replacing `mom_12m`
- `ta_pgr_natr_63d` replacing `vol_63d`
- one representative ratio Bollinger signal, with `ta_ratio_bb_pct_b_6m_vwo`
  the cleanest first candidate

The broader Bollinger-width variants were interesting but should be treated as
diagnostic alternatives until one representative feature proves robust in a
prediction-level shadow run.

## API And Workflow Note

The existing database snapshot already contained the required PGR, benchmark,
and peer daily/monthly history for this research pass. No new Alpha Vantage
workflow schedule was added. If future backfill is needed, avoid collisions
with the existing Friday weekly price fetch, Sunday peer fetch, and monthly
8-K/decision workflows.

## Recommended Next Queue

1. Create a new shadow-only classification replacement plan.
2. Test a minimal feature swap candidate:
   `mom_12m -> ta_pgr_obv_detrended` and `vol_63d -> ta_pgr_natr_63d`.
3. Compare exactly one ratio Bollinger representative against that minimal
   candidate.
4. Require prediction-level WFO diagnostics, regime slices, correlation
   pruning, and reporting-only monthly shadow output before any production
   discussion.

## Verification

```bash
python -m pytest tests/test_research_v160_ta_features.py -q --tb=short
python -m pytest tests/test_research_v162_ta_screen.py -q --tb=short
python -m pytest tests/test_research_v163_ta_confirm.py -q --tb=short
python -m pytest tests/test_feature_engineering.py tests/test_wfo_engine.py tests/test_path_b_classifier.py -q --tb=short
```
