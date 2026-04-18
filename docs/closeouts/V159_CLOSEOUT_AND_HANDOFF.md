# V159 Closeout And Handoff

Created: 2026-04-18

## Completed Block

`v159` closes the Firth shadow integration cycle. The v154 Firth logistic research
winner (VMBS +0.0412, BND +0.0704 BA_covered) is now surfaced as a reporting-only
shadow lane `firth_shadow_v159` alongside the existing `autoresearch_followon_v150`
lane in monthly artifacts.

## Final Outcomes

- New reporting module: `src/reporting/firth_shadow.py`
- Monthly pipeline now produces three shadow variants:
  1. `baseline_shadow` — standard Path A per-benchmark logistic
  2. `autoresearch_followon_v150` — v139-v150 regression research winners
  3. `firth_shadow_v159` — Firth logistic research findings for VMBS and BND
- No production config changes; live recommendation path unchanged
- 6 new tests in `tests/test_firth_shadow.py`; all pass

## Promotion Boundaries

- Production: no change to live monthly recommendation
- Shadow: `firth_shadow_v159` surfaces in monthly `classification_shadow.csv`
  and `monthly_summary.json` as a reporting-only third variant
- Research-only: `results/research/v154_firth_candidate.json` remains the source

## Recommended Next Queue

1. `BL-01` — Black-Litterman tau/view tuning (next open item)
2. `CLS-03` — Path A vs Path B production decision (still time-locked)
3. Future: promote Firth for VMBS/BND in production Path A after 24 prospective months

## Verification

```bash
python -m pytest tests/test_firth_shadow.py \
                 tests/test_shadow_followon.py \
                 tests/test_monthly_pipeline_e2e.py \
                 -q --tb=short
```

## Exact Next Commands

```bash
git checkout master
git pull --ff-only origin master
git checkout -b codex/bl01-black-litterman-tuning
# implement BL-01: Black-Litterman tau/view tuning
```
