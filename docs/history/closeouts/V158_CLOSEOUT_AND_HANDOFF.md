# V158 Closeout And Handoff

Created: 2026-04-17

## Completed Block

`v158` closes the `v153-v158` classification and feature research cycle, initiated
from the 2026-04-17 ChatGPT peer review archived at
`docs/archive/history/repo-peer-reviews/2026-04-17/`.

## Final Outcomes

See `results/research/v158_synthesis_summary.md` for the full table.

- CLS-02 (Firth logistic): **adopted for VMBS (+0.0412) and BND (+0.0704)**
- FEAT-02 (WTI momentum): no benefit — DBC +0.0051, VDE +0.0206 (threshold 0.04)
- FEAT-01 (USD momentum): no benefit — BND -0.0767, VXUS 0.0000, VWO +0.0087
- FEAT-03 (term premium diff): no benefit — best VDE +0.0169 (threshold 0.02)

## Promotion Boundaries

- Production: no v153-v158 change promotes to live monthly decision path
- Shadow: Firth logistic for VMBS and BND is the sole candidate for the next shadow lane
- Research-only: all harnesses, logs, and candidate files under `results/research/`

## Recommended Next Queue

1. `v159` — Wire Firth logistic for VMBS/BND into the shadow classification lane
   (follow `autoresearch_followon_v150` pattern in `src/reporting/shadow_followon.py`)
2. `BL-01` — Black-Litterman tau/view tuning (now unblocked; highest remaining
   decision-layer task)
3. `CLS-03` — Path A vs Path B production decision (still time-locked)

## Verification

```bash
python -m pytest tests/test_research_v154_firth_logistic_eval.py \
                 tests/test_research_v155_wti_momentum_eval.py \
                 tests/test_research_v156_usd_momentum_eval.py \
                 tests/test_research_v157_term_premium_eval.py \
                 -q --tb=short -k "not slow"
```

## Exact Next Commands

```bash
git checkout master
git pull --ff-only origin master
git checkout -b codex/v159-firth-shadow-integration
python -m pytest tests/test_research_v154_firth_logistic_eval.py -q --tb=short -k "not slow"
# implement Firth shadow lane for VMBS and BND
```
