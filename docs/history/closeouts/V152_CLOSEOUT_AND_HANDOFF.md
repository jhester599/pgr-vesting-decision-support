# V152 Closeout And Handoff

Created: 2026-04-17

## Completed Block

`v152` closes the `v139-v152` autoresearch follow-on cycle after the merge of
PR 80.

This final pass does not add new model behavior. It records what changed,
clarifies what did not change, and gives the next autonomous session one clean
restart point.

## Final Outcomes

- Shadow-only promotion outcome:
  - `v151` added the additive reporting lane
    `autoresearch_followon_v150`
  - this lane surfaces the surviving follow-on winners side-by-side with the
    current shadow outputs
  - it does not alter the live recommendation path
- Surviving research winners:
  - `results/research/v141_blend_weight_candidate.txt = 0.60`
  - `results/research/v143_corr_prune_candidate.txt = 0.80`
  - `results/research/v144_conformal_candidate.json = {"coverage": 0.75, "aci_gamma": 0.03}`
  - `results/research/v149_kelly_candidate.json = {"fraction": 0.50, "cap": 0.25}`
- Confirmed no-change follow-through results:
  - `v140` shrinkage stays `0.50`
  - `v142` EDGAR lag stays `2`
  - `v145` WFO window stays `{"train": 60, "test": 6}`
  - `v146` threshold pair stays `{"low": 0.15, "high": 0.70}`
  - `v147` Path B multiplier stays `1.0`
  - `v148` positive class weight stays `1.0`
  - `v150` neutral band stays `0.015`

## Promotion Boundaries

- Production:
  - no `v139-v152` change is promoted directly into the live monthly decision
    path
- Shadow:
  - the `v141`, `v143`, `v144`, and `v149` winners are exposed only through the
    side-by-side reporting lane added in `v151`
- Research-only:
  - all bounded search harnesses, logs, and candidate files under
    `results/research/`

## Recommended Next Queue

1. `BL-01` Black-Litterman tau/view tuning
   - best next decision-layer task now that the Kelly and neutral-band replay
     proxies are settled
2. `CLS-02` Firth logistic for short-history benchmarks
   - best next classifier-stability task that is not blocked on time
3. `FEAT-01` DTWEXBGS post-v128 feature search
   - small, contained benchmark-specific feature follow-up
4. `FEAT-02` WTI 3M momentum for DBC/VDE
   - still promising, but only after a clean external-series verification pass

Keep `CLS-03` blocked on the matured prospective-month gate. Keep `REG-02`
deferred until a later plan shows a stronger ensemble-level reason to reopen
the GBT line.

## Verification

- `git show --stat --summary f692b20ec11a125fba7d3fb38685c38357dd99ff`
- `python -m pytest tests/test_shadow_followon.py tests/test_monthly_pipeline_e2e.py -q --tb=short`

## Exact Next Commands

```bash
git checkout master
git pull --ff-only origin master
git checkout -b codex/<next-cycle-name>
python -m pytest tests/test_shadow_followon.py tests/test_monthly_pipeline_e2e.py -q --tb=short
# then write the next execution plan from docs/research/backlog.md, starting with BL-01 unless priorities change
```
