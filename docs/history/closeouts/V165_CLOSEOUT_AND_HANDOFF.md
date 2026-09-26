# V165 Closeout And Handoff

Created: 2026-04-18

## Completed Block

`v165` implements the first follow-through from the `v164` TA
`replacement_candidate` recommendation. It remains research-only and
reporting-only: no production recommendation path, live monthly decision logic,
or classifier gate behavior changed.

## Scope

The harness tests replacement-style classifier variants against the existing
lean baseline:

- `ta_obv_replaces_mom12`
- `ta_natr_replaces_vol63`
- `ta_minimal_replacement`
- `ta_minimal_plus_vwo_pct_b`

All variants keep the same 12-feature count as the lean baseline. There is no
"all TA features" model.

## Artifacts

- `results/research/v165_ta_shadow_replacement_eval.py`
- `results/research/v165_ta_shadow_replacement_detail.csv`
- `results/research/v165_ta_shadow_replacement_predictions.csv`
- `results/research/v165_ta_shadow_replacement_summary.csv`
- `results/research/v165_ta_shadow_replacement_regime_slices.csv`
- `results/research/v165_ta_shadow_current.csv`
- `results/research/v165_ta_shadow_current_summary.csv`
- `results/research/v165_ta_shadow_candidate.json`

## Outcome

Recommendation: `shadow_monitor`

The strongest historical candidate is `ta_minimal_plus_vwo_pct_b`:

- Mean balanced accuracy: `0.6247`
- Mean Brier score: `0.2348`
- Mean balanced-accuracy delta versus lean baseline: `+0.0584`
- Mean Brier delta versus lean baseline: `-0.0656`
- Positive benchmarks: `8 / 8`
- Degraded benchmarks: `0 / 8`

The simpler `ta_minimal_replacement` candidate also cleared the historical
screen:

- Mean balanced-accuracy delta: `+0.0460`
- Mean Brier delta: `-0.0517`
- Positive benchmarks: `8 / 8`
- Degraded benchmarks: `0 / 8`

## Current Snapshot

The current artifact uses `data_as_of = 2026-04-17` and retains
`feature_anchor_date = 2026-04-30` because the feature matrix is month-end
indexed while April is still in progress.

Current equal-weight P(Actionable Sell):

- Lean baseline: `30.7%` (`NEUTRAL`)
- `ta_minimal_replacement`: `39.2%` (`NEUTRAL`)
- `ta_minimal_plus_vwo_pct_b`: `44.1%` (`NEUTRAL`)

The TA replacement candidates are more sell-aware than the lean baseline in
the current snapshot, but still remain neutral.

## Next Direction

Recommended next step: add a reporting-only monthly artifact lane for
`ta_minimal_plus_vwo_pct_b`, with `ta_minimal_replacement` retained as a
simpler comparison row. The lane should append prospective monthly
probabilities to history, but it must not affect production recommendations,
sell percentages, or the classifier gate overlay.

Promotion discussion remains blocked until prospective history matures.

## Verification

```bash
python -m pytest tests/test_research_v165_ta_shadow_replacement.py -q --tb=short
python results/research/v165_ta_shadow_replacement_eval.py
python -m ruff check results/research/v165_ta_shadow_replacement_eval.py tests/test_research_v165_ta_shadow_replacement.py
```
