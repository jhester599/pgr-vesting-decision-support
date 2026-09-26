# Shadow Follow-On v150 Side-By-Side Design

## Summary

This design adds one new experimental shadow lane to the monthly reporting
pipeline so the winning `v139-v150` autoresearch candidates can be monitored
side-by-side with the current shadow outputs. The new lane is reporting-only:
it must not alter the live recommendation path, sell percentage, or the
existing shadow defaults that are already present in monthly artifacts.

The new lane will be named `autoresearch_followon_v150`. It packages the
surviving bounded-search winners into one comparison bundle:

- regression-side candidate assumptions:
  - `v141` fixed Ridge-vs-GBT blend weight `0.60`
  - `v143` correlation-pruning threshold `0.80`
- confidence / calibration assumptions:
  - `v144` conformal replay candidate
    `{"coverage": 0.75, "aci_gamma": 0.03}`
- decision-layer assumptions:
  - `v149` Kelly replay candidate
    `{"fraction": 0.50, "cap": 0.25}`
- neutral-band policy remains the incumbent `v150` setting `0.015`

## Goal

Expose the promoted follow-on research candidate as a side-by-side shadow
artifact so monthly runs can compare the current shadow interpretation layer
against the newer autoresearch bundle without changing production behavior.

## Non-Goals

- No promotion into the live monthly decision path
- No replacement of the current shadow classifier, shadow gate overlay, or
  existing dashboard cards
- No generic multi-variant framework for arbitrary future shadow lanes
- No change to the current `v131` Path B shadow adoption basis

## Current Context

Today the repo already supports one shadow interpretation layer and a few
related diagnostic artifacts:

- `src/models/classification_shadow.py` builds the current compact shadow
  summary used in reporting
- `src/reporting/monthly_summary.py` serializes a single
  `classification_shadow` payload plus a single `shadow_gate_overlay`
- `src/reporting/dashboard_snapshot.py` renders one “Classification Confidence
  Check” section and one “Agreement Panel”
- `src/reporting/classification_artifacts.py` writes one
  `classification_shadow.csv` and one `decision_overlays.csv`
- `scripts/monthly_decision.py` is the orchestration point that assembles those
  pieces into monthly outputs

That existing path should remain the canonical baseline lane.

## Proposed Design

### 1. Add one named side-by-side shadow variant

Introduce a second reporting payload representing the
`autoresearch_followon_v150` lane. This payload should be parallel to the
current shadow payload, not a replacement for it.

At the data-shape level, the monthly pipeline should be able to carry:

- the existing baseline shadow summary
- one follow-on shadow summary
- the existing shadow gate overlay
- one follow-on decision-layer summary derived from the `v149` candidate

The most important property is explicit naming. Every emitted structure should
identify whether it belongs to:

- `baseline_shadow`
- `autoresearch_followon_v150`

### 2. Keep the lane reporting-only

The follow-on lane should never feed back into:

- the live recommendation mode
- `recommended_sell_pct`
- recommendation headline text
- current promoted classifier/shadow gate logic

It is strictly a comparison lane for operators and future promotion decisions.

### 3. Surface both lanes in existing artifacts

The side-by-side presentation should appear in the same artifacts the user
already reviews each month, rather than creating an entirely separate output
tree.

#### `monthly_summary.json`

Add a new top-level structure for shadow comparisons, for example:

- `classification_shadow_variants`
- `decision_overlay_variants`

The current `classification_shadow` and `shadow_gate_overlay` fields may remain
for backward compatibility, but the new variant-aware structures should carry
both lanes with explicit names and labels.

#### `recommendation.md`

Add a compact comparison subsection that shows:

- baseline shadow snapshot
- follow-on shadow snapshot
- key deltas:
  - actionable-sell probability
  - investable-pool probability if available
  - Path B probability if available
  - Kelly utility / coverage / success-rate proxy where relevant

The tone should make clear that the follow-on lane is experimental.

#### `dashboard.html`

Extend the classification section into a side-by-side comparison card layout.
The page should show the baseline lane and the follow-on lane next to each
other, using the existing dashboard style.

#### CSV artifacts

Prefer additive compatibility over replacement:

- keep `classification_shadow.csv`
- keep `decision_overlays.csv`
- either add a `variant` column to both outputs or introduce a new companion
  comparison CSV if that proves less invasive

The preferred path is to add a `variant` column because it scales better while
remaining easy to filter.

### 4. Compute the follow-on lane from preserved candidate files

The follow-on lane should not hardcode ad hoc values in multiple places.
Instead, it should read from the existing research candidate artifacts:

- `results/research/v141_blend_weight_candidate.txt`
- `results/research/v143_corr_prune_candidate.txt`
- `results/research/v144_conformal_candidate.json`
- `results/research/v149_kelly_candidate.json`
- `results/research/v150_neutral_band_candidate.txt`

This keeps the shadow lane aligned with the recorded research winners and makes
future documentation clearer.

### 5. Minimal implementation strategy

Do not attempt to rebuild the entire monthly model stack around a generalized
variant engine. The lower-risk approach is:

1. add a small helper module that loads the follow-on candidate files and
   produces a compact variant payload
2. thread that payload through the monthly reporting pipeline
3. update report writers to render both lanes side-by-side

This design stays narrow and avoids destabilizing the main monthly-decision
pipeline.

## Data Shape

Recommended new payload shape:

```json
{
  "classification_shadow_variants": [
    {
      "variant": "baseline_shadow",
      "label": "Current Shadow",
      "enabled": true,
      "probability_actionable_sell": 0.534,
      "confidence_tier": "LOW",
      "stance": "NEUTRAL"
    },
    {
      "variant": "autoresearch_followon_v150",
      "label": "Autoresearch Follow-On",
      "enabled": true,
      "probability_actionable_sell": 0.561,
      "confidence_tier": "LOW",
      "stance": "NEUTRAL",
      "candidate_sources": {
        "v141_blend_weight": "0.60",
        "v143_corr_prune": "0.80",
        "v144_conformal": {"coverage": 0.75, "aci_gamma": 0.03},
        "v149_kelly": {"fraction": 0.50, "cap": 0.25},
        "v150_neutral_band": "0.015"
      }
    }
  ]
}
```

The exact schema can be adapted to current repo conventions, but the key
requirement is variant identity plus side-by-side comparability.

## File Responsibilities

- `src/models/classification_shadow.py`
  - remains the source of the baseline shadow summary
  - may expose shared formatting/helpers used by both variants
- new helper module, likely under `src/models/` or `src/reporting/`
  - load follow-on candidate files
  - build the follow-on variant payload
- `scripts/monthly_decision.py`
  - orchestrate baseline + follow-on shadow payload assembly
- `src/reporting/monthly_summary.py`
  - serialize both variants in machine-readable form
- `src/reporting/dashboard_snapshot.py`
  - render side-by-side cards/sections
- `src/reporting/classification_artifacts.py`
  - write variant-aware CSV artifacts

## Testing Strategy

### Unit tests

- candidate-file loader test:
  - verify each required follow-on candidate file is parsed correctly
- follow-on payload builder test:
  - verify the variant name, label, and candidate-source metadata
- artifact writer tests:
  - verify variant-aware CSV outputs include both baseline and follow-on rows
- summary serialization tests:
  - verify `monthly_summary.json` includes both shadow variants

### Integration tests

- monthly reporting smoke test:
  - run the relevant monthly-summary/reporting builders on a fixture payload
  - verify the baseline lane is unchanged
  - verify the follow-on lane appears side-by-side

### Regression safety

- no change to live recommendation fields
- no removal or renaming of legacy baseline shadow fields without compatibility
- no production config mutation

## Risks And Mitigations

### Risk: confusing users with too many shadow numbers

Mitigation:
- keep one extra lane only
- label it clearly as experimental
- render short delta-focused summaries instead of duplicating entire report
  sections verbatim

### Risk: accidental coupling to live recommendation logic

Mitigation:
- compute the follow-on lane in a separate helper path
- add tests asserting live recommendation mode and sell percentage remain
  unchanged when the comparison lane is present

### Risk: brittle report consumers

Mitigation:
- preserve existing baseline fields in `monthly_summary.json`
- add new variant-aware fields rather than replacing old ones in one step

## Open Decision Resolved

This design chooses side-by-side comparison rather than replacement. The
current shadow defaults remain the baseline, and the follow-on lane is a
comparison surface for future shadow monitoring and possible later promotion.
