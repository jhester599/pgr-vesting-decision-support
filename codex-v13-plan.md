# codex-v13-plan.md

## Scope

- implement a limited promotion study for the simpler diversification-first recommendation layer
- keep the live 4-model production stack unchanged
- ship the safest v11-v12 usefulness improvements for the upcoming monthly run

## Implemented Work

- added a production-safe recommendation-layer mode switch:
  - `live_only`
  - `live_with_shadow`
  - `shadow_promoted`
- defaulted production to `live_with_shadow`
- added a simpler-baseline cross-check to the monthly report and email
- added explicit existing-holdings lot guidance to the monthly report
- added explicit diversification-first redeploy guidance to the monthly report and email
- updated the report/email versioning and wording for the v13 recommendation-layer pilot

## Active v13 Default

- live model stack remains primary
- simpler baseline is shown as a cross-check
- existing shares guidance is surfaced in the report
- redeploy guidance prefers diversification away from PGR-like funds
