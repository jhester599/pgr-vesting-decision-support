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
- first defaulted production to `live_with_shadow`
- added a simpler-baseline cross-check to the monthly report and email
- added explicit existing-holdings lot guidance to the monthly report
- added explicit diversification-first redeploy guidance to the monthly report and email
- updated the report/email versioning and wording for the v13 recommendation-layer pilot
- promoted the safer recommendation-layer default to `shadow_promoted` after the
  v12/v13 shadow and production-UX validation
- kept the live 4-model stack intact as a diagnostic cross-check rather than
  changing the underlying prediction engine

## Active v13 Default

- simpler diversification-first baseline is primary
- live model stack is shown as a cross-check
- existing shares guidance is surfaced in the report
- redeploy guidance prefers diversification away from PGR-like funds
