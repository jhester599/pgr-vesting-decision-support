# v118-v121 Prospective Shadow Monitoring

## Purpose

This phase answers the next practical question after `v102-v117`:

- can the selected classifier gate candidate be evaluated as if it had been
  monitored prospectively against the live regression-led baseline?

The goal is not immediate production promotion. The goal is to build a clean,
versioned replay that separates:

- historical all-month uplift
- disagreement-month behavior
- promotion readiness versus real-time shadow-monitoring readiness

## Version Scope

### v118 - Prospective Shadow Replay

- replay the selected `v113` candidate month by month against the current live
  regression baseline
- persist a detailed monthly table with live and shadow hold fractions,
  recommendation-mode proxies, would-change flags, and rolling shadow-minus-live
  metrics

### v119 - Disagreement Scorecard

- summarize the replay with special focus on disagreement months
- answer whether the candidate helped when it actually differed from live

### v120 - Prospective Gate Assessment

- turn the replay and scorecard into a promotion-style decision
- distinguish:
  - ready for real-time shadow monitoring
  - ready for limited production-gate consideration
  - continue offline research only

### v121 - Phase Summary

- consolidate the phase into one short artifact for future reference

## Guardrails

- keep the current production recommendation path unchanged
- use only time-safe historical sequences already produced by the research stack
- do not treat simulated prospective replay as equivalent to real future months
- require matured live monitoring history before any actual classifier-gate promotion

## Expected Artifacts

- `results/research/v118_prospective_shadow_replay_results.csv`
- `results/research/v119_disagreement_scorecard_results.csv`
- `results/research/v120_prospective_shadow_gate_assessment_results.csv`
- `results/research/v121_prospective_shadow_phase_summary_results.csv`
- matching markdown summaries
