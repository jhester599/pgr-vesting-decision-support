# X-Series Resume Summary (2026-04-24)

## Purpose

This document is the restart point for the separate x-series PGR
research lane. It summarizes x1 through x24, records the current repo
state, links useful peer-review prompts, and recommends the next
research steps.

## Current Repo State

- Current working branch when this document was written: `codex/x24-indicator-contract-summary`.
- Current branch HEAD: `49951e2f1b21227fb929e3610be499939fa9e6d2`.
- `origin/master` HEAD at write time: `dbf6a9ac6932025eb0742fdffe7cbe6d6c7c21b8`.
- Important: the x17-x24 chain currently lives on the stacked branch
  lineage above `codex/x17-persistent-bvps` and is not yet reflected in
  `master` in this local/repo state. Before future x-series work, verify
  whether the latest stacked branch has been merged to `master` or open a
  catch-up PR directly to `master`.

## X1-X24 Summary

### x1-x6: Lane Setup And First Baselines

- `x1`: created the separate x-series lane, target utilities, feature
  inventory, and data sufficiency audit. Key constraint: only 22 annual
  special-dividend snapshots.
- `x2`: multi-horizon absolute direction classification baseline. Result:
  did not clear the base-rate gate.
- `x3`: direct forward-return benchmark. Result: mostly baseline-heavy;
  only the 12m drift path clearly cleared the no-change gate.
- `x4`: BVPS forecasting leg. Result: strongest early lane; beat no-change
  BVPS across all four horizons.
- `x5`: BVPS x P/B decomposition benchmark. Result: useful structurally,
  but the stable anchor remained `no_change_pb`.
- `x6`: annual special-dividend two-stage sidecar. Result: low-confidence,
  small-sample annual research only.

### x7-x12: Targeted Follow-Up And Synthesis

- `x7`: targeted TA replacement follow-up. Result: `ta_minimal_plus_vwo_pct_b`
  cleared 2/4 horizons, but broad TA expansion was not justified.
- `x8`: cross-lane synthesis. Result: shadow readiness remained `not_ready`.
- `x9`: BVPS bridge features, interactions, and stronger baselines.
  Result: improved BVPS at 1m and 3m, but not 6m/12m.
- `x10`: capital-enhanced dividend lane using x9 features. Result:
  improved x6 EV MAE but remained low-confidence.
- `x11`: synthesis of x9/x10. Result: `continue_research`.
- `x12`: dividend-adjusted BVPS target audit. Result: adjustment helped
  3m/6m, but not 1m/12m.

### x13-x17: Structural Packaging And Persistent BVPS

- `x13`: adjusted decomposition follow-up. Result: only the 6m adjusted
  structural path clearly survived.
- `x14`: indicator synthesis. Result: narrowed to one bounded 6m
  structural candidate.
- `x15`: P/B regime overlay research. Result: no overlay beat the
  no-change P/B anchor.
- `x16`: packaged the structural indicator `adjusted_structural_bvps_pb_6m`.
- `x17`: persistent BVPS research. Result: persistent BVPS helped medium
  horizons (3m/6m), supporting separation of capital creation from payout policy.

### x18-x23: Dividend Policy Rebuild

- `x18`: rebuilt dividend labels around the December 2018 policy break and
  a December-February payout window.
- `x19`: post-policy-only dividend model using persistent-BVPS capital
  features. Result: better than x10 on the overlapping post-policy years,
  but the sample was only 3 OOS years.
- `x20`: synthesized x19 vs x10. Result: occurrence is one-class on the
  overlap sample, so only size-target experiments are identifiable.
- `x21`: target-scale comparison for dividend size. Result:
  `special_dividend_excess / current_bvps` was best.
- `x22`: baseline challenge on dividend size. Result: the
  `to_current_bvps` size target survived baseline challengers.
- `x23`: packaged the dividend lane. Result: `research_size_indicator_candidate` with occurrence still `underidentified_post_policy`.

### x24: Research Indicator Bundle

- `x24` bundles `adjusted_structural_bvps_pb_6m` and
  `special_dividend_size_watch` into one research-only
  indicator contract for future monthly-report/dashboard discussion.

## Strongest Findings

- BVPS forecasting is still the strongest core x-series leg.
- The structural 6m path is the cleanest price-adjacent packaged signal,
  but it still depends on `no_change_pb` as the valuation anchor.
- The dividend lane became much clearer after the policy-regime rebuild:
  occurrence is still underidentified post-policy, but dividend size has a
  credible research-only path via `special_dividend_excess / current_bvps`.
- The x-series remains research-only overall; it is not ready for shadow or production wiring.

## What Did Not Work

- Broad absolute classification did not establish a robust edge over base rate.
- Direct return prediction was mostly baseline-heavy.
- P/B overlays and regime tricks did not beat the plain no-change anchor.
- Post-policy dividend occurrence currently has too little label variation to support a practical classifier.

## Surviving Packaged Signals

- Structural watch: `adjusted_structural_bvps_pb_6m`
  (`ridge_bridge__no_change_pb`, 6m).
- Dividend watch: `special_dividend_size_watch`
  (`to_current_bvps`, November month-end annual snapshot).

## Recommended Next Steps

1. Merge/catch up the stacked x17-x24 branch chain to `master` cleanly.
2. Start `x25` as a research-only monthly indicator contract/output for the
   x24 bundle, without touching production or shadow artifacts.
3. Start `x26` as a structural P/B revisit focused on disciplined
   insurer-specific valuation features from existing data.
4. Keep the dividend lane annual and size-focused until occurrence gains
   meaningful post-policy label variation.
5. Use the peer-review prompts below before any attempt to promote the x-series into a reporting surface.

## Peer Review / Deep Research Prompts

- Dividend lane prompt: [x23 peer review](/C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/x23_peer_review_prompt.md)
- Holistic bundle prompt: [x24 bundle peer review](/C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/x24_bundle_peer_review_prompt.md)
- Structural P/B prompt: [x24 structural peer review](/C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/x24_structural_peer_review_prompt.md)
- Earlier structural prompt: [x16 peer review](/C:/Users/Jeff/Documents/pgr-vesting-decision-support/results/research/x16_peer_review_prompt.md)

## Resume Advice

When restarting later, read this file first, then `x24_research_memo.md`,
`x23_research_memo.md`, `x16_research_memo.md`, and verify whether the
x17-x24 chain is on `master`. If not, resolve that before doing any new x-series branch work.
