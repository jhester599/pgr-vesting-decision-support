# v102-v117 Exhaustive Post-Review Enhancement Plan

## Purpose

This document converts the full content of the April 11, 2026 external
repo-review reports into one execution program. It intentionally covers both:

- explicit recommendations
- implied actions from the detailed findings and critiques

The source review reports are archived in:

- `docs/archive/history/repo-peer-reviews/2026-04-11/chatgpt_repo_peerreview_20260411.md`
- `docs/archive/history/repo-peer-reviews/2026-04-11/gemini_repo_peerreview_20260411.md`

## Intake Summary

The reports converge on a few core points:

- keep the live regression-led production path intact initially
- treat classification as the promising next decision-layer direction
- avoid broad model-family sprawl or infrastructure expansion
- strengthen monthly artifact contracts and decision-surface clarity
- add a shadow gate overlay before any production classifier promotion
- preserve GitHub Actions and artifact commits as the primary deployment model

The reports also add a few nuanced but important follow-ups:

- harden backdated `--as-of` runs against future-target leakage
- make `monthly_summary.json` the primary surface contract
- persist classifier details and historical classifier outputs
- use agreement/stability-constrained promotion logic rather than
  highest-utility-alone selection
- clarify the intended role of the static dashboard vs local Streamlit
- clean up legacy ensemble references and stale docstrings over time

## Coverage Map

This version block is designed to cover every meaningful suggested action from
the April 11 report set.

### v102 - Archive Intake and Review Provenance

- archive both reports under `docs/archive/history/repo-peer-reviews/2026-04-11/`
- normalize the Gemini report into markdown naming
- add archive `README.md`
- replace old absolute archive links with repo-relative references

### v103 - Summary Contract and Classifier Artifact Completion

- add `classification_shadow.csv`
- expand `monthly_summary.json`
- prefer the structured summary contract over markdown parsing
- update workflow artifact assertions

### v104 - Decision-Surface Clarity and Disagreement Rendering

- add explicit `Hold vs Sell` and `Is this month actionable?`
- add a compact agreement/disagreement panel
- surface the same idea in report, email, dashboard, and Streamlit

### v105 - Dashboard and Surface Architecture Alignment

- keep `dashboard.html` as the primary shareable monthly UI artifact
- keep Streamlit as a local/debugging tool
- tighten Streamlit toward `monthly_summary.json` and CSV artifacts

### v106 - Monthly Orchestration Refactor and Shared Intermediates

- extract small pure helper paths where practical
- reduce repeated artifact-assembly logic
- preserve CLI behavior and outputs

### v107 - Backdated `--as-of` Leakage Hardening

- truncate target rows that would need future prices beyond the simulated
  as-of date
- apply the same guard to regression and classifier shadow paths

### v108 - Decision-Layer Test Hardening and Artifact Contract Tests

- add deterministic tests for decision mapping helpers
- add artifact/schema contract tests

### v109 - Historical Classifier Logging and Maturity Tracking

- add append-only classifier history
- record maturity timing for later calibration monitoring

### v110 - Gemini-Style Hybrid Gate Backtest

- test a veto-style gate:
  regression sell requires classifier confirmation above threshold

### v111 - ChatGPT-Style Permission-to-Deviate Overlay Research

- test a conservative overlay:
  classification only grants permission to become more aggressive than the
  default/live path

### v112 - Target Reformulation Research

- test the actionability target against more decision-aligned alternatives
- keep the feature set lean

### v113 - Constrained Candidate Selection Research

- rank candidates on uplift, agreement, and stability together
- emit explicit promotion-eligibility flags

### v114 - Shadow Gate Overlay in Monthly Production Artifacts

- wire the selected candidate into monthly production as shadow-only
- persist `decision_overlays.csv`
- surface a compact diagnostic summary without changing live behavior

### v115 - Horizon-Aware Classifier Calibration and Drift Monitoring

- compute matured-horizon classifier diagnostics when outcomes are available
- use Brier, log loss, and ECE as governance-facing metrics

### v116 - Limited Production Gate Candidate and Legacy-Layer Cleanup

- only proceed if evidence from `v114-v115` is strong
- evaluate a limited gate candidate, but do not force promotion
- clean up stale comments/docstrings around historical ensemble layers

### v117 - Primary Recommendation-Mode Selector Evaluation

- evaluate whether classification should eventually influence
  recommendation mode directly
- defer promotion unless the stricter evidence bar is clearly met

## Guardrails

- no K-Fold cross-validation
- no broad nonlinear sweep beyond the bounded research already completed
- no large feature-sprawl cycle
- no infrastructure shift away from GitHub Actions
- no production classification override without agreement and stability evidence

## Expected Artifacts

- monthly production artifacts:
  - `classification_shadow.csv`
  - `decision_overlays.csv`
  - expanded `monthly_summary.json`
- history artifact:
  - `results/monthly_decisions/classification_shadow_history.csv`
- research artifacts:
  - `results/research/v110_*.py` through `results/research/v117_*.py`
  - matching CSV / markdown outputs
  - matching flat pytest coverage

## Notes on Already-Implemented Work

Some recommendations from the reports were already partially in motion before
this plan was written:

- the shadow-only classifier confidence block
- `monthly_summary.json`
- the static dashboard snapshot
- the quality-weighted production recommendation path

This plan treats those areas as completion / hardening work rather than
duplicate implementation.
