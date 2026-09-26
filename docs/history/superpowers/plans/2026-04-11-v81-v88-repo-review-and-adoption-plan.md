# v81-v88 Repo Review and Adoption Plan

## Purpose

This document captures a full repo review after the `v66-v80` promotion and
stabilization sequence.

Status update as of 2026-04-11:

- `v81-v84` completed: workflow/email/dashboard/docs aligned to the promoted
  baseline and static dashboard snapshots are now written monthly
- `v85-v86` completed: `monthly_summary.json` added as a structured monthly
  contract and the visible equal-weight cross-check retired from primary
  surfaces while keeping `consensus_shadow.csv` for diagnostics

The goal is to identify:

- useful work that was implemented but not fully adopted
- production/reporting surfaces that drifted from the live baseline
- documentation that no longer reflects the current repo state
- the best next enhancement block after the quality-weighted consensus
  promotion

---

## Review Summary

### Current Production State

- live recommendation path: quality-weighted consensus
- temporary visible cross-check: equal-weight consensus
- current monthly artifact set:
  - `recommendation.md`
  - `diagnostic.md`
  - `signals.csv`
  - `benchmark_quality.csv`
  - `consensus_shadow.csv`
  - `run_manifest.json`

### High-Value Findings

1. Core operator docs had drifted materially behind the live repo state.
   - `README.md`, `ROADMAP.md`, `docs/architecture.md`,
     `docs/workflows.md`, `docs/operations-runbook.md`,
     `docs/decision-output-guide.md`, `docs/artifact-policy.md`,
     `docs/model-governance.md`, and `docs/troubleshooting.md`
     were describing the pre-`v38` / pre-quality-weighted world.

2. The dashboard exists and works locally, but it is not treated as a supported
   production-adjacent surface.
   - `dashboard/app.py` is real and useful.
   - It is not surfaced in the README quick start, monthly workflow, email, or
     release process.
   - It still leans heavily on markdown parsing and does not make first-class
     use of `benchmark_quality.csv` or `consensus_shadow.csv`.

3. The monthly email path is implemented and committed, but it has format drift.
   - `src/reporting/email_sender.py` still parses the old
     `Simple-Baseline Cross-Check` section name for its structured cross-check
     rendering.
   - The current report uses `Consensus Shadow Evaluation`.
   - The plain-text email still picks up some of the new content incidentally,
     but the HTML email is no longer structurally aligned with the live report.

4. Workflow verification is weaker than the live artifact contract.
   - `.github/workflows/monthly_decision.yml` still verifies only the older core
     artifact set, not `benchmark_quality.csv` and `consensus_shadow.csv`.
   - The workflow contract tests also stop short of asserting the newer monthly
     artifact surface.

5. Some promising research work remains unadopted, but it should not be the
   next thing promoted blindly.
   - `v70` remains a credible secondary calibration branch.
   - `v46` and `v73` remain plausible future decision-layer work.
   - None of those are more urgent than fixing the product and documentation
     drift around the already-promoted `v72` / `v76` path.

---

## Recommended Next Block

The best next block is not another prediction-layer bakeoff.

The repo will benefit more from a short adoption-and-contract cycle that:

- makes the current production path easier to observe and trust
- turns the dashboard from an orphaned local tool into a supported surface
- restores email/report parity
- hardens the monthly artifact contract so future promotions do not silently
  drop outputs

---

## Version Map

### v81 - Docs and workflow contract synchronization

- finalize the operator-facing doc refresh for the live `v66-v80` baseline
- update `.github/workflows/monthly_decision.yml` verification so it checks for:
  - `benchmark_quality.csv`
  - `consensus_shadow.csv`
- extend workflow contract tests to assert the newer monthly artifact set

### v82 - Email/report parity refresh

- update `src/reporting/email_sender.py` to parse and render the live
  `Consensus Shadow Evaluation` section directly
- keep backward compatibility with older report sections where easy
- refresh the email tests so they validate the current live report structure,
  not only the older v13/v29-era section names

### v83 - Dashboard adoption and data-model refresh

- upgrade `dashboard/app.py` to consume:
  - `benchmark_quality.csv`
  - `consensus_shadow.csv`
  - `run_manifest.json`
  more directly and less through regex parsing of markdown
- surface current live-vs-cross-check comparison, benchmark-quality, and
  Clark-West-aware health signals in the dashboard
- add a small dashboard smoke test or parsing test suite

### v84 - Dashboard distribution decision

- choose one supported dashboard posture:
  - keep it as a documented local Streamlit tool
  - or add a static monthly HTML snapshot / GitHub Pages path
- if email linking is a real requirement, prefer a static committed monthly
  artifact over a link to a local-only Streamlit command

### v85 - Structured monthly summary payload

- add one machine-readable monthly summary file for consumers such as:
  - email
  - dashboard
  - future automation
- use this to reduce fragile markdown scraping in secondary surfaces

### v86 - Cross-check retirement decision

- after the next clean monthly production cycle, decide whether to remove the
  visible equal-weight cross-check from the report and email
- keep the underlying comparison logic available for diagnostics even if the
  visible section is retired

### v87 - Secondary calibration branch review

- revisit `v70` only if the live quality-weighted path underwhelms or if
  per-benchmark variance control becomes the main concern
- require a narrow promotion gate rather than a broad new sweep

### v88 - Decision-layer follow-on research

- revisit `v46` classification and `v73` hybrid gating only after the
  production surface is stable
- judge these by policy and recommendation usefulness first, not by forecast
  metrics alone
- detailed follow-on plan:
  - `docs/superpowers/plans/2026-04-11-v87-v96-classification-hybrid-research.md`
- the `v87-v96` follow-on now also incorporates the archived deep-review
  guidance on feature families, pooled-vs-separate model structure, composite
  targets, and calibration-first gating

---

## Priority Order

Recommended execution order:

1. `v81`
2. `v82`
3. `v83`
4. `v84`
5. `v86`
6. `v85`
7. `v87-v88`

Rationale:

- fix the production contract first
- then fix the operator-facing surfaces
- then decide how much of the temporary cross-check should remain visible
- only after that resume broader research promotion work

---

## Adoption Notes

### Dashboard

The dashboard should not be treated as "already solved" just because
`dashboard/app.py` exists.

It is best understood as:

- implemented
- locally usable
- not yet fully integrated into the operator workflow

The next cycle should decide whether to:

- support it intentionally
- or archive it intentionally

### Email

The email path is production-relevant today.

That means report-heading changes must be treated as contract changes for:

- `src/reporting/email_sender.py`
- email tests
- any future dashboard or automation reader

### Documentation

Archived historical docs should remain historical.

The focus for "current" accuracy should stay on:

- top-level repo docs
- operator-facing docs
- live workflow and governance docs
- active plan documents in `docs/superpowers/plans/`

---

## Recommended Immediate Next Step

Start `v81-v84` as one focused adoption-and-contract PR stream.

That is the highest-leverage follow-up to the recent promotion work because it
improves trust, usability, and operator clarity without reopening the
prediction-layer promotion question too early.
