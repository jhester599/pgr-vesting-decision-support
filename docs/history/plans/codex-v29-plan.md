# v29 Interpretation And Confidence Plan

Created: 2026-04-05

## Goal

Improve the monthly report and email so they explain the model output more
clearly without changing the underlying prediction layer.

## Why

After v27-v28, the project now distinguishes between:

- the broader forecast benchmark universe
- the narrower investable redeploy universe

That creates a communication problem:

- the monthly output must show useful forecast context
- without implying every benchmark is also a recommended buy

## v29 Scope

v29 focuses on interpretation only:

1. add an explicit confidence snapshot
2. label benchmark roles in the monthly output
3. make the top-line model-view sentence plainer English
4. clarify tax-scenario wording in the next-vest section
5. carry the same interpretation improvements into the HTML email

## Non-Goals

v29 does not:

- change the active recommendation layer
- change the promoted visible cross-check
- change the forecast benchmark universe
- change the v27 redeploy portfolio process
- reopen feature or model research

## Success Criteria

v29 is successful if the monthly output now:

- clearly distinguishes buy candidates from forecast-only context
- shows the confidence gate in a compact pass/fail form
- keeps the report and email structurally aligned
- passes focused rendering tests and the full regression suite
