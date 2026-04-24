# x24 Indicator Contract And Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the surviving x-series signals into one research-only
indicator contract and write a strong x1-x24 resume document for future work.

**Architecture:** Reuse the existing x16 structural package and x23 dividend
package as the two surviving indicator candidates. x24 bundles them into a
single research-only contract, writes companion memos/prompts, and generates a
clear resume document covering the full x-series path from x1 through x24.

**Tech Stack:** Python 3.10+, JSON/Markdown artifact generation, existing
checked-in x-series summaries only.

---

## Scope

- x24 bundles:
  - the `adjusted_structural_bvps_pb_6m` structural watch from x16
  - the `to_current_bvps` annual dividend-size watch from x23
- The summary document must:
  - explain what each x-step did
  - state the strongest findings
  - record what did not work
  - list recommended next steps
  - link peer-review prompts
  - note the current branch/merge state clearly

## File Map

| File | Action | Purpose |
|---|---|---|
| `docs/superpowers/plans/2026-04-24-x24-indicator-contract-and-summary.md` | Create | Plan |
| `src/research/x24_indicator_contract.py` | Create | x24 helpers |
| `scripts/research/x24_indicator_contract.py` | Create | x24 runner |
| `tests/test_research_x24_indicator_contract.py` | Create | x24 tests |
| `results/research/x24_indicator_contract.json` | Create | x24 contract artifact |
| `results/research/x24_research_memo.md` | Create | x24 memo |
| `results/research/x24_bundle_peer_review_prompt.md` | Create | holistic prompt |
| `results/research/x24_structural_peer_review_prompt.md` | Create | structural/P-B prompt |
| `docs/research/x_series_resume_2026-04-24.md` | Create | restart summary |

## Task 1: x24 Tests First

- [ ] Test the x24 contract stays research-only.
- [ ] Test the x24 contract carries both the structural and dividend signals.
- [ ] Test the holistic prompt mentions both surviving signals.

## Task 2: x24 Implementation

- [ ] Implement the x24 contract helper.
- [ ] Implement the x24 prompt helper.
- [ ] Write x24 JSON and memo artifacts.

## Task 3: Resume Document

- [ ] Summarize x1-x24 with short findings for each step or grouped step.
- [ ] Record the current repo/branch state clearly.
- [ ] Link all useful peer-review prompts, including x23 and any new x24 prompts.
- [ ] Write explicit recommended next steps for when work resumes.

## Verification

- [ ] Run x24 unit tests.
- [ ] Run the x24 script and validate generated JSON/Markdown artifacts.
- [ ] Run `py_compile` on x24 module/script/test.
