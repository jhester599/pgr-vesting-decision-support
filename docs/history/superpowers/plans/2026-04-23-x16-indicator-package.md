# x16 Indicator Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the surviving x14 structural signal and the x15 negative P/B
overlay result into a research-only indicator package and a reusable peer-review
prompt for later dashboard/monthly-report discussion.

**Architecture:** Read checked-in x14 and x15 artifacts, produce a compact
indicator specification JSON, a short memo, and a deep-research prompt that
asks for critique of the structural signal, missing P/B features, and
integration risks.

**Tech Stack:** Python 3.10+, JSON/Markdown artifacts only.

---

## Task 1: Tests First

- [x] Test the indicator spec stays research-only and preserves the no-change
      P/B anchor after x15.
- [x] Test the peer-review prompt includes the x14 candidate and x15 null
      result.

## Task 2: Implementation

- [x] Build a research-only indicator spec from x14 and x15 artifacts.
- [x] Build a peer-review prompt for future deep research.
- [x] Write x16 package JSON, prompt markdown, and memo.

## Verification

- [x] Run x16 unit tests.
- [x] Run x16 script and validate artifacts.
- [x] Run focused x-series tests covering x15/x16.
- [x] Run py_compile on new modules, scripts, and tests.
