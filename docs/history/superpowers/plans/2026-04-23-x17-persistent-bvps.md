# x17 Persistent BVPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a dividend-persistent synthetic BVPS history improves
the core book-value-creation modeling path relative to raw BVPS.

**Architecture:** Build a synthetic monthly BVPS series by adding back paid
dividends cumulatively to raw BVPS, then rerun bounded x9-style bridge
forecasting on both raw and persistent variants for side-by-side comparison.

**Tech Stack:** Python 3.10+, pandas, numpy, existing x9/x12 research helpers,
JSON/CSV/Markdown research artifacts only.

---

## Task 1: Tests First

- [x] Test persistent BVPS construction adds cumulative dividends back into the
      level series.
- [x] Test persistent BVPS targets use future persistent levels and growth.
- [x] Test x17 summary separates raw and persistent variants by horizon.

## Task 2: Implementation

- [x] Build persistent BVPS utilities and target-construction helpers.
- [x] Reuse x9 bridge features and bounded model blocks on raw vs persistent
      BVPS.
- [x] Write x17 detail CSV, summary JSON, and memo.

## Verification

- [x] Run x17 unit tests.
- [x] Run x17 script and validate artifacts.
- [x] Run focused x-series tests covering x12/x17.
- [x] Run py_compile on new modules, scripts, and tests.
