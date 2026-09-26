# v159 Firth Shadow Integration Plan

Status: historical plan. Execution is complete; see
`docs/closeouts/V159_CLOSEOUT_AND_HANDOFF.md` and the `v159` section in
`CHANGELOG.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the v154 Firth logistic research winner (VMBS +0.04, BND +0.07 BA_covered) into a reporting-only monthly shadow lane alongside the existing `autoresearch_followon_v150` lane.

**Architecture:** Create `src/reporting/firth_shadow.py` (mirrors `shadow_followon.py` pattern) with a `build_firth_shadow_payload` function that forwards the live baseline probability while surfacing the Firth research findings (winners, deltas, recommendation) from `results/research/v154_firth_candidate.json`. Wire a third variant into `scripts/monthly_decision.py` at the existing two-variant shadow block (lines ~3218–3243). Research-only: no changes to the live classification path or config.

**Tech Stack:** Python 3.10+, pandas, json (stdlib). No new dependencies.

---

## Prerequisite

This branch depends on `results/research/v154_firth_candidate.json` from the v153-v158 cycle (PR #82). Branch from `codex/v153-v158-classification-feature-research` until PR #82 merges to master, then rebase.

```bash
git checkout codex/v153-v158-classification-feature-research
git checkout -b codex/v159-firth-shadow-integration
```

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/reporting/firth_shadow.py` | Create | New reporting module: constants, `load_firth_candidate`, `build_firth_shadow_payload` |
| `tests/test_firth_shadow.py` | Create | Unit tests for the new reporting module |
| `scripts/monthly_decision.py` | Modify | Wire `firth_shadow_v159` as third variant in shadow block (~line 3218) |
| `tests/test_monthly_pipeline_e2e.py` | Modify | Add assertion that `firth_shadow_v159` appears in shadow variants |
| `CHANGELOG.md` | Modify | Prepend v159 entry |
| `ROADMAP.md` | Modify | Update v153-v158 next queue section |
| `docs/closeouts/V159_CLOSEOUT_AND_HANDOFF.md` | Create | Closeout and next-session handoff |

---

## Task 1: Create `src/reporting/firth_shadow.py` (with failing tests first)

**Files:**
- Create: `tests/test_firth_shadow.py`
- Create: `src/reporting/firth_shadow.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_firth_shadow.py`:

```python
"""Tests for the v159 Firth logistic shadow reporting module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_firth_shadow_constants() -> None:
    from src.reporting.firth_shadow import (
        FIRTH_BENCHMARKS,
        FIRTH_SHADOW_VARIANT_LABEL,
        FIRTH_SHADOW_VARIANT_NAME,
    )
    assert "VMBS" in FIRTH_BENCHMARKS
    assert "BND" in FIRTH_BENCHMARKS
    assert FIRTH_SHADOW_VARIANT_NAME == "firth_shadow_v159"
    assert FIRTH_SHADOW_VARIANT_LABEL == "Firth Logistic Shadow"


def test_load_firth_candidate_reads_v154_file() -> None:
    from src.reporting.firth_shadow import load_firth_candidate
    candidate = load_firth_candidate()
    assert "firth_winners" in candidate
    assert "VMBS" in candidate["firth_winners"]
    assert "BND" in candidate["firth_winners"]
    assert candidate["recommendation"] == "adopt_firth_for_thin_benchmarks"


def test_build_firth_shadow_payload_is_reporting_only() -> None:
    from src.reporting.firth_shadow import (
        FIRTH_SHADOW_VARIANT_NAME,
        build_firth_shadow_payload,
    )
    payload = build_firth_shadow_payload(
        probability_actionable_sell=0.40,
        confidence_tier="LOW",
        stance="NEUTRAL",
    )
    assert payload["variant"] == FIRTH_SHADOW_VARIANT_NAME
    assert payload["label"] == "Firth Logistic Shadow"
    assert payload["reporting_only"] is True
    assert payload["probability_actionable_sell"] == 0.40
    assert payload["confidence_tier"] == "LOW"
    assert payload["stance"] == "NEUTRAL"


def test_build_firth_shadow_payload_includes_research_findings() -> None:
    from src.reporting.firth_shadow import build_firth_shadow_payload
    payload = build_firth_shadow_payload(
        probability_actionable_sell=0.36,
        confidence_tier="LOW",
        stance="NEUTRAL",
    )
    assert "VMBS" in payload["firth_winners"]
    assert "BND" in payload["firth_winners"]
    assert payload["firth_winner_deltas"]["VMBS"] > 0.02
    assert payload["firth_winner_deltas"]["BND"] > 0.02
    assert payload["firth_recommendation"] == "adopt_firth_for_thin_benchmarks"
    assert "firth_benchmarks" in payload
    assert sorted(payload["firth_benchmarks"]) == ["BND", "VMBS"]


def test_build_firth_shadow_payload_none_probability() -> None:
    from src.reporting.firth_shadow import build_firth_shadow_payload
    payload = build_firth_shadow_payload(
        probability_actionable_sell=None,
        confidence_tier=None,
        stance=None,
    )
    assert payload["probability_actionable_sell"] is None
    assert payload["reporting_only"] is True
    assert "firth_winners" in payload


def test_build_firth_shadow_payload_no_recommended_sell_pct() -> None:
    """Firth shadow must not contain a recommended_sell_pct — reporting only."""
    from src.reporting.firth_shadow import build_firth_shadow_payload
    payload = build_firth_shadow_payload(
        probability_actionable_sell=0.75,
        confidence_tier="HIGH",
        stance="ACTIONABLE-SELL",
    )
    assert "recommended_sell_pct" not in payload
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && python -m pytest tests/test_firth_shadow.py -q --tb=short
```

Expected: `ModuleNotFoundError: No module named 'src.reporting.firth_shadow'`

- [ ] **Step 3: Create `src/reporting/firth_shadow.py`**

```python
"""Helpers for the Firth logistic shadow reporting lane (v159)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIRTH_SHADOW_VARIANT_NAME: str = "firth_shadow_v159"
FIRTH_SHADOW_VARIANT_LABEL: str = "Firth Logistic Shadow"
FIRTH_BENCHMARKS: frozenset[str] = frozenset({"VMBS", "BND"})


def load_firth_candidate() -> dict[str, Any]:
    """Load the v154 Firth logistic research candidate for shadow reporting."""
    candidate_path = (
        PROJECT_ROOT / "results" / "research" / "v154_firth_candidate.json"
    )
    return json.loads(candidate_path.read_text(encoding="utf-8"))


def build_firth_shadow_payload(
    *,
    probability_actionable_sell: float | None,
    probability_actionable_sell_label: str | None = None,
    confidence_tier: str | None,
    stance: str | None,
    probability_investable_pool_label: str | None = None,
    probability_path_b_temp_scaled_label: str | None = None,
) -> dict[str, Any]:
    """Build the Firth logistic reporting-only shadow payload for monthly artifacts.

    Mirrors the baseline shadow probability while surfacing v154 Firth research
    findings (winners, BA_covered deltas, recommendation) for VMBS and BND.
    Reporting-only: does not alter the live recommendation path.
    """
    candidate = load_firth_candidate()
    firth_winners: list[str] = candidate.get("firth_winners", [])
    winner_deltas: dict[str, float] = {}
    for row in candidate.get("rows", []):
        bm = row.get("benchmark", "")
        delta = row.get("delta_ba_covered")
        if (
            bm in firth_winners
            and isinstance(delta, (int, float))
            and not math.isnan(float(delta))
        ):
            winner_deltas[bm] = float(delta)
    return {
        "variant": FIRTH_SHADOW_VARIANT_NAME,
        "label": FIRTH_SHADOW_VARIANT_LABEL,
        "reporting_only": True,
        "probability_actionable_sell": probability_actionable_sell,
        "probability_actionable_sell_label": probability_actionable_sell_label,
        "confidence_tier": confidence_tier,
        "stance": stance,
        "probability_investable_pool_label": probability_investable_pool_label,
        "probability_path_b_temp_scaled_label": probability_path_b_temp_scaled_label,
        "firth_benchmarks": sorted(FIRTH_BENCHMARKS),
        "firth_winners": firth_winners,
        "firth_winner_deltas": winner_deltas,
        "firth_recommendation": candidate.get("recommendation"),
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && python -m pytest tests/test_firth_shadow.py -q --tb=short
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/reporting/firth_shadow.py tests/test_firth_shadow.py
git commit -m "research: v159 add Firth shadow reporting module and tests"
```

---

## Task 2: Wire Firth Shadow into `scripts/monthly_decision.py`

**Files:**
- Modify: `scripts/monthly_decision.py` (two edits: import block + shadow variant block)
- Modify: `tests/test_monthly_pipeline_e2e.py`

### 2a — Add import

- [ ] **Step 1: Locate the existing shadow_followon import block**

Open `scripts/monthly_decision.py` and find this import block (around line 123):

```python
from src.reporting.shadow_followon import (
    FOLLOWON_VARIANT_LABEL,
    FOLLOWON_VARIANT_NAME,
    build_followon_decision_overlay_payload,
    build_followon_shadow_payload,
)
```

- [ ] **Step 2: Add the Firth shadow import immediately after that block**

Insert these lines immediately after the closing `)` of the shadow_followon import:

```python
from src.reporting.firth_shadow import (
    FIRTH_SHADOW_VARIANT_NAME,
    build_firth_shadow_payload,
)
```

### 2b — Wire the third variant

- [ ] **Step 3: Find the shadow variant assembly block**

Locate the block starting at approximately line 3218 that reads:

```python
        followon_variant = build_followon_shadow_payload(
            probability_actionable_sell=classification_shadow_summary.get("probability_actionable_sell"),
            probability_actionable_sell_label=classification_shadow_summary.get(
                "probability_actionable_sell_label"
            ),
            confidence_tier=str(classification_shadow_summary.get("confidence_tier"))
            if classification_shadow_summary.get("confidence_tier") is not None
            else None,
            stance=str(classification_shadow_summary.get("stance"))
            if classification_shadow_summary.get("stance") is not None
            else None,
            probability_investable_pool_label=classification_shadow_summary.get(
                "probability_investable_pool_label"
            ),
            probability_path_b_temp_scaled_label=classification_shadow_summary.get(
                "probability_path_b_temp_scaled_label"
            ),
        )
        classification_shadow_variants = [baseline_variant, followon_variant]
        if not classification_shadow_artifact_df.empty:
            followon_detail_df = classification_shadow_artifact_df.copy()
            followon_detail_df["variant"] = FOLLOWON_VARIANT_NAME
            classification_shadow_artifact_df = pd.concat(
                [classification_shadow_artifact_df, followon_detail_df],
                ignore_index=True,
            )
```

- [ ] **Step 4: Replace that block with the three-variant version**

Replace the entire block above with:

```python
        followon_variant = build_followon_shadow_payload(
            probability_actionable_sell=classification_shadow_summary.get("probability_actionable_sell"),
            probability_actionable_sell_label=classification_shadow_summary.get(
                "probability_actionable_sell_label"
            ),
            confidence_tier=str(classification_shadow_summary.get("confidence_tier"))
            if classification_shadow_summary.get("confidence_tier") is not None
            else None,
            stance=str(classification_shadow_summary.get("stance"))
            if classification_shadow_summary.get("stance") is not None
            else None,
            probability_investable_pool_label=classification_shadow_summary.get(
                "probability_investable_pool_label"
            ),
            probability_path_b_temp_scaled_label=classification_shadow_summary.get(
                "probability_path_b_temp_scaled_label"
            ),
        )
        firth_variant = build_firth_shadow_payload(
            probability_actionable_sell=classification_shadow_summary.get("probability_actionable_sell"),
            probability_actionable_sell_label=classification_shadow_summary.get(
                "probability_actionable_sell_label"
            ),
            confidence_tier=str(classification_shadow_summary.get("confidence_tier"))
            if classification_shadow_summary.get("confidence_tier") is not None
            else None,
            stance=str(classification_shadow_summary.get("stance"))
            if classification_shadow_summary.get("stance") is not None
            else None,
            probability_investable_pool_label=classification_shadow_summary.get(
                "probability_investable_pool_label"
            ),
            probability_path_b_temp_scaled_label=classification_shadow_summary.get(
                "probability_path_b_temp_scaled_label"
            ),
        )
        classification_shadow_variants = [baseline_variant, followon_variant, firth_variant]
        if not classification_shadow_artifact_df.empty:
            followon_detail_df = classification_shadow_artifact_df.copy()
            followon_detail_df["variant"] = FOLLOWON_VARIANT_NAME
            firth_detail_df = classification_shadow_artifact_df.copy()
            firth_detail_df["variant"] = FIRTH_SHADOW_VARIANT_NAME
            classification_shadow_artifact_df = pd.concat(
                [classification_shadow_artifact_df, followon_detail_df, firth_detail_df],
                ignore_index=True,
            )
```

### 2c — Add e2e test assertion

- [ ] **Step 5: Find the existing followon assertion in `tests/test_monthly_pipeline_e2e.py`**

Search for `autoresearch_followon_v150` or `FOLLOWON_VARIANT_NAME` in `tests/test_monthly_pipeline_e2e.py`.

- [ ] **Step 6: Add a parallel assertion for the Firth variant**

Find the test that checks for `autoresearch_followon_v150` in the shadow variants. Immediately after it, add a matching assertion for `firth_shadow_v159`:

```python
assert any(
    v.get("variant") == "firth_shadow_v159"
    for v in monthly_summary.get("classification_shadow_variants", [])
), "firth_shadow_v159 variant must appear in monthly shadow variants"
```

If no existing test for `autoresearch_followon_v150` exists in the e2e file, find the test function that checks `classification_shadow_variants` and add the assertion at the end of that function.

- [ ] **Step 7: Run the e2e smoke tests**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && python -m pytest tests/test_monthly_pipeline_e2e.py tests/test_firth_shadow.py tests/test_shadow_followon.py -q --tb=short
```

Expected: all tests pass. If `test_monthly_pipeline_e2e.py` fails with an import error, verify the import was added correctly to `scripts/monthly_decision.py`.

- [ ] **Step 8: Commit**

```bash
git add scripts/monthly_decision.py tests/test_monthly_pipeline_e2e.py
git commit -m "research: v159 wire Firth shadow variant into monthly pipeline"
```

---

## Task 3: Documentation and Closeout

**Files:**
- Create: `docs/closeouts/V159_CLOSEOUT_AND_HANDOFF.md`
- Modify: `CHANGELOG.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Create `docs/closeouts/V159_CLOSEOUT_AND_HANDOFF.md`**

```markdown
# V159 Closeout And Handoff

Created: 2026-04-18

## Completed Block

`v159` closes the Firth shadow integration cycle. The v154 Firth logistic research
winner (VMBS +0.0412, BND +0.0704 BA_covered) is now surfaced as a reporting-only
shadow lane `firth_shadow_v159` alongside the existing `autoresearch_followon_v150`
lane in monthly artifacts.

## Final Outcomes

- New reporting module: `src/reporting/firth_shadow.py`
- Monthly pipeline now produces three shadow variants:
  1. `baseline_shadow` — standard Path A per-benchmark logistic
  2. `autoresearch_followon_v150` — v139-v150 regression research winners
  3. `firth_shadow_v159` — Firth logistic research findings for VMBS and BND
- No production config changes; live recommendation path unchanged
- 6 new tests in `tests/test_firth_shadow.py`; all pass

## Promotion Boundaries

- Production: no change to live monthly recommendation
- Shadow: `firth_shadow_v159` surfaces in monthly `classification_shadow_csv`
  and `monthly_summary.json` as a reporting-only third variant
- Research-only: `results/research/v154_firth_candidate.json` remains the source

## Recommended Next Queue

1. `BL-01` — Black-Litterman tau/view tuning (plan:
   `docs/superpowers/plans/2026-04-18-bl01-black-litterman-tuning.md`)
2. `CLS-03` — Path A vs Path B production decision (still time-locked)
3. Future: promote Firth for VMBS/BND in production Path A after 24 prospective months

## Verification

```bash
python -m pytest tests/test_firth_shadow.py \
                 tests/test_shadow_followon.py \
                 tests/test_monthly_pipeline_e2e.py \
                 -q --tb=short
```

## Exact Next Commands

```bash
git checkout master
git pull --ff-only origin master
git checkout -b codex/bl01-black-litterman-tuning
# implement from docs/superpowers/plans/2026-04-18-bl01-black-litterman-tuning.md
```
```

- [ ] **Step 2: Prepend v159 to CHANGELOG.md**

Add this block at the very top of `CHANGELOG.md`:

```markdown
## v159 (2026-04-18)

- Firth shadow integration: wired `firth_shadow_v159` reporting lane into monthly pipeline
- New module: `src/reporting/firth_shadow.py` (load_firth_candidate, build_firth_shadow_payload)
- Monthly shadow now produces three variants: baseline_shadow, autoresearch_followon_v150, firth_shadow_v159
- Research findings surfaced: VMBS +0.0412 BA_cov, BND +0.0704 BA_cov (from v154 Firth candidate)
- No production config or live recommendation changes
- Next: BL-01 Black-Litterman tau/view tuning

```

- [ ] **Step 3: Update ROADMAP.md**

In the `## Active Research Direction: v153-v158` section, locate the "Recommended next queue" block and update it to mark v159 complete:

```markdown
Recommended next queue after `v158`:

1. `v159` — Firth logistic shadow integration for VMBS and BND — **complete (2026-04-18)**
2. `BL-01` — Black-Litterman tau/view tuning — open
3. `CLS-03` — Path A vs Path B (time-locked)
```

Also add a new section immediately after the v153-v158 block:

```markdown
## Active Research Direction: v159 + BL-01

- `v159` complete: Firth shadow lane wired; see `docs/closeouts/V159_CLOSEOUT_AND_HANDOFF.md`
- `BL-01` open: plan at `docs/superpowers/plans/2026-04-18-bl01-black-litterman-tuning.md`
```

- [ ] **Step 4: Run full fast test suite**

```bash
cd C:/Users/Jeff/Documents/pgr-vesting-decision-support && python -m pytest tests/test_firth_shadow.py tests/test_shadow_followon.py tests/test_monthly_pipeline_e2e.py -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/closeouts/V159_CLOSEOUT_AND_HANDOFF.md \
        CHANGELOG.md \
        ROADMAP.md
git commit -m "docs: v159 closeout, CHANGELOG, and ROADMAP update"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| New reporting module `src/reporting/firth_shadow.py` | Task 1 |
| `FIRTH_SHADOW_VARIANT_NAME = "firth_shadow_v159"` | Task 1 Step 3 |
| `FIRTH_BENCHMARKS = frozenset({"VMBS", "BND"})` | Task 1 Step 3 |
| `load_firth_candidate` reads `v154_firth_candidate.json` | Task 1 Step 3 |
| `build_firth_shadow_payload` mirrors baseline probability | Task 1 Step 3 |
| Payload includes `firth_winners`, `firth_winner_deltas`, `firth_recommendation` | Task 1 Step 3 |
| `reporting_only: True` always present | Task 1 Step 3 + tests |
| `recommended_sell_pct` never in payload | Task 1 test |
| Wired as third variant in `scripts/monthly_decision.py` | Task 2 Steps 3–4 |
| `classification_shadow_artifact_df` gets `firth_shadow_v159` rows | Task 2 Step 4 |
| E2e test asserts `firth_shadow_v159` in variants | Task 2 Step 6 |
| No live config changes | All tasks |
| CHANGELOG, ROADMAP, closeout | Task 3 |

### Placeholder scan

No TBD, no "implement later", no "similar to Task N". All code blocks are complete. ✓

### Type consistency

- `load_firth_candidate() -> dict[str, Any]` ✓
- `build_firth_shadow_payload(...) -> dict[str, Any]` — same signature shape as `build_followon_shadow_payload` in `shadow_followon.py` ✓
- `FIRTH_BENCHMARKS: frozenset[str]` — used in `sorted(FIRTH_BENCHMARKS)` → `list[str]` ✓
- `winner_deltas: dict[str, float]` populated only for non-NaN rows ✓
