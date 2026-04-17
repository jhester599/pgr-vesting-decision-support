# Shadow Follow-On v150 Side-By-Side Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one experimental `autoresearch_followon_v150` shadow lane to monthly artifacts so the promoted `v139-v150` winners appear side-by-side with the current shadow outputs without changing production recommendation behavior.

**Architecture:** Keep the current shadow path as the baseline lane and add one narrow helper path that loads the surviving candidate files, builds a follow-on shadow payload, and threads that payload through monthly summary, dashboard, markdown, and CSV artifact writers. Preserve existing baseline fields for compatibility and expose the new lane through additive variant-aware structures.

**Tech Stack:** Python 3.10+, pandas, existing monthly decision/reporting pipeline, pytest

---

## File Map

- Create: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\src\reporting\shadow_followon.py`
  - candidate loading and compact follow-on lane payload builder
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\scripts\monthly_decision.py`
  - orchestrate baseline and follow-on shadow payload assembly
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\src\reporting\monthly_summary.py`
  - serialize both shadow lanes in additive structures
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\src\reporting\dashboard_snapshot.py`
  - render the side-by-side comparison cards/section
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\src\reporting\classification_artifacts.py`
  - emit variant-aware CSV rows or add a `variant` column
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\CHANGELOG.md`
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\ROADMAP.md`
- Test: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\tests\test_shadow_followon.py`
- Test: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\tests\test_monthly_summary.py`
- Test: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\tests\test_dashboard_snapshot.py`
- Test: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\tests\test_classification_artifacts.py`

## Task 1: Add Candidate Loader And Follow-On Payload Builder

**Files:**
- Create: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\src\reporting\shadow_followon.py`
- Test: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\tests\test_shadow_followon.py`

- [ ] **Step 1: Write the failing tests for candidate loading and payload identity**

```python
from src.reporting.shadow_followon import (
    FOLLOWON_VARIANT_NAME,
    load_followon_candidate_bundle,
    build_followon_shadow_payload,
)


def test_load_followon_candidate_bundle_reads_v139_v150_winners():
    bundle = load_followon_candidate_bundle()
    assert bundle["v141_blend_weight"] == 0.60
    assert bundle["v143_corr_prune"] == 0.80
    assert bundle["v144_conformal"] == {"coverage": 0.75, "aci_gamma": 0.03}
    assert bundle["v149_kelly"] == {"fraction": 0.50, "cap": 0.25}
    assert bundle["v150_neutral_band"] == 0.015


def test_build_followon_shadow_payload_is_named_and_reporting_only():
    payload = build_followon_shadow_payload(
        probability_actionable_sell=0.56,
        confidence_tier="LOW",
        stance="NEUTRAL",
    )
    assert payload["variant"] == FOLLOWON_VARIANT_NAME
    assert payload["label"] == "Autoresearch Follow-On"
    assert payload["reporting_only"] is True
    assert payload["candidate_sources"]["v149_kelly"]["fraction"] == 0.50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_shadow_followon.py -q --tb=short`
Expected: FAIL with import errors because `src.reporting.shadow_followon` does not exist yet.

- [ ] **Step 3: Write the minimal helper module**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLLOWON_VARIANT_NAME = "autoresearch_followon_v150"
FOLLOWON_VARIANT_LABEL = "Autoresearch Follow-On"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_followon_candidate_bundle() -> dict[str, Any]:
    research_dir = PROJECT_ROOT / "results" / "research"
    return {
        "v141_blend_weight": float(_read_text(research_dir / "v141_blend_weight_candidate.txt")),
        "v143_corr_prune": float(_read_text(research_dir / "v143_corr_prune_candidate.txt")),
        "v144_conformal": json.loads(_read_text(research_dir / "v144_conformal_candidate.json")),
        "v149_kelly": json.loads(_read_text(research_dir / "v149_kelly_candidate.json")),
        "v150_neutral_band": float(_read_text(research_dir / "v150_neutral_band_candidate.txt")),
    }


def build_followon_shadow_payload(
    *,
    probability_actionable_sell: float | None,
    confidence_tier: str | None,
    stance: str | None,
) -> dict[str, Any]:
    return {
        "variant": FOLLOWON_VARIANT_NAME,
        "label": FOLLOWON_VARIANT_LABEL,
        "reporting_only": True,
        "probability_actionable_sell": probability_actionable_sell,
        "confidence_tier": confidence_tier,
        "stance": stance,
        "candidate_sources": load_followon_candidate_bundle(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_shadow_followon.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_shadow_followon.py src/reporting/shadow_followon.py
git commit -m "add follow-on shadow payload helper"
```

## Task 2: Extend Monthly Summary Serialization

**Files:**
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\src\reporting\monthly_summary.py`
- Test: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\tests\test_monthly_summary.py`

- [ ] **Step 1: Write the failing serialization test**

```python
from src.reporting.monthly_summary import build_monthly_summary_payload


def test_build_monthly_summary_payload_includes_shadow_variants():
    payload = build_monthly_summary_payload(
        as_of_date="2026-04-17",
        run_date="2026-04-17",
        recommendation_layer_label="baseline",
        consensus="HOLD",
        confidence_tier="LOW",
        recommendation_mode="MONITORING-ONLY",
        sell_pct=0.0,
        mean_predicted=0.01,
        mean_ic=0.02,
        mean_hit_rate=0.55,
        mean_prob_outperform=None,
        calibrated_prob_outperform=None,
        aggregate_oos_r2=None,
        aggregate_nw_ic=None,
        warnings=[],
        signals=pd.DataFrame(),
        benchmark_quality_df=None,
        consensus_shadow_df=None,
        visible_cross_check=False,
        classification_shadow_summary={"enabled": True, "stance": "NEUTRAL"},
        shadow_gate_overlay=None,
        classification_shadow_variants=[
            {"variant": "baseline_shadow"},
            {"variant": "autoresearch_followon_v150"},
        ],
    )
    assert len(payload["classification_shadow_variants"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_monthly_summary.py -q --tb=short`
Expected: FAIL because `build_monthly_summary_payload` does not yet accept the new argument.

- [ ] **Step 3: Add additive variant-aware fields without breaking existing ones**

```python
def build_monthly_summary_payload(
    *,
    ...
    classification_shadow_summary: dict[str, Any] | None = None,
    shadow_gate_overlay: dict[str, Any] | None = None,
    classification_shadow_variants: list[dict[str, Any]] | None = None,
    decision_overlay_variants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ...
    return {
        ...
        "classification_shadow": classification_shadow_summary,
        "shadow_gate_overlay": shadow_gate_overlay,
        "classification_shadow_variants": classification_shadow_variants or [],
        "decision_overlay_variants": decision_overlay_variants or [],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_monthly_summary.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_monthly_summary.py src/reporting/monthly_summary.py
git commit -m "serialize side-by-side shadow variants"
```

## Task 3: Make CSV Shadow Artifacts Variant-Aware

**Files:**
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\src\reporting\classification_artifacts.py`
- Test: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\tests\test_classification_artifacts.py`

- [ ] **Step 1: Write the failing CSV test**

```python
def test_write_classification_shadow_csv_keeps_variant_column(tmp_path):
    df = pd.DataFrame(
        [
            {"variant": "baseline_shadow", "benchmark": "VOO"},
            {"variant": "autoresearch_followon_v150", "benchmark": "VOO"},
        ]
    )
    path = write_classification_shadow_csv(tmp_path, df)
    written = pd.read_csv(path)
    assert set(written["variant"]) == {
        "baseline_shadow",
        "autoresearch_followon_v150",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classification_artifacts.py -q --tb=short`
Expected: FAIL because `variant` is not preserved by the current column set.

- [ ] **Step 3: Add `variant` to artifact schemas**

```python
CLASSIFICATION_SHADOW_COLUMNS = [
    "variant",
    "benchmark",
    ...
]

DECISION_OVERLAY_COLUMNS = [
    "variant",
    "recommendation_mode",
    ...
]
```

Also ensure empty CSV outputs still include the `variant` column.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_classification_artifacts.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_classification_artifacts.py src/reporting/classification_artifacts.py
git commit -m "preserve variant labels in shadow artifacts"
```

## Task 4: Render Side-By-Side Dashboard Comparison

**Files:**
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\src\reporting\dashboard_snapshot.py`
- Test: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\tests\test_dashboard_snapshot.py`

- [ ] **Step 1: Write the failing dashboard test**

```python
def test_dashboard_snapshot_renders_shadow_variant_labels(tmp_path):
    write_dashboard_snapshot(
        tmp_path,
        as_of_date="2026-04-17",
        recommendation_mode="MONITORING-ONLY",
        consensus="HOLD",
        sell_pct=0.0,
        mean_predicted=0.01,
        mean_ic=0.02,
        mean_hit_rate=0.55,
        aggregate_oos_r2=None,
        recommendation_layer_label="baseline",
        warnings=[],
        signals=pd.DataFrame(),
        benchmark_quality_df=None,
        consensus_shadow_df=None,
        classification_shadow_summary={"enabled": True},
        shadow_gate_overlay=None,
        classification_shadow_variants=[
            {"variant": "baseline_shadow", "label": "Current Shadow"},
            {"variant": "autoresearch_followon_v150", "label": "Autoresearch Follow-On"},
        ],
    )
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "Current Shadow" in html
    assert "Autoresearch Follow-On" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard_snapshot.py -q --tb=short`
Expected: FAIL because the writer does not accept or render shadow variants yet.

- [ ] **Step 3: Add an additive comparison section**

```python
def write_dashboard_snapshot(
    ...,
    classification_shadow_summary: dict[str, object] | None = None,
    shadow_gate_overlay: dict[str, object] | None = None,
    classification_shadow_variants: list[dict[str, object]] | None = None,
) -> Path:
    ...
    comparison_cards = ""
    if classification_shadow_variants:
        cards = []
        for variant in classification_shadow_variants:
            cards.append(
                "<div class='card'>"
                f"<div class='label'>{escape(str(variant.get('label', '-')))}</div>"
                f"<div class='value'>{escape(str(variant.get('stance', '-')))}</div>"
                f"<div class='muted'>{escape(str(variant.get('probability_actionable_sell_label', '-')))}</div>"
                "</div>"
            )
        comparison_cards = (
            "<section><h2>Shadow Variant Comparison</h2>"
            "<div class='cards'>" + "".join(cards) + "</div></section>"
        )
```

Append the section near the existing classification area.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_snapshot.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_dashboard_snapshot.py src/reporting/dashboard_snapshot.py
git commit -m "render follow-on shadow lane in dashboard"
```

## Task 5: Thread The Follow-On Lane Through Monthly Decision Orchestration

**Files:**
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\scripts\monthly_decision.py`
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\src\reporting\shadow_followon.py`
- Test: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\tests\test_shadow_followon.py`

- [ ] **Step 1: Write the failing orchestration test**

```python
def test_build_followon_shadow_payload_preserves_live_recommendation_fields():
    payload = build_followon_shadow_payload(
        probability_actionable_sell=0.56,
        confidence_tier="LOW",
        stance="NEUTRAL",
    )
    assert payload["reporting_only"] is True
    assert "recommended_sell_pct" not in payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_shadow_followon.py -q --tb=short`
Expected: FAIL until the final payload contract is settled.

- [ ] **Step 3: Add orchestration code in `monthly_decision.py`**

Implement the smallest thread-through that:

- builds the baseline `classification_shadow_summary` as it does today
- builds one follow-on payload using `build_followon_shadow_payload(...)`
- passes both lanes into:
  - `build_monthly_summary_payload(...)`
  - `write_dashboard_snapshot(...)`
  - any variant-aware artifact writer calls

Use code shaped like:

```python
classification_shadow_variants = [
    {
        "variant": "baseline_shadow",
        "label": "Current Shadow",
        **classification_shadow_summary,
    },
    build_followon_shadow_payload(
        probability_actionable_sell=classification_shadow_summary.get("probability_actionable_sell"),
        confidence_tier=classification_shadow_summary.get("confidence_tier"),
        stance=classification_shadow_summary.get("stance"),
    ),
]
```

This preserves narrow scope for the first implementation. If richer follow-on
metrics are added later, they can be layered on top.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `python -m pytest tests/test_shadow_followon.py tests/test_monthly_summary.py tests/test_dashboard_snapshot.py tests/test_classification_artifacts.py -q --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/monthly_decision.py src/reporting/shadow_followon.py src/reporting/monthly_summary.py src/reporting/dashboard_snapshot.py src/reporting/classification_artifacts.py tests/test_shadow_followon.py tests/test_monthly_summary.py tests/test_dashboard_snapshot.py tests/test_classification_artifacts.py
git commit -m "thread follow-on shadow lane through monthly reporting"
```

## Task 6: Refresh Documentation And Verify No Live-Path Regression

**Files:**
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\CHANGELOG.md`
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\ROADMAP.md`
- Modify: `C:\Users\Jeff\Documents\pgr-vesting-decision-support\docs\research\backlog.md`

- [ ] **Step 1: Add the documentation assertions as failing checks**

Document the intended release note bullets before editing code:

```markdown
- add side-by-side `autoresearch_followon_v150` shadow lane
- keep production recommendation behavior unchanged
- expose variant-aware monthly summary / dashboard / CSV artifacts
```

- [ ] **Step 2: Update docs**

Add:

- changelog entry describing the new shadow lane
- roadmap note that the follow-on lane is now visible in shadow mode only
- backlog note that this is a monitoring upgrade, not production promotion

- [ ] **Step 3: Run the final verification command**

Run: `python -m pytest tests/test_shadow_followon.py tests/test_monthly_summary.py tests/test_dashboard_snapshot.py tests/test_classification_artifacts.py tests/test_research_v149_kelly_eval.py tests/test_research_v150_neutral_band_eval.py -q --tb=short`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md ROADMAP.md docs/research/backlog.md
git commit -m "document follow-on shadow comparison lane"
```

## Task 7: Final Validation

**Files:**
- Modify: none
- Test: existing test suite

- [ ] **Step 1: Run the final focused verification sweep**

Run: `python -m pytest tests/test_shadow_followon.py tests/test_monthly_summary.py tests/test_dashboard_snapshot.py tests/test_classification_artifacts.py tests/test_research_v140_shrinkage_eval.py tests/test_research_v141_blend_eval.py tests/test_research_v142_edgar_lag_eval.py tests/test_research_v143_corr_prune_eval.py tests/test_research_v144_conformal_eval.py tests/test_research_v145_wfo_window_sweep.py tests/test_research_v146_threshold_sweep.py tests/test_research_v147_coverage_weighted_aggregate.py tests/test_research_v148_class_weight_eval.py tests/test_research_v149_kelly_eval.py tests/test_research_v150_neutral_band_eval.py -q --tb=short`
Expected: PASS

- [ ] **Step 2: Record the exact files that should change in the PR summary**

```text
src/reporting/shadow_followon.py
scripts/monthly_decision.py
src/reporting/monthly_summary.py
src/reporting/dashboard_snapshot.py
src/reporting/classification_artifacts.py
tests/test_shadow_followon.py
tests/test_monthly_summary.py
tests/test_dashboard_snapshot.py
tests/test_classification_artifacts.py
CHANGELOG.md
ROADMAP.md
docs/research/backlog.md
```

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "verify follow-on shadow lane plan"
```

## Self-Review

- Spec coverage:
  - side-by-side lane: covered by Tasks 1, 2, 4, 5
  - reporting-only behavior: covered by Tasks 1 and 5
  - artifact compatibility: covered by Tasks 2 and 3
  - documentation/monitoring positioning: covered by Task 6
- Placeholder scan:
  - no `TODO`, `TBD`, or “implement later” placeholders remain
- Type consistency:
  - variant naming is consistently `autoresearch_followon_v150`
  - additive payload names are consistently
    `classification_shadow_variants` and `decision_overlay_variants`

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-shadow-followon-v150-side-by-side-plan.md`.

Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - execute tasks in this session using executing-plans, batch execution with checkpoints
