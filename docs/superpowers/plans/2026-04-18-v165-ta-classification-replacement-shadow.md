# v165 TA Classification Replacement Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-only, reporting-only TA classification replacement harness that follows the v164 recommendation without changing production decisions.

**Architecture:** Add a v165 research script under `results/research/` that joins the existing monthly feature matrix with v160 TA features, evaluates a tiny replacement candidate set against the current lean classification baseline, writes benchmark/regime/prediction-level artifacts, and emits a current-month shadow snapshot. Keep live monthly decision code unchanged.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn via existing project WFO/classifier helpers, pytest.

---

### Task 1: Candidate Definitions And Unit Tests

**Files:**
- Create: `tests/test_research_v165_ta_shadow_replacement.py`
- Create: `results/research/v165_ta_shadow_replacement_eval.py`

- [ ] **Step 1: Write failing tests**

```python
def test_candidate_variants_are_replacement_only() -> None:
    from results.research.v165_ta_shadow_replacement_eval import build_candidate_variants

    variants = build_candidate_variants()
    by_name = {variant["variant"]: variant for variant in variants}

    assert "lean_baseline" in by_name
    assert "ta_minimal_replacement" in by_name
    assert by_name["ta_minimal_replacement"]["feature_swaps"] == {
        "mom_12m": "ta_pgr_obv_detrended",
        "vol_63d": "ta_pgr_natr_63d",
    }
    assert all(not variant["variant"].endswith("all_ta") for variant in variants)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python -m pytest tests/test_research_v165_ta_shadow_replacement.py -q --tb=short
```

Expected: fail because the v165 module does not exist yet.

- [ ] **Step 3: Implement candidate helpers**

Add `apply_feature_swaps()`, `build_candidate_variants()`, and baseline-delta helpers in `results/research/v165_ta_shadow_replacement_eval.py`.

- [ ] **Step 4: Verify tests pass**

Run:

```bash
python -m pytest tests/test_research_v165_ta_shadow_replacement.py -q --tb=short
```

Expected: pass.

### Task 2: Empirical Harness

**Files:**
- Modify: `results/research/v165_ta_shadow_replacement_eval.py`

- [ ] **Step 1: Add data-loading and WFO evaluation**

Use `build_feature_matrix_from_db()`, `db_client.get_prices()`, `build_ta_feature_matrix()`, `load_relative_return_matrix()`, `get_X_y_relative()`, `build_target_series()`, and `evaluate_confirmatory_classifier()`.

- [ ] **Step 2: Write deterministic artifacts**

Write:

- `results/research/v165_ta_shadow_replacement_detail.csv`
- `results/research/v165_ta_shadow_replacement_predictions.csv`
- `results/research/v165_ta_shadow_replacement_summary.csv`
- `results/research/v165_ta_shadow_replacement_regime_slices.csv`
- `results/research/v165_ta_shadow_current.csv`
- `results/research/v165_ta_shadow_candidate.json`

- [ ] **Step 3: Run empirical command**

Run:

```bash
python results/research/v165_ta_shadow_replacement_eval.py
```

Expected: all artifacts are written under `results/research/`.

### Task 3: Synthesis And Verification

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `ROADMAP.md`
- Modify: `docs/research/backlog.md`
- Create: `docs/closeouts/V165_CLOSEOUT_AND_HANDOFF.md`

- [ ] **Step 1: Update docs**

Record whether the minimal replacement candidate should remain shadow-only, be narrowed further, or be abandoned.

- [ ] **Step 2: Run verification**

Run:

```bash
python -m pytest tests/test_research_v165_ta_shadow_replacement.py -q --tb=short
python -m pytest tests/test_research_v160_ta_features.py tests/test_research_v162_ta_screen.py tests/test_research_v163_ta_confirm.py -q --tb=short
python -m pytest tests/test_classification_shadow.py tests/test_classification_artifacts.py tests/test_path_b_classifier.py -q --tb=short
python -m ruff check results/research/v165_ta_shadow_replacement_eval.py tests/test_research_v165_ta_shadow_replacement.py
```

Expected: all pass.
