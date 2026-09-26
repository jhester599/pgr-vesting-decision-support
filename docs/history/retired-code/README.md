# Retired Code (was root `archive/`)

The root `archive/` folder held one-off study scripts for research cycles
v11–v24 and one test for them. None of it ran in a workflow or in pytest.
Review 2026-09-25 (section 5, phase 4) removed it from the tree: git history
keeps every file, so this page lists them and how to get one back.

The studies' outputs are in [`research/legacy/`](../../../research/legacy/README.md)
(was `results/v11/` … `results/v24/`), and their result summaries in
[`docs/history/results/`](../results/README.md). The reusable helpers these
scripts used were promoted to `src/portfolio/`, `src/models/` and
`src/reporting/`; the `src/research/v13.py` … `v24.py` study modules remain.

## Getting a file back

The last commit with the folder is `282a6b3` (merge of PR #131, 2026-09-26):

```bash
git show 282a6b3:archive/scripts/v16_promotion_study.py > v16_promotion_study.py
git log --follow -- archive/scripts/v16_promotion_study.py   # its history
```

To run one again, turn its logic into a study under `research/studies/`
(see `CONTRIBUTING.md`, Research Studies) rather than restoring the folder.

## Files

| Path at `282a6b3` | Cycle | Purpose |
|---|---|---|
| `archive/scripts/v11_autonomous_loop.py` | v11 | Autonomous monthly loop prototype |
| `archive/scripts/v12_shadow_study.py` | v12 | Shadow model evaluation framework |
| `archive/scripts/v14_prediction_layer_study.py` | v14 | Prediction layer comparison study |
| `archive/scripts/v15_execute.py` | v15 | Feature replacement execution |
| `archive/scripts/v15_feature_replacement_setup.py` | v15 | Feature replacement setup |
| `archive/scripts/v16_promotion_study.py` | v16 | Model promotion gate study |
| `archive/scripts/v17_shadow_gate.py` | v17 | Shadow promotion gate |
| `archive/scripts/v18_bias_reduction_study.py` | v18 | Bias reduction methodology |
| `archive/scripts/v19_feature_completion.py` | v19 | Feature engineering completion |
| `archive/scripts/v20_synthesis_study.py` | v20 | Ensemble synthesis study |
| `archive/scripts/v21_historical_comparison.py` | v21 | Historical model comparison |
| `archive/scripts/v22_cross_check_promotion.py` | v22 | Cross-check promotion gate |
| `archive/scripts/v23_extended_history_proxy_study.py` | v23 | Extended history proxy study |
| `archive/scripts/v24_vti_replacement_study.py` | v24 | VTI replacement benchmark study |
| `archive/tests/test_v11_research.py` | v11 | Tests for `v11_autonomous_loop` helpers (not collected since v34.2) |
| `archive/README.md`, `archive/scripts/README.md`, `archive/tests/README.md` | — | Folder READMEs (their content is on this page) |
