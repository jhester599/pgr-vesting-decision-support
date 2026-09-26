# Legacy result folders (v9–v28)

The top-level `results/vNN/` folders from the v9–v28 research cycles
(April 2026), moved here unchanged with `git mv` in review 2026-09-25,
section 5, phase 3. They are read-only records: nothing in the monthly run
reads them, and no script writes here. `git log --follow` shows each file's
history from its old path.

| Folder | From | Written by | Closeout |
|---|---|---|---|
| `v9/` | `results/v9/` | `research/studies/v9_experiments/` (feature, target, pooled-benchmark, confirmatory-classifier and weekly-snapshot experiments) and the other v9 scripts still in `scripts/` (`benchmark_suite.py`, `benchmark_reduction.py`, `candidate_model_bakeoff.py`, `classifier_feature_selection.py`, `feature_cost_report.py`, `policy_evaluation.py`, `regime_slice_backtest.py`) | [V9 closeout](../../docs/closeouts/V9_CLOSEOUT_AND_V91_NEXT.md) |
| `v11/` | `results/v11/` | `archive/scripts/v11_autonomous_loop.py` | [V11 closeout](../../docs/closeouts/V11_CLOSEOUT_AND_V12_NEXT.md) |
| `v12/` | `results/v12/` | `archive/scripts/v12_shadow_study.py` | [V12 closeout](../../docs/closeouts/V12_CLOSEOUT_AND_V13_NEXT.md) |
| `v14/` | `results/v14/` | `archive/scripts/v14_prediction_layer_study.py` | [V14 closeout](../../docs/closeouts/V14_CLOSEOUT_AND_V15_NEXT.md) |
| `v15/` | `results/v15/` | `archive/scripts/v15_*.py` | [V15 closeout](../../docs/closeouts/V15_CLOSEOUT_AND_V16_NEXT.md) |
| `v16/` | `results/v16/` | `archive/scripts/v16_promotion_study.py` | [V16 closeout](../../docs/closeouts/V16_CLOSEOUT_AND_V17_NEXT.md) |
| `v17/` | `results/v17/` | `archive/scripts/v17_shadow_gate.py` | [V17 closeout](../../docs/closeouts/V17_CLOSEOUT_AND_V18_NEXT.md) |
| `v18/` | `results/v18/` | `archive/scripts/v18_bias_reduction_study.py` | [V18 closeout](../../docs/closeouts/V18_CLOSEOUT_AND_V19_NEXT.md) |
| `v19/` | `results/v19/` | `archive/scripts/v19_feature_completion.py` | [V19 closeout](../../docs/closeouts/V19_CLOSEOUT_AND_V20_NEXT.md) |
| `v20/` | `results/v20/` | `archive/scripts/v20_synthesis_study.py` | [V20 closeout](../../docs/closeouts/V20_CLOSEOUT_AND_V21_NEXT.md) |
| `v21/` | `results/v21/` | `archive/scripts/v21_historical_comparison.py` | [V21 closeout](../../docs/closeouts/V21_CLOSEOUT_AND_V22_NEXT.md) |
| `v22/` | `results/v22/` | `archive/scripts/v22_cross_check_promotion.py` | [V22 closeout](../../docs/closeouts/V22_CLOSEOUT_AND_V23_NEXT.md) |
| `v23/` | `results/v23/` | `archive/scripts/v23_extended_history_proxy_study.py` | [V23 closeout](../../docs/closeouts/V23_CLOSEOUT_AND_V24_NEXT.md) |
| `v24/` | `results/v24/` | `archive/scripts/v24_vti_replacement_study.py` | [V24 closeout](../../docs/closeouts/V24_CLOSEOUT_AND_V25_NEXT.md) |
| `v27/` | `results/v27/` | `scripts/v27_redeploy_portfolio_study.py` | [V27 closeout](../../docs/closeouts/V27_CLOSEOUT_AND_V28_NEXT.md) |
| `v28/` | `results/v28/` | `scripts/v28_forecast_universe_review.py` | [V28 closeout](../../docs/closeouts/V28_CLOSEOUT_AND_V29_NEXT.md) |

The archived and remaining scripts above still default to `results/vNN/` as
their output folder; a re-run writes a new, untracked `results/vNN/` and
leaves this folder as it is.

## Large files kept as they are

The move is unchanged on purpose, so three `*_detail*.csv` files over 1 MB
stay committed here: `v9/classifier_feature_selection_detail_20260403.csv`
(9.7 MB), `v9/regime_slice_detail_20260403.csv` (1.7 MB) and
`v27/v27_redeploy_backtest_detail_20260405.csv` (1.0 MB). The 1 MB rule for
detail files applies to `research/studies/` and everything outside
`research/legacy/`.
