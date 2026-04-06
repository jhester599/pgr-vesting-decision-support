# archive/scripts/

Completed one-time research study scripts for v11–v24.

These scripts ran their respective research cycles and produced the
artifacts in `results/v11/`–`results/v24/`.  They are preserved here
for reference and auditability but will not run again as part of the
regular workflow.

The companion `src/research/v11.py`–`v24.py` modules remain in
`src/research/` because they export utility functions and dataclasses
that are imported by `scripts/monthly_decision.py` and other production
scripts.  A future refactor will promote those utility functions into
proper production modules and retire the versioned research files.

## Contents

| Script | Research Cycle | Purpose |
|--------|---------------|---------|
| `v11_autonomous_loop.py` | v11 | Autonomous monthly loop prototype |
| `v12_shadow_study.py` | v12 | Shadow model evaluation framework |
| `v14_prediction_layer_study.py` | v14 | Prediction layer comparison study |
| `v15_execute.py` | v15 | Feature replacement execution |
| `v15_feature_replacement_setup.py` | v15 | Feature replacement setup |
| `v16_promotion_study.py` | v16 | Model promotion gate study |
| `v17_shadow_gate.py` | v17 | Shadow promotion gate |
| `v18_bias_reduction_study.py` | v18 | Bias reduction methodology |
| `v19_feature_completion.py` | v19 | Feature engineering completion |
| `v20_synthesis_study.py` | v20 | Ensemble synthesis study |
| `v21_historical_comparison.py` | v21 | Historical model comparison |
| `v22_cross_check_promotion.py` | v22 | Cross-check promotion gate |
| `v23_extended_history_proxy_study.py` | v23 | Extended history proxy study |
| `v24_vti_replacement_study.py` | v24 | VTI replacement benchmark study |
