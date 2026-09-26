# x17 Research Memo

## Scope

x17 compares raw BVPS against a dividend-persistent synthetic BVPS
history, asking whether persistent book-value creation is easier
to forecast than the raw book-value level.

## Results

- 1m raw best `elastic_net_bridge` (logical_interactions, MAE 0.598) vs persistent best `elastic_net_bridge` (logical_interactions, MAE 0.754).
- 3m raw best `ridge_bridge` (bvps_lags, MAE 1.308) vs persistent best `elastic_net_bridge` (logical_interactions, MAE 1.083).
- 6m raw best `drift_bvps_growth` (baseline, MAE 2.057) vs persistent best `trailing_3m_growth` (baseline, MAE 1.980).
- 12m raw best `drift_bvps_growth` (baseline, MAE 3.190) vs persistent best `drift_bvps_growth` (baseline, MAE 4.044).

## Interpretation

- If persistent BVPS helps, it supports separating capital
  creation from dividend policy in later dividend work.
- If it does not help, the remaining challenge is more likely
  feature specification or regime change than dividend noise
  alone.
