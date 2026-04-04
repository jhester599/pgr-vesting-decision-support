# V14 Results Summary

Created: 2026-04-04

## Headline

v14 tested whether the underlying prediction layer can be simplified or replaced on a reduced, diversification-aware benchmark universe without changing the promoted v13.1 recommendation layer.

## Post-v13 Baseline Snapshot

- April production as-of date: `2026-04-02`
- Recommendation-layer mode: `shadow_promoted`
- Live vs simpler-baseline signal agreement at freeze point: `False`
- Live vs simpler-baseline sell agreement at freeze point: `False`

## Universe Selection

- Selected v14 forecast universe: `v13_redeploy_core8`
- Benchmarks: `VOO, VXUS, VWO, VMBS, BND, GLD, DBC, VDE`

## Final Candidate Table

- Best overall row: `ensemble_ridge_gbt`
- Best replacement candidate: `ensemble_ridge_gbt`
- Reduced-universe live stack policy / OOS R^2: `0.0629` / `-0.7285`
- Historical-mean baseline policy / OOS R^2: `0.0702` / `-0.2047`
- Replacement candidate policy / OOS R^2: `0.0707` / `-0.2569`

## Shadow Review Window

- Review snapshots: `6`
- Live signal changes: `1`
- Candidate signal changes: `1`
- Live / candidate agreement rate: `83.3%`

## Key Conclusion

`ensemble_ridge_gbt` improves on the reduced-universe live stack and stays within range of the historical-mean baseline. It is the narrow candidate worth carrying into v15 feature-replacement work.

Detailed CSV outputs are stored in `results\v14`.
