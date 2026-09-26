# v138 Black-Litterman Parameter Search Summary

Bounded autonomous sweep run on 2026-04-14 against the matured `v118` replay proxy frame.

Baseline config: `tau=0.05`, `view_confidence_scalar=1.0` -> accuracy `0.8500`, coverage `0.1235`, mean_kelly_fraction `0.0151`, policy_uplift `0.0008`.

Best success-gate candidate: `tau=0.05`, `view_confidence_scalar=0.75` -> accuracy `0.8293`, coverage `0.2531`, mean_kelly_fraction `0.0200`, policy_uplift `0.0010`, sell_precision `0.0000`.

Top sweep rows:
- `tau=0.03`, `view_confidence_scalar=0.75` -> accuracy `0.8571`, coverage `0.1296`, mean_kelly_fraction `0.0157`, policy_uplift `0.0008`, sell_precision `0.0000`
- `tau=0.5`, `view_confidence_scalar=3.0` -> accuracy `0.8500`, coverage `0.1235`, mean_kelly_fraction `0.0152`, policy_uplift `0.0008`, sell_precision `0.0000`
- `tau=0.05`, `view_confidence_scalar=1.0` -> accuracy `0.8500`, coverage `0.1235`, mean_kelly_fraction `0.0151`, policy_uplift `0.0008`, sell_precision `0.0000`
- `tau=0.2`, `view_confidence_scalar=2.0` -> accuracy `0.8500`, coverage `0.1235`, mean_kelly_fraction `0.0148`, policy_uplift `0.0008`, sell_precision `0.0000`
- `tau=0.05`, `view_confidence_scalar=0.75` -> accuracy `0.8293`, coverage `0.2531`, mean_kelly_fraction `0.0200`, policy_uplift `0.0010`, sell_precision `0.0000`
- `tau=0.2`, `view_confidence_scalar=1.5` -> accuracy `0.8286`, coverage `0.2160`, mean_kelly_fraction `0.0194`, policy_uplift `0.0010`, sell_precision `0.0000`
- `tau=0.35`, `view_confidence_scalar=2.0` -> accuracy `0.8286`, coverage `0.2160`, mean_kelly_fraction `0.0190`, policy_uplift `0.0010`, sell_precision `0.0000`
- `tau=0.5`, `view_confidence_scalar=2.0` -> accuracy `0.8222`, coverage `0.2778`, mean_kelly_fraction `0.0220`, policy_uplift `0.0012`, sell_precision `0.2000`
- `tau=0.1`, `view_confidence_scalar=1.0` -> accuracy `0.8182`, coverage `0.2716`, mean_kelly_fraction `0.0209`, policy_uplift `0.0011`, sell_precision `0.0000`
- `tau=0.1`, `view_confidence_scalar=1.5` -> accuracy `0.8125`, coverage `0.0988`, mean_kelly_fraction `0.0142`, policy_uplift `0.0007`, sell_precision `0.0000`

Interpretation:
- The replay proxy strongly prefers moderate `tau` with slightly tighter view uncertainty than the baseline.
- Several high-accuracy rows achieve coverage below the plan floor, so the chosen candidate is the best row that also satisfies `coverage >= 0.25` and `mean_kelly_fraction <= 0.10`.
- The top candidate still rarely emits correct reduce signals (`sell_precision` remains weak), so this target improves decision alignment on the proxy frame without yet proving a promotable sell overlay.
