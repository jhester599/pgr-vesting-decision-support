# x2 Research Memo

## Scope

x2 runs the first research-only absolute PGR direction classification
baseline. It does not alter production or monthly shadow artifacts.

## Results By Horizon

- 1m best balanced-accuracy row: `base_rate` (BA 0.507, Brier 0.255); base-rate BA 0.507, Brier 0.255; did not clear base-rate gate.
- 3m best balanced-accuracy row: `logistic_l2_balanced` (BA 0.595, Brier 0.382); base-rate BA 0.557, Brier 0.230; did not clear base-rate gate.
- 6m best balanced-accuracy row: `hist_gbt_depth2` (BA 0.501, Brier 0.298); base-rate BA 0.500, Brier 0.212; did not clear base-rate gate.
- 12m best balanced-accuracy row: `base_rate` (BA 0.623, Brier 0.185); base-rate BA 0.623, Brier 0.185; did not clear base-rate gate.

## Interpretation

- Treat any apparent edge as preliminary until x3 return-regression
  and x4/x5 decomposition benchmarks exist.
- x2 uses horizon-specific WFO gaps and fold-local preprocessing.
- x7 should revisit TA only as a bounded follow-up, not as broad
  indicator dumping.
