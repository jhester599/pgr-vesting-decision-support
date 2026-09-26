# x14 Indicator Synthesis Memo

## Scope

x14 translates the adjusted decomposition research into a possible
research-only indicator candidate for monthly report/dashboard
discussion. It does not change production or shadow outputs.
The synthesis explicitly reads x3, x5, x8, x11, x12, and x13
research artifacts.

## Raw vs Adjusted Comparison

- 3m adjusted `elastic_net_bridge__no_change_pb` did not beat raw `drift_bvps_growth__no_change_pb` (delta MAE 0.821).
- 6m adjusted `ridge_bridge__no_change_pb` beat raw `drift_bvps_growth__no_change_pb` (delta MAE -0.468).

## Horizon Evidence

- 3m: x12 adjusted gate=`True`, x13 adjusted gate=`False`, x5 anchor=`drift_bvps_growth__no_change_pb`, x3 return leader=`no_change`, x3 log-return leader=`no_change`.
- 6m: x12 adjusted gate=`True`, x13 adjusted gate=`True`, x5 anchor=`drift_bvps_growth__no_change_pb`, x3 return leader=`drift`, x3 log-return leader=`ridge_log_return`.

## Recommendation

- Status: `research_indicator_candidate`.
- Rationale: One bounded indicator candidate is worth carrying into a later monthly-report/dashboard discussion, but not into production yet.

## Candidate

- Proposed indicator: `adjusted_structural_bvps_pb` at 6m using `ridge_bridge__no_change_pb`.
- Intended use: show as a research-only structural medium-term readout on the monthly report/dashboard, pending a later plan.
