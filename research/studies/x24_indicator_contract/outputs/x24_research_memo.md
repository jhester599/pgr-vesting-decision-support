# x24 Research Memo

## Scope

x24 packages the surviving x-series signals into one research-only
indicator bundle for later monthly-report/dashboard discussion.

## Bundle

- Structural signal: `adjusted_structural_bvps_pb_6m`.
- Structural model: `ridge_bridge__no_change_pb`.
- Dividend signal: `special_dividend_size_watch`.
- Dividend target scale: `to_current_bvps`.
- Bundle status: `research_indicator_bundle_candidate`.

## Interpretation

- x24 is packaging only. It does not wire anything into production,
  monthly outputs, or shadow artifacts.
- The structural side remains a 6m research watch anchored on no-change
  P/B. The dividend side remains a research-only annual size watch.
