# x23 Dividend Lane Peer Review Prompt

Please review the current research-only dividend lane for PGR.

Known findings:
- Post-policy regime anchored at December 1, 2018.
- Annual prediction timestamp is November month-end.
- Overlapping post-policy OOS years are currently one-class on occurrence.
- Best surviving size target is `to_current_bvps`.
- Best current row is `x10_capital_generation` / `ridge_scaled`.

Please challenge:
1. whether the post-policy occurrence problem should be modeled at all yet
2. whether normalizing special-dividend excess by current BVPS is economically sound
3. whether there are stronger but still disciplined insurer-specific capital-return features available from the repo's existing data sources
4. whether a simpler annual baseline should dominate the current ridge result
5. what evidence would justify promoting this lane into a future dashboard/monthly-indicator discussion

Constraints:
- strict temporal validation only
- no K-Fold CV
- research-only; do not suggest production wiring
- prefer low-complexity, small-sample-safe models
