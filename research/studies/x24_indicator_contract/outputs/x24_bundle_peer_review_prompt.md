# x24 X-Series Bundle Peer Review Prompt

Please review the current research-only x-series indicator bundle for PGR.

Current bundle:
- Structural signal: `adjusted_structural_bvps_pb_6m`
- Structural model: `ridge_bridge__no_change_pb`
- Dividend signal: `special_dividend_size_watch`
- Dividend target scale: `to_current_bvps`
- Dividend occurrence status: `underidentified_post_policy`

Existing synthesis context:
- x8 shadow readiness: `not_ready`
- x11 recommendation: `continue_research`
- x23 recommendation: `research_size_indicator_candidate`

Please challenge:
1. whether the structural 6m path and the annual dividend-size watch belong in the same future monthly research surface
2. whether the structural side still needs a stronger P/B leg before any practical packaging
3. whether the dividend-size watch should remain annual-only or be turned into a monthly tracked state variable
4. what evidence would justify promotion from research-only bundle to a reporting-only monthly/dashboard candidate

Constraints:
- strict temporal validation only
- no K-Fold CV
- no production wiring
- prefer low-complexity, small-sample-safe models
