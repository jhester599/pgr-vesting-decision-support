# 0007 — ACTIONABLE sell-percentage mapping (review 2026-09-25, step 6)

| | |
|---|---|
| **Status** | Accepted, live since v179 |
| **Date** | 2026-09-26 |
| **Where it lives** | `src/reporting/decision_rendering.sell_pct_from_consensus`; backtest in `src/models/live_policy_backtest.py` |
| **Findings** | F20 (mapping) of [`REPO_REVIEW_2026-09-25.md`](../reviews/REPO_REVIEW_2026-09-25.md) |
| **Report** | [`2026-09-25_step6_decision_tax_timing_ops.md`](../reviews/2026-09-25_step6_decision_tax_timing_ops.md) |

## Context

The mapping from consensus to sell percentage, used only when every gate of
[0006](0006-validation-gates-and-cpcv-diagnostic.md) passes, had never been
backtested (F20). Replayed on the realised-only OOS record it lost money
against always selling the 50 % default (−0.11 pp per decision), and a
bullish consensus could sell 75 %.

## Decision

| Consensus | Mean forecast | Sell % |
|---|---|---|
| OUTPERFORM | > 15 % | 25 % |
| OUTPERFORM | ≤ 15 % | 50 % (was 75 % at ≤ 5 %) |
| UNDERPERFORM | any | 100 % |
| NEUTRAL, or IC < 0.05 / missing | any | 50 % |

A bullish consensus never sells more than the 50 % default.

## Evidence

The live consensus and mapping were replayed at 186 OOS dates
(2010-09 → 2026-02) of the realised-only record as of 2026-09-21, as if the
gate had passed every month (fixture
`tests/fixtures/live_mapping_oos_panel_2026-09-21.csv`). Mean relative return
kept per decision:

| Policy | Mean per decision |
|---|---:|
| Old mapping | +3.53 % |
| New mapping | +3.81 % |
| Always 50 % | +3.64 % |

The uplift over always-50 % is +0.17 pp per decision (was −0.11 pp).

## Consequences

- `tests/unit/models/test_live_policy_backtest.py` requires the uplift to stay
  ≥ 0, and the monthly "Decision Policy Backtest" section scores this mapping.
- The UNDERPERFORM → 100 % cell rests on one historical decision (it lost
  7.2 pp). It is kept; the regression test will catch it if it starts to cost
  money.
- With the gates as they stand no month is ACTIONABLE, so the mapping has not
  yet set a live sell percentage.
