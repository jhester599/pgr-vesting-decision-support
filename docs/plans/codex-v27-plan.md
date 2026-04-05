# v27 Redeploy Portfolio Plan

Created: 2026-04-05

## Goal

Answer the practical monthly follow-up question:

- if the user sells some PGR shares, what should the proceeds be used for?

The v27 scope is deliberately narrower than the prediction-layer research:

- keep the current sell/hold recommendation layer intact
- build a repeatable redeploy portfolio process
- keep the recommendation aggressively equity-heavy
- let signals and confidence tilt allocations modestly rather than replacing the base portfolio every month
- prune the redeploy universe to funds the user could realistically buy

## External Inputs

Archived under:

- `docs/history/redeploy-portfolio-reports/2026-04-05/diversified_portfolio_chatgpt.md`
- `docs/history/redeploy-portfolio-reports/2026-04-05/diversified_portfolio_gemini.md`

## Desired Portfolio Shape

- >90% equities
- broad-market US core
- explicit tech tilt
- explicit value tilt
- smaller international sleeve
- one bond fund at most
- fewer than 10 funds

## Candidate Design

Use only funds already present in the project universe for the live monthly answer.

Primary investable candidates:

- `VOO`
- `VGT`
- `SCHD`
- `VXUS`
- `VWO`
- `BND`

Optional / research substitutes:

- `VTI`
- `VIG`

Context-only funds:

- `VFH`
- `KIE`

## Process

1. Define a stable high-equity base portfolio.
2. Use current monthly benchmark signals to tilt the weights modestly.
3. Keep the tilts bounded so the monthly answer remains readable and repeatable.
4. Backtest several report-inspired base profiles.
5. Choose the default monthly process using:
   - user-preference fit
   - annualized return
   - volatility
   - correlation to PGR
   - turnover
6. Add the chosen portfolio recommendation to:
   - `recommendation.md`
   - monthly email output

## Expected Outputs

- `results/v27/v27_redeploy_backtest_summary_<date>.csv`
- `results/v27/v27_redeploy_backtest_detail_<date>.csv`
- `results/v27/v27_benchmark_pruning_review_<date>.csv`
- `docs/results/V27_RESULTS_SUMMARY.md`
- `docs/closeouts/V27_CLOSEOUT_AND_V28_NEXT.md`
