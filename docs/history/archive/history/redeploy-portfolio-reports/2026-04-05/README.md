# Redeploy Portfolio Research Reports

Archived on `2026-04-05`.

These reports were used as external input for the v27 redeploy-portfolio work:

- `diversified_portfolio_chatgpt.md`
- `diversified_portfolio_gemini.md`

How they were used:

- The reports helped define the desired shape of the sell-proceeds answer:
  - aggressively equity-heavy
  - broad-market core
  - explicit tech tilt
  - explicit value tilt
  - some international exposure
  - at most one small bond sleeve
- They did **not** directly override the repo’s benchmark or modeling logic.
- v27 converted them into:
  - a repeatable investable redeploy universe
  - a monthly allocation process
  - a backtest over the project’s available ETF history
  - a benchmark-pruning review for funds that should remain contextual only

See:

- [V27 results](/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/results/V27_RESULTS_SUMMARY.md)
- [V27 closeout](/Users/Jeff/.codex/worktrees/3a71/pgr-vesting-decision-support/docs/closeouts/V27_CLOSEOUT_AND_V28_NEXT.md)
