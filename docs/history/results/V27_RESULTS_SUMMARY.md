# V27 Results Summary

Created: 2026-04-05

## Objective

v27 answers a practical question that the monthly sell / hold output had not
yet answered clearly enough:

- if the user sells some PGR shares, what should the proceeds be used for?

The goal was not to replace the existing recommendation layer. The goal was to
add a repeatable, backtested redeploy portfolio that:

- stays aggressively equity-heavy
- reflects the user's stated preference for broad market, tech, value, and a
  smaller international sleeve
- keeps the bond allocation small
- uses only funds already present in the repo
- tilts weights based on projected return and confidence without becoming a
  high-turnover tactical portfolio

## External Research Inputs

Archived under:

- `docs/history/redeploy-portfolio-reports/2026-04-05/diversified_portfolio_chatgpt.md`
- `docs/history/redeploy-portfolio-reports/2026-04-05/diversified_portfolio_gemini.md`

The reports were used to shape the default portfolio profile, not to override
the repo's evidence-based backtests.

## Design Decision

v27 intentionally separates two universes:

1. forecast benchmark universe
2. investable redeploy universe

That split matters.

Some funds remain useful as forecast benchmarks or contextual comparison points
even if they are not good destinations for capital leaving a concentrated PGR
position. Examples include:

- `VFH`
- `KIE`
- `GLD`
- `DBC`
- narrow sector sleeves such as `VDE`, `VPU`, and `VHT`

For the monthly redeploy answer, v27 narrows the live investable set to:

- `VOO`
- `VGT`
- `SCHD`
- `VXUS`
- `VWO`
- `BND`

Optional substitutes documented for research purposes:

- `VTI`
- `VIG`

## Backtested Base Portfolios

The study compared four report-inspired base portfolios:

- `chatgpt_proxy_90_10`
- `gemini_proxy_100_equity`
- `balanced_pref_95_5`
- `hybrid_equity_95_5`

Each base portfolio was then tested with bounded signal tilts at:

- `0.25`
- `0.35`
- `0.45`

The tilt process uses:

- per-fund predicted relative return
- per-fund confidence proxies from IC and hit rate
- diversification score versus PGR

The process is intentionally bounded so the monthly output remains stable and
readable rather than becoming a highly tactical allocation engine.

## Winning Process

Selected monthly redeploy default:

- strategy: `balanced_pref_95_5`
- tilt strength: `0.25`

This is the repeatable v27 answer because it best balanced:

- user-preference fit
- annualized return
- volatility
- correlation to PGR
- turnover

Selected base structure:

- `VOO` core US equity
- `VGT` technology tilt
- `SCHD` value / dividend tilt
- `VXUS` international core
- `VWO` emerging-markets satellite
- `BND` small bond ballast

## Best Backtest Row

From `results/v27/v27_redeploy_backtest_summary_20260405.csv`:

- annualized return: `11.77%`
- annualized volatility: `16.12%`
- correlation to PGR: `0.260`
- max drawdown: `-24.29%`
- mean turnover: `3.46%` per monthly rebalance
- stock / bond mix: `95% / 5%`
- evaluated months: `96`

## Monthly Workflow Outcome

The monthly workflow now produces a new report and email section:

- `Suggested Redeploy Portfolio`

That section is the answer to:

- if the user sells some PGR shares, what should the proceeds be used for?

The recommendation remains dynamic, but only within bounded ranges. The default
portfolio keeps a high-equity shape and tilts weights modestly toward funds
with:

- stronger benchmark-outperformance signals
- stronger diversification characteristics versus PGR
- better confidence support

## Benchmark Pruning Conclusion

v27 does support pruning for the monthly buy recommendation.

Recommended keepers for the redeploy answer:

- `VOO`
- `VGT`
- `SCHD`
- `VXUS`
- `VWO`
- `BND`

Recommended contextual-only or non-default names:

- `VFH`
- `KIE`
- `GLD`
- `DBC`
- `VNQ`
- `BNDX`
- `VCIT`
- `VMBS`
- `VDE`
- `VPU`
- `VHT`
- `VIS`
- `VEA`

Important nuance:

- v27 does **not** automatically prune the production forecast benchmark
  universe
- it prunes the monthly *investable redeploy* answer first

That avoids silently changing the prediction layer while still making the user
facing portfolio recommendation more realistic immediately.

## Practical Recommendation

Use the v27 redeploy portfolio in each monthly run as the default answer for
where sold PGR proceeds should go.

Default process:

1. start from `balanced_pref_95_5`
2. keep equity exposure above `90%`
3. use only modest signal tilts
4. cap the portfolio at a small, readable set of funds
5. keep contextual benchmarks out of the buy list unless a later research cycle
   promotes them explicitly
