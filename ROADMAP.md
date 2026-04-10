# PGR Vesting Decision Support — Roadmap

For full version history see [CHANGELOG.md](CHANGELOG.md).

## Current State

**Master baseline: v33** — config package modularization, expanded mypy CI
coverage, walk-forward diagnostics, conformal prediction monitoring, EDGAR
data expansion, and channel/valuation feature engineering are complete and
passing in CI.

**Active branch: `codex/v36-property-testing`** — Hypothesis property-based
tests (Tier 5.5 from the 2026-04-05 peer review) are added and pending merge.

## Near-Term Backlog

### Data Expansion (EDGAR Phase 2)
- Extend `pgr_edgar_monthly` schema with investment portfolio and capital
  allocation fields (`investment_book_yield`, `net_unrealized_gains_fixed`,
  `fixed_income_duration`, `fte_return_total_portfolio`)
- Extend HTML parser in `scripts/edgar_8k_fetcher.py` to capture all Phase 2
  fields from new live 8-K filings
- Historical backfill for expanded fields once schema is in place

### Model Promotion Readiness
- Accumulate 12 months of live OOS predictions (target: Q1 2027 validation)
- Formal BLP validation: Diebold-Mariano test vs. ensemble baseline
- Conditional coverage tests on conformal intervals

### Operational Hardening
- Monte Carlo tax scenario modeling (Tier 4.5 from 2026-04-05 peer review)
- Calibration diagnostic in monthly report (P2.7)
- Automated email delivery test coverage (P2.8)

## Strategic Backlog

| Item | Description |
|---|---|
| Conformal interval dashboard | Rolling empirical coverage plot in monthly diagnostic; alert if coverage < 70% |
| Peer comparison expansion | EDGAR 8-K fetchers for ALL, TRV, CB, HIG to add peer combined-ratio spread features |
| Property segment CAT tracking | Cross `npw_property` / `npe_property` with FRED property insurance CPI as a CAT-exposure signal |
| BLP formal OOS validation | Diebold-Mariano + conditional coverage tests once 12M of live predictions accumulate (Q1 2027) |

## Development Principles

- Never finalize a module without a passing pytest suite (CLAUDE.md mandate)
- No K-Fold cross-validation — `TimeSeriesSplit` with embargo + purge buffer only
- No `StandardScaler` across full dataset — scaler isolated within each WFO fold pipeline
- No `yfinance` — AV is the canonical price source; FMP/EDGAR for fundamentals
- Python 3.10+, strict PEP 8, full type hinting
- Approved libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `xgboost`,
  `requests`, `statsmodels`, `skfolio`, `PyPortfolioOpt`

## Monthly Decision Log

See [`results/monthly_decisions/decision_log.md`](results/monthly_decisions/decision_log.md)
for the persistent record of all automated monthly recommendations.
