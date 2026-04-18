# v160 Technical Analysis Research Reports

Created: 2026-04-18

## Purpose

These files archive the three external deep-research inputs used to define the
`v160-v164` technical-analysis feature research arc.

They are kept in-repo so future feature work can trace:

- which Alpha Vantage indicator ideas were considered
- where the reports agreed and disagreed
- why the implementation uses a broad-but-pruned screen rather than dumping all
  technical indicators into the model at once

## Archived Reports

- `v160_claude_ta_research_20260418.md`
- `v160_gemini_ta_research_20260418.md`
- `v160_chatgpt_ta_research_20260418.md`

## High-Level Takeaways

- Consensus: technical analysis has a low prior for PGR, but one disciplined
  research-only arc is defensible.
- Primary target: binary PGR-vs-benchmark outperformance classification.
- Secondary target: continuous 6-month relative-return regression.
- Strongest constructs:
  - ratio-series momentum
  - EMA/SMA distance
  - RSI and Bollinger `%B` mean reversion
  - NATR and ADX regime conditioning
  - PC-TECH-like benchmark regime composites
  - peer-relative PGR-vs-insurance-composite signals
- Strongest warning: broad TA screening creates severe false-positive risk, so
  the screen must collapse redundant indicator families before model fitting and
  use strict WFO-only validation.

## Repo Integration

The execution plan built from these reports is:

- `docs/superpowers/plans/2026-04-18-v160-v164-technical-analysis-feature-research.md`

The research-only feature factory and harness files are:

- `src/research/v160_ta_features.py`
- `results/research/v162_ta_broad_screen.py`
- `results/research/v163_ta_survivor_confirm.py`
- `results/research/v164_ta_synthesis_summary.md`
- `results/research/v164_ta_candidate.json`
