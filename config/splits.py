"""
Canonical stock-split registry and reviewed non-split price moves.

This is the single source of split data for every ticker stored in
``daily_prices`` (review 2026-09-25, F03/F05). ``scripts/weekly_fetch.py``,
``scripts/apply_split_history.py`` and ``scripts/rebuild_relative_returns.py``
seed ``split_history`` from ``KNOWN_SPLITS``; ``PGR_KNOWN_SPLITS`` (used by the
v1 loaders and validators) is derived from it.

Conventions:
  - ``split_date`` is the first day the security traded split-adjusted.
  - ``split_ratio`` = new shares per old share = ``numerator / denominator``
    (4.0 for a 4-for-1 split; 0.5 for a 1-for-2 reverse split).
  - ``evidence`` says where the date and ratio were verified.

To add a split: append a row here (``scripts/detect_splits.py`` reports splits
seen in Alpha Vantage adjusted series that are missing from this list), then
run ``python scripts/rebuild_relative_returns.py`` so that stored targets are
recomputed.
"""

KNOWN_SPLITS: list[dict] = [
    # PGR — Progressive Corporation
    {"ticker": "PGR", "split_date": "1992-12-09", "split_ratio": 3.0,
     "numerator": 3.0, "denominator": 1.0, "evidence": "SEC filings (v1 config)"},
    {"ticker": "PGR", "split_date": "2002-04-23", "split_ratio": 3.0,
     "numerator": 3.0, "denominator": 1.0, "evidence": "SEC filings (v1 config)"},
    {"ticker": "PGR", "split_date": "2006-05-19", "split_ratio": 4.0,
     "numerator": 4.0, "denominator": 1.0, "evidence": "SEC filings (v1 config)"},
    # VTI — Vanguard Total Stock Market ETF
    {"ticker": "VTI", "split_date": "2008-06-20", "split_ratio": 2.0,
     "numerator": 2.0, "denominator": 1.0, "evidence": "weekly close 0.487x, no dividend"},
    # VOO — Vanguard S&P 500 ETF: 1-for-2 reverse split (review F03)
    {"ticker": "VOO", "split_date": "2013-10-24", "split_ratio": 0.5,
     "numerator": 1.0, "denominator": 2.0,
     "evidence": "Vanguard 1-for-2 reverse split, split-adjusted trading from "
                 "2013-10-24 (Fox Business; OCC infomemo #33411); weekly close "
                 "79.87 -> 161.20 while VTI +0.8 %"},
    # VWO — Vanguard FTSE Emerging Markets ETF
    {"ticker": "VWO", "split_date": "2008-06-20", "split_ratio": 2.0,
     "numerator": 2.0, "denominator": 1.0, "evidence": "weekly close 0.483x"},
    # VGT — Vanguard Information Technology ETF: 8-for-1 split (review F05)
    {"ticker": "VGT", "split_date": "2026-04-21", "split_ratio": 8.0,
     "numerator": 8.0, "denominator": 1.0,
     "evidence": "Vanguard press release 2026-03-24 'Vanguard announces share "
                 "splits for five equity index ETFs': VGT 8:1, record 2026-04-17, "
                 "split-adjusted trading from 2026-04-21"},
    # SCHD — Schwab US Dividend Equity ETF
    {"ticker": "SCHD", "split_date": "2024-10-11", "split_ratio": 3.0,
     "numerator": 3.0, "denominator": 1.0, "evidence": "weekly close 0.337x"},
    # KIE — SPDR S&P Insurance ETF
    {"ticker": "KIE", "split_date": "2017-12-01", "split_ratio": 3.0,
     "numerator": 3.0, "denominator": 1.0, "evidence": "weekly close 0.342x"},
    # CB — Chubb (peer; not a model benchmark)
    {"ticker": "CB", "split_date": "2006-04-21", "split_ratio": 2.0,
     "numerator": 2.0, "denominator": 1.0, "evidence": "weekly close 0.510x"},
    # FZROX — pre-2018 rows are a VTI proxy, so the proxy carries VTI's split
    {"ticker": "FZROX", "split_date": "2008-06-20", "split_ratio": 2.0,
     "numerator": 2.0, "denominator": 1.0, "evidence": "VTI proxy rows"},
]

# Derived view kept for the v1 loaders (split_loader, corporate_actions,
# migrate_v1_to_v2). Do not edit by hand; edit KNOWN_SPLITS instead.
PGR_KNOWN_SPLITS: list[dict] = [
    {"date": s["split_date"], "ratio": s["split_ratio"]}
    for s in KNOWN_SPLITS
    if s["ticker"] == "PGR"
]

# Weekly close-to-close moves outside [0.6, 1.7] that were reviewed and are
# genuine price moves, not splits. The DB-integrity jump guard
# (src/processing/price_integrity.py) accepts exactly these (ticker, date)
# bars. Hartford's 2008-09 moves are the financial-crisis collapse and
# rebound (both directions, several weeks, no share-count change).
KNOWN_PRICE_JUMPS: list[dict] = [
    {"ticker": "HIG", "date": d, "reason": "2008-09 financial crisis; not a split"}
    for d in (
        "2008-10-03", "2008-10-31", "2008-11-21", "2008-11-28", "2008-12-05",
        "2009-02-20", "2009-03-06", "2009-03-13", "2009-05-08",
    )
]
