# Research Memo — Is a Large PGR Decline a Forward Buying Signal?

**Date:** 2026-07-15
**Trigger:** PGR fell ~-9% intraday on 2026-07-15.
**Script:** `scripts/research/large_decline_buy_signal.py`
**Outputs:** `results/research/large_decline_buy_signal_detail.csv`, `..._summary.csv`

---

## Question

When PGR suffers a large single-bar decline, are forward returns (3 / 6 / 12
months out) above average — i.e. is the sell-off a buying signal — both in
absolute terms and relative to the S&P 500?

## Headline answer

**Yes at 3 and 6 months; no durable edge at 12 months.**

| Horizon | PGR absolute (mean / % positive) | vs S&P 500 (mean / % beat) |
|---------|----------------------------------|----------------------------|
| 3 months  | **+18.9%** / 90% positive | **+9.4%** / 85% beat market |
| 6 months  | **+30.3%** / 95% positive | **+11.4%** / 85% beat market |
| 12 months | **+48.4%** / 90% positive | **+1.4%** / 46% beat market |

- Buying PGR after a large decline was **almost always profitable in absolute
  terms** (90–95% of windows positive at every horizon).
- The *market-relative* edge is concentrated in the **first 3–6 months**: PGR
  beat the S&P proxy ~85% of the time and by a large margin. By 12 months the
  relative return is a coin flip (median slightly negative) — PGR keeps rising,
  but so does the market, and the initial alpha mean-reverts.
- The exact-S&P (VOO) subset for post-2010 events tells the same story
  (+6.8% / +11.3% / +0.8% at 3/6/12m).

**Read:** a large PGR drop has historically been a good *tactical* entry
(strong 3–6 month bounce with market-beating returns), but not a source of
lasting 12-month outperformance versus simply owning the index.

---

## ⚠️ Data caveat — read before citing these numbers

The `daily_prices` table is misleadingly named. It is sourced from Alpha
Vantage **`TIME_SERIES_WEEKLY`** (free tier) — the bars are **weekly Friday
closes, not daily**. True daily "-8% day" detection is **not possible** with
the on-hand data (it needs a premium AV subscription + API key, neither of
which is configured in this environment).

Consequently this study detects **weekly declines of ≥ 8%**. That is a
*stricter* bar than a daily -8% move, so the 21 events here are a conservative
subset of "big down move" episodes. The intraday -9% on 2026-07-15 that
motivated the study is not yet in the DB (data ends 2026-07-10), and a -9%
*intraday* move may or may not close the *week* down ≥8%.

If you can supply daily price history (or an AV premium key), the same script
generalizes to a true daily study — only the input series would change.

## Other methodology notes

- **Returns are DRIP total returns** (split- and dividend-adjusted), computed
  with the repo's own `total_return.build_position_series` — the same engine
  the model's targets use. Event detection uses split-adjusted *price* return
  (dividends off) so a -8% bar reflects a genuine price drop.
- **S&P 500 proxy = VTI** (Vanguard Total Stock Market, ~0.99 correlation with
  the S&P 500), the repo's existing benchmark. VTI history starts 2001-06, so
  the 8 events in 2000–2001 have **no market-relative figure** (PGR absolute is
  still valid). VOO (the exact S&P 500 ETF, 2010+) is reported as a secondary
  cross-check. This is why the relative-return sample is n=13 (VTI) / n=5 (VOO)
  vs n=21 absolute.
- Splits were correctly neutralized: the raw -66.7% (2002) and -74.6% (2006)
  "declines" in the unadjusted close are split artifacts and are **excluded**.
- All 21 events have complete 12-month forward windows (most recent event
  2023-07-14), so no result is truncated by end-of-data.

## Events (21 total, weekly decline ≥ 8%)

Concentrated in three stress regimes: the 2000–01 dot-com/insurance
repricing (8 events), the 2007–09 GFC (8 events), and 2018/2020-COVID/2023
(5 events). Full per-event forward returns are in
`large_decline_buy_signal_detail.csv`.

The single worst 12-month relative outcome was the **2020-03-20** COVID-bottom
event (-41.5% vs market): PGR bounced hard (+40% abs) but the broad index
bounced far harder off the same bottom. The best was **2023-07-14** (+58.9% vs
market over 12m).
