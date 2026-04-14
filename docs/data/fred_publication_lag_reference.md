# FRED Publication Lag Reference

This note documents the conservative publication-lag assumptions used for the
Target 4 `v134` lag sweep. It is an internal research reference, not a live
config source. The goal is to distinguish series that are plausibly safe at
`lag=0` from series that should remain at `lag=1` or higher to avoid
look-ahead bias.

| series_id | release_name | typical_lag_days | safe_lag_months | current_config_months | recommended_min |
| --- | --- | ---: | ---: | ---: | ---: |
| `GS10` | 10-Year Treasury Constant Maturity Rate | 0-1 | 0 | 1 | 0 |
| `GS5` | 5-Year Treasury Constant Maturity Rate | 0-1 | 0 | 1 | 0 |
| `GS2` | 2-Year Treasury Constant Maturity Rate | 0-1 | 0 | 1 | 0 |
| `T10Y2Y` | 10Y minus 2Y Treasury slope | 0-1 | 0 | 1 | 0 |
| `T10YIE` | 10-Year Breakeven Inflation | 0-1 | 0 | 1 | 0 |
| `VIXCLS` | CBOE VIX close | 0-1 | 0 | 1 | 0 |
| `BAA10Y` | Moody's BAA minus 10Y Treasury spread | 0-7 | 0 | 1 | 0 |
| `BAMLH0A0HYM2` | ICE BofA US High Yield OAS | 0-7 | 0 | 1 | 0 |
| `MORTGAGE30US` | 30-Year Fixed Mortgage Average | 0-7 | 0 | 1 | 0 |
| `NFCI` | Chicago Fed National Financial Conditions Index | 7-14 | 1 | 2 | 1 |
| `TRFVOLUSM227NFWA` | Vehicle miles traveled | 45-60 | 2 | 2 | 2 |
| `CUSR0000SETC01` | Motor vehicle insurance CPI | 14-35 | 1 | 1 | 1 |
| `CUSR0000SETA02` | Used car CPI | 14-35 | 1 | 1 | 1 |
| `CUSR0000SAM2` | Medical care CPI | 14-35 | 1 | 1 | 1 |
| `PCU5241265241261` | PPI: private passenger auto insurance | 14-35 | 1 | 1 | 1 |
| `PPIACO` | Producer Price Index: all commodities | 14-35 | 1 | 1 | 1 |
| `WPU45110101` | PPI: legal services | 14-35 | 1 | 1 | 1 |
| `DTWEXBGS` | Broad trade-weighted US dollar index | 0-7 | 1 | 1 | 1 |
| `DCOILWTICO` | WTI crude oil spot price | 0-1 | 0 | 1 | 0 |
| `MRTSSM447USN` | Gasoline stations retail sales | 20-45 | 1 | 1 | 1 |
| `THREEFYTP10` | 10-Year term premium | 0-30 | 1 | 1 | 1 |
| `SP500_PE_RATIO_MULTPL` | S&P 500 PE ratio proxy | 14-35 | 1 | 1 | 1 |
| `SP500_EARNINGS_YIELD_MULTPL` | S&P 500 earnings yield proxy | 14-35 | 1 | 1 | 1 |
| `SP500_PRICE_TO_BOOK_MULTPL` | S&P 500 price/book proxy | 14-35 | 1 | 1 | 1 |

Notes:
- The `v134` candidate search is intentionally limited to the nine
  daily/weekly series listed at the top of the table.
- `recommended_min` is the research floor used to avoid unrealistic lag
  reductions during autonomous sweeps.
- `current_config_months` reflects the live repo state on 2026-04-14.
