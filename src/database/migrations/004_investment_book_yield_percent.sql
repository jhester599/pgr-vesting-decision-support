-- Migration 004: store pgr_edgar_monthly.investment_book_yield in percent.
--
-- Review 2026-09-25, finding F10. The historical CSV stores the book yield in
-- percent (e.g. 3.8 for 3.8 %), but the live 8-K parser divided by 100 before
-- upserting, so 33 rows from 2023-04 onward were stored as fractions
-- (0.03-0.043). The parser no longer divides; this rescales the stored
-- fractions. A book yield below 1 % has never been reported by PGR, so any
-- value below 1 is a fraction. ROUND removes float noise (0.041 * 100).
UPDATE pgr_edgar_monthly
SET investment_book_yield = ROUND(investment_book_yield * 100.0, 6)
WHERE investment_book_yield IS NOT NULL
  AND investment_book_yield < 1;
