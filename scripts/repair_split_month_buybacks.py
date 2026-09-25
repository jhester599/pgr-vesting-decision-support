#!/usr/bin/env python3
"""Record the 2006-05 split-month buybacks from the Q2 2006 10-Q (review step 4c).

The May 2006 8-K (0000950152-06-005098) prints 2.3M shares repurchased and
an average cost of "NM".  Its footnote says the 2.3M adds 0.3M pre-split
shares (at $107.94) and 2.0M post-split shares (at $27.16).  The Q2 2006 10-Q
(0000950152-06-006431, Part II Item 2) has the exact counts:

    May pre-split     331,496 at $107.94
    May post-split  1,932,200 at $27.16

This script:

1. fetches the 10-Q and the 8-K exhibit (cached, throttled to 4 req/s, sent
   with the ``EDGAR_USER_AGENT`` header, which must be set);
2. parses the 10-Q table (``src/ingestion/edgar_10q_repurchases``) and checks
   it against the 8-K footnote, the April and June 8-K rows, and the 10-Q's
   stated Q2 total cost;
3. appends the parse to ``pgr_edgar_filing_parses`` / ``pgr_edgar_monthly_raw``
   (per-leg values as parsed; the month on the post-split basis as derived);
4. applies it to ``pgr_edgar_monthly`` (``apply_pgr_edgar_supplements``):
   ``shares_repurchased`` 2.3 → 3.258184 and ``avg_cost_per_share`` NULL →
   27.09, and recomputes the derived fields.

It works on the DB given by ``--db`` and refuses the committed DB unless
``--allow-committed-db`` is passed.  Re-running it is a no-op: the parse is
recorded once per (accession, parser version).

Usage::

    cp data/pgr_financials.db /tmp/step4c.db
    EDGAR_USER_AGENT="Name email@example.com" \\
        python scripts/repair_split_month_buybacks.py --db /tmp/step4c.db \\
        --export-csv data/processed/pgr_edgar_cache.csv
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from scripts import edgar_8k_fetcher as fetcher  # noqa: E402
from scripts.repair_edgar_history import validate  # noqa: E402
from src.database import db_client  # noqa: E402
from src.ingestion import edgar_10q_repurchases as q10  # noqa: E402

log = logging.getLogger("repair_split_month_buybacks")

MONTH_END = "2006-05-31"
MONTH_LABEL = "May"
TEN_Q = {
    "accession_number": "0000950152-06-006431",
    "filing_date": "2006-08-03",
    "url": "https://www.sec.gov/Archives/edgar/data/80661/000095015206006431/l21391ae10vq.htm",
}
# Q2 2006 MD&A: "we repurchased 7.1 million Common Shares, at a total cost of
# $267.5 million".
_Q2_TOTAL_COST = re.compile(
    r"during the second quarter 2006, we repurchased [0-9.]+ million common shares, "
    r"at a total cost of \$([0-9,.]+) million"
)
# 8-K footnote: "Includes .3 million Common Shares repurchased prior to our
# 4-for-1 stock split at an average cost of $107.94 per share and 2.0 million
# Common Shares repurchased after the stock split at an average cost of $27.16".
_EIGHT_K_FOOTNOTE = re.compile(
    r"includes ([0-9]*\.?[0-9]+) million common shares repurchased prior to our "
    r"4-for-1 stock split at an average cost of \$([0-9.]+) per share and "
    r"([0-9]*\.?[0-9]+) million common shares repurchased after the stock split "
    r"at an average cost of \$([0-9.]+)"
)


def _plain_text(markup: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", markup))
    return " ".join(text.replace("’", "'").split()).lower()


def _split_ratio(conn: sqlite3.Connection) -> float:
    splits = db_client.get_splits(conn, "PGR")
    in_month = splits[splits.index.strftime("%Y-%m") == MONTH_END[:7]]
    if len(in_month) != 1:
        raise RuntimeError(f"expected one PGR split in {MONTH_END[:7]}, found {len(in_month)}")
    return float(in_month["split_ratio"].iloc[0])


def _row(conn: sqlite3.Connection, month_end: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT month_end, accession_number, shares_repurchased, avg_cost_per_share "
        "FROM pgr_edgar_monthly WHERE month_end = ?",
        (month_end,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"pgr_edgar_monthly has no {month_end} row")
    return row


def cross_check(
    conn: sqlite3.Connection,
    rows: list[q10.PurchaseRow],
    purchases: q10.SplitMonthPurchases,
    ten_q_text: str,
    eight_k_text: str,
) -> list[str]:
    """Return the failed checks (empty if the 10-Q agrees with everything)."""
    problems: list[str] = []

    # 1. The 8-K footnote: same prices, counts that round to the printed ones.
    match = _EIGHT_K_FOOTNOTE.search(eight_k_text)
    if match is None:
        problems.append("8-K footnote on the pre/post-split split not found")
    else:
        pre_m, pre_px, post_m, post_px = (float(g) for g in match.groups())
        if pre_px != purchases.pre_split_average_price:
            problems.append(f"pre-split price: 8-K {pre_px} vs 10-Q {purchases.pre_split_average_price}")
        if post_px != purchases.post_split_average_price:
            problems.append(f"post-split price: 8-K {post_px} vs 10-Q {purchases.post_split_average_price}")
        # The 8-K rounds 331,496 to .3 and 1,932,200 to 2.0 (not 1.9); allow 0.1M.
        if abs(pre_m - purchases.pre_split_shares_m) > 0.1:
            problems.append(f"pre-split shares: 8-K {pre_m}M vs 10-Q {purchases.pre_split_shares_m}M")
        if abs(post_m - purchases.post_split_shares_m) > 0.1:
            problems.append(f"post-split shares: 8-K {post_m}M vs 10-Q {purchases.post_split_shares_m}M")

    # 2. The row's printed 2.3M is the unadjusted pre + post count.
    may = _row(conn, MONTH_END)
    printed = conn.execute(
        """
        SELECT value_real FROM pgr_edgar_monthly_raw_values
        WHERE month_end = ? AND field = 'shares_repurchased' AND accession_number = ?
        ORDER BY parse_id LIMIT 1
        """,
        (MONTH_END, may["accession_number"]),
    ).fetchone()
    if printed is None or round(purchases.shares_as_printed_m, 1) != printed[0]:
        problems.append(
            f"8-K printed shares {printed[0] if printed else None} != "
            f"round(pre + post, 1) = {round(purchases.shares_as_printed_m, 1)}"
        )

    # 3. April and June: the 10-Q rows match the 8-K rows in the DB.
    for label, month_end in (("April", "2006-04-30"), ("June", "2006-06-30")):
        found = [r for r in rows if r.label == label]
        db_row = _row(conn, month_end)
        if len(found) != 1:
            problems.append(f"10-Q has no single {label} row")
            continue
        if round(found[0].shares / 1e6, 1) != db_row["shares_repurchased"]:
            problems.append(f"{label} shares: 10-Q {found[0].shares} vs 8-K {db_row['shares_repurchased']}M")
        if found[0].average_price != db_row["avg_cost_per_share"]:
            problems.append(f"{label} price: 10-Q {found[0].average_price} vs 8-K {db_row['avg_cost_per_share']}")

    # 4. The quarter's total cost stated in the 10-Q MD&A.
    match = _Q2_TOTAL_COST.search(ten_q_text)
    if match is None:
        problems.append("10-Q Q2 total repurchase cost not found")
    else:
        stated = float(match.group(1).replace(",", ""))
        computed = sum(r.shares * r.average_price for r in rows) / 1e6
        if abs(stated - computed) > 0.05:
            problems.append(f"Q2 cost: stated ${stated}M vs rows ${computed:.2f}M")
    return problems


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True, help="SQLite DB to repair (use a copy).")
    parser.add_argument(
        "--allow-committed-db",
        action="store_true",
        help="Allow --db to be the committed data/pgr_financials.db.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(config.DATA_RAW_DIR, "edgar_8k_cache"),
        help="EDGAR response cache (default: data/raw/edgar_8k_cache).",
    )
    parser.add_argument(
        "--export-csv", default=None, help="Regenerate pgr_edgar_cache.csv here."
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if not os.getenv("EDGAR_USER_AGENT"):
        log.error("Set EDGAR_USER_AGENT (name and e-mail) before calling EDGAR.")
        return 2
    target = Path(args.db).resolve()
    if target == Path(config.DB_PATH).resolve() and not args.allow_committed_db:
        log.error("Refusing to modify the committed DB; run on a copy.")
        return 2
    if not target.exists():
        log.error("%s does not exist", target)
        return 2

    fetcher.set_http_cache_dir(args.cache_dir)
    conn = db_client.get_connection(str(target))
    try:
        db_client.initialize_schema(conn)
        may = _row(conn, MONTH_END)
        eight_k_url = conn.execute(
            "SELECT source_url FROM pgr_edgar_filing_parses WHERE accession_number = ? "
            "ORDER BY parse_id LIMIT 1",
            (may["accession_number"],),
        ).fetchone()[0]

        ten_q = fetcher._get(TEN_Q["url"])
        eight_k = fetcher._get(eight_k_url)
        rows = q10.parse_issuer_purchases(ten_q.text)
        purchases = q10.split_month_purchases(rows, MONTH_LABEL, _split_ratio(conn))
        for row in rows:
            log.info("10-Q  %-15s %10s shares at $%.2f", row.label, f"{row.shares:,}",
                     row.average_price)

        problems = cross_check(
            conn, rows, purchases, _plain_text(ten_q.text), _plain_text(eight_k.text)
        )
        for problem in problems:
            log.error("cross-check failed: %s", problem)
        if problems:
            return 1

        log.info(
            "%s on the post-split basis: %.6fM shares, $%.4f average, $%.2fM",
            MONTH_END, purchases.shares_post_split_basis_m,
            purchases.avg_cost_post_split_basis, purchases.dollars_m,
        )
        added = db_client.record_pgr_edgar_supplement(
            conn,
            accession_number=TEN_Q["accession_number"],
            parser_version=q10.PARSER_VERSION,
            month_end=MONTH_END,
            filing_date=TEN_Q["filing_date"],
            source_url=TEN_Q["url"],
            fetched_at=getattr(ten_q, "fetched_at", None),
            values=q10.raw_values(purchases),
        )
        log.info("Recorded %d raw values under %s", added, q10.PARSER_VERSION)
        changed = db_client.apply_pgr_edgar_supplements(conn)
        fetcher.recompute_derived_fields(conn)
        after = _row(conn, MONTH_END)
        log.info(
            "%d cell(s) changed; %s: shares_repurchased %s -> %s, avg_cost_per_share %s -> %s",
            changed, MONTH_END, may["shares_repurchased"], after["shares_repurchased"],
            may["avg_cost_per_share"], after["avg_cost_per_share"],
        )

        failures = validate(conn)
        if args.export_csv:
            n = fetcher.export_edgar_cache_csv(conn, args.export_csv)
            log.info("Wrote %d rows to %s", n, args.export_csv)
    finally:
        conn.close()
    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
