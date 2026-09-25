"""Split-month buybacks from a 10-Q "Issuer Purchases of Equity Securities" table.

PGR's monthly 8-K for May 2006 (0000950152-06-005098), the month of the
2006-05-18 4-for-1 split, prints "Shares repurchased — May 2.3" and "Average
cost per share NM".  Its footnote splits the 2.3M into 0.3M shares bought
before the split at $107.94 and 2.0M after it at $27.16 ("we did not split
treasury shares").  The 2.3M therefore adds pre-split and post-split shares
and is on neither basis.

The Q2 2006 10-Q (0000950152-06-006431), Part II Item 2, gives the exact
counts on separate rows::

    May pre-split     331,496   107.94
    May post-split  1,932,200    27.16

This module reads those rows and combines them onto the share basis of the
report month (post-split for 2006-05-31, the basis every other per-share
value of that row is on):

    shares   = pre_shares × split_ratio + post_shares
    dollars  = pre_shares × pre_price + post_shares × post_price
    avg cost = dollars / shares

The per-leg values are recorded as parsed and the combined month as derived
(``pgr_edgar_monthly_raw``, parser ``PARSER_VERSION``); see
``scripts/repair_split_month_buybacks.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

# Version tag of these parses in pgr_edgar_filing_parses.  It is listed in
# db_client.PGR_EDGAR_SUPPLEMENT_PARSER_VERSIONS.
PARSER_VERSION: str = "10q-issuer-purchases/2026-09-25"

_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)
_NUMBER = re.compile(r"^\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)$")


@dataclass(frozen=True)
class PurchaseRow:
    """One row of the issuer-purchases table (shares as whole shares)."""

    label: str
    shares: int
    average_price: float


@dataclass(frozen=True)
class SplitMonthPurchases:
    """A split month's pre- and post-split purchases, combined onto one basis."""

    pre_split_shares_m: float
    pre_split_average_price: float
    post_split_shares_m: float
    post_split_average_price: float
    split_ratio: float

    @property
    def shares_post_split_basis_m(self) -> float:
        """Shares repurchased (millions) restated onto the post-split basis."""
        return self.pre_split_shares_m * self.split_ratio + self.post_split_shares_m

    @property
    def dollars_m(self) -> float:
        """Total cost ($M): each leg at its own price and share count."""
        return (
            self.pre_split_shares_m * self.pre_split_average_price
            + self.post_split_shares_m * self.post_split_average_price
        )

    @property
    def avg_cost_post_split_basis(self) -> float:
        """Average cost per post-split share ($)."""
        return self.dollars_m / self.shares_post_split_basis_m

    @property
    def shares_as_printed_m(self) -> float:
        """Pre- plus post-split share count, as the 8-K and 10-Q add them."""
        return self.pre_split_shares_m + self.post_split_shares_m


def _number(text: str) -> float | None:
    match = _NUMBER.match(text.strip())
    return float(match.group(1).replace(",", "")) if match else None


def parse_issuer_purchases(html: str) -> list[PurchaseRow]:
    """Return the month rows of the "Issuer Purchases of Equity Securities" table.

    Each row starts with a month name (optionally "pre-split"/"post-split");
    the first two numbers are the total shares purchased and the average price
    paid.  "$" cells are ignored.  The "Total" row is not returned.

    Raises:
        ValueError: if no such table is found.
    """
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        if "issuer purchases of equity securities" not in " ".join(
            table.get_text(" ").lower().split()
        ):
            continue
        rows: list[PurchaseRow] = []
        for tr in table.find_all("tr"):
            cells = [
                " ".join(td.get_text(" ").split())
                for td in tr.find_all(["td", "th"])
            ]
            cells = [c for c in cells if c and c != "$"]
            if not cells or cells[0].split(" ")[0].lower() not in _MONTHS:
                continue
            numbers = [n for n in (_number(c) for c in cells[1:]) if n is not None]
            if len(numbers) < 2:
                continue
            rows.append(PurchaseRow(cells[0], int(numbers[0]), numbers[1]))
        if rows:
            return rows
    raise ValueError("no 'Issuer Purchases of Equity Securities' table with month rows")


def split_month_purchases(
    rows: list[PurchaseRow],
    month: str,
    split_ratio: float,
) -> SplitMonthPurchases:
    """Pick the "<month> pre-split" and "<month> post-split" rows and combine them.

    Raises:
        ValueError: if either row is missing or appears twice.
    """
    def _one(suffix: str) -> PurchaseRow:
        label = f"{month} {suffix}".lower()
        found = [r for r in rows if r.label.lower() == label]
        if len(found) != 1:
            raise ValueError(f"expected one '{month} {suffix}' row, found {len(found)}")
        return found[0]

    pre = _one("pre-split")
    post = _one("post-split")
    return SplitMonthPurchases(
        pre_split_shares_m=pre.shares / 1e6,
        pre_split_average_price=pre.average_price,
        post_split_shares_m=post.shares / 1e6,
        post_split_average_price=post.average_price,
        split_ratio=split_ratio,
    )


def raw_values(purchases: SplitMonthPurchases) -> list[tuple[str, float, str]]:
    """``(field, value, method)`` rows for ``pgr_edgar_monthly_raw``.

    The four per-leg values are ``parsed``; ``shares_repurchased`` and
    ``avg_cost_per_share`` on the report month's (post-split) basis are
    ``derived``.
    """
    return [
        ("shares_repurchased_pre_split", purchases.pre_split_shares_m, "parsed"),
        ("avg_cost_per_share_pre_split", purchases.pre_split_average_price, "parsed"),
        ("shares_repurchased_post_split", purchases.post_split_shares_m, "parsed"),
        ("avg_cost_per_share_post_split", purchases.post_split_average_price, "parsed"),
        ("shares_repurchased", purchases.shares_post_split_basis_m, "derived"),
        ("avg_cost_per_share", purchases.avg_cost_post_split_basis, "derived"),
    ]
