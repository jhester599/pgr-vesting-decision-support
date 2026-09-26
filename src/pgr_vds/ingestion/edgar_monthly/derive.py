"""Derived ``pgr_edgar_monthly`` fields (YoY growth, Gainshare, channel mix, ...).

Derived fields are recomputed over the whole table, by calendar month, after
every write (``recompute_derived_fields``): YoY windows need the full
history, so they are never computed on a partial fetch window (F16/F33).
The formulas are in ``src.processing.pgr_edgar_derived``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from src.processing import pgr_edgar_derived


def _prior_year_key(month_end: str) -> str:
    """Return the month_end string for the same month one year prior.

    Period arithmetic on the calendar month (F11): 2025-02-28 maps to
    2024-02-29, and 2024-02-29 maps to 2023-02-28.
    """
    return pgr_edgar_derived.prior_year_month_end(month_end)


# Fields that ``compute_derived_fields`` owns.  They are always recomputed
# from the parsed fields, never read from a filing or the CSV.
DERIVED_FIELDS: tuple[str, ...] = (
    "pif_total",
    "pif_total_personal_lines",
    "pif_growth_yoy",
    "gainshare_estimate",
    "channel_mix_agency_pct",
    "underwriting_income",
    "npw_growth_yoy",
    "unearned_premium_growth_yoy",
)


def compute_derived_fields(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute all derived fields for the full 8-K time series, in place.

    Uses the single definitions in ``src.processing.pgr_edgar_derived``:

      - ``pif_total``                — agency auto + direct auto + special lines
        + commercial lines (property excluded); None unless all are present
      - ``pif_total_personal_lines`` — agency auto + direct auto + special lines
      - ``pif_growth_yoy``           — calendar-month YoY of ``pif_total``
      - ``gainshare_estimate``       — 0.5 × CR score + 0.5 × PIF score (0–2);
        None unless both CR and PIF growth exist
      - ``channel_mix_agency_pct``   — Agency NPW / (Agency + Direct NPW)
      - ``underwriting_income``      — NPE × (1 − CR/100)
      - ``npw_growth_yoy``           — calendar-month YoY of total NPW
      - ``unearned_premium_growth_yoy`` — calendar-month YoY of unearned premiums

    YoY values are None when the same month one year earlier is missing, so
    a gap never produces a 13-month change (F16).  Every derived field is
    assigned (possibly None), so recomputing over the full table clears
    stale values.

    Args:
        records: Record dicts with a ``month_end`` key (any order).

    Returns:
        The same list with all derived fields assigned.
    """
    for rec in records:
        rec["pif_total"] = pgr_edgar_derived.pif_sum(
            rec, pgr_edgar_derived.PIF_TOTAL_COMPONENTS
        )
        rec["pif_total_personal_lines"] = pgr_edgar_derived.pif_sum(
            rec, pgr_edgar_derived.PIF_PERSONAL_LINES_COMPONENTS
        )

    def _yoy(field: str) -> dict[pd.Period, float | None]:
        return pgr_edgar_derived.yoy_growth_by_period(
            {r["month_end"]: r.get(field) for r in records}
        )

    pif_yoy = _yoy("pif_total")
    npw_yoy = _yoy("net_premiums_written")
    unprem_yoy = _yoy("unearned_premiums")

    for rec in records:
        period = pgr_edgar_derived.month_key(rec["month_end"])
        rec["pif_growth_yoy"] = pif_yoy.get(period)
        rec["npw_growth_yoy"] = npw_yoy.get(period)
        rec["unearned_premium_growth_yoy"] = unprem_yoy.get(period)

        cr = rec.get("combined_ratio")
        rec["gainshare_estimate"] = pgr_edgar_derived.gainshare_estimate(
            cr, rec["pif_growth_yoy"]
        )

        # channel_mix_agency_pct = npw_agency / (npw_agency + npw_direct)
        rec["channel_mix_agency_pct"] = None
        npw_ag = rec.get("npw_agency")
        npw_di = rec.get("npw_direct")
        if npw_ag is not None and npw_di is not None and npw_ag + npw_di > 0.0:
            rec["channel_mix_agency_pct"] = npw_ag / (npw_ag + npw_di)

        # underwriting_income = net_premiums_earned × (1 − CR / 100)
        rec["underwriting_income"] = None
        npe = rec.get("net_premiums_earned")
        if npe is not None and cr is not None:
            rec["underwriting_income"] = npe * (1.0 - cr / 100.0)

    return records


def recompute_derived_fields(conn: sqlite3.Connection) -> int:
    """Recompute every derived column over the whole ``pgr_edgar_monthly`` table.

    YoY windows need the full history, so derived fields are never computed
    on a partial fetch window (F16/F33).  Returns the number of rows updated.
    """
    cur = conn.execute("SELECT * FROM pgr_edgar_monthly ORDER BY month_end")
    names = [d[0] for d in cur.description]
    records = [dict(zip(names, row)) for row in cur.fetchall()]
    if not records:
        return 0
    compute_derived_fields(records)
    sql = (
        "UPDATE pgr_edgar_monthly SET "
        + ", ".join(f"{f} = ?" for f in DERIVED_FIELDS)
        + " WHERE month_end = ?"
    )
    conn.executemany(
        sql,
        [tuple(rec[f] for f in DERIVED_FIELDS) + (rec["month_end"],) for rec in records],
    )
    conn.commit()
    return len(records)
