"""Helpers for the v19 remaining-feature completion cycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
import re
from typing import Any

import pandas as pd
import requests

from src.ingestion.fred_loader import upsert_fred_to_db


FREDGRAPH_SERIES: tuple[str, ...] = (
    "DTWEXBGS",
    "DCOILWTICO",
    "MORTGAGE30US",
    "WPU45110101",
    "PPIACO",
    "MRTSSM447USN",
    "THREEFYTP10",
)

BLS_SERIES: tuple[str, ...] = (
    "CUSR0000SETE",
)

MULTPL_SERIES: dict[str, tuple[str, bool]] = {
    "SP500_PE_RATIO_MULTPL": ("s-p-500-pe-ratio", False),
    "SP500_EARNINGS_YIELD_MULTPL": ("s-p-500-earnings-yield", True),
    "SP500_PRICE_TO_BOOK_MULTPL": ("s-p-500-price-to-book", False),
}

BLOCKED_FEATURE_REASONS: dict[str, str] = {
    "pgr_cr_vs_peer_cr": (
        "Requires point-in-time peer combined-ratio history for ALL/TRV/CB/HIG, "
        "which the repo does not currently ingest."
    ),
    "pgr_fcf_yield": (
        "Requires quarterly operating-cash-flow and capex ingestion from EDGAR, "
        "which is not present in the current fundamentals schema."
    ),
}


@dataclass(frozen=True)
class PublicMacroFetchSummary:
    """Simple status row for one v19 public macro series fetch."""

    series_id: str
    source: str
    rows_loaded: int


def _clean_numeric_string(value: Any) -> float | None:
    """Parse a numeric value from a scraped string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned in {"", "-", ".", "-."}:
        return None
    return float(cleaned)


def fetch_fredgraph_series(
    series_id: str,
    observation_start: str = "2008-01-01",
) -> pd.DataFrame:
    """Fetch a monthly-compatible series from FRED's public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    value_col = df.columns[-1]
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["observation_date"]).set_index("observation_date").sort_index()
    df = df.loc[df.index >= pd.Timestamp(observation_start)]
    monthly = df[[value_col]].resample("ME").last().ffill(limit=5)
    monthly.columns = [series_id]
    return monthly


def fetch_bls_series(
    series_id: str,
    observation_start: str = "2008-01-01",
) -> pd.DataFrame:
    """Fetch a monthly BLS series via the public BLS API."""
    start_year = str(pd.Timestamp(observation_start).year)
    end_year = str(date.today().year)
    url = f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}"
    response = requests.get(
        url,
        params={"startyear": start_year, "endyear": end_year},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    series_list = payload.get("Results", {}).get("series", [])
    if not series_list:
        return pd.DataFrame(columns=[series_id])

    rows: list[dict[str, Any]] = []
    for row in series_list[0].get("data", []):
        period = str(row.get("period", ""))
        if not period.startswith("M") or period == "M13":
            continue
        year = int(row["year"])
        month = int(period[1:])
        month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        rows.append({"month_end": month_end, series_id: _clean_numeric_string(row.get("value"))})

    if not rows:
        return pd.DataFrame(columns=[series_id])

    df = pd.DataFrame(rows).dropna(subset=["month_end"]).sort_values("month_end").set_index("month_end")
    df = df.loc[df.index >= pd.Timestamp(observation_start)]
    return df[[series_id]]


def fetch_multpl_series(
    *,
    slug: str,
    series_id: str,
    observation_start: str = "2008-01-01",
) -> pd.DataFrame:
    """Fetch a monthly valuation series from Multpl."""
    url = f"https://www.multpl.com/{slug}/table/by-month"
    table = pd.read_html(url)[0]
    table["Date"] = pd.to_datetime(table["Date"], errors="coerce")
    table[series_id] = table["Value"].map(_clean_numeric_string)
    table = table.dropna(subset=["Date"]).set_index("Date").sort_index()
    table.index = table.index + pd.offsets.MonthEnd(0)
    table = table.loc[table.index >= pd.Timestamp(observation_start)]
    monthly = table[[series_id]].resample("ME").last().ffill(limit=3)
    return monthly


def fetch_v19_public_macro(
    observation_start: str = "2008-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all public-macro series needed for the remaining v19 queue."""
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for series_id in FREDGRAPH_SERIES:
        df = fetch_fredgraph_series(series_id, observation_start=observation_start)
        frames.append(df)
        summary_rows.append(
            PublicMacroFetchSummary(series_id=series_id, source="fredgraph", rows_loaded=len(df)).__dict__
        )

    for series_id in BLS_SERIES:
        df = fetch_bls_series(series_id, observation_start=observation_start)
        frames.append(df)
        summary_rows.append(
            PublicMacroFetchSummary(series_id=series_id, source="bls", rows_loaded=len(df)).__dict__
        )

    for series_id, (slug, _percent) in MULTPL_SERIES.items():
        df = fetch_multpl_series(slug=slug, series_id=series_id, observation_start=observation_start)
        frames.append(df)
        summary_rows.append(
            PublicMacroFetchSummary(series_id=series_id, source="multpl", rows_loaded=len(df)).__dict__
        )

    wide = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return wide, summary


def upsert_v19_public_macro(conn: Any, df: pd.DataFrame) -> int:
    """Store public-macro series in fred_macro_monthly for feature reuse."""
    return upsert_fred_to_db(conn, df)


def build_v19_traceability_matrix(
    inventory_df: pd.DataFrame,
    *,
    feature_columns: set[str],
    phase0_summary: pd.DataFrame,
    blocked_reasons: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize tested/blocked status for every original v15 feature idea."""
    blocked_reasons = blocked_reasons or BLOCKED_FEATURE_REASONS
    rows: list[dict[str, Any]] = []

    for row in inventory_df.itertuples(index=False):
        feature_name = str(row.feature_name)
        tested_rows = phase0_summary[phase0_summary["candidate_feature"] == feature_name].copy()
        blocked_reason = blocked_reasons.get(feature_name, "")

        if not tested_rows.empty:
            best_row = tested_rows.sort_values(
                by=[
                    "mean_policy_return_sign_delta",
                    "mean_oos_r2_delta",
                    "mean_ic_delta",
                ],
                ascending=[False, False, False],
            ).iloc[0]
            evaluation_status = "tested"
            best_model = str(best_row["candidate_name"])
            best_replace = str(best_row["replace_feature"])
            best_policy_delta = float(best_row["mean_policy_return_sign_delta"])
            best_oos_r2_delta = float(best_row["mean_oos_r2_delta"])
            best_ic_delta = float(best_row["mean_ic_delta"])
        elif blocked_reason:
            evaluation_status = "blocked"
            best_model = ""
            best_replace = ""
            best_policy_delta = float("nan")
            best_oos_r2_delta = float("nan")
            best_ic_delta = float("nan")
        elif feature_name in feature_columns:
            evaluation_status = "available_not_queued"
            best_model = ""
            best_replace = ""
            best_policy_delta = float("nan")
            best_oos_r2_delta = float("nan")
            best_ic_delta = float("nan")
        else:
            evaluation_status = "not_available"
            best_model = ""
            best_replace = ""
            best_policy_delta = float("nan")
            best_oos_r2_delta = float("nan")
            best_ic_delta = float("nan")

        rows.append(
            {
                "feature_name": feature_name,
                "category": str(row.category),
                "target_model": str(row.target_model),
                "priority_rank": row.priority_rank,
                "research_source": getattr(row, "research_source", ""),
                "available_in_matrix": feature_name in feature_columns,
                "evaluation_status": evaluation_status,
                "blocked_reason": blocked_reason,
                "best_model": best_model,
                "best_replace_feature": best_replace,
                "best_policy_return_sign_delta": best_policy_delta,
                "best_oos_r2_delta": best_oos_r2_delta,
                "best_ic_delta": best_ic_delta,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["priority_rank", "feature_name"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
