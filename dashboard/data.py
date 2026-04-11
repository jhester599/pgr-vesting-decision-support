"""Pure data-loading helpers for the local Streamlit dashboard."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).parent.parent
DECISIONS_DIR = BASE_DIR / "results" / "monthly_decisions"
DB_PATH = BASE_DIR / "data" / "pgr_financials.db"
DECISION_LOG = DECISIONS_DIR / "decision_log.md"


def monthly_dirs(decisions_dir: Path = DECISIONS_DIR) -> list[Path]:
    """Return monthly artifact directories in reverse chronological order."""
    if not decisions_dir.exists():
        return []
    return sorted(
        [path for path in decisions_dir.iterdir() if path.is_dir() and re.match(r"\d{4}-\d{2}$", path.name)],
        reverse=True,
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def load_latest_run_bundle(decisions_dir: Path = DECISIONS_DIR) -> dict[str, Any]:
    """Load the current monthly artifact bundle from disk."""
    dirs = monthly_dirs(decisions_dir)
    if not dirs:
        return {
            "latest_dir": None,
            "manifest": None,
            "signals": pd.DataFrame(),
            "recommendation_text": None,
            "benchmark_quality": pd.DataFrame(),
            "consensus_shadow": pd.DataFrame(),
            "summary": None,
        }

    latest = dirs[0]
    return {
        "latest_dir": latest,
        "manifest": _read_json(latest / "run_manifest.json"),
        "signals": _read_csv(latest / "signals.csv"),
        "recommendation_text": _read_text(latest / "recommendation.md"),
        "benchmark_quality": _read_csv(latest / "benchmark_quality.csv"),
        "consensus_shadow": _read_csv(latest / "consensus_shadow.csv"),
        "summary": _read_json(latest / "monthly_summary.json"),
    }


def load_decision_history(decision_log: Path = DECISION_LOG) -> pd.DataFrame:
    """Parse decision_log.md markdown table into a DataFrame."""
    if not decision_log.exists():
        return pd.DataFrame()

    text = decision_log.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if "|" in line and "---" not in line]

    rows: list[list[str]] = []
    header: list[str] | None = None
    for line in lines:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if header is None:
            header = cells
        elif len(cells) == len(header):
            rows.append(cells)

    if not rows or header is None:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)
    return df.rename(columns={column: column.strip() for column in df.columns})


def load_pgr_prices(db_path: Path = DB_PATH, n_days: int = 365) -> pd.DataFrame:
    """Load recent PGR daily close prices from SQLite."""
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            "SELECT date, close FROM daily_prices WHERE ticker='PGR' ORDER BY date DESC LIMIT ?",
            conn,
            params=(n_days,),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def parse_aggregate_health(rec_text: str | None) -> dict[str, float | None]:
    """Extract aggregate OOS R^2, IC, and hit rate from recommendation text."""
    match = re.search(
        r"OOS R\^2\s+([+-]?\d+\.?\d*)%.*?IC\s+([-\d.]+).*?hit rate\s+([\d.]+)%",
        rec_text or "",
    )
    if not match:
        return {"oos_r2": None, "ic": None, "hit_rate": None}
    return {
        "oos_r2": float(match.group(1)),
        "ic": float(match.group(2)),
        "hit_rate": float(match.group(3)),
    }


def parse_recommendation_summary(rec_text: str | None) -> dict[str, str | None]:
    """Extract top-level recommendation fields from recommendation markdown."""
    if not rec_text:
        return {
            "signal": None,
            "sell_pct": None,
            "predicted_return": None,
            "calibration_ece": None,
        }

    signal_match = re.search(r"\| Signal \| \*\*(.+?)\*\* \|", rec_text)
    sell_match = re.search(r"Recommended Sell %.*?\*\*(\d+)%\*\*", rec_text)
    predicted_match = re.search(r"\| Predicted 6M Relative Return \| ([^|]+)\|", rec_text)
    ece_match = re.search(r"ECE\s*=\s*([\d.]+)%", rec_text)
    return {
        "signal": signal_match.group(1).strip() if signal_match else None,
        "sell_pct": f"{sell_match.group(1)}%" if sell_match else None,
        "predicted_return": predicted_match.group(1).strip() if predicted_match else None,
        "calibration_ece": f"{ece_match.group(1)}%" if ece_match else None,
    }
