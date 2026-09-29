"""Reproduce the v201 preflight blocker; this closed session cannot fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from pgr_vds.research_lib.provenance import (
    export_git_blob,
    read_immutable,
    sha256_file,
    verify_baseline_lock,
    verify_runtime,
)


ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
BASELINE = ROOT / "research/studies/v200_clean_baseline/outputs"
LOCK_SHA = "c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0"
SEED = 20260926
LAGS = {
    "T10Y2Y": 1,
    "GS10": 1,
    "T10YIE": 1,
    "VIXCLS": 1,
    "NFCI": 2,
    "BAMLH0A0HYM2": 1,
    "PCU5241265241261": 1,
    "CUSR0000SETA02": 1,
    "CUSR0000SAM2": 1,
}
BLUEPRINTS = {
    "P1": {
        "remove": ["mom_3m", "mom_6m", "mom_12m"],
        "add": ["pm_mom3", "pm_mom6", "pm_mom12"],
    },
    "P2": {"remove": ["vol_63d"], "add": ["pm_vol13w", "pm_high52w"]},
    "P3": {
        "remove": ["mom_6m", "mom_12m"],
        "add": ["pm_ema12_voo", "pm_rsi6_voo"],
        "inventory": ["ta_ratio_ema_gap_12m_voo", "ta_ratio_rsi_6m_voo"],
    },
    "M1": {
        "remove": ["yield_slope", "real_rate_10y", "real_yield_change_6m"],
        "add": ["pm_slope", "pm_real_change6"],
    },
    "M2": {
        "remove": ["vix", "nfci", "credit_spread_hy"],
        "add": ["pm_vix", "pm_nfci", "pm_credit_hy"],
    },
    "M3": {"remove": ["rate_adequacy_gap_yoy"], "add": ["pm_rate_gap"]},
}
SQL = {
    "prices": "SELECT ticker,date,close FROM daily_prices WHERE ticker IN "
    "('PGR','VOO') AND date<=? AND proxy_fill=0 ORDER BY ticker,date",
    "splits": "SELECT ticker,split_date,split_ratio FROM split_history "
    "WHERE ticker IN ('PGR','VOO') AND split_date<=?",
    "macro": "SELECT series_id,month_end,value FROM fred_macro_monthly "
    "WHERE month_end<=? ORDER BY series_id,month_end",
}


def save_json(path: Path, value: object) -> None:
    """Emit strict JSON; numpy/pandas scalars have explicit representations."""

    def normalize(item: object) -> object:
        if isinstance(item, dict):
            return {str(k): normalize(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        if isinstance(item, (pd.Timestamp, Path)):
            return str(item)
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    """Preserve exact portable artifact bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def git(*args: str) -> str:
    """Inspect local Git state; never fetch providers or send messages."""
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def preflight(output: Path, scratch: Path) -> dict:
    """Verify the accepted lock, complete manifest and exact dependencies.

    Its original current-checkout verifier is preserved. It intentionally
    exposes the registry mismatch rather than weakening an accepted pin.
    """
    output.mkdir(parents=True, exist_ok=True)
    try:
        lock_path = BASELINE / "baseline_lock.json"
        lock = verify_baseline_lock(lock_path)
        if sha256_file(lock_path) != LOCK_SHA:
            raise ValueError("Accepted v200 baseline lock bytes changed")
        if lock.get("comparison_status") != "accepted_research_comparator":
            raise ValueError("No provisional v200 comparison is permitted")
        manifest = json.loads((BASELINE / "output_manifest.json").read_text())
        for relative, digest in manifest["sha256"].items():
            if sha256_file(BASELINE.parent / relative) != digest:
                raise ValueError(f"v200 output hash mismatch: {relative}")
        verify_runtime(
            json.loads((BASELINE / "runtime_lock.json").read_text())
        )
        scratch.mkdir(parents=True, exist_ok=True)
        db = lock["input_db"]
        export_git_blob(
            db["git_commit"],
            db["relative_path"],
            scratch / "pinned.db",
            db["sha256"],
        )
        with read_immutable(scratch / "pinned.db", db["sha256"]) as conn:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("DB integrity preflight failed")
        return lock
    except Exception as error:
        (output / "README.md").write_text(
            "v201 is blocked by its exact baseline/input preflight. "
            "No fitting or holdout scoring started.\n\n"
            f"Failure: {error!r}.\n\nWhat changed: the blocker was recorded. "
            "What is left: resolve it before running the six candidates.\n",
            encoding="utf-8",
        )
        save_json(
            output / "preflight.json",
            {"status": "blocked", "error": repr(error), "fits": 0},
        )
        raise


def load_csv(path: Path, dates: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=dates, float_precision="round_trip")


def require_register(output: Path) -> dict:
    """Require committed, byte-exact preregistration and consumed artifacts."""
    path = output / "registered.json"
    if not path.is_file():
        raise ValueError("A committed preregistration is required")
    relative = str(path.relative_to(ROOT)).replace("\\", "/")
    saved = subprocess.check_output(
        ["git", "show", "HEAD:" + relative], cwd=ROOT
    )
    if saved != path.read_bytes():
        raise ValueError("Commit the exact preregistration before execution")
    register = json.loads(saved)
    for relative, digest in register["input_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"Preregistered input changed: {relative}")
    return register


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    output = STUDY / "outputs"
    preflight(output, args.scratch)
    raise RuntimeError("Blocked session closeout: fitting is disabled")


if __name__ == "__main__":
    main()
