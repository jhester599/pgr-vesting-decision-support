"""Pinned, development-only v205 dividend/BVPS x-series lane runner.

Two phases, run in order and committed separately:

    python research/studies/v205_dividend_bvps/run.py --preregister \
        --scratch <external-dir>
    python research/studies/v205_dividend_bvps/run.py --execute \
        --scratch <external-dir> [--output-dir <dir>]

``--preregister`` verifies the accepted v200 lock, the exact runtime and the
pinned DB copy, recomputes every target by hand and checks it against v200,
builds rolling monthly features, seals the partition lock and writes the
frozen candidate register. It fits no model and scores no forecast.
``--execute`` refuses to run unless the committed preregistration is
byte-identical to the one the code rebuilds, then fits the registered
candidates on development data only. No step fetches data, sends email,
writes a database or reads the v207 quarantine.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

from pgr_vds.research_lib import xseries
from pgr_vds.research_lib.adapters import drip_return
from pgr_vds.research_lib.metrics import honest_r2
from pgr_vds.research_lib.provenance import (
    BASELINE_CODE_COMMIT,
    INPUT_DB_COMMIT,
    INPUT_DB_SHA256,
    PARENT_SEED_COMMIT,
    PARENT_SEED_SHA256,
    export_git_blob,
    read_immutable,
    runtime_lock,
    sha256_file,
    source_sha256,
    verify_baseline_lock,
    verify_runtime,
)
from pgr_vds.research_lib.temporal import chronological_splits, label_end


ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
OUTPUTS = STUDY / "outputs"
V200 = "research/studies/v200_clean_baseline"
REGISTRY = "research/registry.yaml"
AS_OF = pd.Timestamp("2026-09-26")
BOUNDARY = pd.Timestamp("2023-09-29")
SEED = 20260926
REPLICATES = 2000
CALIBRATION_MIN_SUPPORT = 12
# Exact bytes of every earlier output this study consumes. They match the
# v200 output manifest and cannot be substituted by a mutable latest file.
V200_PINS = {
    f"{V200}/outputs/baseline_lock.json": (
        "c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0"
    ),
    f"{V200}/outputs/control_predictions.csv": (
        "4198eef62332cfa02ab59b8cf1b52328b66dc8cd547181751dc0949c4943830b"
    ),
    f"{V200}/outputs/runtime_lock.json": (
        "ed8a5eea976ce21c816beb3b305b3a1d6fde6c59ea87c261aedab7ee0304d1e9"
    ),
    f"{V200}/outputs/dependency_environment.json": (
        "0cd4bf33c6dc2052cf31ed7dc300ed2fd7a5462c2db1660a429dfaa6ba30f298"
    ),
    f"{V200}/outputs/partitions.json": None,
    f"{V200}/outputs/control_support.json": None,
}
SOURCE_DOCS = [
    "AGENTS.md",
    "docs/research/RERUN_PLAN_v200_codex.md",
    "docs/reviews/REPO_REVIEW_2026-09-25.md",
    "docs/reviews/VERIFICATION_2026-09-26.md",
    "docs/reviews/VERIFICATION_2026-09-26_claude.md",
    "docs/research/x_series_resume_2026-04-24.md",
    "docs/model-governance.md",
    "src/research/x1_targets.py",
    "src/research/x9_bvps_bridge.py",
    "src/research/x12_bvps_target_audit.py",
    "src/research/x16_indicator_package.py",
    "src/research/x17_persistent_bvps.py",
    "src/research/x18_dividend_policy_regime.py",
    "src/research/x21_dividend_target_scales.py",
    "src/research/x22_dividend_size_baselines.py",
    "src/research/x23_dividend_lane_package.py",
    f"{V200}/README.md",
]
SQL = {
    "prices": (
        "SELECT date,close FROM daily_prices WHERE ticker='PGR' "
        "AND date < ? ORDER BY date"
    ),
    "dividends": (
        "SELECT ex_date,amount FROM daily_dividends WHERE ticker='PGR' "
        "AND ex_date < ? ORDER BY ex_date"
    ),
    "splits": (
        "SELECT split_date,split_ratio FROM split_history WHERE "
        "ticker='PGR' AND split_date < ? ORDER BY split_date"
    ),
    "reports": (
        "SELECT month_end,filing_date,book_value_per_share,combined_ratio,"
        "pif_total,net_premiums_written,roe_net_income_ttm,"
        "gainshare_estimate,investment_book_yield FROM pgr_edgar_monthly "
        "WHERE month_end < ? AND filing_date < ? ORDER BY month_end"
    ),
    "first_reported": (
        "SELECT month_end,field,value_real,filing_date FROM "
        "pgr_edgar_monthly_first_reported WHERE month_end < ? AND "
        "filing_date < ? AND field IN ('book_value_per_share',"
        "'combined_ratio','pif_total','net_premiums_written',"
        "'roe_net_income_ttm','investment_book_yield') "
        "ORDER BY month_end,field"
    ),
}
VINTAGE_FIELDS = [
    "book_value_per_share",
    "combined_ratio",
    "pif_total",
    "net_premiums_written",
    "roe_net_income_ttm",
    "investment_book_yield",
]
DOLLAR_FEATURES = {
    "past12_cash",
    "prior_past12_cash",
    "current_bvps",
    "bvps_yoy_dollar_change",
}
ARCHIVED_BVPS_LAGS = [
    "current_bvps",
    "bvps_growth_1m",
    "bvps_growth_3m",
    "bvps_growth_6m",
    "bvps_growth_ytd",
    "bvps_yoy_dollar_change",
    "month_of_year",
    "q4_flag",
    "dividend_season_flag",
]
ENDPOINTS = {
    "cash_dividend_12m": {
        "horizon": 12,
        "unit": "USD cash per one origin-date PGR share",
        "event": "next-12M cash strictly above the fixed past-12M control",
        "threshold": "control",
    },
    "bvps_growth_12m": {
        "horizon": 12,
        "unit": "fractional BVPS growth on the origin share basis",
        "event": "BVPS growth strictly above zero",
        "threshold": "zero",
    },
    "pgr_drip_return_6m": {
        "horizon": 6,
        "unit": "signed fractional PGR total return, splits and DRIP",
        "event": "total return strictly above zero",
        "threshold": "zero",
    },
    "annual_excess_to_bvps": {
        "horizon": 12,
        "unit": "December-February excess cash / November current BVPS",
        "event": "not scored: conditional positive-event endpoint",
        "threshold": "none",
    },
}


def candidate_register() -> list[dict[str, Any]]:
    """Frozen ordered v205 register: six blueprints, campaign slots 29-34."""
    d1 = ["past12_cash", "prior_past12_cash", "cr_ttm"]
    b1 = ["bvps_growth_yoy", "bvps_growth_3m", "bvps_growth_6m", "roe_ttm"]
    common = {
        "model": "median imputer, standard scaler, Ridge; all fitted inside "
        "each training history",
        "alpha_grid": list(xseries.ALPHA_GRID),
        "inner_selection": "summed absolute error over three inner folds; "
        "exact ties choose the larger alpha",
        "primary_test": "one-sided paired moving-date-block bootstrap of the "
        "per-date absolute-error improvement over the matched v200 control",
    }
    return [
        {
            **common,
            "id": "D1",
            "slot": 29,
            "lane": "dividend_12m",
            "endpoint": "cash_dividend_12m",
            "horizon": 12,
            "features": d1,
            "training_target": "next-12M cash minus the fixed past-12M "
            "control (offset added back after prediction)",
            "control": "v200 fixed past-12M cash per origin share",
            "status": "registered",
        },
        {
            **common,
            "id": "D2",
            "slot": 30,
            "lane": "dividend_12m",
            "endpoint": "cash_dividend_12m",
            "horizon": 12,
            "features": d1 + ["pif_growth_yoy_cal", "npw_growth_ttm"],
            "training_target": "next-12M cash minus the fixed past-12M "
            "control (offset added back after prediction)",
            "control": "v200 fixed past-12M cash per origin share",
            "status": "registered",
        },
        {
            **common,
            "id": "D3",
            "slot": 31,
            "lane": "dividend_12m",
            "endpoint": "annual_excess_to_bvps",
            "horizon": 12,
            "features": ["gainshare_estimate", "investment_book_yield"],
            "training_target": "archived x23 annual excess / current BVPS",
            "control": "v200 past-only mean of matured positive annual "
            "ratios (same label)",
            "status": "registered",
            "closure_rule": "Fit only if at least 5 positive annual events "
            "matured before a scored November origin and a three-fold inner "
            "history with 60 usable monthly labels exists; otherwise close "
            "as insufficient annual support with primary p=1. Exclusions "
            "and gaps are never reduced.",
        },
        {
            **common,
            "id": "B1",
            "slot": 32,
            "lane": "bvps_12m",
            "endpoint": "bvps_growth_12m",
            "horizon": 12,
            "features": b1,
            "training_target": "next-12M BVPS growth (the endpoint label)",
            "control": "v200 past-only prevailing mean BVPS growth",
            "status": "registered",
        },
        {
            **common,
            "id": "B2",
            "slot": 33,
            "lane": "bvps_12m",
            "endpoint": "bvps_growth_12m",
            "horizon": 12,
            "features": b1 + [
                "cr_ttm", "pif_growth_yoy_cal", "npw_growth_ttm"
            ],
            "training_target": "next-12M BVPS growth (the endpoint label)",
            "control": "v200 past-only prevailing mean BVPS growth",
            "status": "registered",
        },
        {
            **common,
            "id": "B3",
            "slot": 34,
            "lane": "bvps_12m",
            "endpoint": "pgr_drip_return_6m",
            "horizon": 6,
            "features": list(ARCHIVED_BVPS_LAGS),
            "training_target": "archived x12/x16 dividend-adjusted 6M BVPS "
            "growth, split-consistent, available on the future report's "
            "filing",
            "mapping": "x16 adjusted_structural_bvps_pb_6m, "
            "ridge_bridge__no_change_pb on the x9 bvps_lags block: implied "
            "price = forecast adjusted BVPS x current P/B, so the forecast "
            "6M PGR return equals the forecast adjusted BVPS growth",
            "control": "v200 past-only mean absolute 6M PGR split/DRIP return",
            "status": "registered",
        },
    ]


def preregistered_rules() -> dict[str, Any]:
    """Every threshold, test and support rule, frozen before fitting."""
    return {
        "thresholds": dict(xseries.THRESHOLDS),
        "disposition": (
            "Pass only if relative MAE reduction >= .10 vs the matched v200 "
            "control, Holm-adjusted primary p < .05 (38-test campaign "
            "family, pending slots p=1), delta honest R2 >= .01, candidate "
            "hit rate >= control hit rate, candidate Brier <= control Brier, "
            "|coverage80-.80| <= control's + .05, scored dates / horizon >= 5 "
            "independent blocks and supported block inference. NaN fails."
        ),
        "nomination": (
            "At most one winner across both lanes: lowest adjusted p, then "
            "largest relative MAE reduction; none if nothing passes"
        ),
        "matched_rows": (
            "Outer-test rows with a finite candidate forecast, a finite "
            "non-warmup v200 control forecast and naive mean, and a finite "
            "label"
        ),
        "honest_r2_naive": (
            "v200 prevailing mean of same-endpoint labels matured by origin"
        ),
        "primary_test": {
            "statistic": "mean over dates of |y-control| - |y-candidate|",
            "sidedness": "one-sided, centered bootstrap",
            "block_length": "horizon months (12 or 6)",
            "replicates": REPLICATES,
            "seed": SEED,
            "minimum_dates": "two full blocks",
        },
        "ic": "Spearman, two-sided centered moving-date-block p (reported)",
        "direction": {
            "cash_dividend_12m": ENDPOINTS["cash_dividend_12m"]["event"],
            "bvps_growth_12m": ENDPOINTS["bvps_growth_12m"]["event"],
            "pgr_drip_return_6m": ENDPOINTS["pgr_drip_return_6m"]["event"],
            "base": "constant majority event of labels matured by origin",
            "skill_test": "one-sided block test of per-date hit minus base "
            "hit, reported raw; a safeguard, not a success route",
        },
        "calibration": {
            "method": "prequential empirical residual quantiles/frequency "
            "from the same stream's matured residuals only",
            "interval": "nominal 80%: forecast + 10th/90th residual quantile",
            "probability": "share of matured residuals with forecast + "
            "residual above the event threshold",
            "minimum_matured_residuals": CALIBRATION_MIN_SUPPORT,
            "scores": "Brier, log loss clipped [1e-6,1-1e-6], 10-bin ECE",
            "matched": "rows where candidate and control are both "
            "past warmup",
        },
        "folds": {
            "h12": "outer TimeSeriesSplit 120 train / 6 test / gap 24; "
            "inner 3 folds test 6 gap 24, minimum 60 usable months",
            "h6": "outer TimeSeriesSplit 60 train / 6 test / gap 12; inner "
            "3 folds test 6 gap 12, minimum 24 usable months",
            "dates": "contiguous business month-ends spanning the v200 "
            "control origins of each endpoint; unique monthly dates split "
            "before any row expansion",
            "availability": "training labels need target end, filing or "
            "label availability and origin before the (inner) test origin; "
            "inner validation labels must have arrived by the outer origin",
            "unsupported": "unscorable; gaps and support are never reduced",
        },
        "share_basis": (
            "Dollar features, targets and offsets are restated to the share "
            "basis of each fold's first test origin before fitting; "
            "forecasts are restated back to each test origin's basis"
        ),
        "feature_rows": "all registered features finite; no imputation "
        "of a missing report month",
        "campaign_slots": {
            "v201": "1-6", "v202": "7-14", "v203": "15-22",
            "v204": "23-28", "v205": "29-34", "v206": "35-38",
        },
        "post_result_search": "none; no feature, lag, threshold, window or "
        "grid change after results",
        "holdout": "no quarantine label, feature or metric is read",
    }


def json_value(value: object) -> object:
    """Strict portable JSON: missing values are null, never NaN."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, date, Path)):
        return str(value)
    return value


def save_json(path: Path, payload: object) -> None:
    """Save a reviewable research artifact, never a database mutation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_value(payload), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    """Round-trip floats and deterministic row order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def frame_hash(frame: pd.DataFrame) -> str:
    """Seal a partition's bytes."""
    return hashlib.sha256(
        frame.to_csv(
            index=False, float_format="%.17g", lineterminator="\n"
        ).encode()
    ).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def write_blocked(error: Exception, stage: str) -> None:
    """Stop before fitting and leave an explicit owner-readable blocker."""
    save_json(
        OUTPUTS / "blocked_attempt.json",
        {"status": "blocked", "stage": stage, "error": repr(error)},
    )
    (STUDY / "README.md").write_text(
        "v205 is blocked. A required pin, runtime, target or preregistration "
        f"check failed during {stage}, so no model was fitted and no "
        "forecast was scored.\n\n"
        f"Error: `{error!r}`\n\nFix the named input under a separately "
        "authorized change and rerun; never loosen the check.\n",
        encoding="utf-8",
    )


def verify_lock_with_registry_append(
    lock_path: Path, scratch: Path,
) -> dict[str, Any]:
    """Verify the v200 lock; allow only append-only registry growth.

    v200 consumed ``research/registry.yaml``, which every later study must
    extend with its own entry. That one file is verified at the v200
    execution commit and must keep every pinned entry unchanged; all other
    consumed files must match their pins in the working tree.
    """
    import yaml

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    pins = [
        item for item in lock["input_files"] if item["path"] == REGISTRY
    ]
    if len(pins) != 1:
        raise ValueError("The v200 lock must pin the research registry once")
    filtered = dict(lock)
    filtered["input_files"] = [
        item for item in lock["input_files"] if item["path"] != REGISTRY
    ]
    scratch.mkdir(parents=True, exist_ok=True)
    copy = scratch / "v200_lock_without_registry.json"
    copy.write_text(json.dumps(filtered), encoding="utf-8")
    verify_baseline_lock(copy)
    commit = lock["research_execution_code_commit"]
    pinned = subprocess.run(
        ["git", "show", f"{commit}:{REGISTRY}"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    ).stdout.replace(b"\r\n", b"\n")
    if hashlib.sha256(pinned).hexdigest() != pins[0]["sha256"]:
        raise ValueError("Pinned registry bytes differ at the v200 commit")
    before = yaml.safe_load(pinned)["studies"]
    after = yaml.safe_load(
        (ROOT / REGISTRY).read_text(encoding="utf-8")
    )["studies"]
    missing = [entry["id"] for entry in before if entry not in after]
    if missing:
        raise ValueError(f"Pinned registry entries changed: {missing}")
    return {
        "lock": lock,
        "registry_check": {
            "path": REGISTRY,
            "pinned_sha256": pins[0]["sha256"],
            "pinned_at_commit": commit,
            "current_sha256": source_sha256(ROOT / REGISTRY),
            "pinned_entries_unchanged": len(before),
            "added_entries": [
                entry["id"] for entry in after if entry not in before
            ],
        },
    }


def preflight(scratch: Path) -> dict[str, Any]:
    """Verify the accepted lock, runtime, v200 pins and pinned DB bytes."""
    lock_path = ROOT / V200 / "outputs/baseline_lock.json"
    checked = verify_lock_with_registry_append(lock_path, scratch)
    lock = checked["lock"]
    verify_runtime(
        json.loads(
            (ROOT / V200 / "outputs/runtime_lock.json").read_text("utf-8")
        )
    )
    manifest = json.loads(
        (ROOT / V200 / "outputs/output_manifest.json").read_text("utf-8")
    )["sha256"]
    pins = {}
    for relative, expected in V200_PINS.items():
        actual = sha256_file(ROOT / relative)
        listed = manifest.get(relative.removeprefix(f"{V200}/"))
        if expected is not None and actual != expected:
            raise ValueError(f"Pinned v200 file changed: {relative}")
        if listed is not None and listed != actual:
            raise ValueError(f"v200 manifest mismatch: {relative}")
        pins[relative] = actual
    database = export_git_blob(
        INPUT_DB_COMMIT,
        "data/pgr_financials.db",
        scratch / "pgr_financials_ed7997f.db",
        INPUT_DB_SHA256,
    )
    tracked = ROOT / "data/pgr_financials.db"
    return {
        "lock": lock,
        "registry_check": checked["registry_check"],
        "lock_sha256": sha256_file(lock_path),
        "v200_pins": pins,
        "database": database,
        "tracked_db_sha256": sha256_file(tracked) if tracked.exists() else None,
    }


def load_sources(connection: sqlite3.Connection) -> dict[str, Any]:
    """Bounded PGR sources: nothing dated on or after the boundary."""
    cutoff = str(BOUNDARY.date())
    prices = pd.read_sql_query(
        SQL["prices"], connection, params=(cutoff,), parse_dates=["date"]
    ).set_index("date")["close"].astype(float)
    dividends = pd.read_sql_query(
        SQL["dividends"], connection, params=(cutoff,),
        parse_dates=["ex_date"],
    ).set_index("ex_date")["amount"].astype(float)
    splits = pd.read_sql_query(
        SQL["splits"], connection, params=(cutoff,),
        parse_dates=["split_date"],
    ).set_index("split_date")["split_ratio"].astype(float)
    reports = pd.read_sql_query(
        SQL["reports"], connection, params=(cutoff, cutoff),
        parse_dates=["month_end", "filing_date"],
    )
    first = pd.read_sql_query(
        SQL["first_reported"], connection, params=(cutoff, cutoff),
        parse_dates=["month_end", "filing_date"],
    )
    if prices.index.has_duplicates or dividends.index.has_duplicates:
        raise ValueError("Duplicate PGR economic dates are unsupported")
    if reports["month_end"].duplicated().any():
        raise ValueError("Duplicate monthly PGR reports are unsupported")
    if (reports["filing_date"] < reports["month_end"]).any():
        raise ValueError("A report cannot be filed before its month ends")
    return {
        "prices": prices,
        "dividends": dividends,
        "splits": splits,
        "reports": reports,
        "first_reported": first,
    }


def load_controls() -> pd.DataFrame:
    """The exact pinned v200 development control forecasts."""
    controls = pd.read_csv(
        ROOT / V200 / "outputs/control_predictions.csv",
        parse_dates=["date", "target_end", "available"],
    )
    controls["warmup"] = controls["warmup"].astype(bool)
    return controls


def build_targets(
    sources: dict[str, Any], controls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Hand-calculate every label and check it against v200 exactly."""
    prices = sources["prices"]
    dividends = sources["dividends"]
    splits = sources["splits"]
    reports = sources["reports"]
    rows: list[dict[str, Any]] = []
    for control in controls.to_dict("records"):
        origin = pd.Timestamp(control["date"])
        endpoint = control["endpoint"]
        record = {
            "date": origin,
            "endpoint": endpoint,
            "v200_y_true": control["y_true"],
            "v200_target_end": control["target_end"],
            "v200_available": control["available"],
            "v200_y_hat": control["y_hat"],
            "v200_naive": control["naive"],
            "v200_warmup": control["warmup"],
            "control_value": float("nan"),
        }
        if endpoint == "cash_dividend_12m":
            value, end = xseries.next_cash_12m(dividends, splits, origin)
            prior = origin - pd.DateOffset(months=12)
            control_value = (
                xseries.past_cash(dividends, splits, origin, 12, 0)
                if prices.index.min() <= prior
                else float("nan")
            )
            record.update(
                y_true=value, target_end=end, available=end,
                control_value=control_value,
            )
        elif endpoint == "bvps_growth_12m":
            target = xseries.bvps_growth_target(reports, splits, origin, 12)
            if target is None:
                raise ValueError(f"Missing BVPS label at {origin.date()}")
            record.update(
                y_true=target["value"],
                target_end=target["target_end"],
                available=target["available"],
            )
        elif endpoint == "pgr_drip_return_6m":
            end = label_end(origin, 6)
            _, value = drip_return(prices, dividends, splits, origin, end)
            record.update(y_true=value, target_end=end, available=end)
        elif endpoint == "annual_excess_to_bvps":
            target = xseries.annual_excess_target(
                dividends, splits, reports, origin
            )
            if target is None:
                raise ValueError(f"Missing annual label at {origin.date()}")
            record.update(
                y_true=target["value"],
                target_end=target["target_end"],
                available=target["available"],
            )
        else:
            raise ValueError(f"Unregistered v200 endpoint {endpoint}")
        rows.append(record)
    # v200's prevailing means also count matured labels from business
    # month-end origins after the first price bar that precede its first
    # output row. Rebuild them here so the naive check is independent.
    first_bar = pd.offsets.BMonthEnd().rollforward(prices.index.min())
    for endpoint in ("cash_dividend_12m", "pgr_drip_return_6m"):
        first_output = controls.loc[
            controls["endpoint"] == endpoint, "date"
        ].min()
        for origin in pd.date_range(first_bar, first_output, freq="BME"):
            if origin >= first_output:
                continue
            if endpoint == "cash_dividend_12m":
                value, end = xseries.next_cash_12m(dividends, splits, origin)
            else:
                end = label_end(origin, 6)
                _, value = drip_return(prices, dividends, splits, origin, end)
            rows.append(
                {
                    "date": origin,
                    "endpoint": endpoint,
                    "y_true": value,
                    "target_end": end,
                    "available": end,
                    "control_value": float("nan"),
                    "in_v200_output": False,
                }
            )
    targets = pd.DataFrame(rows)
    targets["in_v200_output"] = (
        targets["in_v200_output"].astype("boolean").fillna(True).astype(bool)
    )
    # Pre-output history rows have no v200 forecast: always warmup.
    targets["v200_warmup"] = (
        targets["v200_warmup"].astype("boolean").fillna(True).astype(bool)
    )
    targets = targets.sort_values(["endpoint", "date"], kind="stable")
    targets = targets.reset_index(drop=True)
    output_rows = targets["in_v200_output"]
    targets["abs_diff"] = (targets["y_true"] - targets["v200_y_true"]).abs()
    control_diff = (
        targets["control_value"] - targets["v200_y_hat"]
    ).abs().where(targets["endpoint"] == "cash_dividend_12m")
    naive = []
    for row in targets.to_dict("records"):
        same = targets.loc[targets["endpoint"] == row["endpoint"]]
        past = same.loc[
            (same["date"] < row["date"])
            & (same["available"] <= row["date"])
            & (same["target_end"] <= row["date"])
        ]
        naive.append(float(past["y_true"].mean()) if len(past) else np.nan)
    targets["naive"] = naive
    naive_diff = (targets["naive"] - targets["v200_naive"]).abs()
    checks = {
        "n_rows": int(output_rows.sum()),
        "n_pre_output_history_rows": int((~output_rows).sum()),
        "max_abs_label_diff": float(targets["abs_diff"].max()),
        "max_abs_cash_control_diff": float(control_diff.max()),
        "max_abs_naive_diff": float(naive_diff.max()),
        "end_mismatches": int(
            (targets["target_end"] != targets["v200_target_end"])[
                output_rows
            ].sum()
        ),
        "availability_mismatches": int(
            (targets["available"] != targets["v200_available"])[
                output_rows
            ].sum()
        ),
        "cash_control_nan_mismatches": int(
            (
                targets["control_value"].isna()
                != targets["v200_y_hat"].isna()
            )[targets["endpoint"] == "cash_dividend_12m"].sum()
        ),
        "naive_nan_mismatches": int(
            (targets["naive"].isna() != targets["v200_naive"].isna())[
                output_rows
            ].sum()
        ),
        "tolerance": 1e-12,
    }
    failed = (
        checks["max_abs_label_diff"] > 1e-12
        or checks["max_abs_cash_control_diff"] > 1e-12
        or checks["max_abs_naive_diff"] > 1e-12
        or checks["end_mismatches"]
        or checks["availability_mismatches"]
        or checks["cash_control_nan_mismatches"]
        or checks["naive_nan_mismatches"]
    )
    checks["status"] = "failed" if failed else "passed"
    if failed:
        raise ValueError(f"Hand-calculated targets differ from v200: {checks}")
    b3_rows = []
    for origin in targets.loc[
        (targets["endpoint"] == "pgr_drip_return_6m") & output_rows, "date"
    ]:
        target = xseries.adjusted_bvps_growth_target(
            reports, dividends, splits, origin, 6
        )
        if target is None:
            continue
        b3_rows.append(
            {
                "date": origin,
                "endpoint": "adjusted_bvps_growth_6m",
                "y_true": target["value"],
                "report_month": target["report_month"],
                "future_month": target["future_month"],
                "target_end": target["target_end"],
                "available": target["available"],
            }
        )
    training = pd.DataFrame(b3_rows)
    return targets, training, checks


def feature_table(
    sources: dict[str, Any], origins: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Rolling monthly features at every business month-end origin."""
    rows = []
    for origin in origins:
        values = xseries.lane_features(
            sources["reports"], sources["dividends"], sources["splits"],
            origin,
        )
        values["date"] = origin
        values["max_ex_date_used"] = (
            sources["dividends"].index[
                sources["dividends"].index <= origin
            ].max()
        )
        rows.append(values)
    frame = pd.DataFrame(rows)
    columns = ["date"] + xseries.FEATURE_COLUMNS + ["max_ex_date_used"]
    frame = frame[columns]
    late = frame["report_filing_date"] > frame["date"]
    if late.any():
        raise ValueError("A feature used a report filed after its origin")
    return frame


def target_audit(
    sources: dict[str, Any], targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Descriptive: archived raw handling versus the clean labels.

    Label arithmetic only; nothing is fitted or selected here.
    """
    prices = sources["prices"]
    dividends = sources["dividends"]
    reports = sources["reports"]
    no_splits = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    rows = []
    for row in targets.loc[targets["in_v200_output"]].to_dict("records"):
        origin = row["date"]
        if row["endpoint"] == "bvps_growth_12m":
            raw = xseries.bvps_growth_target(reports, no_splits, origin, 12)
            archived = raw["value"] if raw else np.nan
            handling = "raw BVPS without split restatement (F15)"
        elif row["endpoint"] == "pgr_drip_return_6m":
            start = prices.loc[prices.index <= origin]
            end = prices.loc[prices.index <= row["target_end"]]
            archived = float(end.iloc[-1] / start.iloc[-1] - 1.0)
            handling = "raw close ratio: no splits, no dividends (F25)"
        elif row["endpoint"] == "cash_dividend_12m":
            window = dividends.loc[
                (dividends.index > origin)
                & (dividends.index <= row["target_end"])
            ]
            archived = float(window.sum())
            handling = "raw per-share cash, no split restatement"
        else:
            year = origin.year + 1
            start, end = xseries.annual_window(origin)
            december_to_february = float(
                dividends.loc[
                    (dividends.index >= start) & (dividends.index <= end)
                ].sum()
            )
            archived = float(
                dividends.loc[
                    (dividends.index >= pd.Timestamp(year, 1, 1))
                    & (dividends.index <= pd.Timestamp(year, 3, 31))
                ].sum()
                - december_to_february
            )
            handling = (
                "x1 January-March cash minus December-February cash; "
                "negative when a December special is missed (F25)"
            )
        rows.append(
            {
                "date": origin,
                "endpoint": row["endpoint"],
                "clean": row["y_true"],
                "archived_handling": archived,
                "difference": archived
                if row["endpoint"] == "annual_excess_to_bvps"
                else archived - row["y_true"],
                "handling": handling,
            }
        )
    audit = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for endpoint, group in audit.groupby("endpoint"):
        diff = group["difference"].abs()
        summary[endpoint] = {
            "n": len(group),
            "n_abs_diff_over_01": int((diff > 0.01).sum()),
            "max_abs_diff": float(diff.max()) if diff.notna().any() else None,
            "mean_signed_diff": float(group["difference"].mean())
            if group["difference"].notna().any()
            else None,
            "first_over_01": str(group.loc[diff > 0.01, "date"].min())
            if (diff > 0.01).any()
            else None,
            "last_over_01": str(group.loc[diff > 0.01, "date"].max())
            if (diff > 0.01).any()
            else None,
        }
    december = dividends.loc[dividends.index.month == 12]
    summary["december_ex_dates_before_boundary"] = [
        {"ex_date": str(day.date()), "amount": float(amount)}
        for day, amount in december.items()
    ]
    return audit, summary


def annual_support(sources: dict[str, Any]) -> pd.DataFrame:
    """Every post-policy November origin and why it is or is not a label."""
    rows = []
    for year in range(2018, BOUNDARY.year + 1):
        origin = pd.offsets.BMonthEnd().rollback(pd.Timestamp(year, 11, 30))
        start, end = xseries.annual_window(origin)
        record: dict[str, Any] = {
            "date": origin,
            "window_start": start,
            "window_end": end,
        }
        if end >= BOUNDARY:
            record["status"] = "outcome after development boundary"
            rows.append(record)
            continue
        target = xseries.annual_excess_target(
            sources["dividends"], sources["splits"], sources["reports"],
            origin,
        )
        features = xseries.lane_features(
            sources["reports"], sources["dividends"], sources["splits"],
            origin,
        )
        record.update(
            gainshare_estimate=features["gainshare_estimate"],
            investment_book_yield=features["investment_book_yield"],
        )
        if target is None:
            record["status"] = (
                "no causal ordinary baseline (no payment <= .25 in 24M)"
            )
        else:
            record.update(
                ordinary=target["ordinary"],
                window_cash=target["window_cash"],
                current_bvps=target["current_bvps"],
                value=target["value"],
                status="positive annual event"
                if target["positive"]
                else "zero excess: outside conditional-positive endpoint",
            )
        rows.append(record)
    return pd.DataFrame(rows)


def dividend_decomposition(sources: dict[str, Any]) -> pd.DataFrame:
    """Descriptive ordinary/special split by payment year, raw per share."""
    dividends = sources["dividends"]
    frame = pd.DataFrame(
        {"ex_date": dividends.index, "amount": dividends.to_numpy()}
    )
    frame["component"] = np.where(
        frame["ex_date"] < xseries.POLICY_CHANGE,
        "pre_policy_annual_variable",
        np.where(
            frame["amount"] <= 0.25, "regular_quarterly", "special",
        ),
    )
    frame["year"] = frame["ex_date"].dt.year
    return (
        frame.groupby(["year", "component"])["amount"]
        .agg(cash="sum", payments="count")
        .reset_index()
    )


def vintage_audit(sources: dict[str, Any]) -> dict[str, Any]:
    """Disclose current-versus-first-reported differences in used fields."""
    first = sources["first_reported"]
    current = sources["reports"].set_index("month_end")
    summary: dict[str, Any] = {
        "basis": "features use the repaired current table with actual "
        "filing-date gates, matching the v200 controls; first-reported "
        "values are compared only for disclosure",
    }
    for field in VINTAGE_FIELDS:
        values = first.loc[first["field"] == field].set_index("month_end")
        joined = values[["value_real"]].join(current[[field]], how="inner")
        diff = (joined["value_real"] - joined[field]).abs()
        tolerance = 1e-9 * np.maximum(1.0, joined[field].abs())
        changed = diff > tolerance
        summary[field] = {
            "n_compared": int(diff.notna().sum()),
            "n_changed": int(changed.sum()),
            "months_changed": [str(day.date()) for day in joined.index[changed]][
                :20
            ],
        }
    return summary


def partition_lock(
    targets: pd.DataFrame, training: pd.DataFrame,
) -> dict[str, Any]:
    """Seal every development label set; fail if any touches the boundary."""
    lock: dict[str, Any] = {
        "boundary": str(BOUNDARY.date()),
        "rule": "target_end and label availability strictly before the "
        "boundary; quarantine definitions inherited from v200 unchanged",
        "endpoints": {},
    }
    frames = [
        (name, group)
        for name, group in targets.groupby("endpoint", sort=True)
    ] + [("adjusted_bvps_growth_6m", training)]
    for name, group in frames:
        columns = group[["date", "target_end", "available"]].sort_values(
            "date"
        )
        if (columns["target_end"] >= BOUNDARY).any() or (
            columns["available"] >= BOUNDARY
        ).any():
            raise ValueError(f"{name} label reaches the quarantine boundary")
        lock["endpoints"][name] = {
            "n_rows": len(columns),
            "first_origin": str(columns["date"].min().date()),
            "last_origin": str(columns["date"].max().date()),
            "last_available": str(columns["available"].max().date()),
            "sha256_dates_ends_availability": frame_hash(columns),
        }
    v200_partitions = json.loads(
        (ROOT / V200 / "outputs/partitions.json").read_text("utf-8")
    )
    if pd.Timestamp(v200_partitions["boundary"]) != BOUNDARY:
        raise ValueError("v200 development boundary changed")
    lock["v200_quarantine"] = v200_partitions["quarantine"]
    lock["v200_partitions_sha256"] = sha256_file(
        ROOT / V200 / "outputs/partitions.json"
    )
    return lock


def build_preregistration(
    context: dict[str, Any], scratch: Path,
) -> dict[str, Any]:
    """Everything that must be frozen before any candidate is fitted."""
    connection = read_immutable(context["database"], INPUT_DB_SHA256)
    try:
        sources = load_sources(connection)
    finally:
        connection.close()
    controls = load_controls()
    targets, training, checks = build_targets(sources, controls)
    listed = targets.loc[targets["in_v200_output"], "date"]
    origins = pd.date_range(listed.min(), listed.max(), freq="BME")
    features = feature_table(sources, origins)
    audit, audit_summary = target_audit(sources, targets)
    annual = annual_support(sources)
    lock = partition_lock(targets, training)
    return {
        "sources": sources,
        "targets": targets,
        "training": training,
        "checks": checks,
        "features": features,
        "audit": audit,
        "audit_summary": audit_summary,
        "annual": annual,
        "decomposition": dividend_decomposition(sources),
        "vintage": vintage_audit(sources),
        "partition_lock": lock,
    }


def preregistration_payload(
    context: dict[str, Any], frozen: dict[str, Any],
) -> dict[str, Any]:
    """The frozen document the execute phase must reproduce exactly."""
    annual = frozen["annual"]
    positives = int((annual["status"] == "positive annual event").sum())
    return {
        "study": "v205_dividend_bvps",
        "as_of": str(AS_OF.date()),
        "development_boundary": str(BOUNDARY.date()),
        "candidates": candidate_register(),
        "rules": preregistered_rules(),
        "endpoints": ENDPOINTS,
        "archived_contracts": {
            "x23_excess": "x18/x23: November post-policy origin, December 1 "
            "to next February 28/29 cash minus the median positive payment "
            "<= .25 in the prior 24 months (v200 freezes it at the "
            "November origin), divided by the latest filed November BVPS; "
            "conditional positive-excess survivor to_current_bvps",
            "x16_structural": "adjusted_structural_bvps_pb_6m = "
            "ridge_bridge__no_change_pb on x9 bvps_lags; x12 adjusted "
            "target (future BVPS + dividends paid in the next h report "
            "months) / current BVPS - 1; P/B held at its current value",
            "x16_alpha_note": "The archived ridge_bridge used a fixed "
            "alpha 1000; this campaign's budget allows only the nested "
            "{1,10,100} grid, so B3 selects inside that grid",
            "x17_persistent": "BVPS plus cumulative dividends; the x12 "
            "adjusted target is the same idea over each forecast window",
        },
        "d3_support": {
            "positive_annual_events": positives,
            "rule_outcome": "closed_insufficient_annual_support"
            if positives < 5
            else "eligible",
        },
        "target_checks": frozen["checks"],
        "partition_lock": frozen["partition_lock"],
        "pins": {
            "baseline_lock_sha256": context["lock_sha256"],
            "baseline_code_commit": BASELINE_CODE_COMMIT,
            "input_db_commit": INPUT_DB_COMMIT,
            "input_db_sha256": INPUT_DB_SHA256,
            "parent_seed_commit": PARENT_SEED_COMMIT,
            "parent_seed_db_sha256": PARENT_SEED_SHA256,
            "v200_files": context["v200_pins"],
        },
        "artifact_sha256": {
            "features.csv": frame_hash(frozen["features"]),
            "targets.csv": frame_hash(frozen["targets"]),
            "b3_training_labels.csv": frame_hash(frozen["training"]),
        },
        "extraction_sql": SQL,
    }


def write_preregistration(
    context: dict[str, Any], frozen: dict[str, Any], destination: Path,
) -> None:
    save_csv(destination / "targets.csv", frozen["targets"])
    save_csv(destination / "b3_training_labels.csv", frozen["training"])
    save_csv(destination / "features.csv", frozen["features"])
    save_csv(destination / "target_audit.csv", frozen["audit"])
    save_json(destination / "target_audit.json", frozen["audit_summary"])
    save_csv(destination / "annual_support.csv", frozen["annual"])
    save_csv(
        destination / "dividend_decomposition.csv", frozen["decomposition"]
    )
    save_json(destination / "source_vintage_audit.json", frozen["vintage"])
    save_json(destination / "partition_lock.json", frozen["partition_lock"])
    save_json(
        destination / "preregistration.json",
        preregistration_payload(context, frozen),
    )
    save_json(
        destination / "access_ledger.json",
        {
            "study": "v205",
            "forecast_and_metric_scope": "development only",
            "holdout_metric_accesses": 0,
            "quarantine_label_reads": 0,
            "source_bound": f"every SELECT is dated before {BOUNDARY.date()}",
            "next_unseal": "v207 one frozen batch, D2",
        },
    )


# ---------------------------------------------------------------------------
# Execution: fitting and development scoring only.
# ---------------------------------------------------------------------------


def lane_frame(
    candidate: dict[str, Any],
    frozen: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly fold frame and the matched outcome/control rows."""
    endpoint = candidate["endpoint"]
    targets = frozen["targets"]
    outcomes = targets.loc[
        (targets["endpoint"] == endpoint) & targets["in_v200_output"]
    ].copy()
    outcomes = outcomes.sort_values("date").set_index("date")
    dates = pd.date_range(outcomes.index.min(), outcomes.index.max(), freq="BME")
    if not dates.equals(pd.DatetimeIndex(outcomes.index)):
        raise ValueError(f"{endpoint} origins are not contiguous")
    features = frozen["features"].set_index("date").reindex(dates)
    frame = features[candidate["features"]].copy()
    if candidate["id"] == "B3":
        labels = frozen["training"].set_index("date").reindex(dates)
        frame["y_train"] = labels["y_true"]
        frame["label_end"] = labels["target_end"]
        frame["available"] = labels["available"]
    else:
        frame["y_train"] = outcomes["y_true"]
        frame["label_end"] = outcomes["target_end"]
        frame["available"] = outcomes["available"]
    frame["offset"] = (
        features["past12_cash"] if endpoint == "cash_dividend_12m" else 0.0
    )
    return frame, outcomes


def run_candidate(
    candidate: dict[str, Any],
    frozen: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Nested chronological Ridge forecasts for one registered blueprint."""
    frame, outcomes = lane_frame(candidate, frozen)
    splits = frozen["sources"]["splits"]
    horizon = candidate["horizon"]
    dates = pd.DatetimeIndex(frame.index)
    dollar = [
        column
        for column in candidate["features"]
        if column in DOLLAR_FEATURES
    ]
    cash = candidate["endpoint"] == "cash_dividend_12m"
    records: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    folds = chronological_splits(dates, horizon)
    if not folds:
        ledger.append(
            {
                "candidate": candidate["id"],
                "kind": "outer",
                "status": "unscorable",
                "reason": "Insufficient history for fixed outer folds",
            }
        )
    for fold, (train, test) in enumerate(folds):
        basis = dates[test[0]]
        work = frame.copy()
        for column in dollar + (["y_train", "offset"] if cash else []):
            work[column] = xseries.restate_to_basis(
                work[column].to_numpy(dtype=float), dates, splits, basis
            )
        work["target"] = work["y_train"] - (work["offset"] if cash else 0.0)
        result = xseries.nested_ridge_forecast(
            work,
            candidate["features"],
            "target",
            train,
            test,
            horizon,
            xseries.ALPHA_GRID,
        )
        outer = {
            "candidate": candidate["id"],
            "kind": "outer",
            "fold": fold,
            "horizon": horizon,
            "gap": 2 * horizon,
            "purge": horizon,
            "embargo": horizon,
            "train_window": 60 if horizon == 6 else 120,
            "test_size": len(test),
            "basis_date": str(basis.date()),
            "status": result["status"],
            "reason": result["reason"],
            "n_train": result["n_train"],
            "train_start": result["train_start"],
            "train_end": result["train_end"],
            "test_start": result["test_start"],
            "test_end": result["test_end"],
            "selected_alpha": result["alpha"],
            "inner_absolute_error": json.dumps(
                result["inner_absolute_error"], sort_keys=True
            ),
        }
        ledger.append(outer)
        for entry in result["inner"]:
            ledger.append(
                {
                    "candidate": candidate["id"],
                    "kind": "inner",
                    "fold": fold,
                    "horizon": horizon,
                    "gap": entry["gap"],
                    "test_size": entry["test_size"],
                    "inner_fold": entry["inner_fold"],
                    "status": entry["status"],
                    "n_train": entry["n_train"],
                    "n_test_usable": entry["n_test_usable"],
                    "train_start": entry["train_start"],
                    "train_end": entry["train_end"],
                    "test_start": entry["test_start"],
                    "test_end": entry["test_end"],
                }
            )
        forecast = result["prediction"]
        if cash:
            forecast = forecast + work["offset"].to_numpy(dtype=float)[test]
            forecast = np.array([
                xseries.restate_to_basis(
                    np.array([value]),
                    pd.DatetimeIndex([basis]),
                    splits,
                    dates[index],
                )[0]
                for value, index in zip(forecast, test)
            ])
        for value, index in zip(forecast, test):
            origin = dates[index]
            outcome = outcomes.loc[origin]
            records.append(
                {
                    "date": origin,
                    "candidate": candidate["id"],
                    "endpoint": candidate["endpoint"],
                    "fold": fold,
                    "fold_status": result["status"],
                    "selected_alpha": result["alpha"],
                    "y_hat": float(value),
                    "y_true": float(outcome["y_true"]),
                    "target_end": outcome["target_end"],
                    "available": outcome["available"],
                    "control_y_hat": float(outcome["v200_y_hat"]),
                    "naive": float(outcome["v200_naive"]),
                    "control_warmup": bool(outcome["v200_warmup"]),
                }
            )
    return pd.DataFrame(records), ledger


def _threshold(endpoint: str, frame: pd.DataFrame, column: str) -> np.ndarray:
    if ENDPOINTS[endpoint]["threshold"] == "control":
        return frame[column].to_numpy(dtype=float)
    return np.zeros(len(frame))


def evaluate(
    candidate: dict[str, Any],
    predictions: pd.DataFrame,
    frozen: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Development metrics on matched rows; nothing here selects a model."""
    endpoint = candidate["endpoint"]
    horizon = candidate["horizon"]
    targets = frozen["targets"]
    history = targets.loc[targets["endpoint"] == endpoint].sort_values("date")
    matched = predictions.loc[
        np.isfinite(predictions["y_hat"])
        & np.isfinite(predictions["control_y_hat"])
        & np.isfinite(predictions["naive"])
        & np.isfinite(predictions["y_true"])
        & ~predictions["control_warmup"]
    ].sort_values("date")
    summary: dict[str, Any] = {
        "candidate": candidate["id"],
        "endpoint": endpoint,
        "unit": ENDPOINTS[endpoint]["unit"],
        "horizon": horizon,
        "n_outer_test_rows": len(predictions),
        "n_forecasts": int(np.isfinite(predictions["y_hat"]).sum()),
        "n_scored": len(matched),
    }
    control_stream = history.loc[
        np.isfinite(history["v200_y_hat"]) & ~history["v200_warmup"]
    ].assign(y_hat=lambda f: f["v200_y_hat"])
    control_stream["threshold"] = _threshold(
        endpoint, control_stream, "v200_y_hat"
    )
    control_cal = xseries.prequential_stream(
        control_stream[["date", "available", "y_true", "y_hat", "threshold"]],
        CALIBRATION_MIN_SUPPORT,
    ).set_index("date")
    stream = predictions.loc[np.isfinite(predictions["y_hat"])].copy()
    stream["threshold"] = _threshold(endpoint, stream, "control_y_hat")
    candidate_cal = xseries.prequential_stream(
        stream[["date", "available", "y_true", "y_hat", "threshold"]],
        CALIBRATION_MIN_SUPPORT,
    ).set_index("date")
    calibration = candidate_cal.join(
        control_cal[["lower", "upper", "probability", "covered", "warmup"]],
        rsuffix="_control",
    ).reset_index()
    calibration.insert(0, "candidate", candidate["id"])
    if len(matched) < 2:
        summary["status"] = "unscorable"
        summary["reason"] = "Fewer than two matched development rows"
        return summary, calibration
    y = matched["y_true"].to_numpy(dtype=float)
    p = matched["y_hat"].to_numpy(dtype=float)
    c = matched["control_y_hat"].to_numpy(dtype=float)
    naive = matched["naive"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(matched["date"])
    mae = float(np.mean(np.abs(y - p)))
    control_mae = float(np.mean(np.abs(y - c)))
    r2 = honest_r2(y, p, naive)
    control_r2 = honest_r2(y, c, naive)
    primary = xseries.paired_block_test(
        dates, np.abs(y - c) - np.abs(y - p), horizon, REPLICATES, SEED
    )
    threshold = _threshold(endpoint, matched, "control_y_hat")
    events = (y > threshold).astype(float)
    predicted = (p > threshold).astype(float)
    control_predicted = (c > threshold).astype(float)
    history_threshold = _threshold(endpoint, history, "v200_y_hat")
    history_events = np.where(
        np.isfinite(history_threshold),
        (history["y_true"].to_numpy(dtype=float) > history_threshold)
        .astype(float),
        np.nan,
    )
    base = xseries.majority_base(
        history["date"], history["available"], history_events, dates
    )
    base_ready = np.isfinite(base)
    hits = (predicted == events).astype(float)
    control_hits = (control_predicted == events).astype(float)
    base_hits = (base == events).astype(float)
    skill = xseries.paired_block_test(
        dates[base_ready],
        (hits - base_hits)[base_ready],
        horizon,
        REPLICATES,
        SEED,
    )
    matched_cal = calibration.set_index("date").loc[dates]
    both = ~matched_cal["warmup"].to_numpy(bool) & ~matched_cal[
        "warmup_control"
    ].fillna(True).to_numpy(bool)
    scores = xseries.probability_scores(
        matched_cal["probability"].to_numpy(dtype=float)[both],
        matched_cal["event"].to_numpy(dtype=float)[both],
    )
    control_scores = xseries.probability_scores(
        matched_cal["probability_control"].to_numpy(dtype=float)[both],
        matched_cal["event"].to_numpy(dtype=float)[both],
    )
    covered = matched_cal["covered"].to_numpy(dtype=float)[both]
    control_covered = matched_cal["covered_control"].to_numpy(dtype=float)[
        both
    ]
    ic = xseries.block_spearman(dates, y, p, horizon, REPLICATES, SEED)
    control_ic = xseries.block_spearman(
        dates, y, c, horizon, REPLICATES, SEED
    )
    alphas = predictions.loc[
        predictions["fold_status"] == "scorable", ["fold", "selected_alpha"]
    ].drop_duplicates()
    summary.update(
        status="scored",
        origin_start=str(dates.min().date()),
        origin_end=str(dates.max().date()),
        availability_end=str(pd.Timestamp(matched["available"].max()).date()),
        n_dates=len(dates),
        independent_blocks=len(dates) / horizon,
        mae=mae,
        control_mae=control_mae,
        relative_mae_reduction=1.0 - mae / control_mae,
        oos_r2=r2,
        control_oos_r2=control_r2,
        delta_r2=r2 - control_r2,
        primary_improvement=primary["observed"],
        primary_ci=primary["ci"],
        raw_p=primary["p_value"],
        inference_supported=primary["inference_supported"],
        ic=ic["ic"],
        ic_ci=ic["ci"],
        ic_p=ic["p_value"],
        control_ic=control_ic["ic"],
        hit_rate=float(hits.mean()),
        control_hit_rate=float(control_hits.mean()),
        base_hit_rate=float(base_hits[base_ready].mean())
        if base_ready.any()
        else float("nan"),
        base_rate=float(events.mean()),
        directional_skill=skill["observed"],
        directional_skill_p=skill["p_value"],
        n_calibration=int(both.sum()),
        n_calibration_warmup=int((~both).sum()),
        brier=scores["brier"],
        control_brier=control_scores["brier"],
        log_loss=scores["log_loss"],
        control_log_loss=control_scores["log_loss"],
        ece=scores["ece"],
        control_ece=control_scores["ece"],
        coverage_80=float(covered.mean()) if len(covered) else float("nan"),
        control_coverage_80=float(control_covered.mean())
        if len(control_covered)
        else float("nan"),
        selected_alpha_counts={
            str(alpha): int(count)
            for alpha, count in alphas["selected_alpha"]
            .value_counts()
            .sort_index()
            .items()
        },
    )
    return summary, calibration


def execute(scratch: Path, destination: Path) -> None:
    """Fit registered candidates after verifying the frozen register."""
    started = time.time()
    try:
        context = preflight(scratch)
        frozen = build_preregistration(context, scratch)
        rebuilt = json.dumps(
            json_value(preregistration_payload(context, frozen)),
            indent=2, sort_keys=True, allow_nan=False,
        ) + "\n"
        committed_path = OUTPUTS / "preregistration.json"
        committed = committed_path.read_text(encoding="utf-8")
        if rebuilt != committed:
            raise ValueError("Preregistration differs from the committed one")
        tracked = git("ls-files", "--", str(committed_path.relative_to(ROOT)))
        dirty = git(
            "status", "--porcelain", "--",
            str(committed_path.relative_to(ROOT)),
        )
        if not tracked or dirty:
            raise ValueError("Preregistration must be committed and clean")
        prereg_commit = git(
            "log", "-1", "--format=%H", "--",
            str(committed_path.relative_to(ROOT)),
        )
    except Exception as error:  # noqa: BLE001 - recorded blocker
        write_blocked(error, "execute preflight")
        raise
    predictions = []
    ledger: list[dict[str, Any]] = []
    summaries = []
    calibration = []
    attempts = []
    for candidate in candidate_register():
        if candidate["id"] == "D3":
            outcome = preregistration_payload(context, frozen)["d3_support"]
            summaries.append(
                {
                    "candidate": "D3",
                    "endpoint": candidate["endpoint"],
                    "status": outcome["rule_outcome"],
                    "positive_annual_events": outcome[
                        "positive_annual_events"
                    ],
                    "raw_p": 1.0,
                    "reason": "Preregistered closure: fewer than five "
                    "positive annual events and no 60-month inner history "
                    "for an annual endpoint; not fitted",
                }
            )
            attempts.append({"candidate": "D3", "fits": 0, "closed": True})
            continue
        frame, entries = run_candidate(candidate, frozen)
        predictions.append(frame)
        ledger.extend(entries)
        summary, stream = evaluate(candidate, frame, frozen)
        summaries.append(summary)
        calibration.append(stream)
        attempts.append(
            {
                "candidate": candidate["id"],
                "outer_folds": sum(
                    1 for entry in entries if entry["kind"] == "outer"
                ),
                "scorable_outer_folds": sum(
                    1
                    for entry in entries
                    if entry["kind"] == "outer"
                    and entry["status"] == "scorable"
                ),
                "closed": False,
            }
        )
    raw = [
        float(summary.get("raw_p", 1.0))
        if np.isfinite(summary.get("raw_p", np.nan))
        else 1.0
        for summary in summaries
    ]
    adjusted = xseries.campaign_holm(raw)
    rows = []
    for summary, value, adj in zip(summaries, raw, adjusted):
        summary["holm_p_input"] = value
        summary["adjusted_p"] = adj
        if summary.get("status") == "scored":
            decision = xseries.lane_disposition(summary)
        else:
            decision = {
                "passes": False,
                "failed": [summary.get("status", "unscorable")],
                "gates": {},
            }
        summary["disposition"] = decision
        rows.append(
            {
                "candidate": summary["candidate"],
                "passes": decision["passes"],
                "adjusted_p": adj,
                "relative_mae_reduction": summary.get(
                    "relative_mae_reduction", float("nan")
                ),
            }
        )
    winner = xseries.nominate_winner(rows)
    destination.mkdir(parents=True, exist_ok=True)
    save_csv(
        destination / "predictions.csv",
        pd.concat(predictions, ignore_index=True),
    )
    save_csv(destination / "fold_ledger.csv", pd.DataFrame(ledger))
    save_csv(
        destination / "calibration_streams.csv",
        pd.concat(calibration, ignore_index=True),
    )
    save_json(
        destination / "metrics.json",
        {summary["candidate"]: summary for summary in summaries},
    )
    save_json(
        destination / "multiplicity.json",
        {
            "family": "38 campaign primary tests; v205 slots 29-34; "
            "pending slots p=1",
            "raw_p": dict(zip([s["candidate"] for s in summaries], raw)),
            "holm_adjusted_p": dict(
                zip([s["candidate"] for s in summaries], adjusted)
            ),
            "familywise_alpha": 0.05,
            "v207": "completes the campaign adjustment",
        },
    )
    save_json(
        destination / "comparison.json",
        {
            "rows": rows,
            "winner": winner,
            "rule": preregistered_rules()["nomination"],
            "note": "Each candidate is compared only with its own matched "
            "v200 control; endpoints and units are never pooled",
        },
    )
    save_json(
        destination / "candidate_ledger.json",
        {
            "register": candidate_register(),
            "attempts": attempts,
            "preregistration_commit": prereg_commit,
            "alternatives_beyond_register": 0,
        },
    )
    save_json(
        destination / "run_record.json",
        {
            "execution_code_commit": git("rev-parse", "HEAD"),
            "preregistration_commit": prereg_commit,
            "runtime_seconds": round(time.time() - started, 1),
            "runtime": runtime_lock(),
            "tracked_db_sha256_before": context["tracked_db_sha256"],
            "tracked_db_sha256_after": sha256_file(
                ROOT / "data/pgr_financials.db"
            ),
        },
    )


def prepare(scratch: Path) -> None:
    """Freeze the register and partition locks; fit nothing."""
    try:
        context = preflight(scratch)
        frozen = build_preregistration(context, scratch)
    except Exception as error:  # noqa: BLE001 - recorded blocker
        write_blocked(error, "preregistration preflight")
        raise
    write_preregistration(context, frozen, OUTPUTS)
    save_json(
        OUTPUTS / "preflight.json",
        {
            "status": "passed",
            "baseline_lock_sha256": context["lock_sha256"],
            "v200_pins": context["v200_pins"],
            "input_db_sha256": sha256_file(context["database"]),
            "input_db_location": "external scratch copy, immutable read",
            "tracked_db_sha256": context["tracked_db_sha256"],
            "runtime": "matches v200 runtime lock",
            "registry_check": context["registry_check"],
            "target_checks": frozen["checks"],
            "source_docs": {
                path: source_sha256(ROOT / path) for path in SOURCE_DOCS
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--preregister", action="store_true")
    phase.add_argument("--execute", action="store_true")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    arguments = parser.parse_args()
    scratch = arguments.scratch.resolve()
    if scratch.is_relative_to(ROOT.resolve()):
        raise SystemExit("Use a scratch directory outside the repository")
    if arguments.preregister:
        prepare(scratch)
    else:
        execute(scratch, arguments.output_dir.resolve())


if __name__ == "__main__":
    main()
