"""One bounded, offline v202 insurance and valuation development study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

import config
from pgr_vds.research_lib.adapters import (
    RIDGE_GRID,
    SHRINKAGE_GRID,
    ensemble_stream,
    prequential_intervals,
)
from pgr_vds.research_lib.baseline import calibration_summary, strict_predictions
from pgr_vds.research_lib.insurance_valuation import (
    fiscal_month_shift,
    insurance_features,
    paired_delta_bootstrap,
    select_causal_recipe,
)
from pgr_vds.research_lib.metrics import holm38, panel_summary
from pgr_vds.research_lib.price_macro_disposition import (
    candidate_disposition,
    nominate_finalist,
)
from pgr_vds.research_lib.provenance import (
    export_git_blob,
    read_immutable,
    sha256_file,
    verify_runtime,
)
from pgr_vds.research_lib.snapshot import verify_registry_growth
from pgr_vds.research_lib.temporal import label_end


ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
OUTPUTS = STUDY / "outputs"
BASELINE = ROOT / "research/studies/v200_clean_baseline/outputs"
V201 = ROOT / "research/studies/v201_price_macro/outputs/attempt2"
V205 = ROOT / "research/studies/v205_dividend_bvps/outputs"
LOCK_SHA = "c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0"
SEED = 20260926
AS_OF = "2026-09-26"
BOUNDARY = pd.Timestamp("2023-09-29")
FEATURES = {
    key: list(config.MODEL_FEATURE_OVERRIDES[key]) for key in ("ridge", "gbt")
}
BENCHMARKS = list(config.PRIMARY_FORECAST_UNIVERSE)
BLUEPRINTS = {
    "I1": {"remove": ["combined_ratio_ttm"],
           "add": ["iv_cr_ttm", "iv_cr_change"]},
    "I2": {"remove": ["pif_growth_yoy"], "add": ["iv_pif_growth"]},
    "I3": {"remove": [], "add": ["iv_gainshare"]},
    "I4": {"remove": ["investment_income_growth_yoy",
                      "investment_book_yield"],
           "add": ["iv_income_growth", "iv_book_yield"]},
    "I5": {"remove": ["book_value_per_share_growth_yoy"],
           "add": ["iv_bvps_growth"]},
    "I6": {"remove": [], "add": ["iv_roe"]},
    "I7": {"remove": [],
           "add": ["iv_pb", "iv_pe", "iv_pb_pe_spread"]},
    "I8": {"remove": ["npw_growth_yoy", "rate_adequacy_gap_yoy"],
           "add": ["iv_npw_t12_yoy", "iv_rate_gap"]},
}
SQL = {
    "monthly": "SELECT * FROM pgr_edgar_monthly WHERE month_end<=? "
               "ORDER BY month_end",
    "quarterly": "SELECT * FROM pgr_fundamentals_quarterly "
                 "WHERE period_end<=? ORDER BY period_end",
    "prices": "SELECT date,close FROM daily_prices WHERE ticker='PGR' "
              "AND date<=? AND proxy_fill=0 ORDER BY date",
    "splits": "SELECT split_date,split_ratio FROM split_history "
              "WHERE ticker='PGR' AND split_date<=? ORDER BY split_date",
    "ppi": "SELECT month_end,value FROM fred_macro_monthly "
           "WHERE series_id='PCU5241265241261' AND month_end<=? "
           "ORDER BY month_end",
}


def save_json(path: Path, value: Any) -> None:
    """Write portable JSON and preserve missing numeric values as null."""
    def normal(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): normal(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [normal(val) for val in item]
        if isinstance(item, (pd.Timestamp, Path)):
            return str(item)
        if isinstance(item, np.generic):
            return normal(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normal(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write deterministic full precision CSV artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def load_csv(path: Path, dates: list[str]) -> pd.DataFrame:
    """Load pinned CSV with round-trip floats and declared date columns."""
    return pd.read_csv(path, parse_dates=dates, float_precision="round_trip")


def git(*arguments: str) -> str:
    """Read the local Git repository without provider access."""
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True
    ).strip()


def preflight(scratch: Path) -> dict[str, Any]:
    """Stop before fitting if the accepted repaired v200 pin changes."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    try:
        lock_path = BASELINE / "baseline_lock.json"
        if sha256_file(lock_path) != LOCK_SHA:
            raise ValueError("Accepted v200 baseline lock bytes changed")
        checked = verify_registry_growth(lock_path, scratch)
        lock = checked["lock"]
        if lock["comparison_status"] != "accepted_research_comparator":
            raise ValueError("Provisional v200 comparison is prohibited")
        manifest = json.loads((BASELINE / "output_manifest.json").read_text())
        for relative, digest in manifest["sha256"].items():
            if sha256_file(BASELINE.parent / relative) != digest:
                raise ValueError(f"v200 output changed: {relative}")
        verify_runtime(json.loads((BASELINE / "runtime_lock.json").read_text()))
        database = lock["input_db"]
        scratch.mkdir(parents=True, exist_ok=True)
        db_copy = scratch / "pinned.db"
        export_git_blob(
            database["git_commit"], database["relative_path"],
            db_copy, database["sha256"],
        )
        with read_immutable(db_copy, database["sha256"]) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Pinned DB integrity check failed")
        return {
            "status": "passed", "lock_sha256": LOCK_SHA,
            "baseline_code_commit": lock["baseline_code_commit"],
            "parent_seed": lock["parent_seed"],
            "repairs": lock["repairs"], "database": database,
            "db_copy": db_copy,
            "tracked_db_sha256": sha256_file(ROOT / "data/pgr_financials.db"),
            "registry_check": checked["registry_check"],
            "fit_count": 0,
            "accepted_exception": "VWO March 2026 provider gap; "
                                  "12 affected targets quarantined",
        }
    except Exception as error:
        (STUDY / "README.md").write_text(
            "v202 is blocked before fitting. No holdout was opened.\n\n"
            f"Preflight failure: {error!r}.\n\n"
            "What changed: the blocker was recorded. "
            "What is left: repair the pin in a separate authorized session.\n",
            encoding="utf-8",
        )
        save_json(OUTPUTS / "preflight.json", {
            "status": "blocked", "error": repr(error), "fits": 0,
        })
        raise


def procedure() -> dict[str, Any]:
    """Freeze all eight feature blocks and one primary endpoint."""
    return {
        "candidate_order": list(BLUEPRINTS),
        "candidate_slots": list(range(7, 15)),
        "blueprints": BLUEPRINTS,
        "columns": {
            name: {
                model: [column for column in original
                        if column not in blueprint["remove"]]
                       + blueprint["add"]
                for model, original in FEATURES.items()
            }
            for name, blueprint in BLUEPRINTS.items()
        },
        "ridge_grid": RIDGE_GRID.tolist(),
        "shrinkage_grid": SHRINKAGE_GRID.tolist(),
        "gbt": {"max_depth": 2, "n_estimators": 50,
                "learning_rate": 0.1, "subsample": 0.8,
                "random_state": 42},
        "outer": {"h6": [60, 6, 12], "h12": [120, 6, 24]},
        "inner": {"folds": 3, "test_size": 6,
                  "gap": "2*h", "max_train_size": "outer window",
                  "minimum_months": {"h6": 24, "h12": 60}},
        "source_availability": "actual filing to same-or-next business month end; "
                               "missing filing fallback report month+2; "
                               "no optimized lag; maximum staleness 3 months",
        "source_vintage": "stored revised rows; first-reported historical "
                          "values unrecoverable",
        "npw": "trailing twelve complete calendar-month sums after A3",
        "ppi": "frozen one-month calendar publication lag; latest vintage",
        "descriptive_pb_pe": {
            "eras": [["2004-01-01", "2014-12-31"],
                     ["2015-01-01", "2023-09-28"]],
            "signals": ["iv_pb", "iv_pe", "iv_pb_pe_spread"],
            "tests": 6, "holm_family": "separate from 38 candidates",
        },
        "upstream_v201": "select only from its six frozen candidates "
                         "inside each downstream three-fold inner history; "
                         "unsupported early folds stay missing",
        "primary": "paired 6M squared-error gain vs exact v200 support",
        "secondary": "12M, per-era valuation and v201 comparator cannot rescue primary",
        "inference": {"seed": SEED, "replicates": 2000,
                      "date_block_length": "h", "family_size": 38},
        "thresholds": {"delta_r2": 0.010, "holm_p": 0.05,
                       "max_ic_loss": 0.01,
                       "safeguards": "same strict no-deterioration rule as v201"},
        "costs": "forecast research only; no policy or transaction P&L",
        "sql": SQL,
    }


def audit_sources(monthly: pd.DataFrame, quarterly: pd.DataFrame,
                  splits: pd.DataFrame) -> dict[str, Any]:
    """Record repaired units, calendar gaps and filing limitations."""
    months = monthly.copy()
    months["month_end"] = pd.to_datetime(months.month_end)
    months = months.set_index("month_end").sort_index()
    calendar = pd.date_range(months.index.min(), months.index.max(), freq="ME")
    missing = calendar.difference(months.index)
    q = quarterly.copy()
    q["period_end"] = pd.to_datetime(q.period_end)
    q4 = q.loc[q.period_end.dt.month == 12]
    reconciled = []
    for row in q4.itertuples():
        start = row.period_end - pd.DateOffset(months=3)
        window = months.loc[(months.index > start) &
                            (months.index <= row.period_end), "net_income"]
        if len(window) == 3 and pd.notna(row.net_income):
            reconciled.append(float(window.sum() - row.net_income / 1e6))
    equity_gap = (
        months.total_assets - months.total_liabilities
        - months.shareholders_equity
    )
    return {
        "monthly_rows": len(months),
        "missing_calendar_months": [str(value.date()) for value in missing],
        "leap_february_rows": [str(value.date()) for value in months.index
                                if value.month == 2 and value.day == 29],
        "negative_net_income_months": int((months.net_income < 0).sum()),
        "book_yield_unit": "stored percent, not fraction",
        "book_yield_min_max": [float(months.investment_book_yield.min()),
                               float(months.investment_book_yield.max())],
        "common_equity_identity_max_abs_millions": float(
            equity_gap.abs().max()),
        "q4_quarterly_vs_monthly_ni_max_abs_millions":
            max(map(abs, reconciled), default=None),
        "q4_compared": len(reconciled),
        "split_actions": splits.to_dict("records"),
        "source_vintage_limit": "stored repaired values are not a "
                                "historical first-report vintage",
        "pif_vintage_limit": "pif_total differs from first-reported "
                             "values in 50 months from 2019-07; stable "
                             "components are not verified first vintages",
    }


def prepare(scratch: Path) -> None:
    """Perform source audit, freeze feature artifacts and preregistration."""
    context = preflight(scratch)
    original = load_csv(BASELINE / "development_features.csv", ["date"])
    original = original.set_index("date")
    origins = pd.DatetimeIndex(original.index)
    cutoff = str(origins.max().date())
    with read_immutable(context["db_copy"],
                        context["database"]["sha256"]) as connection:
        monthly_all = pd.read_sql_query(
            SQL["monthly"], connection, params=[AS_OF],
            parse_dates=["month_end", "filing_date"],
        )
        monthly = monthly_all.loc[monthly_all.month_end <= origins.max()]
        quarterly = pd.read_sql_query(
            SQL["quarterly"], connection, params=[cutoff],
            parse_dates=["period_end", "filing_date"],
        )
        prices = pd.read_sql_query(
            SQL["prices"], connection, params=[cutoff],
            parse_dates=["date"],
        )
        actions = pd.read_sql_query(
            SQL["splits"], connection, params=[cutoff],
            parse_dates=["split_date"],
        )
        ppi = pd.read_sql_query(
            SQL["ppi"], connection, params=[cutoff],
        )
    pattern = monthly_all.loc[
        monthly_all.month_end.between("2016-01-01", AS_OF),
        ["month_end", "net_premiums_written", "net_premiums_earned",
         "npw_growth_yoy"],
    ].copy()
    pattern["year"] = pattern.month_end.dt.year
    pattern["month"] = pattern.month_end.dt.month
    pattern = pattern.rename(columns={
        "net_premiums_written": "npw", "net_premiums_earned": "npe",
    })
    fiscal = fiscal_month_shift(pattern)
    fiscal["npw_yoy_sd_2016_2023"] = float(
        pattern.loc[pattern.year.between(2016, 2023),
                    "npw_growth_yoy"].std())
    fiscal["npw_yoy_sd_2024"] = float(
        pattern.loc[pattern.year == 2024, "npw_growth_yoy"].std())
    fiscal["decision"] = "trailing12 NPW growth in I8"
    save_json(OUTPUTS / "fiscal_month_audit.json", fiscal)
    save_json(OUTPUTS / "source_audit.json",
              audit_sources(monthly, quarterly, actions))
    close = prices.set_index("date").close.sort_index()
    price_at_origin = close.reindex(origins, method="ffill")
    split_series = actions.set_index("split_date").split_ratio
    additions, availability = insurance_features(
        monthly, split_series, origins, price_at_origin
    )
    ppi["period"] = pd.to_datetime(ppi.month_end).dt.to_period("M")
    ppi_level = ppi.set_index("period").value.astype(float).sort_index()
    ppi_change = ppi_level / ppi_level.shift(12) - 1.0
    available_ppi = ppi_change.reindex(origins.to_period("M") - 1)
    additions["iv_rate_gap"] = (
        additions.iv_npw_t12_yoy.to_numpy()
        - available_ppi.to_numpy(dtype=float)
    )
    availability = pd.concat([
        availability,
        pd.DataFrame({
            "origin": origins, "feature": "iv_rate_gap",
            "report_month": (origins.to_period("M") - 1).to_timestamp("M"),
            "available": origins,
            "fallback": False, "age_months": 1,
            "publication_basis": "one calendar-month PPI assumption",
        }),
    ], ignore_index=True)
    features = original.join(additions)
    if np.isinf(features.select_dtypes(include="number").to_numpy()).any():
        raise ValueError("Infinite research features")
    save_csv(OUTPUTS / "features.csv", features.reset_index())
    save_csv(OUTPUTS / "availability_ledger.csv", availability)
    save_json(OUTPUTS / "preflight.json", {
        **context, "db_copy": str(context["db_copy"]),
        "max_feature_origin": origins.max(),
        "max_available_source_date": monthly.filing_date.max(),
        "source_query_cutoff": cutoff,
        "holdout_metrics": 0,
    })
    (OUTPUTS / "registry_snapshot.yaml").write_bytes(
        (ROOT / "research/registry.yaml").read_bytes()
    )
    inputs = [
        BASELINE / name for name in (
            "baseline_lock.json", "runtime_lock.json", "output_manifest.json",
            "development_features.csv", "development_targets.csv",
            "predictions.csv", "fold_ledger.csv", "partitions.json",
            "access_ledger.json",
        )
    ] + [
        V201 / name for name in (
            "registered.json", "metrics.json", "comparison.json",
            "multiplicity.json",
        )
    ] + [
        V201 / f"predictions_{name}_h{horizon}.csv"
        for name in ("P1", "P2", "P3", "M1", "M2", "M3")
        for horizon in (6, 12)
    ] + [
        V205 / "multiplicity.json", STUDY / "run.py",
        ROOT / "src/pgr_vds/research_lib/insurance_valuation.py",
        ROOT / "tests/research/test_v202_insurance_math.py",
    ] + [
        OUTPUTS / name for name in (
            "features.csv", "availability_ledger.csv",
            "fiscal_month_audit.json", "source_audit.json",
            "registry_snapshot.yaml",
        )
    ]
    save_json(OUTPUTS / "registered.json", {
        **procedure(),
        "as_of": AS_OF, "development_boundary": str(BOUNDARY.date()),
        "base_master_commit": git("merge-base", "HEAD", "origin/master"),
        "baseline_lock_sha256": LOCK_SHA,
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in inputs
        },
        "candidate_rule": "eight one-factor blocks, no combinations "
                          "or post-result search; at most one finalist",
    })
    save_json(OUTPUTS / "candidate_ledger.json", {
        "status": "registered_before_fitting", "slots": list(range(7, 15)),
        "candidates": BLUEPRINTS, "fits": 0,
    })
    print("Prepared eight frozen candidates; commit registered.json before fitting")


def require_registration() -> dict[str, Any]:
    """Reject a changed or uncommitted procedure and all changed inputs."""
    path = OUTPUTS / "registered.json"
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise ValueError("Commit exact preregistration before fitting")
    registered = json.loads(committed)
    for key, value in procedure().items():
        if registered.get(key) != value:
            raise ValueError(f"Preregistered procedure changed: {key}")
    for relative, digest in registered["input_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"Consumed input changed: {relative}")
    return registered


def verify_panel(candidate: pd.DataFrame, control: pd.DataFrame) -> None:
    """Require exact fixed labels, target ends, origins and naive forecasts."""
    keys = ["date", "benchmark", "horizon"]
    columns = keys + ["fold", "target_end", "available", "y_true",
                      "naive", "base_prediction"]
    left = candidate.sort_values(keys)[columns].reset_index(drop=True)
    right = control.sort_values(keys)[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False,
                                  check_exact=True)
    if (left.target_end >= BOUNDARY).any() or (
        left.available >= BOUNDARY
    ).any():
        raise ValueError("Quarantine target entered development comparison")
    for row in left.itertuples():
        if row.target_end != label_end(row.date, int(row.horizon)):
            raise ValueError("Incorrect target end")


def verify_folds(candidate: pd.DataFrame, control: pd.DataFrame) -> None:
    """Require the same strict v200 outer and inner support."""
    columns = ["kind", "fold", "benchmark", "horizon", "train_start",
               "train_end", "test_start", "test_end", "n_train",
               "n_test", "status"]
    keys = ["horizon", "benchmark", "fold", "kind", "test_start"]
    left = candidate.reindex(columns=columns).sort_values(keys).reset_index(
        drop=True)
    right = control.reindex(columns=columns).sort_values(keys).reset_index(
        drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    scored = candidate.loc[candidate.status == "scorable"]
    for row in scored.itertuples():
        if row.kind == "outer":
            train_end = pd.Timestamp(row.train_end)
            test_start = pd.Timestamp(row.test_start)
            if train_end >= test_start:
                raise ValueError("Outer training is not earlier than test")


def valuation_eras(features: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Six fixed descriptive date-block IC tests in a separate Holm family."""
    from statsmodels.stats.multitest import multipletests

    outcome = targets.loc[
        (targets.horizon == 6) & (targets.benchmark == "VOO"),
        ["date", "pgr_return"],
    ].drop_duplicates("date")
    joined = features.reset_index().merge(outcome, on="date", how="inner")
    rows = []
    for first, last in (("2004-01-01", "2014-12-31"),
                        ("2015-01-01", "2023-09-28")):
        era = joined.loc[joined.date.between(first, last)]
        for name, direction in (("iv_pb", -1), ("iv_pe", -1),
                                ("iv_pb_pe_spread", -1)):
            valid = era.loc[
                np.isfinite(era[name]) & np.isfinite(era.pgr_return),
                ["date", "pgr_return", name],
            ].copy()
            valid = valid.rename(columns={"pgr_return": "y_true",
                                          name: "y_hat"})
            valid["y_hat"] *= direction
            valid["benchmark"] = "PGR"
            valid["naive"] = 0.0
            valid["base_prediction"] = 0.0
            if len(valid) < 12:
                rows.append({"era": [first, last], "signal": name,
                             "n_dates": len(valid), "ic": None,
                             "raw_p": 1.0})
                continue
            metric = panel_summary(valid, 6, 2000, SEED)
            rows.append({
                "era": [first, last], "signal": name,
                "n_dates": metric["n_dates"], "ic": metric["panel_ic"],
                "raw_p": metric["panel_ic_p"],
                "block_length": 6,
                "source": "matched repaired PGR six-month DRIP targets",
            })
    adjusted = multipletests([row["raw_p"] for row in rows],
                            method="holm")[1]
    for row, pvalue in zip(rows, adjusted):
        row["holm6_p"] = float(pvalue)
    return {"tests": rows, "family_size": 6, "promotable": False,
            "note": "Descriptive; cannot rescue a failed forecast primary"}


def causal_v201_comparator(horizon: int, old_folds: pd.DataFrame,
                           all_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame,
                                                                  pd.DataFrame]:
    """Select from frozen v201 OOS recipes within downstream inner history."""
    names = ("P1", "P2", "P3", "M1", "M2", "M3")
    frames = []
    for name in names:
        frame = load_csv(
            V201 / f"predictions_{name}_h{horizon}.csv",
            ["date", "available", "target_end"],
        )
        frame["recipe"] = name
        frames.append(frame)
    catalog = pd.concat(frames, ignore_index=True)
    outer = old_folds.loc[
        (old_folds.horizon == horizon) & (old_folds.kind == "outer")
        & (old_folds.status == "scorable")
    ]
    choices = []
    for fold, group in outer.groupby("fold"):
        first = pd.Timestamp(group.test_start.min())
        train_start = pd.Timestamp(group.train_start.min())
        train_end = pd.Timestamp(group.train_end.max())
        training_dates = all_dates[
            (all_dates >= train_start) & (all_dates <= train_end)
        ]
        choice = select_causal_recipe(
            catalog, training_dates, first, horizon
        )
        choices.append({"horizon": horizon, "fold": fold,
                        "test_start": first, "train_start": train_start,
                        "train_end": train_end, "selected": choice,
                        "support": len(training_dates)})
    selection = pd.DataFrame(choices)
    keep = selection.dropna(subset=["selected"])
    if keep.empty:
        return pd.DataFrame(), selection
    chosen = catalog.merge(
        keep[["fold", "selected"]], left_on=["fold", "recipe"],
        right_on=["fold", "selected"], how="inner"
    )
    return chosen, selection


def execute(scratch: Path) -> None:
    """Run the single registered development comparison and close it out."""
    context = preflight(scratch)
    registered = require_registration()
    code_commit = git("rev-parse", "HEAD")
    tracked_dirty = bool(git("status", "--porcelain",
                             "--untracked-files=no"))
    features = load_csv(OUTPUTS / "features.csv", ["date"]).set_index("date")
    targets = load_csv(BASELINE / "development_targets.csv",
                       ["date", "target_end", "available"])
    targets = targets.loc[targets.benchmark.isin(BENCHMARKS)]
    control = load_csv(BASELINE / "predictions.csv",
                       ["date", "target_end", "available"])
    old_folds = pd.read_csv(BASELINE / "fold_ledger.csv")
    all_predictions = []
    all_folds = []
    metrics = {}
    comparisons = {}
    attempts = []
    for name in registered["candidate_order"]:
        for horizon in (6, 12):
            print(f"START {name} h{horizon}", flush=True)
            selected = targets.loc[targets.horizon == horizon]
            components, entries = strict_predictions(
                features, selected, horizon,
                registered["columns"][name],
            )
            folds = pd.DataFrame(entries)
            verify_folds(folds, old_folds.loc[old_folds.horizon == horizon])
            panel = ensemble_stream(
                components,
                selected.rename(columns={"available": "label_available"}),
            )
            panel = prequential_intervals(panel)
            incumbent = control.loc[control.horizon == horizon]
            verify_panel(panel, incumbent)
            key = f"{name}_h{horizon}"
            summary = panel_summary(panel, horizon, 2000, SEED)
            summary.update(calibration_summary(panel))
            summary["naive_diagnostic_p"] = summary.pop("primary_p")
            metrics[key] = summary
            comparisons[key] = paired_delta_bootstrap(
                incumbent, panel, horizon, 2000, SEED
            )
            panel["candidate"] = name
            folds["candidate"] = name
            all_predictions.append(panel)
            all_folds.append(folds)
            save_csv(OUTPUTS / f"predictions_{key}.csv", panel)
            attempts.append({"candidate": name, "horizon": horizon,
                             "status": "completed", "rows": len(panel)})
            save_json(OUTPUTS / "progress.json", {
                "attempts": attempts, "metrics": metrics,
                "comparisons": comparisons,
            })
            print(f"DONE {key}: {len(panel)} rows", flush=True)
    family = np.ones(38)
    prior_v201 = json.loads((V201 / "multiplicity.json").read_text())[
        "raw_primary_p_by_slot"]
    family[:6] = prior_v201[:6]
    prior_v205 = json.loads((V205 / "multiplicity.json").read_text())["raw_p"]
    for slot, name in zip(range(28, 34),
                          ("D1", "D2", "D3", "B1", "B2", "B3")):
        family[slot] = prior_v205[name]
    for slot, name in enumerate(registered["candidate_order"], start=6):
        family[slot] = comparisons[f"{name}_h6"]["primary_p"]
    adjusted = holm38(family)
    controls = {}
    for horizon in (6, 12):
        incumbent = control.loc[control.horizon == horizon]
        controls[f"h{horizon}"] = panel_summary(
            incumbent, horizon, 2000, SEED
        )
        controls[f"h{horizon}"].update(calibration_summary(incumbent))
    dispositions = {}
    for slot, name in enumerate(registered["candidate_order"], start=6):
        comparisons[f"{name}_h6"]["holm38_p"] = adjusted[slot]
        dispositions[name] = candidate_disposition(
            metrics[f"{name}_h6"], controls["h6"],
            comparisons[f"{name}_h6"],
        )
    save_json(OUTPUTS / "descriptive_pb_pe.json",
              valuation_eras(features, targets))
    upstream = {}
    for horizon in (6, 12):
        chosen, ledger = causal_v201_comparator(
            horizon, old_folds, pd.DatetimeIndex(features.index)
        )
        save_csv(OUTPUTS / f"v201_selection_h{horizon}.csv", ledger)
        if chosen.empty:
            upstream[f"h{horizon}"] = {
                "status": "unsupported", "n_rows": 0,
            }
            continue
        save_csv(OUTPUTS / f"v201_causal_h{horizon}.csv", chosen)
        upstream[f"h{horizon}"] = {
            "status": "partial matched support",
            "n_rows": len(chosen), "n_dates": chosen.date.nunique(),
            "selected_counts": ledger.selected.value_counts().to_dict(),
        }
    save_json(OUTPUTS / "v201_comparison.json", {
        "fold_causal_reconstruction": upstream,
        "frozen_attempt2": {
            "metrics_sha256": sha256_file(V201 / "metrics.json"),
            "comparison_sha256": sha256_file(V201 / "comparison.json"),
            "status": "six frozen candidates, no global winner",
        },
        "globally_selected_recipe": "excluded from honest inference",
    })
    save_csv(OUTPUTS / "predictions.csv", pd.concat(all_predictions))
    save_csv(OUTPUTS / "fold_ledger.csv", pd.concat(all_folds))
    save_json(OUTPUTS / "metrics.json", metrics)
    save_json(OUTPUTS / "control_metrics.json", controls)
    save_json(OUTPUTS / "comparison.json", comparisons)
    save_json(OUTPUTS / "multiplicity.json", {
        "raw_primary_p_by_slot": family.tolist(),
        "holm_adjusted_p_by_slot": adjusted,
        "family_size": 38, "v202_slots": list(range(7, 15)),
        "v201_slots": list(range(1, 7)),
        "v205_slots": list(range(29, 35)),
        "remaining_slots": "p=1; v203/v204 cancelled; v206 fills none",
    })
    finalist = nominate_finalist(registered["candidate_order"],
                                 dispositions)
    save_json(OUTPUTS / "closeout.json", {
        "candidate_results": dispositions,
        "v207_finalist": finalist, "promotion": False,
        "active_policy_proposal": False,
        "primary": "paired h6 only; first-Holm spread is descriptive",
        "upstream_v201": "fold-causal partial-support reconstruction",
    })
    save_json(OUTPUTS / "candidate_ledger.json", {
        "slots": list(range(7, 15)), "candidates": BLUEPRINTS,
        "attempts": attempts, "finalist": finalist,
    })
    save_json(OUTPUTS / "access_ledger.json", {
        "development_boundary": str(BOUNDARY.date()),
        "quarantine_target_rows_loaded": 0, "quarantine_metrics": 0,
        "feature_max_origin": features.index.max(),
        "baseline_partition_sha256": sha256_file(BASELINE / "partitions.json"),
    })
    after = sha256_file(ROOT / "data/pgr_financials.db")
    if after != context["tracked_db_sha256"]:
        raise ValueError("Tracked DB changed during v202")
    save_json(OUTPUTS / "run_record.json", {
        "code_commit": code_commit,
        "tracked_dirty_at_start": tracked_dirty,
        "input_git": context["baseline_code_commit"],
        "input_db": context["database"],
        "tracked_db_sha256_before": context["tracked_db_sha256"],
        "tracked_db_sha256_after": after,
        "baseline_lock_sha256": LOCK_SHA,
        "registered_sha256": sha256_file(OUTPUTS / "registered.json"),
        "seed": SEED, "providers": 0, "emails": 0,
        "live_changes": 0,
    })


def main() -> None:
    """Prepare once, commit registration, then execute the frozen run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.scratch.resolve().is_relative_to(ROOT):
        raise ValueError("DB copies and scratch must be outside the checkout")
    if args.prepare:
        prepare(args.scratch)
    else:
        execute(args.scratch)


if __name__ == "__main__":
    main()
