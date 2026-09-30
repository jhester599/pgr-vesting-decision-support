"""Offline v201 attempt 2: frozen six-block development comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import config
from pgr_vds.research_lib.adapters import (
    RIDGE_GRID,
    SHRINKAGE_GRID,
    ensemble_stream,
    prequential_intervals,
)
from pgr_vds.research_lib.baseline import (
    calibration_summary,
    strict_predictions,
)
from pgr_vds.research_lib.metrics import holm38, panel_summary
from pgr_vds.research_lib.price_macro import (
    available_macro,
    macro_features,
    paired_comparison,
    price_features,
    relative_trends,
)
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
BASELINE = ROOT / "research/studies/v200_clean_baseline/outputs"
LOCK_SHA = "c558ecb5215cf960b2685d687b83275d43e22ff3c976221e03dc734a1f02ffb0"
SEED = 20260926
OUTPUTS = STUDY / "outputs/attempt2"
BOUNDARY = pd.Timestamp("2023-09-29")
FEATURES = {
    key: list(config.MODEL_FEATURE_OVERRIDES[key]) for key in ("ridge", "gbt")
}
BENCHMARKS = list(config.PRIMARY_FORECAST_UNIVERSE)
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

    The historical registry procedure is the sole verification amendment.
    All approved data/code pins and the accepted lock remain byte-identical.
    """
    output.mkdir(parents=True, exist_ok=True)
    try:
        lock_path = BASELINE / "baseline_lock.json"
        if sha256_file(lock_path) != LOCK_SHA:
            raise ValueError("Accepted v200 baseline lock bytes changed")
        checked = verify_registry_growth(lock_path, scratch)
        lock = checked["lock"]
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
        return {
            **checked,
            "database": scratch / "pinned.db",
            "lock_sha256": LOCK_SHA,
            "tracked_db_sha256": sha256_file(ROOT / "data/pgr_financials.db"),
        }
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


def procedure() -> dict:
    """Freeze procedures rather than select anything from observed results."""
    return {
        "attempt": 2,
        "candidate_order": list(BLUEPRINTS),
        "candidate_slots": list(range(1, 7)),
        "blueprints": BLUEPRINTS,
        "columns": {
            candidate: {
                model: [c for c in cols if c not in blueprint["remove"]]
                + blueprint["add"]
                for model, cols in FEATURES.items()
            }
            for candidate, blueprint in BLUEPRINTS.items()
        },
        "ridge_grid": RIDGE_GRID.tolist(),
        "shrinkage_grid": SHRINKAGE_GRID.tolist(),
        "gbt": {
            "max_depth": 2,
            "n_estimators": 50,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        },
        "publication_lags": LAGS,
        "macro_limits": "latest vintage; calendar availability proxy, not "
        "observed release timestamps; exact period, one lag, no filling",
        "outer": {"6": [60, 6, 12], "12": [120, 6, 24]},
        "inner": {
            "folds": 3,
            "test_size": 6,
            "gap": "2*h",
            "max_train_size": "outer window",
            "minimum": [24, 60],
        },
        "ensemble": "unchanged v200 mature-past MAE weights and 10-value "
        "shrinkage; no new upstream recipe or tuning",
        "calibration": "unchanged v200 mature-past ACI and Platt; missing "
        "warmup never evaluated or backfilled",
        "primary": "paired equal-date squared-error gain at h6 vs v200",
        "secondary": "h12 diagnostics, never another primary success path",
        "inference": {
            "seed": SEED,
            "replicates": 2000,
            "date_block_length": "h",
            "family_size": 38,
        },
        "thresholds": {
            "delta_r2": 0.010,
            "holm_p": 0.05,
            "max_ic_loss": 0.01,
            "direction_calibration": "no deterioration vs matched control",
        },
        "safeguards": "hit rate and directional skill >= control; Brier, "
        "log loss and ECE <= control; distance of coverage from .8 <= "
        "control; missing safeguards fail; float epsilon 1e-12",
        "nomination": "highest h6 delta R2 among passers, ties retain "
        "candidate_order; at most one; h12 never rescues failed primary",
        "v207": "quarantine sealed; at most one research finalist",
        "costs": "forecasts only; no trading policy or cost-adjusted P&L",
        "sql": SQL,
    }


def verify_matched_support(
    candidate: pd.DataFrame, control: pd.DataFrame
) -> None:
    """Require exact keys, labels, endpoints and realised mature controls."""
    keys = ["date", "benchmark", "horizon"]
    cols = keys + [
        "fold",
        "target_end",
        "available",
        "y_true",
        "naive",
        "base_prediction",
    ]
    left = candidate.sort_values(keys)[cols].reset_index(drop=True)
    right = control.sort_values(keys)[cols].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "Candidate differs from matched v200 support"
        ) from error


def verify_label_ends(targets: pd.DataFrame, ledger: pd.DataFrame) -> None:
    """Check target definitions and label ends before every test origin."""
    for row in targets.to_dict("records"):
        if (
            row["target_end"] != label_end(row["date"], int(row["horizon"]))
            or row["available"] < row["target_end"]
            or row["available"] >= BOUNDARY
        ):
            raise ValueError("Target label end/availability contract failed")
    for row in ledger.loc[ledger.status == "scorable"].to_dict("records"):
        train = targets.loc[
            (targets.horizon == row["horizon"])
            & (targets.benchmark == row["benchmark"])
            & (targets.date >= pd.Timestamp(row["train_start"]))
            & (targets.date <= pd.Timestamp(row["train_end"]))
            & (targets.available <= pd.Timestamp(row["test_start"]))
        ]
        tests = pd.date_range(row["test_start"], row["test_end"], freq="BME")
        for origin in tests:
            if (train.target_end > origin).any() or (
                train.date >= origin
            ).any():
                raise ValueError("Training label end exceeds a test origin")


def verify_fold_support(
    candidate: pd.DataFrame, control: pd.DataFrame
) -> None:
    """Only learned transforms and selected penalties may change by block."""
    cols = [
        "kind",
        "fold",
        "benchmark",
        "horizon",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "n_train",
        "n_test",
        "status",
    ]
    keys = ["horizon", "benchmark", "fold", "kind", "test_start"]
    left = (
        candidate.reindex(columns=cols)
        .sort_values(keys)
        .reset_index(drop=True)
    )
    right = (
        control.reindex(columns=cols).sort_values(keys).reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as error:
        raise ValueError(
            "Candidate folds differ from fixed v200 folds"
        ) from error


def prepare(scratch: Path) -> None:
    """Freeze causal features, source pins and catalog before fitting."""
    context = preflight(OUTPUTS, scratch)
    original = load_csv(BASELINE / "development_features.csv", ["date"])
    original = original.set_index("date")
    origins = original.index
    cutoff = str(origins.max().date())
    with read_immutable(
        context["database"],
        context["lock"]["input_db"]["sha256"],
    ) as connection:
        prices = pd.read_sql_query(
            SQL["prices"],
            connection,
            params=[cutoff],
            parse_dates=["date"],
        )
        actions = pd.read_sql_query(
            SQL["splits"],
            connection,
            params=[cutoff],
            parse_dates=["split_date"],
        )
        macro = pd.read_sql_query(
            SQL["macro"],
            connection,
            params=[cutoff],
        )
    closes = {
        ticker: prices.loc[prices.ticker == ticker].set_index("date").close
        for ticker in ("PGR", "VOO")
    }
    splits = {
        ticker: actions.loc[actions.ticker == ticker]
        .set_index(
            "split_date",
        )
        .split_ratio
        for ticker in closes
    }
    technical = price_features(closes["PGR"], splits["PGR"], origins)
    technical = technical.join(
        relative_trends(
            closes["PGR"],
            closes["VOO"],
            splits,
            origins,
        )
    )
    values, availability = available_macro(macro, origins, LAGS)
    features = original.join(technical).join(macro_features(values))
    if np.isinf(features.select_dtypes(include="number").to_numpy()).any():
        raise ValueError("Infinite feature values; do not silently clean them")
    save_csv(OUTPUTS / "features.csv", features.reset_index())
    save_csv(OUTPUTS / "availability_ledger.csv", availability)
    examples = []
    for date, archived, corrected in [
        ("2023-02-28", "mom_12m", "pm_mom12"),
        ("2020-04-30", "vix", "pm_vix"),
        ("2023-02-28", "rate_adequacy_gap_yoy", "pm_rate_gap"),
    ]:
        examples.append(
            {
                "date": date,
                "archived_column": archived,
                "corrected_column": corrected,
                "archived_v200": original.loc[date, archived],
                "corrected": features.loc[date, corrected],
                "comparison_role": "matched recipe example; archived v200 "
                "already repaired the original window/publication defect",
            }
        )
    save_json(OUTPUTS / "feature_examples.json", examples)
    save_json(
        OUTPUTS / "archived_finding_examples.json",
        {
            "source": "docs/reviews/REPO_REVIEW_2026-09-25.md",
            "role": "documentary archived findings, not rerun forecasts",
            "mom12": {
                "finding": "F01",
                "origin": "2006-05-31",
                "archived_report_value_rounded": -0.80,
                "legacy_rule": "252 raw weekly rows, contaminated by split",
                "repaired_calendar_value": features.loc[
                    "2006-05-31", "pm_mom12"
                ],
            },
            "april2020_vix": {
                "finding": "F06",
                "period": "2020-04",
                "legacy_stored_value": 53.54,
                "legacy_value_role": "March close stored under April; "
                "then lagged again",
                "repaired_raw_april_value": float(
                    macro.loc[
                        (macro.series_id == "VIXCLS")
                        & (
                            pd.to_datetime(macro.month_end).dt.to_period("M")
                            == pd.Period("2020-04")
                        ),
                        "value",
                    ].iloc[0]
                ),
                "single_lag_feature_at_april_origin": features.loc[
                    "2020-04-30", "pm_vix"
                ],
            },
            "rate_gap": {
                "finding": "F07",
                "archived_report_origin": "2026-04",
                "archived_report_value": None,
                "archived_reason": "Insurance PPI stale after February 2026; "
                "live gap missing. Documentary report only, "
                "no quarantine query.",
                "development_origin": "2023-02-28",
                "repaired_value": features.loc["2023-02-28", "pm_rate_gap"],
            },
        },
    )
    save_json(
        OUTPUTS / "preflight.json",
        {
            **context,
            "status": "passed",
            "fit_count": 0,
            "max_feature_origin": origins.max(),
            "query_cutoff": cutoff,
            "source_windows": {
                "prices": [prices.date.min(), prices.date.max()],
                "macro": [macro.month_end.min(), macro.month_end.max()],
            },
            "accepted_exception": "VWO March 2026 provider gap remains "
            "accepted; all 12 affected v200 labels are quarantined",
        },
    )
    save_csv(OUTPUTS / "raw_macro_observations.csv", macro)
    registry = OUTPUTS / "registry_snapshot.yaml"
    registry.write_bytes((ROOT / "research/registry.yaml").read_bytes())
    input_paths = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in [
            *[
                BASELINE / name
                for name in (
                    "baseline_lock.json",
                    "runtime_lock.json",
                    "output_manifest.json",
                    "development_features.csv",
                    "development_targets.csv",
                    "predictions.csv",
                    "fold_ledger.csv",
                    "partitions.json",
                    "access_ledger.json",
                    "dependency_environment.json",
                )
            ],
            OUTPUTS / "features.csv",
            OUTPUTS / "availability_ledger.csv",
            OUTPUTS / "raw_macro_observations.csv",
            registry,
            STUDY / "outputs/attempt1/outputs/candidate_ledger.json",
            ROOT
            / "research/studies/v205_dividend_bvps/outputs/multiplicity.json",
            STUDY / "run.py",
            ROOT / "src/pgr_vds/research_lib/price_macro.py",
            ROOT / "src/pgr_vds/research_lib/snapshot.py",
            ROOT / "src/pgr_vds/research_lib/price_macro_disposition.py",
        ]
    ]
    references = json.loads(
        (STUDY / "outputs/attempt1/provenance.json").read_text(
            encoding="utf-8"
        )
    )["reference_file_sha256"]
    input_paths.extend(
        path for path in references if path != "research/registry.yaml"
    )
    input_paths = list(dict.fromkeys(input_paths))
    save_json(
        OUTPUTS / "registered.json",
        {
            **procedure(),
            "input_sha256": {
                path: sha256_file(ROOT / path) for path in input_paths
            },
            "base_master_commit": "ad5d98dde4163721fde63723fbcd97dc912d4ae7",
            "baseline_lock_sha256": LOCK_SHA,
            "as_of": "2026-09-26",
            "development_boundary": str(BOUNDARY.date()),
            "archive_sha256": sha256_file(
                STUDY / "outputs/attempt1/archive_manifest.json",
            ),
            "amendment": "A2: historical-registry verification, completed "
            "execution runner; unchanged six features/grids/lags; "
            "zero prior fits",
        },
    )
    print("Prepared all six blueprints; commit registered.json before fitting")


def execute(scratch: Path, destination: Path) -> None:
    """Execute once; fixed h6 primary and h12 secondary, no holdout read."""
    context = preflight(destination, scratch)
    registered = require_register(OUTPUTS)
    if any(registered[key] != value for key, value in procedure().items()):
        raise ValueError("Committed procedure differs from current code")
    started_commit = git("rev-parse", "HEAD")
    started_dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    features = load_csv(OUTPUTS / "features.csv", ["date"]).set_index("date")
    targets = load_csv(
        BASELINE / "development_targets.csv",
        ["date", "available", "target_end"],
    )
    targets = targets.loc[targets.benchmark.isin(BENCHMARKS)]
    control = load_csv(
        BASELINE / "predictions.csv",
        ["date", "available", "target_end"],
    )
    old_folds = pd.read_csv(BASELINE / "fold_ledger.csv")
    all_predictions, all_folds, summaries, comparisons = [], [], {}, {}
    for candidate in registered["candidate_order"]:
        for horizon in (6, 12):
            print(
                f"START {candidate} h{horizon}; development only", flush=True
            )
            selected = targets.loc[targets.horizon == horizon].copy()
            components, entries = strict_predictions(
                features,
                selected,
                horizon,
                registered["columns"][candidate],
            )
            ledger = pd.DataFrame(entries)
            verify_label_ends(selected, ledger)
            verify_fold_support(
                ledger, old_folds.loc[old_folds.horizon == horizon]
            )
            panel = ensemble_stream(
                components,
                selected.rename(columns={"available": "label_available"}),
            )
            panel = prequential_intervals(panel)
            incumbent = control.loc[control.horizon == horizon].copy()
            verify_matched_support(panel, incumbent)
            key = f"{candidate}_h{horizon}"
            summary = panel_summary(panel, horizon, 2000, SEED)
            summary.update(calibration_summary(panel))
            # The naive-based p is a diagnostic; the sole registered primary
            # test is the paired candidate-minus-v200 improvement below.
            summary["naive_squared_error_diagnostic_p"] = summary.pop(
                "primary_p"
            )
            summaries[key] = summary
            comparisons[key] = paired_comparison(incumbent, panel, horizon)
            panel["candidate"] = candidate
            ledger["candidate"] = candidate
            all_predictions.append(panel)
            all_folds.append(ledger)
            save_csv(destination / f"predictions_{key}.csv", panel)
            save_json(
                destination / "progress.json",
                {
                    "completed": list(summaries),
                    "metrics": summaries,
                    "comparisons": comparisons,
                },
            )
            print(f"DONE {key}, {len(panel)} matched rows", flush=True)
    family = np.ones(38)
    for slot, candidate in enumerate(registered["candidate_order"]):
        family[slot] = comparisons[f"{candidate}_h6"]["primary_p"]
    prior = json.loads(
        (
            ROOT
            / ("research/studies/v205_dividend_bvps/outputs/multiplicity.json")
        ).read_text(encoding="utf-8")
    )["raw_p"]
    for slot, name in zip(range(28, 34), ("D1", "D2", "D3", "B1", "B2", "B3")):
        family[slot] = prior[name]
    adjusted = holm38(family)
    control_metrics = {}
    for horizon in (6, 12):
        incumbent = control.loc[control.horizon == horizon]
        control_metrics[f"h{horizon}"] = panel_summary(
            incumbent,
            horizon,
            2000,
            SEED,
        )
        control_metrics[f"h{horizon}"].update(calibration_summary(incumbent))
    dispositions = {}
    for slot, candidate in enumerate(registered["candidate_order"]):
        comparisons[f"{candidate}_h6"]["holm38_p"] = adjusted[slot]
        dispositions[candidate] = candidate_disposition(
            summaries[f"{candidate}_h6"],
            control_metrics["h6"],
            comparisons[f"{candidate}_h6"],
        )
    save_json(
        destination / "closeout.json",
        {
            "candidate_results": dispositions,
            "v207_finalist": nominate_finalist(
                registered["candidate_order"],
                dispositions,
            ),
            "promotion": False,
            "active_policy_proposal": False,
            "primary": "h6 only; all candidates and safeguards reported",
        },
    )
    save_json(destination / "control_metrics.json", control_metrics)
    save_csv(destination / "predictions.csv", pd.concat(all_predictions))
    save_csv(destination / "fold_ledger.csv", pd.concat(all_folds))
    save_json(destination / "metrics.json", summaries)
    save_json(destination / "comparison.json", comparisons)
    save_json(
        destination / "multiplicity.json",
        {
            "raw_primary_p_by_slot": family.tolist(),
            "holm_adjusted_p_by_slot": adjusted,
            "family_size": 38,
            "alpha": 0.05,
            "v201_slots": list(range(1, 7)),
            "v205_slots": list(range(29, 35)),
            "pending_slots": "p=1",
            "h12": "secondary; excluded from the primary family",
        },
    )
    save_json(
        destination / "access_ledger.json",
        {
            "attempt": 2,
            "quarantine_target_rows_loaded": 0,
            "quarantine_metrics": 0,
            "development_boundary": str(BOUNDARY.date()),
            "features_max_origin": features.index.max(),
            "quarantine_access": "v207 only; labels consumed from pinned "
            "development CSV",
            "baseline_partition_sha256": sha256_file(
                BASELINE / "partitions.json"
            ),
        },
    )
    after = sha256_file(ROOT / "data/pgr_financials.db")
    if after != context["tracked_db_sha256"]:
        raise ValueError("Tracked DB changed")
    save_json(
        destination / "run_record.json",
        {
            "code_commit": started_commit,
            "tracked_dirty_at_start": started_dirty,
            "input_git": context["lock"]["baseline_code_commit"],
            "input_db": context["lock"]["input_db"],
            "tracked_db_sha256_before": context["tracked_db_sha256"],
            "tracked_db_sha256_after": after,
            "seed": SEED,
            "registered_sha256": sha256_file(OUTPUTS / "registered.json"),
            "baseline_lock_sha256": LOCK_SHA,
            "fits": "one procedure per candidate/horizon",
            "providers": 0,
            "emails": 0,
            "live_changes": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    if args.scratch.resolve().is_relative_to(ROOT):
        raise ValueError(
            "DB copies and scratch must stay outside the checkout"
        )
    if args.prepare:
        prepare(args.scratch)
    else:
        execute(args.scratch, args.output_dir)


if __name__ == "__main__":
    main()
