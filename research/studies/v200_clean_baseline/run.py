"""Offline, pinned v200 preflight and development-only baseline runner."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from pgr_vds.research_lib.adapters import (
    bounded_features,
    drip_return,
    ensemble_stream,
    feature_source_ledger,
    prequential_intervals,
    RIDGE_GRID,
    SHRINKAGE_GRID,
)
from pgr_vds.research_lib.baseline import (
    calibration_summary,
    strict_predictions,
)
from pgr_vds.research_lib.metrics import holm38, panel_summary
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
from pgr_vds.research_lib.temporal import label_end


ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parent
OUTPUTS = STUDY / "outputs"
AS_OF = pd.Timestamp("2026-09-26")
BOUNDARY = pd.Timestamp("2023-09-29")
BRIDGE_AS_OF = pd.Timestamp("2022-08-31")
SEED = 20260926
BENCHMARKS = list(config.PRIMARY_FORECAST_UNIVERSE)
CLASSIFIER_BENCHMARKS = list(config.INVESTABLE_CLASSIFIER_BASE_WEIGHTS)
ALL_BENCHMARKS = sorted(set(BENCHMARKS + CLASSIFIER_BENCHMARKS))
FEATURES = {
    key: list(config.MODEL_FEATURE_OVERRIDES[key]) for key in ("ridge", "gbt")
}
SOURCE_DOCS = [
    "AGENTS.md",
    "docs/reviews/REPO_REVIEW_2026-09-25.md",
    "docs/reviews/VERIFICATION_2026-09-26.md",
    "docs/reviews/VERIFICATION_2026-09-26_claude.md",
    "docs/reviews/2026-09-26_step13v_comparison.md",
    "docs/reviews/2026-09-28_R2_dividend_refresh_check.md",
    "docs/reviews/R3_validation_closeout.md",
    "docs/reviews/R3b_baseline_closeout.md",
    "docs/research/RERUN_PLAN_v200_codex.md",
    "docs/model-governance.md",
    "docs/history/superpowers/plans/2026-04-10-v37-v60-results-summary.md",
    (
        "docs/history/superpowers/plans/2026-04-10-v66-v7"
        "3-calibration-and-decision-layer.md"
    ),
    "src/research/v37_utils.py",
    "src/research/x1_targets.py",
    "src/research/x23_dividend_lane_package.py",
    "src/research/x18_dividend_policy_regime.py",
    "src/research/x21_dividend_target_scales.py",
    "src/research/x22_dividend_size_baselines.py",
    "docs/reviews/R3b_refreshed_replay_rows.csv",
    "docs/reviews/2026-09-25_step5_rebaseline_rows.csv",
]
SQL = {
    "targets": (
        "SELECT * FROM monthly_relative_returns ORDER BY "
        "date,benchmark,target_horizon"
    ),
    "prices": (
        "SELECT * FROM daily_prices WHERE date < ? AND "
        "ticker != 'CB' ORDER BY ticker,date"
    ),
    "dividends": (
        "SELECT * FROM daily_dividends WHERE ex_date < ? "
        "ORDER BY ticker,ex_date"
    ),
    "splits": (
        "SELECT * FROM split_history WHERE split_date < ?"
        " ORDER BY ticker,split_date"
    ),
    "monthly": (
        "SELECT * FROM pgr_edgar_monthly WHERE "
        "filing_date < ? ORDER BY month_end"
    ),
    "quarterly": (
        "SELECT * FROM pgr_fundamentals_quarterly WHERE "
        "filing_date < ? ORDER BY period_end"
    ),
    "macro": (
        "SELECT * FROM fred_macro_monthly WHERE month_end"
        " < ? ORDER BY series_id,month_end"
    ),
}


def json_value(value: object) -> object:
    """Make JSON strict and portable; missing metrics are null, never NaN."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
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
    """Preserve round-trip floating values and deterministic row order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def frame_hash(frame: pd.DataFrame) -> str:
    """Seal bytes for partitions without printing quarantined outcomes."""
    return hashlib.sha256(
        frame.to_csv(
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        ).encode()
    ).hexdigest()


def load_targets(conn: sqlite3.Connection) -> pd.DataFrame:
    """Classify all required target rows before any fit or outcome score."""
    frame = pd.read_sql_query(SQL["targets"], conn)
    frame = frame.loc[frame["benchmark"].isin(ALL_BENCHMARKS)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.rename(
        columns={"relative_return": "y_true", "target_horizon": "horizon"}
    )
    frame["target_end"] = [
        label_end(d, int(h)) for d, h in zip(frame["date"], frame["horizon"])
    ]
    frame["available"] = frame["target_end"]
    frame["partition"] = np.where(
        frame["target_end"] < BOUNDARY,
        "development",
        np.where(
            frame["date"] < BOUNDARY, "purged_boundary", "quarantine_union"
        ),
    )
    if (frame["available"] > AS_OF).any():
        raise ValueError(
            "Stored target claims maturity beyond the pinned as-of"
        )
    if frame.duplicated(["date", "benchmark", "horizon"]).any():
        raise ValueError("Duplicate monthly target")
    if not (frame["proxy_fill"] == 0).all():
        raise ValueError("Unresolved proxy target in required universe")
    return frame.sort_values(["horizon", "date", "benchmark"]).reset_index(
        drop=True
    )


def preflight(
    db: Path, scratch: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fail closed on the repaired input and development targets."""
    from src.database import db_client
    from src.processing import pgr_edgar_validation as checks
    from src.processing.price_integrity import (
        find_duplicate_week_bars,
        find_unexplained_price_jumps,
    )

    conn = read_immutable(db, INPUT_DB_SHA256)
    conn.row_factory = sqlite3.Row
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert find_unexplained_price_jumps(conn).empty
        assert find_duplicate_week_bars(conn).empty
        assert not conn.execute(
            "SELECT series_id, substr(month_end,1,7),COUNT(*) "
            "FROM fred_macro_monthly GROUP BY 1,2 HAVING COUNT(*)>1"
        ).fetchall()
        readiness = db_client.check_required_feed_readiness(conn, AS_OF.date())
        assert not readiness["stale_required_feeds"], readiness[
            "stale_required_feeds"
        ]
        monthly = db_client.get_pgr_edgar_monthly(conn)
        quarterly = db_client.get_pgr_fundamentals(conn)
        assert len(monthly) == 265 and len(quarterly) == 73
        assert not checks.missing_months(monthly)
        for check in (
            checks.income_identity_violations,
            checks.combined_ratio_violations,
            checks.equity_violations,
            checks.pif_jump_violations,
        ):
            assert check(monthly).empty
        assert checks.quarterly_net_income_violations(monthly, quarterly).empty
        for frame in (monthly, quarterly):
            filed = pd.to_datetime(frame["filing_date"])
            assert filed.notna().all() and (filed >= frame.index).all()
        targets = load_targets(conn)
        development = targets.loc[targets["partition"] == "development"]
        raw: dict = {}
        for table, key, time_column, value_column in [
            ("daily_prices", "prices", "date", "close"),
            ("daily_dividends", "dividends", "ex_date", "amount"),
            ("split_history", "splits", "split_date", "split_ratio"),
        ]:
            values = pd.read_sql_query(
                SQL[key], conn, params=(str(BOUNDARY.date()),)
            )
            values[time_column] = pd.to_datetime(values[time_column])
            raw[key] = {
                ticker: group.set_index(time_column)[value_column]
                for ticker, group in values.groupby("ticker")
            }
        manual: dict = {}
        max_error = 0.0
        for row in development.to_dict("records"):
            for ticker, stored in [
                ("PGR", row["pgr_return"]),
                (row["benchmark"], row["benchmark_return"]),
            ]:
                key = (ticker, row["date"], row["horizon"])
                if key not in manual:
                    _, manual[key] = drip_return(
                        raw["prices"][ticker],
                        raw["dividends"].get(ticker, pd.Series(dtype=float)),
                        raw["splits"].get(ticker, pd.Series(dtype=float)),
                        row["date"],
                        row["target_end"],
                    )
                max_error = max(max_error, abs(manual[key] - stored))
            assert (
                abs(
                    row["y_true"] - row["pgr_return"] + row["benchmark_return"]
                )
                < 1e-12
            )
        assert max_error < 1e-10, max_error
        migrations = [
            dict(row)
            for row in conn.execute("SELECT * FROM schema_migrations")
        ]
    finally:
        conn.close()
    live = bounded_features(db, AS_OF, scratch / "live_preflight")
    cols = sorted(set(FEATURES["ridge"] + FEATURES["gbt"]))
    assert set(cols).issubset(live.columns), (
        "Missing configured feature columns"
    )
    assert np.isfinite(live.iloc[-1][cols]).all(), (
        "Missing live inputs before imputation"
    )
    # Only raw features, no fitted model or quarantine forecast is computed.
    features = bounded_features(
        db, BOUNDARY - pd.Timedelta(days=1), scratch / "development"
    )
    earlier = bounded_features(
        db, pd.Timestamp("2015-04-30"), scratch / "causality"
    )
    common = earlier.index.intersection(features.index)
    np.testing.assert_allclose(
        earlier.loc[common, cols],
        features.loc[common, cols],
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    )
    return (
        features[cols],
        targets,
        {
            "status": "passed",
            "required_feed_readiness": readiness,
            "development_targets_checked": len(development),
            "independent_raw_split_drip_max_abs_error": max_error,
            "quarterly_rows": 73,
            "monthly_rows": 265,
            "live_feature_origin": str(live.index[-1].date()),
            "live_missing_features": [],
            "future_source_cutoff_invariance": True,
            "schema_migrations": migrations,
            "historical_vintage_limitation": (
                "Current-vintage FRED; calendar lags are "
                "conservative reconstructed release assumptions, "
                "not vintage proof."
            ),
        },
    )


def prepare(scratch: Path) -> None:
    """Freeze approved pins, procedure and partitions before fitting."""
    if (OUTPUTS / "predictions.csv").exists():
        raise ValueError("The executed register cannot be silently refrozen")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    tracked_before = sha256_file(ROOT / "data/pgr_financials.db")
    db = export_git_blob(
        INPUT_DB_COMMIT,
        "data/pgr_financials.db",
        scratch / "approved.db",
        INPUT_DB_SHA256,
    )
    try:
        features, targets, audit = preflight(db, scratch)
        conn = read_immutable(db, INPUT_DB_SHA256)
        try:
            source_ledger = feature_source_ledger(
                conn,
                BOUNDARY,
                features.index,
            )
        finally:
            conn.close()
        save_json(OUTPUTS / "preflight.json", audit)
    except Exception as error:
        save_json(
            OUTPUTS / "preflight.json",
            {"status": "blocked", "error": repr(error)},
        )
        save_json(
            OUTPUTS / "baseline_lock.json",
            {
                "status": "blocked",
                "baseline_code_commit": BASELINE_CODE_COMMIT,
                "input_db": {
                    "git_commit": INPUT_DB_COMMIT,
                    "sha256": INPUT_DB_SHA256,
                },
                "reason": repr(error),
            },
        )
        (STUDY / "README.md").write_text(
            "v200 is blocked: the approved repair failed required preflight. "
            "No model was fitted and no clean comparator exists.\n\n"
            f"Failure: `{error!r}`. See outputs/preflight.json.\n\n"
            "What changed: reproducible preflight evidence. What is left: "
            "resolve the blocker in its separately authorized"
            " PR, then rerun v200.\n",
            encoding="utf-8",
        )
        raise
    development = targets.loc[targets["partition"] == "development"].copy()
    save_csv(OUTPUTS / "source_availability_ledger.csv", source_ledger)
    quarantine = targets.loc[targets["partition"] == "quarantine_union"].copy()
    purged = targets.loc[targets["partition"] == "purged_boundary"].copy()
    save_csv(OUTPUTS / "development_targets.csv", development)
    save_csv(
        OUTPUTS / "development_features.csv",
        features.reset_index().rename(columns={"index": "date"}),
    )
    metadata = targets[
        [
            "date",
            "benchmark",
            "horizon",
            "target_end",
            "available",
            "partition",
        ]
    ]
    save_csv(OUTPUTS / "availability_ledger.csv", metadata)
    affected = metadata.loc[
        (metadata["benchmark"] == "VWO")
        & (
            (
                (metadata["horizon"] == 6)
                & metadata["date"].between("2025-09-01", "2026-02-28")
            )
            | (
                (metadata["horizon"] == 12)
                & metadata["date"].between("2025-03-01", "2025-08-31")
            )
        )
    ].copy()
    affected["exception"] = (
        "Accepted provider gap: March 2026 VWO ex-date absent"
    )
    assert len(affected) == 12
    save_csv(OUTPUTS / "vwo_accepted_gap.csv", affected)
    save_json(
        OUTPUTS / "partitions.json",
        {
            "boundary": BOUNDARY,
            "development_sha256": frame_hash(development),
            "quarantine_sha256": frame_hash(quarantine),
            "purged_sha256": frame_hash(purged),
            "development_rows": len(development),
            "quarantine_rows": len(quarantine),
            "purged_rows": len(purged),
            "target_end_boundary_rule": (
                "target_end and label_available strictly before 2023-09-29"
            ),
            "quarantine": {
                "h6": ["2024-03-28", "2026-02-27"],
                "h12": ["2023-09-29", "2025-08-29"],
            },
            "quarantine_prior_exposure": [
                "v75",
                "v129",
                "v132",
                "both step13V replays",
            ],
        },
    )
    save_json(
        OUTPUTS / "access_ledger.json",
        {
            "study": "v200",
            "holdout_metric_accesses": 0,
            "label_access_purposes": [
                "partition classification and SHA256 sealing only"
            ],
            "forecast_and_metric_scope": "development only",
            "prohibited_replay": (
                "September full-history step13V replay deferred to v207"
            ),
            "next_unseal": "v207 one frozen batch, D2",
        },
    )
    save_json(OUTPUTS / "runtime_lock.json", runtime_lock())
    save_json(
        OUTPUTS / "candidate_ledger.json",
        {
            "study": "v200_clean_baseline",
            "alternatives": 0,
            "forecast_blueprints": 1,
            "features": FEATURES,
            "ridge_penalties_ordered": RIDGE_GRID.tolist(),
            "shrinkage_grid_ordered": SHRINKAGE_GRID.tolist(),
            "shrinkage_min_mature_panel_rows": 36,
            "gbt": {
                "max_depth": 2,
                "n_estimators": 50,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "random_state": 42,
            },
            "outer": {
                "h6": {"train": 60, "gap": 12, "test": 6},
                "h12": {"train": 120, "gap": 24, "test": 6},
            },
            "inner": {
                "folds": 3,
                "test": 6,
                "gap": "2*h",
                "minimum_months": {"h6": 24, "h12": 60},
            },
            "primary_test": (
                "6M paired per-date squared-error improvement vs "
                "honest mature prevailing mean; baseline control "
                "only"
            ),
            "secondary_endpoint": "12M diagnostics; no primary success route",
            "inference": {
                "block_length": "h",
                "replicates": 2000,
                "seed": SEED,
            },
            "campaign_slots": [
                {"slot": slot, "status": "pending_or_unused", "p": 1.0}
                for slot in range(1, 39)
            ],
            "matched_controls": [
                "same-label PathB",
                "past12M cash dividend",
                "mature mean annual excess/currentBVPS",
                "mature prevailing BVPSgrowth",
                "mature mean absolute PGR6M DRIP",
            ],
            "path_b": {
                "C": 0.5,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 1000,
                "weights": config.INVESTABLE_CLASSIFIER_BASE_WEIGHTS,
                "threshold": -0.03,
                "min_calibration": 24,
                "temperature_grid_ordered": np.concatenate(
                    [
                        np.linspace(0.50, 0.95, 10),
                        np.linspace(1.0, 3.0, 41),
                    ]
                ).tolist(),
            },
            "calibration": (
                "mature per-benchmark Platt C1e10/scaler, "
                "minimum20; unscored warmup"
            ),
            "interval": (
                "incumbent80% ACI gamma.05, finite-sample "
                "quantile correction, minimum4 mature "
                "same-benchmark residuals; unscored warmup"
            ),
            "costs": (
                "No active policy/trade is tested; gross forecast"
                " targets, trading costs not applicable"
            ),
        },
    )
    sources = []
    reference_sources = []
    reference_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    for relative in SOURCE_DOCS:
        if Path(relative).suffix == ".csv":
            reference = OUTPUTS / "references" / Path(relative).name
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference.write_bytes(
                subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{reference_commit}:{relative}",
                    ]
                )
            )
            sources.append(str(reference.relative_to(ROOT)).replace("\\", "/"))
            reference_sources.append(
                {
                    "original_path": relative,
                    "git_commit": reference_commit,
                    "copy_path": sources[-1],
                    "sha256": sha256_file(reference),
                    "use": "provisional citation only; never fitted",
                }
            )
        else:
            sources.append(relative)
    sources += [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "docs/history/results").glob(
            "V*_RESULTS_SUMMARY.md"
        )
        if path.stem.split("_")[0]
        in {"V9", "V18", "V20", "V21", "V22", "V23", "V24", "V25"}
    ]
    sources += [
        "research/studies/v200_clean_baseline/run.py",
        "research/registry.yaml",
        "pyproject.toml",
        "config/model.py",
        "config/features.py",
    ]
    sources += [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "src/pgr_vds/research_lib").glob("*.py")
    ]
    for directory in (
        "src/models",
        "src/processing",
        "src/database",
        "src/reporting",
        "src/pgr_vds/decision",
        "src/pgr_vds/reporting",
        "config",
    ):
        sources += [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in (ROOT / directory).rglob("*.py")
        ]
    sources += [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in OUTPUTS.glob("*")
        if path.suffix in (".json", ".csv")
        and path.name != "baseline_lock.json"
    ]
    file_pins = []
    for relative in sorted(set(sources)):
        exact = relative.startswith(
            (
                "research/studies/v200_clean_baseline/",
                "src/pgr_vds/research_lib/",
            )
        ) or Path(relative).suffix not in {".py", ".md", ".yaml", ".toml"}
        file_pins.append(
            {
                "path": relative,
                "sha256": sha256_file(ROOT / relative)
                if exact
                else source_sha256(ROOT / relative),
                "hash_basis": "exact_bytes" if exact else "git_source_lf",
            }
        )
    lock = {
        "schema_version": 1,
        "status": "clean",
        "baseline_code_commit": BASELINE_CODE_COMMIT,
        "input_db": {
            "git_commit": INPUT_DB_COMMIT,
            "relative_path": "data/pgr_financials.db",
            "sha256": INPUT_DB_SHA256,
        },
        "parent_seed": {
            "git_commit": PARENT_SEED_COMMIT,
            "db_sha256": PARENT_SEED_SHA256,
        },
        "repairs": [
            {
                "id": "R2-lite",
                "commit": INPUT_DB_COMMIT,
                "rebuild_id": "Weekly Data Accumulation run36410317252",
                "review_pr": 144,
                "record": (
                    "docs/reviews/2026-09-28_R2_dividend_refresh_check.md"
                ),
            },
            {
                "id": "R3b",
                "commit": BASELINE_CODE_COMMIT,
                "review_pr": 145,
                "record": "docs/reviews/R3b_baseline_closeout.md",
            },
        ],
        "accepted_exceptions": [
            {
                "id": "VWO_MARCH_2026_GAP",
                "description": (
                    "Owner accepted provider history without "
                    "March2026 ex-date; exactly12 affected targets in"
                    " vwo_accepted_gap.csv, all quarantined"
                ),
            }
        ],
        "as_of": "2026-09-26",
        "development_boundary": str(BOUNDARY.date()),
        "data_status": (
            "approved repair preflight passed; reconstructed "
            "historical availability, vintage limits "
            "disclosed"
        ),
        "comparison_status": (
            "pending development execution; not usable as a "
            "v201+ comparator until accepted closeout"
        ),
        "endpoint_contracts": {
            "regression": (
                "fractional raw split/fractional-share DRIP "
                "PGR-minus-benchmark h6/h12; available at BME "
                "origin+h"
            ),
            "PathB": (
                "1 iff fixed investable weighted "
                "PGR-minus-portfolio6M return < -.03; production "
                "C.5 balanced logistic; same-label mature "
                "temperature stream"
            ),
            "cash12": (
                "next12M per origin-share cash vs past12M per "
                "origin-share cash"
            ),
            "annual_excess": (
                "x23/x18 post-policy November origin "
                "Dec1-nextFeb28/29 excess cash over past24M "
                "positive-payment median<=.25 / current "
                "originBVPS; unique annual events; "
                "positive-excess survivor"
            ),
            "bvps_growth": (
                "latest-filed reportBVPS at origin to exact "
                "reportmonth+12 BVPS; available on target filing;"
                " mature mean growth comparator"
            ),
            "absolute_pgr6": (
                "unbenchmarked PGR6M fractional split/DRIP "
                "return, not absolute-value magnitude; mature "
                "mean comparator"
            ),
        },
        "input_files": file_pins,
        "reference_sources": reference_sources,
        "artifact_byte_policy": (
            "Scoped -text attributes preserve exact new artifact/code bytes; "
            "inherited source text hashes use Git LF line-ending equivalence"
        ),
        "schema_migrations": audit["schema_migrations"],
        "runtime_lock_sha256": sha256_file(OUTPUTS / "runtime_lock.json"),
        "partition_ledger_sha256": sha256_file(OUTPUTS / "partitions.json"),
        "access_ledger_sha256": sha256_file(OUTPUTS / "access_ledger.json"),
        "tracked_db_sha256_before": tracked_before,
    }
    save_json(OUTPUTS / "baseline_lock.json", lock)
    verify_baseline_lock(OUTPUTS / "baseline_lock.json")
    assert sha256_file(ROOT / "data/pgr_financials.db") == tracked_before
    print(
        (
            "PREFLIGHT PASSED; exact repair, procedure and "
            "partitions sealed; no fit performed"
        ),
        flush=True,
    )


def production_bridge(db: Path, scratch: Path) -> list[dict]:
    """A1: actual entry functions vs independent production inputs."""
    from src.processing import feature_engineering as fe
    from pgr_vds.decision.signal_generation import generate_signals
    from pgr_vds.decision.health import compute_aggregate_health
    from src.models.multi_benchmark_wfo import run_ensemble_benchmarks
    from src.models.prequential import build_prequential_panel

    rows: list[dict] = []
    for horizon in (6, 12):
        conn = read_immutable(db, INPUT_DB_SHA256)
        try:
            with patch.object(
                fe,
                "_PROCESSED_PATH",
                str(scratch / f"bridge_h{horizon}.parquet"),
            ):
                signals, ensembles, diagnostics = generate_signals(
                    conn,
                    BRIDGE_AS_OF.date(),
                    horizon,
                )
            original = compute_aggregate_health(
                ensembles, horizon, panel=diagnostics["prequential_panel"]
            )
            features = bounded_features(
                db, BRIDGE_AS_OF, scratch / f"bridge_bounded_h{horizon}"
            )
            targets = load_targets(conn)
            targets = targets.loc[
                (targets["horizon"] == horizon)
                & (targets["available"] <= BRIDGE_AS_OF)
                & targets["benchmark"].isin(BENCHMARKS)
            ]
            matrix = targets.pivot(
                index="date", columns="benchmark", values="y_true"
            )
            matrix = matrix.reindex(columns=BENCHMARKS)
            reproduced = run_ensemble_benchmarks(
                features,
                matrix,
                target_horizon_months=horizon,
                model_feature_overrides=FEATURES,
            )
            panel = build_prequential_panel(reproduced)
            manual = compute_aggregate_health(reproduced, horizon, panel=panel)
            assert original is not None and manual is not None
            fields = ["oos_r2", "nw_ic", "agg_hit", "pt_p_value"]
            for field in fields:
                assert abs(original[field] - manual[field]) <= 1e-9, (
                    horizon,
                    field,
                )
            ew_original = float(
                np.mean([r["nw_ic"] for r in original["per_benchmark_rows"]])
            )
            ew_manual = float(
                np.mean([r["nw_ic"] for r in manual["per_benchmark_rows"]])
            )
            assert abs(ew_original - ew_manual) <= 1e-9
            assert set(ensembles) == set(BENCHMARKS)
            assert len(signals) == len(BENCHMARKS)
            rows.append(
                {
                    "horizon": horizon,
                    "as_of": str(BRIDGE_AS_OF.date()),
                    "protocol": "actual production vs independent assembly",
                    "train": 60,
                    "gap": 8 if horizon == 6 else 15,
                    "test": 6,
                    "oos_r2": original["oos_r2"],
                    "equal_weight_ic": ew_original,
                    "panel_ic": original["nw_ic"],
                    "hit_rate": original["agg_hit"],
                    "pt_p": original["pt_p_value"],
                    "max_metric_difference": max(
                        abs(original[field] - manual[field])
                        for field in fields
                    ),
                    "equivalence_passed": True,
                    "support_dates": original["n_dates"],
                    "interpretation": (
                        "development-only A1; live60-month h12 retained "
                        "only in bridge"
                    ),
                }
            )
            save_csv(scratch / f"production_panel_h{horizon}.csv", panel)
        finally:
            conn.close()
    return rows


def run(scratch: Path, destination: Path, bridge: bool) -> None:
    """Fit only after verifying the sealed clean data/procedure lock."""
    from pgr_vds.research_lib.controls import (
        classifier_summary,
        endpoint_controls,
        endpoint_summary,
        path_b_stream,
    )

    try:
        lock = verify_baseline_lock(OUTPUTS / "baseline_lock.json")
        verify_runtime(json.loads((OUTPUTS / "runtime_lock.json").read_text()))
        db = export_git_blob(
            INPUT_DB_COMMIT,
            "data/pgr_financials.db",
            scratch / "approved.db",
            INPUT_DB_SHA256,
        )
    except (OSError, ValueError) as error:
        save_json(
            OUTPUTS / "blocked_attempt.json",
            {
                "status": "blocked",
                "reason": repr(error),
                "fit_count": 0,
            },
        )
        (STUDY / "README.md").write_text(
            "v200 is blocked because its exact input/procedure lock "
            "failed verification. No fitting started.\n\n"
            f"Failure: `{error!r}`.\n\n"
            "What changed: a failed preflight attempt was recorded. "
            "What is left: resolve the pin or availability blocker "
            "before a clean baseline can be accepted.\n",
            encoding="utf-8",
        )
        raise
    before = sha256_file(ROOT / "data/pgr_financials.db")
    features = pd.read_csv(
        OUTPUTS / "development_features.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    targets = pd.read_csv(
        OUTPUTS / "development_targets.csv",
        parse_dates=["date", "target_end", "available"],
        float_precision="round_trip",
    )
    folds: list[dict] = []
    metrics: dict = {}
    all_panels: list[pd.DataFrame] = []
    bridge_rows: list[dict] = []
    for horizon in (6, 12):
        print(f"STRICT h{horizon} started; development only", flush=True)
        selected = targets.loc[
            (targets["horizon"] == horizon)
            & targets["benchmark"].isin(BENCHMARKS)
        ].copy()
        components, ledger = strict_predictions(
            features, selected, horizon, FEATURES
        )
        folds.extend(ledger)
        if components.empty:
            metrics[f"h{horizon}"] = {
                "status": "insufficient chronological evidence"
            }
            continue
        history = selected.rename(columns={"available": "label_available"})
        panel = ensemble_stream(components, history)
        panel = prequential_intervals(panel)
        panel["residual"] = panel["y_true"] - panel["y_hat"]
        panel["naive_residual"] = panel["y_true"] - panel["naive"]
        assert (panel["available"] < BOUNDARY).all()
        summary = panel_summary(panel, horizon, 2000, SEED)
        summary.update(calibration_summary(panel))
        for key in ("primary", "directional_skill"):
            probability = summary[key + "_p"]
            conservative = probability if np.isfinite(probability) else 1.0
            summary[key + "_holm38_p"] = holm38([conservative])[0]
        if horizon == 12:
            summary["secondary_squared_error_p"] = summary.pop("primary_p")
            summary.pop("primary_holm38_p")
            summary["test_role"] = "secondary 12M diagnostic only"
        else:
            summary["test_role"] = "preregistered 6M baseline reference test"
        summary["disposition"] = "research comparator; no promotion"
        summary["unscorable_outer_pairs"] = sum(
            row["kind"] == "outer" and row["status"] != "scorable"
            for row in ledger
        )
        metrics[f"h{horizon}"] = summary
        all_panels.append(panel)
        # Non-searched protocol bridge: same label support under fixed as-of.
        if bridge:
            limited = selected.loc[selected["available"] <= BRIDGE_AS_OF]
            strict_parts, _ = strict_predictions(
                features.loc[features.index <= BRIDGE_AS_OF],
                limited,
                horizon,
                FEATURES,
            )
            if not strict_parts.empty:
                strict_panel = ensemble_stream(
                    strict_parts,
                    limited.rename(columns={"available": "label_available"}),
                )
                save_csv(
                    scratch / f"strict_panel_h{horizon}.csv", strict_panel
                )
    save_csv(
        destination / "predictions.csv",
        pd.concat(all_panels, ignore_index=True)
        if all_panels
        else pd.DataFrame(columns=["date", "benchmark", "horizon"]),
    )
    save_csv(destination / "fold_ledger.csv", pd.DataFrame(folds))
    print("Fixed endpoint controls started", flush=True)
    conn = read_immutable(db, INPUT_DB_SHA256)
    try:
        controls, support = endpoint_controls(conn, features, BOUNDARY)
    finally:
        conn.close()
    pathb, classifier_folds = path_b_stream(
        features, targets.loc[targets["horizon"] == 6]
    )
    save_csv(destination / "control_predictions.csv", controls)
    save_csv(destination / "path_b_predictions.csv", pathb)
    save_csv(
        destination / "path_b_fold_ledger.csv", pd.DataFrame(classifier_folds)
    )
    save_json(destination / "control_support.json", support)
    metrics["path_b"] = classifier_summary(pathb)
    for key in ("raw", "calibrated"):
        probability = metrics["path_b"][key]["directional_skill_p"]
        conservative = probability if np.isfinite(probability) else 1.0
        metrics["path_b"][key]["directional_skill_holm38_p"] = holm38(
            [conservative]
        )[0]
    metrics["endpoint_controls"] = endpoint_summary(controls, support)
    save_json(destination / "metrics.json", metrics)
    observed = metrics.get("h6", {}).get("primary_p", float("nan"))
    observed = observed if np.isfinite(observed) else 1.0
    family = [observed] + [1.0] * 37
    save_json(
        destination / "multiplicity.json",
        {
            "primary_endpoint": "6M regression",
            "baseline_reference_raw_p": observed,
            "conservative_reference_family_p": family,
            "conservative_reference_family_holm_p": holm38(family),
            "campaign_candidates_executed": 0,
            "campaign_pending_or_unused_slots": 38,
            "campaign_p": [1.0] * 38,
            "secondary_endpoint": "12M, no primary success route",
            "interpretation": (
                "Baseline reference diagnostics use a conservative 38-test "
                "envelope; no campaign alternative or promotion is claimed"
            ),
        },
    )
    if bridge:
        print("Development-date production equivalence started", flush=True)
        bridge_rows = production_bridge(db, scratch)
        for horizon in (6, 12):
            strict_path = scratch / f"strict_panel_h{horizon}.csv"
            if not strict_path.exists():
                continue
            strict = pd.read_csv(strict_path, parse_dates=["date"])
            production = pd.read_csv(
                scratch / f"production_panel_h{horizon}.csv",
                parse_dates=["date"],
            )
            common = strict[["date", "benchmark"]].merge(
                production[["date", "benchmark"]]
            )
            for name, frame in [
                ("strict", strict),
                ("production", production),
            ]:
                matched = frame.merge(common, on=["date", "benchmark"])
                if "base_prediction" not in matched:
                    history = targets.loc[targets["horizon"] == horizon]
                    matched["base_prediction"] = [
                        int(
                            (
                                history.loc[
                                    (history["benchmark"] == row["benchmark"])
                                    & (history["available"] <= row["date"]),
                                    "y_true",
                                ]
                                > 0
                            ).mean()
                            >= 0.5
                        )
                        for row in matched.to_dict("records")
                    ]
                comparison = panel_summary(matched, horizon, 2000, SEED)
                bridge_rows.append(
                    {
                        "horizon": horizon,
                        "as_of": str(BRIDGE_AS_OF.date()),
                        "protocol": name + " on matched development support",
                        "metric_basis": (
                            "issued shrunk forecast IC; honest naive"
                        ),
                        "train": (60 if horizon == 6 else 120)
                        if name == "strict"
                        else 60,
                        "gap": 2 * horizon
                        if name == "strict"
                        else (8 if horizon == 6 else 15),
                        **comparison,
                    }
                )
        save_csv(destination / "bridge_table.csv", pd.DataFrame(bridge_rows))
    after = sha256_file(ROOT / "data/pgr_financials.db")
    assert before == after == INPUT_DB_SHA256
    save_json(
        destination / "db_hash_check.json",
        {"before": before, "after": after, "exit": 0},
    )
    outputs = {
        path.name: sha256_file(path)
        for path in sorted(destination.glob("*"))
        if path.is_file()
    }
    provenance = {
        "study": "v200_clean_baseline",
        "baseline_code_commit": BASELINE_CODE_COMMIT,
        "input_db_git": INPUT_DB_COMMIT,
        "input_db_sha256": INPUT_DB_SHA256,
        "parent_seed_git": PARENT_SEED_COMMIT,
        "parent_seed_db_sha256": PARENT_SEED_SHA256,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_dirty_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "baseline_lock_sha256": sha256_file(OUTPUTS / "baseline_lock.json"),
        "as_of": AS_OF,
        "max_available_development": targets["available"].max(),
        "extraction_sql": SQL,
        "target_definitions": lock["endpoint_contracts"],
        "runtime": runtime_lock(),
        "dependency_lock_sha256": lock["runtime_lock_sha256"],
        "seed": SEED,
        "folds": (
            "TimeSeriesSplit outer60/120,test6,gap12/24; "
            "inner3,test6,min24/60; unsupported recorded"
        ),
        "purge": "h",
        "embargo": "h",
        "attempts": [
            {
                "id": "incumbent_strict",
                "alternative_count": 0,
                "status": "completed development",
            }
        ],
        "budget": (
            "one incumbent blueprint, 50 inner Ridge "
            "penalties, 10 mature shrinkage values, fixed "
            "endpoint controls; no searched alternatives"
        ),
        "source_limits": [
            "Current-vintage FRED",
            "Repaired historical EDGAR with no per-row fetch timestamp",
            "Accepted VWO March2026 provider gap",
            "Weekly-bar ex-date reinvestment approximation",
        ],
        "splits_dividends_fred_edgar": (
            "Approved DB exact Git/SHA; economic-date bounded"
            " raw extraction; filings and one calendar macro "
            "publication lag; no providers called"
        ),
        "input_files": lock["input_files"],
        "output_sha256": outputs,
        "quarantine_ledger_sha256": sha256_file(OUTPUTS / "partitions.json"),
        "access_ledger_sha256": sha256_file(OUTPUTS / "access_ledger.json"),
        "holdout_metric_accesses": 0,
        "metrics": (
            "Honest R2 vs mature benchmark mean; EW and "
            "panelIC, paired date-block inference "
            "lengthh,2000 replicates; past majority base; "
            "mature Platt/ECE/Brier/logloss and incumbent80% "
            "ACI gamma.05 intervals"
        ),
    }
    save_json(
        (STUDY / "provenance.json")
        if destination == OUTPUTS
        else destination / "provenance.json",
        provenance,
    )
    print(
        (
            "DEVELOPMENT RUN COMPLETE; no quarantine metrics;"
            " tracked DB unchanged"
        ),
        flush=True,
    )


def main() -> None:
    """Bounded CLI: prepare only or one fixed development execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--scratch",
        type=Path,
        default=Path(tempfile.gettempdir()) / "pgr-v200-run",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument(
        "--skip-bridge",
        action="store_true",
        help="Determinism replay reuses A1 evidence; no second A1 fit",
    )
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args.scratch)
    else:
        run(args.scratch, args.output_dir, not args.skip_bridge)


if __name__ == "__main__":
    main()
