"""Write v201 closeout/provenance from saved outputs; never fit or query DB."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from pgr_vds.research_lib.provenance import runtime_lock, sha256_file


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
OUT = STUDY / "outputs/attempt2"
V200 = ROOT / "research/studies/v200_clean_baseline/outputs"


def load(name: str) -> Any:
    """Read one recorded JSON artifact."""
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    """Preserve explicit UTF8/LF strict JSON bytes."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def number(value: Any, digits: int = 3) -> str:
    """Format reported values without recomputing any metric."""
    return "—" if value is None else f"{value:.{digits}f}"


def finalize() -> None:
    """Publish all results, uncertainty, negatives, pins and test evidence."""
    registered = load("registered.json")
    record = load("run_record.json")
    metrics = load("metrics.json")
    comparison = load("comparison.json")
    closeout = load("closeout.json")
    controls = load("control_metrics.json")
    preflight = load("preflight.json")
    verification = load("verification/full_pytest_exit.json")
    rows = []
    for candidate in registered["candidate_order"]:
        primary = metrics[f"{candidate}_h6"]
        paired = comparison[f"{candidate}_h6"]
        result = closeout["candidate_results"][candidate]
        gates_text = (
            "passes" if result["passes"] else ", ".join(result["failed_gates"])
        )
        rows.append(
            f"| {candidate} | {number(primary['oos_r2'])} | "
            f"{number(paired['delta_r2'])} | "
            f"{number(paired['primary_p'], 4)} / "
            f"{number(paired['holm38_p'], 4)} | "
            f"{number(primary['equal_weight_ic'])} | "
            f"{number(primary['hit_rate'])} / "
            f"{number(primary['base_hit_rate'])} | "
            f"{number(primary['ece'])} / "
            f"{number(primary['coverage_80'])} | "
            f"{gates_text} |"
        )
    finalist = closeout["v207_finalist"]
    result_text = (
        f"{finalist} clears the preregistered development gates and is the "
        "single research finalist for v207. This is not a live promotion."
        if finalist
        else "None of the six candidates clears every preregistered gate. "
        "No finalist advances to v207 and the incumbent remains in place."
    )
    ci_rows, secondary_rows, calibration_rows = [], [], []
    for candidate in registered["candidate_order"]:
        for horizon in (6, 12):
            key = f"{candidate}_h{horizon}"
            m, c = metrics[key], comparison[key]
            ci_rows.append(
                f"| {candidate} / {horizon}M | "
                f"{number(c['delta_r2_ci'][0])}, "
                f"{number(c['delta_r2_ci'][1])} | "
                f"{number(m['equal_weight_ic_ci'][0])}, "
                f"{number(m['equal_weight_ic_ci'][1])} | "
                f"{number(m['panel_ic'])} / {number(m['panel_ic_p'], 4)} | "
                f"{number(m['directional_skill'])} / "
                f"{number(m['directional_skill_p'], 4)} |"
            )
            calibration_rows.append(
                f"| {candidate} / {horizon}M | {m['n_rows']} / "
                f"{m['n_dates']} | {m['n_calibration']} / "
                f"{m['n_intervals']} | {number(m['brier'])} / "
                f"{number(m['log_loss'])} | {number(m['ece'])} / "
                f"{number(m['coverage_80'])} |"
            )
            if horizon == 12:
                secondary_rows.append(
                    f"| {candidate} | {number(m['oos_r2'])} | "
                    f"{number(c['delta_r2'])} | "
                    f"{number(c['primary_p'], 4)} | "
                    f"{number(m['equal_weight_ic'])} |"
                )
    summary = verification["summary"]
    template = (STUDY / "README_TEMPLATE.md").read_text(encoding="utf-8")
    fields = {
        "field_0": result_text,
        "field_1": number(controls["h6"]["oos_r2"]),
        "field_2": number(controls["h6"]["equal_weight_ic"]),
        "field_3": number(controls["h6"]["hit_rate"]),
        "field_4": number(controls["h6"]["base_hit_rate"]),
        "field_5": number(controls["h6"]["ece"]),
        "field_6": number(controls["h6"]["coverage_80"]),
        "field_7": chr(10).join(rows),
        "field_8": number(controls["h12"]["oos_r2"]),
        "field_9": number(controls["h12"]["equal_weight_ic"]),
        "field_10": chr(10).join(secondary_rows),
        "field_11": chr(10).join(ci_rows),
        "field_12": chr(10).join(calibration_rows),
        "field_13": record["baseline_lock_sha256"],
        "field_14": record["input_git"],
        "field_15": record["input_db"]["git_commit"],
        "field_16": record["input_db"]["sha256"],
        "field_17": preflight["registry_check"]["pinned_at_commit"],
        "field_18": record["code_commit"],
        "field_19": summary,
        "field_20": verification["exit_code"],
        "field_21": verification["inherited_failures"],
        "field_22": verification["db_sha256_after"],
        "field_23": (
            "v207 quarantine synthesis for the frozen finalist; "
            "promotion remains separate."
        )
        if finalist
        else (
            "no v201 finalist; v207 campaign synthesis and genuinely "
            "unused forward evidence remain."
        ),
    }
    text = template.format_map(fields)
    (STUDY / "README.md").write_text(text, encoding="utf-8", newline="\n")
    (STUDY / "outputs/README.md").write_text(
        "Continuation results and verification are in [attempt 2](attempt2/). "
        "[Attempt 1](attempt1/) preserves the original blocked session. "
        "Other root files are its historical unscored evidence.\n",
        encoding="utf-8",
        newline="\n",
    )
    attempts = load("attempts.json")
    save(
        OUT / "candidate_ledger.json",
        {
            candidate: {
                "slot": slot,
                "blueprint": registered["blueprints"][candidate],
                "columns": registered["columns"][candidate],
                "fit_attempts": {"h6": 1, "h12": 1},
                "primary": comparison[f"{candidate}_h6"],
                "disposition": closeout["candidate_results"][candidate],
                "source_commit": record["code_commit"],
            }
            for slot, candidate in enumerate(registered["candidate_order"], 1)
        },
    )
    provenance = {
        "study": "v201_price_macro",
        "attempt": 2,
        "status": "completed",
        "as_of": "2026-09-26",
        "code_commit": record["code_commit"],
        "code_dirty_at_execution": record["tracked_dirty_at_start"],
        "artifact_finalization_git": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "artifact_finalizer_sha256": sha256_file(Path(__file__)),
        "baseline_code_commit": record["input_git"],
        "input_db": record["input_db"],
        "parent_seed": preflight["lock"]["parent_seed"],
        "repairs": preflight["lock"]["repairs"],
        "accepted_exceptions": preflight["lock"]["accepted_exceptions"],
        "baseline_lock_sha256": record["baseline_lock_sha256"],
        "registered_sha256": sha256_file(OUT / "registered.json"),
        "consumed_file_sha256": registered["input_sha256"],
        "attempt1_archive_sha256": registered["archive_sha256"],
        "registry_verification": preflight["registry_check"],
        "runtime": runtime_lock(),
        "runtime_lock_sha256": sha256_file(V200 / "runtime_lock.json"),
        "dependency_environment_sha256": sha256_file(
            V200 / "dependency_environment.json",
        ),
        "seed": 20260926,
        "bootstrap_replicates": 2000,
        "candidate_register": registered,
        "attempts": attempts,
        "target_definitions": preflight["lock"]["endpoint_contracts"][
            "regression"
        ],
        "target_end_availability": "origin+h BME; strictly before 2023-09-29",
        "max_feature_origin": "2023-08-31",
        "max_available_development_outcome": "2023-08-31",
        "extraction_sql": registered["sql"],
        "splits_dividends_provenance": "D7 approved DB; raw weekly closes; "
        "manual past splits; frozen fractional-share DRIP targets",
        "fred_provenance": registered["macro_limits"],
        "edgar_provenance": "unchanged v200 filing-gated feature CSV and "
        "its accepted current-table vintage limits",
        "purge": "h",
        "embargo": "h",
        "outer": registered["outer"],
        "inner": registered["inner"],
        "metric_definitions": "honest realised mean R2; matched paired "
        "delta; equal-benchmark/panel IC; date-block inference; past-learned "
        "majority base; prequential Brier/log loss/ECE/nominal80% coverage",
        "quarantine_partition_sha256": sha256_file(V200 / "partitions.json"),
        "v200_access_ledger_sha256": sha256_file(V200 / "access_ledger.json"),
        "access_ledger_sha256": sha256_file(OUT / "access_ledger.json"),
        "quarantine_metrics": 0,
        "provider_calls": 0,
        "email_calls": 0,
        "data_writes": 0,
        "live_changes": 0,
        "promotion": False,
        "full_pytest": verification,
        "run_record": record,
        "v207_finalist": finalist,
        "output_sha256": {
            p.relative_to(STUDY).as_posix(): sha256_file(p)
            for p in sorted(OUT.rglob("*"))
            if p.is_file()
        },
    }
    save(STUDY / "provenance.json", provenance)
    files = [
        p
        for p in STUDY.rglob("*")
        if p.is_file()
        and p != STUDY / "outputs/output_manifest.json"
        and "__pycache__" not in p.parts
    ]
    save(
        STUDY / "outputs/output_manifest.json",
        {
            "scope": "all study files except this manifest; exact bytes",
            "sha256": {
                p.relative_to(STUDY).as_posix(): sha256_file(p)
                for p in sorted(files)
            },
        },
    )
    print(result_text)


if __name__ == "__main__":
    finalize()
