"""R4 synthetic monitoring replay; no provider, model fit or tracked DB use."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pgr_vds.decision.artifacts import update_shadow_ledgers
from src.reporting.classification_artifacts import (
    append_ta_shadow_variant_history,
    build_ta_shadow_variant_history_entries,
    ta_shadow_variant_history_path,
)


@pytest.fixture
def synthetic_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[sqlite3.Connection]:
    """Two independently specified relative returns: mean(-.08, -.04)=-.06."""
    import config

    monkeypatch.setattr(config, "PRIMARY_FORECAST_UNIVERSE", ["AAA", "BBB"])
    conn = sqlite3.connect(tmp_path / "synthetic.db")
    conn.execute(
        "CREATE TABLE monthly_relative_returns (date TEXT, benchmark TEXT, "
        "target_horizon INTEGER, relative_return REAL)"
    )
    conn.executemany(
        "INSERT INTO monthly_relative_returns VALUES (?, ?, ?, ?)",
        [("2020-01-31", "AAA", 6, -0.08), ("2020-01-31", "BBB", 6, -0.04)],
    )
    conn.commit()
    yield conn
    conn.close()


def _seed(base: Path) -> Path:
    entries = build_ta_shadow_variant_history_entries(
        as_of_date=date(2020, 2, 1),
        run_date=date(2020, 2, 1),
        forecast_horizon_months=6,
        classification_shadow_variants=[
            {
                "variant": "ta_minimal_replacement",
                "reporting_only": True,
                "feature_anchor_date": "2020-01-31",
                "probability_actionable_sell": 0.31,
                "stance": "NON-ACTIONABLE",
                "benchmark_count": 2,
            }
        ],
    )
    return append_ta_shadow_variant_history(base_dir=base, entries=entries)


def _run(
    conn: sqlite3.Connection,
    base: Path,
    as_of: date,
    *,
    dry_run: bool = False,
) -> pd.DataFrame:
    return update_shadow_ledgers(
        conn,
        as_of=as_of,
        run_date=date(2021, 1, 1),
        history_base_dir=base,
        dry_run=dry_run,
        classification_shadow_summary=None,
        classification_shadow_variants=[],
        live_recommendation_mode="ACTIONABLE",
        live_sell_pct=0.75,
        shadow_gate_overlay=None,
    )


def test_ta_history_matures_on_actual_horizon_end(
    synthetic_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    path = _seed(tmp_path)
    before = pd.read_csv(path)
    _run(synthetic_db, tmp_path, date(2020, 7, 31))
    after = pd.read_csv(path)
    assert bool(after.loc[0, "is_horizon_mature"])
    assert after.loc[0, "mature_on_date"] == "2020-07-31"
    assert after.loc[0, "actual_basket_relative_return"] == pytest.approx(
        -0.06
    )
    assert after.loc[0, "actual_actionable_sell"] == 1.0
    monitoring = {
        "mature_on_date",
        "is_horizon_mature",
        "actual_basket_relative_return",
        "actual_actionable_sell",
    }
    issued = [column for column in before if column not in monitoring]
    pd.testing.assert_frame_equal(before[issued], after[issued])


def test_ta_history_does_not_mature_early(
    synthetic_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    path = _seed(tmp_path)
    _run(synthetic_db, tmp_path, date(2020, 7, 30))
    after = pd.read_csv(path)
    assert not bool(after.loc[0, "is_horizon_mature"])
    assert after["actual_basket_relative_return"].isna().all()


def test_ta_maturity_is_idempotent(
    synthetic_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    path = _seed(tmp_path)
    _run(synthetic_db, tmp_path, date(2020, 7, 31))
    first = path.read_bytes()
    assert pd.read_csv(path).loc[0, "actual_actionable_sell"] == 1.0
    _run(synthetic_db, tmp_path, date(2020, 7, 31))
    assert path.read_bytes() == first
    assert len(pd.read_csv(path)) == 1


def test_backdated_ta_run_ignores_later_outcomes(
    synthetic_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    path = _seed(tmp_path)
    _run(synthetic_db, tmp_path, date(2020, 7, 31))
    assert pd.read_csv(path).loc[0, "actual_actionable_sell"] == 1.0
    synthetic_db.execute(
        "INSERT INTO monthly_relative_returns VALUES "
        "('2020-08-31', 'AAA', 6, 9.0)"
    )
    _run(synthetic_db, tmp_path, date(2020, 7, 30))
    after = pd.read_csv(path)
    assert not bool(after.loc[0, "is_horizon_mature"])
    assert after["actual_actionable_sell"].isna().all()
    assert after["actual_basket_relative_return"].isna().all()


@pytest.mark.parametrize("missing", ["row", "null", "infinite"])
def test_ta_missing_basket_outcome_remains_unknown(
    synthetic_db: sqlite3.Connection,
    tmp_path: Path,
    missing: str,
) -> None:
    path = _seed(tmp_path)
    stale = pd.read_csv(path)
    stale["is_horizon_mature"] = True
    stale["actual_actionable_sell"] = 0.0
    stale["actual_basket_relative_return"] = 0.9
    stale.to_csv(path, index=False)
    if missing == "row":
        synthetic_db.execute(
            "DELETE FROM monthly_relative_returns WHERE benchmark='BBB'"
        )
    else:
        synthetic_db.execute(
            "UPDATE monthly_relative_returns SET relative_return=? "
            "WHERE benchmark='BBB'",
            (None if missing == "null" else float("inf"),),
        )
    _run(synthetic_db, tmp_path, date(2020, 8, 31))
    after = pd.read_csv(path)
    assert not bool(after.loc[0, "is_horizon_mature"])
    assert after["actual_actionable_sell"].isna().all()


def test_ta_replay_preserves_extra_issuance_metadata(
    synthetic_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    path = _seed(tmp_path)
    before = pd.read_csv(path)
    before["recipe"] = "synthetic-pinned-recipe"
    before["candidate_identity"] = "synthetic-fit-001"
    before["issued_regression_prediction"] = -0.02
    before["issued_sell_recommendation"] = 0.75
    before.to_csv(path, index=False)
    _seed(tmp_path)
    _run(synthetic_db, tmp_path, date(2020, 7, 31))
    after = pd.read_csv(path)
    for column in (
        "recipe",
        "candidate_identity",
        "issued_regression_prediction",
        "issued_sell_recommendation",
        "probability_actionable_sell",
    ):
        pd.testing.assert_series_equal(before[column], after[column])
    assert after.loc[0, "actual_basket_relative_return"] == pytest.approx(
        -0.06
    )


def test_ta_repair_script_is_copy_only_and_deterministic(
    synthetic_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    from scripts import repair_ta_shadow_history

    source = _seed(tmp_path)
    output = tmp_path / "repaired.csv"
    diff = tmp_path / "row_diff.csv"
    original = source.read_bytes()
    db_path = tmp_path / "synthetic.db"
    db_before = db_path.read_bytes()
    repair_ta_shadow_history.repair_copies(
        db_copy=db_path,
        ledger_copy=source,
        output=output,
        row_diff=diff,
        as_of=date(2020, 7, 31),
    )
    assert source.read_bytes() == original
    assert db_path.read_bytes() == db_before
    after = pd.read_csv(output)
    assert after.loc[0, "actual_basket_relative_return"] == pytest.approx(
        -0.06
    )
    changes = pd.read_csv(diff)
    assert set(changes["column"]) == {
        "is_horizon_mature",
        "actual_actionable_sell",
        "actual_basket_relative_return",
    }
    first = (output.read_bytes(), diff.read_bytes())
    repair_ta_shadow_history.repair_copies(
        db_copy=db_path,
        ledger_copy=source,
        output=output,
        row_diff=diff,
        as_of=date(2020, 7, 31),
    )
    assert (output.read_bytes(), diff.read_bytes()) == first
    with pytest.raises(ValueError, match="distinct"):
        repair_ta_shadow_history.repair_copies(
            db_copy=db_path,
            ledger_copy=source,
            output=source,
            row_diff=diff,
            as_of=date(2020, 7, 31),
        )


def test_ta_repair_rejects_repository_paths(tmp_path: Path) -> None:
    from scripts import repair_ta_shadow_history

    with pytest.raises(ValueError, match="outside"):
        repair_ta_shadow_history.repair_copies(
            db_copy=repair_ta_shadow_history.REPO_ROOT
            / "data/pgr_financials.db",
            ledger_copy=tmp_path / "ledger.csv",
            output=tmp_path / "output.csv",
            row_diff=tmp_path / "diff.csv",
            as_of=date(2020, 7, 31),
        )


def test_ta_maturity_uses_each_rows_horizon(
    synthetic_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    path = _seed(tmp_path)
    history = pd.read_csv(path)
    history.loc[0, "forecast_horizon_months"] = 12
    history.to_csv(path, index=False)
    synthetic_db.executemany(
        "INSERT INTO monthly_relative_returns VALUES (?, ?, ?, ?)",
        [("2020-01-31", "AAA", 12, 0.04), ("2020-01-31", "BBB", 12, 0.08)],
    )
    _run(synthetic_db, tmp_path, date(2021, 1, 29))
    after = pd.read_csv(path)
    assert after.loc[0, "mature_on_date"] == "2021-01-29"
    assert after.loc[0, "actual_basket_relative_return"] == pytest.approx(0.06)
    assert after.loc[0, "actual_actionable_sell"] == 0.0


def test_dry_run_does_not_rewrite_ta_ledger(
    synthetic_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    path = _seed(tmp_path)
    classifier = tmp_path / "classification_shadow_history.csv"
    pd.DataFrame(
        [
            {
                "feature_anchor_date": "2020-01-31",
                "is_horizon_mature": False,
                "classifier_prob_actionable_sell": 0.4,
            }
        ]
    ).to_csv(classifier, index=False)
    before = {p: p.read_bytes() for p in (path, classifier)}
    result = _run(synthetic_db, tmp_path, date(2020, 7, 31), dry_run=True)
    assert result.loc[0, "actual_actionable_sell"] == 1.0
    assert {p: p.read_bytes() for p in before} == before
    assert ta_shadow_variant_history_path(tmp_path) == path
