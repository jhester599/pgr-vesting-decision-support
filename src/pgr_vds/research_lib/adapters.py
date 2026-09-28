"""Causal adapters for the frozen incumbent; no production mutations."""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pgr_vds.research_lib.temporal import chronological_splits, label_end


TABLE_DATES = {
    "daily_prices": "date",
    "daily_dividends": "ex_date",
    "split_history": "split_date",
    "fred_macro_monthly": "month_end",
    "pgr_edgar_monthly": "filing_date",
    "pgr_fundamentals_quarterly": "filing_date",
}
RIDGE_GRID = np.logspace(-4, 4, 50)
SHRINKAGE_GRID = np.array(
    [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.75,
        1.00,
    ]
)


class AsOfCursor(sqlite3.Cursor):
    """Restrict raw economic dates and filings before legacy extraction."""

    def execute(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> sqlite3.Cursor:
        return super().execute(self.connection.bound_query(sql), parameters)


class AsOfConnection(sqlite3.Connection):
    """Immutable SQLite connection exposing only data available by cutoff.

    Macro release timing additionally follows the frozen calendar-lag rule
    in feature engineering. Revised historical vintages cannot be recovered.
    """

    as_of: pd.Timestamp

    def bound_query(self, sql: str) -> str:
        """Wrap each raw source with unchanged query filters."""
        bound = self.as_of.strftime("%Y-%m-%d")
        for table, column in TABLE_DATES.items():
            predicate = f"{column} <= '{bound}'"
            if table == "daily_prices":
                predicate += " AND ticker != 'CB'"
            sql = re.sub(
                rf"\bFROM\s+{table}\b",
                f"FROM (SELECT * FROM {table} WHERE {predicate})",
                sql,
                flags=re.IGNORECASE,
            )
        return sql

    def cursor(self, factory: type = AsOfCursor) -> sqlite3.Cursor:
        return super().cursor(factory)

    def execute(
        self,
        sql: str,
        parameters: tuple = (),
    ) -> sqlite3.Cursor:
        return super().execute(self.bound_query(sql), parameters)


def bounded_features(
    path: Path,
    as_of: pd.Timestamp,
    scratch: Path,
) -> pd.DataFrame:
    """Extract bounded rolling features with a fixed incumbent catalog.

    The legacy full-frame observation-count pruning is disabled only within
    this research call. Structurally absent incumbent columns remain NaN;
    training-fold imputers handle them without choosing columns using the
    observations available after an earlier forecast origin.
    """
    import config
    from src.processing import feature_engineering as features

    conn = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
        factory=AsOfConnection,
    )
    conn.row_factory = sqlite3.Row
    conn.as_of = pd.Timestamp(as_of)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with (
            patch.object(
                features, "_PROCESSED_PATH", str(scratch / "features.parquet")
            ),
            patch.object(config, "WFO_MIN_GAINSHARE_OBS", 0),
        ):
            frame = features.build_feature_matrix_from_db(conn)
        output = frame.loc[frame.index <= as_of].copy()
        catalog = {
            feature
            for model in ("ridge", "gbt")
            for feature in config.MODEL_FEATURE_OVERRIDES.get(model, [])
        }
        for feature in sorted(catalog - set(output.columns)):
            output[feature] = np.nan
        return output
    finally:
        conn.close()


def feature_source_ledger(
    conn: sqlite3.Connection,
    boundary: pd.Timestamp,
    origins: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Read development source dates without reading economic values.

    The caller supplies the immutable input connection. Every SELECT is
    bounded before the quarantine boundary; only source metadata is read.
    Actual EDGAR filings use the incumbent calendar placement helper. FRED
    dates use its frozen calendar-month lag assumption, which cannot recover
    release dates or the vintages available to historical decision makers.
    If supplied, origins identify the first development decision that can
    use each source row. This gate does not certify historical vintages.
    """
    import config
    from src.processing.feature_engineering import edgar_availability_dates

    cutoff = pd.Timestamp(boundary).normalize()
    date_bound = cutoff.strftime("%Y-%m-%d")
    columns = [
        "report_period",
        "filing_date",
        "usable_from",
        "source",
        "series_id",
        "lag_months",
        "availability_basis",
        "historical_vintage_assumption",
    ]
    frames: list[pd.DataFrame] = []
    for table, period in (
        ("pgr_edgar_monthly", "month_end"),
        ("pgr_fundamentals_quarterly", "period_end"),
    ):
        query = (
            f"SELECT {period} AS report_period,filing_date FROM {table} "
            f"WHERE {period} < ? "
            "AND (filing_date < ? OR filing_date IS NULL) "
            f"ORDER BY {period}"
        )
        frame = pd.read_sql_query(
            query, conn, params=(date_bound, date_bound)
        )
        if frame.empty:
            continue
        frame["report_period"] = pd.to_datetime(
            frame["report_period"], errors="coerce"
        )
        frame["filing_date"] = pd.to_datetime(
            frame["filing_date"], errors="coerce"
        )
        if (
            frame[["report_period", "filing_date"]].isna().any().any()
            or (frame["filing_date"] < frame["report_period"]).any()
        ):
            raise ValueError(f"Invalid or unknown filing metadata: {table}")
        frame["usable_from"] = edgar_availability_dates(
            pd.DatetimeIndex(frame["report_period"]), frame["filing_date"]
        )
        if (frame["usable_from"] < frame["filing_date"]).any():
            raise ValueError("EDGAR source placed before actual filing")
        frame["source"] = table
        frame["series_id"] = "PGR"
        frame["lag_months"] = np.nan
        frame["availability_basis"] = "actual filing; next business month end"
        frame["historical_vintage_assumption"] = (
            "stored filing metadata; original-vintage values unverified"
        )
        frames.append(frame.loc[frame["usable_from"] < cutoff, columns])

    selected = {
        feature
        for model in ("ridge", "gbt")
        for feature in config.MODEL_FEATURE_OVERRIDES.get(model, [])
    }
    required = sorted(
        {
            series
            for feature in selected
            for series in config.FRED_FEATURE_SOURCES.get(feature, ())
        }
    )
    if required:
        placeholders = ",".join("?" for _ in required)
        query = (
            "SELECT series_id,month_end AS report_period "
            "FROM fred_macro_monthly WHERE month_end < ? "
            f"AND series_id IN ({placeholders}) ORDER BY series_id,month_end"
        )
        frame = pd.read_sql_query(
            query, conn, params=(date_bound, *required)
        )
        if not frame.empty:
            frame["report_period"] = pd.to_datetime(
                frame["report_period"], errors="coerce"
            )
            if frame["report_period"].isna().any():
                raise ValueError("Invalid FRED source month")
            frame["lag_months"] = [
                int(config.FRED_SERIES_LAGS.get(
                    series, config.FRED_DEFAULT_LAG_MONTHS
                ))
                for series in frame["series_id"]
            ]
            if (frame["lag_months"] < 0).any():
                raise ValueError("Negative FRED calendar publication lag")
            frame["usable_from"] = [
                (date.to_period("M") + lag).to_timestamp()
                + pd.offsets.BMonthEnd(0)
                for date, lag in zip(
                    frame["report_period"], frame["lag_months"]
                )
            ]
            frame["filing_date"] = pd.NaT
            frame["source"] = "fred_macro_monthly"
            frame["availability_basis"] = (
                "assumed calendar publication lag; actual release absent"
            )
            frame["historical_vintage_assumption"] = (
                "current vintage with frozen lag; historical releases and "
                "vintages unavailable"
            )
            frames.append(frame.loc[frame["usable_from"] < cutoff, columns])
    ledger = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=columns)
    )
    if origins is not None:
        decisions = pd.DatetimeIndex(pd.to_datetime(origins)).sort_values()
        if decisions.hasnans or (decisions >= cutoff).any():
            raise ValueError("Source-use origins must precede quarantine")
        first = [
            decisions[decisions >= date][0]
            if (decisions >= date).any()
            else pd.NaT
            for date in ledger["usable_from"]
        ]
        ledger["first_origin_using_source"] = pd.to_datetime(first)
        ledger = ledger.dropna(subset=["first_origin_using_source"])
        ledger["source_available_by_first_origin"] = (
            ledger["usable_from"] <= ledger["first_origin_using_source"]
        )
        if not ledger["source_available_by_first_origin"].all():
            raise ValueError("Source availability exceeds decision origin")
    return ledger.sort_values(
        ["usable_from", "source", "series_id", "report_period"]
    ).reset_index(drop=True)


def drip_return(
    prices: pd.Series,
    dividends: pd.Series,
    splits: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, float]:
    """One raw share, explicit splits and fractional-share ex-date DRIP.

    Weekly source bars require reinvestment at the last observed raw close
    on or before the ex-date, matching the approved stored target contract.
    Start-date actions are excluded. Calendar target availability is checked
    separately; a partial final month must never be called a mature target.
    """
    raw = prices.dropna().sort_index().astype(float)
    first = raw.loc[raw.index <= start]
    last = raw.loc[raw.index <= end]
    if first.empty or last.empty or last.index[-1] <= first.index[-1]:
        raise ValueError("Insufficient observed endpoints")
    start_bar, end_bar = first.index[-1], last.index[-1]
    shares = 1.0
    events = [
        (d, 0, float(v)) for d, v in splits.items() if start_bar < d <= end_bar
    ]
    events += [
        (d, 1, float(v))
        for d, v in dividends.items()
        if start_bar < d <= end_bar
    ]
    for event, kind, value in sorted(events):
        if kind == 0:
            if value <= 0:
                raise ValueError("Invalid split")
            shares *= value
        else:
            price = float(raw.loc[raw.index <= event].iloc[-1])
            if price <= 0 or value < 0:
                raise ValueError("Invalid dividend or raw close")
            shares *= 1.0 + value / price
    return shares, shares * float(last.iloc[-1]) / float(first.iloc[-1]) - 1.0


def regular_dividend_baseline(
    dividends: pd.Series,
    origin: pd.Timestamp,
) -> float:
    """Archived x18/x23 small-payment median fixed at November origin."""
    past = dividends.loc[
        (dividends.index <= origin)
        & (dividends.index > origin - pd.DateOffset(months=24))
        & (dividends > 0)
        & (dividends <= 0.25)
    ].sort_index()
    if past.empty:
        return float("nan")
    return float(past.median())


def cash_window(
    dividends: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """Unadjusted per-share cash in (start, end], with date boundaries."""
    return float(
        dividends.loc[
            (dividends.index > start) & (dividends.index <= end)
        ].sum()
    )


def excess_ratio(total: float, ordinary: float, current_bvps: float) -> float:
    """Annual Q1 excess cash divided by current origin-date BVPS."""
    if current_bvps <= 0:
        raise ValueError("BVPS must be positive")
    return max(total - ordinary, 0.0) / current_bvps


def bvps_growth(current: float, future: float) -> float:
    """Fractional change on one share basis, on the origin basis."""
    if current <= 0:
        raise ValueError("BVPS must be positive")
    return future / current - 1.0


def ridge_pipeline(alpha: float) -> Pipeline:
    """Every imputer and scaler is fitted inside its own training history."""
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def nested_ridge(
    x: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    horizon: int,
    available: pd.Series | None = None,
) -> tuple[np.ndarray, float, list[dict]]:
    """Select the unchanged 50 penalties in three explicit inner folds."""
    splits = chronological_splits(x.index, horizon, inner=True)
    if len(splits) != 3:
        raise ValueError("Unsupported three-fold inner history")
    if available is None:
        available = pd.Series(
            [label_end(origin, horizon) for origin in x.index], index=x.index
        )
    available = available.reindex(x.index)
    errors = np.zeros(len(RIDGE_GRID))
    ledger: list[dict] = []
    minimum = 24 if horizon == 6 else 60
    for train, validation in splits:
        matured = np.array(
            [
                index
                for index in train
                if label_end(x.index[index], horizon) <= x.index[validation[0]]
                and available.iloc[index] <= x.index[validation[0]]
                and np.isfinite(y.iloc[index])
            ],
            dtype=int,
        )
        if len(matured) < minimum:
            raise ValueError("Unsupported inner training history")
        for index, alpha in enumerate(RIDGE_GRID):
            model = ridge_pipeline(float(alpha))
            model.fit(x.iloc[matured], y.iloc[matured])
            prediction = model.predict(x.iloc[validation])
            usable = np.isfinite(y.iloc[validation].to_numpy())
            if not usable.any():
                raise ValueError("Unsupported inner validation labels")
            errors[index] += np.sum(
                (y.iloc[validation].to_numpy()[usable] - prediction[usable])
                ** 2
            )
            if index == 0:
                ledger.append(
                    {
                        "n_train": len(matured),
                        "n_test": len(validation),
                        "train_start": str(x.index[matured[0]].date()),
                        "train_end": str(x.index[matured[-1]].date()),
                        "test_start": str(x.index[validation[0]].date()),
                        "test_end": str(x.index[validation[-1]].date()),
                        "scale_mean": model["scale"].mean_.tolist(),
                    }
                )
    alpha = float(RIDGE_GRID[int(np.argmin(errors))])
    selected = ridge_pipeline(alpha)
    usable = np.isfinite(y)
    selected.fit(x.loc[usable], y.loc[usable])
    return selected.predict(test), alpha, ledger


def ensemble_stream(
    components: pd.DataFrame,
    history: pd.DataFrame,
    min_shrinkage: int = 36,
) -> pd.DataFrame:
    """Production weight/shrinkage recipes using matured past predictions."""
    rows: list[dict] = []
    output = components.sort_values(["date", "benchmark"]).copy()
    for origin, current in output.groupby("date", sort=True):
        prior = output.loc[
            (output["available"] <= origin) & (output["date"] < origin)
        ]
        prior_scored = pd.DataFrame(rows)
        if not prior_scored.empty:
            prior_scored = prior_scored.loc[
                (prior_scored["available"] <= origin)
                & (prior_scored["date"] < origin)
            ]
        for row in current.to_dict("records"):
            benchmark = row["benchmark"]
            mature = prior.loc[prior["benchmark"] == benchmark]
            weights = np.array([0.5, 0.5])
            if not mature.empty:
                maes = np.array(
                    [
                        np.mean(np.abs(mature["y_true"] - mature[key]))
                        for key in ("ridge", "gbt")
                    ]
                )
                raw = np.where(maes > 1e-9, 1 / np.maximum(maes, 1e-9) ** 2, 1)
                weights = raw / raw.sum()
            z = float(weights @ [row["ridge"], row["gbt"]])
            alpha = 1.0
            if len(prior_scored) >= min_shrinkage:
                losses = [
                    float(
                        np.sum(
                            (prior_scored["y_true"] - a * prior_scored["z"])
                            ** 2
                        )
                    )
                    for a in SHRINKAGE_GRID
                ]
                best = min(losses)
                alpha = float(
                    max(
                        a
                        for a, loss in zip(SHRINKAGE_GRID, losses)
                        if loss <= best + 1e-15
                    )
                )
            mature_labels = history.loc[
                (history["benchmark"] == benchmark)
                & (history["label_available"] <= origin)
                & (history["date"] < origin)
            ]
            naive = float(mature_labels["y_true"].mean())
            majority = (
                float((mature_labels["y_true"] > 0).mean()) >= 0.5
                if len(mature_labels)
                else None
            )
            rows.append(
                {
                    **row,
                    "z": z,
                    "alpha": alpha,
                    "y_hat": alpha * z,
                    "ridge_weight": weights[0],
                    "naive": naive,
                    "n_mature_labels": len(mature_labels),
                    "base_prediction": int(majority)
                    if majority is not None
                    else float("nan"),
                    "shrinkage_warmup": len(prior_scored) < min_shrinkage,
                }
            )
    return pd.DataFrame(rows)


def prequential_intervals(
    panel: pd.DataFrame,
    min_support: int = 4,
) -> pd.DataFrame:
    """Incumbent nominal80% ACI, gamma.05, on mature residuals only."""
    from src.models.calibration import _build_platt_pipeline
    from src.models.conformal import aci_adjusted_interval

    output = panel.copy()
    output["lower"] = np.nan
    output["upper"] = np.nan
    output["probability"] = np.nan
    output["interval_warmup"] = True
    output["calibration_warmup"] = True
    for origin, current in panel.groupby("date", sort=True):
        mature = panel.loc[
            (panel["available"] <= origin) & (panel["date"] < origin)
        ]
        for index, row in current.iterrows():
            past = mature.loc[mature["benchmark"] == row["benchmark"]]
            if len(past) < min_support:
                continue
            interval = aci_adjusted_interval(
                row["y_hat"],
                (past["y_true"] - past["y_hat"]).to_numpy(),
                nominal_coverage=0.8,
                gamma=0.05,
            )
            output.loc[index, ["lower", "upper"]] = [
                interval.lower,
                interval.upper,
            ]
            output.loc[index, "interval_warmup"] = False
            binary = (past["y_true"] > 0).astype(int)
            if len(past) < 20 or binary.nunique() < 2:
                continue
            calibrator = _build_platt_pipeline()
            calibrator.fit(past[["z"]].to_numpy(), binary.to_numpy())
            probability = calibrator.predict_proba(np.array([[row["z"]]]))[
                0, 1
            ]
            output.loc[index, "probability"] = float(probability)
            output.loc[index, "calibration_warmup"] = False
    return output
