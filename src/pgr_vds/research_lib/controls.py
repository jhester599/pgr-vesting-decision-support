"""Frozen matched endpoint controls with explicit label availability."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

import config
from pgr_vds.research_lib.adapters import (
    bvps_growth,
    cash_window,
    drip_return,
    excess_ratio,
    regular_dividend_baseline,
)
from pgr_vds.research_lib.baseline import (
    calibration_summary,
    temperature_stream,
)
from pgr_vds.research_lib.metrics import honest_r2, panel_summary
from pgr_vds.research_lib.temporal import chronological_splits, label_end


ENDPOINT_COLUMNS = [
    "date",
    "endpoint",
    "benchmark",
    "horizon",
    "target_end",
    "available",
    "y_true",
    "y_hat",
    "naive",
    "n_mature_labels",
    "warmup",
    "residual",
    "event_id",
    "ordinary_baseline",
]
CLASSIFIER_COLUMNS = [
    "date",
    "benchmark",
    "horizon",
    "target_end",
    "available",
    "y_true",
    "composite_relative_return",
    "raw_probability",
    "probability",
    "temperature",
    "calibration_warmup",
    "raw_residual",
    "residual",
    "base_prediction",
    "base_positive_rate",
    "n_mature_labels",
    "fold",
]


def _definitions() -> dict[str, dict[str, Any]]:
    """Freeze endpoint definitions before calculating any target values."""
    return {
        "cash_dividend_12m": {
            "unit": "USD cash per one origin-date PGR share",
            "target": "Cash ex-dates in (origin, business month-end +12M]",
            "forecast": "Cash in the past 12 calendar months, origin basis",
            "availability": "Target business month-end",
            "horizon": 12,
        },
        "bvps_growth_12m": {
            "unit": "fractional BVPS growth on the origin share basis",
            "target": (
                "Latest filed report at origin to that report month +12; "
                "future/current BVPS minus one"
            ),
            "forecast": "Mean of earlier labels matured by origin",
            "availability": "Actual filing date of the exact future report",
            "horizon": 12,
        },
        "annual_excess_to_bvps": {
            "unit": "December-February excess cash / November current BVPS",
            "target": (
                "Post-policy November origin: max(December-February cash "
                "minus ordinary "
                "quarter baseline fixed at November,0)/latest filed BVPS; "
                "conditional positive-event survivor endpoint"
            ),
            "forecast": "Mean of earlier matured positive annual ratios",
            "availability": "Next-year February 28 or 29",
            "ordinary_rule": (
                "Median positive payments <=.25 in prior 24 months, "
                "fixed at November origin; no December information"
            ),
            "horizon": 12,
            "dependence": "One annual event per November, no monthly repeats",
        },
        "pgr_drip_return_6m": {
            "unit": "signed fractional total return of PGR alone",
            "target": (
                "One raw origin share, manual splits and fractional "
                "ex-date DRIP to business month-end +6M; unbenchmarked "
                "absolute asset return, never absolute-value magnitude"
            ),
            "forecast": "Mean of earlier matured PGR DRIP return labels",
            "availability": "Target business month-end",
            "horizon": 6,
        },
    }


def _share_factor(
    splits: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """Explicit share multiplier in (start,end], with no inferred actions."""
    values = splits.loc[(splits.index > start) & (splits.index <= end)]
    if (values <= 0).any() or not np.isfinite(values).all():
        raise ValueError("Invalid split ratio in endpoint source")
    return float(values.prod())


def _cash_on_origin_basis(
    dividends: pd.Series,
    splits: pd.Series,
    origin: pd.Timestamp,
) -> pd.Series:
    """Convert raw per-share payments into cash per one origin-date share."""
    amounts = []
    for date, amount in dividends.items():
        if date <= origin:
            amount /= _share_factor(splits, date, origin)
        else:
            amount *= _share_factor(splits, origin, date)
        amounts.append(float(amount))
    return pd.Series(amounts, index=dividends.index, dtype=float)


def _current_bvps(
    reports: pd.DataFrame,
    splits: pd.Series,
    origin: pd.Timestamp,
) -> tuple[float, pd.Series | None]:
    """Use the most recent report actually filed by the origin."""
    eligible = reports.loc[
        (reports["filing_date"] <= origin) & (reports["month_end"] <= origin)
    ]
    if eligible.empty:
        return float("nan"), None
    row = eligible.sort_values(["month_end", "filing_date"]).iloc[-1]
    raw = float(row["book_value_per_share"])
    value = raw / _share_factor(splits, row["month_end"], origin)
    return value, row


def _record(
    origin: pd.Timestamp,
    endpoint: str,
    horizon: int,
    end: pd.Timestamp,
    available: pd.Timestamp,
    actual: float,
    event_id: str,
    prediction: float = float("nan"),
    ordinary: float = float("nan"),
) -> dict[str, Any]:
    """Store endpoint identity and chronology before later mean forecasts."""
    return {
        "date": origin,
        "endpoint": endpoint,
        "benchmark": "PGR",
        "horizon": horizon,
        "target_end": end,
        "available": available,
        "y_true": actual,
        "y_hat": prediction,
        "event_id": event_id,
        "ordinary_baseline": ordinary,
    }


def endpoint_controls(
    conn: sqlite3.Connection,
    features: pd.DataFrame,
    boundary: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Read only bounded PGR sources and build fixed development controls.

    Source queries exclude the quarantine boundary and all later economic
    dates or filings. Calendar ends and filing availability are checked before
    future cash summation or BVPS arithmetic. Earlier raw monthly histories
    contribute to prevailing means even before features become usable.
    """
    definitions = _definitions()
    boundary = pd.Timestamp(boundary)
    requested = pd.DatetimeIndex(features.index).sort_values()
    requested = requested[requested < boundary]
    if requested.empty:
        return pd.DataFrame(columns=ENDPOINT_COLUMNS), definitions
    cutoff = str(boundary.date())
    prices_frame = pd.read_sql_query(
        "SELECT date,close FROM daily_prices "
        "WHERE ticker='PGR' AND date < ? ORDER BY date",
        conn,
        params=(cutoff,),
        parse_dates=["date"],
    )
    dividends_frame = pd.read_sql_query(
        "SELECT ex_date,amount FROM daily_dividends "
        "WHERE ticker='PGR' AND ex_date < ? ORDER BY ex_date",
        conn,
        params=(cutoff,),
        parse_dates=["ex_date"],
    )
    splits_frame = pd.read_sql_query(
        "SELECT split_date,split_ratio FROM split_history "
        "WHERE ticker='PGR' AND split_date < ? ORDER BY split_date",
        conn,
        params=(cutoff,),
        parse_dates=["split_date"],
    )
    reports = pd.read_sql_query(
        "SELECT month_end,filing_date,book_value_per_share "
        "FROM pgr_edgar_monthly WHERE month_end < ? AND filing_date < ? "
        "ORDER BY month_end,filing_date",
        conn,
        params=(cutoff, cutoff),
        parse_dates=["month_end", "filing_date"],
    )
    prices = prices_frame.set_index("date")["close"].astype(float)
    dividends = dividends_frame.set_index("ex_date")["amount"].astype(float)
    splits = splits_frame.set_index("split_date")["split_ratio"].astype(float)
    if dividends.index.has_duplicates or prices.index.has_duplicates:
        raise ValueError("Duplicate PGR economic dates are unsupported")
    first = (
        min(requested.min(), prices.index.min())
        if len(prices)
        else requested.min()
    )
    origins = pd.date_range(first, requested.max(), freq="BME")
    records: list[dict[str, Any]] = []
    zero_annual_events = 0
    for origin in origins:
        cash = _cash_on_origin_basis(dividends, splits, origin)
        end12 = label_end(origin, 12)
        if end12 < boundary:
            actual = cash_window(cash, origin, end12)
            prior_start = origin - pd.DateOffset(months=12)
            forecast = (
                cash_window(cash, prior_start, origin)
                if len(prices) and prices.index.min() <= prior_start
                else float("nan")
            )
            records.append(
                _record(
                    origin,
                    "cash_dividend_12m",
                    12,
                    end12,
                    end12,
                    actual,
                    f"cash:{origin:%Y-%m}",
                    forecast,
                )
            )
        current, current_report = _current_bvps(reports, splits, origin)
        if current_report is not None and np.isfinite(current) and current > 0:
            future_month = current_report["month_end"].to_period("M") + 12
            future_end = pd.offsets.BMonthEnd().rollback(
                future_month.to_timestamp(how="end").normalize()
            )
            if future_end < boundary:
                future_rows = reports.loc[
                    (reports["month_end"].dt.to_period("M") == future_month)
                    & (reports["filing_date"] < boundary)
                ]
                if len(future_rows):
                    future_report = future_rows.iloc[0]
                    arrival = future_report["filing_date"]
                    future = float(future_report["book_value_per_share"])
                    future *= _share_factor(
                        splits,
                        origin,
                        future_report["month_end"],
                    )
                    if np.isfinite(future):
                        records.append(
                            _record(
                                origin,
                                "bvps_growth_12m",
                                12,
                                future_end,
                                arrival,
                                bvps_growth(current, future),
                                f"bvps:{future_month}",
                            )
                        )
            if origin.month == 11 and origin.year >= 2018:
                annual_start = pd.Timestamp(origin.year, 12, 1)
                annual_end = (
                    pd.Timestamp(origin.year + 1, 2, 1) + pd.offsets.MonthEnd()
                )
                if annual_end < boundary:
                    ordinary = regular_dividend_baseline(cash, origin)
                    if np.isfinite(ordinary):
                        total = cash_window(
                            cash,
                            annual_start - pd.Timedelta(days=1),
                            annual_end,
                        )
                        ratio = excess_ratio(total, ordinary, current)
                        if ratio > 0:
                            records.append(
                                _record(
                                    origin,
                                    "annual_excess_to_bvps",
                                    12,
                                    annual_end,
                                    annual_end,
                                    ratio,
                                    f"annual-dec-feb:{origin.year + 1}",
                                    ordinary=ordinary,
                                )
                            )
                        else:
                            zero_annual_events += 1
        end6 = label_end(origin, 6)
        if end6 < boundary:
            try:
                _, actual = drip_return(
                    prices,
                    dividends,
                    splits,
                    origin,
                    end6,
                )
            except ValueError:
                continue
            records.append(
                _record(
                    origin,
                    "pgr_drip_return_6m",
                    6,
                    end6,
                    end6,
                    actual,
                    f"pgr-return:{origin:%Y-%m}",
                )
            )
    history = pd.DataFrame(records)
    if history.empty:
        return pd.DataFrame(columns=ENDPOINT_COLUMNS), definitions
    rows = []
    for row in history.to_dict("records"):
        if row["date"] not in requested:
            continue
        past = history.loc[
            (history["endpoint"] == row["endpoint"])
            & (history["date"] < row["date"])
            & (history["available"] <= row["date"])
            & (history["target_end"] <= row["date"])
        ]
        mean = float(past["y_true"].mean())
        if row["endpoint"] != "cash_dividend_12m":
            row["y_hat"] = mean
        row.update(
            naive=mean,
            n_mature_labels=len(past),
            warmup=not np.isfinite(row["y_hat"]) or not np.isfinite(mean),
            residual=row["y_true"] - row["y_hat"],
        )
        rows.append(row)
    output = pd.DataFrame(rows, columns=ENDPOINT_COLUMNS)
    for name, definition in definitions.items():
        selected = output.loc[output["endpoint"] == name]
        minimum = 24 if definition["horizon"] == 6 else 60
        unique = selected["event_id"].nunique()
        definition["support"] = {
            "n_origins": len(selected),
            "n_forecasts": int(selected["y_hat"].notna().sum()),
            "n_warmup": int(selected["warmup"].sum()),
            "n_raw_history_labels": int((history["endpoint"] == name).sum()),
            "n_unique_annual_events": int(unique)
            if name == "annual_excess_to_bvps"
            else None,
            "minimum_training_months": minimum,
            "status": "unscorable"
            if len(selected) < minimum
            else "control_only",
            "reason": (
                "Fixed control; model evidence requires strict "
                "chronological folds"
            ),
        }
    definitions["annual_excess_to_bvps"]["support"].update(
        n_zero_excess_events_excluded=zero_annual_events,
        status="unscorable",
        reason=(
            "Sparse annual positive events are not independent "
            "monthly observations"
        ),
    )
    return output, definitions


def path_b_stream(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fixed Path B on strict monthly folds; mature temperature calibration."""
    from src.models.path_b_classifier import make_path_b_model

    required = {"date", "benchmark", "y_true", "available", "horizon"}
    if not required.issubset(targets.columns):
        raise ValueError(
            "Path B targets need explicit dates, labels and availability"
        )
    source = targets.loc[targets["horizon"] == 6].copy()
    if source.empty:
        return pd.DataFrame(columns=CLASSIFIER_COLUMNS), []
    source["date"] = pd.to_datetime(source["date"])
    source["available"] = pd.to_datetime(source["available"])
    expected_ends = source["date"].map(lambda date: label_end(date, 6))
    if (source["available"] < expected_ends).any():
        raise ValueError("Label availability cannot precede the target end")
    if source.duplicated(["date", "benchmark"]).any():
        raise ValueError("Duplicate benchmark monthly labels are unsupported")
    dates = pd.date_range(
        source["date"].min(), source["date"].max(), freq="BME"
    )
    weights = config.INVESTABLE_CLASSIFIER_BASE_WEIGHTS
    values = source.pivot(index="date", columns="benchmark", values="y_true")
    arrivals = source.pivot(
        index="date", columns="benchmark", values="available"
    )
    values = values.reindex(index=dates, columns=list(weights))
    arrivals = arrivals.reindex(index=dates, columns=list(weights))
    arrivals = arrivals.apply(pd.to_datetime)
    complete = values.notna().all(axis=1) & arrivals.notna().all(axis=1)
    composite = values.mul(pd.Series(weights)).sum(axis=1).where(complete)
    available = arrivals.max(axis=1).where(complete)
    binary = (composite < -0.03).astype(float).where(complete)
    ends = pd.Series([label_end(date, 6) for date in dates], index=dates)
    columns = config.MODEL_FEATURE_OVERRIDES["ridge"]
    missing = [column for column in columns if column not in features]
    if missing:
        return pd.DataFrame(columns=CLASSIFIER_COLUMNS), [
            {
                "kind": "outer",
                "status": "unscorable",
                "n_train": 0,
                "reason": "Missing frozen Path B features: "
                + ",".join(missing),
            }
        ]
    x = features.reindex(dates)[columns]
    ledger: list[dict[str, Any]] = []
    records = []
    splits = chronological_splits(dates, 6)
    if not splits:
        ledger.append(
            {
                "kind": "outer",
                "status": "unscorable",
                "n_train": 0,
                "reason": "Insufficient monthly history for fixed outer folds",
            }
        )
    for fold, (train, test) in enumerate(splits):
        first_test = dates[test[0]]
        usable = (
            binary.iloc[train].notna()
            & (available.iloc[train] <= first_test)
            & (ends.iloc[train] <= first_test)
            & x.iloc[train].notna().any(axis=1)
        ).to_numpy()
        mature = train[usable]
        entry = {
            "kind": "outer",
            "fold": fold,
            "endpoint": "path_b",
            "horizon": 6,
            "gap": 12,
            "purge": 6,
            "embargo": 6,
            "n_train": len(mature),
            "n_test": len(test),
            "train_start": str(dates[train[0]].date()),
            "train_end": str(dates[train[-1]].date()),
            "test_start": str(first_test.date()),
            "test_end": str(dates[test[-1]].date()),
            "status": "unscorable",
        }
        ledger.append(entry)
        if len(mature) < 24 or binary.iloc[mature].nunique() < 2:
            entry["reason"] = (
                "Insufficient mature support or single training class"
            )
            continue
        inner_valid = True
        for inner_fold, (inner_train, validation) in enumerate(
            chronological_splits(dates[train], 6, inner=True)
        ):
            absolute_train = train[inner_train]
            absolute_test = train[validation]
            origin = dates[absolute_test[0]]
            keep = (
                binary.iloc[absolute_train].notna()
                & (available.iloc[absolute_train] <= origin)
                & (ends.iloc[absolute_train] <= origin)
                & x.iloc[absolute_train].notna().any(axis=1)
            ).to_numpy()
            usable_inner = absolute_train[keep]
            valid_validation = absolute_test[
                binary.iloc[absolute_test].notna().to_numpy()
                & (available.iloc[absolute_test] <= first_test).to_numpy()
                & (ends.iloc[absolute_test] <= first_test).to_numpy()
                & x.iloc[absolute_test].notna().any(axis=1).to_numpy()
            ]
            scorable = (
                len(usable_inner) >= 24
                and binary.iloc[usable_inner].nunique() >= 2
                and len(valid_validation) > 0
            )
            inner_entry = {
                "kind": "inner",
                "fold": fold,
                "inner_fold": inner_fold,
                "endpoint": "path_b",
                "horizon": 6,
                "gap": 12,
                "purge": 6,
                "embargo": 6,
                "test_size": 6,
                "n_train": len(usable_inner),
                "n_test": len(validation),
                "n_test_usable": len(valid_validation),
                "test_start": str(origin.date()),
                "status": "scorable" if scorable else "unscorable",
                "reason": "Fixed C=.5; no hyperparameter selection",
            }
            if scorable:
                inner_model = make_path_b_model()
                inner_model.fit(
                    x.iloc[usable_inner],
                    binary.iloc[usable_inner].astype(int),
                )
                inner_entry["scale_mean"] = inner_model["scale"].mean_.tolist()
            ledger.append(inner_entry)
            inner_valid &= scorable
        if not inner_valid:
            entry["reason"] = "Unsupported fixed three-fold inner history"
            continue
        valid_test = test[
            binary.iloc[test].notna().to_numpy()
            & x.iloc[test].notna().any(axis=1).to_numpy()
        ]
        if not len(valid_test):
            entry["reason"] = "Missing matched composite test labels"
            continue
        model = make_path_b_model()
        model.fit(x.iloc[mature], binary.iloc[mature].astype(int))
        probabilities = model.predict_proba(x.iloc[valid_test])[:, 1]
        entry["status"] = "scorable"
        for index, probability in zip(valid_test, probabilities):
            origin = dates[index]
            past = (
                (dates < origin)
                & (available <= origin)
                & (ends <= origin)
                & binary.notna()
            )
            prior = binary.loc[past]
            positive = float(prior.mean()) if len(prior) else float("nan")
            records.append(
                {
                    "date": origin,
                    "benchmark": "PATH_B",
                    "horizon": 6,
                    "target_end": ends.iloc[index],
                    "available": available.iloc[index],
                    "y_true": int(binary.iloc[index]),
                    "composite_relative_return": float(composite.iloc[index]),
                    "raw_probability": float(probability),
                    "fold": fold,
                    "base_prediction": int(positive >= 0.5)
                    if len(prior)
                    else float("nan"),
                    "base_positive_rate": positive,
                    "n_mature_labels": len(prior),
                }
            )
    if not records:
        return pd.DataFrame(columns=CLASSIFIER_COLUMNS), ledger
    output = temperature_stream(pd.DataFrame(records))
    output["raw_residual"] = output["y_true"] - output["raw_probability"]
    output["residual"] = output["y_true"] - output["probability"]
    return output.reindex(columns=CLASSIFIER_COLUMNS), ledger


def _classifier_probability_summary(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """Reuse tested proper scores and date-block directional inference."""
    work = frame.loc[
        np.isfinite(frame["y_true"]) & np.isfinite(frame[column])
    ].copy()
    calibration = work.assign(
        probability=work[column], lower=np.nan, upper=np.nan
    )
    proper = calibration_summary(calibration)
    panel = work.assign(
        y_true=2 * work["y_true"] - 1,
        y_hat=work[column] - 0.5,
        naive=2 * work["base_positive_rate"] - 1,
        base_prediction=2 * work["base_prediction"] - 1,
    )
    statistics = panel_summary(panel, 6, replicates=2000, seed=20260926)
    directional = (
        "hit_rate",
        "base_hit_rate",
        "directional_skill",
        "directional_skill_p",
        "equal_weight_ic",
        "equal_weight_ic_ci",
        "equal_weight_ic_p",
        "panel_ic",
        "panel_ic_ci",
        "panel_ic_p",
        "block_length",
        "bootstrap_replicates",
        "seed",
        "inference_supported",
        "inference_method",
    )
    result = {name: statistics[name] for name in directional}
    result.update(
        brier=proper["brier"],
        log_loss=proper["log_loss"],
        ece=proper["ece"],
        n_probability=proper["n_calibration"],
        n_dates=work["date"].nunique(),
        n_direction_rows=statistics["n_rows"],
        n_direction_dates=statistics["n_dates"],
        observed_positive_rate=statistics["base_rate"],
        mean_past_positive_rate=float(work["base_positive_rate"].mean()),
        probability_threshold="class 1 if probability > .5; tie class 0",
        majority_rule="class 1 if mature past positive rate >= .5",
        ece_bins="10 fixed equal-width bins",
        probability_clip_for_log_loss="[1e-6,1-1e-6]",
        test_role="directional skill safeguard, separate binary endpoint",
    )
    return result


def classifier_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Score fixed Path B without treating classifier labels as returns.

    Calibration warmup remains unevaluated even if a caller filled its
    probability column. The direction comparator is each origin's stored
    majority of matured same-label history, including training history.
    Moving date blocks reuse the regression panel inference implementation,
    but its regression R-squared and primary loss test are not reported.
    """
    required = {
        "date",
        "benchmark",
        "y_true",
        "raw_probability",
        "probability",
        "calibration_warmup",
        "base_prediction",
        "base_positive_rate",
        "n_mature_labels",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Classifier summary requires probabilities and base")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    numeric = [
        "y_true",
        "raw_probability",
        "probability",
        "base_prediction",
        "base_positive_rate",
        "n_mature_labels",
    ]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    if not work["y_true"].dropna().isin([0, 1]).all():
        raise ValueError("Classifier outcomes must be binary labels")
    for column in ("raw_probability", "probability", "base_positive_rate"):
        finite = work[column].dropna()
        if ((finite < 0) | (finite > 1)).any():
            raise ValueError("Classifier probability must lie in [0,1]")
    base_ready = (
        (work["n_mature_labels"] > 0)
        & work["base_positive_rate"].notna()
        & work["base_prediction"].notna()
    )
    majority = (work.loc[base_ready, "base_positive_rate"] >= 0.5).astype(int)
    if not work.loc[base_ready, "base_prediction"].eq(majority).all():
        raise ValueError("Base direction must match mature past majority")
    work.loc[~base_ready, ["base_prediction", "base_positive_rate"]] = np.nan
    warmup = work["calibration_warmup"].fillna(True).astype(bool)
    calibrated = work.loc[~warmup]
    support: dict[str, Any] = {
        "n_rows": len(work),
        "n_origins": work["date"].nunique(),
        "n_calibration_warmup": int(warmup.sum()),
        "n_base_warmup": int((~base_ready).sum()),
        "origin_start": str(work["date"].min()),
        "origin_end": str(work["date"].max()),
        "minimum_mature_base_support": int(
            work.loc[base_ready, "n_mature_labels"].min()
        )
        if base_ready.any()
        else None,
    }
    if "available" in work:
        arrivals = pd.to_datetime(work["available"])
        support.update(
            availability_start=str(arrivals.min()),
            availability_end=str(arrivals.max()),
        )
    return {
        "endpoint": "path_b_same_label_classifier",
        "target": (
            "1 if fixed six-benchmark 6M composite relative return < -.03"
        ),
        "unit": "binary probability of the registered same-label event",
        "horizon": 6,
        "support": support,
        "raw": _classifier_probability_summary(work, "raw_probability"),
        "calibrated": _classifier_probability_summary(
            calibrated, "probability"
        ),
        "disposition": (
            "fixed research control; no endpoint pooling or promotion"
        ),
    }


def endpoint_summary(
    frame: pd.DataFrame,
    definitions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Report each fixed endpoint descriptively against its mature mean.

    No statistical test pools endpoint units or treats repeated BVPS reports
    as independent events. Sparse annual positive-event controls retain their
    support disclosure but have no scored performance or inference.
    """
    required = {
        "date",
        "endpoint",
        "y_true",
        "y_hat",
        "naive",
        "warmup",
        "event_id",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Endpoint summary requires targets and honest naive")
    result: dict[str, dict[str, Any]] = {}
    for name, definition in definitions.items():
        selected = frame.loc[frame["endpoint"] == name].copy()
        numeric = ["y_true", "y_hat", "naive"]
        selected[numeric] = selected[numeric].apply(
            pd.to_numeric, errors="coerce"
        )
        warmup = selected["warmup"].fillna(True).astype(bool)
        eligible = selected.loc[
            ~warmup & np.isfinite(selected[numeric]).all(axis=1)
        ]
        annual = name == "annual_excess_to_bvps"
        support = dict(definition.get("support", {}))
        summary: dict[str, Any] = {
            "unit": definition["unit"],
            "horizon": definition["horizon"],
            "n_rows": len(selected),
            "n_origins": selected["date"].nunique(),
            "n_unique_events": selected["event_id"].nunique(),
            "n_warmup": int(warmup.sum()),
            "n_finite_forecasts": len(eligible),
            "n_scored": 0 if annual else len(eligible),
            "status": "unscorable"
            if annual or len(eligible) < 2
            else "descriptive_only",
            "support": support,
            "oos_r2": None,
            "mae": None,
            "rmse": None,
            "inference_supported": False,
            "reason": (
                "Sparse unique annual events; no performance scores"
                if annual
                else "Fixed descriptive control; no inferential endpoint test"
            ),
            "comparator": "Mean of same-endpoint labels matured by origin",
        }
        if not annual and len(eligible) >= 2:
            residual = eligible["y_true"] - eligible["y_hat"]
            summary.update(
                oos_r2=honest_r2(
                    eligible["y_true"], eligible["y_hat"], eligible["naive"]
                ),
                mae=float(residual.abs().mean()),
                rmse=float(np.sqrt(np.mean(residual**2))),
            )
        result[name] = summary
    return result
