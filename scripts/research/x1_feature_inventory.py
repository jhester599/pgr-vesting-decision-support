"""Build x1 feature inventory and target sufficiency artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.database import db_client
from src.processing.feature_engineering import get_feature_columns
from src.research.x1_targets import (
    build_decomposition_targets,
    build_forward_return_targets,
    build_special_dividend_targets,
)

OUTPUT_DIR = Path("results") / "research"
INVENTORY_PATH = OUTPUT_DIR / "x1_feature_inventory.json"
MEMO_PATH = OUTPUT_DIR / "x1_research_memo.md"
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)


FEATURE_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "price_momentum_volatility": (
        "mom_",
        "vol_",
        "high_52w",
    ),
    "technical_indicators_existing": (
        "sma_",
        "rsi",
        "macd",
        "bb_",
        "ta_",
    ),
    "book_value_related": (
        "book_value",
        "pb_ratio",
    ),
    "buyback_capital_return": (
        "buyback",
        "shares_repurchased",
    ),
    "gainshare": (
        "gainshare",
    ),
    "peer_market_relative": (
        "pgr_vs_",
        "vwo_vxus",
        "gold_vs_",
        "commodity_equity",
    ),
    "valuation": (
        "pe_ratio",
        "roe",
    ),
    "underwriting_profitability": (
        "combined_ratio",
        "loss_ratio",
        "expense_ratio",
        "underwriting",
        "cr_",
    ),
    "growth_pif_premium": (
        "pif",
        "npw",
        "npe",
        "unearned",
        "channel_mix",
    ),
    "capital_balance_sheet": (
        "book_value",
        "bvps",
        "debt",
        "equity",
        "investments",
        "assets",
        "liabilities",
        "capital",
    ),
    "macro_rates_spreads": (
        "yield",
        "rate",
        "spread",
        "nfci",
        "vix",
        "cpi",
        "vmt",
        "term_premium",
    ),
}


def _month_end_close(prices: pd.DataFrame) -> pd.Series:
    close = prices["close"].copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    result = close.resample("ME").last()
    result.name = "close_price"
    return result


def _series_summary(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    return {
        "non_null": int(non_null.shape[0]),
        "missing": int(series.isna().sum()),
        "start": (
            non_null.index.min().date().isoformat()
            if not non_null.empty
            else None
        ),
        "end": non_null.index.max().date().isoformat() if not non_null.empty else None,
    }


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "columns": 0,
            "start": None,
            "end": None,
        }
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "start": frame.index.min().date().isoformat(),
        "end": frame.index.max().date().isoformat(),
    }


def categorize_features(feature_columns: list[str]) -> dict[str, list[str]]:
    """Categorize feature names into exclusive x-series planning groups."""
    groups: dict[str, list[str]] = {name: [] for name in FEATURE_GROUP_PATTERNS}
    groups["uncategorized"] = []

    for column in feature_columns:
        lower = column.lower()
        for group, patterns in FEATURE_GROUP_PATTERNS.items():
            if any(pattern in lower for pattern in patterns):
                groups[group].append(column)
                break
        else:
            groups["uncategorized"].append(column)

    return {key: sorted(value) for key, value in groups.items()}


def _target_summary(targets: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        column: _series_summary(targets[column])
        for column in targets.columns
        if column.startswith("target_")
    }


def _build_monthly_snapshot_frame(
    monthly_close: pd.Series,
    pgr_monthly: pd.DataFrame,
) -> pd.DataFrame:
    snapshots = pgr_monthly.copy()
    snapshots = snapshots.join(monthly_close, how="left")
    if "eps_basic" in snapshots.columns:
        snapshots["net_income_ttm_per_share"] = snapshots["eps_basic"].rolling(
            12,
            min_periods=6,
        ).sum()
    return snapshots


def _load_feature_matrix_read_only() -> pd.DataFrame:
    """Load the existing processed feature matrix without refreshing the cache."""
    path = Path(config.DATA_PROCESSED_DIR) / "feature_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "x1 inventory requires the existing processed feature matrix cache "
            f"at {path}. Run a production refresh outside the x-series lane "
            "before regenerating this research artifact."
        )
    return pd.read_parquet(path)


def build_inventory() -> dict[str, Any]:
    """Build the x1 inventory payload from the checked-in database."""
    conn = db_client.get_connection(config.DB_PATH)
    try:
        feature_df = _load_feature_matrix_read_only()
        feature_columns = get_feature_columns(feature_df)
        prices = db_client.get_prices(conn, "PGR")
        dividends = db_client.get_dividends(conn, "PGR")
        pgr_monthly = db_client.get_pgr_edgar_monthly(conn)
    finally:
        conn.close()

    monthly_close = (
        _month_end_close(prices)
        if not prices.empty
        else pd.Series(dtype=float)
    )
    forward_targets = build_forward_return_targets(monthly_close, horizons=HORIZONS)

    decomposition_targets = pd.DataFrame()
    if not pgr_monthly.empty and "book_value_per_share" in pgr_monthly.columns:
        decomposition_targets = build_decomposition_targets(
            monthly_close,
            pgr_monthly["book_value_per_share"],
            horizons=HORIZONS,
        )

    special_dividend_targets = pd.DataFrame()
    if not pgr_monthly.empty and not dividends.empty:
        snapshot_frame = _build_monthly_snapshot_frame(monthly_close, pgr_monthly)
        special_dividend_targets = build_special_dividend_targets(
            snapshot_frame,
            dividends,
        )

    feature_groups = categorize_features(feature_columns)
    group_summaries = {
        group: {
            "feature_count": len(columns),
            "features": columns,
        }
        for group, columns in feature_groups.items()
    }

    return {
        "version": "x1",
        "artifact_classification": "research",
        "production_changes": False,
        "feature_matrix_source": "read_existing_processed_cache",
        "feature_group_mode": "exclusive_first_match",
        "source_reports": [
            "docs/archive/history/x1-pgr-model-reports/20260421-pgrmodels-chatgpt.md",
            "docs/archive/history/x1-pgr-model-reports/20260421-pgrmodels-gemini.md",
        ],
        "data_frames": {
            "feature_matrix": _frame_summary(feature_df),
            "pgr_monthly_edgar": _frame_summary(pgr_monthly),
            "pgr_monthly_close": _series_summary(monthly_close),
            "pgr_dividends": _frame_summary(dividends),
        },
        "feature_groups": group_summaries,
        "target_summaries": {
            "forward_returns": _target_summary(forward_targets),
            "decomposition": _target_summary(decomposition_targets),
            "special_dividend": {
                column: _series_summary(special_dividend_targets[column])
                for column in special_dividend_targets.columns
            },
        },
        "recommended_next_order": ["x2", "x3", "x4", "x5", "x6", "x7", "x8"],
        "methodology_notes": [
            "Use strict TimeSeriesSplit-based WFO with horizon-specific purge/embargo.",
            "Fit preprocessing inside fold pipelines only.",
            (
                "Keep x-series artifacts separate from v-series production "
                "and shadow lanes."
            ),
            (
                "Treat annual special-dividend models as small-sample "
                "capital-allocation research."
            ),
        ],
    }


def _target_count(payload: dict[str, Any], section: str, name: str) -> int:
    section_payload = payload["target_summaries"].get(section, {})
    return int(section_payload.get(name, {}).get("non_null", 0))


def write_memo(payload: dict[str, Any], path: Path = MEMO_PATH) -> None:
    """Write the human-readable x1 memo."""
    feature_matrix = payload["data_frames"]["feature_matrix"]
    edgar = payload["data_frames"]["pgr_monthly_edgar"]
    dividends = payload["data_frames"]["pgr_dividends"]
    special_count = _target_count(
        payload,
        "special_dividend",
        "special_dividend_excess",
    )
    lines = [
        "# x1 Research Memo",
        "",
        "## Scope",
        "",
        "x1 sets up the separate x-series research lane for absolute PGR",
        "forecasting and Q1 special-dividend forecasting. It does not fit",
        "models and does not alter monthly decision outputs.",
        "",
        "## Available History",
        "",
        (
            f"- Feature matrix: {feature_matrix['rows']} rows x "
            f"{feature_matrix['columns']} columns, "
            f"{feature_matrix['start']} to {feature_matrix['end']}."
        ),
        "- Feature matrix source: existing processed cache, read without refresh.",
        (
            f"- PGR monthly EDGAR: {edgar['rows']} rows, "
            f"{edgar['start']} to {edgar['end']}."
        ),
        (
            f"- PGR dividends: {dividends['rows']} rows, "
            f"{dividends['start']} to {dividends['end']}."
        ),
        "",
        "## Candidate Target Depth",
        "",
    ]
    for horizon in HORIZONS:
        lines.append(
            f"- {horizon}m forward return: "
            f"{_target_count(payload, 'forward_returns', f'target_{horizon}m_return')} "
            "non-null observations."
        )
    lines.extend(
        [
            (
                "- Special-dividend annual snapshots: "
                f"{special_count} labeled observations."
            ),
            "",
            "## Feature Inventory Takeaways",
            "",
        ]
    )
    for group, summary in payload["feature_groups"].items():
        lines.append(f"- {group}: {summary['feature_count']} features.")
    lines.extend(
        [
            "",
            "## Recommended Ordering",
            "",
            "1. x2 multi-horizon classification baseline.",
            "2. x3 direct forward-return/log-return implied-price benchmark.",
            "3. x4/x5 BVPS and P/B decomposition benchmark.",
            "4. x6 special-dividend two-stage annual sidecar.",
            "5. x7 targeted feature expansion, including bounded TA follow-up.",
            "",
            "## Caveats",
            "",
            "- Annual special-dividend labels are sparse and should be treated as",
            "  high-uncertainty capital-allocation research.",
            "- Multi-month absolute targets overlap by construction, so x2+ must use",
            "  horizon-specific purge/embargo logic.",
            "- The normal quarterly dividend baseline is inferred from repo dividend",
            "  history rather than hardcoded.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inventory(payload: dict[str, Any], path: Path = INVENTORY_PATH) -> None:
    """Write the machine-readable x1 inventory artifact."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_inventory()
    write_inventory(payload)
    write_memo(payload)
    print(f"Wrote {INVENTORY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
