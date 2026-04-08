"""Helpers for diversified redeploy portfolio recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import config


@dataclass(frozen=True)
class RedeployFundSpec:
    """One investable fund used by the monthly redeploy portfolio."""

    ticker: str
    sleeve: str
    base_weight: float
    min_weight: float
    max_weight: float
    rationale: str


@dataclass(frozen=True)
class RedeployPortfolioRow:
    """Rendered row for one recommended fund."""

    ticker: str
    allocation: float
    sleeve: str
    rationale: str
    corr_to_pgr: float
    predicted_relative_return: float | None
    signal_view: str
    benchmark_outperform_prob: float | None


def v27_redeploy_specs() -> dict[str, RedeployFundSpec]:
    """Return the investable redeploy universe used by the monthly workflow."""
    return {
        "VTI": RedeployFundSpec(
            ticker="VTI",
            sleeve="Broad US equity substitute",
            base_weight=0.40,
            min_weight=0.28,
            max_weight=0.50,
            rationale="Research substitute for VOO when a total-market wrapper is preferred over the S&P 500 core sleeve.",
        ),
        "VOO": RedeployFundSpec(
            ticker="VOO",
            sleeve="Broad US equity core",
            base_weight=0.40,
            min_weight=0.28,
            max_weight=0.50,
            rationale="Core US beta sleeve that keeps the portfolio equity-heavy without recreating single-stock PGR risk.",
        ),
        "VGT": RedeployFundSpec(
            ticker="VGT",
            sleeve="Technology tilt",
            base_weight=0.20,
            min_weight=0.10,
            max_weight=0.30,
            rationale="Growth engine and explicit tech tilt when the relative signal supports owning more innovation exposure than a pure core index.",
        ),
        "SCHD": RedeployFundSpec(
            ticker="SCHD",
            sleeve="Value / dividend tilt",
            base_weight=0.15,
            min_weight=0.08,
            max_weight=0.22,
            rationale="Closest current project proxy for a value sleeve; adds a cheaper, income-oriented counterweight to the tech allocation.",
        ),
        "VXUS": RedeployFundSpec(
            ticker="VXUS",
            sleeve="International core",
            base_weight=0.10,
            min_weight=0.05,
            max_weight=0.18,
            rationale="Primary geographic diversifier away from a US employer-stock concentration.",
        ),
        "VWO": RedeployFundSpec(
            ticker="VWO",
            sleeve="Emerging-markets satellite",
            base_weight=0.10,
            min_weight=0.04,
            max_weight=0.16,
            rationale="Higher-growth international sleeve kept modest because it is more volatile than the core international allocation.",
        ),
        "BND": RedeployFundSpec(
            ticker="BND",
            sleeve="Bond ballast",
            base_weight=0.05,
            min_weight=0.00,
            max_weight=0.10,
            rationale="Small stabilizer sleeve kept intentionally light so the redeploy portfolio stays above 90% equities in normal months.",
        ),
        "VIG": RedeployFundSpec(
            ticker="VIG",
            sleeve="Quality / dividend substitute",
            base_weight=0.10,
            min_weight=0.04,
            max_weight=0.18,
            rationale="Research substitute for the value sleeve when a lower-volatility quality/dividend tilt is preferred.",
        ),
    }


def v27_investable_redeploy_universe() -> list[str]:
    """Return the investable redeploy tickers in display order."""
    return ["VOO", "VGT", "SCHD", "VXUS", "VWO", "BND"]


def v27_benchmark_pruning_review() -> list[dict[str, str]]:
    """Return a benchmark-review table for buyable vs contextual funds."""
    review: list[dict[str, str]] = []
    keep = {
        "VOO": "Core US equity sleeve.",
        "VGT": "Technology tilt.",
        "SCHD": "Value / dividend proxy.",
        "VXUS": "International core.",
        "VWO": "Emerging-markets satellite.",
        "BND": "Single bond sleeve.",
    }
    optional = {
        "VTI": "Broad-market substitute for VOO, but kept off the main list to avoid redundant US core funds.",
        "VIG": "Reasonable quality/dividend substitute, but secondary to SCHD for the stated value-tilt preference.",
    }
    contextual = {
        "VFH": "Too close to PGR's financial-sector exposure to be a preferred redeploy target.",
        "KIE": "Useful insurance context benchmark, but not a concentration-reduction destination.",
    }
    not_preferred = {
        "VHT": "Sector sleeve not central to the preferred broad-market redeploy policy.",
        "VIS": "Sector sleeve not central to the preferred broad-market redeploy policy.",
        "VDE": "Energy is too narrow for the default sell-proceeds answer.",
        "VPU": "Utilities are too defensive and narrow for the default equity-heavy answer.",
        "VEA": "Redundant with VXUS + VWO as the preferred international combination.",
        "BNDX": "Second bond sleeve is unnecessary in the preferred >90% equity profile.",
        "VCIT": "Credit sleeve is narrower than the preferred single-core bond fund.",
        "VMBS": "Mortgage sleeve is narrower than the preferred single-core bond fund.",
        "VNQ": "Real estate is optional, not part of the preferred repeatable default.",
        "GLD": "Gold may diversify, but it was not part of the preferred equity-heavy redeploy framework.",
        "DBC": "Broad commodities may diversify, but not part of the preferred repeatable default.",
    }

    for ticker in config.ETF_BENCHMARK_UNIVERSE:
        if ticker in keep:
            status = "keep_for_redeploy"
            reason = keep[ticker]
        elif ticker in optional:
            status = "optional_substitute"
            reason = optional[ticker]
        elif ticker in contextual:
            status = "contextual_only"
            reason = contextual[ticker]
        else:
            status = "not_preferred_for_redeploy"
            reason = not_preferred.get(
                ticker,
                "Not part of the preferred repeatable sell-proceeds answer.",
            )
        review.append({"benchmark": ticker, "status": status, "reason": reason})
    return review


def tilt_strength_for_mode(mode_label: str) -> float:
    """Map the current recommendation mode into a redeploy tilt budget."""
    normalized = str(mode_label).strip().upper()
    if normalized == "ACTIONABLE":
        return 0.45
    if normalized == "MONITORING-ONLY":
        return 0.35
    return 0.25


def _benchmark_outperform_probability(row: pd.Series) -> float | None:
    """Return P(benchmark beats PGR) when a calibrated/raw probability exists."""
    prob = row.get("calibrated_prob_outperform")
    if prob is None or pd.isna(prob):
        prob = row.get("prob_outperform")
    if prob is None or pd.isna(prob):
        return None
    return float(max(0.0, min(1.0, 1.0 - float(prob))))


def _confidence_multiplier(row: pd.Series) -> float:
    """Convert per-fund hit-rate / IC into a bounded confidence multiplier."""
    hit_rate = float(row.get("hit_rate", 0.5) or 0.5)
    ic = float(row.get("ic", 0.0) or 0.0)
    multiplier = 1.0 + (2.0 * (hit_rate - 0.5)) + max(ic, 0.0)
    return float(min(1.5, max(0.5, multiplier)))


def _raw_signal_score(row: pd.Series, diversification_score: float) -> float:
    """Return a non-negative attractiveness score for one benchmark fund."""
    predicted_relative_return = float(row.get("predicted_relative_return", 0.0) or 0.0)
    benchmark_edge = max(0.0, -predicted_relative_return)
    benchmark_prob = _benchmark_outperform_probability(row)
    prob_multiplier = 1.0
    if benchmark_prob is not None:
        prob_multiplier = 0.5 + (2.0 * max(0.0, benchmark_prob - 0.5))
    return float(benchmark_edge * diversification_score * _confidence_multiplier(row) * prob_multiplier)


def _apply_weight_bounds(
    weights: dict[str, float],
    specs: dict[str, RedeployFundSpec],
) -> dict[str, float]:
    """Project weights into the configured min/max bands."""
    active = {ticker: float(weight) for ticker, weight in weights.items()}
    locked: dict[str, float] = {}
    free = set(active)
    remaining = 1.0

    while free:
        changed = False
        free_sum = sum(active[ticker] for ticker in free)
        if free_sum <= 0:
            equal_weight = remaining / len(free)
            for ticker in list(free):
                active[ticker] = equal_weight
            free_sum = sum(active[ticker] for ticker in free)

        for ticker in list(free):
            spec = specs[ticker]
            candidate = active[ticker] / free_sum * remaining
            if candidate < spec.min_weight:
                locked[ticker] = spec.min_weight
                remaining -= spec.min_weight
                free.remove(ticker)
                changed = True
            elif candidate > spec.max_weight:
                locked[ticker] = spec.max_weight
                remaining -= spec.max_weight
                free.remove(ticker)
                changed = True
        if not changed:
            break

    if free:
        free_sum = sum(active[ticker] for ticker in free)
        if free_sum <= 0:
            equal_weight = remaining / len(free)
            for ticker in free:
                locked[ticker] = equal_weight
        else:
            for ticker in free:
                locked[ticker] = active[ticker] / free_sum * remaining

    total = sum(locked.values())
    if total <= 0:
        equal_weight = 1.0 / len(specs)
        return {ticker: equal_weight for ticker in specs}
    if np.isclose(total, 1.0, atol=1e-10):
        return locked
    return {ticker: weight / total for ticker, weight in locked.items()}


def recommend_redeploy_portfolio(
    signals: pd.DataFrame,
    diversification_scoreboard: pd.DataFrame,
    recommendation_mode_label: str,
    max_funds: int = 8,
) -> dict[str, Any]:
    """Build the concrete sell-proceeds portfolio recommendation for one month."""
    specs = v27_redeploy_specs()
    investable = v27_investable_redeploy_universe()
    diversification_map = diversification_scoreboard.set_index("benchmark").to_dict("index")

    signal_subset = signals.reset_index().rename(columns={"index": "benchmark"})
    signal_subset = signal_subset[signal_subset["benchmark"].isin(investable)].copy()
    signal_rows = signal_subset.set_index("benchmark").to_dict("index")

    base_weights = {ticker: specs[ticker].base_weight for ticker in investable}
    raw_scores: dict[str, float] = {}
    for ticker in investable:
        row = pd.Series(signal_rows.get(ticker, {}), dtype=object)
        raw_scores[ticker] = _raw_signal_score(
            row,
            float(diversification_map.get(ticker, {}).get("diversification_score", 0.0) or 0.0),
        )

    tilt_strength = tilt_strength_for_mode(recommendation_mode_label)
    total_raw = sum(raw_scores.values())
    if total_raw > 0:
        signal_target = {ticker: raw_scores[ticker] / total_raw for ticker in investable}
    else:
        signal_target = dict(base_weights)

    blended = {
        ticker: ((1.0 - tilt_strength) * base_weights[ticker]) + (tilt_strength * signal_target[ticker])
        for ticker in investable
    }
    bounded = _apply_weight_bounds(blended, specs)

    rows: list[RedeployPortfolioRow] = []
    for ticker, weight in sorted(bounded.items(), key=lambda item: item[1], reverse=True):
        if len(rows) >= max_funds:
            break
        spec = specs[ticker]
        row = pd.Series(signal_rows.get(ticker, {}), dtype=object)
        predicted_relative_return = row.get("predicted_relative_return")
        predicted_relative_value = (
            float(predicted_relative_return)
            if predicted_relative_return is not None and not pd.isna(predicted_relative_return)
            else None
        )
        benchmark_prob = _benchmark_outperform_probability(row)
        if (
            benchmark_prob is not None
            and predicted_relative_value is not None
            and (
                (predicted_relative_value < 0 and benchmark_prob < 0.5)
                or (predicted_relative_value > 0 and benchmark_prob > 0.5)
            )
        ):
            benchmark_prob = None
        signal_view = _signal_view_label(predicted_relative_value, benchmark_prob)
        rows.append(
            RedeployPortfolioRow(
                ticker=ticker,
                allocation=float(weight),
                sleeve=spec.sleeve,
                rationale=spec.rationale,
                corr_to_pgr=float(diversification_map.get(ticker, {}).get("corr_to_pgr", float("nan"))),
                predicted_relative_return=predicted_relative_value,
                signal_view=signal_view,
                benchmark_outperform_prob=benchmark_prob,
            )
        )

    total_equity = sum(row.allocation for row in rows if row.ticker != "BND")
    total_bonds = sum(row.allocation for row in rows if row.ticker == "BND")
    return {
        "rows": rows,
        "tilt_strength": tilt_strength,
        "total_equity": total_equity,
        "total_bonds": total_bonds,
        "universe": investable,
        "base_weights": base_weights,
        "note": (
            "The current project universe does not yet include a dedicated small-cap ETF, so the value sleeve uses SCHD and the broad-market sleeve stays in VOO."
        ),
    }


def _signal_view_label(
    predicted_relative_return: float | None,
    benchmark_outperform_prob: float | None,
) -> str:
    """Convert the model view into a short user-facing label."""
    if predicted_relative_return is None:
        return "Base-weight only"
    if predicted_relative_return <= -0.03:
        if benchmark_outperform_prob is not None and benchmark_outperform_prob >= 0.55:
            return "Highest-conviction buy"
        return "Preferred this month"
    if predicted_relative_return < 0:
        return "Supportive"
    if predicted_relative_return < 0.03:
        return "Keep near base"
    return "Only keep at floor weight"


def render_redeploy_portfolio_markdown_lines(portfolio: dict[str, Any]) -> list[str]:
    """Render the suggested redeploy portfolio into recommendation.md lines."""
    rows: list[RedeployPortfolioRow] = portfolio.get("rows", [])
    lines = ["## Suggested Redeploy Portfolio", ""]
    if not rows:
        return lines + ["- Suggested redeploy portfolio unavailable.", ""]

    lines += [
        (
            f"- Default posture: `{portfolio['total_equity']:.0%}` equities / "
            f"`{portfolio['total_bonds']:.0%}` bonds across the curated investable universe."
        ),
        (
            f"- Monthly tilts use a `{portfolio['tilt_strength']:.0%}` signal overlay around the base weights, "
            "so the recommendation can adapt without becoming a full tactical allocation model."
        ),
        f"- Investable universe used in the monthly workflow: `{', '.join(portfolio['universe'])}`.",
        f"- Constraint note: {portfolio['note']}",
        "",
        "| Fund | Allocation | Sleeve | Why it is included | PGR Correlation | Relative Signal | P(Benchmark Beats PGR) |",
        "|------|------------|--------|--------------------|-----------------|-----------------|------------------------|",
    ]
    for row in rows:
        corr_text = f"{row.corr_to_pgr:.2f}" if np.isfinite(row.corr_to_pgr) else "n/a"
        rel_text = (
            f"{row.predicted_relative_return:+.1%}"
            if row.predicted_relative_return is not None
            else "n/a"
        )
        prob_text = (
            f"{row.benchmark_outperform_prob:.1%}"
            if row.benchmark_outperform_prob is not None
            else "n/a"
        )
        lines.append(
            f"| {row.ticker} | {row.allocation:.0%} | {row.sleeve} | {row.rationale} | "
            f"{corr_text} | {row.signal_view} ({rel_text}) | {prob_text} |"
        )
    lines.append("")
    return lines


def portfolio_rows_to_frame(portfolio: dict[str, Any], *, as_of: str) -> pd.DataFrame:
    """Convert a live recommendation into a CSV-friendly DataFrame."""
    rows: list[RedeployPortfolioRow] = portfolio.get("rows", [])
    records = [
        {
            "as_of": as_of,
            "fund": row.ticker,
            "allocation": row.allocation,
            "sleeve": row.sleeve,
            "rationale": row.rationale,
            "corr_to_pgr": row.corr_to_pgr,
            "predicted_relative_return": row.predicted_relative_return,
            "signal_view": row.signal_view,
            "benchmark_outperform_prob": row.benchmark_outperform_prob,
        }
        for row in rows
    ]
    return pd.DataFrame.from_records(records)


def simulate_dynamic_portfolio(
    predicted_relative_panel: pd.DataFrame,
    monthly_return_panel: pd.DataFrame,
    signal_metric_frame: pd.DataFrame,
    diversification_scoreboard: pd.DataFrame,
    *,
    base_weights: dict[str, float],
    tilt_strength: float,
) -> pd.DataFrame:
    """Simulate the dynamic redeploy sleeve over historical monthly data."""
    tickers = list(base_weights)
    pred_panel = predicted_relative_panel[tickers].copy()
    ret_panel = monthly_return_panel[tickers].copy()
    common_index = pred_panel.index.intersection(ret_panel.index)
    pred_panel = pred_panel.loc[common_index].dropna(how="any")
    ret_panel = ret_panel.loc[common_index].dropna(how="any")
    common_index = pred_panel.index.intersection(ret_panel.index)
    pred_panel = pred_panel.loc[common_index]
    ret_panel = ret_panel.loc[common_index]

    metric_map = signal_metric_frame.set_index("benchmark").to_dict("index")
    div_map = diversification_scoreboard.set_index("benchmark").to_dict("index")
    specs = {ticker: v27_redeploy_specs()[ticker] for ticker in tickers}

    detail_rows: list[dict[str, Any]] = []
    previous_weights = dict(base_weights)
    for as_of, pred_row in pred_panel.iterrows():
        raw_scores: dict[str, float] = {}
        for ticker in tickers:
            synthetic_row = pd.Series(
                {
                    "predicted_relative_return": float(pred_row[ticker]),
                    "ic": metric_map.get(ticker, {}).get("ic", 0.0),
                    "hit_rate": metric_map.get(ticker, {}).get("hit_rate", 0.5),
                },
                dtype=float,
            )
            raw_scores[ticker] = _raw_signal_score(
                synthetic_row,
                float(div_map.get(ticker, {}).get("diversification_score", 0.0) or 0.0),
            )
        total_raw = sum(raw_scores.values())
        if total_raw > 0:
            signal_target = {ticker: raw_scores[ticker] / total_raw for ticker in tickers}
        else:
            signal_target = dict(base_weights)
        blended = {
            ticker: ((1.0 - tilt_strength) * base_weights[ticker]) + (tilt_strength * signal_target[ticker])
            for ticker in tickers
        }
        weights = _apply_weight_bounds(blended, specs)
        portfolio_return = float(sum(weights[ticker] * ret_panel.at[as_of, ticker] for ticker in tickers))
        turnover = float(sum(abs(weights[ticker] - previous_weights[ticker]) for ticker in tickers) / 2.0)
        previous_weights = weights
        for ticker in tickers:
            detail_rows.append(
                {
                    "date": as_of,
                    "fund": ticker,
                    "weight": weights[ticker],
                    "predicted_relative_return": float(pred_row[ticker]),
                    "fund_return_1m": float(ret_panel.at[as_of, ticker]),
                    "portfolio_return_1m": portfolio_return,
                    "turnover": turnover,
                }
            )
    return pd.DataFrame(detail_rows)


def summarize_dynamic_portfolio(
    detail_frame: pd.DataFrame,
    pgr_monthly_returns: pd.Series,
) -> dict[str, float]:
    """Summarize the historical behavior of one dynamic redeploy sleeve."""
    if detail_frame.empty:
        return {
            "annualized_return": float("nan"),
            "annualized_vol": float("nan"),
            "corr_to_pgr": float("nan"),
            "max_drawdown": float("nan"),
            "mean_turnover": float("nan"),
            "n_months": 0.0,
        }

    portfolio_series = (
        detail_frame[["date", "portfolio_return_1m"]]
        .drop_duplicates()
        .set_index("date")["portfolio_return_1m"]
        .sort_index()
    )
    aligned_pgr = pgr_monthly_returns.reindex(portfolio_series.index)
    cumulative = (1.0 + portfolio_series).cumprod()
    max_drawdown = float((cumulative / cumulative.cummax() - 1.0).min())
    annualized_return = float((1.0 + portfolio_series).prod() ** (12.0 / len(portfolio_series)) - 1.0)
    annualized_vol = float(portfolio_series.std(ddof=1) * np.sqrt(12.0))
    return {
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "corr_to_pgr": float(portfolio_series.corr(aligned_pgr)),
        "max_drawdown": max_drawdown,
        "mean_turnover": float(detail_frame["turnover"].mean()),
        "n_months": float(len(portfolio_series)),
    }
