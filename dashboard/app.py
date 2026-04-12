"""PGR Vesting Decision Support - Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py

Reads from:
- results/monthly_decisions/ (manifest, markdown, and CSV artifacts)
- data/pgr_financials.db     (daily_prices for the PGR price chart)
"""

from __future__ import annotations

import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    from dashboard.data import (
        DB_PATH,
        load_decision_history as load_decision_history_data,
        load_latest_run_bundle,
        load_pgr_prices as load_pgr_prices_data,
        parse_aggregate_health,
        parse_recommendation_summary,
        top_level_summary_fields,
    )
except ImportError:  # pragma: no cover - local Streamlit execution fallback
    from data import (  # type: ignore
        DB_PATH,
        load_decision_history as load_decision_history_data,
        load_latest_run_bundle,
        load_pgr_prices as load_pgr_prices_data,
        parse_aggregate_health,
        parse_recommendation_summary,
        top_level_summary_fields,
    )


@st.cache_data(ttl=300)
def load_latest_run() -> dict:
    """Load the latest monthly artifact bundle."""
    return load_latest_run_bundle()


@st.cache_data(ttl=300)
def load_decision_history() -> pd.DataFrame:
    """Load decision history from the committed monthly log."""
    return load_decision_history_data()


@st.cache_data(ttl=300)
def load_pgr_prices(n_days: int = 365) -> pd.DataFrame:
    """Load recent PGR prices for the price chart."""
    return load_pgr_prices_data(n_days=n_days)


def _format_pct(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def _summary_lookup(summary: dict | None, *keys: str) -> object | None:
    current: object | None = summary
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _style_signal_rows(row: pd.Series) -> list[str]:
    signal = str(row.get("Signal", ""))
    if "OUTPERFORM" in signal:
        background = "background-color: #d4edda"
    elif "UNDERPERFORM" in signal:
        background = "background-color: #f8d7da"
    else:
        background = ""
    return [background] * len(row)


def _format_signal_table(signals: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "benchmark",
        "signal",
        "predicted_relative_return",
        "ic",
        "hit_rate",
        "confidence_tier",
        "calibrated_prob_outperform",
        "classifier_prob_actionable_sell",
        "classifier_shadow_tier",
    ]
    available = [column for column in display_cols if column in signals.columns]
    df_display = signals[available].copy()

    for column in ["predicted_relative_return", "ic"]:
        if column in df_display.columns:
            df_display[column] = df_display[column].apply(lambda value: f"{value * 100:+.2f}%")
    for column in [
        "hit_rate",
        "calibrated_prob_outperform",
        "classifier_prob_actionable_sell",
    ]:
        if column in df_display.columns:
            df_display[column] = df_display[column].apply(lambda value: f"{value * 100:.1f}%")

    return df_display.rename(
        columns={
            "benchmark": "Benchmark",
            "signal": "Signal",
            "predicted_relative_return": "Pred. Return",
            "ic": "IC",
            "hit_rate": "Hit Rate",
            "confidence_tier": "Confidence",
            "calibrated_prob_outperform": "P(Outperform)",
            "classifier_prob_actionable_sell": "P(Actionable Sell)",
            "classifier_shadow_tier": "Cls Tier",
        }
    )


def _format_shadow_table(consensus_shadow: pd.DataFrame) -> pd.DataFrame:
    df = consensus_shadow.copy()
    rename_map = {
        "variant": "Variant",
        "consensus": "Consensus",
        "mean_predicted_return": "Mean Pred. Return",
        "mean_ic": "Mean IC",
        "mean_hit_rate": "Mean Hit Rate",
        "mean_prob_outperform": "P(Outperform)",
        "recommendation_mode": "Mode",
        "recommended_sell_pct": "Sell %",
        "top_benchmark": "Top Benchmark",
        "top_benchmark_weight": "Top Weight",
        "is_live_path": "Live Path",
    }
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})
    for column in ["Mean Pred. Return", "Mean Hit Rate", "P(Outperform)", "Sell %", "Top Weight"]:
        if column in df.columns:
            df[column] = df[column].apply(lambda value: f"{value * 100:.1f}%" if pd.notna(value) else "-")
    if "Mean IC" in df.columns:
        df["Mean IC"] = df["Mean IC"].apply(lambda value: f"{value:.4f}" if pd.notna(value) else "-")
    if "Live Path" in df.columns:
        df["Live Path"] = df["Live Path"].map({True: "Live", False: "Shadow"}).fillna("-")
    return df


def _format_benchmark_quality_table(benchmark_quality: pd.DataFrame) -> pd.DataFrame:
    df = benchmark_quality.copy()
    rename_map = {
        "benchmark": "Benchmark",
        "oos_r2": "OOS R^2",
        "nw_ic": "NW IC",
        "hit_rate": "Hit Rate",
        "cw_t_stat": "CW t-stat",
        "cw_p_value": "CW p-value",
    }
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})
    if "OOS R^2" in df.columns:
        df["OOS R^2"] = df["OOS R^2"].apply(lambda value: f"{value * 100:+.2f}%" if pd.notna(value) else "-")
    if "NW IC" in df.columns:
        df["NW IC"] = df["NW IC"].apply(lambda value: f"{value:.4f}" if pd.notna(value) else "-")
    if "Hit Rate" in df.columns:
        df["Hit Rate"] = df["Hit Rate"].apply(lambda value: f"{value * 100:.1f}%" if pd.notna(value) else "-")
    for column in ["CW t-stat", "CW p-value"]:
        if column in df.columns:
            df[column] = df[column].apply(lambda value: f"{value:.4f}" if pd.notna(value) else "-")
    return df


st.set_page_config(
    page_title="PGR Vesting Decision Support",
    page_icon="📈",
    layout="wide",
)

st.title("PGR Vesting Decision Support")

bundle = load_latest_run()
manifest = bundle["manifest"]
signals = bundle["signals"]
rec_text = bundle["recommendation_text"]
benchmark_quality = bundle["benchmark_quality"]
consensus_shadow = bundle["consensus_shadow"]
decision_overlays = bundle["decision_overlays"]
summary_payload = bundle["summary"]

if manifest is None and signals.empty:
    st.error("No monthly decision data found in results/monthly_decisions/.")
    st.stop()

health = parse_aggregate_health(rec_text)
summary = parse_recommendation_summary(rec_text)
top_level = top_level_summary_fields(summary_payload)
as_of = (
    _summary_lookup(summary_payload, "as_of_date")
    or (manifest.get("as_of_date", "-") if manifest else "-")
)
signal_display = _summary_lookup(summary_payload, "recommendation", "signal_label") or summary["signal"] or "-"
sell_display = _summary_lookup(summary_payload, "recommendation", "recommended_sell_pct_label") or summary["sell_pct"] or "-"
predicted_display = (
    _summary_lookup(summary_payload, "recommendation", "predicted_6m_relative_return_label")
    or summary["predicted_return"]
    or "-"
)
oos_display = (
    _summary_lookup(summary_payload, "recommendation", "aggregate_oos_r2_label")
    or _format_pct(health["oos_r2"])
)

cols = st.columns(5)
cols[0].metric("As-of Date", as_of)
cols[1].metric("Signal", str(signal_display))
cols[2].metric("Recommended Sell %", str(sell_display))
cols[3].metric("Predicted 6M Return", str(predicted_display))
cols[4].metric("Aggregate OOS R^2", str(oos_display))

st.divider()

tab_current, tab_history, tab_health, tab_freshness = st.tabs(
    ["Current Decision", "History", "Model Health", "Data Freshness"]
)

with tab_current:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        if top_level["decision_headline"] or top_level["hold_vs_sell_label"]:
            st.subheader("Decision At A Glance")
            if top_level["decision_headline"]:
                st.info(str(top_level["decision_headline"]))
            if top_level["hold_vs_sell_label"] or top_level["actionability_label"]:
                st.caption(
                    " | ".join(
                        part
                        for part in [
                            top_level["hold_vs_sell_label"],
                            top_level["actionability_label"],
                        ]
                        if part
                    )
                )

        st.subheader("Per-Benchmark Signals")
        if not signals.empty:
            st.dataframe(
                _format_signal_table(signals).style.apply(_style_signal_rows, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No signals data available for the latest run.")

        cross_check_note = _summary_lookup(summary_payload, "cross_check", "retired_reason")
        if cross_check_note:
            st.info(
                "Consensus cross-check retired from the main dashboard surface. "
                f"{cross_check_note}"
            )

        classification_shadow = _summary_lookup(summary_payload, "classification_shadow")
        if isinstance(classification_shadow, dict) and classification_shadow.get("enabled"):
            st.subheader("Classification Confidence Check")
            st.caption(
                "Shadow-only interpretation layer from the v87-v96 classifier research. "
                "It does not change the live recommendation."
            )
            cls_cols = st.columns(4)
            cls_cols[0].metric(
                "P(Actionable Sell)",
                str(classification_shadow.get("probability_actionable_sell_label", "-")),
            )
            cls_cols[1].metric(
                "Confidence Tier",
                str(classification_shadow.get("confidence_tier", "-")),
            )
            cls_cols[2].metric(
                "Classifier Stance",
                str(classification_shadow.get("stance", "-")),
            )
            cls_cols[3].metric(
                "Agreement",
                str(classification_shadow.get("agreement_label", "-")),
            )
            interpretation = classification_shadow.get("interpretation")
            if interpretation:
                st.info(str(interpretation))

        overlay = _summary_lookup(summary_payload, "shadow_gate_overlay")
        if isinstance(summary_payload, dict):
            overlay = summary_payload.get("shadow_gate_overlay")
        if not isinstance(overlay, dict) and not decision_overlays.empty:
            shadow_rows = decision_overlays[
                decision_overlays["variant"].astype(str) == "shadow_gate"
            ]
            if not shadow_rows.empty:
                overlay = shadow_rows.iloc[0].to_dict()
        if isinstance(overlay, dict):
            st.subheader("Shadow Gate Overlay")
            st.caption(
                "Shadow-only classifier gate candidate for future promotion review."
            )
            overlay_cols = st.columns(4)
            overlay_cols[0].metric(
                "Mode",
                str(overlay.get("recommendation_mode", "-")),
            )
            overlay_cols[1].metric(
                "Sell %",
                f"{float(overlay.get('recommended_sell_pct', 0.0)):.0%}"
                if overlay.get("recommended_sell_pct") is not None
                else "-",
            )
            overlay_cols[2].metric(
                "Would Change",
                "Yes" if overlay.get("would_change") else "No",
            )
            overlay_cols[3].metric(
                "Reason",
                str(overlay.get("reason", "-")),
            )

    with col_right:
        st.subheader("PGR Price (1 Year)")
        prices = load_pgr_prices(365)
        if not prices.empty:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(prices["date"], prices["close"], color="#1f77b4", linewidth=1.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Price ($)", fontsize=9)
            ax.set_title("PGR Daily Close", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Latest: ${prices['close'].iloc[-1]:.2f} on {prices['date'].iloc[-1].date()}")
        else:
            st.info("No price data available.")

        if manifest:
            warnings = manifest.get("warnings", [])
            if warnings:
                st.subheader("Model Warnings")
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success("No model quality warnings.")

with tab_history:
    st.subheader("Decision Log")
    history = load_decision_history()
    if history.empty:
        st.info("No decision history found.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        return_col = "Predicted 6M Return"
        date_col = "As-Of Date"
        if return_col in history.columns and date_col in history.columns:
            st.subheader("Predicted 6M Return Over Time")
            hist_plot = history[[date_col, return_col, "Consensus Signal"]].copy()
            hist_plot[date_col] = pd.to_datetime(hist_plot[date_col], errors="coerce")
            hist_plot[return_col] = (
                hist_plot[return_col]
                .str.replace("%", "", regex=False)
                .str.replace("+", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
            hist_plot = hist_plot.dropna(subset=[date_col, return_col])
            if len(hist_plot) >= 2:
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = hist_plot["Consensus Signal"].map(
                    {"OUTPERFORM": "#2ca02c", "UNDERPERFORM": "#d62728", "NEUTRAL": "#7f7f7f"}
                ).fillna("#aec7e8")
                ax.bar(hist_plot[date_col], hist_plot[return_col], color=colors, width=15)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
                plt.xticks(rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("Pred. Return (%)", fontsize=9)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Green = OUTPERFORM, Red = UNDERPERFORM, Gray = NEUTRAL")

with tab_health:
    st.subheader("Aggregate Model Metrics")
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "OOS R^2",
        str(_summary_lookup(summary_payload, "recommendation", "aggregate_oos_r2_label") or _format_pct(health["oos_r2"])),
    )
    metric_cols[1].metric(
        "Mean IC",
        str(
            _summary_lookup(summary_payload, "recommendation", "mean_ic_label")
            or (f"{health['ic']:.4f}" if health["ic"] is not None else "-")
        ),
    )
    metric_cols[2].metric(
        "Hit Rate",
        str(
            _summary_lookup(summary_payload, "recommendation", "mean_hit_rate_label")
            or _format_pct(health["hit_rate"], 1)
        ),
    )
    metric_cols[3].metric("Calibration ECE", summary["calibration_ece"] or "-")

    if not signals.empty and "confidence_tier" in signals.columns:
        st.subheader("Confidence Tier Distribution")
        tier_counts = signals["confidence_tier"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 2.5))
        tier_order = [tier for tier in ["HIGH", "MODERATE", "LOW"] if tier in tier_counts.index]
        tier_vals = [tier_counts.get(tier, 0) for tier in tier_order]
        tier_colors = {"HIGH": "#d62728", "MODERATE": "#ff7f0e", "LOW": "#aec7e8"}
        ax.barh(
            tier_order,
            tier_vals,
            color=[tier_colors.get(tier, "#aec7e8") for tier in tier_order],
        )
        ax.set_xlabel("# Benchmarks", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Benchmark Quality Snapshot")
    if not benchmark_quality.empty:
        cw_passes = int((benchmark_quality["cw_p_value"] < 0.05).sum()) if "cw_p_value" in benchmark_quality else 0
        positive_ic = int((benchmark_quality["nw_ic"] > 0).sum()) if "nw_ic" in benchmark_quality else 0
        best_row = (
            benchmark_quality.sort_values("nw_ic", ascending=False).iloc[0]
            if "nw_ic" in benchmark_quality and not benchmark_quality.empty
            else None
        )
        quality_cols = st.columns(3)
        quality_cols[0].metric("CW passes (p < 0.05)", f"{cw_passes}/{len(benchmark_quality)}")
        quality_cols[1].metric("Positive benchmark IC", f"{positive_ic}/{len(benchmark_quality)}")
        quality_cols[2].metric(
            "Best benchmark NW IC",
            f"{best_row['benchmark']} ({best_row['nw_ic']:.4f})" if best_row is not None else "-",
        )
        st.dataframe(
            _format_benchmark_quality_table(benchmark_quality),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No benchmark quality artifact available for the latest run.")

    if not consensus_shadow.empty:
        st.caption(
            "The equal-weight comparison is still written to `consensus_shadow.csv` "
            "for diagnostic review, but it is no longer shown as a primary decision surface."
        )

with tab_freshness:
    st.subheader("Data Freshness")
    if manifest:
        latest_dates = manifest.get("latest_dates", {})
        row_counts = manifest.get("row_counts", {})
        run_ts = manifest.get("run_timestamp_utc", "-")
        schema = manifest.get("schema_version", "-")

        st.caption(f"Report generated at {run_ts} | Schema: `{schema}`")
        if latest_dates:
            fresh_df = pd.DataFrame(
                [
                    {
                        "Data Source": source,
                        "Latest Date": latest_date,
                        "Row Count": row_counts.get(source.replace(".", "_").split(".")[0], "-"),
                    }
                    for source, latest_date in sorted(latest_dates.items())
                ]
            )
            st.dataframe(fresh_df, use_container_width=True, hide_index=True)
        st.caption(f"Git SHA: `{manifest.get('git_sha', '-')}`")
    else:
        st.info("No manifest available for the latest run.")

    st.subheader("Database Row Counts")
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                conn,
            )["name"].tolist()
            counts = []
            for table in tables:
                count = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", conn)["n"].iloc[0]
                counts.append({"Table": table, "Rows": count})
            st.dataframe(pd.DataFrame(counts), use_container_width=True, hide_index=True)
        finally:
            conn.close()
    else:
        st.warning(f"Database not found at {DB_PATH}")

st.caption("PGR Vesting Decision Support - jhester599/pgr-vesting-decision-support")
