"""PGR Vesting Decision Support — Streamlit Dashboard.

Run with:
    streamlit run dashboard/app.py

Reads from:
- results/monthly_decisions/  (run_manifest.json, signals.csv, recommendation.md)
- data/pgr_financials.db       (daily_prices for PGR price chart)
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DECISIONS_DIR = BASE_DIR / "results" / "monthly_decisions"
DB_PATH = BASE_DIR / "data" / "pgr_financials.db"
DECISION_LOG = DECISIONS_DIR / "decision_log.md"

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def _monthly_dirs() -> list[Path]:
    return sorted(
        [d for d in DECISIONS_DIR.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}$", d.name)],
        reverse=True,
    )


@st.cache_data(ttl=300)
def load_latest_run() -> tuple[dict | None, pd.DataFrame | None, str | None]:
    """Return (manifest, signals_df, recommendation_text) for the most recent run."""
    dirs = _monthly_dirs()
    if not dirs:
        return None, None, None

    latest = dirs[0]
    manifest = None
    manifest_path = latest / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    signals: pd.DataFrame | None = None
    signals_path = latest / "signals.csv"
    if signals_path.exists():
        signals = pd.read_csv(signals_path)

    rec_text: str | None = None
    rec_path = latest / "recommendation.md"
    if rec_path.exists():
        rec_text = rec_path.read_text()

    return manifest, signals, rec_text


@st.cache_data(ttl=300)
def load_decision_history() -> pd.DataFrame:
    """Parse decision_log.md markdown table into a DataFrame."""
    if not DECISION_LOG.exists():
        return pd.DataFrame()

    text = DECISION_LOG.read_text()
    lines = [ln.strip() for ln in text.splitlines() if "|" in ln and "---" not in ln]

    rows: list[list[str]] = []
    header: list[str] | None = None
    for line in lines:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if header is None:
            header = cells
        else:
            if len(cells) == len(header):
                rows.append(cells)

    if not rows or header is None:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header)
    # Rename columns defensively
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)
    return df


@st.cache_data(ttl=300)
def load_pgr_prices(n_days: int = 365) -> pd.DataFrame:
    """Load recent PGR daily close prices from SQLite."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT date, close FROM daily_prices WHERE ticker='PGR'"
            " ORDER BY date DESC LIMIT ?",
            conn,
            params=(n_days,),
        )
    finally:
        conn.close()

    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_aggregate_health(rec_text: str) -> dict[str, float | None]:
    """Extract aggregate OOS R², IC, hit rate from recommendation text."""
    m = re.search(
        r"OOS R\^2\s+([+-]?\d+\.?\d*)%.*?IC\s+([-\d.]+).*?hit rate\s+([\d.]+)%",
        rec_text or "",
    )
    if m:
        return {
            "oos_r2": float(m.group(1)),
            "ic": float(m.group(2)),
            "hit_rate": float(m.group(3)),
        }
    return {"oos_r2": None, "ic": None, "hit_rate": None}


def _signal_color(signal: str) -> str:
    colors = {"OUTPERFORM": "green", "UNDERPERFORM": "red", "NEUTRAL": "gray"}
    return colors.get(signal.upper(), "gray")


def _confidence_badge(tier: str) -> str:
    badges = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "⚪"}
    return badges.get(tier.upper(), "")


def _format_pct(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PGR Vesting Decision Support",
    page_icon="📈",
    layout="wide",
)

st.title("📈 PGR Vesting Decision Support")

manifest, signals, rec_text = load_latest_run()

if manifest is None and signals is None:
    st.error("No monthly decision data found in results/monthly_decisions/.")
    st.stop()

# ---------------------------------------------------------------------------
# Key metrics bar
# ---------------------------------------------------------------------------

as_of = manifest.get("as_of_date", "—") if manifest else "—"
health = _parse_aggregate_health(rec_text or "")

# Derive consensus from signals
consensus_signal = "—"
sell_pct = "—"
predicted_return_str = "—"
if signals is not None and not signals.empty:
    outperform = (signals["signal"] == "OUTPERFORM").sum()
    underperform = (signals["signal"] == "UNDERPERFORM").sum()
    total = len(signals)
    if outperform > total / 2:
        consensus_signal = "OUTPERFORM"
    elif underperform > total / 2:
        consensus_signal = "UNDERPERFORM"
    else:
        consensus_signal = "NEUTRAL"
    predicted_return_str = _format_pct(
        signals["predicted_relative_return"].mean() * 100
    )

# Try to parse sell % and signal from recommendation.md
if rec_text:
    m_sell = re.search(r"Recommended Sell %.*?\*\*(\d+)%\*\*", rec_text)
    if m_sell:
        sell_pct = f"{m_sell.group(1)}%"
    m_sig = re.search(r"Signal.*?\*\*([A-Z ]+(?:\(.*?\))?)\*\*", rec_text)
    if m_sig:
        consensus_signal = m_sig.group(1).strip()

cols = st.columns(5)
cols[0].metric("As-of Date", as_of)
cols[1].metric("Signal", consensus_signal)
cols[2].metric("Recommended Sell %", sell_pct)
cols[3].metric("Predicted 6M Return", predicted_return_str)
oos_str = _format_pct(health["oos_r2"]) if health["oos_r2"] is not None else "—"
cols[4].metric("Aggregate OOS R²", oos_str)

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_current, tab_history, tab_health, tab_freshness = st.tabs(
    ["📋 Current Decision", "📅 History", "🔬 Model Health", "🗄️ Data Freshness"]
)

# ── Tab 1: Current Decision ─────────────────────────────────────────────────
with tab_current:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Per-Benchmark Signals")
        if signals is not None and not signals.empty:
            display_cols = [
                "benchmark",
                "signal",
                "predicted_relative_return",
                "ic",
                "hit_rate",
                "confidence_tier",
                "calibrated_prob_outperform",
            ]
            available = [c for c in display_cols if c in signals.columns]
            df_display = signals[available].copy()

            # Format numeric columns
            for col in ["predicted_relative_return", "ic"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x*100:+.2f}%"
                    )
            for col in ["hit_rate", "calibrated_prob_outperform"]:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x*100:.1f}%"
                    )

            col_labels = {
                "benchmark": "Benchmark",
                "signal": "Signal",
                "predicted_relative_return": "Pred. Return",
                "ic": "IC",
                "hit_rate": "Hit Rate",
                "confidence_tier": "Confidence",
                "calibrated_prob_outperform": "P(Outperform)",
            }
            df_display = df_display.rename(
                columns={k: v for k, v in col_labels.items() if k in df_display.columns}
            )

            def _row_color(row: pd.Series) -> list[str]:
                sig = row.get("Signal", "")
                if "OUTPERFORM" in str(sig):
                    bg = "background-color: #d4edda"
                elif "UNDERPERFORM" in str(sig):
                    bg = "background-color: #f8d7da"
                else:
                    bg = ""
                return [bg] * len(row)

            styled = df_display.style.apply(_row_color, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("No signals data available for the latest run.")

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
            latest_price = prices["close"].iloc[-1]
            st.caption(
                f"Latest: **${latest_price:.2f}** on {prices['date'].iloc[-1].date()}"
            )
        else:
            st.info("No price data available.")

        # Model warnings
        if manifest:
            warnings = manifest.get("warnings", [])
            if warnings:
                st.subheader("Model Warnings")
                for w in warnings:
                    st.warning(w)
            else:
                st.success("No model quality warnings.")

# ── Tab 2: History ──────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Decision Log")
    history = load_decision_history()
    if history.empty:
        st.info("No decision history found.")
    else:
        # Display full log table
        st.dataframe(history, use_container_width=True, hide_index=True)

        # Plot predicted return over time if columns present
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
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                colors = hist_plot["Consensus Signal"].map(
                    {"OUTPERFORM": "#2ca02c", "UNDERPERFORM": "#d62728", "NEUTRAL": "#7f7f7f"}
                ).fillna("#aec7e8")
                ax2.bar(hist_plot[date_col], hist_plot[return_col], color=colors, width=15)
                ax2.axhline(0, color="black", linewidth=0.8)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
                plt.xticks(rotation=30, ha="right", fontsize=8)
                ax2.set_ylabel("Pred. Return (%)", fontsize=9)
                ax2.grid(axis="y", linestyle="--", alpha=0.4)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                st.caption(
                    "Green = OUTPERFORM, Red = UNDERPERFORM, Gray = NEUTRAL"
                )

# ── Tab 3: Model Health ─────────────────────────────────────────────────────
with tab_health:
    st.subheader("Aggregate Model Metrics")

    m_col1, m_col2, m_col3 = st.columns(3)
    oos_r2 = health["oos_r2"]
    ic_val = health["ic"]
    hr_val = health["hit_rate"]
    m_col1.metric(
        "OOS R²",
        _format_pct(oos_r2),
        delta="Pass ≥ 2%" if oos_r2 is not None and oos_r2 >= 2.0 else "Below threshold",
        delta_color="normal" if (oos_r2 or 0) >= 2.0 else "inverse",
    )
    m_col2.metric(
        "Mean IC",
        f"{ic_val:.4f}" if ic_val is not None else "—",
        delta="Pass ≥ 0.07" if ic_val is not None and ic_val >= 0.07 else "Below threshold",
        delta_color="normal" if (ic_val or 0) >= 0.07 else "inverse",
    )
    m_col3.metric(
        "Hit Rate",
        _format_pct(hr_val, 1),
        delta="Pass ≥ 55%" if hr_val is not None and hr_val >= 55.0 else "Below threshold",
        delta_color="normal" if (hr_val or 0) >= 55.0 else "inverse",
    )

    # Confidence distribution
    if signals is not None and not signals.empty and "confidence_tier" in signals.columns:
        st.subheader("Confidence Tier Distribution")
        tier_counts = signals["confidence_tier"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(5, 2.5))
        tier_order = [t for t in ["HIGH", "MODERATE", "LOW"] if t in tier_counts.index]
        tier_vals = [tier_counts.get(t, 0) for t in tier_order]
        tier_colors = {"HIGH": "#d62728", "MODERATE": "#ff7f0e", "LOW": "#aec7e8"}
        ax3.barh(
            tier_order,
            tier_vals,
            color=[tier_colors.get(t, "#aec7e8") for t in tier_order],
        )
        ax3.set_xlabel("# Benchmarks", fontsize=9)
        ax3.grid(axis="x", linestyle="--", alpha=0.4)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # CPCV / ECE from recommendation.md
    if rec_text:
        st.subheader("Quality Gate Summary")
        # Extract ECE
        ece_match = re.search(r"ECE\s*=\s*([\d.]+)%", rec_text)
        cpcv_match = re.search(r"CPCV verdict.*?(PASS|FAIL)", rec_text, re.IGNORECASE)
        obs_match = re.search(
            r"Observation-to-feature.*?(PASS|FAIL).*?ratio=([\d.]+)", rec_text, re.IGNORECASE
        )

        qcols = st.columns(3)
        if ece_match:
            qcols[0].metric("Calibration ECE", f"{ece_match.group(1)}%")
        if cpcv_match:
            verdict = cpcv_match.group(1).upper()
            qcols[1].metric("CPCV Verdict", verdict)
        if obs_match:
            verdict2 = obs_match.group(1).upper()
            ratio = obs_match.group(2)
            qcols[2].metric("Obs/Feature Ratio", ratio, delta=verdict2,
                            delta_color="normal" if verdict2 == "PASS" else "inverse")

# ── Tab 4: Data Freshness ───────────────────────────────────────────────────
with tab_freshness:
    st.subheader("Data Freshness")
    if manifest:
        latest_dates = manifest.get("latest_dates", {})
        row_counts = manifest.get("row_counts", {})
        run_ts = manifest.get("run_timestamp_utc", "—")
        schema = manifest.get("schema_version", "—")

        st.caption(f"Report generated at **{run_ts}** | Schema: `{schema}`")

        if latest_dates:
            fresh_df = pd.DataFrame(
                [
                    {
                        "Data Source": source,
                        "Latest Date": date,
                        "Row Count": row_counts.get(source.replace(".", "_").split(".")[0], "—"),
                    }
                    for source, date in sorted(latest_dates.items())
                ]
            )
            st.dataframe(fresh_df, use_container_width=True, hide_index=True)

        # Git sha
        git_sha = manifest.get("git_sha", "—")
        st.caption(f"Git SHA: `{git_sha}`")
    else:
        st.info("No manifest available for the latest run.")

    # DB table row counts direct from SQLite
    st.subheader("Database Row Counts")
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                conn,
            )["name"].tolist()
            counts = []
            for tbl in tables:
                count = pd.read_sql(f"SELECT COUNT(*) AS n FROM {tbl}", conn)["n"].iloc[0]
                counts.append({"Table": tbl, "Rows": count})
            st.dataframe(pd.DataFrame(counts), use_container_width=True, hide_index=True)
        finally:
            conn.close()
    else:
        st.warning(f"Database not found at {DB_PATH}")

st.caption("PGR Vesting Decision Support — jhester599/pgr-vesting-decision-support")
