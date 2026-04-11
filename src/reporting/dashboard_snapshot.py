"""Static monthly dashboard snapshot writer."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _format_pct(value: float | int | None, *, scale: float = 100.0, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    numeric = float(value) * scale
    sign = "+" if numeric >= 0 else ""
    return f"{sign}{numeric:.{decimals}f}%"


def _format_number(value: float | int | None, *, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}"


def _render_table(title: str, df: pd.DataFrame) -> str:
    if df.empty:
        return (
            f"<section><h2>{escape(title)}</h2>"
            "<p class='muted'>No data available for this section.</p></section>"
        )
    return (
        f"<section><h2>{escape(title)}</h2>"
        f"{df.to_html(index=False, classes='data-table', border=0, justify='left')}"
        "</section>"
    )


def _prepare_signal_table(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    df = signals.copy()
    keep = [
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
    df = df[[column for column in keep if column in df.columns]]
    rename_map = {
        "benchmark": "Benchmark",
        "signal": "Signal",
        "predicted_relative_return": "Predicted Return",
        "ic": "IC",
        "hit_rate": "Hit Rate",
        "confidence_tier": "Confidence",
        "calibrated_prob_outperform": "P(Outperform)",
        "classifier_prob_actionable_sell": "P(Actionable Sell)",
        "classifier_shadow_tier": "Cls Tier",
    }
    df = df.rename(columns=rename_map)
    if "Predicted Return" in df.columns:
        df["Predicted Return"] = df["Predicted Return"].apply(_format_pct)
    if "IC" in df.columns:
        df["IC"] = df["IC"].apply(_format_number)
    if "Hit Rate" in df.columns:
        df["Hit Rate"] = df["Hit Rate"].apply(lambda value: _format_pct(value, decimals=1))
    if "P(Outperform)" in df.columns:
        df["P(Outperform)"] = df["P(Outperform)"].apply(lambda value: _format_pct(value, decimals=1))
    if "P(Actionable Sell)" in df.columns:
        df["P(Actionable Sell)"] = df["P(Actionable Sell)"].apply(
            lambda value: _format_pct(value, decimals=1)
        )
    return df


def _prepare_benchmark_quality_table(benchmark_quality_df: pd.DataFrame | None) -> pd.DataFrame:
    if benchmark_quality_df is None or benchmark_quality_df.empty:
        return pd.DataFrame()
    df = benchmark_quality_df.copy()
    keep = ["benchmark", "oos_r2", "nw_ic", "hit_rate", "cw_t_stat", "cw_p_value"]
    df = df[[column for column in keep if column in df.columns]]
    rename_map = {
        "benchmark": "Benchmark",
        "oos_r2": "OOS R^2",
        "nw_ic": "NW IC",
        "hit_rate": "Hit Rate",
        "cw_t_stat": "CW t-stat",
        "cw_p_value": "CW p-value",
    }
    df = df.rename(columns=rename_map)
    if "OOS R^2" in df.columns:
        df["OOS R^2"] = df["OOS R^2"].apply(_format_pct)
    if "NW IC" in df.columns:
        df["NW IC"] = df["NW IC"].apply(_format_number)
    if "Hit Rate" in df.columns:
        df["Hit Rate"] = df["Hit Rate"].apply(lambda value: _format_pct(value, decimals=1))
    if "CW t-stat" in df.columns:
        df["CW t-stat"] = df["CW t-stat"].apply(_format_number)
    if "CW p-value" in df.columns:
        df["CW p-value"] = df["CW p-value"].apply(_format_number)
    return df


def write_dashboard_snapshot(
    out_dir: Path,
    *,
    as_of_date: str,
    recommendation_mode: str,
    consensus: str,
    sell_pct: float,
    mean_predicted: float,
    mean_ic: float,
    mean_hit_rate: float,
    aggregate_oos_r2: float | None,
    recommendation_layer_label: str,
    warnings: list[str],
    signals: pd.DataFrame,
    benchmark_quality_df: pd.DataFrame | None,
    consensus_shadow_df: pd.DataFrame | None,
    classification_shadow_summary: dict[str, object] | None = None,
) -> Path:
    """Write a static HTML snapshot summarizing the latest monthly decision."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dashboard.html"

    signal_table = _prepare_signal_table(signals)
    benchmark_quality_table = _prepare_benchmark_quality_table(benchmark_quality_df)
    del consensus_shadow_df

    classification_section = ""
    if isinstance(classification_shadow_summary, dict) and classification_shadow_summary.get("enabled"):
        probability = escape(
            str(
                classification_shadow_summary.get(
                    "probability_actionable_sell_label",
                    "-",
                )
            )
        )
        tier = escape(str(classification_shadow_summary.get("confidence_tier", "-")))
        stance = escape(str(classification_shadow_summary.get("stance", "-")))
        agreement = escape(str(classification_shadow_summary.get("agreement_label", "-")))
        interpretation = escape(
            str(classification_shadow_summary.get("interpretation", "-"))
        )
        classification_section = (
            "<section>"
            "<h2>Classification Confidence Check</h2>"
            "<p class='muted'>"
            "Shadow-only interpretation layer from the v87-v96 classifier research. "
            "It does not change the live recommendation or sell percentage."
            "</p>"
            "<div class='cards'>"
            f"<div class='card'><div class='label'>P(Actionable Sell)</div><div class='value'>{probability}</div></div>"
            f"<div class='card'><div class='label'>Confidence Tier</div><div class='value'>{tier}</div></div>"
            f"<div class='card'><div class='label'>Classifier Stance</div><div class='value'>{stance}</div></div>"
            f"<div class='card'><div class='label'>Agreement</div><div class='value'>{agreement}</div></div>"
            "</div>"
            f"<p class='muted' style='margin-top:14px;'>{interpretation}</p>"
            "</section>"
        )

    warnings_html = (
        "<ul>" + "".join(f"<li>{escape(warning)}</li>" for warning in warnings) + "</ul>"
        if warnings
        else "<p class='muted'>No workflow warnings for this run.</p>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PGR Monthly Dashboard - {escape(as_of_date)}</title>
  <style>
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      background: linear-gradient(180deg, #f4f7fb 0%, #eef3f8 100%);
      color: #122033;
      margin: 0;
      padding: 24px;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
    }}
    .hero {{
      background: #ffffff;
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(18, 32, 51, 0.08);
      margin-bottom: 20px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .card {{
      background: #f7fafc;
      border: 1px solid #d8e3ef;
      border-radius: 16px;
      padding: 14px 16px;
    }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #4f6780;
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    section {{
      background: #ffffff;
      border-radius: 20px;
      padding: 22px 24px;
      box-shadow: 0 10px 30px rgba(18, 32, 51, 0.06);
      margin-bottom: 20px;
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    .muted {{
      color: #596f86;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .data-table th,
    .data-table td {{
      border-bottom: 1px solid #e2e8f0;
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      background: #f7fafc;
    }}
    code {{
      background: #eef3f8;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <p class="label">Monthly Decision Dashboard</p>
      <h1>PGR Monthly Snapshot</h1>
      <p class="muted">
        Static monthly dashboard generated from the committed artifact bundle.
        This snapshot is intended for email linking and lightweight review
        without a live Streamlit process.
      </p>
      <p><strong>As-of Date:</strong> {escape(as_of_date)}<br>
      <strong>Recommendation Layer:</strong> {escape(recommendation_layer_label)}<br>
      <strong>Recommendation Mode:</strong> {escape(recommendation_mode)}</p>
      <div class="cards">
        <div class="card"><div class="label">Consensus</div><div class="value">{escape(consensus)}</div></div>
        <div class="card"><div class="label">Sell %</div><div class="value">{_format_pct(sell_pct, decimals=0)}</div></div>
        <div class="card"><div class="label">Predicted 6M Return</div><div class="value">{_format_pct(mean_predicted)}</div></div>
        <div class="card"><div class="label">Mean IC</div><div class="value">{_format_number(mean_ic)}</div></div>
        <div class="card"><div class="label">Mean Hit Rate</div><div class="value">{_format_pct(mean_hit_rate, decimals=1)}</div></div>
        <div class="card"><div class="label">Aggregate OOS R^2</div><div class="value">{_format_pct(aggregate_oos_r2)}</div></div>
      </div>
    </div>

    <section>
      <h2>Workflow Warnings</h2>
      {warnings_html}
    </section>

    {_render_table("Benchmark Quality", benchmark_quality_table)}
    {classification_section}
    {_render_table("Per-Benchmark Signals", signal_table)}

    <section>
      <h2>Artifact Notes</h2>
      <p class="muted">
        Source files for this snapshot live alongside this HTML file in the same
        monthly folder:
        <code>recommendation.md</code>,
        <code>diagnostic.md</code>,
        <code>signals.csv</code>,
        <code>benchmark_quality.csv</code>,
        <code>consensus_shadow.csv</code>, and
        <code>monthly_summary.json</code>, and
        <code>run_manifest.json</code>.
      </p>
      <p class="muted">
        The equal-weight comparison remains available in
        <code>consensus_shadow.csv</code> for diagnostics, but it is no longer
        promoted as a primary decision surface in the monthly snapshot.
      </p>
    </section>
  </div>
</body>
</html>
"""

    path.write_text(html, encoding="utf-8")
    return path
