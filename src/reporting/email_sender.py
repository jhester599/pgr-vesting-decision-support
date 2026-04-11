"""
Monthly decision email delivery.

Builds a multipart email with:
- a plain-text fallback summary
- an HTML decision memo optimized for desktop and mobile inboxes

The HTML message is intentionally shorter and more action-oriented than the
full markdown artifact, while still preserving the key "what changed" and
per-benchmark detail sections lower in the email.
"""

from __future__ import annotations

import json
import os
import re
import smtplib
import ssl
from datetime import date, datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from pathlib import Path

from src.tax.capital_gains import load_position_lots

_DEFAULT_LOTS_PATH = Path("data/processed/position_lots.csv")


def _clean_markdown_cell(value: str) -> str:
    """Strip simple markdown emphasis from extracted table cells."""
    cleaned = value.strip()
    while cleaned.startswith("**") and cleaned.endswith("**") and len(cleaned) >= 4:
        cleaned = cleaned[2:-2].strip()
    while cleaned.startswith("`") and cleaned.endswith("`") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _extract_report_value(body: str, field: str) -> str | None:
    match = re.search(rf"\| {re.escape(field)} \| \*\*(.+?)\*\* \|", body)
    if match:
        return _clean_markdown_cell(match.group(1))
    match = re.search(rf"\| {re.escape(field)} \| (.+?) \|", body)
    if match:
        return _clean_markdown_cell(match.group(1))
    return None


def _extract_executive_summary(body: str) -> list[str]:
    match = re.search(r"## Executive Summary\s+(.*?)\n---", body, flags=re.S)
    if not match:
        return []
    return [
        line.strip()[2:]
        for line in match.group(1).splitlines()
        if line.strip().startswith("- ")
    ]


def _extract_markdown_tables(text: str) -> list[list[str]]:
    tables: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line.rstrip())
            continue
        if current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _parse_markdown_table(lines: list[str]) -> list[dict[str, str]]:
    if len(lines) < 2:
        return []
    headers = [_clean_markdown_cell(cell) for cell in lines[0].strip().strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [_clean_markdown_cell(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _extract_section(body: str, heading: str) -> str:
    match = re.search(rf"## {re.escape(heading)}\s+(.*?)(?=\n## |\Z)", body, flags=re.S)
    return match.group(1) if match else ""


def _extract_section_bullets(body: str, heading: str) -> list[str]:
    section = _extract_section(body, heading)
    return [
        line.strip()[2:]
        for line in section.splitlines()
        if line.strip().startswith("- ")
    ]


def _extract_table_by_header(body: str, header_name: str) -> list[dict[str, str]]:
    for table_lines in _extract_markdown_tables(body):
        if table_lines and table_lines[0].strip().startswith(f"| {header_name} |"):
            return _parse_markdown_table(table_lines)
    return []


def _extract_next_vest_data(body: str) -> dict[str, str]:
    section = _extract_section(body, "Next Vest Decision")
    if not section:
        section = _extract_section(body, "Per-Benchmark Signals")
    for table_lines in _extract_markdown_tables(section):
        if table_lines and table_lines[0].strip().startswith("| Field | Value |"):
            rows = _parse_markdown_table(table_lines)
            return {
                row.get("Field", ""): row.get("Value", "")
                for row in rows
                if row.get("Field")
            }
    return {}


def _extract_consensus_cross_check(body: str) -> list[dict[str, str]]:
    """Extract the live-vs-cross-check comparison from old or new report layouts."""
    section = _extract_section(body, "Consensus Shadow Evaluation")
    for table_lines in _extract_markdown_tables(section):
        if table_lines and table_lines[0].strip().startswith("| Variant |"):
            rows = _parse_markdown_table(table_lines)
            return [
                {
                    "Label": row.get("Variant", ""),
                    "Mode": row.get("Mode", ""),
                    "Sell %": row.get("Sell %", ""),
                    "Predicted": row.get("Mean Pred. Return", ""),
                    "Details": (
                        f"{row.get('Consensus', '')}; "
                        f"IC {row.get('Mean IC', '')}; "
                        f"Hit {row.get('Mean Hit Rate', '')}; "
                        f"Top {row.get('Top Weight', '')}"
                    ).strip("; "),
                }
                for row in rows
            ]

    section = _extract_section(body, "Simple-Baseline Cross-Check")
    for table_lines in _extract_markdown_tables(section):
        if table_lines and table_lines[0].strip().startswith("| Path |"):
            rows = _parse_markdown_table(table_lines)
            return [
                {
                    "Label": row.get("Path", ""),
                    "Mode": row.get("Recommendation Mode", ""),
                    "Sell %": row.get("Sell %", ""),
                    "Predicted": row.get("Predicted 6M Return", ""),
                    "Details": (
                        f"{row.get('Signal', '')}; "
                        f"OOS R^2 {row.get('Aggregate OOS R^2', '')}"
                    ).strip("; "),
                }
                for row in rows
            ]
    return []


def _extract_redeploy_portfolio_rows(body: str) -> list[dict[str, str]]:
    section = _extract_section(body, "Suggested Redeploy Portfolio")
    for table_lines in _extract_markdown_tables(section):
        if table_lines and table_lines[0].strip().startswith("| Fund |"):
            return _parse_markdown_table(table_lines)
    return []


def _extract_confidence_snapshot_rows(body: str) -> list[dict[str, str]]:
    section = _extract_section(body, "Confidence Snapshot")
    for table_lines in _extract_markdown_tables(section):
        if table_lines and table_lines[0].strip().startswith("| Check |"):
            return _parse_markdown_table(table_lines)
    return []


def _extract_recommendation_layer(body: str) -> str | None:
    match = re.search(r"\*\*Recommendation Layer:\*\*\s+(.+?)\s{2,}", body)
    if match:
        return _clean_markdown_cell(match.group(1))
    return None


def _extract_as_of_date(body: str) -> date | None:
    match = re.search(r"\*\*As-of Date:\*\*\s+(\d{4}-\d{2}-\d{2})", body)
    if not match:
        return None
    return date.fromisoformat(match.group(1))


def _parse_money(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_lot_label(vest_date: date, basis: float) -> str:
    return f"{vest_date.isoformat()} @ ${basis:,.0f}"


def _build_existing_shares_guidance(
    body: str,
    lots_csv_path: str | Path | None = None,
) -> dict | None:
    lots_path = Path(lots_csv_path) if lots_csv_path is not None else _DEFAULT_LOTS_PATH
    if not lots_path.exists():
        return None

    as_of = _extract_as_of_date(body)
    next_vest_data = _extract_next_vest_data(body)
    current_price = _parse_money(next_vest_data.get("Current PGR price"))
    recommendation_mode = _extract_report_value(body, "Recommendation Mode") or "MONITORING-ONLY"
    if as_of is None or current_price is None:
        return None

    lots = load_position_lots(str(lots_path))
    if not lots:
        return None

    loss_lots = []
    ltcg_gain_lots = []
    stcg_gain_lots = []
    for lot in lots:
        gain = current_price - lot.cost_basis_per_share
        item: dict[str, str | float] = {
            "shares": float(lot.shares_remaining or lot.shares),
            "label": _format_lot_label(lot.vest_date, lot.cost_basis_per_share),
            "tax_status": "LTCG" if lot.is_ltcg_eligible(as_of) else "STCG",
        }
        if gain < 0:
            loss_lots.append(item)
        elif lot.is_ltcg_eligible(as_of):
            ltcg_gain_lots.append(item)
        else:
            stcg_gain_lots.append(item)

    def _bucket_summary(items: list[dict[str, str | float]]) -> tuple[float, str]:
        shares = sum(float(item["shares"]) for item in items)
        labels = ", ".join(str(item["label"]) for item in items[:4])
        if len(items) > 4:
            labels += f", +{len(items) - 4} more"
        return shares, labels

    loss_shares, loss_labels = _bucket_summary(loss_lots)
    ltcg_shares, ltcg_labels = _bucket_summary(ltcg_gain_lots)
    stcg_shares, stcg_labels = _bucket_summary(stcg_gain_lots)

    if recommendation_mode == "DEFER-TO-TAX-DEFAULT":
        headline = "For shares already held, avoid a model-led trim today."
    elif recommendation_mode == "MONITORING-ONLY":
        headline = "For shares already held, trim only if you already want to reduce concentration."
    else:
        headline = "For shares already held, any additional trim should still follow tax-aware lot order."

    bullets: list[str] = []
    rows: list[dict[str, str]] = []

    if loss_lots:
        bullets.append(
            f"If trimming the legacy position, sell loss lots first: {loss_shares:.0f} share(s) across {loss_labels}."
        )
        rows.append({
            "Category": "Loss lots first",
            "Shares": f"{loss_shares:.0f}",
            "Tax status": ", ".join(sorted({str(item['tax_status']) for item in loss_lots})),
            "Tranches": loss_labels,
            "Guidance": "Harvest these first if reducing existing holdings.",
        })
    if ltcg_gain_lots:
        bullets.append(
            f"After loss lots, the next trim candidates are LTCG gain lots: {ltcg_shares:.0f} share(s) across {ltcg_labels}."
        )
        rows.append({
            "Category": "LTCG gains next",
            "Shares": f"{ltcg_shares:.0f}",
            "Tax status": "LTCG",
            "Tranches": ltcg_labels,
            "Guidance": "Prefer these before realizing any STCG gains.",
        })
    if stcg_gain_lots:
        bullets.append(
            f"Avoid STCG gain lots unless risk reduction is urgent: {stcg_shares:.0f} share(s) across {stcg_labels}."
        )
        rows.append({
            "Category": "Avoid STCG gains",
            "Shares": f"{stcg_shares:.0f}",
            "Tax status": "STCG",
            "Tranches": stcg_labels,
            "Guidance": "Let these age into LTCG when possible.",
        })

    total_shares = sum((lot.shares_remaining or lot.shares) for lot in lots)
    return {
        "headline": headline,
        "bullets": bullets,
        "rows": rows,
        "total_shares": total_shares,
        "current_price": current_price,
    }


def _recommendation_headline(body: str) -> str:
    next_vest_data = _extract_next_vest_data(body)
    vest_date = next_vest_data.get("Next vest date")
    rsu_type = next_vest_data.get("RSU type")
    action = next_vest_data.get("Suggested default vest action")
    if vest_date and rsu_type and action:
        return f"{vest_date} {rsu_type.title()} vest: {action}."
    sell_pct = _extract_report_value(body, "Recommended Sell %") or "n/a"
    return f"Next vest action: sell {sell_pct}."


def _infer_dashboard_snapshot_url(dashboard_path: Path) -> str | None:
    """Infer a GitHub URL for a committed dashboard snapshot when possible."""
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    ref_name = os.environ.get("GITHUB_REF_NAME", "").strip() or "master"
    if not repo:
        return None
    relative_path = dashboard_path.as_posix()
    return f"https://github.com/{repo}/blob/{ref_name}/{relative_path}"


def _summary_lookup(
    summary_payload: dict[str, object] | None,
    *keys: str,
) -> str | None:
    current: object | None = summary_payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None:
        return None
    return str(current)


def build_email_summary(
    body: str,
    lots_csv_path: str | Path | None = None,
    dashboard_snapshot_label: str | None = None,
    dashboard_snapshot_url: str | None = None,
    summary_payload: dict[str, object] | None = None,
) -> str:
    """Build a concise plaintext decision memo from recommendation.md."""
    signal = (
        _summary_lookup(summary_payload, "recommendation", "signal_label")
        or _extract_report_value(body, "Signal")
        or "UNKNOWN"
    )
    mode = (
        _summary_lookup(summary_payload, "recommendation", "recommendation_mode")
        or _extract_report_value(body, "Recommendation Mode")
        or "MONITORING-ONLY"
    )
    predicted = (
        _summary_lookup(summary_payload, "recommendation", "predicted_6m_relative_return_label")
        or _extract_report_value(body, "Predicted 6M Relative Return")
        or "n/a"
    )
    exec_lines = _extract_executive_summary(body)
    existing_guidance = _build_existing_shares_guidance(body, lots_csv_path)
    cross_check_rows = _extract_consensus_cross_check(body)
    recommendation_layer = (
        _summary_lookup(summary_payload, "recommendation_layer", "label")
        or _extract_recommendation_layer(body)
        or ""
    )
    redeploy_lines = _extract_section_bullets(body, "Redeploy Guidance")
    redeploy_portfolio_rows = _extract_redeploy_portfolio_rows(body)
    confidence_rows = _extract_confidence_snapshot_rows(body)
    classification_shadow = None
    if isinstance(summary_payload, dict):
        raw_classification_shadow = summary_payload.get("classification_shadow")
        if isinstance(raw_classification_shadow, dict):
            classification_shadow = raw_classification_shadow

    lines = [
        "PGR Monthly Decision Summary",
        "",
        f"Recommendation: {_recommendation_headline(body)}",
        f"Model view: {signal}",
        f"Recommendation mode: {mode}",
        f"Predicted 6M relative return: {predicted}",
    ]
    if exec_lines:
        lines += ["", "What's changed:"]
        lines.extend(f"- {line}" for line in exec_lines)
    if existing_guidance:
        lines += ["", "Existing shares already held:"]
        lines.append(existing_guidance["headline"])
        lines.extend(f"- {line}" for line in existing_guidance["bullets"])
    if confidence_rows:
        lines += ["", "Confidence checks:"]
        for row in confidence_rows:
            lines.append(f"- {row.get('Check', '')}: {row.get('Current', '')} / {row.get('Status', '')}")
    if isinstance(classification_shadow, dict) and classification_shadow.get("enabled"):
        lines += ["", "Classification confidence check:"]
        lines.append(
            "- Shadow-only interpretation layer; it does not override the live recommendation."
        )
        lines.append(
            "- P(actionable sell): "
            f"{classification_shadow.get('probability_actionable_sell_label', 'n/a')}"
        )
        lines.append(
            f"- Confidence tier: {classification_shadow.get('confidence_tier', 'n/a')}"
        )
        lines.append(f"- Classifier stance: {classification_shadow.get('stance', 'n/a')}")
        lines.append(
            "- Agreement with live recommendation: "
            f"{classification_shadow.get('agreement_label', 'n/a')}"
        )
        interpretation = classification_shadow.get("interpretation")
        if interpretation:
            lines.append(f"- Interpretation: {interpretation}")
    if cross_check_rows:
        lines += ["", "Consensus cross-check:"]
        if recommendation_layer:
            lines.append(f"- Active layer: {recommendation_layer}")
        for row in cross_check_rows:
            lines.append(
                f"- {row.get('Label', 'Path')}: {row.get('Mode', 'n/a')}, "
                f"sell {row.get('Sell %', 'n/a')}, predicted {row.get('Predicted', 'n/a')}"
            )
            details = row.get("Details", "")
            if details:
                lines.append(f"  {details}")
    if redeploy_lines:
        lines += ["", "If redeploying sold exposure:"]
        lines.extend(f"- {line}" for line in redeploy_lines)
    if redeploy_portfolio_rows:
        lines += ["", "Suggested redeploy portfolio:"]
        for row in redeploy_portfolio_rows:
            lines.append(
                f"- {row.get('Fund', '')}: {row.get('Allocation', '')} "
                f"({row.get('Sleeve', '')}) - {row.get('Relative Signal', '')}"
            )

    lines += [
        "",
        "Full report:",
        body,
    ]
    if dashboard_snapshot_label:
        lines += ["", f"Dashboard snapshot: {dashboard_snapshot_label}"]
        if dashboard_snapshot_url:
            lines.append(f"Dashboard link: {dashboard_snapshot_url}")
    return "\n".join(lines)


def _html_badge(text: str, background: str, color: str = "#ffffff") -> str:
    return (
        f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
        f'background:{background};color:{color};font-size:12px;font-weight:700;'
        f'letter-spacing:0.02em;">{escape(text)}</span>'
    )


def _build_benchmark_html_table(body: str) -> str:
    rows = _extract_table_by_header(body, "Benchmark")
    if not rows:
        return ""
    html_rows = []
    for row in rows:
        signal = row.get("Signal", "")
        signal_bg = {
            "OUTPERFORM": "#d1fae5",
            "UNDERPERFORM": "#fee2e2",
            "NEUTRAL": "#e5e7eb",
        }.get(signal, "#e5e7eb")
        signal_fg = {
            "OUTPERFORM": "#065f46",
            "UNDERPERFORM": "#991b1b",
            "NEUTRAL": "#374151",
        }.get(signal, "#374151")
        ci_range = f"{row.get('CI Lower', 'n/a')} to {row.get('CI Upper', 'n/a')}"
        html_rows.append(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;font-weight:700;'>{escape(row.get('Benchmark', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Benchmark Role', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Description', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Predicted Return', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(ci_range)}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('P(cal)', row.get('P(raw)', 'n/a')))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Confidence', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'><span style='display:inline-block;padding:4px 8px;border-radius:999px;background:{signal_bg};color:{signal_fg};font-weight:700;font-size:12px;'>{escape(signal)}</span></td>"
            "</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Benchmark</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Role</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Description</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Predicted</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>80% CI</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>P(cal)</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Confidence</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Signal</th>"
        "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )


def _build_scenario_html_table(body: str) -> str:
    scenario_rows = _extract_table_by_header(body, "Scenario")
    if not scenario_rows:
        return ""
    html_rows = []
    for row in scenario_rows:
        html_rows.append(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;font-weight:700;'>{escape(row.get('Scenario', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Timing', row.get('Sell Date', '')))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Tax Rate', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Predicted Return', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Probability', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Use when', ''))}</td>"
            "</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Scenario</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Timing</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Tax rate</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Predicted return</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Probability</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Use when</th>"
        "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )


def _build_redeploy_portfolio_html_table(body: str) -> str:
    rows = _extract_redeploy_portfolio_rows(body)
    if not rows:
        return ""
    html_rows = []
    for row in rows:
        html_rows.append(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;font-weight:700;'>{escape(row.get('Fund', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Allocation', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Sleeve', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Why it is included', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('PGR Correlation', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Relative Signal', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('P(Benchmark Beats PGR)', ''))}</td>"
            "</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Fund</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Allocation</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Sleeve</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Why</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Corr</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Signal</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>P(Benchmark>PGR)</th>"
        "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )


def build_email_html(
    body: str,
    lots_csv_path: str | Path | None = None,
    dashboard_snapshot_label: str | None = None,
    dashboard_snapshot_url: str | None = None,
    summary_payload: dict[str, object] | None = None,
) -> str:
    """Build the HTML decision memo."""
    signal = (
        _summary_lookup(summary_payload, "recommendation", "signal_label")
        or _extract_report_value(body, "Signal")
        or "UNKNOWN"
    )
    mode = (
        _summary_lookup(summary_payload, "recommendation", "recommendation_mode")
        or _extract_report_value(body, "Recommendation Mode")
        or "MONITORING-ONLY"
    )
    predicted = (
        _summary_lookup(summary_payload, "recommendation", "predicted_6m_relative_return_label")
        or _extract_report_value(body, "Predicted 6M Relative Return")
        or "n/a"
    )
    mean_ic = (
        _summary_lookup(summary_payload, "recommendation", "mean_ic_label")
        or _extract_report_value(body, "Mean IC (across benchmarks)")
        or "n/a"
    )
    mean_hr = (
        _summary_lookup(summary_payload, "recommendation", "mean_hit_rate_label")
        or _extract_report_value(body, "Mean Hit Rate")
        or "n/a"
    )
    mean_cal = (
        _summary_lookup(summary_payload, "recommendation", "prob_outperform_calibrated_label")
        or _extract_report_value(body, "P(Outperform, calibrated)")
        or "n/a"
    )
    oos_r2 = (
        _summary_lookup(summary_payload, "recommendation", "aggregate_oos_r2_label")
        or _extract_report_value(body, "Aggregate OOS R^2")
        or "n/a"
    )
    executive_summary = _extract_executive_summary(body)
    next_vest_data = _extract_next_vest_data(body)
    existing_guidance = _build_existing_shares_guidance(body, lots_csv_path)
    benchmark_table = _build_benchmark_html_table(body)
    scenario_table = _build_scenario_html_table(body)
    cross_check_rows = _extract_consensus_cross_check(body)
    recommendation_layer = (
        _summary_lookup(summary_payload, "recommendation_layer", "label")
        or _extract_recommendation_layer(body)
        or ""
    )
    redeploy_lines = _extract_section_bullets(body, "Redeploy Guidance")
    redeploy_portfolio_table = _build_redeploy_portfolio_html_table(body)
    confidence_rows = _extract_confidence_snapshot_rows(body)
    classification_shadow = None
    if isinstance(summary_payload, dict):
        raw_classification_shadow = summary_payload.get("classification_shadow")
        if isinstance(raw_classification_shadow, dict):
            classification_shadow = raw_classification_shadow
    mode_badge = {
        "ACTIONABLE": _html_badge("ACTIONABLE", "#0f766e"),
        "MONITORING-ONLY": _html_badge("MONITORING ONLY", "#a16207"),
        "DEFER-TO-TAX-DEFAULT": _html_badge("DEFER TO TAX DEFAULT", "#9a3412"),
    }.get(mode, _html_badge(mode, "#475569"))

    confidence_badges = {
        row.get("Check", ""): _html_badge(
            row.get("Status", "UNKNOWN"),
            {
                "PASS": "#166534",
                "FAIL": "#991b1b",
                "UNKNOWN": "#475569",
            }.get(row.get("Status", "UNKNOWN"), "#475569"),
        )
        for row in confidence_rows
    }

    if mode == "DEFER-TO-TAX-DEFAULT":
        lead_text = (
            f"The model leans {escape(signal.lower())}, but the quality checks still point to the default 50% vest sale."
        )
    elif mode == "MONITORING-ONLY":
        lead_text = (
            f"The model leans {escape(signal.lower())}, but the signal is still better treated as monitoring evidence than as a trading edge."
        )
    else:
        lead_text = "The model direction and the quality gate both support a prediction-led decision this month."

    existing_section = ""
    if existing_guidance:
        row_html = "".join(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;font-weight:700;'>{escape(row['Category'])}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row['Shares'])}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row['Tax status'])}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row['Tranches'])}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row['Guidance'])}</td>"
            "</tr>"
            for row in existing_guidance["rows"]
        )
        bullet_html = "".join(
            f"<li style='margin:0 0 8px 0;'>{escape(line)}</li>"
            for line in existing_guidance["bullets"]
        )
        existing_section = (
            "<div style='margin-top:20px;padding:20px;border:1px solid #dbeafe;"
            "border-radius:16px;background:#f8fbff;'>"
            "<h2 style='margin:0 0 10px 0;font-size:20px;color:#0f172a;'>Existing shares already held</h2>"
            f"<p style='margin:0 0 12px 0;color:#334155;line-height:1.5;'>{escape(existing_guidance['headline'])}</p>"
            f"<p style='margin:0 0 12px 0;color:#475569;line-height:1.5;'>Current lot file: {existing_guidance['total_shares']:.0f} share(s) at ${existing_guidance['current_price']:.2f}.</p>"
            f"<ul style='margin:0 0 14px 20px;padding:0;color:#334155;line-height:1.5;'>{bullet_html}</ul>"
            "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Bucket</th>"
            "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Shares</th>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Tax</th>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Tranches</th>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Guidance</th>"
            "</tr></thead><tbody>"
            f"{row_html}"
            "</tbody></table>"
            "</div>"
        )

    executive_html = "".join(
        f"<li style='margin:0 0 8px 0;'>{escape(line)}</li>"
        for line in executive_summary
    )

    shadow_section = ""
    if cross_check_rows:
        shadow_html = "".join(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;font-weight:700;'>{escape(row.get('Label', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Mode', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Sell %', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Predicted', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Details', ''))}</td>"
            "</tr>"
            for row in cross_check_rows
        )
        shadow_section = (
            "<div style='margin-top:20px;padding:20px;border:1px solid #dbeafe;border-radius:16px;background:#f8fbff;'>"
            "<h2 style='margin:0 0 12px 0;font-size:20px;color:#0f172a;'>Consensus cross-check</h2>"
            f"<p style='margin:0 0 12px 0;color:#334155;line-height:1.5;'><strong>Active layer:</strong> {escape(recommendation_layer or 'n/a')}</p>"
            "<p style='margin:0 0 12px 0;color:#334155;line-height:1.5;'>"
            "This compares the live consensus path to the visible cross-check so the monthly memo stays honest about whether the current recommendation would change under the alternate weighting path."
            "</p>"
            "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Variant</th>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Mode</th>"
            "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Sell %</th>"
            "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Predicted</th>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Details</th>"
            "</tr></thead><tbody>"
            f"{shadow_html}"
            "</tbody></table>"
            "</div>"
        )

    redeploy_section = ""
    if redeploy_lines:
        redeploy_html = "".join(
            f"<li style='margin:0 0 8px 0;'>{escape(line)}</li>"
            for line in redeploy_lines
        )
        redeploy_section = (
            "<div style='margin-top:20px;padding:20px;border:1px solid #dbeafe;border-radius:16px;background:#f8fbff;'>"
            "<h2 style='margin:0 0 12px 0;font-size:20px;color:#0f172a;'>If redeploying sold exposure</h2>"
            f"<ul style='margin:0 0 0 20px;padding:0;color:#334155;line-height:1.5;'>{redeploy_html}</ul>"
            "</div>"
        )

    redeploy_portfolio_section = ""
    if redeploy_portfolio_table:
        redeploy_portfolio_section = (
            "<div style='margin-top:20px;padding:20px;border:1px solid #dbeafe;border-radius:16px;background:#f8fbff;'>"
            "<h2 style='margin:0 0 12px 0;font-size:20px;color:#0f172a;'>Suggested redeploy portfolio</h2>"
            "<p style='margin:0 0 12px 0;color:#334155;line-height:1.5;'>"
            "This is the repeatable answer to the practical follow-up question after a partial PGR sale: what to buy with the proceeds. "
            "The workflow keeps a stable high-equity base and then tilts the allocations modestly toward funds with stronger benchmark-outperformance signals and better diversification versus PGR."
            "</p>"
            f"{redeploy_portfolio_table}"
            "</div>"
        )

    confidence_section = ""
    if confidence_rows:
        confidence_section = (
            "<div style='margin-top:20px;padding:20px;border:1px solid #dbeafe;border-radius:16px;background:#f8fbff;'>"
            "<h2 style='margin:0 0 10px 0;font-size:20px;color:#0f172a;'>Confidence snapshot</h2>"
            "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Check</th>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Current</th>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Threshold</th>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Status</th>"
            "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;font-weight:700;'>{escape(row.get('Check', ''))}</td>"
                f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Current', ''))}</td>"
                f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Threshold', ''))}</td>"
                f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{confidence_badges.get(row.get('Check', ''), '')}</td>"
                "</tr>"
                for row in confidence_rows
            )
            + "</tbody></table></div>"
        )

    classification_section = ""
    if isinstance(classification_shadow, dict) and classification_shadow.get("enabled"):
        classification_section = (
            "<div style='margin-top:20px;padding:20px;border:1px solid #dbeafe;border-radius:16px;background:#f8fbff;'>"
            "<h2 style='margin:0 0 10px 0;font-size:20px;color:#0f172a;'>Classification confidence check</h2>"
            "<p style='margin:0 0 12px 0;color:#334155;line-height:1.5;'>"
            "Shadow-only interpretation layer from the v87-v96 classifier research. "
            "It does not change the live recommendation or sell percentage."
            "</p>"
            "<table style='width:100%;border-collapse:collapse;font-size:13px;'><tbody>"
            f"<tr><td style='padding:8px 0;color:#475569;width:220px;'>P(actionable sell)</td><td style='padding:8px 0;'><strong>{escape(str(classification_shadow.get('probability_actionable_sell_label', 'n/a')))}</strong></td></tr>"
            f"<tr><td style='padding:8px 0;color:#475569;'>Confidence tier</td><td style='padding:8px 0;'>{escape(str(classification_shadow.get('confidence_tier', 'n/a')))}</td></tr>"
            f"<tr><td style='padding:8px 0;color:#475569;'>Classifier stance</td><td style='padding:8px 0;'>{escape(str(classification_shadow.get('stance', 'n/a')))}</td></tr>"
            f"<tr><td style='padding:8px 0;color:#475569;'>Agreement</td><td style='padding:8px 0;'>{escape(str(classification_shadow.get('agreement_label', 'n/a')))}</td></tr>"
            "</tbody></table>"
            f"<p style='margin:12px 0 0 0;color:#334155;line-height:1.5;'>{escape(str(classification_shadow.get('interpretation', '')))}</p>"
            "</div>"
        )

    return (
        "<html><body style='margin:0;padding:0;background:#f3f6fb;font-family:Segoe UI,Arial,sans-serif;color:#0f172a;'>"
        "<div style='max-width:960px;margin:0 auto;padding:24px 14px;'>"
        "<div style='background:#ffffff;border-radius:20px;padding:24px 24px 18px 24px;"
        "box-shadow:0 8px 24px rgba(15,23,42,0.08);'>"
        "<p style='margin:0 0 10px 0;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;'>PGR Monthly Decision</p>"
        f"<h1 style='margin:0 0 10px 0;font-size:28px;line-height:1.2;color:#0f172a;'>{escape(_recommendation_headline(body))}</h1>"
        f"<div style='margin:0 0 14px 0;'>{mode_badge}</div>"
        f"<p style='margin:0 0 18px 0;font-size:16px;line-height:1.6;color:#334155;'>{lead_text}</p>"
        "<table role='presentation' style='width:100%;border-collapse:separate;border-spacing:12px 12px;'>"
        "<tr>"
        f"<td style='width:20%;background:#eff6ff;border:1px solid #bfdbfe;border-radius:16px;padding:16px;vertical-align:top;'><div style='font-size:12px;color:#1d4ed8;font-weight:700;text-transform:uppercase;'>Predicted 6M relative return</div><div style='font-size:24px;font-weight:800;color:#0f172a;margin-top:6px;'>{escape(predicted)}</div><div style='margin-top:8px;font-size:12px;color:#475569;'>PGR vs. the benchmark set</div></td>"
        f"<td style='width:20%;background:#eff6ff;border:1px solid #bfdbfe;border-radius:16px;padding:16px;vertical-align:top;'><div style='font-size:12px;color:#1d4ed8;font-weight:700;text-transform:uppercase;'>P(outperform)</div><div style='font-size:24px;font-weight:800;color:#0f172a;margin-top:6px;'>{escape(mean_cal)}</div><div style='margin-top:8px;font-size:12px;color:#475569;'>Calibrated probability</div></td>"
        f"<td style='width:20%;background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:16px;vertical-align:top;'><div style='font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;'>Mean IC</div><div style='font-size:24px;font-weight:800;color:#0f172a;margin-top:6px;'>{escape(mean_ic)}</div><div style='margin-top:8px;'>{confidence_badges.get('Mean IC', '')}</div></td>"
        f"<td style='width:20%;background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:16px;vertical-align:top;'><div style='font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;'>Mean hit rate</div><div style='font-size:24px;font-weight:800;color:#0f172a;margin-top:6px;'>{escape(mean_hr)}</div><div style='margin-top:8px;'>{confidence_badges.get('Mean hit rate', '')}</div></td>"
        f"<td style='width:20%;background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:16px;vertical-align:top;'><div style='font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;'>Aggregate OOS R^2</div><div style='font-size:24px;font-weight:800;color:#0f172a;margin-top:6px;'>{escape(oos_r2)}</div><div style='margin-top:8px;'>{confidence_badges.get('Aggregate OOS R^2', '')}</div></td>"
        "</tr></table>"
        f"{confidence_section}"
        f"{classification_section}"
        "<div style='margin-top:20px;padding:20px;border:1px solid #dbeafe;border-radius:16px;background:#f8fbff;'>"
        "<h2 style='margin:0 0 10px 0;font-size:20px;color:#0f172a;'>What's changed</h2>"
        f"<ul style='margin:0 0 0 20px;padding:0;color:#334155;line-height:1.5;'>{executive_html}</ul>"
        "</div>"
        "<div style='margin-top:20px;padding:20px;border:1px solid #fde68a;border-radius:16px;background:#fffdf5;'>"
        "<h2 style='margin:0 0 10px 0;font-size:20px;color:#0f172a;'>New vested shares</h2>"
        f"<p style='margin:0 0 12px 0;color:#334155;line-height:1.5;'><strong>{escape(_recommendation_headline(body))}</strong></p>"
        "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
        "<tbody>"
        f"<tr><td style='padding:8px 0;color:#475569;width:180px;'>Next vest</td><td style='padding:8px 0;'><strong>{escape(next_vest_data.get('Next vest date', 'n/a'))}</strong> ({escape(next_vest_data.get('RSU type', 'n/a'))})</td></tr>"
        f"<tr><td style='padding:8px 0;color:#475569;'>Default action</td><td style='padding:8px 0;'><strong>{escape(next_vest_data.get('Suggested default vest action', 'n/a'))}</strong></td></tr>"
        f"<tr><td style='padding:8px 0;color:#475569;'>Current PGR price</td><td style='padding:8px 0;'>{escape(next_vest_data.get('Current PGR price', 'n/a'))}</td></tr>"
        f"<tr><td style='padding:8px 0;color:#475569;'>Decision mode</td><td style='padding:8px 0;'>{escape(next_vest_data.get('Recommendation mode', mode))}</td></tr>"
        "</tbody></table>"
        "<p style='margin:12px 0 0 0;color:#334155;line-height:1.5;'>Use the next-vest action above for the incoming tranche. "
        "The scenario table below is still useful for tax framing, but it is not the primary recommendation when the mode is not actionable.</p>"
        "</div>"
        + existing_section +
        redeploy_section +
        redeploy_portfolio_section +
        shadow_section +
        (
            "<div style='margin-top:20px;padding:20px;border:1px solid #dbeafe;border-radius:16px;background:#f8fbff;'>"
            "<h2 style='margin:0 0 12px 0;font-size:20px;color:#0f172a;'>Dashboard snapshot</h2>"
            f"<p style='margin:0 0 12px 0;color:#334155;line-height:1.5;'>A static monthly dashboard snapshot is available for lightweight review: <strong>{escape(dashboard_snapshot_label or '-')}</strong>.</p>"
            + (
                f"<p style='margin:0;color:#334155;line-height:1.5;'><a href='{escape(dashboard_snapshot_url)}'>Open dashboard snapshot</a></p>"
                if dashboard_snapshot_url
                else ""
            )
            + "</div>"
            if dashboard_snapshot_label
            else ""
        ) +
        "<div style='margin-top:20px;padding:20px;border:1px solid #e2e8f0;border-radius:16px;background:#ffffff;'>"
        "<h2 style='margin:0 0 12px 0;font-size:20px;color:#0f172a;'>Tax timing scenarios for the next tranche</h2>"
        "<p style='margin:0 0 12px 0;color:#475569;line-height:1.5;'>These scenarios are informational unless the recommendation mode is ACTIONABLE. They help explain timing tradeoffs; they do not override the main vest instruction by themselves.</p>"
        f"{scenario_table}"
        "</div>"
        "<div style='margin-top:20px;padding:20px;border:1px solid #e2e8f0;border-radius:16px;background:#ffffff;'>"
        "<h2 style='margin:0 0 12px 0;font-size:20px;color:#0f172a;'>Benchmark detail</h2>"
        "<p style='margin:0 0 12px 0;color:#475569;line-height:1.5;'>Detailed benchmark signal detail is kept below the main decision memo so the recommendation stays readable on desktop and mobile. Predicted return is always from the perspective of PGR versus each fund, and the role column shows whether a fund is a buy candidate, optional substitute, or forecast-only context.</p>"
        f"{benchmark_table}"
        "</div>"
        "</div></div></body></html>"
    )


def build_email_message(
    body: str,
    from_addr: str,
    to_addr: str,
    month_label: str | None = None,
    lots_csv_path: str | Path | None = None,
    dashboard_snapshot_label: str | None = None,
    dashboard_snapshot_url: str | None = None,
    summary_payload: dict[str, object] | None = None,
) -> MIMEMultipart:
    """Construct a multipart monthly decision email."""
    if month_label is None:
        month_label = datetime.now(timezone.utc).strftime("%B %Y")

    signal = (
        _summary_lookup(summary_payload, "recommendation", "signal_label")
        or "UNKNOWN"
    )
    if signal == "UNKNOWN":
        match = re.search(r"\| Signal \| \*\*(.+?)\*\* \|", body)
        if match:
            signal = match.group(1)

    subject = f"PGR Monthly Decision \u2014 {month_label}: {signal}"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(
        MIMEText(
            build_email_summary(
                body,
                lots_csv_path=lots_csv_path,
                dashboard_snapshot_label=dashboard_snapshot_label,
                dashboard_snapshot_url=dashboard_snapshot_url,
                summary_payload=summary_payload,
            ),
            "plain",
            "utf-8",
        )
    )
    msg.attach(
        MIMEText(
            build_email_html(
                body,
                lots_csv_path=lots_csv_path,
                dashboard_snapshot_label=dashboard_snapshot_label,
                dashboard_snapshot_url=dashboard_snapshot_url,
                summary_payload=summary_payload,
            ),
            "html",
            "utf-8",
        )
    )
    return msg


def send_monthly_email(
    report_path: str | Path | None = None,
    *,
    smtp_server: str | None = None,
    smtp_port: int | None = None,
    username: str | None = None,
    password: str | None = None,
    from_addr: str | None = None,
    to_addr: str | None = None,
    month_label: str | None = None,
    dry_run: bool = False,
    lots_csv_path: str | Path | None = None,
) -> str:
    """Send the monthly decision report by email."""
    smtp_server = smtp_server or os.environ.get("SMTP_SERVER", "").strip()
    smtp_port = smtp_port or int(os.environ.get("SMTP_PORT", "465") or "465")
    username = username or os.environ.get("SMTP_USERNAME", "").strip()
    password = password or os.environ.get("SMTP_PASSWORD", "").strip()
    from_addr = from_addr or os.environ.get("EMAIL_FROM", "").strip()
    to_addr = to_addr or os.environ.get("EMAIL_TO", "").strip()

    if not dry_run:
        missing = [k for k, v in {
            "SMTP_SERVER": smtp_server,
            "SMTP_USERNAME": username,
            "SMTP_PASSWORD": password,
            "EMAIL_FROM": from_addr,
            "EMAIL_TO": to_addr,
        }.items() if not v]
        if missing:
            raise ValueError(f"send_monthly_email: missing SMTP configuration: {missing}")

    if report_path is None:
        ym = (month_label or datetime.now(timezone.utc).strftime("%Y-%m"))
        if " " in str(ym):
            try:
                ym = datetime.strptime(str(ym), "%B %Y").strftime("%Y-%m")
            except ValueError:
                pass
        report_path = Path(f"results/monthly_decisions/{ym}/recommendation.md")

    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Monthly report not found: {report_path}")

    summary_payload: dict[str, object] | None = None
    summary_path = report_path.with_name("monthly_summary.json")
    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    dashboard_snapshot_path = report_path.with_name("dashboard.html")
    dashboard_snapshot_label: str | None = str(dashboard_snapshot_path)
    dashboard_snapshot_url = (
        _infer_dashboard_snapshot_url(dashboard_snapshot_path)
        if dashboard_snapshot_path.exists()
        else None
    )
    if not dashboard_snapshot_path.exists():
        dashboard_snapshot_label = None

    body = report_path.read_text(encoding="utf-8")
    msg = build_email_message(
        body,
        from_addr or "noreply@example.com",
        to_addr or "",
        month_label,
        lots_csv_path=lots_csv_path,
        dashboard_snapshot_label=dashboard_snapshot_label,
        dashboard_snapshot_url=dashboard_snapshot_url,
        summary_payload=summary_payload,
    )
    subject: str = msg["Subject"]

    if dry_run:
        return subject

    try:
        if smtp_port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=ctx) as server:
                server.login(username, password)
                server.sendmail(from_addr, [to_addr], msg.as_string())
        else:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.login(username, password)
                server.sendmail(from_addr, [to_addr], msg.as_string())
    except smtplib.SMTPException as exc:
        raise RuntimeError(f"SMTP send failed: {exc}") from exc

    return subject
