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


def _extract_shadow_check(body: str) -> list[dict[str, str]]:
    section = _extract_section(body, "Simple-Baseline Cross-Check")
    for table_lines in _extract_markdown_tables(section):
        if table_lines and table_lines[0].strip().startswith("| Path |"):
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
        item = {
            "shares": lot.shares_remaining or lot.shares,
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


def build_email_summary(body: str, lots_csv_path: str | Path | None = None) -> str:
    """Build a concise plaintext decision memo from recommendation.md."""
    signal = _extract_report_value(body, "Signal") or "UNKNOWN"
    mode = _extract_report_value(body, "Recommendation Mode") or "MONITORING-ONLY"
    predicted = _extract_report_value(body, "Predicted 6M Relative Return") or "n/a"
    exec_lines = _extract_executive_summary(body)
    existing_guidance = _build_existing_shares_guidance(body, lots_csv_path)
    shadow_rows = _extract_shadow_check(body)
    recommendation_layer = _extract_recommendation_layer(body) or ""
    redeploy_lines = _extract_section_bullets(body, "Redeploy Guidance")

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
    if shadow_rows:
        lines += ["", "Simple-baseline cross-check:"]
        if recommendation_layer:
            lines.append(f"- Active layer: {recommendation_layer}")
        for row in shadow_rows:
            lines.append(
                f"- {row.get('Path', 'Path')}: {row.get('Recommendation Mode', 'n/a')}, "
                f"sell {row.get('Sell %', 'n/a')}, predicted {row.get('Predicted 6M Return', 'n/a')}"
            )
    if redeploy_lines:
        lines += ["", "If redeploying sold exposure:"]
        lines.extend(f"- {line}" for line in redeploy_lines)

    lines += [
        "",
        "Full report:",
        body,
    ]
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
    rename = {
        "SELL_NOW_STCG": "Sell at vest (STCG)",
        "HOLD_TO_LTCG": "Hold to LTCG date",
        "HOLD_FOR_LOSS": "Hold for downside / loss case",
    }
    for row in scenario_rows:
        html_rows.append(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;font-weight:700;'>{escape(rename.get(row.get('Scenario', ''), row.get('Scenario', '')))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Sell Date', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Tax Rate', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Predicted Return', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Net Proceeds', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Probability', ''))}</td>"
            "</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Scenario</th>"
        "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Date</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Tax rate</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Predicted return</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Net proceeds</th>"
        "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Probability</th>"
        "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )


def build_email_html(body: str, lots_csv_path: str | Path | None = None) -> str:
    """Build the HTML decision memo."""
    signal = _extract_report_value(body, "Signal") or "UNKNOWN"
    mode = _extract_report_value(body, "Recommendation Mode") or "MONITORING-ONLY"
    predicted = _extract_report_value(body, "Predicted 6M Relative Return") or "n/a"
    mean_ic = _extract_report_value(body, "Mean IC (across benchmarks)") or "n/a"
    mean_hr = _extract_report_value(body, "Mean Hit Rate") or "n/a"
    executive_summary = _extract_executive_summary(body)
    next_vest_data = _extract_next_vest_data(body)
    existing_guidance = _build_existing_shares_guidance(body, lots_csv_path)
    benchmark_table = _build_benchmark_html_table(body)
    scenario_table = _build_scenario_html_table(body)
    shadow_rows = _extract_shadow_check(body)
    recommendation_layer = _extract_recommendation_layer(body) or ""
    redeploy_lines = _extract_section_bullets(body, "Redeploy Guidance")
    mode_badge = {
        "ACTIONABLE": _html_badge("ACTIONABLE", "#0f766e"),
        "MONITORING-ONLY": _html_badge("MONITORING ONLY", "#a16207"),
        "DEFER-TO-TAX-DEFAULT": _html_badge("DEFER TO TAX DEFAULT", "#9a3412"),
    }.get(mode, _html_badge(mode, "#475569"))

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
    if shadow_rows:
        shadow_html = "".join(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;font-weight:700;'>{escape(row.get('Path', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;'>{escape(row.get('Recommendation Mode', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Sell %', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Predicted 6M Return', ''))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;'>{escape(row.get('Aggregate OOS R^2', ''))}</td>"
            "</tr>"
            for row in shadow_rows
        )
        shadow_section = (
            "<div style='margin-top:20px;padding:20px;border:1px solid #dbeafe;border-radius:16px;background:#f8fbff;'>"
            "<h2 style='margin:0 0 12px 0;font-size:20px;color:#0f172a;'>Recommendation-layer cross-check</h2>"
            f"<p style='margin:0 0 12px 0;color:#334155;line-height:1.5;'><strong>Active layer:</strong> {escape(recommendation_layer or 'n/a')}</p>"
            "<p style='margin:0 0 12px 0;color:#334155;line-height:1.5;'>"
            "This compares the active recommendation layer to the alternate path so the monthly memo stays honest about whether the simpler baseline and live stack agree."
            "</p>"
            "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Path</th>"
            "<th style='text-align:left;padding:8px;border-bottom:2px solid #cbd5e1;'>Mode</th>"
            "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Sell %</th>"
            "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>Predicted</th>"
            "<th style='text-align:right;padding:8px;border-bottom:2px solid #cbd5e1;'>OOS R²</th>"
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

    return (
        "<html><body style='margin:0;padding:0;background:#f3f6fb;font-family:Segoe UI,Arial,sans-serif;color:#0f172a;'>"
        "<div style='max-width:960px;margin:0 auto;padding:24px 14px;'>"
        "<div style='background:#ffffff;border-radius:20px;padding:24px 24px 18px 24px;"
        "box-shadow:0 8px 24px rgba(15,23,42,0.08);'>"
        "<p style='margin:0 0 10px 0;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;'>PGR Monthly Decision</p>"
        f"<h1 style='margin:0 0 10px 0;font-size:28px;line-height:1.2;color:#0f172a;'>{escape(_recommendation_headline(body))}</h1>"
        f"<div style='margin:0 0 14px 0;'>{mode_badge}</div>"
        f"<p style='margin:0 0 18px 0;font-size:16px;line-height:1.6;color:#334155;'>Model view: <strong>{escape(signal)}</strong>. "
        "This email separates the model direction from whether that direction is strong enough to drive an action.</p>"
        "<table role='presentation' style='width:100%;border-collapse:separate;border-spacing:12px 12px;'>"
        "<tr>"
        f"<td style='width:33%;background:#eff6ff;border:1px solid #bfdbfe;border-radius:16px;padding:16px;vertical-align:top;'><div style='font-size:12px;color:#1d4ed8;font-weight:700;text-transform:uppercase;'>Predicted 6M return</div><div style='font-size:24px;font-weight:800;color:#0f172a;margin-top:6px;'>{escape(predicted)}</div></td>"
        f"<td style='width:33%;background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:16px;vertical-align:top;'><div style='font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;'>Mean IC</div><div style='font-size:24px;font-weight:800;color:#0f172a;margin-top:6px;'>{escape(mean_ic)}</div></td>"
        f"<td style='width:33%;background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:16px;vertical-align:top;'><div style='font-size:12px;color:#475569;font-weight:700;text-transform:uppercase;'>Mean hit rate</div><div style='font-size:24px;font-weight:800;color:#0f172a;margin-top:6px;'>{escape(mean_hr)}</div></td>"
        "</tr></table>"
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
        shadow_section +
        "<div style='margin-top:20px;padding:20px;border:1px solid #e2e8f0;border-radius:16px;background:#ffffff;'>"
        "<h2 style='margin:0 0 12px 0;font-size:20px;color:#0f172a;'>Tax scenario view for the next tranche</h2>"
        f"{scenario_table}"
        "</div>"
        "<div style='margin-top:20px;padding:20px;border:1px solid #e2e8f0;border-radius:16px;background:#ffffff;'>"
        "<h2 style='margin:0 0 12px 0;font-size:20px;color:#0f172a;'>Benchmark detail</h2>"
        "<p style='margin:0 0 12px 0;color:#475569;line-height:1.5;'>Detailed benchmark signal detail is kept below the main decision memo so the recommendation stays readable on desktop and mobile.</p>"
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
) -> MIMEMultipart:
    """Construct a multipart monthly decision email."""
    if month_label is None:
        month_label = datetime.now(timezone.utc).strftime("%B %Y")

    signal = "UNKNOWN"
    match = re.search(r"\| Signal \| \*\*(.+?)\*\* \|", body)
    if match:
        signal = match.group(1)

    subject = f"PGR Monthly Decision \u2014 {month_label}: {signal}"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(build_email_summary(body, lots_csv_path=lots_csv_path), "plain", "utf-8"))
    msg.attach(MIMEText(build_email_html(body, lots_csv_path=lots_csv_path), "html", "utf-8"))
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

    body = report_path.read_text(encoding="utf-8")
    msg = build_email_message(
        body,
        from_addr or "noreply@example.com",
        to_addr or "",
        month_label,
        lots_csv_path=lots_csv_path,
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
