"""
Monthly decision email delivery (P2.8).

Extracts the SMTP send logic from the GitHub Actions workflow YAML into a
testable Python module.  This lets tests mock ``smtplib.SMTP_SSL`` /
``smtplib.SMTP`` to verify correct email construction without network access.

The module has two public functions:
  build_email_message()  — pure function; constructs the MIMEMultipart message
  send_monthly_email()   — side-effectful; reads env-vars and sends via SMTP

Both functions raise ``ValueError`` on missing / invalid configuration rather
than silently exiting, so callers can decide whether to treat that as fatal.
"""

from __future__ import annotations

import os
import re
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _extract_report_value(body: str, field: str) -> str | None:
    match = re.search(rf"\| {re.escape(field)} \| \*\*(.+?)\*\* \|", body)
    if match:
        return match.group(1)
    match = re.search(rf"\| {re.escape(field)} \| (.+?) \|", body)
    if match:
        return match.group(1)
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


def build_email_summary(body: str) -> str:
    """Build a concise plaintext decision memo from recommendation.md."""
    signal = _extract_report_value(body, "Signal") or "UNKNOWN"
    mode = _extract_report_value(body, "Recommendation Mode") or "MONITORING-ONLY"
    sell_pct = _extract_report_value(body, "Recommended Sell %") or "n/a"
    predicted = _extract_report_value(body, "Predicted 6M Relative Return") or "n/a"
    exec_lines = _extract_executive_summary(body)

    lines = [
        "PGR Monthly Decision Summary",
        "",
        f"Signal: {signal}",
        f"Recommendation mode: {mode}",
        f"Suggested vest action: {sell_pct}",
        f"Predicted 6M relative return: {predicted}",
    ]
    if exec_lines:
        lines += ["", "Executive summary:"]
        lines.extend(f"- {line}" for line in exec_lines)

    lines += [
        "",
        "Full report:",
        body,
    ]
    return "\n".join(lines)


def build_email_message(
    body: str,
    from_addr: str,
    to_addr: str,
    month_label: str | None = None,
) -> MIMEMultipart:
    """Construct a MIMEMultipart email from a recommendation.md body string.

    Extracts the signal line from the report body to produce an informative
    subject line, then attaches a concise plaintext summary followed by the
    full report body.

    Args:
        body:        Full text of the monthly recommendation.md report.
        from_addr:   Sender email address (e.g. ``"me@example.com"``).
        to_addr:     Recipient email address.
        month_label: Human-readable month string for the subject line, e.g.
                     ``"April 2026"``.  Defaults to the current UTC month.

    Returns:
        A ``MIMEMultipart`` message ready to pass to ``smtplib``.
    """
    if month_label is None:
        month_label = datetime.now(timezone.utc).strftime("%B %Y")

    # Extract signal from "| Signal | **NEUTRAL (LOW CONFIDENCE)** |"
    signal = "UNKNOWN"
    m = re.search(r"\| Signal \| \*\*(.+?)\*\* \|", body)
    if m:
        signal = m.group(1)

    subject = f"PGR Monthly Decision \u2014 {month_label}: {signal}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(build_email_summary(body), "plain", "utf-8"))
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
) -> str:
    """Send the monthly decision report by email.

    Configuration can be passed as keyword arguments or read from environment
    variables (``SMTP_SERVER``, ``SMTP_PORT``, ``SMTP_USERNAME``,
    ``SMTP_PASSWORD``, ``EMAIL_FROM``, ``EMAIL_TO``).  Explicit keyword
    arguments take precedence over environment variables.

    Args:
        report_path: Path to ``recommendation.md``.  Defaults to the current
                     UTC month's report at
                     ``results/monthly_decisions/YYYY-MM/recommendation.md``.
        smtp_server: SMTP hostname.
        smtp_port:   SMTP port (465 = SMTP_SSL, 587 = STARTTLS).
        username:    SMTP login username.
        password:    SMTP login password.
        from_addr:   Sender address.
        to_addr:     Recipient address.
        month_label: Month label for the subject line (default: current UTC month).
        dry_run:     If ``True``, build the message and return the subject line
                     without making any network connection.

    Returns:
        The email subject line string (useful for logging / dry-run output).

    Raises:
        FileNotFoundError: If ``report_path`` does not exist.
        ValueError:        If required SMTP configuration is missing.
        RuntimeError:      If the SMTP send fails.
    """
    # --- Resolve config from args → env-vars ---
    smtp_server = smtp_server or os.environ.get("SMTP_SERVER", "").strip()
    smtp_port   = smtp_port   or int(os.environ.get("SMTP_PORT", "465") or "465")
    username    = username    or os.environ.get("SMTP_USERNAME", "").strip()
    password    = password    or os.environ.get("SMTP_PASSWORD", "").strip()
    from_addr   = from_addr   or os.environ.get("EMAIL_FROM", "").strip()
    to_addr     = to_addr     or os.environ.get("EMAIL_TO", "").strip()

    if not dry_run:
        missing = [k for k, v in {
            "SMTP_SERVER": smtp_server,
            "SMTP_USERNAME": username,
            "SMTP_PASSWORD": password,
            "EMAIL_FROM": from_addr,
            "EMAIL_TO": to_addr,
        }.items() if not v]
        if missing:
            raise ValueError(
                f"send_monthly_email: missing SMTP configuration: {missing}"
            )

    # --- Locate report ---
    if report_path is None:
        ym = (month_label or datetime.now(timezone.utc).strftime("%Y-%m"))
        # If month_label like "April 2026" → convert to "2026-04"
        if " " in str(ym):
            try:
                ym = datetime.strptime(str(ym), "%B %Y").strftime("%Y-%m")
            except ValueError:
                pass
        report_path = Path(
            f"results/monthly_decisions/{ym}/recommendation.md"
        )

    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(
            f"Monthly report not found: {report_path}"
        )

    body = report_path.read_text(encoding="utf-8")
    msg = build_email_message(body, from_addr or "noreply@example.com", to_addr or "", month_label)
    subject: str = msg["Subject"]

    if dry_run:
        return subject

    # --- Send ---
    try:
        if smtp_port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=ctx) as s:
                s.login(username, password)
                s.sendmail(from_addr, [to_addr], msg.as_string())
        else:
            with smtplib.SMTP(smtp_server, smtp_port) as s:
                s.ehlo()
                s.starttls(context=ssl.create_default_context())
                s.login(username, password)
                s.sendmail(from_addr, [to_addr], msg.as_string())
    except smtplib.SMTPException as exc:
        raise RuntimeError(f"SMTP send failed: {exc}") from exc

    return subject
