"""Safety at the MCP boundary.

The outbound PII guard that scrubs a question before it leaves the machine,
and sanitizing of untrusted external content before it becomes evidence.
"""

from __future__ import annotations

import re
from typing import Any

from jarvisrag.config import JARVIS_MCP_PII_GUARD


# Control characters (except tab/newline) are stripped from external content.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# --- Outbound PII guard (egress boundary) ----------------------------------
# Only the user's question text leaves the machine (it is substituted into the
# MCP call arguments). Before any MCP request, scrub recognizable PII so a PII
# question asked against LOCAL data never egresses raw to a third-party MCP
# provider. Each entry: (category, pattern, placeholder). Order matters — more
# specific digit patterns (SSN/EIN) run before the looser phone pattern.
_PII_PATTERNS: list[tuple[str, "re.Pattern[str]", str]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED-EMAIL]"),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED-SSN]"),
    ("EIN", re.compile(r"\b\d{2}-\d{7}\b"), "[REDACTED-EIN]"),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED-PHONE]"),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), "[REDACTED-IP]"),
    ("ADDRESS", re.compile(
        r"\b\d{1,6}[A-Za-z]?\s+[A-Za-z0-9.\s]{2,40}?\s"
        r"(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Ln|Lane|Dr|Drive|"
        r"Ct|Court|Way|Pl|Place|Ter|Terrace|Cir|Circle|Hwy|Highway)\b\.?",
        re.IGNORECASE), "[REDACTED-ADDRESS]"),
]


# Card-like digit runs are Luhn-validated before redacting, so tax section
# numbers / years / long IDs are not mistaken for credit cards.
_CREDIT_CARD_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")


def _sanitize_external_text(text: str, max_chars: int) -> str:
    """Treat external MCP content as untrusted DATA: strip control characters,
    collapse whitespace, and cap length. Content is NOT interpreted; the
    synthesis prompt additionally instructs the model to ignore any embedded
    directives."""
    if not text:
        return ""
    cleaned = _CONTROL_CHAR_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + " …[truncated]"
    return cleaned


def _luhn_ok(digits: str) -> bool:
    """Luhn check-digit validation (used to confirm a digit run is a real card)."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace recognizable PII in ``text`` with placeholders.

    Returns ``(redacted_text, categories)`` where ``categories`` is the ordered,
    de-duplicated list of PII kinds found (empty if the text is clean). Used at
    the MCP egress boundary so PII never leaves the machine in a request.

    Note: structured PII (SSN/EIN, email, phone, credit card, IP, street
    address) is detected by pattern. Free-form identifiers like a person's name
    cannot be detected by regex alone and are NOT redacted here — keep names out
    of MCP queries, or use ``JARVIS_MCP_PII_GUARD=block`` for a hard stop."""
    if not text:
        return text, []
    found: list[str] = []

    def _cc_sub(match: "re.Match[str]") -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("CREDIT_CARD")
            return "[REDACTED-CC]"
        return match.group(0)

    redacted = _CREDIT_CARD_RE.sub(_cc_sub, text)
    for label, pattern, placeholder in _PII_PATTERNS:
        def _sub(match: "re.Match[str]", _label=label, _ph=placeholder) -> str:
            found.append(_label)
            return _ph
        redacted = pattern.sub(_sub, redacted)

    seen: set[str] = set()
    categories = [c for c in found if not (c in seen or seen.add(c))]
    return redacted, categories


def _guard_outbound_args(
    arguments: dict[str, Any], connector_name: str, tool: str
) -> tuple[dict[str, Any], bool]:
    """Scrub PII from outbound MCP call arguments per ``JARVIS_MCP_PII_GUARD``.

    Returns ``(args_to_send, blocked)``. In ``redact`` mode (default) any
    detected PII is replaced with placeholders and the user is warned; in
    ``block`` mode the call is refused (``blocked=True``) and nothing is sent;
    in ``off`` mode the arguments pass through verbatim. The user is always
    warned BEFORE anything is sent."""
    mode = (JARVIS_MCP_PII_GUARD or "redact").strip().lower()
    if mode == "off":
        return arguments, False

    all_categories: list[str] = []
    sanitized: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            redacted, categories = _redact_pii(value)
            sanitized[key] = redacted
            all_categories.extend(categories)
        else:
            sanitized[key] = value

    if not all_categories:
        return arguments, False

    categories = sorted(set(all_categories))
    if mode == "block":
        print(
            f"[mcp] PII guard: detected {categories} in the outbound query for "
            f"'{connector_name}:{tool}'. BLOCKING the request — nothing was sent "
            f"to the provider. Answering on local evidence only. "
            f"(Set JARVIS_MCP_PII_GUARD=redact to send a sanitized query instead.)"
        )
        return arguments, True

    print(
        f"[mcp] PII guard: detected {categories} in the outbound query for "
        f"'{connector_name}:{tool}'. Sanitizing before sending — the provider "
        f"receives a redacted query; your raw PII never leaves this machine."
    )
    return sanitized, False
