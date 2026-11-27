"""Pluggable MCP (Model Context Protocol) connector layer for Jarvis.

Purpose
-------
The general LLM corpus is stale on fast-moving, authoritative facts (current tax
law, regulations, policy) and has no access to a user's private account data.
This module lets Jarvis pull that information on demand from external MCP servers
and inject it as the HIGHEST-PRECEDENCE evidence for a question, so live
authoritative/account data usurps the model's general knowledge when relevant.

Design goals
------------
- **Scalable / extensible:** connectors are pure DATA in a JSON registry
  (``jarvisrag/mcp/registry.json``). Adding a new source = adding an entry; no code change.
- **Gated:** a connector only fires when the query matches its routing rules
  (e.g. a Fidelity/Prudential/Raymond James question for Truthify, an IRS/tax
  question for taxmcp.io). Non-matching questions are completely unaffected.
- **Opt-in:** disabled by default (global ``JARVIS_ENABLE_MCP`` + per-connector
  ``enabled``).
- **Fail-safe:** ANY failure (server down, missing token, provider does not
  expose MCP, protocol error, timeout) is caught and the connector is silently
  skipped — the turn proceeds on local/document evidence as before. MCP can
  never break a query.
- **Security:** the registry stores only secret *references* (env var names),
  never secrets. External MCP content is treated as untrusted DATA (labeled,
  size-capped, control-characters stripped), and the synthesis prompt is told to
  treat it as data, not instructions, to blunt prompt-injection.

Leaf module: depends only on ``jarvis_config`` and ``jarvis_io`` (plus a lazily
imported ``httpx`` for the default transport), so it never risks a circular
import with the RAG pipeline.

Split by job: ``registry`` (connector definitions), ``transport`` (HTTP and
stdio JSON-RPC), ``oauth`` (sign-in and secrets), ``guard`` (outbound PII
guard and external-content sanitizing); this module fetches the evidence.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from jarvisrag.config import debug_print
from jarvisrag.iotools import time_limit
from jarvisrag.mcp.guard import _guard_outbound_args, _sanitize_external_text
from jarvisrag.mcp.registry import load_mcp_registry, MCPRegistry
from jarvisrag.mcp.transport import _transport_for, MCPTransport
from jarvisrag.mcp.oauth import _resolve_oauth_token, _resolve_secret


def _result_to_text(result: dict[str, Any]) -> str:
    """Extract human-readable text from an MCP ``tools/call`` result.

    Handles the standard ``content: [{type:'text', text:...}]`` shape and falls
    back to ``structuredContent`` / raw JSON for non-text tools."""
    if not isinstance(result, dict):
        return str(result)
    parts: list[str] = []
    for item in result.get("content", []) or []:
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            parts.append(str(item["text"]))
    if parts:
        return "\n".join(parts)
    if result.get("structuredContent"):
        try:
            return json.dumps(result["structuredContent"], ensure_ascii=False)
        except Exception:
            return str(result["structuredContent"])
    return ""


def _substitute_args(args: dict[str, Any], query: str, keywords: Optional[str] = None) -> dict[str, Any]:
    """Fill placeholders in a call's argument template.

    ``{query}`` -> the raw (effective) user query. ``{keywords}`` -> a compact
    keyword form for literal full-text-search back-ends (falls back to the raw
    query when the caller supplies none)."""
    kw = keywords if keywords is not None else query
    out: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str):
            out[key] = value.replace("{query}", query).replace("{keywords}", kw)
        else:
            out[key] = value
    return out


def fetch_mcp_evidence(
    query: str,
    registry: Optional[MCPRegistry],
    transport: Optional[MCPTransport] = None,
    search_terms: Optional[str] = None,
) -> list[str]:
    """Pull evidence lines from every connector that matches ``query``.

    Returns evidence lines in the SAME format the answer pipeline already
    consumes, each tagged so the synthesizer treats it as the most current
    authoritative/account source. Fail-safe: any connector/tool error is caught
    and skipped, and a fully inert/empty result is returned rather than raised,
    so a failed MCP pull never breaks the turn."""
    if not registry or not registry.active():
        return []
    selected = registry.select(query)
    if not selected:
        return []

    injected_transport = transport
    search_keywords = (search_terms or "").strip() or query
    lines: list[str] = []

    for connector in selected:
        if len(lines) >= registry.max_evidence_lines:
            break
        token = _resolve_secret(connector)
        active_transport = injected_transport or _transport_for(connector)
        if connector.auth in ("bearer", "oauth") and not token:
            detail = (
                f"env {connector.auth_env}" if connector.auth == "bearer"
                else "browser sign-in unavailable or declined"
            )
            print(
                f"[mcp] Connector '{connector.name}' has no usable credentials "
                f"({detail}) — skipping."
            )
            continue

        for call in connector.calls:
            if len(lines) >= registry.max_evidence_lines:
                break
            arguments = _substitute_args(call.args, query, search_keywords)
            arguments, blocked = _guard_outbound_args(arguments, connector.name, call.tool)
            if blocked:
                continue
            try:
                with time_limit(registry.timeout_seconds, operation=f"mcp:{connector.name}:{call.tool}"):
                    result = active_transport.call_tool(
                        connector.url, token, call.tool, arguments, registry.timeout_seconds
                    )
                text = _sanitize_external_text(
                    _result_to_text(result), connector.max_chars or registry.max_chars_per_result
                )
                if text:
                    label = connector.source_label or connector.name
                    lines.append(
                        f"[MCP · {label} · live authoritative source · {call.tool}] {text}"
                    )
                    debug_print(f"[mcp] {connector.name}:{call.tool} returned {len(text)} chars.")
                else:
                    debug_print(f"[mcp] {connector.name}:{call.tool} returned no text content.")
            except Exception as exc:
                # Fail-safe: never let an MCP failure surface to the user.
                print(f"[mcp] {connector.name}:{call.tool} failed ({exc}); omitting and moving on.")
                continue

    return lines[: registry.max_evidence_lines]


def mcp_login(config_path: Optional[str] = None) -> int:
    """Pre-authorize every enabled OAuth connector (one-time browser sign-in).

    Lets the user complete the interactive flow up front (``make mcp-login``)
    instead of during the first matching query. Returns the number of connectors
    successfully authorized. Always returns (never raises) so it is safe to run
    repeatedly; already-cached connectors just refresh silently."""
    registry = load_mcp_registry(config_path)
    oauth_connectors = [
        c for c in registry.connectors if c.enabled and c.auth == "oauth"
    ]
    if not oauth_connectors:
        print("[mcp-oauth] No enabled OAuth connectors to authorize.")
        return 0

    authorized = 0
    for connector in oauth_connectors:
        print(f"[mcp-oauth] Authorizing connector '{connector.name}' ({connector.url}) …")
        try:
            token = _resolve_oauth_token(connector)
        except Exception as exc:
            print(f"[mcp-oauth] '{connector.name}' authorization failed ({exc}).")
            continue
        if token:
            authorized += 1
        else:
            print(f"[mcp-oauth] '{connector.name}' was not authorized.")
    print(f"[mcp-oauth] Done. {authorized}/{len(oauth_connectors)} connector(s) authorized.")
    return authorized


if __name__ == "__main__":
    # `python -m jarvisrag.mcp.connector` / `make mcp-login` -> run the interactive sign-in.
    raise SystemExit(0 if mcp_login() >= 0 else 1)
