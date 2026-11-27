"""The MCP connector registry: connector definitions and their routing rules.

Connectors are pure data in ``registry.json``; this module loads them.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from jarvisrag.config import debug_print, JARVIS_ENABLE_MCP, JARVIS_MCP_CONFIG, JARVIS_MCP_TIMEOUT


# Default connector registry shipped inside this package (used when neither a
# config_path argument nor the JARVIS_MCP_CONFIG env var is provided).
_DEFAULT_REGISTRY = os.path.join(os.path.dirname(__file__), "registry.json")


# Hard caps so a misbehaving/hostile server cannot flood the LLM context.
_DEFAULT_MAX_EVIDENCE_LINES = 12


_DEFAULT_MAX_CHARS_PER_RESULT = 1500


# ---------------------------------------------------------------------------
# Data model (loaded from the JSON registry)
# ---------------------------------------------------------------------------
def routing_term_matches(term: str, text: str) -> bool:
    """Whether a routing term occurs in ``text`` starting at the beginning of a word.

    Shared by MCP connectors and domain corpora. A plain substring test let a
    term match inside an unrelated word ("irs" in "first", "opm" in
    "development", "irc" in "circumstance"); anchoring the start keeps derived
    forms ("tax" in "taxable", "deduction" in "deductions"). Case-insensitive."""
    if not term or not text:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(term.lower()), text.lower()) is not None


@dataclass
class MCPCall:
    """One tool invocation against a connector's MCP server.

    ``args`` values may contain the ``{query}`` placeholder, substituted with the
    user's (effective) query at call time."""
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPConnector:
    """A single MCP source, loaded from the registry as pure data."""
    name: str
    enabled: bool
    transport: str                      # "http" (streamable HTTP) | "stdio" (on-demand local process)
    url: str
    auth: str = "none"                  # "none" | "bearer" | "oauth"
    auth_env: Optional[str] = None      # bearer: env var holding the token (never the token)
    oauth_scope: str = ""               # oauth: scope requested during authorization
    command: list[str] = field(default_factory=list)    # stdio: process argv, spawned per query then torn down
    env: dict[str, str] = field(default_factory=dict)   # stdio: extra environment variables for the child
    match_any: list[str] = field(default_factory=list)   # case-insensitive terms, matched from the start of a word
    match_regex: Optional[str] = None
    calls: list[MCPCall] = field(default_factory=list)
    source_label: str = ""              # human label cited in evidence/References
    precedence: int = 100               # higher = more authoritative / injected first
    max_chars: Optional[int] = None     # per-connector result char cap (falls back to the registry default)

    def matches(self, query: str) -> bool:
        """Routing gate: does this connector apply to ``query``?"""
        if not query:
            return False
        if any(routing_term_matches(term, query) for term in self.match_any if term):
            return True
        if self.match_regex:
            try:
                if re.search(self.match_regex, query, re.IGNORECASE):
                    return True
            except re.error:
                debug_print(f"[mcp] Invalid match_regex on connector '{self.name}'; ignoring it.")
        return False


@dataclass
class MCPRegistry:
    """The full set of connectors plus global behavior settings."""
    enabled: bool = False
    timeout_seconds: float = JARVIS_MCP_TIMEOUT
    max_evidence_lines: int = _DEFAULT_MAX_EVIDENCE_LINES
    max_chars_per_result: int = _DEFAULT_MAX_CHARS_PER_RESULT
    connectors: list[MCPConnector] = field(default_factory=list)

    def active(self) -> bool:
        """True only if the layer is enabled AND at least one connector is on."""
        return self.enabled and any(c.enabled for c in self.connectors)

    def select(self, query: str) -> list[MCPConnector]:
        """Enabled connectors whose routing rules fire for ``query``,
        most-authoritative first."""
        if not self.active():
            return []
        chosen = [c for c in self.connectors if c.enabled and c.matches(query)]
        return sorted(chosen, key=lambda c: c.precedence, reverse=True)


# ---------------------------------------------------------------------------
# Registry loading (pure data, no network)
# ---------------------------------------------------------------------------
def load_mcp_registry(config_path: Optional[str] = None) -> MCPRegistry:
    """Load the connector registry from JSON.

    Never raises: a missing/invalid file (or the layer being disabled) yields an
    inert registry so the rest of Jarvis is unaffected. The global
    ``JARVIS_ENABLE_MCP`` env var force-enables the layer even if the file's
    ``settings.enabled`` is false."""
    path = config_path or JARVIS_MCP_CONFIG or _DEFAULT_REGISTRY
    if not path or not os.path.exists(path):
        if JARVIS_ENABLE_MCP:
            print(f"[mcp] JARVIS_ENABLE_MCP=1 but no connector registry found at '{path}'.")
        return MCPRegistry(enabled=False)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        print(f"[mcp] Could not read connector registry '{path}': {exc}. MCP disabled.")
        return MCPRegistry(enabled=False)

    settings = raw.get("settings", {}) if isinstance(raw, dict) else {}
    connectors: list[MCPConnector] = []
    for entry in (raw.get("connectors", []) if isinstance(raw, dict) else []):
        try:
            connectors.append(_connector_from_dict(entry))
        except Exception as exc:
            debug_print(f"[mcp] Skipping malformed connector entry: {exc}")

    enabled = bool(settings.get("enabled", False)) or JARVIS_ENABLE_MCP
    registry = MCPRegistry(
        enabled=enabled,
        timeout_seconds=float(settings.get("timeout_seconds", JARVIS_MCP_TIMEOUT)),
        max_evidence_lines=int(settings.get("max_evidence_lines", _DEFAULT_MAX_EVIDENCE_LINES)),
        max_chars_per_result=int(settings.get("max_chars_per_result", _DEFAULT_MAX_CHARS_PER_RESULT)),
        connectors=connectors,
    )
    if registry.active():
        live = [c.name for c in connectors if c.enabled]
        print(f"[mcp] Connector layer enabled. Active connectors: {live}")
    return registry


def _connector_from_dict(entry: dict[str, Any]) -> MCPConnector:
    calls = [
        MCPCall(tool=str(c["tool"]), args=dict(c.get("args", {})))
        for c in entry.get("calls", [])
        if isinstance(c, dict) and c.get("tool")
    ]
    return MCPConnector(
        name=str(entry["name"]),
        enabled=bool(entry.get("enabled", False)),
        transport=str(entry.get("transport", "http")),
        url=str(entry.get("url", "")),
        auth=str(entry.get("auth", "none")),
        auth_env=entry.get("auth_env"),
        oauth_scope=str(entry.get("oauth_scope", "")),
        command=[str(a) for a in entry.get("command", [])],
        env={str(k): str(v) for k, v in (entry.get("env", {}) or {}).items()},
        match_any=[str(t) for t in entry.get("match_any", [])],
        match_regex=entry.get("match_regex"),
        calls=calls,
        source_label=str(entry.get("source_label", entry.get("name", ""))),
        precedence=int(entry.get("precedence", 100)),
        max_chars=(int(entry["max_chars_per_result"]) if entry.get("max_chars_per_result") is not None else None),
    )
