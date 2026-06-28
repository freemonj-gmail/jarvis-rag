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
  (``jarvis_mcp.json``). Adding a new source = adding an entry; no code change.
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
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import secrets
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from jarvis_config import (
    JARVIS_ENABLE_MCP,
    JARVIS_MCP_CONFIG,
    JARVIS_MCP_PII_GUARD,
    JARVIS_MCP_TIMEOUT,
    JARVIS_MCP_TOKEN_CACHE,
    debug_print,
)
from jarvis_io import time_limit

# Protocol version advertised during the MCP initialize handshake.
_MCP_PROTOCOL_VERSION = "2025-06-18"
# Hard caps so a misbehaving/hostile server cannot flood the LLM context.
_DEFAULT_MAX_EVIDENCE_LINES = 12
_DEFAULT_MAX_CHARS_PER_RESULT = 1500
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

# OAuth 2.0 (RFC 7591 DCR + RFC 7636 PKCE) settings for connectors whose MCP
# endpoint requires browser sign-in (e.g. taxmcp.io). One-time browser auth,
# then a cached refresh token keeps every later run non-interactive.
_OAUTH_REDIRECT_HOST = "127.0.0.1"
_OAUTH_CALLBACK_PATH = "/callback"
_OAUTH_HTTP_TIMEOUT = 30.0          # discovery/registration/token HTTP calls
_OAUTH_BROWSER_WAIT_SECONDS = 180.0  # how long to wait for the user to authorize
_TOKEN_EXPIRY_SKEW_SECONDS = 60      # refresh a little before actual expiry


# ---------------------------------------------------------------------------
# Data model (loaded from the JSON registry)
# ---------------------------------------------------------------------------
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
    transport: str                      # currently only "http" (streamable HTTP)
    url: str
    auth: str = "none"                  # "none" | "bearer" | "oauth"
    auth_env: Optional[str] = None      # bearer: env var holding the token (never the token)
    oauth_scope: str = ""               # oauth: scope requested during authorization
    match_any: list[str] = field(default_factory=list)   # case-insensitive substrings
    match_regex: Optional[str] = None
    calls: list[MCPCall] = field(default_factory=list)
    source_label: str = ""              # human label cited in evidence/References
    precedence: int = 100               # higher = more authoritative / injected first

    def matches(self, query: str) -> bool:
        """Routing gate: does this connector apply to ``query``?"""
        if not query:
            return False
        ql = query.lower()
        if any(term.lower() in ql for term in self.match_any if term):
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
    path = config_path or JARVIS_MCP_CONFIG
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
        match_any=[str(t) for t in entry.get("match_any", [])],
        match_regex=entry.get("match_regex"),
        calls=calls,
        source_label=str(entry.get("source_label", entry.get("name", ""))),
        precedence=int(entry.get("precedence", 100)),
    )


# ---------------------------------------------------------------------------
# Transport seam (so the HTTP client can be swapped / mocked in tests)
# ---------------------------------------------------------------------------
class MCPTransport(Protocol):
    """Calls a single MCP tool and returns its parsed JSON-RPC ``result``."""
    def call_tool(
        self,
        url: str,
        token: Optional[str],
        tool: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        ...


class HttpMCPTransport:
    """Default transport: streamable-HTTP MCP over JSON-RPC 2.0 via ``httpx``.

    Performs the minimal MCP handshake (``initialize`` -> ``notifications/
    initialized`` -> ``tools/call``) per call. ``httpx`` is imported lazily so the
    module loads even when the MCP layer is never used."""

    def call_tool(
        self,
        url: str,
        token: Optional[str],
        tool: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        import httpx  # lazy: only needed when a connector actually fires

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(timeout=timeout) as client:
            init = self._rpc(
                client, url, headers,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": _MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "jarvis", "version": "1.0"},
                    },
                },
            )
            session_id = init.get("_session_id")
            if session_id:
                headers["Mcp-Session-Id"] = session_id

            # initialized notification (no id, response ignored)
            try:
                client.post(
                    url, headers=headers,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
            except Exception:
                pass

            result = self._rpc(
                client, url, headers,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": arguments},
                },
            )
            return result

    @staticmethod
    def _rpc(
        client: Any,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        session_id = response.headers.get("Mcp-Session-Id")
        body = _parse_jsonrpc_body(response.text, response.headers.get("content-type", ""))
        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(f"MCP error: {body['error']}")
        result = body.get("result", {}) if isinstance(body, dict) else {}
        if session_id:
            result = {**result, "_session_id": session_id}
        return result


def _parse_jsonrpc_body(text: str, content_type: str) -> dict[str, Any]:
    """Parse a JSON-RPC response that may arrive as JSON or an SSE stream."""
    text = text.strip()
    if not text:
        return {}
    if "text/event-stream" in content_type or text.startswith("event:") or text.startswith("data:"):
        # Server-Sent Events: take the last well-formed JSON ``data:`` payload.
        last: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                chunk = line[len("data:"):].strip()
                try:
                    parsed = json.loads(chunk)
                    if isinstance(parsed, dict):
                        last = parsed
                except json.JSONDecodeError:
                    continue
        return last
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# OAuth 2.0 (Dynamic Client Registration + PKCE authorization-code) flow
# ---------------------------------------------------------------------------
# Some MCP endpoints (e.g. taxmcp.io) are protected by OAuth and issue NO static
# API key — the only way in is the same browser sign-in Claude/ChatGPT use. We
# implement it here: discover the auth server, dynamically register Jarvis as a
# client, open the browser once for the user to approve, exchange the code for an
# access + refresh token via PKCE, cache them (0600), and silently refresh after.
# The whole flow is best-effort: any failure returns None so the connector is
# simply skipped and the turn proceeds on local evidence (fail-safe).
def _load_token_cache() -> dict[str, Any]:
    path = JARVIS_MCP_TOKEN_CACHE
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        debug_print(f"[mcp-oauth] Could not read token cache: {exc}")
        return {}


def _save_token_cache(cache: dict[str, Any]) -> None:
    """Persist the token cache with owner-only (0600) permissions."""
    path = JARVIS_MCP_TOKEN_CACHE
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except Exception as exc:
        debug_print(f"[mcp-oauth] Could not persist token cache: {exc}")


def _pkce_pair() -> tuple[str, str]:
    """Return a (code_verifier, code_challenge) PKCE pair using S256."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _http_get_json(client: Any, url: str) -> dict[str, Any]:
    resp = client.get(
        url,
        headers={"Accept": "application/json", "MCP-Protocol-Version": _MCP_PROTOCOL_VERSION},
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _discover_oauth_metadata(client: Any, resource_url: str) -> dict[str, Any]:
    """Discover OAuth endpoints for an MCP resource.

    Follows RFC 9728 (protected-resource metadata) to find the authorization
    server, then RFC 8414 / OIDC discovery for its endpoints, with sane
    fallbacks so a partially-compliant server still works."""
    parsed = urllib.parse.urlsplit(resource_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")

    auth_servers: list[str] = []
    pr_candidates = []
    if path:
        pr_candidates.append(f"{origin}/.well-known/oauth-protected-resource{path}")
    pr_candidates.append(f"{origin}/.well-known/oauth-protected-resource")
    for candidate in pr_candidates:
        try:
            meta = _http_get_json(client, candidate)
            servers = meta.get("authorization_servers") or []
            if servers:
                auth_servers = [str(s) for s in servers]
                break
        except Exception:
            continue
    if not auth_servers:
        auth_servers = [origin]

    as_url = auth_servers[0].rstrip("/")
    as_parsed = urllib.parse.urlsplit(as_url)
    as_origin = f"{as_parsed.scheme}://{as_parsed.netloc}"
    as_path = as_parsed.path.rstrip("/")
    as_candidates = [
        f"{as_origin}/.well-known/oauth-authorization-server{as_path}",
        f"{as_url}/.well-known/oauth-authorization-server",
        f"{as_url}/.well-known/openid-configuration",
        f"{as_origin}/.well-known/oauth-authorization-server",
    ]
    for candidate in as_candidates:
        try:
            meta = _http_get_json(client, candidate)
            if meta.get("authorization_endpoint") and meta.get("token_endpoint"):
                return meta
        except Exception:
            continue

    # Last-resort conventional endpoints.
    return {
        "authorization_endpoint": f"{as_url}/authorize",
        "token_endpoint": f"{as_url}/token",
        "registration_endpoint": f"{as_url}/register",
    }


def _register_client(client: Any, metadata: dict[str, Any], redirect_uri: str) -> dict[str, Any]:
    """Dynamic Client Registration (RFC 7591). Returns the registration response
    (with ``client_id``) or {} if the server has no registration endpoint."""
    reg_endpoint = metadata.get("registration_endpoint")
    if not reg_endpoint:
        return {}
    payload = {
        "client_name": "Jarvis RAG",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "application_type": "native",
    }
    resp = client.post(
        reg_endpoint,
        json=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """Captures the single OAuth redirect on the loopback interface."""

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != _OAUTH_CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        self.server.oauth_response = params  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body style='font-family:sans-serif'>"
            b"<h2>Jarvis: authorization received.</h2>"
            b"<p>You can close this tab and return to your terminal.</p>"
            b"</body></html>"
        )

    def log_message(self, *args: Any) -> None:  # silence default logging
        return


def _await_authorization_code(httpd: Any, expected_state: str, timeout: float) -> str:
    """Block (bounded by ``timeout``) until the browser redirect arrives."""
    httpd.timeout = 1.0
    deadline = time.time() + timeout
    while time.time() < deadline:
        httpd.handle_request()
        response = getattr(httpd, "oauth_response", None)
        if response:
            if response.get("state") != expected_state:
                raise RuntimeError("OAuth state mismatch (possible CSRF); aborting.")
            if response.get("error"):
                raise RuntimeError(
                    f"Authorization denied: {response.get('error_description') or response['error']}"
                )
            code = response.get("code")
            if code:
                return code
            raise RuntimeError("Authorization response missing 'code'.")
    raise RuntimeError("Timed out waiting for browser authorization.")


def _post_token(client: Any, token_endpoint: str, data: dict[str, str]) -> dict[str, Any]:
    resp = client.post(token_endpoint, data=data, headers={"Accept": "application/json"})
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, dict) else {}


def _store_token(
    cache: dict[str, Any], key: str, entry: dict[str, Any], token: dict[str, Any]
) -> Optional[str]:
    """Persist a token response under ``key`` and return the access token."""
    access = token.get("access_token")
    if not access:
        return None
    merged = dict(entry)
    merged["access_token"] = access
    if token.get("refresh_token"):
        merged["refresh_token"] = token["refresh_token"]
    expires_in = token.get("expires_in")
    merged["expires_at"] = time.time() + (float(expires_in) if expires_in else 3600.0)
    cache[key] = merged
    _save_token_cache(cache)
    return str(access)


def _interactive_authorize(
    client: Any, connector: MCPConnector, cache: dict[str, Any], key: str
) -> Optional[str]:
    """Run the one-time browser authorization-code (PKCE) flow."""
    metadata = _discover_oauth_metadata(client, connector.url)
    httpd = http.server.HTTPServer((_OAUTH_REDIRECT_HOST, 0), _OAuthCallbackHandler)
    httpd.oauth_response = None  # type: ignore[attr-defined]
    port = httpd.server_address[1]
    redirect_uri = f"http://{_OAUTH_REDIRECT_HOST}:{port}{_OAUTH_CALLBACK_PATH}"
    try:
        existing = cache.get(key, {}) if isinstance(cache.get(key), dict) else {}
        registration = _register_client(client, metadata, redirect_uri)
        client_id = registration.get("client_id") or existing.get("client_id")
        if not client_id:
            raise RuntimeError("No client_id available (registration unsupported).")

        verifier, challenge = _pkce_pair()
        state = secrets.token_urlsafe(16)
        auth_params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        if connector.oauth_scope:
            auth_params["scope"] = connector.oauth_scope
        if connector.url:
            auth_params["resource"] = connector.url
        auth_url = metadata["authorization_endpoint"] + "?" + urllib.parse.urlencode(auth_params)

        print(
            f"[mcp-oauth] Authorizing '{connector.name}'. Opening your browser to sign in.\n"
            f"            If it does not open, paste this URL:\n            {auth_url}"
        )
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

        code = _await_authorization_code(httpd, state, _OAUTH_BROWSER_WAIT_SECONDS)
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        }
        if connector.url:
            token_data["resource"] = connector.url
        token = _post_token(client, metadata["token_endpoint"], token_data)

        entry = dict(existing)
        entry["client_id"] = str(client_id)
        entry["token_endpoint"] = metadata["token_endpoint"]
        access = _store_token(cache, key, entry, token)
        if access:
            print(f"[mcp-oauth] '{connector.name}' authorized; token cached for future runs.")
        return access
    finally:
        try:
            httpd.server_close()
        except Exception:
            pass


def _resolve_oauth_token(connector: MCPConnector) -> Optional[str]:
    """Return a valid access token for an OAuth connector.

    Uses a cached token while fresh, then a refresh token, then falls back to the
    one-time interactive browser flow. Returns None on any failure (caller
    skips)."""
    import httpx  # lazy: only when an OAuth connector actually fires

    cache = _load_token_cache()
    key = connector.name
    entry = cache.get(key, {}) if isinstance(cache.get(key), dict) else {}

    access = entry.get("access_token")
    expires_at = float(entry.get("expires_at", 0) or 0)
    if access and time.time() < (expires_at - _TOKEN_EXPIRY_SKEW_SECONDS):
        return str(access)

    with httpx.Client(timeout=_OAUTH_HTTP_TIMEOUT, follow_redirects=True) as client:
        refresh_token = entry.get("refresh_token")
        token_endpoint = entry.get("token_endpoint")
        client_id = entry.get("client_id")
        if refresh_token and token_endpoint and client_id:
            try:
                data = {
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                }
                if connector.url:
                    data["resource"] = connector.url
                token = _post_token(client, token_endpoint, data)
                refreshed = _store_token(cache, key, entry, token)
                if refreshed:
                    debug_print(f"[mcp-oauth] Refreshed access token for '{connector.name}'.")
                    return refreshed
            except Exception as exc:
                debug_print(
                    f"[mcp-oauth] Refresh failed for '{connector.name}' ({exc}); "
                    "starting browser authorization."
                )
        return _interactive_authorize(client, connector, cache, key)


# ---------------------------------------------------------------------------
# Evidence fetch + normalization
# ---------------------------------------------------------------------------
def _resolve_secret(connector: MCPConnector) -> Optional[str]:
    """Resolve a connector's access token.

    - ``none``   -> no token.
    - ``bearer`` -> token read from the referenced env var (never the registry).
    - ``oauth``  -> cached/refreshed token, or a one-time browser sign-in.

    Returns None (caller skips the connector) when a required token cannot be
    obtained — secrets are NEVER read from the registry file."""
    if connector.auth == "none":
        return None
    if connector.auth == "bearer":
        if not connector.auth_env:
            return None
        return os.environ.get(connector.auth_env) or None
    if connector.auth == "oauth":
        try:
            return _resolve_oauth_token(connector)
        except Exception as exc:
            print(f"[mcp] OAuth authorization for '{connector.name}' failed ({exc}); skipping.")
            return None
    return None


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


def _substitute_args(args: dict[str, Any], query: str) -> dict[str, Any]:
    """Fill ``{query}`` placeholders in a call's argument template."""
    out: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str):
            out[key] = value.replace("{query}", query)
        else:
            out[key] = value
    return out


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


def fetch_mcp_evidence(
    query: str,
    registry: Optional[MCPRegistry],
    transport: Optional[MCPTransport] = None,
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

    transport = transport or HttpMCPTransport()
    lines: list[str] = []

    for connector in selected:
        if len(lines) >= registry.max_evidence_lines:
            break
        token = _resolve_secret(connector)
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
            arguments = _substitute_args(call.args, query)
            arguments, blocked = _guard_outbound_args(arguments, connector.name, call.tool)
            if blocked:
                continue
            try:
                with time_limit(registry.timeout_seconds, operation=f"mcp:{connector.name}:{call.tool}"):
                    result = transport.call_tool(
                        connector.url, token, call.tool, arguments, registry.timeout_seconds
                    )
                text = _sanitize_external_text(
                    _result_to_text(result), registry.max_chars_per_result
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
    # `python -m jarvis_mcp` / `make mcp-login` -> run the interactive sign-in.
    raise SystemExit(0 if mcp_login() >= 0 else 1)
