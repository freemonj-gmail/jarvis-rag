"""OAuth sign-in for MCP servers that need it, and connector secret resolution.

RFC 7591 dynamic client registration plus RFC 7636 PKCE: one browser sign-in,
then a cached refresh token keeps later runs non-interactive.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import time
import urllib.parse
import webbrowser
from typing import Any, Optional

from jarvisrag.config import debug_print, JARVIS_MCP_TOKEN_CACHE
from jarvisrag.mcp.registry import MCPConnector
from jarvisrag.mcp.transport import _MCP_PROTOCOL_VERSION


# OAuth 2.0 (RFC 7591 DCR + RFC 7636 PKCE) settings for connectors whose MCP
# endpoint requires browser sign-in (e.g. taxmcp.io). One-time browser auth,
# then a cached refresh token keeps every later run non-interactive.
_OAUTH_REDIRECT_HOST = "127.0.0.1"


_OAUTH_CALLBACK_PATH = "/callback"


_OAUTH_HTTP_TIMEOUT = 30.0          # discovery/registration/token HTTP calls


_OAUTH_BROWSER_WAIT_SECONDS = 180.0  # how long to wait for the user to authorize


_TOKEN_EXPIRY_SKEW_SECONDS = 60      # refresh a little before actual expiry


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
