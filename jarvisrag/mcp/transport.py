"""MCP transports: JSON-RPC over HTTP and over a local stdio subprocess.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Protocol

from jarvisrag.mcp.registry import MCPConnector


# Protocol version advertised during the MCP initialize handshake.
_MCP_PROTOCOL_VERSION = "2025-06-18"


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


class StdioMCPTransport:
    """On-demand transport: spawns a local MCP server over stdio for the call and
    tears it down afterward, so a keyless local server (e.g. the eCFR
    federal-regulations server via ``npx``) consumes NO idle resources between
    queries. Speaks newline-delimited JSON-RPC 2.0 on the child's stdin/stdout.

    Command + environment come from the connector (``command``/``env``);
    ``url``/``token`` are unused. Any failure raises so ``fetch_mcp_evidence``
    catches it and skips the connector (fail-safe), and the child process is
    always terminated in ``finally``."""

    def __init__(self, command: list[str], env: Optional[dict[str, str]] = None) -> None:
        self._command = list(command or [])
        self._env = dict(env or {})

    def call_tool(
        self,
        url: str,
        token: Optional[str],
        tool: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        import subprocess  # lazy: only needed when a stdio connector fires

        if not self._command:
            raise RuntimeError("stdio transport requires a non-empty 'command'")

        proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, **self._env},
            text=True,
            bufsize=1,
        )
        try:
            self._send(proc, {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "jarvis", "version": "1.0"},
                },
            })
            self._read_result(proc, 1)
            self._send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
            self._send(proc, {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": tool, "arguments": arguments},
            })
            return self._read_result(proc, 2)
        finally:
            self._terminate(proc)

    @staticmethod
    def _send(proc: Any, message: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise RuntimeError("stdio MCP server has no stdin")
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    @staticmethod
    def _read_result(proc: Any, want_id: int) -> dict[str, Any]:
        if proc.stdout is None:
            raise RuntimeError("stdio MCP server has no stdout")
        for line in proc.stdout:  # blocks; the caller's time_limit SIGALRM interrupts
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON log noise on stdout — ignore
            if isinstance(message, dict) and message.get("id") == want_id:
                if message.get("error"):
                    raise RuntimeError(f"MCP error: {message['error']}")
                result = message.get("result", {})
                return result if isinstance(result, dict) else {}
        raise RuntimeError("stdio MCP server closed before responding")

    @staticmethod
    def _terminate(proc: Any) -> None:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


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


def _transport_for(connector: MCPConnector) -> MCPTransport:
    """Pick the transport a connector's ``transport`` field asks for."""
    if connector.transport == "stdio":
        return StdioMCPTransport(connector.command, connector.env)
    return HttpMCPTransport()
