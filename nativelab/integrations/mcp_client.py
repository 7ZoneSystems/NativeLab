"""
MCP (Model Context Protocol) client for pipeline blocks.

Supports two transports:
  - SSE  : HTTP Server-Sent Events (streamable HTTP)
  - stdio: local subprocess with JSON-RPC over stdin/stdout

Used by the MCP_SERVER pipeline block to call external tools.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]


class McpClient:
    """Lightweight MCP client that can test connections and call tools."""

    def __init__(self):
        self.connected: bool = False
        self.tools: List[Dict[str, Any]] = []
        self._session = None
        self._proc: Optional[subprocess.Popen] = None
        self._transport: str = ""
        self._url: str = ""
        self._response_queue: list = []
        self._notif_queue: list = []
        self._reader_thread: Optional[threading.Thread] = None
        self._request_id: int = 0
        self._auth_token: Optional[str] = None
        self._auth_env: Optional[Dict[str, str]] = None
        self._extra_headers: Dict[str, str] = {}

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    # ── public API ──────────────────────────────────────────────────────────

    def test_connection(self, transport: str, url: str, *,
                        auth_token: Optional[str] = None,
                        auth_env: Optional[Dict[str, str]] = None) -> Tuple[bool, Any]:
        """
        Connect to MCP server, list tools, return (True, tools_list) or (False, error_string).
        Does NOT leave the connection open — call shutdown() after.
        """
        try:
            ok = self._connect(transport, url, auth_token=auth_token, auth_env=auth_env)
            if not ok:
                return False, "Connection failed"
            return True, list(self.tools)
        except Exception as e:
            return False, str(e)
        finally:
            self.shutdown()

    def execute(self, transport: str, url: str,
                tool_name: str, arguments: dict, *,
                auth_token: Optional[str] = None,
                auth_env: Optional[Dict[str, str]] = None) -> Tuple[bool, Any]:
        """
        Connect, call a tool, return (True, result_text) or (False, error_string).
        """
        try:
            if not self._connect(transport, url, auth_token=auth_token, auth_env=auth_env):
                return False, "Connection failed"
            result = self._call_tool(tool_name, arguments)
            if result is None:
                return False, "Tool call returned no result"
            return True, result
        except Exception as e:
            return False, str(e)
        finally:
            self.shutdown()

    def shutdown(self):
        """Close any open connection."""
        if self._session:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._session.close())
                loop.close()
            except Exception:
                pass
            self._session = None
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self.connected = False
        self.tools = []

    # ── connection ──────────────────────────────────────────────────────────

    def _connect(self, transport: str, url: str, *,
                 auth_token: Optional[str] = None,
                 auth_env: Optional[Dict[str, str]] = None) -> bool:
        self._transport = transport
        self._url = url
        self._auth_token = auth_token
        self._auth_env = auth_env or {}
        # Build extra headers from auth token
        self._extra_headers = {}
        if auth_token:
            self._extra_headers["Authorization"] = f"Bearer {auth_token}"
        if transport == "sse":
            return self._connect_sse(url)
        else:
            return self._connect_stdio(url)

    def _connect_sse(self, url: str) -> bool:
        """Connect to an SSE-based MCP server."""
        if aiohttp is None:
            raise RuntimeError(
                "aiohttp is required for SSE MCP servers.\n"
                "Install it:  pip install aiohttp")
        try:
            import asyncio

            loop = asyncio.new_event_loop()

            async def _do_connect():
                self._session = aiohttp.ClientSession()  # type: ignore[union-attr]
                resp = await self._session.get(url, timeout=aiohttp.ClientTimeout(total=10))  # type: ignore[union-attr]
                if resp.status != 200:
                    await self._session.close()
                    self._session = None
                    return False
                await resp.release()
                return True

            ok = loop.run_until_complete(_do_connect())
            if not ok:
                return False

            # Initialize
            init_result = loop.run_until_complete(
                self._send_sse_request(loop, "initialize", {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "nativelab-mcp", "version": "1.0.0"},
                }))
            if init_result is None:
                return False

            loop.run_until_complete(
                self._send_sse_notification(loop, "notifications/initialized", {}))

            # List tools
            tools_result = loop.run_until_complete(
                self._send_sse_request(loop, "tools/list", {}))
            loop.close()

            if tools_result and "tools" in tools_result:
                self.tools = tools_result["tools"]
                self.connected = True
                return True
            return False
        except Exception as e:
            raise RuntimeError(f"SSE connection error: {e}")

    async def _send_sse_request(self, loop, method: str, params: dict) -> Optional[dict]:
        """Send JSON-RPC request via SSE POST and read response."""
        if not self._session:
            return None
        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self._extra_headers)
            async with self._session.post(  # type: ignore[union-attr]
                self._url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),  # type: ignore[union-attr]
            ) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()
                for line in text.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data:
                            try:
                                msg = json.loads(data)
                                if "result" in msg:
                                    return msg["result"]
                                if "error" in msg:
                                    return None
                            except json.JSONDecodeError:
                                continue
                # Try parsing as direct JSON response
                try:
                    msg = json.loads(text)
                    if "result" in msg:
                        return msg["result"]
                except Exception:
                    pass
                return None
        except Exception:
            return None

    async def _send_sse_notification(self, loop, method: str, params: dict):
        """Send JSON-RPC notification (no response expected)."""
        if not self._session:
            return
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self._extra_headers)
            await self._session.post(  # type: ignore[union-attr]
                self._url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),  # type: ignore[union-attr]
            )
        except Exception:
            pass

    def _connect_stdio(self, cmd: str) -> bool:
        """Connect to a stdio-based MCP server (including npx commands)."""
        try:
            import os
            env = dict(os.environ)
            # Inject auth token as env var
            if self._auth_token:
                env["MCP_AUTH_TOKEN"] = self._auth_token
            # Inject user-provided env vars
            if self._auth_env:
                env.update(self._auth_env)
            self._proc = subprocess.Popen(
                cmd, shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=env,
            )
            self._response_queue = []
            self._notif_queue = []
            self._reader_thread = threading.Thread(
                target=self._read_stdout_loop, daemon=True)
            self._reader_thread.start()

            # Initialize
            init_result = self._send_stdio_request("initialize", {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "nativelab-mcp", "version": "1.0.0"},
            })
            if init_result is None:
                return False

            self._send_stdio_notification("notifications/initialized", {})

            # List tools
            tools_result = self._send_stdio_request("tools/list", {})
            if tools_result and "tools" in tools_result:
                self.tools = tools_result["tools"]
                self.connected = True
                return True
            return False
        except Exception as e:
            raise RuntimeError(f"stdio connection error: {e}")

    def _read_stdout_loop(self):
        """Background thread reading LSP-framed JSON-RPC messages from stdout."""
        import re
        buffer = b""
        while self._proc and self._proc.poll() is None:
            stdout = self._proc.stdout
            if stdout is None:
                break
            try:
                chunk = stdout.read(1)
                if not chunk:
                    break
                buffer += chunk
                # Check for Content-Length header
                header_end = buffer.find(b"\r\n\r\n")
                if header_end >= 0:
                    header = buffer[:header_end].decode("utf-8", errors="replace")
                    match = re.search(r"Content-Length:\s*(\d+)", header, re.IGNORECASE)
                    if match:
                        content_len = int(match.group(1))
                        body_start = header_end + 4
                        while len(buffer) < body_start + content_len:
                            more = stdout.read(
                                content_len - (len(buffer) - body_start))
                            if not more:
                                break
                            buffer += more
                        body = buffer[body_start:body_start + content_len]
                        buffer = buffer[body_start + content_len:]
                        try:
                            msg = json.loads(body.decode("utf-8"))
                            if "id" in msg:
                                self._response_queue.append(msg)
                            else:
                                self._notif_queue.append(msg)
                        except json.JSONDecodeError:
                            pass
                    else:
                        buffer = buffer[header_end + 2:]
            except Exception:
                break

    def _send_stdio_request(self, method: str, params: dict,
                            timeout: float = 30.0) -> Optional[dict]:
        """Send JSON-RPC request to stdio server, wait for response."""
        if not self._proc or self._proc.poll() is not None:
            return None
        req_id = self._next_id()
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
        stdin = self._proc.stdin
        if stdin is None:
            return None
        try:
            stdin.write(header + payload)
            stdin.flush()
        except Exception:
            return None

        # Wait for response
        deadline = time.time() + timeout
        while time.time() < deadline:
            for i, msg in enumerate(self._response_queue):
                if msg.get("id") == req_id:
                    self._response_queue.pop(i)
                    if "result" in msg:
                        return msg["result"]
                    if "error" in msg:
                        return None
            time.sleep(0.05)
        return None

    def _send_stdio_notification(self, method: str, params: dict):
        """Send JSON-RPC notification (no response expected)."""
        if not self._proc or self._proc.poll() is not None:
            return
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
        stdin = self._proc.stdin
        if stdin is None:
            return
        try:
            stdin.write(header + payload)
            stdin.flush()
        except Exception:
            pass

    def _call_tool(self, tool_name: str, arguments: dict) -> Optional[str]:
        """Call a tool and return the text result."""
        if self._transport == "sse":
            import asyncio
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                self._send_sse_request(loop, "tools/call", {
                    "name": tool_name,
                    "arguments": arguments,
                }))
            loop.close()
        else:
            result = self._send_stdio_request("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })
        if result is None:
            return None
        return self._extract_text(result)

    def get_auth_error(self) -> Optional[str]:
        """
        Check stderr output from a stdio process for auth-related messages.
        Returns the error string if auth failure detected, None otherwise.
        """
        if not self._proc:
            return None
        stderr_stream = self._proc.stderr
        if stderr_stream is None:
            return None
        try:
            import select
            if self._proc.poll() is not None:
                # Process exited — check stderr
                stderr = stderr_stream.read()
            else:
                # Non-blocking read of available stderr
                ready, _, _ = select.select([stderr_stream], [], [], 0.5)
                if not ready:
                    return None
                stderr = stderr_stream.read(4096)
            text = stderr.decode("utf-8", errors="replace").lower()
            auth_markers = [
                "unauthorized", "authentication", "auth required",
                "login required", "token", "permission denied",
                "401", "403", "api key", "credential",
                "oauth", "authorization",
            ]
            for marker in auth_markers:
                if marker in text:
                    return stderr.decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Extract text from MCP tool result content."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            content = result.get("content", result)
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif "text" in item:
                            parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                return "\n".join(parts) if parts else json.dumps(result)
            if isinstance(content, str):
                return content
            if "text" in result:
                return str(result["text"])
            return json.dumps(result)
        return str(result)
