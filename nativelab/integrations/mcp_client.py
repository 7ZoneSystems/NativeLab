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
import aiohttp
import uuid
from typing import Any, Dict, List, Optional, Tuple


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

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    # ── public API ──────────────────────────────────────────────────────────

    def test_connection(self, transport: str, url: str) -> Tuple[bool, Any]:
        """
        Connect to MCP server, list tools, return (True, tools_list) or (False, error_string).
        Does NOT leave the connection open — call shutdown() after.
        """
        try:
            ok = self._connect(transport, url)
            if not ok:
                return False, "Connection failed"
            return True, list(self.tools)
        except Exception as e:
            return False, str(e)
        finally:
            self.shutdown()

    def execute(self, transport: str, url: str,
                tool_name: str, arguments: dict) -> Tuple[bool, Any]:
        """
        Connect, call a tool, return (True, result_text) or (False, error_string).
        """
        try:
            if not self._connect(transport, url):
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

    def _connect(self, transport: str, url: str) -> bool:
        self._transport = transport
        self._url = url
        if transport == "sse":
            return self._connect_sse(url)
        else:
            return self._connect_stdio(url)

    def _connect_sse(self, url: str) -> bool:
        """Connect to an SSE-based MCP server."""
        try:
            import aiohttp
            import asyncio

            loop = asyncio.new_event_loop()

            async def _do_connect():
                self._session = aiohttp.ClientSession()
                resp = await self._session.get(url, timeout=aiohttp.ClientTimeout(total=10))
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
        except ImportError:
            raise RuntimeError(
                "aiohttp is required for SSE MCP servers.\n"
                "Install it:  pip install aiohttp")
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
            async with self._session.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
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
            await self._session.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=5),
            )
        except Exception:
            pass

    def _connect_stdio(self, cmd: str) -> bool:
        """Connect to a stdio-based MCP server (including npx commands)."""
        try:
            self._proc = subprocess.Popen(
                cmd, shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
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
            try:
                chunk = self._proc.stdout.read(1)
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
                            more = self._proc.stdout.read(
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
        try:
            self._proc.stdin.write(header + payload)
            self._proc.stdin.flush()
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
        try:
            self._proc.stdin.write(header + payload)
            self._proc.stdin.flush()
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
