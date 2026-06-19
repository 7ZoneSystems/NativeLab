"""
MCP server verification for AI-generated pipelines.

After the AI generates a pipeline containing MCP_SERVER blocks, this module
tests each server, auto-installs missing packages, retries on failure,
and reports results back to the build worker.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class McpVerifyResult:
    """Result of verifying one MCP server block."""
    block_bid: int
    block_label: str
    transport: str
    url: str
    tool_name: str
    success: bool
    tools_found: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    install_attempted: bool = False
    install_success: bool = False
    retries: int = 0
    auth_needed: bool = False
    auth_error_detail: str = ""


@dataclass
class McpVerificationReport:
    """Aggregated report for all MCP blocks in a pipeline."""
    results: List[McpVerifyResult] = field(default_factory=list)
    all_ok: bool = True
    fixed_blocks: List[int] = field(default_factory=list)
    removed_blocks: List[int] = field(default_factory=list)

    @property
    def has_mcp(self) -> bool:
        return len(self.results) > 0

    @property
    def failed(self) -> List[McpVerifyResult]:
        return [r for r in self.results if not r.success]

    @property
    def passed(self) -> List[McpVerifyResult]:
        return [r for r in self.results if r.success]

    @property
    def auth_needed(self) -> List[McpVerifyResult]:
        return [r for r in self.results if r.auth_needed]


def extract_mcp_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract MCP server blocks from a pipeline block list."""
    return [b for b in blocks if b.get("btype") == "mcp_server"]


def _is_npx_command(url: str) -> bool:
    """Check if a stdio URL is an npx command."""
    return bool(re.match(r"^\s*npx\s+", url or ""))


def _npm_package_name(url: str) -> Optional[str]:
    """Extract npm package name from an npx command."""
    match = re.search(r"(?:npx\s+(?:-y\s+)?)(@[^\s]+/[^\s]+|[^\s]+)", url or "")
    if match:
        return match.group(1)
    return None


def _install_npm_package(package: str, log_cb: Optional[Callable[[str], None]] = None) -> bool:
    """Try to install an npm package globally."""
    if log_cb:
        log_cb(f"  Installing npm package: {package}")
    try:
        result = subprocess.run(
            ["npm", "install", "-g", package],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            if log_cb:
                log_cb(f"  ✓ Installed {package}")
            return True
        if log_cb:
            log_cb(f"  ✗ npm install failed: {result.stderr[:200]}")
        return False
    except FileNotFoundError:
        if log_cb:
            log_cb("  ✗ npm not found - cannot auto-install packages")
        return False
    except subprocess.TimeoutExpired:
        if log_cb:
            log_cb(f"  ✗ npm install timed out for {package}")
        return False
    except Exception as e:
        if log_cb:
            log_cb(f"  ✗ npm install error: {e}")
        return False


def _parse_env_block(text: str) -> Dict[str, str]:
    """Parse KEY=value lines into a dict."""
    env = {}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key:
                env[key] = val
    return env


def _test_stdio_server(url: str, timeout: float = 15.0, *,
                       auth_token: Optional[str] = None,
                       auth_env: Optional[Dict[str, str]] = None,
                       log_cb: Optional[Callable[[str], None]] = None) -> Tuple[bool, List[Dict], str]:
    """
    Start a stdio MCP server, run initialize + tools/list, return (ok, tools, error).
    """
    try:
        import os
        env = dict(os.environ)
        if auth_token:
            env["MCP_AUTH_TOKEN"] = auth_token
        if auth_env:
            env.update(auth_env)
        proc = subprocess.Popen(
            url, shell=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=False,
            env=env,
        )
    except Exception as e:
        return False, [], f"Failed to start process: {e}"

    tools: List[Dict[str, Any]] = []
    error = ""

    try:
        # Send initialize
        init_msg = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "nativelab-verify", "version": "1.0.0"},
            },
        }).encode("utf-8")
        header = f"Content-Length: {len(init_msg)}\r\n\r\n".encode("utf-8")
        if proc.stdin is None:
            return False, [], "Server stdin not available"
        proc.stdin.write(header + init_msg)
        proc.stdin.flush()

        # Read initialize response
        resp = _read_lsp_response(proc, timeout=timeout)
        if resp is None:
            error = "Server did not respond to initialize"
            return False, [], error
        if "error" in resp:
            error = f"Initialize error: {resp['error']}"
            return False, [], error

        # Send initialized notification
        notif = json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized", "params": {},
        }).encode("utf-8")
        notif_header = f"Content-Length: {len(notif)}\r\n\r\n".encode("utf-8")
        proc.stdin.write(notif_header + notif)
        proc.stdin.flush()

        # Send tools/list
        tools_msg = json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        }).encode("utf-8")
        tools_header = f"Content-Length: {len(tools_msg)}\r\n\r\n".encode("utf-8")
        proc.stdin.write(tools_header + tools_msg)
        proc.stdin.flush()

        # Read tools/list response
        resp2 = _read_lsp_response(proc, timeout=timeout)
        if resp2 is None:
            error = "Server did not respond to tools/list"
            return False, [], error
        if "error" in resp2:
            error = f"tools/list error: {resp2['error']}"
            return False, [], error

        result = resp2.get("result", {})
        tools = result.get("tools", [])
        if not tools:
            error = "Server returned no tools"
            return False, [], error

        return True, tools, ""

    except Exception as e:
        error = str(e)
        return False, [], error
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _read_lsp_response(proc: subprocess.Popen, timeout: float = 15.0) -> Optional[dict]:
    """Read one LSP-framed JSON-RPC response from a subprocess."""
    import select

    stdout = proc.stdout
    if stdout is None:
        return None

    deadline = time.time() + timeout
    buffer = b""

    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        ready, _, _ = select.select([stdout], [], [], min(remaining, 0.5))
        if not ready:
            if proc.poll() is not None:
                return None
            continue

        chunk = stdout.read(1)
        if not chunk:
            if proc.poll() is not None:
                return None
            continue

        buffer += chunk

        header_end = buffer.find(b"\r\n\r\n")
        if header_end < 0:
            continue

        header = buffer[:header_end].decode("utf-8", errors="replace")
        match = re.search(r"Content-Length:\s*(\d+)", header, re.IGNORECASE)
        if not match:
            buffer = buffer[header_end + 4:]
            continue

        content_len = int(match.group(1))
        body_start = header_end + 4

        while len(buffer) < body_start + content_len:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([stdout], [], [], min(remaining, 0.5))
            if not ready:
                continue
            more = stdout.read(content_len - (len(buffer) - body_start))
            if not more:
                break
            buffer += more

        body = buffer[body_start:body_start + content_len]
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    return None


def _test_sse_server(url: str, timeout: float = 15.0, *,
                     auth_token: Optional[str] = None,
                     log_cb: Optional[Callable[[str], None]] = None) -> Tuple[bool, List[Dict], str]:
    """
    Test an SSE MCP server by sending initialize + tools/list via HTTP POST.
    """
    import urllib.request
    import urllib.error
    try:

        # Try POST to the SSE endpoint
        init_payload = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "nativelab-verify", "version": "1.0.0"},
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=init_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if auth_token:
            req.add_header("Authorization", f"Bearer {auth_token}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")

        # Parse response (might be SSE format or direct JSON)
        result_data = None
        for line in body.strip().split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        msg = json.loads(data)
                        if "result" in msg:
                            result_data = msg["result"]
                    except json.JSONDecodeError:
                        continue

        if result_data is None:
            try:
                msg = json.loads(body)
                if "result" in msg:
                    result_data = msg["result"]
                elif "error" in msg:
                    return False, [], f"Server error: {msg['error']}"
            except json.JSONDecodeError:
                return False, [], "Could not parse server response"

        if result_data is None:
            return False, [], "No result in server response"

        # Now request tools/list
        tools_payload = json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        }).encode("utf-8")

        req2 = urllib.request.Request(
            url, data=tools_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if auth_token:
            req2.add_header("Authorization", f"Bearer {auth_token}")
        with urllib.request.urlopen(req2, timeout=timeout) as resp2:
            body2 = resp2.read().decode("utf-8", errors="replace")

        tools = []
        for line in body2.strip().split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        msg = json.loads(data)
                        if "result" in msg:
                            tools = msg["result"].get("tools", [])
                    except json.JSONDecodeError:
                        continue

        if not tools:
            try:
                msg = json.loads(body2)
                tools = msg.get("result", {}).get("tools", [])
            except Exception:
                pass

        if not tools:
            return False, [], "Server returned no tools"

        return True, tools, ""

    except urllib.error.HTTPError as e:
        return False, [], f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, [], f"Connection error: {e.reason}"
    except Exception as e:
        return False, [], str(e)


def _extract_error_detail(error: str) -> str:
    """Extract a human-readable detail from an error string."""
    if not error:
        return "unknown error"
    # Common patterns
    for pattern in [
        r"Cannot find module '([^']+)'",
        r"command not found",
        r"No such file or directory",
        r"ECONNREFUSED",
        r"ENOENT",
    ]:
        if re.search(pattern, error, re.IGNORECASE):
            return error[:200]
    return error[:200]


def _is_auth_error(error: str) -> bool:
    """Check if an error string indicates an authentication/authorization failure."""
    lower = (error or "").lower()
    markers = [
        "unauthorized", "authentication", "auth required",
        "login required", "login", "permission denied",
        "401", "403", "api key", "credential",
        "oauth", "authorization", "token",
    ]
    return any(m in lower for m in markers)


def verify_mcp_block(
    block: Dict[str, Any],
    *,
    max_retries: int = 2,
    auto_install: bool = True,
    log_cb: Optional[Callable[[str], None]] = None,
) -> McpVerifyResult:
    """
    Verify a single MCP server block.
    Tests connection, auto-installs npm packages if needed, retries on failure.
    Detects auth errors and sets auth_needed flag.
    """
    meta = block.get("metadata", {})
    bid = block.get("bid", 0)
    label = block.get("label", "MCP Server")
    transport = meta.get("mcp_transport", "sse")
    url = meta.get("mcp_url", "")
    tool_name = meta.get("mcp_tool_name", "")
    auth_token = meta.get("mcp_auth_token", "")
    auth_env_text = meta.get("mcp_auth_env", "")
    auth_env = _parse_env_block(auth_env_text)

    result = McpVerifyResult(
        block_bid=bid,
        block_label=label,
        transport=transport,
        url=url,
        tool_name=tool_name,
        success=False,
    )

    if not url:
        result.error = "No server URL configured"
        return result

    if log_cb:
        log_cb(f"  Testing MCP server '{label}': {transport} → {url[:60]}")

    # First attempt (with auth if provided)
    if transport == "stdio":
        ok, tools, err = _test_stdio_server(
            url, auth_token=auth_token or None, auth_env=auth_env or None, log_cb=log_cb)
    else:
        ok, tools, err = _test_sse_server(
            url, auth_token=auth_token or None, log_cb=log_cb)

    if ok:
        result.success = True
        result.tools_found = tools
        if log_cb:
            log_cb(f"  ✓ '{label}' connected - {len(tools)} tool(s) found")
        return result

    # Check for auth-specific errors
    if _is_auth_error(err):
        result.auth_needed = True
        result.auth_error_detail = err[:300]
        if log_cb:
            log_cb(f"  🔒 '{label}' requires authentication: {_extract_error_detail(err)}")
        result.error = err
        return result

    if log_cb:
        log_cb(f"  ✗ '{label}' failed: {_extract_error_detail(err)}")

    # Auto-install for npx commands
    if auto_install and transport == "stdio" and _is_npx_command(url):
        pkg = _npm_package_name(url)
        if pkg:
            result.install_attempted = True
            if log_cb:
                log_cb(f"  Attempting to install npm package: {pkg}")
            installed = _install_npm_package(pkg, log_cb=log_cb)
            result.install_success = installed
            if installed:
                # Retry after install
                result.retries += 1
                if log_cb:
                    log_cb(f"  Retrying after install (attempt 2/{max_retries + 1})...")
                ok2, tools2, err2 = _test_stdio_server(
                    url, auth_token=auth_token or None, auth_env=auth_env or None, log_cb=log_cb)
                if ok2:
                    result.success = True
                    result.tools_found = tools2
                    if log_cb:
                        log_cb(f"  ✓ '{label}' connected after install - {len(tools2)} tool(s)")
                    return result
                if log_cb:
                    log_cb(f"  ✗ Still failed after install: {_extract_error_detail(err2)}")
                err = err2

    # Retry loop (without install)
    for attempt in range(max_retries):
        result.retries += 1
        if log_cb:
            log_cb(f"  Retry {attempt + 1}/{max_retries}...")
        time.sleep(1)

        if transport == "stdio":
            ok_r, tools_r, err_r = _test_stdio_server(
                url, auth_token=auth_token or None, auth_env=auth_env or None, log_cb=log_cb)
        else:
            ok_r, tools_r, err_r = _test_sse_server(
                url, auth_token=auth_token or None, log_cb=log_cb)

        if ok_r:
            result.success = True
            result.tools_found = tools_r
            if log_cb:
                log_cb(f"  ✓ '{label}' connected on retry - {len(tools_r)} tool(s)")
            return result
        err = err_r
        # Check for auth error on retry too
        if _is_auth_error(err_r):
            result.auth_needed = True
            result.auth_error_detail = err_r[:300]
            if log_cb:
                log_cb(f"  🔒 '{label}' requires authentication: {_extract_error_detail(err_r)}")
            result.error = err_r
            return result
        if log_cb:
            log_cb(f"  ✗ Retry failed: {_extract_error_detail(err_r)}")

    result.error = err
    return result


def verify_all_mcp_blocks(
    blocks: List[Dict[str, Any]],
    *,
    max_retries: int = 2,
    auto_install: bool = True,
    log_cb: Optional[Callable[[str], None]] = None,
    abort_cb: Optional[Callable[[], bool]] = None,
) -> McpVerificationReport:
    """
    Verify all MCP server blocks in a pipeline.
    Returns a report with per-block results.
    """
    mcp_blocks = extract_mcp_blocks(blocks)
    report = McpVerificationReport()

    if not mcp_blocks:
        return report

    if log_cb:
        log_cb(f"Verifying {len(mcp_blocks)} MCP server block(s)...")

    for block in mcp_blocks:
        if abort_cb and abort_cb():
            if log_cb:
                log_cb("MCP verification cancelled.")
            break

        result = verify_mcp_block(
            block,
            max_retries=max_retries,
            auto_install=auto_install,
            log_cb=log_cb,
        )
        report.results.append(result)

        if not result.success:
            report.all_ok = False

    passed = len(report.passed)
    failed = len(report.failed)
    if log_cb:
        if failed:
            log_cb(f"MCP verification: {passed} passed, {failed} failed")
        else:
            log_cb(f"MCP verification: all {passed} server(s) OK")

    return report


def _find_tool_for_block(
    block: Dict[str, Any],
    available_tools: List[Dict[str, Any]],
) -> Optional[str]:
    """Find the best matching tool name for a block's configured tool."""
    configured = block.get("metadata", {}).get("mcp_tool_name", "")
    if not configured:
        return None

    # Exact match
    for t in available_tools:
        if t.get("name") == configured:
            return configured

    # Case-insensitive match
    for t in available_tools:
        if t.get("name", "").lower() == configured.lower():
            return t["name"]

    return None


def fix_mcp_blocks_after_verification(
    blocks: List[Dict[str, Any]],
    report: McpVerificationReport,
    *,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Fix or remove MCP blocks based on verification results.
    Returns (fixed_blocks, messages) where messages describe what was changed.
    """
    fixed = []
    messages = []

    result_map = {r.block_bid: r for r in report.results}

    for block in blocks:
        bid = block.get("bid", 0)

        if bid not in result_map:
            fixed.append(block)
            continue

        result = result_map[bid]

        if result.success:
            # Update tools list from verification
            block["metadata"]["mcp_connected"] = True
            block["metadata"]["mcp_tools"] = result.tools_found

            # Verify configured tool still exists
            tool_name = _find_tool_for_block(block, result.tools_found)
            if tool_name and tool_name != block["metadata"].get("mcp_tool_name"):
                block["metadata"]["mcp_tool_name"] = tool_name
                messages.append(
                    f"MCP '{result.block_label}': tool name updated to '{tool_name}'"
                )
            elif not tool_name and result.tools_found:
                # Configured tool not found, pick first available
                first_tool = result.tools_found[0].get("name", "")
                if first_tool:
                    block["metadata"]["mcp_tool_name"] = first_tool
                    messages.append(
                        f"MCP '{result.block_label}': configured tool not found, "
                        f"switched to '{first_tool}'"
                    )

            fixed.append(block)
        elif result.auth_needed:
            # Server needs auth - keep block but mark it unconfigured
            block["metadata"]["mcp_connected"] = False
            block["metadata"]["mcp_auth_required"] = True
            messages.append(
                f"MCP '{result.block_label}' needs authentication - "
                f"configure token/env vars before running"
            )
            if log_cb:
                log_cb(f"  🔒 MCP block '{result.block_label}' kept - needs auth setup")
            fixed.append(block)
        else:
            # Server failed verification - remove the block
            report.removed_blocks.append(bid)
            messages.append(
                f"MCP '{result.block_label}' removed: server unreachable "
                f"({result.error[:80]})"
            )
            if log_cb:
                log_cb(f"  Removed MCP block '{result.block_label}' - server failed verification")

    return fixed, messages
