"""
MCP server verification for AI-generated pipelines.

After the AI generates a pipeline containing MCP_SERVER blocks, this module
tests each server, auto-installs missing packages, retries on failure,
and reports results back to the build worker.
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


VALID_WEB_SEARCH_CATEGORIES = {
    "general", "images", "videos", "news", "science", "it", "files", "music", "social media"
}

WEB_SEARCH_CATEGORY_SYNONYMS: Dict[str, str] = {
    "tech": "it", "technology": "it", "code": "it", "coding": "it",
    "developer": "it", "programming": "it", "software": "it", "github": "it", "computer": "it",
    "paper": "science", "papers": "science", "academic": "science", "research": "science",
    "arxiv": "science", "scientific": "science", "scholar": "science",
    "finance": "news", "economy": "news", "politics": "news", "world": "news",
    "headlines": "news", "current": "news", "breaking": "news",
    "image": "images", "picture": "images", "pictures": "images", "photo": "images", "photos": "images",
    "video": "videos", "youtube": "videos", "movies": "videos", "clips": "videos",
    "audio": "music", "song": "music", "songs": "music", "podcast": "music",
    "social": "social media", "twitter": "social media", "reddit": "social media",
    "web": "general", "all": "general", "search": "general",
}


@dataclass
class McpVerifyResult:
    """Result of verifying one MCP server or tool block."""
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
    block_type: str = "mcp_server"
    auto_repaired: bool = False
    repair_notes: str = ""


@dataclass
class McpVerificationReport:
    """Aggregated report for all tool/MCP blocks in a pipeline."""
    results: List[McpVerifyResult] = field(default_factory=list)
    all_ok: bool = True
    fixed_blocks: List[int] = field(default_factory=list)
    removed_blocks: List[int] = field(default_factory=list)

    @property
    def has_mcp(self) -> bool:
        return any(r.block_type == "mcp_server" for r in self.results)

    @property
    def has_tools(self) -> bool:
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

    @property
    def web_search_results(self) -> List[McpVerifyResult]:
        return [r for r in self.results if r.block_type == "web_search"]

    @property
    def mcp_results(self) -> List[McpVerifyResult]:
        return [r for r in self.results if r.block_type == "mcp_server"]


def extract_mcp_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract MCP server blocks from a pipeline block list."""
    return [b for b in blocks if b.get("btype") == "mcp_server"]


def extract_tool_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract all tool blocks (MCP server and Web Search) from pipeline block list."""
    return [b for b in blocks if b.get("btype") in ("mcp_server", "web_search")]


def verify_web_search_block(
    block: Dict[str, Any],
    *,
    log_cb: Optional[Callable[[str], None]] = None,
) -> McpVerifyResult:
    """
    Verify and normalize a Web Search block (SearXNG).
    Validates categories with synonym repair, clamps limits, checks subsystem.
    """
    bid = block.get("bid", 0)
    label = block.get("label", "Web Search")
    meta = block.setdefault("metadata", {})

    raw_cats = meta.get("ws_categories", ["general"])
    if not isinstance(raw_cats, list):
        raw_cats = [str(raw_cats)]

    clean_cats = []
    for c in raw_cats:
        c_str = str(c).strip().lower()
        if c_str in VALID_WEB_SEARCH_CATEGORIES:
            clean_cats.append(c_str)
        elif c_str in WEB_SEARCH_CATEGORY_SYNONYMS:
            clean_cats.append(WEB_SEARCH_CATEGORY_SYNONYMS[c_str])
    if not clean_cats:
        clean_cats = ["general"]

    seen = set()
    deduped = []
    for c in clean_cats:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    meta["ws_categories"] = deduped

    lang = str(meta.get("ws_language") or "en").strip().lower()
    meta["ws_language"] = lang[:10] if lang else "en"

    try:
        max_r = int(meta.get("ws_max_results", 10))
    except Exception:
        max_r = 10
    meta["ws_max_results"] = min(50, max(1, max_r))

    try:
        timeout = int(meta.get("ws_timeout", 10))
    except Exception:
        timeout = 10
    meta["ws_timeout"] = min(30, max(3, timeout))

    fmt = str(meta.get("ws_output_format") or "text").strip().lower()
    meta["ws_output_format"] = fmt if fmt in ("text", "json") else "text"

    result = McpVerifyResult(
        block_bid=bid,
        block_label=label,
        transport="in-process",
        url="searxng",
        tool_name="web_search",
        success=True,
        block_type="web_search",
    )

    try:
        import nativelab.web_search  # noqa: F401
        if log_cb:
            log_cb(f"  ✓ Web Search '{label}': verified with categories {deduped}")
    except ImportError as e:
        result.success = False
        result.error = f"SearXNG web search dependencies missing: {e}"
        if log_cb:
            log_cb(f"  ✗ Web Search '{label}': {result.error}")
    except Exception as e:
        result.error = str(e)

    return result


def heal_connections_after_removal(
    connections: List[Dict[str, Any]],
    removed_bids: List[int],
) -> List[Dict[str, Any]]:
    """
    Auto-bridge connections across removed blocks so the pipeline graph remains connected.
    For each removed block X, all (A -> X) and (X -> B) become (A -> B).
    Connections involving X are removed, duplicate or self-loop edges are pruned.
    """
    if not removed_bids or not connections:
        return [dict(c) for c in connections]

    removed_set = set(removed_bids)
    in_edges: Dict[int, List[Dict[str, Any]]] = {}
    out_edges: Dict[int, List[Dict[str, Any]]] = {}
    clean_connections = []

    for c in connections:
        from_id = c.get("from_block_id")
        to_id = c.get("to_block_id")
        if from_id in removed_set or to_id in removed_set:
            out_edges.setdefault(from_id, []).append(c)
            in_edges.setdefault(to_id, []).append(c)
        else:
            clean_connections.append(dict(c))

    seen_edges = {
        (c["from_block_id"], c.get("from_port", "E"), c["to_block_id"], c.get("to_port", "W"))
        for c in clean_connections
    }

    for r_bid in removed_bids:
        preds = in_edges.get(r_bid, [])
        succs = out_edges.get(r_bid, [])
        for p in preds:
            from_bid = p.get("from_block_id")
            from_port = p.get("from_port", "E")
            if from_bid in removed_set:
                continue
            for s in succs:
                to_bid = s.get("to_block_id")
                to_port = s.get("to_port", "W")
                if to_bid in removed_set or from_bid == to_bid:
                    continue
                edge_key = (from_bid, from_port, to_bid, to_port)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    clean_connections.append({
                        "from_block_id": from_bid,
                        "from_port": from_port,
                        "to_block_id": to_bid,
                        "to_port": to_port,
                        "is_loop": False,
                        "loop_times": 1,
                    })

    return clean_connections


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


def _infer_arg_name(block: Dict[str, Any], tool_dict: Dict[str, Any]) -> None:
    """If block's mcp_arg_name is empty or default 'input', infer best parameter from tool schema."""
    meta = block.setdefault("metadata", {})
    curr = str(meta.get("mcp_arg_name") or "").strip()
    schema = tool_dict.get("inputSchema") or tool_dict.get("parameters") or {}
    props = schema.get("properties") if isinstance(schema, dict) else {}
    if not isinstance(props, dict) or not props:
        return
    # If currently specified arg name is already valid, keep it
    if curr and curr in props:
        return

    # Check required fields
    raw_required = schema.get("required")
    required_fields: List[str] = [str(r) for r in raw_required] if isinstance(raw_required, list) else []
    for req in required_fields:
        if req in props:
            meta["mcp_arg_name"] = str(req)
            return

    # Check common semantic parameter names for tools
    for cand in ("query", "q", "text", "prompt", "input_text", "path", "file_path", "url", "message", "content"):
        if cand in props:
            meta["mcp_arg_name"] = cand
            return

    # Default to first property in schema
    first_key = next(iter(props.keys()))
    meta["mcp_arg_name"] = str(first_key)


def _find_tool_for_block(
    block: Dict[str, Any],
    available_tools: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Find the best matching tool name for a block's configured tool using multi-tier
    matching (exact, case-insensitive, normalized, substring, token overlap, and fuzzy ratio).
    Also auto-infers input argument name from schema.
    """
    configured = str(block.get("metadata", {}).get("mcp_tool_name", "")).strip()
    if not configured or not available_tools:
        return None

    # 1. Exact match
    for t in available_tools:
        name = str(t.get("name") or "")
        if name == configured:
            _infer_arg_name(block, t)
            return configured

    # 2. Case-insensitive match
    cfg_lower = configured.lower()
    for t in available_tools:
        name = str(t.get("name") or "")
        if name.lower() == cfg_lower:
            _infer_arg_name(block, t)
            return name

    # 3. Normalized punctuation / snake / kebab / camel match
    def _norm(s: str) -> str:
        return re.sub(r"[-_.\s]+", "", s.lower())

    cfg_norm = _norm(configured)
    for t in available_tools:
        name = str(t.get("name") or "")
        if _norm(name) == cfg_norm:
            _infer_arg_name(block, t)
            return name

    # 4. Substring containment match
    substring_matches = []
    for t in available_tools:
        name = str(t.get("name") or "")
        name_lower = name.lower()
        if cfg_lower in name_lower or name_lower in cfg_lower:
            substring_matches.append(name)
    if substring_matches:
        best = min(substring_matches, key=lambda n: abs(len(n) - len(configured)))
        match_tool = next((t for t in available_tools if t.get("name") == best), None)
        if match_tool:
            _infer_arg_name(block, match_tool)
        return best

    # 5. Fuzzy match using difflib & token overlap
    best_name = None
    best_score = 0.0
    cfg_tokens = set(re.findall(r"[a-z0-9]+", cfg_lower))
    for t in available_tools:
        name = str(t.get("name") or "")
        name_lower = name.lower()
        name_tokens = set(re.findall(r"[a-z0-9]+", name_lower))
        jaccard = (
            len(cfg_tokens & name_tokens) / max(1, len(cfg_tokens | name_tokens))
            if (cfg_tokens and name_tokens)
            else 0.0
        )
        ratio = difflib.SequenceMatcher(None, cfg_lower, name_lower).ratio()
        score = max(jaccard, ratio)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name and best_score >= 0.55:
        match_tool = next((t for t in available_tools if t.get("name") == best_name), None)
        if match_tool:
            _infer_arg_name(block, match_tool)
        return best_name

    return None


def verify_all_tool_blocks(
    blocks: List[Dict[str, Any]],
    *,
    max_retries: int = 2,
    auto_install: bool = True,
    log_cb: Optional[Callable[[str], None]] = None,
    abort_cb: Optional[Callable[[], bool]] = None,
    max_workers: int = 4,
) -> McpVerificationReport:
    """
    Verify all tool blocks (MCP servers and Web Search) in a pipeline concurrently.
    Runs probes across a worker thread pool for high efficiency and fail-safe execution.
    """
    mcp_blocks = extract_mcp_blocks(blocks)
    web_blocks = [b for b in blocks if b.get("btype") == "web_search"]
    all_tool_blocks = mcp_blocks + web_blocks
    report = McpVerificationReport()

    if not all_tool_blocks:
        return report

    if log_cb:
        log_cb(f"Verifying {len(all_tool_blocks)} tool block(s) concurrently...")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _verify_one(b: Dict[str, Any]) -> McpVerifyResult:
        if abort_cb and abort_cb():
            return McpVerifyResult(
                block_bid=b.get("bid", 0),
                block_label=b.get("label", "Tool"),
                transport="aborted",
                url="",
                tool_name="",
                success=False,
                error="Cancelled",
                block_type=b.get("btype", "tool"),
            )
        if b.get("btype") == "web_search":
            return verify_web_search_block(b, log_cb=log_cb)
        return verify_mcp_block(
            b,
            max_retries=max_retries,
            auto_install=auto_install,
            log_cb=log_cb,
        )

    workers = min(max_workers, len(all_tool_blocks) or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_verify_one, b): b for b in all_tool_blocks}
        for future in as_completed(futures):
            if abort_cb and abort_cb():
                if log_cb:
                    log_cb("Tool verification cancelled.")
                break
            try:
                res = future.result()
                report.results.append(res)
                if not res.success:
                    report.all_ok = False
            except Exception as e:
                b = futures[future]
                err_res = McpVerifyResult(
                    block_bid=b.get("bid", 0),
                    block_label=b.get("label", "Tool"),
                    transport="unknown",
                    url="",
                    tool_name="",
                    success=False,
                    error=str(e),
                    block_type=b.get("btype", "tool"),
                )
                report.results.append(err_res)
                report.all_ok = False

    passed = len(report.passed)
    failed = len(report.failed)
    if log_cb:
        if failed:
            log_cb(f"Tool verification: {passed} passed, {failed} failed")
        else:
            log_cb(f"Tool verification: all {passed} tool(s) OK")

    return report


def verify_all_mcp_blocks(
    blocks: List[Dict[str, Any]],
    *,
    max_retries: int = 2,
    auto_install: bool = True,
    log_cb: Optional[Callable[[str], None]] = None,
    abort_cb: Optional[Callable[[], bool]] = None,
) -> McpVerificationReport:
    """Backwards-compatible wrapper calling verify_all_tool_blocks."""
    return verify_all_tool_blocks(
        blocks,
        max_retries=max_retries,
        auto_install=auto_install,
        log_cb=log_cb,
        abort_cb=abort_cb,
    )


def fix_mcp_blocks_after_verification(
    blocks: List[Dict[str, Any]],
    report: McpVerificationReport,
    *,
    connections: Optional[List[Dict[str, Any]]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Fix or remove tool blocks based on verification results.
    If connections list is provided and blocks are removed, connections are automatically healed.
    Returns (fixed_blocks, messages).
    """
    fixed = []
    messages = []

    result_map = {r.block_bid: r for r in report.results}

    for block in blocks:
        bid = block.get("bid", 0)
        btype = block.get("btype", "")

        if bid not in result_map:
            fixed.append(block)
            continue

        result = result_map[bid]

        if result.block_type == "web_search":
            if result.success:
                fixed.append(block)
                if result.error:
                    messages.append(f"Web Search '{result.block_label}': noted {result.error}")
            else:
                # Web search failed (e.g. dependencies missing)
                messages.append(f"Web Search '{result.block_label}': {result.error}")
                fixed.append(block)  # Keep block with normalized metadata
            continue

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
                    _infer_arg_name(block, result.tools_found[0])
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

    # If blocks were removed and connections list was provided, auto-heal connections
    if report.removed_blocks and connections is not None:
        healed = heal_connections_after_removal(connections, report.removed_blocks)
        connections.clear()
        connections.extend(healed)
        messages.append(f"Auto-healed {len(connections)} connection(s) across removed tool block(s).")
        if log_cb:
            log_cb(f"  Auto-healed pipeline graph connections across removed tool block(s).")

    return fixed, messages
