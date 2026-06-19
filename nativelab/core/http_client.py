"""
Centralized HTTP client for all NativeLab network operations.

Replaces scattered urllib.request/http.client calls with a unified interface.
Handles timeouts, retries, authentication, and error normalization.
"""

from __future__ import annotations

import json
import ssl
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from nativelab.core.llm_errors import explain_llm_error

DEFAULT_TIMEOUT = 30  # seconds
LONG_TIMEOUT = 300    # 5 min for generations
USER_AGENT = "NativeLab/1"


@dataclass
class HttpResponse:
    """Normalized HTTP response."""
    status: int
    data: Any
    raw_text: str
    headers: Dict[str, str]
    ok: bool

    @property
    def error_message(self) -> str:
        if self.ok:
            return ""
        if isinstance(self.data, dict) and "error" in self.data:
            err = self.data["error"]
            if isinstance(err, dict):
                return err.get("message", str(err))
            return str(err)
        return self.raw_text[:500]


class HttpError(Exception):
    """HTTP request failed."""
    def __init__(self, message: str, status: int = 0, response: Optional[HttpResponse] = None):
        super().__init__(message)
        self.status = status
        self.response = response


class NativeLabHttpClient:
    """
    Unified HTTP client for NativeLab.

    Usage:
        client = NativeLabHttpClient()
        resp = client.get("http://localhost:8080/health")
        resp = client.post("http://api.openai.com/v1/chat/completions", body={...}, headers={...})
        for chunk in client.stream("http://...", body={...}):
            process(chunk)
    """

    def __init__(self, default_timeout: int = DEFAULT_TIMEOUT, user_agent: str = USER_AGENT):
        self.default_timeout = default_timeout
        self.user_agent = user_agent
        self._ssl_ctx = ssl.create_default_context()

    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        auth_token: str = "",
    ) -> HttpResponse:
        """GET request. Returns HttpResponse."""
        hdrs = self._make_headers(headers, auth_token)
        req = urllib.request.Request(url, headers=hdrs, method="GET")
        return self._execute(req, timeout or self.default_timeout)

    def post(
        self,
        url: str,
        body: Any = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        auth_token: str = "",
    ) -> HttpResponse:
        """POST request with JSON body. Returns HttpResponse."""
        hdrs = self._make_headers(headers, auth_token)
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            data = body.encode("utf-8")
        elif isinstance(body, bytes):
            data = body
        else:
            data = b""
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        return self._execute(req, timeout or self.default_timeout)

    def stream(
        self,
        url: str,
        body: Any,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        auth_token: str = "",
    ) -> Iterator[str]:
        """
        POST request with SSE streaming. Yields data lines.
        Raises HttpError on non-200 responses.
        """
        hdrs = self._make_headers(headers, auth_token)
        hdrs["Accept"] = "text/event-stream"
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            data = body.encode("utf-8")
        else:
            data = body or b""

        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        timeout_val = timeout or LONG_TIMEOUT

        try:
            resp = urllib.request.urlopen(req, timeout=timeout_val, context=self._ssl_ctx)
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise HttpError(
                f"HTTP {e.code}: {e.reason}",
                status=e.code,
                response=HttpResponse(e.code, None, raw, {}, False),
            )
        except Exception as e:
            raise HttpError(str(e))

        try:
            for line in resp:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.strip()
                if line:
                    yield line
        finally:
            resp.close()

    def post_openai_stream(
        self,
        url: str,
        messages: list,
        model: str = "",
        api_key: str = "",
        max_tokens: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        repeat_penalty: float = 1.1,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        OpenAI-compatible streaming chat completion.
        Returns the full generated text.
        Calls on_token(token) for each token.
        """
        body: Dict[str, Any] = {
            "messages": messages,
            "stream": True,
        }
        if model:
            body["model"] = model
        if max_tokens > 0:
            body["max_tokens"] = max_tokens
        if temperature >= 0:
            body["temperature"] = temperature
        if top_p > 0:
            body["top_p"] = top_p
        if top_k > 0:
            body["top_k"] = top_k
        if repeat_penalty > 1.0:
            body["repeat_penalty"] = repeat_penalty
        if extra_body:
            body.update(extra_body)

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        result = []
        try:
            for line in self.stream(url, body, headers=headers, timeout=timeout):
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            result.append(token)
                            if on_token:
                                on_token(token)
                except json.JSONDecodeError:
                    continue
        except HttpError as e:
            if result:
                return "".join(result)
            raise

        return "".join(result)

    def post_anthropic_stream(
        self,
        url: str,
        messages: list,
        model: str = "",
        api_key: str = "",
        max_tokens: int = 1024,
        system: str = "",
        temperature: float = 0.7,
        timeout: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Anthropic-compatible streaming messages.
        Returns the full generated text.
        """
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system:
            body["system"] = system
        if temperature >= 0:
            body["temperature"] = temperature

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        result = []
        try:
            for line in self.stream(url, body, headers=headers, timeout=timeout):
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                try:
                    event = json.loads(data)
                    if event.get("type") == "content_block_delta":
                        text = event.get("delta", {}).get("text", "")
                        if text:
                            result.append(text)
                            if on_token:
                                on_token(text)
                except json.JSONDecodeError:
                    continue
        except HttpError as e:
            if result:
                return "".join(result)
            raise

        return "".join(result)

    # ── Internal ──────────────────────────────────────────────────

    def _make_headers(self, headers: Optional[Dict[str, str]], auth_token: str) -> Dict[str, str]:
        hdrs = {"User-Agent": self.user_agent}
        if headers:
            hdrs.update(headers)
        if auth_token:
            hdrs["Authorization"] = f"Bearer {auth_token}"
        return hdrs

    def _execute(self, req: urllib.request.Request, timeout: int) -> HttpResponse:
        try:
            resp = urllib.request.urlopen(req, timeout=timeout, context=self._ssl_ctx)
            raw = resp.read().decode("utf-8", errors="replace")
            resp_headers = dict(resp.headers)
            status = resp.status

            data = None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = raw

            return HttpResponse(
                status=status,
                data=data,
                raw_text=raw,
                headers=resp_headers,
                ok=200 <= status < 300,
            )
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            data = None
            if raw:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = raw
            return HttpResponse(
                status=e.code,
                data=data,
                raw_text=raw,
                headers=dict(e.headers) if hasattr(e, "headers") else {},
                ok=False,
            )
        except Exception as e:
            raise HttpError(str(e))


# ── Singleton accessor ────────────────────────────────────────────

_http_client: Optional[NativeLabHttpClient] = None

def get_http_client() -> NativeLabHttpClient:
    """Get the global HTTP client instance."""
    global _http_client
    if _http_client is None:
        _http_client = NativeLabHttpClient()
    return _http_client
