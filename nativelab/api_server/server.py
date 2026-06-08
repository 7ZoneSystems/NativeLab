from __future__ import annotations

import ipaddress
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from nativelab.core.engine_status import active_engine_status

from .catalog import anthropic_model_list, model_catalog, openai_model_list, resolve_model_ref
from .config import ACTIVE_MODEL_REF, ApiServerConfig
from .protocol import (
    anthropic_message_response,
    error_payload,
    normalize_anthropic_messages,
    normalize_openai_messages,
    openai_chat_response,
    openai_completion_response,
    sampling_options,
)


class NativeLabApiServer:
    """OpenAI/Anthropic compatible local HTTP boundary for NativeLab engines."""

    def __init__(self, endpoints, config: ApiServerConfig | None = None):
        self.endpoints = endpoints
        self.config = config or ApiServerConfig.load()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._generation_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{int(self.config.port)}"

    def start(self) -> str:
        if self.is_running:
            return self.base_url
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_OPTIONS(self):
                self._send_json({"ok": True})

            def do_GET(self):
                owner._handle(self, "GET")

            def do_POST(self):
                owner._handle(self, "POST")

            def log_message(self, fmt, *args):
                return

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0:
                    return {}
                raw = self.rfile.read(length).decode("utf-8", errors="replace")
                return json.loads(raw) if raw.strip() else {}

            def _send_json(self, payload: dict[str, Any], status: int = 200):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", owner.config.allowed_origins or "*")
                self.send_header("Access-Control-Allow-Headers", "authorization, x-api-key, anthropic-version, content-type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def _send_sse(self, lines: list[str], status: int = 200):
                body = "".join(lines).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", owner.config.allowed_origins or "*")
                self.send_header("Access-Control-Allow-Headers", "authorization, x-api-key, anthropic-version, content-type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.config.host, int(self.config.port)), Handler)
        self.config.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.config.save()
        return self.base_url

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    def _handle(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        path = urlparse(handler.path).path.rstrip("/") or "/"
        if method == "GET" and path in ("/", "/health", "/v1/health"):
            handler._send_json(self._health_payload())
            return
        if not self._authorized(handler):
            handler._send_json(error_payload("Invalid API key for this client address.", "authentication_error", 401), 401)
            return
        try:
            if method == "GET":
                self._handle_get(handler, path)
            elif method == "POST":
                self._handle_post(handler, path)
            else:
                handler._send_json(error_payload("Unsupported method.", "method_not_allowed", 405), 405)
        except Exception as exc:
            handler._send_json(error_payload(str(exc), "server_error", 500), 500)

    def _handle_get(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        if path in ("/models", "/v1/models"):
            if self.config.supports_openai():
                handler._send_json(openai_model_list())
            else:
                handler._send_json(anthropic_model_list())
            return
        if path in ("/capabilities", "/v1/capabilities"):
            handler._send_json({
                "object": "nativelab.capabilities",
                "models": model_catalog(),
                "runtime": self._runtime_payload(),
                "endpoints": self._endpoint_payload(),
            })
            return
        handler._send_json(error_payload(f"Unknown endpoint: {path}", "not_found", 404), 404)

    def _handle_post(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        payload = handler._read_json()
        if path in ("/chat/completions", "/v1/chat/completions"):
            if not self.config.supports_openai():
                handler._send_json(error_payload("OpenAI-compatible mode is disabled.", "not_found", 404), 404)
                return
            self._handle_openai_chat(handler, payload)
            return
        if path in ("/completions", "/v1/completions"):
            if not self.config.supports_openai():
                handler._send_json(error_payload("OpenAI-compatible mode is disabled.", "not_found", 404), 404)
                return
            self._handle_openai_completion(handler, payload)
            return
        if path in ("/messages", "/v1/messages"):
            if not self.config.supports_anthropic():
                handler._send_json(error_payload("Anthropic-compatible mode is disabled.", "not_found", 404), 404)
                return
            self._handle_anthropic_messages(handler, payload)
            return
        handler._send_json(error_payload(f"Unknown endpoint: {path}", "not_found", 404), 404)

    def _handle_openai_chat(self, handler, payload: dict[str, Any]) -> None:
        model_id = self._ensure_model(payload.get("model"))
        messages, images = normalize_openai_messages(payload)
        text = self._generate(messages, images, sampling_options(payload))
        if payload.get("stream"):
            handler._send_sse(self._openai_chat_sse(model_id, text))
        else:
            handler._send_json(openai_chat_response(model_id, text))

    def _handle_openai_completion(self, handler, payload: dict[str, Any]) -> None:
        model_id = self._ensure_model(payload.get("model"))
        prompt = payload.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(str(p) for p in prompt)
        messages = [{"role": "user", "content": str(prompt or "")}]
        text = self._generate(messages, [], sampling_options(payload))
        if payload.get("stream"):
            handler._send_sse(self._openai_completion_sse(model_id, text))
        else:
            handler._send_json(openai_completion_response(model_id, text))

    def _handle_anthropic_messages(self, handler, payload: dict[str, Any]) -> None:
        model_id = self._ensure_model(payload.get("model"))
        messages, images = normalize_anthropic_messages(payload)
        text = self._generate(messages, images, sampling_options(payload))
        if payload.get("stream"):
            handler._send_sse(self._anthropic_sse(model_id, text))
        else:
            handler._send_json(anthropic_message_response(model_id, text))

    def _generate(self, messages: list[dict[str, str]], images: list[dict[str, Any]], opts: dict[str, Any]) -> str:
        if not messages:
            raise RuntimeError("No messages were provided.")
        with self._generation_lock:
            return self.endpoints.call_llm(
                messages=messages,
                image_data=images,
                n_predict=opts["n_predict"],
                temperature=opts["temperature"],
                top_p=opts["top_p"],
                repeat_penalty=opts["repeat_penalty"],
                top_k=opts["top_k"],
                min_p=opts["min_p"],
                typical_p=opts["typical_p"],
                seed=opts["seed"],
            )

    def _ensure_model(self, requested_model: Any) -> str:
        requested = str(requested_model or "").strip()
        selected_ref = str(self.config.model_ref or ACTIVE_MODEL_REF)
        target_ref = resolve_model_ref(requested) if requested else ""
        if not target_ref and selected_ref != ACTIVE_MODEL_REF:
            target_ref = selected_ref
        active_ref = str(self.endpoints.model_path or "")
        if target_ref:
            model_id = self._model_id_for_ref(target_ref)
            if target_ref != active_ref:
                if not self.config.auto_load_model:
                    raise RuntimeError(f"Requested model is not loaded: {model_id}")
                if not self.endpoints.request_load_model(target_ref):
                    raise RuntimeError(f"Could not request model load: {model_id}")
                if not self.endpoints.wait_until_loaded(int(self.config.load_timeout_ms)):
                    raise RuntimeError(f"Timed out loading model: {model_id}")
            return model_id
        if not self.endpoints.is_loaded:
            raise RuntimeError("No NativeLab engine is loaded.")
        return requested or self._model_id_for_ref(active_ref) or "nativelab-active"

    @staticmethod
    def _model_id_for_ref(model_ref: str) -> str:
        for row in model_catalog():
            if str(row.get("native_ref") or "") == str(model_ref or ""):
                return str(row.get("id") or row.get("name") or "nativelab-model")
        return str(model_ref or "nativelab-active").rstrip("/").split("/")[-1]

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        if not self.config.require_api_key:
            return True
        key = str(handler.headers.get("Authorization") or "").strip()
        if key.lower().startswith("bearer "):
            key = key[7:].strip()
        if not key:
            key = str(handler.headers.get("x-api-key") or "").strip()
        expected = self.config.local_api_key if self._is_loopback(handler.client_address[0]) else self.config.lan_api_key
        return bool(key and expected and key == expected)

    @staticmethod
    def _is_loopback(ip: str) -> bool:
        try:
            return ipaddress.ip_address(str(ip)).is_loopback
        except Exception:
            return str(ip).startswith("127.")

    def _health_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "name": "NativeLab API Server",
            "runtime": self._runtime_payload(),
            "endpoints": self._endpoint_payload(),
            "auth": {
                "required": bool(self.config.require_api_key),
                "local_key_header": "Authorization: Bearer <local_api_key>",
                "lan_key_header": "Authorization: Bearer <lan_api_key>",
            },
        }

    def _runtime_payload(self) -> dict[str, Any]:
        try:
            status = active_engine_status(self.endpoints.llama_engine, self.endpoints.api_engine, is_loading=self.endpoints.is_loading)
            return {
                "loaded": bool(status.is_loaded),
                "state": status.state,
                "status": status.status_text,
                "model": self.endpoints.model_name,
                "model_path": self.endpoints.model_path,
                "mode": self.endpoints.mode,
                "ctx": self.endpoints.ctx_value,
            }
        except Exception:
            return {"loaded": False, "state": "err", "status": "Runtime unavailable"}

    def _endpoint_payload(self) -> dict[str, Any]:
        return {
            "protocol": self.config.protocol,
            "bind": self.config.host,
            "local_base_url": self.config.local_base_url,
            "lan_base_url": self.config.lan_base_url,
            "openai_chat": f"{self.config.local_base_url}/chat/completions",
            "anthropic_messages": f"{self.config.local_base_url}/messages",
            "models": f"{self.config.local_base_url}/models",
            "capabilities": f"{self.config.local_base_url}/capabilities",
        }

    @staticmethod
    def _openai_chat_sse(model: str, text: str) -> list[str]:
        chunk_id = "chatcmpl-stream"
        return [
            "data: " + json.dumps({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
            }) + "\n\n",
            "data: " + json.dumps({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }) + "\n\n",
            "data: [DONE]\n\n",
        ]

    @staticmethod
    def _openai_completion_sse(model: str, text: str) -> list[str]:
        return [
            "data: " + json.dumps({
                "id": "cmpl-stream",
                "object": "text_completion",
                "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            }) + "\n\n",
            "data: [DONE]\n\n",
        ]

    @staticmethod
    def _anthropic_sse(model: str, text: str) -> list[str]:
        return [
            "event: message_start\ndata: " + json.dumps(anthropic_message_response(model, "")) + "\n\n",
            "event: content_block_delta\ndata: " + json.dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }) + "\n\n",
            "event: message_delta\ndata: " + json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            }) + "\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]
