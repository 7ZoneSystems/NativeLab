from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from .endpoints import IntegrationEndpoints


class IntegrationHttpEndpoint:
    """Small localhost JSON server for external integration prototypes."""

    def __init__(
        self,
        endpoints: IntegrationEndpoints,
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        self.endpoints = endpoints
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        if self.is_running:
            return self.base_url

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_OPTIONS(self):
                self._send_json({"ok": True})

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/health":
                    self._send_json({"ok": True, "runtime": owner.endpoints.runtime()})
                    return
                self._send_json(owner.endpoints.handle(path))

            def do_POST(self):
                path = urlparse(self.path).path
                if path != "/call_llm":
                    self._send_json({"error": "unknown POST endpoint", "path": path}, 404)
                    return
                try:
                    payload = self._read_json()
                    text = owner.endpoints.call_llm(
                        messages=payload.get("messages"),
                        prompt=payload.get("prompt"),
                        system_prompt=payload.get("system_prompt"),
                        n_predict=int(payload.get("n_predict", 512)),
                        temperature=float(payload.get("temperature", 0.3)),
                        top_p=float(payload.get("top_p", 0.9)),
                        repeat_penalty=float(payload.get("repeat_penalty", 1.15)),
                    )
                    self._send_json({"text": text, "runtime": owner.endpoints.runtime()})
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)

            def log_message(self, fmt, *args):
                return

            def _read_json(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0:
                    return {}
                raw = self.rfile.read(length).decode("utf-8")
                return json.loads(raw) if raw.strip() else {}

            def _send_json(self, payload: Dict[str, Any], status: int = 200):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
                self.send_header("Access-Control-Allow-Headers", "content-type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None
