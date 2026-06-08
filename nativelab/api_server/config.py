from __future__ import annotations

import json
import secrets
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


API_SERVER_CONFIG_FILE = Path("./localllm/api_server_config.json")
ACTIVE_MODEL_REF = "__active_native_lab_engine__"


def generate_api_key() -> str:
    return "nl-" + secrets.token_urlsafe(32)


def detect_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    try:
        host = socket.gethostname()
        for ip in socket.gethostbyname_ex(host)[2]:
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    return "127.0.0.1"


@dataclass
class ApiServerConfig:
    host: str = "0.0.0.0"
    port: int = 8787
    protocol: str = "both"
    model_ref: str = ACTIVE_MODEL_REF
    auto_load_model: bool = True
    require_api_key: bool = True
    local_api_key: str = field(default_factory=generate_api_key)
    lan_api_key: str = field(default_factory=generate_api_key)
    load_timeout_ms: int = 600000
    allowed_origins: str = "*"

    @property
    def bind_label(self) -> str:
        if self.host in ("127.0.0.1", "localhost"):
            return "Localhost only"
        return "WiFi/LAN"

    @property
    def local_base_url(self) -> str:
        return f"http://127.0.0.1:{int(self.port)}/v1"

    @property
    def lan_base_url(self) -> str:
        return f"http://{detect_lan_ip()}:{int(self.port)}/v1"

    def supports_openai(self) -> bool:
        return self.protocol in ("openai", "both")

    def supports_anthropic(self) -> bool:
        return self.protocol in ("anthropic", "both")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["port"] = int(data.get("port") or 8787)
        data["protocol"] = str(data.get("protocol") or "both")
        data["host"] = str(data.get("host") or "0.0.0.0")
        return data

    def save(self) -> None:
        API_SERVER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        API_SERVER_CONFIG_FILE.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> "ApiServerConfig":
        if API_SERVER_CONFIG_FILE.exists():
            try:
                raw = json.loads(API_SERVER_CONFIG_FILE.read_text(encoding="utf-8"))
                cfg = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
                if not cfg.local_api_key:
                    cfg.local_api_key = generate_api_key()
                if not cfg.lan_api_key:
                    cfg.lan_api_key = generate_api_key()
                if cfg.protocol not in ("openai", "anthropic", "both"):
                    cfg.protocol = "both"
                if cfg.host not in ("127.0.0.1", "0.0.0.0"):
                    cfg.host = "0.0.0.0"
                return cfg
            except Exception:
                pass
        cfg = cls()
        cfg.save()
        return cfg
