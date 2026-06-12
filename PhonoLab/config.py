from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import CONFIG_FILE, atomic_write_text, ensure_dirs


def _cpu_count() -> int:
    return max(1, int(os.cpu_count() or 4))


def _bound_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _bound_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


@dataclass
class MobileConfig:
    ctx: int = 2048
    threads: int = min(4, _cpu_count())
    n_predict: int = 384
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_prompt_chars: int = 18000
    max_download_bytes: int = 7 * 1024 * 1024 * 1024
    llama_cli_path: str = ""
    llama_server_path: str = ""
    active_model: str = ""
    hf_token: str = ""
    user_agent: str = "PhonoLabMobile/1"
    stream_timeout_seconds: int = 180

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MobileConfig":
        cfg = cls()
        for key in asdict(cfg):
            if key in data:
                setattr(cfg, key, data[key])
        cfg.ctx = _bound_int(cfg.ctx, 1024, 512, 32768)
        cfg.threads = _bound_int(cfg.threads, min(4, _cpu_count()), 1, min(8, _cpu_count()))
        cfg.n_predict = _bound_int(cfg.n_predict, 384, 32, 4096)
        cfg.temperature = _bound_float(cfg.temperature, 0.7, 0.0, 2.0)
        cfg.top_p = _bound_float(cfg.top_p, 0.9, 0.05, 1.0)
        cfg.repeat_penalty = _bound_float(cfg.repeat_penalty, 1.1, 0.8, 2.0)
        cfg.max_prompt_chars = _bound_int(cfg.max_prompt_chars, 18000, 1000, 300000)
        cfg.max_download_bytes = _bound_int(
            cfg.max_download_bytes,
            7 * 1024 * 1024 * 1024,
            50 * 1024 * 1024,
            50 * 1024 * 1024 * 1024,
        )
        cfg.stream_timeout_seconds = _bound_int(cfg.stream_timeout_seconds, 180, 10, 1200)
        cfg.llama_cli_path = str(cfg.llama_cli_path or "")
        cfg.llama_server_path = str(cfg.llama_server_path or "")
        cfg.active_model = str(cfg.active_model or "")
        cfg.hf_token = str(cfg.hf_token or "")
        cfg.user_agent = str(cfg.user_agent or "PhonoLabMobile/1")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path = CONFIG_FILE) -> MobileConfig:
    ensure_dirs()
    if not path.exists():
        cfg = MobileConfig()
        save_config(cfg, path)
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return MobileConfig.from_dict(data)
    except Exception:
        pass
    return MobileConfig()


def save_config(config: MobileConfig, path: Path = CONFIG_FILE) -> None:
    ensure_dirs()
    atomic_write_text(path, json.dumps(config.to_dict(), indent=2, sort_keys=True))
