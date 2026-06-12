from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import MobileConfig, load_config, save_config
from .hardware import profile_hardware, recommended_threads
from .paths import MODELS_DIR, REGISTRY_FILE, atomic_write_text, ensure_dirs
from .safety import SafetyError, validate_model_path


@dataclass
class MobileModelConfig:
    path: str
    name: str = ""
    repo: str = ""
    quant: str = ""
    family: str = "default"
    ctx: int = 2048
    threads: int = 4
    n_predict: int = 384
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MobileModelConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in dict(data or {}).items() if k in fields})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MobileModelRegistry:
    def __init__(self, path: Path = REGISTRY_FILE):
        ensure_dirs()
        self.path = Path(path)
        self._models: dict[str, MobileModelConfig] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        rows = data.get("models") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            cfg = MobileModelConfig.from_dict(row)
            if cfg.path:
                self._models[cfg.path] = cfg

    def save(self) -> None:
        payload = {"version": 1, "models": [m.to_dict() for m in self.all()]}
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True))

    def all(self) -> list[MobileModelConfig]:
        return sorted(self._models.values(), key=lambda item: item.name.lower() or item.path.lower())

    def get(self, path: str) -> MobileModelConfig | None:
        return self._models.get(str(path))

    def add(self, path: str | Path, *, repo: str = "", quant: str = "", set_active: bool = True) -> MobileModelConfig:
        model_path = validate_model_path(path)
        cfg = self._models.get(str(model_path))
        if cfg is None:
            hw = profile_hardware()
            cfg = MobileModelConfig(
                path=str(model_path),
                name=model_path.name,
                repo=repo,
                quant=quant,
                ctx=2048 if hw.ram_total_mb < 6144 else 4096,
                threads=recommended_threads(hw),
                n_predict=384,
            )
            self._models[str(model_path)] = cfg
        else:
            cfg.repo = repo or cfg.repo
            cfg.quant = quant or cfg.quant
            cfg.name = cfg.name or model_path.name
        self.save()
        if set_active:
            app_cfg = load_config()
            app_cfg.active_model = str(model_path)
            app_cfg.ctx = cfg.ctx
            app_cfg.threads = cfg.threads
            app_cfg.n_predict = cfg.n_predict
            save_config(app_cfg)
        return cfg

    def remove(self, path: str | Path) -> None:
        key = str(Path(path).expanduser())
        self._models.pop(key, None)
        self.save()
        app_cfg = load_config()
        if app_cfg.active_model == key:
            app_cfg.active_model = ""
            save_config(app_cfg)

    def discover(self) -> list[MobileModelConfig]:
        if not MODELS_DIR.exists():
            return self.all()
        for gguf in sorted(MODELS_DIR.rglob("*.gguf")):
            try:
                self.add(gguf, set_active=False)
            except SafetyError:
                continue
        self.save()
        return self.all()


def get_registry() -> MobileModelRegistry:
    return MobileModelRegistry()
