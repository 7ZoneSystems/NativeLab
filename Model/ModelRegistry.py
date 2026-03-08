from imports.import_global import Dict, dataclass, Path, json
from .model_family import *
from GlobalConfig.config_global import DEFAULT_CTX, DEFAULT_THREADS, DEFAULT_N_PRED, CUSTOM_MODELS_FILE, MODEL_CONFIGS_FILE, MODELS_DIR
# ═════════════════════════════ MODEL REGISTRY ═══════════════════════════════
@dataclass
class ModelConfig:
    path:           str
    role:           str   = "general"
    threads:        int   = DEFAULT_THREADS
    ctx:            int   = DEFAULT_CTX
    temperature:    float = 0.7
    top_p:          float = 0.9
    repeat_penalty: float = 1.1
    n_predict:      int   = DEFAULT_N_PRED
    # auto-detected, stored for display
    family:         str   = "default"

    def to_dict(self) -> Dict:
        return {
            "path": self.path, "role": self.role,
            "threads": self.threads, "ctx": self.ctx,
            "temperature": self.temperature, "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty, "n_predict": self.n_predict,
            "family": self.family,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def size_mb(self) -> float:
        p = Path(self.path)
        return round(p.stat().st_size / 1e6, 1) if p.exists() else 0.0

    @property
    def detected_family(self) -> ModelFamily:
        return detect_model_family(self.path)

    @property
    def quant_type(self) -> str:
        return detect_quant_type(self.path)


class ModelRegistry:
    def __init__(self):
        self._custom:  List[str]              = []
        self._configs: Dict[str, ModelConfig] = {}
        self._load()

    def _load(self):
        if CUSTOM_MODELS_FILE.exists():
            try:
                self._custom = json.loads(CUSTOM_MODELS_FILE.read_text())
            except Exception:
                self._custom = []
        if MODEL_CONFIGS_FILE.exists():
            try:
                raw = json.loads(MODEL_CONFIGS_FILE.read_text())
                self._configs = {p: ModelConfig.from_dict(d) for p, d in raw.items()}
            except Exception:
                self._configs = {}

    def save(self):
        CUSTOM_MODELS_FILE.write_text(json.dumps(self._custom, indent=2))
        MODEL_CONFIGS_FILE.write_text(
            json.dumps({p: c.to_dict() for p, c in self._configs.items()}, indent=2))

    def add(self, path: str):
        if path not in self._custom:
            self._custom.append(path)
        if path not in self._configs:
            fam = detect_model_family(path)
            self._configs[path] = ModelConfig(path=path, family=fam.family)
        self.save()

    def remove(self, path: str):
        self._custom  = [p for p in self._custom if p != path]
        self._configs.pop(path, None)
        self.save()

    def get_config(self, path: str) -> ModelConfig:
        if path not in self._configs:
            fam = detect_model_family(path)
            self._configs[path] = ModelConfig(path=path, family=fam.family)
        return self._configs[path]

    def set_config(self, path: str, cfg: ModelConfig):
        self._configs[path] = cfg
        self.save()

    def all_models(self) -> List[Dict]:
        seen:   set        = set()
        models: List[Dict] = []
        if MODELS_DIR.exists():
            for f in sorted(MODELS_DIR.glob("*.gguf")):
                seen.add(str(f))
                cfg = self.get_config(str(f))
                fam = detect_model_family(str(f))
                qt  = detect_quant_type(str(f))
                models.append({
                    "path": str(f), "name": f.name,
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                    "source": "auto", "role": cfg.role,
                    "family": fam.name, "quant": qt,
                })
        for p in self._custom:
            fp = Path(p)
            if fp.exists() and p not in seen:
                seen.add(p)
                cfg = self.get_config(p)
                fam = detect_model_family(p)
                qt  = detect_quant_type(p)
                models.append({
                    "path": p, "name": fp.name,
                    "size_mb": round(fp.stat().st_size / 1e6, 1),
                    "source": "custom", "role": cfg.role,
                    "family": fam.name, "quant": qt,
                })
        return models


MODEL_REGISTRY = ModelRegistry()

