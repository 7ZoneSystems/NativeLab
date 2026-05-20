from nativelab.imports.import_global import Dict, dataclass, Path, json, field
from .model_family import *
# ═════════════════════════════ MODEL REGISTRY ═══════════════════════════════
OLLAMA_REF_PREFIX = "ollama:"
HF_REF_PREFIX = "hf:"


def is_ollama_model_ref(ref: str) -> bool:
    return str(ref or "").strip().lower().startswith(OLLAMA_REF_PREFIX)


def is_hf_model_ref(ref: str) -> bool:
    return str(ref or "").strip().lower().startswith(HF_REF_PREFIX)


def is_external_model_ref(ref: str) -> bool:
    return is_ollama_model_ref(ref) or is_hf_model_ref(ref)


def model_ref_backend(ref: str) -> str:
    if is_ollama_model_ref(ref):
        return "ollama"
    if is_hf_model_ref(ref):
        return "hf_transformers"
    return "llama_cpp"


def model_ref_payload(ref: str) -> str:
    text = str(ref or "")
    if is_ollama_model_ref(text):
        return text[len(OLLAMA_REF_PREFIX):].strip()
    if is_hf_model_ref(text):
        return text[len(HF_REF_PREFIX):].strip()
    return text


def make_ollama_model_ref(model_name: str) -> str:
    return OLLAMA_REF_PREFIX + str(model_name or "").strip()


def make_hf_model_ref(model_id_or_path: str) -> str:
    return HF_REF_PREFIX + str(model_id_or_path or "").strip()


def model_ref_display_name(ref: str) -> str:
    text = str(ref or "")
    payload = model_ref_payload(text)
    if is_ollama_model_ref(text):
        return payload or "Ollama model"
    if is_hf_model_ref(text):
        p = Path(payload)
        if p.exists():
            return p.name
        return payload.rstrip("/").split("/")[-1] or payload or "HF model"
    p = Path(text)
    return p.name if text else ""


def is_model_ref_valid(ref: str) -> bool:
    text = str(ref or "").strip()
    if not text:
        return False
    if is_ollama_model_ref(text):
        return bool(model_ref_payload(text))
    if is_hf_model_ref(text):
        return bool(model_ref_payload(text))
    return Path(text).exists()


def _looks_like_hf_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
    has_index = any(path.glob("*.safetensors.index.json")) or any(path.glob("pytorch_model*.bin.index.json"))
    has_tokenizer = any((path / name).exists() for name in (
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "preprocessor_config.json",
        "processor_config.json",
    ))
    return bool((has_weights or has_index) and has_tokenizer)


def _path_size_mb(path: Path) -> float:
    try:
        if path.is_file():
            return round(path.stat().st_size / 1e6, 1)
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return round(total / 1e6, 1)
    except Exception:
        return 0.0


def _cfg():
    from nativelab.GlobalConfig import config_global
    return config_global

def _default_ctx():
    v = _cfg().DEFAULT_CTX
    return v() if callable(v) else int(v)

def _default_threads():
    v = _cfg().DEFAULT_THREADS
    return v() if callable(v) else int(v)

def _default_n_pred():
    v = _cfg().DEFAULT_N_PRED
    return v() if callable(v) else int(v)

@dataclass
class ModelConfig:
    path: str
    role: str = "general"
    backend: str = "llama_cpp"
    vision: bool = False

    threads: int = field(default_factory=_default_threads)
    ctx: int = field(default_factory=_default_ctx)
    n_predict: int = field(default_factory=_default_n_pred)

    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    family: str = "default"

    def to_dict(self) -> Dict:
        return {
            "path": self.path, "role": self.role,
            "backend": self.backend or model_ref_backend(self.path),
            "vision": bool(self.vision),
            "threads":        int(self.threads)        if not callable(self.threads)        else int(self.threads()),
            "ctx":            int(self.ctx)             if not callable(self.ctx)             else int(self.ctx()),
            "n_predict":      int(self.n_predict)       if not callable(self.n_predict)       else int(self.n_predict()),
            "temperature":    self.temperature,
            "top_p":          self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "family":         self.family,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def name(self) -> str:
        return model_ref_display_name(self.path)

    @property
    def size_mb(self) -> float:
        p = Path(model_ref_payload(self.path))
        return _path_size_mb(p) if p.exists() else 0.0

    @property
    def detected_family(self) -> ModelFamily:
        return detect_model_family(model_ref_payload(self.path) or self.path)

    @property
    def quant_type(self) -> str:
        if is_ollama_model_ref(self.path):
            return "OLLAMA"
        if is_hf_model_ref(self.path):
            return "TRANSFORMERS"
        return detect_quant_type(self.path)

    @property
    def vision_info(self) -> VisionModelInfo:
        vi = detect_vision_model(model_ref_payload(self.path) or self.path)
        if self.vision and not vi.is_vision:
            return VisionModelInfo(True, "Vision model", "vlm", False)
        return vi


class ModelRegistry:
    def __init__(self):
        self._custom:  List[str]              = []
        self._configs: Dict[str, ModelConfig] = {}
        self._load()

    def _load(self):
        from nativelab.GlobalConfig.config_global import DEFAULT_CTX, DEFAULT_THREADS, DEFAULT_N_PRED, CUSTOM_MODELS_FILE, MODEL_CONFIGS_FILE, MODELS_DIR
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
        from nativelab.GlobalConfig.config_global import DEFAULT_CTX, DEFAULT_THREADS, DEFAULT_N_PRED, CUSTOM_MODELS_FILE, MODEL_CONFIGS_FILE, MODELS_DIR
        CUSTOM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        MODEL_CONFIGS_FILE.parent.mkdir(parents=True, exist_ok=True)

        CUSTOM_MODELS_FILE.write_text(json.dumps(self._custom, indent=2))
        MODEL_CONFIGS_FILE.write_text(
            json.dumps({p: c.to_dict() for p, c in self._configs.items()}, indent=2))

    def add(self, path: str):
        if path not in self._custom:
            self._custom.append(path)
        if path not in self._configs:
            fam = detect_model_family(model_ref_payload(path) or path)
            vi = detect_vision_model(model_ref_payload(path) or path)
            self._configs[path] = ModelConfig(
                path=path,
                family=fam.family,
                backend=model_ref_backend(path),
                vision=vi.is_vision,
            )
        self.save()

    def remove(self, path: str):
        self._custom  = [p for p in self._custom if p != path]
        self._configs.pop(path, None)
        self.save()

    def get_config(self, path: str) -> ModelConfig:
        if path not in self._configs:
            fam = detect_model_family(model_ref_payload(path) or path)
            vi = detect_vision_model(model_ref_payload(path) or path)
            self._configs[path] = ModelConfig(
                path=path,
                family=fam.family,
                backend=model_ref_backend(path),
                vision=vi.is_vision,
            )
        else:
            cfg = self._configs[path]
            if not getattr(cfg, "backend", ""):
                cfg.backend = model_ref_backend(path)
        return self._configs[path]

    def set_config(self, path: str, cfg: ModelConfig):
        self._configs[path] = cfg
        self.save()

    def all_models(self) -> List[Dict]:
        from nativelab.GlobalConfig.config_global import DEFAULT_CTX, DEFAULT_THREADS, DEFAULT_N_PRED, CUSTOM_MODELS_FILE, MODEL_CONFIGS_FILE, MODELS_DIR
        seen:   set        = set()
        models: List[Dict] = []
        if MODELS_DIR.exists():
            for f in sorted(MODELS_DIR.glob("*.gguf")):
                seen.add(str(f))
                cfg = self.get_config(str(f))
                fam = detect_model_family(str(f))
                qt  = detect_quant_type(str(f))
                vi  = detect_vision_model(str(f))
                models.append({
                    "path": str(f), "name": f.name,
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                    "source": "auto", "role": cfg.role, "backend": "llama_cpp",
                    "family": fam.name, "quant": qt,
                    "vision": vi.is_vision, "vision_label": vi.label,
                    "mmproj": detect_mmproj_for_model(str(f)),
                })
            for d in sorted((p for p in MODELS_DIR.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
                ref = make_hf_model_ref(str(d))
                if _looks_like_hf_dir(d) and ref not in seen:
                    seen.add(ref)
                    cfg = self.get_config(ref)
                    fam = detect_model_family(d.name)
                    vi = detect_vision_model(d.name)
                    models.append({
                        "path": ref, "name": d.name,
                        "size_mb": _path_size_mb(d),
                        "source": "auto", "role": cfg.role, "backend": "hf_transformers",
                        "family": fam.name, "quant": "TRANSFORMERS",
                        "vision": vi.is_vision or cfg.vision,
                        "vision_label": vi.label or ("Vision model" if cfg.vision else ""),
                        "mmproj": "",
                    })
        for p in self._custom:
            fp = Path(model_ref_payload(p))
            if is_model_ref_valid(p) and p not in seen:
                seen.add(p)
                cfg = self.get_config(p)
                payload = model_ref_payload(p) or p
                fam = detect_model_family(payload)
                qt  = cfg.quant_type
                vi  = cfg.vision_info
                backend = cfg.backend or model_ref_backend(p)
                models.append({
                    "path": p, "name": model_ref_display_name(p),
                    "size_mb": _path_size_mb(fp) if fp.exists() else 0.0,
                    "source": "custom", "role": cfg.role, "backend": backend,
                    "family": fam.name, "quant": qt,
                    "vision": vi.is_vision, "vision_label": vi.label,
                    "mmproj": detect_mmproj_for_model(p) if backend == "llama_cpp" else "",
                })
        return models


_model_registry = None

def get_model_registry():
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
