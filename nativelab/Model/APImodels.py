from nativelab.imports.import_global import dataclass, List, json, Dict
from urllib.parse import quote, unquote
def _cfg():
    from nativelab.GlobalConfig import config_global
    return config_global
@dataclass
class ApiConfig:
    name:        str   = ""
    provider:    str   = "OpenAI"
    model_id:    str   = ""
    api_key:     str   = ""
    base_url:    str   = ""
    api_format:  str   = "openai"
    max_tokens:  int   = 2048
    temperature: float = 0.7
    # ── custom prompt format ─────────────────────────────────────────────────
    use_custom_prompt:   bool = False
    system_prompt:       str  = ""
    user_prefix:         str  = ""
    user_suffix:         str  = ""
    assistant_prefix:    str  = ""
    prompt_template:     str  = "default"   # "default"|"chatml"|"llama2"|"alpaca"|"custom"
    # ── custom provider display ──────────────────────────────────────────────
    custom_provider_name: str = ""

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Dict) -> "ApiConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ApiRegistry:
    def __init__(self):
        self._configs: List[ApiConfig] = []
        self._load()

    def _load(self):
        if _cfg().API_MODELS_FILE.exists():
            try:
                self._configs = [ApiConfig.from_dict(d)
                                 for d in json.loads(_cfg().API_MODELS_FILE.read_text())]
            except Exception:
                self._configs = []

    def save(self):
        _cfg().API_MODELS_FILE.write_text(
            json.dumps([c.to_dict() for c in self._configs], indent=2))

    def add(self, cfg: ApiConfig):
        self._configs = [c for c in self._configs if c.name != cfg.name]
        self._configs.append(cfg)
        self.save()

    def remove(self, name: str):
        self._configs = [c for c in self._configs if c.name != name]
        self.save()

    def all(self) -> List[ApiConfig]:
        return list(self._configs)

    def get(self, name: str) -> ApiConfig | None:
        for cfg in self._configs:
            if cfg.name == name:
                return cfg
        return None

    def get_by_ref(self, ref: str) -> ApiConfig | None:
        if not is_api_model_ref(ref):
            return None
        return self.get(api_model_name_from_ref(ref))


api_registry = None

def getapi_registry():
    global api_registry
    if api_registry is None:
        api_registry = ApiRegistry()
    return api_registry


API_MODEL_PREFIX = "@api/"


def api_model_ref(name: str) -> str:
    return API_MODEL_PREFIX + quote(name or "", safe="")


def api_model_name_from_ref(ref: str) -> str:
    return unquote((ref or "")[len(API_MODEL_PREFIX):])


def is_api_model_ref(ref: str) -> bool:
    return bool(ref) and str(ref).startswith(API_MODEL_PREFIX)


def api_model_label(cfg: ApiConfig) -> str:
    provider = getattr(cfg, "custom_provider_name", "") or cfg.provider
    return f"API: {cfg.name} ({provider} / {cfg.model_id})"
