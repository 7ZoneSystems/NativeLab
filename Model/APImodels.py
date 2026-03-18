from imports.import_global import dataclass, List, json, Dict
def _cfg():
    from GlobalConfig import config_global
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


_api_registry = None

def get_api_registry():
    global _api_registry
    if _api_registry is None:
        _api_registry = ApiRegistry()
    return _api_registry
