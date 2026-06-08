from nativelab.imports.import_global import Optional, List, Dict, json, QThread
from nativelab.core.streamer_global import ApiStreamWorker
from nativelab.core.engine_status import engine_status
from nativelab.GlobalConfig.config_global import DEFAULT_N_PRED, LONG_TIMEOUT_SECONDS
from nativelab.Model.model_global import ApiConfig, api_model_ref
class ApiEngine:
    def __init__(self):
        self.model_path:           str               = ""
        self.mode:                 str               = "unloaded"
        self.ctx_value:            int               = 8192
        self._config:    Optional[ApiConfig]         = None
        self._pending_messages: List[Dict]           = []
        self._log = lambda m: None


    def set_messages(self, messages: List[Dict]):
        """Store structured messages to be used by the next create_worker call."""
        self._pending_messages = list(messages)

    def load(self, config: ApiConfig, log_cb=None) -> bool:
        self._config  = config
        self._log     = log_cb or (lambda m: None)
        self.model_path = api_model_ref(config.name)
        self.ctx_value  = max(int(config.max_tokens or 0), 8192) * 6
        try:
            import urllib.request, urllib.error
            msgs = [{"role": "user", "content": "hi"}]
            if config.api_format == "anthropic":
                body = json.dumps({"model": config.model_id, "messages": msgs,
                                   "max_tokens": 1, "stream": False}).encode()
                req  = urllib.request.Request(
                    f"{config.base_url.rstrip('/')}/v1/messages", data=body,
                    headers={"Content-Type": "application/json",
                             "x-api-key": config.api_key,
                             "anthropic-version": "2023-06-01"})
            else:
                body = json.dumps({"model": config.model_id, "messages": msgs,
                                   "max_tokens": 1, "stream": False}).encode()
                req  = urllib.request.Request(
                    f"{config.base_url.rstrip('/')}/chat/completions", data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {config.api_key}"})
            with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
                r.read()
            self.mode = "api"
            print(f"[APIENGINE][INFO] API model verified: {config.model_id}", flush=True)
            self._log(f"[INFO] API model verified: {config.model_id}")
            return True
        except Exception as e:
            print(f"[APIENGINE][ERROR] API test failed: {e}", flush=True)
            self._log(f"[ERROR] API test failed: {e}")
            self.mode = "unloaded"
            return False

    def create_worker(self, prompt: str, n_predict: int = DEFAULT_N_PRED,
                      model_path: str = "") -> QThread:
        cfg  = self._config
        if cfg is None:
            raise RuntimeError("API config not loaded. Call load() first.")
        if self._pending_messages:
            msgs = list(self._pending_messages)
            self._pending_messages = []
        else:
            msgs = [{"role": "user", "content": prompt}]
        try:
            output_tokens = int(n_predict)
        except (TypeError, ValueError):
            output_tokens = 0
        w = ApiStreamWorker(
            messages    = msgs,
            api_key     = cfg.api_key,
            base_url    = cfg.base_url,
            model_id    = cfg.model_id,
            api_format  = cfg.api_format,
            max_tokens  = output_tokens if output_tokens > 0 else None,
            temperature = cfg.temperature,
        )
        w.log.connect(self._log)
        if cfg.use_custom_prompt:
            w.use_custom_prompt  = True
            w.system_prompt      = cfg.system_prompt
            w.user_prefix        = cfg.user_prefix
            w.user_suffix        = cfg.user_suffix
            w.assistant_prefix   = cfg.assistant_prefix
        return w

    def ensure_server(self, log_cb=None) -> bool:
        return self.is_loaded

    def ensure_server_or_reload(self, log_cb=None) -> bool:
        return self.is_loaded

    def shutdown(self):
        self.mode             = "unloaded"
        self._config          = None
        self._pending_messages = []

    @property
    def is_loaded(self) -> bool:
        return self.mode == "api"

    @property
    def status_text(self) -> str:
        return engine_status(self).status_text
