from imports.import_global import Path, json, Dict
from .config import LLAMA_CLI_DEFAULT, LLAMA_SERVER_DEFAULT
from .const import APP_CONFIG_DEFAULTS, APP_CONFIG_FILE
def refresh_binary_paths():
    """Re-read SERVER_CONFIG and update module-level LLAMA_CLI / LLAMA_SERVER."""
    from Server.server_global import SERVER_CONFIG  # local import

    global LLAMA_CLI, LLAMA_SERVER
    LLAMA_CLI    = _resolve_binary(SERVER_CONFIG.cli_path,    LLAMA_CLI_DEFAULT)
    LLAMA_SERVER = _resolve_binary(SERVER_CONFIG.server_path, LLAMA_SERVER_DEFAULT)
MODELS_DIR   = Path("./localllm")
SESSIONS_DIR = Path("./sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

def _load_app_config() -> Dict:
    cfg = dict(APP_CONFIG_DEFAULTS)
    if APP_CONFIG_FILE.exists():
        try:
            saved = json.loads(APP_CONFIG_FILE.read_text())
            cfg.update({k: v for k, v in saved.items() if k in cfg})
        except Exception:
            pass
    return cfg

def save_app_config(cfg: Dict):
    APP_CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

APP_CONFIG = _load_app_config()

def _resolve_binary(cfg_path: str, fallback: str) -> str:
    """Return cfg_path if set and exists, else fallback."""
    if cfg_path and Path(cfg_path).exists():
        return cfg_path
    return fallback
