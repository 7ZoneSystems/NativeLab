from imports.import_global import Path, _sys, re, os
_BASE = Path(getattr(_sys, "_MEIPASS", Path(".")))
_EXT  = ".exe" if __import__("platform").system() == "Windows" else ""

LLAMA_CLI_DEFAULT    = str(_BASE / f"llama-bin/llama-cli{_EXT}")
LLAMA_SERVER_DEFAULT = str(_BASE / f"llama-bin/llama-server{_EXT}")

# Fallback to local llama/bin for dev mode
if not Path(LLAMA_CLI_DEFAULT).exists():
    LLAMA_CLI_DEFAULT    = f"./llama/bin/llama-cli{_EXT}"
    LLAMA_SERVER_DEFAULT = f"./llama/bin/llama-server{_EXT}"

# These are resolved at runtime via SERVER_CONFIG (set after ServerConfig loads
LLAMA_CLI    = LLAMA_CLI_DEFAULT
LLAMA_SERVER = LLAMA_SERVER_DEFAULT

DEFAULT_MODEL      = "./localllm/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
def DEFAULT_CTX() -> int:
    return 2048

def DEFAULT_THREADS() -> int:
    return os.cpu_count() or 4
DEFAULT_N_PRED     = 512
CUSTOM_MODELS_FILE  = Path("./localllm/custom_models.json")
MODEL_CONFIGS_FILE  = Path("./localllm/model_configs.json")
PARALLEL_PREFS_FILE  = Path("./localllm/parallel_prefs.json")
SERVER_CONFIG_FILE   = Path("./localllm/server_config.json")
API_MODELS_FILE      = Path("./localllm/api_models.json")

# ── model roles ───────────────────────────────────────────────────────────────
MODEL_ROLES = ["general", "reasoning", "summarization", "coding", "secondary"]
ROLE_ICONS  = {
    "general":       "💬",
    "reasoning":     "🧠",
    "summarization": "📄",
    "coding":        "💻",
    "secondary":     "🔀",
}

# ═════════════════════════════ MODEL FAMILY DETECTION ═══════════════════════

# All GGUF quantization formats supported by llama.cpp
GGUF_QUANT_PATTERNS = [
    # imatrix importance quants
    r'IQ1_S', r'IQ1_M', r'IQ2_XXS', r'IQ2_XS', r'IQ2_S', r'IQ2_M',
    r'IQ3_XXS', r'IQ3_XS', r'IQ3_S', r'IQ3_M', r'IQ4_XS', r'IQ4_NL',
    # K-quants
    r'Q2_K(?:_S)?', r'Q3_K_(?:S|M|L)', r'Q3_K',
    r'Q4_K_(?:S|M)', r'Q4_K',
    r'Q5_K_(?:S|M)', r'Q5_K',
    r'Q6_K',
    # legacy quants
    r'Q4_0', r'Q4_1', r'Q5_0', r'Q5_1', r'Q8_0',
    # float
    r'F16', r'F32', r'BF16',
    # GGML legacy
    r'Q4_0_4_4', r'Q4_0_4_8', r'Q4_0_8_8',
]

QUANT_REGEX = re.compile(
    r'\.(' + '|'.join(GGUF_QUANT_PATTERNS) + r')\.gguf$',
    re.IGNORECASE
)

# Dynamic accessors (use these everywhere instead of bare constants)
def RAM_WATCHDOG_MB() -> float:
    from .binaryResolve import APP_CONFIG
    return float(APP_CONFIG["ram_watchdog_mb"])

def CHUNK_INDEX_SIZE() -> int:
    from .binaryResolve import APP_CONFIG
    return int(APP_CONFIG["chunk_index_size"])

def MAX_RAM_CHUNKS() -> int:
    from .binaryResolve import APP_CONFIG
    return int(APP_CONFIG["max_ram_chunks"])

def get_default_threads() -> int:
    from .binaryResolve import APP_CONFIG
    return int(APP_CONFIG["default_threads"])

def get_default_ctx() -> int:
    from .binaryResolve import APP_CONFIG
    return int(APP_CONFIG["default_ctx"])

def get_default_n_pred() -> int:
    from .binaryResolve import APP_CONFIG
    return int(APP_CONFIG["default_n_predict"])
from Model.model_global import SCRIPT_LANGUAGES
SCRIPT_EXTENSIONS_FILTER = (
    "Source files ("
    + " ".join(f"*{ext}" for ext in SCRIPT_LANGUAGES.keys())
    + ");;All Files (*)"
)