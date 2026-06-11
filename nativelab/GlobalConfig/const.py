from nativelab.imports.import_global import _sys
from pathlib import Path
import shutil
from .hardwareUtil import cpu_count
from .timeouts import LONG_TIMEOUT_MS, LONG_TIMEOUT_NONE, LONG_TIMEOUT_SECONDS
_sys.path.append(str(Path(__file__).resolve().parents[1]))
# ── Reference system constants ────────────────────────────────────────────────
REFS_DIR          = Path("chat_refs")
REF_CACHE_DIR     = Path("ref_cache")
REF_INDEX_DIR     = Path("ref_index")
PAUSED_JOBS_DIR   = Path("paused_jobs")
LOCAL_LLM_DIR     = Path("localllm")
APP_CONFIG_FILE   = Path("app_config.json")
MAX_CONTEXT_TOKENS = 300000
LOCAL_LLM_DIR.mkdir(parents=True, exist_ok=True)
REFS_DIR.mkdir(parents=True, exist_ok=True)
REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
REF_INDEX_DIR.mkdir(parents=True, exist_ok=True)
PAUSED_JOBS_DIR.mkdir(parents=True, exist_ok=True)
PIPELINES_DIR = LOCAL_LLM_DIR / "pipelines"
LEGACY_PIPELINES_DIR = Path.home() / ".native_lab" / "pipelines"
PIPELINES_DIR.mkdir(parents=True, exist_ok=True)


def migrate_legacy_pipelines() -> int:
    """Copy old ~/.native_lab/pipelines JSON files into localllm/pipelines."""
    if not LEGACY_PIPELINES_DIR.exists() or not LEGACY_PIPELINES_DIR.is_dir():
        return 0
    moved = 0
    for src in LEGACY_PIPELINES_DIR.glob("*.json"):
        if src.is_symlink() or not src.is_file():
            continue
        dst = PIPELINES_DIR / src.name
        if dst.exists():
            continue
        try:
            PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            moved += 1
        except Exception:
            continue
    return moved


migrate_legacy_pipelines()
APP_CONFIG_DEFAULTS = {
    "ram_watchdog_mb":        800,
    "chunk_index_size":       400,
    "max_ram_chunks":         120,
    "summary_chunk_chars":    3000,
    "summary_ctx_carry":      600,
    "summary_n_pred_sect":    380,
    "summary_n_pred_final":   700,
    "multipdf_n_pred_sect":   380,
    "multipdf_n_pred_final":  900,
    "ref_top_k":              6,
    "ref_max_context_chars":  3000,
    "pause_after_chunks":     2,   # auto-pause suggestion threshold
    "default_threads":        12,
    "default_ctx":            4096,
    "default_n_predict":      512,
    "tps_display":            True,
    "auto_spill_on_start":    False,
    "server_startup_timeout": LONG_TIMEOUT_SECONDS,
    "stream_socket_timeout":  LONG_TIMEOUT_SECONDS,   # seconds before a silent read errors
    "stream_stall_timeout":   LONG_TIMEOUT_SECONDS,   # seconds of no tokens before giving up
    "stream_max_buf_bytes":   65536, # max line buffer size
    "hf_transformers_dir":    "localllm/hf_transformers",
    "hf_token":               "",
    "hf_revision":            "main",
    "hf_trust_remote_code":   True,
    "hf_local_files_only":    False,
    "hf_use_safetensors":     "auto",
    "hf_torch_dtype":         "auto",
    "hf_device_map":          "auto",
    "hf_low_cpu_mem_usage":   True,
    "hf_attn_implementation": "",
    "hf_max_memory":          "",
    "hf_quantization":        "none",
    "ollama_host":            "http://127.0.0.1:11434",
    "ollama_keep_alive":      "5m",
    "developer_mode":         False,
}
# ─── Config descriptions ──────────────────────────────────────────────────────
CONFIG_FIELD_META = {
    "ram_watchdog_mb": {
        "label": "RAM Watchdog (MB)",
        "desc":  (
            "Free RAM threshold in megabytes. When available RAM drops below "
            "this value, reference chunk caches are automatically spilled to disk "
            "to prevent system memory exhaustion. Lower = more aggressive spilling. "
            "Recommended: 400–1200 MB depending on your total RAM."
        ),
        "min": 100, "max": 8000, "type": "int",
    },
    "chunk_index_size": {
        "label": "Ref Chunk Size (chars)",
        "desc":  (
            "Character size of each indexed chunk when a reference file is loaded. "
            "Smaller chunks = finer retrieval granularity but more memory entries. "
            "Larger chunks = broader context per hit but less precise. "
            "Recommended: 300–600."
        ),
        "min": 100, "max": 2000, "type": "int",
    },
    "max_ram_chunks": {
        "label": "Max RAM Chunks (per ref)",
        "desc":  (
            "Maximum number of reference chunks kept in RAM per loaded file. "
            "Chunks beyond this limit are evicted to disk. On systems with "
            "8 GB RAM, 80–120 is safe. On 4 GB, consider 40–60."
        ),
        "min": 10, "max": 500, "type": "int",
    },
    "summary_chunk_chars": {
        "label": "Summary Chunk Size (chars)",
        "desc":  (
            "Size of each text chunk sent to the model during PDF summarization. "
            "Smaller values = more chunks, less RAM per step, but more inference calls. "
            "Larger values = fewer calls but each prompt uses more context window. "
            "Recommended: 2000–4000."
        ),
        "min": 500, "max": 8000, "type": "int",
    },
    "summary_ctx_carry": {
        "label": "Summary Context Carry (chars)",
        "desc":  (
            "Number of characters from the previous chunk's summary to carry forward "
            "as context for the next chunk. Helps maintain narrative continuity "
            "across long documents. Recommended: 400–800."
        ),
        "min": 100, "max": 2000, "type": "int",
    },
    "summary_n_pred_sect": {
        "label": "Summary Tokens / Section",
        "desc":  (
            "Maximum tokens the model generates for each section summary. "
            "Higher = more detailed section summaries but slower processing. "
            "Recommended: 250–500."
        ),
        "min": 64, "max": 1024, "type": "int",
    },
    "summary_n_pred_final": {
        "label": "Summary Tokens / Final Pass",
        "desc":  (
            "Maximum tokens for the final consolidation pass that synthesizes "
            "all section summaries into one cohesive document summary. "
            "Recommended: 500–900."
        ),
        "min": 128, "max": 2048, "type": "int",
    },
    "multipdf_n_pred_sect": {
        "label": "Multi-PDF Tokens / Section",
        "desc":  (
            "Tokens per section for multi-PDF batch summarization. "
            "Same as Summary Tokens / Section but applied to each document "
            "in a multi-document batch job. Recommended: 300–500."
        ),
        "min": 64, "max": 1024, "type": "int",
    },
    "multipdf_n_pred_final": {
        "label": "Multi-PDF Tokens / Final",
        "desc":  (
            "Tokens for the final cross-document consolidation in multi-PDF mode. "
            "This pass synthesizes summaries across all loaded documents, "
            "so a higher value gives richer cross-document analysis. "
            "Recommended: 700–1200."
        ),
        "min": 128, "max": 2048, "type": "int",
    },
    "ref_top_k": {
        "label": "Reference Top-K Chunks",
        "desc":  (
            "Number of most relevant chunks retrieved from each reference file "
            "per query. Higher = more context injected but larger prompts. "
            "Lower = faster, smaller prompts. Recommended: 4–8."
        ),
        "min": 1, "max": 20, "type": "int",
    },
    "ref_max_context_chars": {
        "label": "Ref Max Context (chars)",
        "desc":  (
            "Maximum total characters injected from all references combined "
            "into each prompt. Guards against overflowing the model's context window. "
            "Recommended: 1500–4000."
        ),
        "min": 200, "max": 12000, "type": "int",
    },
    "pause_after_chunks": {
        "label": "Pause-Suggest Threshold (chunks)",
        "desc":  (
            "After this many chunks are processed, the app will show a "
            "pause/save banner if many chunks remain. Set to 0 to disable "
            "auto-pause suggestions. Recommended: 2–5."
        ),
        "min": 0, "max": 50, "type": "int",
    },
    "default_threads": {
        "label": "Default CPU Threads",
        "desc":  (
            "Number of CPU threads used by llama.cpp for inference. "
            "Setting this higher than your physical core count may reduce performance. "
            f"Your system has {cpu_count()} logical CPUs. "
            "Recommended: physical core count (not hyperthreaded)."
        ),
        "min": 1, "max": 64, "type": "int",
    },
    "default_ctx": {
        "label": "Default Context Window (tokens)",
        "desc":  (
            "Default token context window loaded with each model. "
            "Larger context = more conversation history, more RAM. "
            "Each 4096 additional tokens uses ~0.5 GB extra RAM. "
            "Recommended: 2048–8192 for most systems."
        ),
        "min": 512, "max": MAX_CONTEXT_TOKENS, "type": "int",
    },
    "default_n_predict": {
        "label": "Default Max New Tokens",
        "desc":  (
            "Default maximum number of tokens the model generates per response. "
            "Does not affect RAM usage, only generation length. "
            "Recommended: 256–1024."
        ),
        "min": 32, "max": 4096, "type": "int",
    },
    "auto_spill_on_start": {
        "label": "Auto-Spill Refs on Startup",
        "desc":  (
            "When enabled, all reference chunk caches are immediately spilled to disk "
            "on app startup regardless of available RAM. Useful on systems with "
            "very low RAM where you need the model to have maximum headroom."
        ),
        "min": 0, "max": 1, "type": "bool",
    },
    "hf_transformers_dir": {
        "label": "HF Download Directory",
        "desc": (
            "Local directory where full Hugging Face Transformers snapshots are "
            "downloaded. Completed snapshots are registered as hf:<folder> models."
        ),
        "type": "path",
    },
    "hf_token": {
        "label": "HF Access Token",
        "desc": (
            "Optional Hugging Face token used for private/gated repos. It is saved "
            "locally in app_config.json and masked in the UI."
        ),
        "type": "password",
    },
    "hf_revision": {
        "label": "HF Default Revision",
        "desc": "Branch, tag, or commit used for HF snapshot downloads and model loads.",
        "type": "str",
    },
    "hf_trust_remote_code": {
        "label": "HF Trust Remote Code",
        "desc": (
            "Pass trust_remote_code to Transformers. Required by some custom model "
            "repos, but only enable it for repos you trust."
        ),
        "type": "bool",
    },
    "hf_local_files_only": {
        "label": "HF Local Files Only",
        "desc": "Prevent Transformers from downloading missing files during load.",
        "type": "bool",
    },
    "hf_use_safetensors": {
        "label": "HF Use Safetensors",
        "desc": (
            "auto lets Transformers choose. true forces safetensors. false allows "
            "PyTorch binary weights."
        ),
        "type": "choice", "choices": ["auto", "true", "false"],
    },
    "hf_torch_dtype": {
        "label": "HF Torch Dtype",
        "desc": "torch_dtype passed to Transformers when loading models.",
        "type": "choice", "choices": ["auto", "float16", "bfloat16", "float32"],
    },
    "hf_device_map": {
        "label": "HF Device Map",
        "desc": (
            "device_map for Transformers. auto uses Accelerate when installed. "
            "none omits device_map."
        ),
        "type": "choice", "choices": ["auto", "none", "cpu", "balanced", "sequential"],
    },
    "hf_low_cpu_mem_usage": {
        "label": "HF Low CPU Memory",
        "desc": "Pass low_cpu_mem_usage=True to reduce peak RAM during model load.",
        "type": "bool",
    },
    "hf_attn_implementation": {
        "label": "HF Attention Implementation",
        "desc": (
            "Optional attn_implementation value, such as sdpa or flash_attention_2. "
            "Leave blank for Transformers default."
        ),
        "type": "str",
    },
    "hf_max_memory": {
        "label": "HF Max Memory Map",
        "desc": (
            "Optional comma-separated map, e.g. cuda:0=12GiB,cpu=32GiB. "
            "Passed as max_memory when loading."
        ),
        "type": "str",
    },
    "hf_quantization": {
        "label": "HF Quantization",
        "desc": (
            "none, 8bit, or 4bit. 8bit/4bit require bitsandbytes installed in "
            "the Python environment."
        ),
        "type": "choice", "choices": ["none", "8bit", "4bit"],
    },
    "ollama_host": {
        "label": "Ollama Host",
        "desc": "Base URL for the already-running Ollama daemon.",
        "type": "str",
    },
    "ollama_keep_alive": {
        "label": "Ollama Keep Alive",
        "desc": "keep_alive value sent to Ollama chat requests, for example 5m or 1h.",
        "type": "str",
    },
    "developer_mode": {
        "label": "Developer Mode",
        "desc": (
            "Show the Dev workspace with Labs, Logs, Integrations, Pipeline, MCP, "
            "and Skills. Turn this off for a cleaner main workspace."
        ),
        "type": "bool",
    },
}
