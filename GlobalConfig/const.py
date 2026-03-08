from imports.import_global import _sys
from pathlib import Path

_sys.path.append(str(Path(__file__).resolve().parents[1]))
# ── Reference system constants ────────────────────────────────────────────────
REFS_DIR          = Path("chat_refs")
REF_CACHE_DIR     = Path("ref_cache")
REF_INDEX_DIR     = Path("ref_index")
PAUSED_JOBS_DIR   = Path("paused_jobs")
APP_CONFIG_FILE   = Path("app_config.json")
REFS_DIR.mkdir(parents=True, exist_ok=True)
REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
REF_INDEX_DIR.mkdir(parents=True, exist_ok=True)
PAUSED_JOBS_DIR.mkdir(parents=True, exist_ok=True)
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
}