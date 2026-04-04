from imports.import_global import hashlib, psutil, HAS_PSUTIL, Dict

def ram_free_mb() -> float:
    if HAS_PSUTIL:
        return psutil.virtual_memory().available / (1024 * 1024)
    return 9999.0

def cpu_count() -> int:
    try:
        import multiprocessing
        n = multiprocessing.cpu_count()
        return n if n and n > 0 else 4
    except Exception:
        return 4

def simple_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:12]

_SESSION_REF_STORES: Dict = {}

def get_ref_store(session_id: str):
    from codeparser.codeparser_global import SessionReferenceStore
    if session_id not in _SESSION_REF_STORES:
        _SESSION_REF_STORES[session_id] = SessionReferenceStore(session_id)
    return _SESSION_REF_STORES[session_id]

class RamWatchdog:
    triggered = False

    @staticmethod
    def check_and_spill(session_id: str) -> bool:
        from GlobalConfig.config_global import RAM_WATCHDOG_MB
        if ram_free_mb() < RAM_WATCHDOG_MB:
            store = get_ref_store(session_id)
            store.flush_ram()
            RamWatchdog.triggered = True
            return True
        return False