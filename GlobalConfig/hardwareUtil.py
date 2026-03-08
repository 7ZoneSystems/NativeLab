from imports.import_global import hashlib, psutil, HAS_PSUTIL
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