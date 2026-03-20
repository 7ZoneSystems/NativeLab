from imports.import_global import Optional, List, json
from GlobalConfig.config_global import PAUSED_JOBS_DIR
# ── Paused job persistence ────────────────────────────────────────────────────

def save_paused_job(job_id: str, state: dict):
    p = PAUSED_JOBS_DIR / f"{job_id}.json"
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def load_paused_job(job_id: str) -> Optional[dict]:
    p = PAUSED_JOBS_DIR / f"{job_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def delete_paused_job(job_id: str):
    p = PAUSED_JOBS_DIR / f"{job_id}.json"
    if p.exists():
        try: p.unlink()
        except Exception: pass

def list_paused_jobs() -> List[dict]:
    jobs = []
    for p in PAUSED_JOBS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            jobs.append(d)
        except Exception:
            pass
    return jobs
