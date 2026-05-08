"""Synchronous HuggingFace search + download for the CLI.

Plain `urllib` and a blocking download loop; mirrors the resume + retry
behaviour of `nativelab.Server.hfdwld.HfDownloadWorker` but without QThreads.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from . import ui


_HEADERS = {"User-Agent": "NativeLabPro-CLI/1"}


# ─────────────────────────────────────────────────────────────────────────────
#  Search
# ─────────────────────────────────────────────────────────────────────────────

def list_repo_ggufs(repo_id: str) -> List[dict]:
    """Return GGUF siblings of a HuggingFace model repo (with size when available)."""
    repo_id = repo_id.strip().strip("/")
    url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8", errors="replace"))

    siblings = data.get("siblings") or []
    out = []
    for s in siblings:
        name = s.get("rfilename", "")
        if not name.lower().endswith(".gguf"):
            continue
        size = s.get("size") or s.get("lfs", {}).get("size") or 0
        out.append({"name": name, "size": int(size)})
    return out


def quick_picks() -> List[dict]:
    """A small list of well-known small GGUFs for first-time users."""
    return [
        {"repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
         "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
         "label": "Mistral 7B Instruct (Q4_K_M, ~4.4 GB) — solid general-purpose"},
        {"repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
         "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
         "label": "Qwen 2.5 3B Instruct (Q4_K_M, ~2 GB) — fast, low RAM"},
        {"repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
         "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
         "label": "Phi-3.5 Mini (Q4_K_M, ~2.4 GB) — strong for size"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Download
# ─────────────────────────────────────────────────────────────────────────────

CHUNK = 262_144   # 256 KB
MAX_RETRIES = 3
RETRY_WAIT  = 3   # seconds


def download_gguf(repo_id: str, filename: str, dest_dir: Path,
                  expected_size: int = 0) -> Path:
    """Download `filename` from `repo_id` into `dest_dir`.

    Resumes from a `.part` file if present and retries up to `MAX_RETRIES` on
    transient failures. Returns the final file path.
    """
    repo_id = repo_id.strip().strip("/")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        ui.ok(f"Already present: {dest}")
        return dest

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    try:
        head_req = urllib.request.Request(url, headers=_HEADERS, method="HEAD")
        with urllib.request.urlopen(head_req, timeout=30) as r:
            url = r.url
    except Exception:
        pass  # use the original URL — the next request will follow redirects

    bar = ui.ProgressBar(prefix=f"  {filename}")

    for attempt in range(1, MAX_RETRIES + 1):
        resume_from = part.stat().st_size if part.exists() else 0
        headers = dict(_HEADERS)
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                cl = r.headers.get("Content-Length")
                if resume_from and r.status == 206:
                    total = resume_from + (int(cl) if cl else 0)
                else:
                    total = expected_size or (int(cl) if cl and int(cl) > 0 else 0)
                    resume_from = 0

                done = resume_from
                mode = "ab" if resume_from else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        bar.update(done, total)

                if total and done != total:
                    raise RuntimeError(f"Size mismatch: expected {total} got {done}")

            part.replace(dest)
            bar.done(f"Downloaded {filename}  →  {dest}")
            return dest

        except (urllib.error.URLError, ConnectionError, TimeoutError, RuntimeError) as e:
            if attempt >= MAX_RETRIES:
                bar.done()
                raise RuntimeError(
                    f"Download failed after {MAX_RETRIES} attempts: {e}"
                ) from e
            ui.warn(f"Network blip ({e}) — retrying in {RETRY_WAIT}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_WAIT)

    raise RuntimeError("Download failed")
