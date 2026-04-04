# hfwld.py - HuggingFace GGUF download workers
from nativelab.imports.import_global import QThread, pyqtSignal, Path
import threading


class HfSearchWorker(QThread):
    """Queries the HuggingFace API for GGUF siblings in a repo."""
    results_ready = pyqtSignal(list)
    err           = pyqtSignal(str)

    def __init__(self, repo_id: str):
        super().__init__()
        self._repo = repo_id.strip().strip("/")

    def run(self):
        import urllib.request as _ur, json as _j
        url = f"https://huggingface.co/api/models/{self._repo}"
        try:
            req = _ur.Request(url, headers={"User-Agent": "NativeLabPro/2"})
            with _ur.urlopen(req, timeout=15) as r:
                data = _j.loads(r.read())
            siblings = data.get("siblings", [])
            gguf = [s for s in siblings
                    if s.get("rfilename", "").lower().endswith(".gguf")]
            self.results_ready.emit(gguf)
        except Exception as e:
            self.err.emit(str(e))


class _AbortedError(Exception):
    """Raised internally when the user aborts a download."""
    def __init__(self, delete_part: bool = False):
        self.delete_part = delete_part


class HfDownloadWorker(QThread):
    """
    Downloads a single GGUF file from HuggingFace with:
      - Pause / resume  (threading.Event)
      - Abort keeping .part  (default) or wiping it (Cancel & Delete)
      - Resume on restart    (Range header, .part file survives crashes)
      - Auto-retry on transient network failure (3 attempts, resumes each time)
    """
    progress = pyqtSignal(int, int)   # bytes_done, bytes_total
    done     = pyqtSignal(str)        # final save path
    err      = pyqtSignal(str)
    paused   = pyqtSignal(bool)       # True = just paused, False = just resumed

    MAX_RETRIES = 3
    RETRY_WAIT  = 4        # seconds between retries
    CHUNK       = 262144   # 256 KB

    def __init__(self, repo_id: str, filename: str, dest_dir: Path,
                 expected_size: int = 0):
        super().__init__()
        self._repo          = repo_id.strip().strip("/")
        self._filename      = filename
        self._dest_dir      = dest_dir
        self._expected_size = expected_size
        self._abort         = False
        self._abort_delete  = False
        self._pause_event   = threading.Event()
        self._pause_event.set()   # not paused initially

    # ── public controls ───────────────────────────────────────────────────────

    def abort(self, delete_part: bool = False):
        """
        Stop the download.
        delete_part=False (default) : keep .part file so download can resume later.
        delete_part=True            : wipe the .part file (user explicitly discarding).
        """
        self._abort_delete = delete_part
        self._abort = True
        self._pause_event.set()   # unblock if currently paused so thread can exit

    def pause(self):
        self._pause_event.clear()
        self.paused.emit(True)

    def resume(self):
        self._pause_event.set()
        self.paused.emit(False)

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    # ── main thread entry ─────────────────────────────────────────────────────

    def run(self):
        import urllib.request as _ur, time

        url  = (f"https://huggingface.co/{self._repo}"
                f"/resolve/main/{self._filename}")
        dest = self._dest_dir / self._filename
        part = self._part_path(dest)
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        # Resolve CDN URL once upfront (avoids repeated redirect overhead)
        try:
            final_url = self._resolve_final_url(url)
        except Exception:
            final_url = url

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._download_loop(final_url, part)
                break   # clean finish — exit retry loop
            except _AbortedError as e:
                if e.delete_part:
                    try: part.unlink()
                    except Exception: pass
                return   # exit silently either way — not an error
            except Exception as exc:
                if attempt >= self.MAX_RETRIES:
                    # Genuine failure — keep .part so user can resume next session
                    self.err.emit(str(exc))
                    return
                time.sleep(self.RETRY_WAIT)
                # Next iteration resumes automatically from .part offset

        # Atomically rename .part -> final filename
        try:
            part.replace(dest)
        except Exception as e:
            self.err.emit(f"Could not finalise file: {e}")
            return

        self.done.emit(str(dest))

    # ── internals ─────────────────────────────────────────────────────────────

    def _resolve_final_url(self, url: str) -> str:
        """Follow HuggingFace CDN redirects via HEAD and return the final URL."""
        import urllib.request as _ur
        req = _ur.Request(url, headers={"User-Agent": "NativeLabPro/2"},
                          method="HEAD")
        with _ur.urlopen(req, timeout=30) as r:
            return r.url

    def _part_path(self, dest: Path) -> Path:
        return dest.with_suffix(dest.suffix + ".part")

    def _download_loop(self, final_url: str, part: Path):
        import urllib.request as _ur

        # Resume: how many bytes do we already have from a previous run / retry?
        resume_from = part.stat().st_size if part.exists() else 0

        headers = {"User-Agent": "NativeLabPro/2"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        req = _ur.Request(final_url, headers=headers)
        with _ur.urlopen(req, timeout=60) as r:
            cl = r.headers.get("Content-Length")

            if resume_from and r.status == 206:   # server honoured Range
                total = resume_from + (int(cl) if cl else 0)
            else:
                # Server ignored Range (e.g. returned 200) — start fresh
                total       = self._expected_size or (int(cl) if cl and int(cl) > 0 else 0)
                resume_from = 0

            done = resume_from
            mode = "ab" if resume_from else "wb"

            with open(part, mode) as f:
                while True:
                    # Block here when paused; wakes immediately on resume/abort
                    self._pause_event.wait()

                    if self._abort:
                        raise _AbortedError(delete_part=self._abort_delete)

                    chunk = r.read(self.CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    self.progress.emit(done, total)

        if total and done != total:
            raise ValueError(f"Size mismatch: expected {total} B, got {done} B")