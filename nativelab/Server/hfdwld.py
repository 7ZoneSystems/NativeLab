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


class LlamaCppReleaseFetcher(QThread):
    """Fetches llama.cpp GitHub releases and available assets for the current platform."""
    results_ready = pyqtSignal(list)   # list of {tag, assets: [{name, url, size}]}
    err           = pyqtSignal(str)

    def run(self):
        import urllib.request as _ur, json as _j, platform as _pl
        url = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
        try:
            req = _ur.Request(url, headers={"User-Agent": "NativeLabPro/2",
                                            "Accept": "application/vnd.github+json"})
            with _ur.urlopen(req, timeout=15) as r:
                releases = _j.loads(r.read())

            sys_name = _pl.system().lower()
            machine  = _pl.machine().lower()

            # Pick platform filter keywords
            if sys_name == "windows":
                kw = ["win"]
            elif sys_name == "darwin":
                kw = ["macos", "osx", "macos"]
            else:
                kw = ["ubuntu", "linux"]

            # Prefer cuda/vulkan/metal builds, include cpu as fallback
            result = []
            for rel in releases:
                tag    = rel.get("tag_name", "")
                assets = []
                for a in rel.get("assets", []):
                    n = a["name"].lower()
                    if not any(k in n for k in kw):
                        continue
                    if not n.endswith(".zip") and not n.endswith(".tar.gz"):
                        continue
                    if n.startswith("cudart-"):
                        continue
                    assets.append({
                        "name":  a["name"],
                        "url":   a["browser_download_url"],
                        "size":  a.get("size", 0),
                    })
                if assets:
                    result.append({"tag": tag, "assets": assets})
            self.results_ready.emit(result)
        except Exception as e:
            self.err.emit(str(e))


class LlamaCppDownloadWorker(QThread):
    """Downloads and extracts a llama.cpp release zip/tar.gz into ./llama/."""
    progress  = pyqtSignal(int, int)   # bytes_done, bytes_total
    done      = pyqtSignal(str)        # install path
    err       = pyqtSignal(str)
    status    = pyqtSignal(str)        # status message

    CHUNK = 262144

    def __init__(self, url: str, filename: str, dest_dir: Path, expected_size: int = 0):
        super().__init__()
        self._url      = url
        self._filename = filename
        self._dest_dir = dest_dir
        self._expected_size = expected_size
        self._abort    = False

    def abort(self):
        self._abort = True

    def run(self):
        import urllib.request as _ur, zipfile, tarfile, tempfile, shutil

        tmp = Path(tempfile.mkdtemp())
        archive = tmp / self._filename
        try:
            # Download
            self.status.emit(f"Downloading {self._filename}…")
            req = _ur.Request(self._url, headers={"User-Agent": "NativeLabPro/2"})
            with _ur.urlopen(req, timeout=60) as r:
                total = self._expected_size or int(r.headers.get("Content-Length") or 0)
                done  = 0
                with open(archive, "wb") as f:
                    while not self._abort:
                        chunk = r.read(self.CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        self.progress.emit(done, total)
            if self._abort:
                return

            # Extract
            self.status.emit("Extracting…")
            extract_to = tmp / "extracted"
            extract_to.mkdir()
            if self._filename.endswith(".zip"):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(extract_to)
            else:
                with tarfile.open(archive) as t:
                    t.extractall(extract_to)

            # Find the bin folder (look for llama-server or llama-cli)
            bin_src = None
            for p in extract_to.rglob("llama-server*"):
                if p.is_file():
                    bin_src = p.parent
                    break
            if not bin_src:
                for p in extract_to.rglob("llama-cli*"):
                    if p.is_file():
                        bin_src = p.parent
                        break
            if not bin_src:
                subdirs = [x for x in extract_to.iterdir() if x.is_dir()]
                bin_src = subdirs[0] if subdirs else extract_to

            self.status.emit(f"Found binaries at: {bin_src}")

            # Copy into ./llama/bin/
            # Verify the binary is actually executable on this platform
            import platform as _plat
            _machine = _plat.machine().lower()
            _bin_name = bin_src.name.lower() if bin_src else ""
            # Warn if arm64 binary on x86 or vice versa
            if "aarch64" in str(bin_src).lower() and "x86" in _machine:
                self.err.emit("Downloaded ARM64 build but running on x86_64 — pick the x64 build instead")
                return
            if "x64" in str(bin_src).lower() and "aarch64" in _machine:
                self.err.emit("Downloaded x64 build but running on ARM64 — pick the aarch64 build instead")
                return

            self.status.emit("Installing to ./llama/bin/…")
            dest_bin = self._dest_dir / "bin"
            if dest_bin.exists():
                shutil.rmtree(dest_bin)
            shutil.copytree(bin_src, dest_bin)

            # Make binaries executable on Linux/macOS
            import platform, stat
            if platform.system() != "Windows":
                for f in dest_bin.iterdir():
                    if f.is_file():
                        f.chmod(f.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

            self.done.emit(str(dest_bin))
        except Exception as e:
            self.err.emit(str(e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)