# hfwld.py - HuggingFace GGUF download workers
from nativelab.imports.import_global import QThread, pyqtSignal, Path
from nativelab.GlobalConfig.config_global import LONG_TIMEOUT_SECONDS
from nativelab.Server.hfauth import hf_auth_headers, normalize_hf_exception
from nativelab.Server.ollama_helpers import normalize_ollama_exception, normalize_ollama_host
import threading


class HfSearchWorker(QThread):
    """Queries the HuggingFace API for GGUF siblings in a repo."""
    results_ready = pyqtSignal(list)
    err           = pyqtSignal(str)

    def __init__(self, repo_id: str, token: str = ""):
        super().__init__()
        self._repo = repo_id.strip().strip("/")
        self._token = token

    def run(self):
        try:
            self.results_ready.emit(fetch_hf_gguf_files(self._repo, self._token))
        except Exception as e:
            self.err.emit(normalize_hf_exception(e, repo_id=self._repo))


def fetch_hf_gguf_files(repo_id: str, token: str = "") -> list:
    """Return GGUF file metadata for a Hugging Face repo."""
    import urllib.request as _ur, json as _j

    repo = str(repo_id or "").strip().strip("/")
    url = f"https://huggingface.co/api/models/{repo}"
    req = _ur.Request(url, headers=hf_auth_headers(token))
    with _ur.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
        data = _j.loads(r.read())
    siblings = data.get("siblings", [])
    return [
        s for s in siblings
        if str(s.get("rfilename", "")).lower().endswith(".gguf")
    ]


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
                 expected_size: int = 0, token: str = ""):
        super().__init__()
        self._repo          = repo_id.strip().strip("/")
        self._filename      = filename
        self._dest_dir      = dest_dir
        self._expected_size = expected_size
        self._token         = token
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
        dest = _safe_child_path(self._dest_dir, self._filename)
        part = self._part_path(dest)
        self._dest_dir.mkdir(parents=True, exist_ok=True)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Resolve CDN URL once upfront (avoids repeated redirect overhead)
        try:
            final_url = self._resolve_final_url(url)
        except Exception:
            final_url = url

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._download_loop(final_url, part)
                break   # clean finish - exit retry loop
            except _AbortedError as e:
                if e.delete_part:
                    try: part.unlink()
                    except Exception: pass
                return   # exit silently either way - not an error
            except Exception as exc:
                if attempt >= self.MAX_RETRIES:
                    # Genuine failure - keep .part so user can resume next session
                    self.err.emit(normalize_hf_exception(exc, repo_id=self._repo))
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
        req = _ur.Request(url, headers=hf_auth_headers(self._token),
                          method="HEAD")
        with _ur.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
            return r.url

    def _part_path(self, dest: Path) -> Path:
        return dest.with_suffix(dest.suffix + ".part")

    def _download_loop(self, final_url: str, part: Path):
        import urllib.request as _ur

        # Resume: how many bytes do we already have from a previous run / retry?
        resume_from = part.stat().st_size if part.exists() else 0

        headers = hf_auth_headers(self._token)
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        req = _ur.Request(final_url, headers=headers)
        with _ur.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
            cl = r.headers.get("Content-Length")

            if resume_from and r.status == 206:   # server honoured Range
                total = resume_from + (int(cl) if cl else 0)
            else:
                # Server ignored Range (e.g. returned 200) - start fresh
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


def _hf_auth_headers(token: str = "") -> dict:
    return hf_auth_headers(token)


def _safe_child_path(root: Path, name: str) -> Path:
    raw = str(name or "").replace("\\", "/").strip("/")
    if not raw:
        raise ValueError("Empty download filename")
    rel = Path(raw)
    if rel.is_absolute() or any(part in {"..", ""} for part in rel.parts):
        raise ValueError(f"Unsafe download filename: {name}")
    base = Path(root).resolve()
    dest = (Path(root) / rel).resolve()
    try:
        dest.relative_to(base)
    except ValueError:
        raise ValueError(f"Download path escapes destination: {name}")
    return Path(root) / rel


def _hf_runtime_file(name: str) -> bool:
    n = str(name or "").lower()
    base = Path(n).name
    if not n or n.endswith("/"):
        return False
    if base in {"readme.md", "license", "license.txt", ".gitattributes"}:
        return True
    if n.endswith((".safetensors", ".safetensors.index.json")):
        return True
    if base.startswith("pytorch_model") and n.endswith((".bin", ".bin.index.json")):
        return True
    if base in {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "preprocessor_config.json", "processor_config.json",
        "chat_template.json", "added_tokens.json", "merges.txt",
        "vocab.json", "vocab.txt", "tokenizer.model",
        "sentencepiece.bpe.model", "modeling.py",
    }:
        return True
    if n.endswith(".py"):
        return True
    if n.endswith(".json") and any(k in base for k in ("config", "tokenizer", "processor", "index")):
        return True
    return False


class HfSnapshotSearchWorker(QThread):
    """Queries a Hugging Face repo and returns files needed for Transformers."""
    results_ready = pyqtSignal(dict)
    err = pyqtSignal(str)

    def __init__(self, repo_id: str, revision: str = "main", token: str = ""):
        super().__init__()
        self._repo = repo_id.strip().strip("/")
        self._revision = (revision or "main").strip()
        self._token = token

    def run(self):
        try:
            self.results_ready.emit(fetch_hf_snapshot_files(self._repo, self._revision, self._token))
        except Exception as e:
            self.err.emit(normalize_hf_exception(e, repo_id=self._repo))


def fetch_hf_snapshot_files(repo_id: str, revision: str = "main", token: str = "") -> dict:
    """Return files needed for a local HF Transformers runtime snapshot."""
    import urllib.request as _ur, json as _j
    from urllib.parse import quote as _quote

    repo = str(repo_id or "").strip().strip("/")
    rev_text = (revision or "main").strip()
    rev = _quote(rev_text, safe="")
    url = f"https://huggingface.co/api/models/{repo}/revision/{rev}"
    req = _ur.Request(url, headers=_hf_auth_headers(token))
    with _ur.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
        data = _j.loads(r.read().decode("utf-8", errors="replace"))
    files = []
    for sib in data.get("siblings", []):
        name = sib.get("rfilename", "")
        if not _hf_runtime_file(name):
            continue
        size = int(sib.get("size") or sib.get("lfs", {}).get("size") or 0)
        files.append({"name": name, "size": size})
    files.sort(key=lambda x: x["name"].lower())
    return {
        "repo": repo,
        "revision": rev_text,
        "sha": data.get("sha", ""),
        "files": files,
        "total_size": sum(int(f.get("size") or 0) for f in files),
    }


class HfSnapshotDownloadWorker(QThread):
    """Downloads a full HF Transformers runtime snapshot with resume support."""
    progress = pyqtSignal(int, int, str)
    status = pyqtSignal(str)
    done = pyqtSignal(str)
    err = pyqtSignal(str)
    paused = pyqtSignal(bool)

    CHUNK = 262144
    MAX_RETRIES = 3
    RETRY_WAIT = 4

    def __init__(self, repo_id: str, revision: str, files: list, dest_root: Path, token: str = ""):
        super().__init__()
        self._repo = repo_id.strip().strip("/")
        self._revision = (revision or "main").strip()
        self._files = list(files or [])
        self._dest_root = Path(dest_root)
        self._token = token
        self._abort = False
        self._abort_delete = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._done_total = 0
        self._expected_total = sum(int(f.get("size") or 0) for f in self._files)

    def abort(self, delete_part: bool = False):
        self._abort = True
        self._abort_delete = bool(delete_part)
        self._pause_event.set()

    def pause(self):
        self._pause_event.clear()
        self.paused.emit(True)

    def resume(self):
        self._pause_event.set()
        self.paused.emit(False)

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def run(self):
        import time

        dest_dir = self._snapshot_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        self._done_total = self._existing_bytes(dest_dir)
        self.progress.emit(self._done_total, self._expected_total, "Preparing snapshot")
        for fdata in self._files:
            if self._abort:
                if self._abort_delete:
                    self._delete_partials(dest_dir)
                return
            name = fdata.get("name", "")
            expected = int(fdata.get("size") or 0)
            self.status.emit(f"Downloading {name}")
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    self._download_one(dest_dir, name, expected)
                    break
                except _AbortedError as e:
                    if e.delete_part:
                        self._delete_partials(dest_dir)
                    return
                except Exception as exc:
                    if attempt >= self.MAX_RETRIES:
                        self.err.emit(f"{name}: {normalize_hf_exception(exc, repo_id=self._repo)}")
                        return
                    time.sleep(self.RETRY_WAIT)
        self.done.emit(str(dest_dir))

    def _snapshot_dir(self) -> Path:
        parts = [p for p in self._repo.split("/") if p]
        if len(parts) >= 2:
            return self._dest_root / parts[0] / parts[1]
        return self._dest_root / self._repo.replace("/", "--")

    def _existing_bytes(self, dest_dir: Path) -> int:
        total = 0
        for fdata in self._files:
            dest = _safe_child_path(dest_dir, fdata.get("name", ""))
            part = self._part_path(dest)
            expected = int(fdata.get("size") or 0)
            if dest.exists():
                size = dest.stat().st_size
                if not expected or size >= expected:
                    total += size
                    continue
            if part.exists():
                total += part.stat().st_size
        return total

    def _part_path(self, dest: Path) -> Path:
        return dest.with_name(dest.name + ".part")

    def _delete_partials(self, dest_dir: Path):
        for fdata in self._files:
            name = fdata.get("name", "")
            if not name:
                continue
            try:
                self._part_path(_safe_child_path(dest_dir, name)).unlink()
            except Exception:
                pass

    def _resolve_url(self, filename: str) -> str:
        from urllib.parse import quote as _quote
        quoted = "/".join(_quote(part, safe="") for part in filename.split("/"))
        rev = _quote(self._revision, safe="")
        return f"https://huggingface.co/{self._repo}/resolve/{rev}/{quoted}"

    def _download_one(self, dest_dir: Path, filename: str, expected_size: int):
        import urllib.request as _ur

        dest = _safe_child_path(dest_dir, filename)
        part = self._part_path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (not expected_size or dest.stat().st_size >= expected_size):
            return
        resume_from = part.stat().st_size if part.exists() else 0
        headers = _hf_auth_headers(self._token)
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        req = _ur.Request(self._resolve_url(filename), headers=headers)
        with _ur.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
            if resume_from and getattr(r, "status", 200) != 206:
                self._done_total = max(0, self._done_total - resume_from)
                resume_from = 0
            mode = "ab" if resume_from else "wb"
            with open(part, mode) as fh:
                while True:
                    self._pause_event.wait()
                    if self._abort:
                        raise _AbortedError(delete_part=self._abort_delete)
                    chunk = r.read(self.CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    self._done_total += len(chunk)
                    self.progress.emit(self._done_total, self._expected_total, filename)
        part.replace(dest)


class OllamaListWorker(QThread):
    results_ready = pyqtSignal(list)
    err = pyqtSignal(str)

    def __init__(self, host: str):
        super().__init__()
        self._host = normalize_ollama_host(host)

    def run(self):
        import urllib.request as _ur, json as _j
        try:
            req = _ur.Request(f"{self._host}/api/tags", headers={"User-Agent": "NativeLabPro/2"})
            with _ur.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
                data = _j.loads(r.read().decode("utf-8", errors="replace"))
            self.results_ready.emit(data.get("models") or [])
        except Exception as e:
            self.err.emit(normalize_ollama_exception(e, self._host, action="list models from"))


class OllamaPullWorker(QThread):
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(str)
    err = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, host: str, model_name: str):
        super().__init__()
        self._host = normalize_ollama_host(host)
        self._model = model_name.strip()
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        import urllib.request as _ur, json as _j
        try:
            body = _j.dumps({"name": self._model, "stream": True}).encode("utf-8")
            req = _ur.Request(
                f"{self._host}/api/pull",
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "NativeLabPro/2"},
            )
            with _ur.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
                last_status = ""
                for raw in r:
                    if self._abort:
                        return
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    data = _j.loads(line)
                    if data.get("error"):
                        self.err.emit(str(data.get("error")))
                        return
                    status = str(data.get("status") or "")
                    last_status = status or last_status
                    completed = int(data.get("completed") or 0)
                    total = int(data.get("total") or 0)
                    if status:
                        self.status.emit(status)
                    self.progress.emit(completed, total, status or last_status)
            self.done.emit(self._model)
        except Exception as e:
            self.err.emit(normalize_ollama_exception(e, self._host, action="pull from"))


class LlamaCppReleaseFetcher(QThread):
    """Fetches llama.cpp GitHub releases and available assets for the current platform."""
    results_ready = pyqtSignal(list)   # list of {tag, assets: [{name, url, size}]}
    err           = pyqtSignal(str)

    def run(self):
        try:
            self.results_ready.emit(fetch_llama_cpp_releases())
        except Exception as e:
            self.err.emit(str(e))


def fetch_llama_cpp_releases(per_page: int = 10) -> list:
    """Fetch compatible llama.cpp release assets for this platform."""
    import urllib.request as _ur, json as _j, platform as _pl

    url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page={int(per_page)}"
    req = _ur.Request(url, headers={
        "User-Agent": "NativeLabPro/2",
        "Accept": "application/vnd.github+json",
    })
    with _ur.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
        releases = _j.loads(r.read())

    sys_name = _pl.system().lower()
    if sys_name == "windows":
        kw = ["win"]
    elif sys_name == "darwin":
        kw = ["macos", "osx"]
    else:
        kw = ["ubuntu", "linux"]

    result = []
    for rel in releases:
        tag = rel.get("tag_name", "")
        assets = []
        for a in rel.get("assets", []):
            n = str(a.get("name", "")).lower()
            if not any(k in n for k in kw):
                continue
            if not n.endswith(".zip") and not n.endswith(".tar.gz"):
                continue
            if n.startswith("cudart-"):
                continue
            assets.append({
                "name": a.get("name", ""),
                "url": a.get("browser_download_url", ""),
                "size": a.get("size", 0),
            })
        if assets:
            result.append({"tag": tag, "assets": assets})
    return result


class LlamaCppDownloadWorker(QThread):
    """Downloads and extracts a llama.cpp release zip/tar.gz into ./llama/."""
    progress  = pyqtSignal(int, int)   # bytes_done, bytes_total
    done      = pyqtSignal(str)        # install path
    err       = pyqtSignal(str)
    status    = pyqtSignal(str)        # status message
    paused    = pyqtSignal(bool)

    CHUNK = 262144

    def __init__(self, url: str, filename: str, dest_dir: Path, expected_size: int = 0):
        super().__init__()
        self._url      = url
        self._filename = filename
        self._dest_dir = dest_dir
        self._expected_size = expected_size
        self._abort    = False
        self._abort_delete = False
        self._pause_event = threading.Event()
        self._pause_event.set()

    def abort(self, delete_part: bool = False):
        self._abort_delete = bool(delete_part)
        self._abort = True
        self._pause_event.set()

    def pause(self):
        self._pause_event.clear()
        self.paused.emit(True)

    def resume(self):
        self._pause_event.set()
        self.paused.emit(False)

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def run(self):
        import zipfile, tarfile, tempfile, shutil

        tmp = Path(tempfile.mkdtemp())
        archive = self._archive_path()
        try:
            self._download_archive(archive)
            if self._abort:
                if self._abort_delete:
                    try:
                        archive.unlink()
                    except Exception:
                        pass
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
                self.err.emit("Downloaded ARM64 build but running on x86_64 - pick the x64 build instead")
                return
            if "x64" in str(bin_src).lower() and "aarch64" in _machine:
                self.err.emit("Downloaded x64 build but running on ARM64 - pick the aarch64 build instead")
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

    def _archive_path(self) -> Path:
        downloads = self._dest_dir / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        return downloads / self._filename

    def _part_path(self, archive: Path) -> Path:
        return archive.with_name(archive.name + ".part")

    def _download_archive(self, archive: Path):
        import urllib.request as _ur

        expected = int(self._expected_size or 0)
        if archive.exists() and (not expected or archive.stat().st_size >= expected):
            self.status.emit(f"Using cached {self._filename}")
            self.progress.emit(archive.stat().st_size, expected or archive.stat().st_size)
            return

        part = self._part_path(archive)
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "NativeLabPro/2"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        self.status.emit(f"Downloading {self._filename}...")
        req = _ur.Request(self._url, headers=headers)
        with _ur.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
            cl = r.headers.get("Content-Length")
            if resume_from and getattr(r, "status", 200) == 206:
                total = resume_from + (int(cl) if cl else 0)
            else:
                total = expected or (int(cl) if cl and int(cl) > 0 else 0)
                resume_from = 0
            done = resume_from
            mode = "ab" if resume_from else "wb"
            with open(part, mode) as f:
                while True:
                    self._pause_event.wait()
                    if self._abort:
                        if self._abort_delete:
                            try:
                                part.unlink()
                            except Exception:
                                pass
                        return
                    chunk = r.read(self.CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    self.progress.emit(done, total)
        if total and done != total:
            raise ValueError(f"Size mismatch: expected {total} B, got {done} B")
        part.replace(archive)
