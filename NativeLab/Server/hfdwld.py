from imports.import_global import QThread, pyqtSignal, Path
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


class HfDownloadWorker(QThread):
    """Downloads a single GGUF file from HuggingFace."""
    progress = pyqtSignal(int, int)   # bytes_done, bytes_total
    done     = pyqtSignal(str)        # final save path
    err      = pyqtSignal(str)

    def __init__(self, repo_id: str, filename: str, dest_dir: Path):
        super().__init__()
        self._repo     = repo_id.strip().strip("/")
        self._filename = filename
        self._dest_dir = dest_dir
        self._abort    = False

    def abort(self):
        self._abort = True

    def run(self):
        import urllib.request as _ur
        url  = f"https://huggingface.co/{self._repo}/resolve/main/{self._filename}"
        dest = self._dest_dir / self._filename
        self._dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            req = _ur.Request(url, headers={"User-Agent": "NativeLabPro/2"})
            with _ur.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0))
                done  = 0
                CHUNK = 262144   # 256 KB
                with open(dest, "wb") as f:
                    while True:
                        if self._abort:
                            try: dest.unlink()
                            except Exception: pass
                            return
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        self.progress.emit(done, total)
            self.done.emit(str(dest))
        except Exception as e:
            try: dest.unlink()
            except Exception: pass
            self.err.emit(str(e))
