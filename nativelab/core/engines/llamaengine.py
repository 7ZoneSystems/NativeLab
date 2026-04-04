from imports.import_global import Optional, subprocess, time, json, Path, QThread, HAS_PSUTIL, psutil
from components.components_global import detect_model_family
from Model.model_global import get_model_registry
from core.streamer_global import ServerStreamWorker, CliStreamWorker
from GlobalConfig.config_global import (
    LLAMA_CLI, LLAMA_SERVER, DEFAULT_CTX, DEFAULT_THREADS, DEFAULT_N_PRED, APP_CONFIG
)
from Server.server_global import free_port, SERVER_CONFIG, PORT_RANGE_START, PORT_RANGE_END


class LlamaEngine:
    def __init__(self):
        self.server_proc: Optional[subprocess.Popen] = None
        self.server_port: int = 0
        self.model_path:  str = ""
        self.ctx_value:   int = DEFAULT_CTX()
        self.mode = "unloaded"
        self._log = lambda m: None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def load(self, model_path: str,
             threads: int = DEFAULT_THREADS(),
             ctx:     int = DEFAULT_CTX(),
             log_cb=None) -> bool:
        self.model_path = model_path
        self._log = log_cb or (lambda m: None)
        self.ctx_value = ctx

        if not Path(model_path).exists():
            self._log(f"[ERROR] Model not found: {model_path}")
            return False

        if Path(LLAMA_SERVER).exists():
            ok = self._start_server(model_path, threads, ctx)
            if ok:
                return True
            self._log("[WARN] Server start failed — falling back to llama-cli mode")

        if Path(LLAMA_CLI).exists():
            self._log("[INFO] Using llama-cli (per-prompt) mode")
            self.mode = "cli"
            return True

        self._log("[ERROR] Neither llama-server nor llama-cli found")
        return False

    def create_worker(self, prompt: str, n_predict: int = DEFAULT_N_PRED,
                      model_path: str = "") -> QThread:
        fam = detect_model_family(model_path or self.model_path)
        cfg = get_model_registry().get_config(self.model_path)

        if self.mode == "server":
            return ServerStreamWorker(
                self.server_port, prompt, n_predict,
                stop_tokens=fam.stop_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repeat_penalty=cfg.repeat_penalty,
            )

        _extra_cli = SERVER_CONFIG.extra_cli_args.split() if SERVER_CONFIG.extra_cli_args else []
        cmd = [
            LLAMA_CLI, "-m", self.model_path,
            "-t", str(DEFAULT_THREADS()), "--ctx-size", str(self.ctx_value),
            "-n", str(n_predict), "--no-display-prompt", "--no-escape",
            "-p", prompt,
        ] + _extra_cli
        return CliStreamWorker(cmd)

    def ensure_server(self, log_cb=None) -> bool:
        """
        Return True immediately if already in server mode.
        Otherwise attempt to start a server; return False on failure.
        """
        if self.mode == "server":
            return True
        if not self.model_path or not Path(self.model_path).exists():
            return False
        if log_cb:
            self._log = log_cb

        self._log(f"[INFO] ensure_server: starting server for {Path(self.model_path).name}")
        ok = self._start_server(self.model_path, DEFAULT_THREADS(), self.ctx_value)
        level = "INFO" if ok else "WARN"
        msg   = f"started on port {self.server_port}" if ok else "start failed"
        self._log(f"[{level}] ensure_server: {msg}")
        return ok

    def ensure_server_or_reload(self, log_cb=None) -> bool:
        """
        Like ensure_server but kills any stale server proc first,
        then retries once on a fresh port.
        """
        if self.mode == "server":
            return True
        if self.server_proc:
            self._kill_proc(self.server_proc)
            self.server_proc = None
        self.server_port = free_port()
        return self.ensure_server(log_cb=log_cb)

    def shutdown(self):
        if self.server_proc:
            self._kill_proc(self.server_proc)
        self.server_proc = None
        self.mode = "unloaded"

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self.mode != "unloaded"

    @property
    def status_text(self) -> str:
        if self.mode == "server": return f"🟢 Server  :{self.server_port}"
        if self.mode == "cli":    return "🟡 CLI Mode"
        return "⚪ Not Loaded"

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _kill_proc(self, proc: subprocess.Popen) -> None:
        """Terminate a subprocess, killing child processes first if psutil is available."""
        pid = getattr(proc, "pid", None)
        try:
            if HAS_PSUTIL and pid:
                try:
                    parent = psutil.Process(pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    parent.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    proc.terminate()
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass

    def _check_existing_server(self, port: int, timeout: float = 2.0) -> bool:
        """Ping /health on the given port; return True if server is alive."""
        import http.client
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            conn.request("GET", "/health")
            res = conn.getresponse()
            return res.status in (200, 404)
        except Exception:
            return False

    def _start_server(self, model_path: str, threads: int, ctx: int) -> bool:
        import http.client

        # ── 1. Try to reuse an already-running server for this exact model ──
        for test_port in range(PORT_RANGE_START, PORT_RANGE_END):
            if not self._check_existing_server(test_port):
                continue
            try:
                conn = http.client.HTTPConnection("127.0.0.1", test_port, timeout=2)
                conn.request("GET", "/props")
                res   = conn.getresponse()
                props = json.loads(res.read().decode("utf-8", errors="replace"))
                running_model = props.get(
                    "model_path",
                    props.get("default_generation_settings", {}).get("model", "")
                )
                if Path(running_model).resolve() == Path(model_path).resolve():
                    self.server_port = test_port
                    self.mode = "server"
                    self._log(f"[INFO] Reusing existing server for same model on port {test_port}")
                    return True
                self._log(f"[INFO] Port {test_port} has a different model — skipping")
            except Exception:
                pass

        # ── 2. Kill our own stale server proc if still alive ──
        if self.server_proc and self.server_proc.poll() is None:
            self._kill_proc(self.server_proc)
            self.server_proc = None

        # ── 3. Launch a fresh server ──
        self.server_port = free_port()
        _extra_srv = SERVER_CONFIG.extra_server_args.split() if SERVER_CONFIG.extra_server_args else []
        cmd = [
            LLAMA_SERVER, "-m", model_path,
            "-t", str(threads), "--ctx-size", str(ctx),
            "--port", str(self.server_port),
            "--host", SERVER_CONFIG.host or "127.0.0.1",
        ] + _extra_srv

        self._log(f"[INFO] Starting llama-server on port {self.server_port}…")

        # Log to a temp file instead of DEVNULL so failures are diagnosable
        import tempfile
        log_path = Path(tempfile.gettempdir()) / f"llama_server_{self.server_port}.log"
        self._log(f"[INFO] Server output → {log_path}")

        try:
            log_fh = open(log_path, "w")
            self.server_proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        except Exception as e:
            self._log(f"[ERROR] Could not start server: {e}")
            self.server_proc = None
            return False

        # ── 4. Poll until ready or timeout ──
        max_wait   = int(APP_CONFIG.get("server_startup_timeout", 120))  # seconds, configurable
        poll_every = 0.5
        polls      = int(max_wait / poll_every)

        for _ in range(polls):
            time.sleep(poll_every)

            if self.server_proc is None:
                return False

            if self.server_proc.poll() is not None:
                self._log(
                    f"[ERROR] llama-server exited unexpectedly — "
                    f"check {log_path} for details"
                )
                self.server_proc = None
                return False

            if self._check_existing_server(self.server_port):
                self._log(f"[INFO] llama-server ready on port {self.server_port}")
                self.mode = "server"
                return True

        self._log(
            f"[ERROR] Server did not respond within {max_wait}s — "
            f"check {log_path} for details"
        )
        try:
            self.server_proc.terminate()
        except Exception:
            pass
        self.server_proc = None
        return False