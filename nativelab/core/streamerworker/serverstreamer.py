from imports.import_global import QThread, pyqtSignal, json, time, List, Optional
from GlobalConfig.config_global import DEFAULT_N_PRED, APP_CONFIG

class ServerStreamWorker(QThread):
    token = pyqtSignal(str)
    done  = pyqtSignal(float)
    err   = pyqtSignal(str)

    def __init__(self, port: int, prompt: str, n_predict: int = DEFAULT_N_PRED,
                 stop_tokens: Optional[List[str]] = None, temperature: float = 0.7,
                 top_p: float = 0.9, repeat_penalty: float = 1.1):
        super().__init__()
        self.port           = port
        self.prompt         = prompt
        self.n_predict      = n_predict
        self.stop_tokens    = stop_tokens or ["</s>", "[INST]", "### Human:"]
        self.temperature    = temperature
        self.top_p          = top_p
        self.repeat_penalty = repeat_penalty
        self._abort         = False

    def run(self):
        import http.client

        # Configurable timeouts
        socket_timeout = int(APP_CONFIG.get("stream_socket_timeout", 32000))   # per-read stall limit
        stall_timeout  = int(APP_CONFIG.get("stream_stall_timeout", 32000))     # no-token stall detection
        max_buf        = int(APP_CONFIG.get("stream_max_buf_bytes", 65536))  # runaway buffer guard

        t0 = time.time()
        last_token_at = time.time()
        n = 0

        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=socket_timeout)
            body = json.dumps({
                "prompt":         self.prompt,
                "n_predict":      self.n_predict,
                "stream":         True,
                "temperature":    self.temperature,
                "top_p":          self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "stop":           self.stop_tokens,
            })
            conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200:
                self.err.emit(f"HTTP {r.status}"); return

            buf = b""
            # Fix: read in chunks instead of byte-by-byte
            while not self._abort:
                # Stall detection — emit error if no token for stall_timeout seconds
                if time.time() - last_token_at > stall_timeout:
                    self.err.emit(f"Stream stalled — no tokens for {stall_timeout}s"); return

                chunk = r.read(64)   # 64 bytes per read — balanced latency vs syscall cost
                if not chunk:
                    break

                buf += chunk

                # Guard against runaway buffer
                if len(buf) > max_buf:
                    self.err.emit("Stream buffer overflow — possible malformed response"); return

                # Process all complete lines in the buffer
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        d = json.loads(line[6:])
                        if d.get("stop"):
                            buf = b""  # discard remainder
                            break
                        c = d.get("content", "")
                        if c:
                            n += 1
                            last_token_at = time.time()
                            self.token.emit(c)
                    except json.JSONDecodeError:
                        pass
                else:
                    continue
                break  # inner break (stop token) propagates out

        except Exception as e:
            self.err.emit(f"{type(e).__name__}: {e}")
            return

        elapsed = time.time() - t0
        self.done.emit(n / elapsed if elapsed > 0 else 0.0)

    def abort(self):
        self._abort = True