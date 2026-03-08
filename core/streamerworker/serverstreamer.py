from imports.import_global import QThread, pyqtSignal, json, time, List
from GlobalConfig.config_global import DEFAULT_N_PRED
class ServerStreamWorker(QThread):
    token  = pyqtSignal(str)
    done   = pyqtSignal(float)
    err    = pyqtSignal(str)

    def __init__(self, port: int, prompt: str, n_predict: int = DEFAULT_N_PRED,
                 stop_tokens: List[str] = None, temperature: float = 0.7,
                 top_p: float = 0.9, repeat_penalty: float = 1.1):
        super().__init__()
        self.port          = port
        self.prompt        = prompt
        self.n_predict     = n_predict
        self.stop_tokens   = stop_tokens or ["</s>", "[INST]", "### Human:"]
        self.temperature   = temperature
        self.top_p         = top_p
        self.repeat_penalty = repeat_penalty
        self._abort        = False

    def run(self):
        import http.client
        t0 = time.time(); n = 0
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=900)
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
            while not self._abort:
                b = r.read(1)
                if not b: break
                buf += b
                if b == b"\n":
                    line = buf.decode("utf-8", errors="replace").strip()
                    buf  = b""
                    if line.startswith("data: "):
                        try:
                            d = json.loads(line[6:])
                            if d.get("stop"): break
                            c = d.get("content", "")
                            if c:
                                n += 1; self.token.emit(c)
                        except json.JSONDecodeError:
                            pass

            elapsed = time.time() - t0
            self.done.emit(n / elapsed if elapsed > 0 else 0.0)
        except Exception as e:
            self.err.emit(str(e))

    def abort(self):
        self._abort = True
