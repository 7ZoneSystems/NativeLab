from imports.import_global import QThread, pyqtSignal, json, time
class ApiStreamWorker(QThread):
    """Streams tokens from any OpenAI-compatible or Anthropic API endpoint."""
    token = pyqtSignal(str)
    done  = pyqtSignal(float)
    err   = pyqtSignal(str)

    def __init__(self, messages: list, api_key: str, base_url: str,
                 model_id: str, api_format: str = "openai",
                 max_tokens: int = 1024, temperature: float = 0.7):
        super().__init__()
        self.messages    = messages
        self.api_key     = api_key
        self.base_url    = base_url.rstrip("/")
        self.model_id    = model_id
        self.api_format  = api_format
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._abort      = False

    def run(self):
        import urllib.request, urllib.error
        t0 = time.time()
        n  = 0
        try:
            # Apply custom prompt wrapping when requested
            if getattr(self, "use_custom_prompt", False) and self.messages:
                up  = getattr(self, "user_prefix", "")
                us  = getattr(self, "user_suffix", "")
                ap  = getattr(self, "assistant_prefix", "")
                sys = getattr(self, "system_prompt", "")
                wrapped = []
                if sys:
                    wrapped.append({"role": "system", "content": sys})
                for m in self.messages:
                    if m["role"] == "user":
                        wrapped.append({"role": "user",
                                        "content": f"{up}{m['content']}{us}"})
                    elif m["role"] == "assistant":
                        wrapped.append({"role": "assistant",
                                        "content": f"{ap}{m['content']}"})
                    else:
                        wrapped.append(m)
                self.messages = wrapped

            if self.api_format == "anthropic":
                # ── Anthropic streaming ───────────────────────────────────────
                sys_msg = next((m["content"] for m in self.messages
                                if m["role"] == "system"), "")
                msgs    = [m for m in self.messages if m["role"] != "system"]
                body    = json.dumps({
                    "model":      self.model_id,
                    "messages":   msgs,
                    "stream":     True,
                    "max_tokens": self.max_tokens,
                    **({"system": sys_msg} if sys_msg else {}),
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/v1/messages", data=body,
                    headers={"Content-Type": "application/json",
                             "x-api-key": self.api_key,
                             "anthropic-version": "2023-06-01"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw in resp:
                        if self._abort: break
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data: "): continue
                        try:
                            obj = json.loads(line[6:])
                            tok = obj.get("delta", {}).get("text", "")
                            if tok:
                                n += 1; self.token.emit(tok)
                        except Exception:
                            pass
            else:
                # ── OpenAI-compatible streaming ───────────────────────────────
                body = json.dumps({
                    "model":       self.model_id,
                    "messages":    self.messages,
                    "stream":      True,
                    "max_tokens":  self.max_tokens,
                    "temperature": self.temperature,
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions", data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw in resp:
                        if self._abort: break
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data: "): continue
                        data = line[6:]
                        if data == "[DONE]": break
                        try:
                            obj = json.loads(data)
                            tok = (obj.get("choices", [{}])[0]
                                      .get("delta", {}).get("content") or "")
                            if tok:
                                n += 1; self.token.emit(tok)
                        except Exception:
                            pass
            elapsed = time.time() - t0
            self.done.emit(n / elapsed if elapsed > 0 else 0.0)
        except Exception as e:
            self.err.emit(str(e))

    def abort(self):
        self._abort = True
