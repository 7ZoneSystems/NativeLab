from nativelab.imports.import_global import QThread, pyqtSignal, json, time
from nativelab.GlobalConfig.config_global import LONG_TIMEOUT_SECONDS

ANTHROPIC_UNCAPPED_FALLBACK_TOKENS = 8192


class ApiStreamWorker(QThread):
    """Streams tokens from any OpenAI-compatible or Anthropic API endpoint."""
    token = pyqtSignal(str)
    done  = pyqtSignal(float)
    err   = pyqtSignal(str)
    log   = pyqtSignal(str)

    def __init__(self, messages: list, api_key: str, base_url: str,
                 model_id: str, api_format: str = "openai",
                 max_tokens=None, temperature: float = 0.7):
        super().__init__()
        self.messages    = messages
        self.api_key     = api_key
        self.base_url    = base_url.rstrip("/")
        self.model_id    = model_id
        self.api_format  = api_format
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self._abort      = False
        self.use_custom_prompt: bool = False
        self.system_prompt:     str  = ""
        self.user_prefix:       str  = ""
        self.user_suffix:       str  = ""
        self.assistant_prefix:  str  = ""

    def _max_tokens_value(self):
        try:
            value = int(self.max_tokens)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
        
    def run(self):
        import urllib.request, urllib.error
        t0 = time.time()
        n  = 0
        try:
            self.log.emit(
                f"[API] Request started — format={self.api_format.upper()}"
                f"  model={self.model_id}  url={self.base_url}"
            )
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
                    content = m.get("content", "")
                    if m["role"] == "user":
                        if isinstance(content, list):
                            new_content = []
                            text_wrapped = False
                            for part in content:
                                if (isinstance(part, dict)
                                        and part.get("type") == "text"
                                        and not text_wrapped):
                                    new_content.append({
                                        **part,
                                        "text": f"{up}{part.get('text', '')}{us}",
                                    })
                                    text_wrapped = True
                                else:
                                    new_content.append(part)
                            wrapped.append({"role": "user", "content": new_content})
                        else:
                            wrapped.append({"role": "user",
                                            "content": f"{up}{content}{us}"})
                    elif m["role"] == "assistant":
                        wrapped.append({"role": "assistant",
                                        "content": f"{ap}{content}"})
                    else:
                        wrapped.append(m)
                self.messages = wrapped

            if self.api_format == "anthropic":
                # ── Anthropic streaming ───────────────────────────────────────
                sys_msg = next((m["content"] for m in self.messages
                                if m["role"] == "system"), "")
                msgs    = [m for m in self.messages if m["role"] != "system"]
                max_tokens = self._max_tokens_value() or ANTHROPIC_UNCAPPED_FALLBACK_TOKENS
                body    = json.dumps({
                    "model":      self.model_id,
                    "messages":   msgs,
                    "stream":     True,
                    "max_tokens": max_tokens,
                    **({"system": sys_msg} if sys_msg else {}),
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/v1/messages", data=body,
                    headers={"Content-Type": "application/json",
                             "x-api-key": self.api_key,
                             "anthropic-version": "2023-06-01"})
                with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as resp:
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
                payload = {
                    "model":       self.model_id,
                    "messages":    self.messages,
                    "stream":      True,
                    "temperature": self.temperature,
                }
                max_tokens = self._max_tokens_value()
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions", data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"})
                with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as resp:
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
            tps = n / elapsed if elapsed > 0 else 0.0
            self.log.emit(
                f"[API] Done — {n} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)"
            )
            self.done.emit(tps)
        except Exception as e:
            self.log.emit(f"[API] Error: {e}")
            self.err.emit(str(e))

    def abort(self):
        self._abort = True
