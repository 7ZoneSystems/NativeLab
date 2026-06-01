import sys
print("LLAMAENGINE LOADED FROM:", __file__, file=sys.stderr, flush=True)
from nativelab.imports.import_global import Optional, subprocess, time, json, Path, QThread, HAS_PSUTIL, psutil
from nativelab.Model.model_global import (
    detect_mmproj_for_model,
    detect_model_family,
    detect_vision_model,
    get_model_registry,
    is_hf_model_ref,
    is_ollama_model_ref,
    model_ref_display_name,
    model_ref_payload,
)
from nativelab.core.streamer_global import ServerStreamWorker, CliStreamWorker, BackendStreamWorker
from nativelab.GlobalConfig.config_global import (
    DEFAULT_CTX, DEFAULT_THREADS, DEFAULT_N_PRED, APP_CONFIG,
    LONG_TIMEOUT_SECONDS,
)
from nativelab.Server.hfauth import get_hf_access_token, normalize_hf_exception
from nativelab.Server.ollama_helpers import normalize_ollama_exception, normalize_ollama_host
import nativelab.GlobalConfig.binaryResolve as _binres
from nativelab.Server.server_global import free_port, SERVER_CONFIG, PORT_RANGE_START, PORT_RANGE_END


class LlamaEngine:
    def __init__(self):
        self.server_proc: Optional[subprocess.Popen] = None
        self.server_port: int = 0
        self.model_path:  str = ""
        self.ctx_value:   int = DEFAULT_CTX()
        self.mode = "unloaded"
        self._log = lambda m: None
        self._pending_images: list = []
        self.ollama_host = normalize_ollama_host(str(APP_CONFIG.get("ollama_host", "http://127.0.0.1:11434")))
        self._hf_model = None
        self._hf_tokenizer = None
        self._hf_processor = None
        self._hf_is_vision = False

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
        print(f"LOAD CALLED: {model_path!r}", file=sys.stderr, flush=True)
        if is_ollama_model_ref(model_path):
            return self._load_ollama(model_path)
        if is_hf_model_ref(model_path):
            return self._load_hf_transformers(model_path)
        import os
        self._log(f"[DEBUG] model_path = {model_path!r}")
        self._log(f"[DEBUG] resolved   = {Path(model_path).resolve()}")
        self._log(f"[DEBUG] cwd        = {os.getcwd()}")
        self._log(f"[DEBUG] exists     = {Path(model_path).exists()}")
        self._log(f"[DEBUG] _binres.LLAMA_SERVER = {_binres.LLAMA_SERVER!r}")
        self._log(f"[DEBUG] srv config  = {SERVER_CONFIG.server_path!r}")
        self._log(f"[DEBUG] _server_bin = {(SERVER_CONFIG.server_path or _binres.LLAMA_SERVER)!r}")
        self._log(f"[DEBUG] srv exists  = {Path(SERVER_CONFIG.server_path or _binres.LLAMA_SERVER).exists()}")
        if not Path(model_path).exists():
            self._log(f"[ERROR] Model not found: {model_path}")
            return False
        _server_bin = SERVER_CONFIG.server_path or _binres.LLAMA_SERVER
        _cli_bin    = SERVER_CONFIG.cli_path    or _binres.LLAMA_CLI
        if Path(_server_bin).exists():
            ok = self._start_server(model_path, threads, ctx)
            if ok:
                return True
            self._log("[WARN] Server start failed - falling back to llama-cli mode")

        if Path(_cli_bin).exists():
            self._log("[INFO] Using llama-cli (per-prompt) mode")
            self.mode = "cli"
            return True

        self._log("[ERROR] Neither llama-server nor llama-cli found")
        return False

    def create_worker(self, prompt: str, n_predict: int = DEFAULT_N_PRED,
                      model_path: str = "") -> QThread:
        cfg = get_model_registry().get_config(self.model_path)
        fam = detect_model_family(model_ref_payload(model_path or self.model_path) or self.model_path)

        if self.mode in ("ollama", "hf_transformers"):
            image_data = list(self._pending_images)
            self._pending_images = []
            return BackendStreamWorker(
                self,
                prompt,
                n_predict,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repeat_penalty=cfg.repeat_penalty,
                image_data=image_data,
                raw_prompt=True,
            )

        if self.mode == "server":
            image_data = list(self._pending_images)
            self._pending_images = []
            return ServerStreamWorker(
                self.server_port, prompt, n_predict,
                stop_tokens=fam.stop_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                repeat_penalty=cfg.repeat_penalty,
                image_data=image_data,
            )

        _extra_cli = SERVER_CONFIG.extra_cli_args.split() if SERVER_CONFIG.extra_cli_args else []
        _cli_bin = SERVER_CONFIG.cli_path or _binres.LLAMA_CLI
        cmd = [
            _cli_bin, "-m", self.model_path,
            "-t", str(DEFAULT_THREADS()), "--ctx-size", str(self.ctx_value),
            "-n", str(n_predict), "--no-display-prompt", "--no-escape",
            "-p", prompt,
        ] + _extra_cli
        return CliStreamWorker(cmd)

    def set_images(self, image_data: list):
        """Attach image payloads to the next backend request."""
        self._pending_images = list(image_data or [])

    def ensure_server(self, log_cb=None) -> bool:
        """
        Return True immediately if already in server mode.
        Otherwise attempt to start a server; return False on failure.
        """
        if self.mode in ("server", "ollama", "hf_transformers"):
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
        self._unload_hf()
        self.mode = "unloaded"

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_loaded(self) -> bool:
        return self.mode != "unloaded"

    @property
    def status_text(self) -> str:
        if self.mode == "server": return f"Server  :{self.server_port}"
        if self.mode == "cli":    return "CLI Mode"
        if self.mode == "ollama": return f"Ollama  ·  {model_ref_display_name(self.model_path)}"
        if self.mode == "hf_transformers": return f"Transformers  ·  {model_ref_display_name(self.model_path)}"
        return "Not Loaded"

    def generate_sync(
        self,
        messages=None,
        prompt: str = "",
        n_predict: int = DEFAULT_N_PRED,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        token_cb=None,
        abort_cb=None,
        image_data=None,
        raw_prompt: bool = False,
    ) -> str:
        """Synchronous generation shared by Labs, pipelines, and backend stream workers."""
        messages = list(messages or [])
        if not messages and prompt:
            messages = [{"role": "user", "content": prompt}]
        if not messages:
            raise RuntimeError("generate_sync: no prompt/messages provided")
        if self.mode == "server":
            return self._generate_server(messages, n_predict, temperature, top_p, repeat_penalty, token_cb, abort_cb, image_data, raw_prompt)
        if self.mode == "cli":
            return self._generate_cli(messages, n_predict, temperature, repeat_penalty, token_cb, abort_cb, raw_prompt)
        if self.mode == "ollama":
            return self._generate_ollama(messages, n_predict, temperature, top_p, repeat_penalty, token_cb, abort_cb, image_data)
        if self.mode == "hf_transformers":
            return self._generate_hf(messages, n_predict, temperature, top_p, token_cb, abort_cb, image_data, raw_prompt)
        raise RuntimeError("No local engine loaded")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _load_ollama(self, ref: str) -> bool:
        import urllib.error
        import urllib.request

        model = model_ref_payload(ref)
        if not model:
            self._log("[ERROR] Empty Ollama model ref")
            return False
        self._unload_hf()
        if self.server_proc:
            self._kill_proc(self.server_proc)
            self.server_proc = None
        self.ollama_host = normalize_ollama_host(str(APP_CONFIG.get("ollama_host", self.ollama_host)))
        try:
            req = urllib.request.Request(
                f"{self.ollama_host.rstrip('/')}/api/show",
                data=json.dumps({"model": model}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as r:
                if r.status >= 400:
                    self._log(f"[ERROR] Ollama rejected model '{model}'")
                    return False
        except urllib.error.HTTPError as exc:
            self._log(f"[ERROR] Ollama model not available: {model}. {normalize_ollama_exception(exc, self.ollama_host, action='load')}")
            return False
        except Exception as exc:
            self._log(f"[ERROR] {normalize_ollama_exception(exc, self.ollama_host, action='load')}")
            return False
        self.mode = "ollama"
        self.model_path = ref
        self._log(f"[INFO] Ollama model ready: {model}")
        return True

    def _load_hf_transformers(self, ref: str) -> bool:
        target = model_ref_payload(ref)
        if not target:
            self._log("[ERROR] Empty Hugging Face model ref")
            return False
        expanded_target = Path(target).expanduser()
        if expanded_target.exists():
            target = str(expanded_target)
        self._unload_hf()
        if self.server_proc:
            self._kill_proc(self.server_proc)
            self.server_proc = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except Exception:
                AutoModelForImageTextToText = None
                AutoProcessor = None
        except Exception as exc:
            self._log(
                "[ERROR] Hugging Face backend is not installed. "
                "Install it with: pip install -e \".[hf]\""
            )
            self._log(f"[ERROR] Import failure: {exc}")
            return False

        is_vision = self._hf_target_is_vision(target)
        common_kwargs = self._hf_common_pretrained_kwargs(target)
        load_kwargs = self._hf_load_kwargs(target)

        try:
            if is_vision:
                if AutoModelForImageTextToText is None or AutoProcessor is None:
                    self._log("[ERROR] Installed transformers version does not expose AutoModelForImageTextToText")
                    return False
                self._hf_processor = AutoProcessor.from_pretrained(
                    target,
                    trust_remote_code=bool(APP_CONFIG.get("hf_trust_remote_code", True)),
                    **common_kwargs,
                )
                self._hf_tokenizer = getattr(self._hf_processor, "tokenizer", None)
                self._hf_model = AutoModelForImageTextToText.from_pretrained(
                    target, trust_remote_code=bool(APP_CONFIG.get("hf_trust_remote_code", True)), **load_kwargs
                )
                self._hf_is_vision = True
            else:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(
                    target,
                    trust_remote_code=bool(APP_CONFIG.get("hf_trust_remote_code", True)),
                    **common_kwargs,
                )
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    target, trust_remote_code=bool(APP_CONFIG.get("hf_trust_remote_code", True)), **load_kwargs
                )
                self._hf_processor = None
                self._hf_is_vision = False
            try:
                self._hf_model.eval()
            except Exception:
                pass
        except Exception as exc:
            self._unload_hf()
            repo_id = "" if Path(str(target)).expanduser().exists() else str(target)
            self._log(f"[ERROR] Could not load HF Transformers model '{target}': {normalize_hf_exception(exc, repo_id=repo_id)}")
            self._log(f"[ERROR] HF settings used: {self._hf_settings_summary()}")
            self._log("[ERROR] Check Settings → App Configuration → HF Transformers for dtype/device/safetensors/quantization options.")
            return False

        self.mode = "hf_transformers"
        self.model_path = ref
        self._log(
            f"[INFO] HF Transformers model ready: {model_ref_display_name(ref)}"
            + (" [vision]" if self._hf_is_vision else "")
        )
        return True

    def _unload_hf(self):
        self._hf_model = None
        self._hf_tokenizer = None
        self._hf_processor = None
        self._hf_is_vision = False
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _hf_target_is_vision(self, target: str) -> bool:
        vi = detect_vision_model(target)
        if vi.is_vision:
            return True
        p = Path(target)
        cfg_path = p / "config.json" if p.is_dir() else None
        if cfg_path and cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                text = json.dumps(cfg).lower()
                return any(k in text for k in (
                    "vision_config", "vision_model", "image_token_index",
                    "llava", "qwen2_vl", "qwen2_5_vl", "mllama",
                    "paligemma", "pixtral", "idefics", "internvl",
                ))
            except Exception:
                return False
        return False

    def _hf_common_pretrained_kwargs(self, target: str = "") -> dict:
        kwargs = {}
        is_local = bool(target and Path(target).expanduser().exists())
        token = get_hf_access_token()
        revision = str(APP_CONFIG.get("hf_revision", "main") or "").strip()
        cache_dir = str(APP_CONFIG.get("hf_transformers_dir", "") or "").strip()
        if token and not is_local:
            kwargs["token"] = token
        if revision and not is_local:
            kwargs["revision"] = revision
        if cache_dir and not is_local:
            kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
        kwargs["local_files_only"] = bool(APP_CONFIG.get("hf_local_files_only", False))
        return kwargs

    def _parse_hf_max_memory(self) -> dict:
        raw = str(APP_CONFIG.get("hf_max_memory", "") or "").strip()
        if not raw:
            return {}
        out = {}
        for part in raw.split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                out[key] = value
        return out

    def _hf_settings_summary(self) -> str:
        return (
            f"revision={APP_CONFIG.get('hf_revision', 'main')}, "
            f"local_files_only={bool(APP_CONFIG.get('hf_local_files_only', False))}, "
            f"use_safetensors={APP_CONFIG.get('hf_use_safetensors', 'auto')}, "
            f"torch_dtype={APP_CONFIG.get('hf_torch_dtype', 'auto')}, "
            f"device_map={APP_CONFIG.get('hf_device_map', 'auto')}, "
            f"low_cpu_mem_usage={bool(APP_CONFIG.get('hf_low_cpu_mem_usage', True))}, "
            f"attn_implementation={APP_CONFIG.get('hf_attn_implementation', '') or '<default>'}, "
            f"max_memory={APP_CONFIG.get('hf_max_memory', '') or '<default>'}, "
            f"quantization={APP_CONFIG.get('hf_quantization', 'none')}"
        )

    def _hf_load_kwargs(self, target: str = "") -> dict:
        common = self._hf_common_pretrained_kwargs(target)
        kwargs = dict(common)
        dtype = str(APP_CONFIG.get("hf_torch_dtype", "auto") or "auto").strip()
        if dtype:
            if dtype == "auto":
                kwargs["torch_dtype"] = "auto"
            else:
                try:
                    import torch
                    kwargs["torch_dtype"] = {
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "float32": torch.float32,
                    }.get(dtype, "auto")
                except Exception:
                    kwargs["torch_dtype"] = dtype

        device_map = str(APP_CONFIG.get("hf_device_map", "auto") or "auto").strip()
        if device_map and device_map.lower() != "none":
            try:
                import accelerate  # noqa: F401
                kwargs["device_map"] = device_map
            except Exception:
                self._log("[WARN] hf_device_map ignored because accelerate is not installed.")

        use_safetensors = str(APP_CONFIG.get("hf_use_safetensors", "auto") or "auto").lower()
        if use_safetensors == "true":
            kwargs["use_safetensors"] = True
        elif use_safetensors == "false":
            kwargs["use_safetensors"] = False

        if bool(APP_CONFIG.get("hf_low_cpu_mem_usage", True)):
            kwargs["low_cpu_mem_usage"] = True
        attn_impl = str(APP_CONFIG.get("hf_attn_implementation", "") or "").strip()
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        max_memory = self._parse_hf_max_memory()
        if max_memory:
            kwargs["max_memory"] = max_memory

        quant = str(APP_CONFIG.get("hf_quantization", "none") or "none").lower()
        if quant in ("8bit", "4bit"):
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:
                raise RuntimeError(
                    f"HF quantization '{quant}' requires bitsandbytes/Transformers quantization support. "
                    "Set HF Quantization to 'none' or install bitsandbytes."
                ) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=(quant == "8bit"),
                load_in_4bit=(quant == "4bit"),
            )
        return kwargs

    def _build_text_prompt(self, messages) -> str:
        fam = detect_model_family(model_ref_payload(self.model_path) or self.model_path)
        out = []
        sys_buf = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
            if role == "system":
                sys_buf += str(content) + "\n"
            elif role == "user":
                u = (sys_buf + str(content)) if sys_buf else str(content)
                sys_buf = ""
                out.append(getattr(fam, "user_prefix", "") + u + getattr(fam, "user_suffix", ""))
            elif role == "assistant":
                out.append(getattr(fam, "assistant_prefix", "") + str(content) + getattr(fam, "assistant_suffix", ""))
        out.append(getattr(fam, "assistant_prefix", ""))
        return getattr(fam, "bos", "") + "".join(out)

    def _image_b64_list(self, image_data=None) -> list:
        out = []
        for item in image_data or []:
            if isinstance(item, dict):
                data = item.get("data") or ""
                if isinstance(data, str) and data:
                    out.append(data.split(";base64,", 1)[-1])
        return out

    def _raw_prompt_text(self, messages) -> str:
        if len(messages) == 1:
            content = messages[0].get("content", "")
            if isinstance(content, str):
                return content
        return self._build_text_prompt(messages)

    def _generate_server(self, messages, n_predict, temperature, top_p, repeat_penalty, token_cb=None, abort_cb=None, image_data=None, raw_prompt: bool = False) -> str:
        import http.client

        prompt_text = self._raw_prompt_text(messages) if raw_prompt else self._build_text_prompt(messages)
        fam = detect_model_family(self.model_path)
        body_obj = {
            "prompt": prompt_text,
            "n_predict": int(n_predict),
            "stream": bool(token_cb),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repeat_penalty": float(repeat_penalty),
            "stop": getattr(fam, "stop_tokens", []),
        }
        images = list(image_data or [])
        if images:
            body_obj["image_data"] = images
            markers = "\n".join(f"[img-{img.get('id', i + 1)}]" for i, img in enumerate(images) if isinstance(img, dict))
            body_obj["prompt"] = f"{markers}\n\n{prompt_text}"
        conn = http.client.HTTPConnection("127.0.0.1", self.server_port, timeout=LONG_TIMEOUT_SECONDS)
        conn.request("POST", "/completion", json.dumps(body_obj), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw_parts = []
        if resp.status != 200:
            raw = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {resp.status}: {raw}")
        if token_cb:
            buf = b""
            while not (abort_cb and abort_cb()):
                chunk = resp.read(64)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except Exception:
                        continue
                    if data.get("stop"):
                        return "".join(raw_parts).strip()
                    tok = data.get("content", "")
                    if tok:
                        raw_parts.append(tok)
                        token_cb(tok)
            return "".join(raw_parts).strip()
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return (data.get("content") or "").strip()

    def _generate_cli(self, messages, n_predict, temperature, repeat_penalty, token_cb=None, abort_cb=None, raw_prompt: bool = False) -> str:
        from nativelab.Server.server_global import SERVER_CONFIG

        prompt_text = self._raw_prompt_text(messages) if raw_prompt else self._build_text_prompt(messages)
        cli_bin = SERVER_CONFIG.cli_path or _binres.LLAMA_CLI
        cmd = [
            cli_bin, "-m", self.model_path,
            "-t", str(DEFAULT_THREADS()), "--ctx-size", str(self.ctx_value),
            "-n", str(int(n_predict)), "--no-display-prompt", "--no-escape",
            "--temp", str(temperature), "--repeat-penalty", str(repeat_penalty),
            "-p", prompt_text,
        ]
        if token_cb:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, bufsize=0
            )
            out = []
            try:
                while not (abort_cb and abort_cb()):
                    if proc.stdout is None:
                        break
                    b = proc.stdout.read(1)
                    if not b:
                        break
                    c = b.decode("utf-8", errors="replace")
                    out.append(c)
                    token_cb(c)
                proc.wait(timeout=LONG_TIMEOUT_SECONDS)
            finally:
                if proc.poll() is None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            return "".join(out).strip()
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, timeout=LONG_TIMEOUT_SECONDS,
        )
        return result.stdout.decode("utf-8", errors="replace").strip()

    def _generate_ollama(self, messages, n_predict, temperature, top_p, repeat_penalty, token_cb=None, abort_cb=None, image_data=None) -> str:
        import urllib.request
        import urllib.error

        model = model_ref_payload(self.model_path)
        chat = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            chat.append({"role": role if role in ("system", "user", "assistant") else "user", "content": str(content)})
        images = self._image_b64_list(image_data)
        if images:
            for msg in reversed(chat):
                if msg.get("role") == "user":
                    msg["images"] = images
                    break
            else:
                chat.append({"role": "user", "content": "Please inspect the attached image(s).", "images": images})
        payload = {
            "model": model,
            "messages": chat or [{"role": "user", "content": ""}],
            "stream": bool(token_cb),
            "keep_alive": str(APP_CONFIG.get("ollama_keep_alive", "5m") or "5m"),
            "options": {
                "num_ctx": int(self.ctx_value),
                "num_predict": int(n_predict),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repeat_penalty": float(repeat_penalty),
            },
        }
        req = urllib.request.Request(
            f"{self.ollama_host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=LONG_TIMEOUT_SECONDS) as resp:
                if token_cb:
                    out = []
                    for raw_line in resp:
                        if abort_cb and abort_cb():
                            break
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        tok = ((data.get("message") or {}).get("content") or "")
                        if tok:
                            out.append(tok)
                            token_cb(tok)
                        if data.get("done"):
                            break
                    return "".join(out).strip()
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            raise RuntimeError(normalize_ollama_exception(exc, self.ollama_host, action="chat with")) from exc
        return (((data.get("message") or {}).get("content")) or data.get("response") or "").strip()

    def _hf_prompt_text(self, messages) -> str:
        tokenizer = self._hf_tokenizer
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                chat = []
                for msg in messages:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            str(part.get("text", "")) if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    chat.append({"role": msg.get("role", "user"), "content": str(content)})
                return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return self._build_text_prompt(messages)

    def _hf_inputs_to_device(self, inputs):
        try:
            device = next(self._hf_model.parameters()).device
            return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        except Exception:
            return inputs

    def _generate_hf(self, messages, n_predict, temperature, top_p, token_cb=None, abort_cb=None, image_data=None, raw_prompt: bool = False) -> str:
        if self._hf_model is None:
            raise RuntimeError("HF Transformers model is not loaded")
        try:
            import base64
            import threading
            from io import BytesIO
            from PIL import Image
            from transformers import TextIteratorStreamer
        except Exception as exc:
            raise RuntimeError(f"HF generation dependencies missing: {exc}") from exc

        tokenizer = self._hf_tokenizer or getattr(self._hf_processor, "tokenizer", None)
        images = []
        for data in self._image_b64_list(image_data):
            try:
                images.append(Image.open(BytesIO(base64.b64decode(data))).convert("RGB"))
            except Exception:
                pass

        if self._hf_is_vision:
            if not images:
                raise RuntimeError("This HF vision model needs at least one image input")
            processor = self._hf_processor
            text = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") != "assistant")
            vision_messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": text}],
            }]
            try:
                inputs = processor.apply_chat_template(
                    vision_messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
            except Exception:
                prompt_text = processor.apply_chat_template(
                    vision_messages, add_generation_prompt=True, tokenize=False,
                )
                inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
        else:
            prompt_text = self._raw_prompt_text(messages) if raw_prompt else self._hf_prompt_text(messages)
            inputs = tokenizer(prompt_text, return_tensors="pt")

        inputs = self._hf_inputs_to_device(inputs)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": int(n_predict),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "do_sample": float(temperature) > 0.0,
        }

        if token_cb and tokenizer is not None:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            error = []

            def _run_generate():
                try:
                    self._hf_model.generate(**gen_kwargs)
                except Exception as exc:
                    error.append(exc)

            thread = threading.Thread(target=_run_generate, daemon=True)
            thread.start()
            out = []
            for chunk in streamer:
                if abort_cb and abort_cb():
                    break
                if chunk:
                    out.append(chunk)
                    token_cb(chunk)
            thread.join(timeout=1)
            if error:
                raise error[0]
            return "".join(out).strip()

        output = self._hf_model.generate(**gen_kwargs)
        input_len = int(inputs.get("input_ids").shape[-1]) if "input_ids" in inputs else 0
        generated = output[0][input_len:] if input_len else output[0]
        if tokenizer is None:
            return str(generated)
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

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
            proc.wait(timeout=LONG_TIMEOUT_SECONDS)
        except Exception:
            pass

    def _check_existing_server(self, port: int, timeout: float = LONG_TIMEOUT_SECONDS) -> bool:
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
                conn = http.client.HTTPConnection("127.0.0.1", test_port, timeout=LONG_TIMEOUT_SECONDS)
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
                self._log(f"[INFO] Port {test_port} has a different model - skipping")
            except Exception:
                pass

        # ── 2. Kill our own stale server proc if still alive ──
        if self.server_proc and self.server_proc.poll() is None:
            self._kill_proc(self.server_proc)
            self.server_proc = None

        # ── 3. Launch a fresh server ──
        self.server_port = free_port()
        _extra_srv = SERVER_CONFIG.extra_server_args.split() if SERVER_CONFIG.extra_server_args else []
        vi = detect_vision_model(model_path)
        if vi.is_vision and "--mmproj" not in _extra_srv:
            mmproj = detect_mmproj_for_model(model_path)
            if mmproj:
                _extra_srv = ["--mmproj", mmproj] + _extra_srv
                self._log(f"[INFO] VLM detected ({vi.label}) - using mmproj: {Path(mmproj).name}")
            elif vi.needs_mmproj:
                self._log(
                    f"[WARN] VLM detected ({vi.label}) but no mmproj/projector GGUF "
                    f"was found next to the model. Add --mmproj <path> in server extra flags.")
        _server_bin = SERVER_CONFIG.server_path or _binres.LLAMA_SERVER
        cmd = [
            _server_bin, "-m", model_path,
            "-t", str(threads), "--ctx-size", str(ctx),
            "--port", str(self.server_port),
            "--host", SERVER_CONFIG.host or "127.0.0.1",
        ] + _extra_srv

        self._log(f"[INFO] Starting llama-server on port {self.server_port}…")

        # Log to a temp file instead of DEVNULL so failures are diagnosable
        import tempfile
        log_path = Path(tempfile.gettempdir()) / f"_binres.LLAMA_SERVER_{self.server_port}.log"
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
        max_wait   = int(APP_CONFIG.get("server_startup_timeout", LONG_TIMEOUT_SECONDS))  # seconds, configurable
        poll_every = 0.5
        polls      = int(max_wait / poll_every)

        for _ in range(polls):
            time.sleep(poll_every)

            if self.server_proc is None:
                return False

            if self.server_proc.poll() is not None:
                self._log(
                    f"[ERROR] llama-server exited unexpectedly - "
                    f"check {log_path} for details"
                )
                self.server_proc = None
                return False

            if self._check_existing_server(self.server_port):
                self._log(f"[INFO] llama-server ready on port {self.server_port}")
                self.mode = "server"
                return True

        self._log(
            f"[ERROR] Server did not respond within {max_wait}s - "
            f"check {log_path} for details"
        )
        try:
            self.server_proc.terminate()
        except Exception:
            pass
        self.server_proc = None
        return False
