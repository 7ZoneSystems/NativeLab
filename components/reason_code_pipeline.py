from imports.import_global import QThread, pyqtSignal, subprocess, time, json
from Model.model_global import detect_model_family
from GlobalConfig.config_global import LLAMA_CLI, DEFAULT_THREADS, DEFAULT_CTX
class PipelineWorker(QThread):
    """
    Multi-engine structural insight → coding pipeline.

    Stage 1 (parallel):  Every active non-coding engine is asked to give a
                         *structural-level insight* about the user's coding
                         request — no code, just intent, architecture and design.
                         Each engine runs sequentially (llama.cpp single-process
                         constraint) but their outputs are collected and labelled.

    Stage 2 (coding):    The coding engine receives ALL structural insights from
                         stage 1 as rich context and generates the final code.
    """

    # per-engine insight signals
    insight_started  = pyqtSignal(int, str)   # (engine_idx, engine_label)
    insight_token    = pyqtSignal(int, str)   # (engine_idx, token)
    insight_done     = pyqtSignal(int, str)   # (engine_idx, full_text)

    # coding stage signals
    coding_token     = pyqtSignal(str)
    coding_done      = pyqtSignal(float)      # tok/s

    err              = pyqtSignal(str)
    stage_changed    = pyqtSignal(str)        # "insights" | "coding"

    # keep old signal names alive so existing connections don't break
    reasoning_token  = pyqtSignal(str)
    reasoning_done   = pyqtSignal(str)

    INSIGHT_PROMPT_TEMPLATE = (
        "You are a senior software architect reviewing a coding request.\n"
        "Your task: provide STRUCTURAL INSIGHT ONLY — no code, no pseudocode.\n\n"
        "Describe:\n"
        "1. High-level purpose and user intent\n"
        "2. Recommended architecture / design pattern (e.g. class hierarchy, "
        "pipeline, event-driven, functional)\n"
        "3. Key components / modules and their responsibilities\n"
        "4. Data flow between components\n"
        "5. Important algorithms or data structures to use\n"
        "6. Critical edge cases and error-handling considerations\n"
        "7. Suggested external libraries / imports\n\n"
        "Be thorough but concise. Structure your answer with short headers. "
        "Do NOT write any actual code.\n\n"
        "Coding request: {prompt}"
    )

    def __init__(self,
                 insight_engines,
                 coding_eng,
                 prompt: str,
                 n_predict_insight: int = 512,
                 n_predict_code: int = 1024):
        """
        insight_engines : list of (label, engine) for structural-insight stage
        coding_eng      : engine that will generate the final code
        """
        super().__init__()
        self.insight_engines    = insight_engines   # [(label, eng), ...]
        self.coding_eng         = coding_eng
        self.prompt             = prompt
        self.n_predict_insight  = n_predict_insight
        self.n_predict_code     = n_predict_code
        self._abort             = False
        self._insights          = []  # [(label, text), ...]

    def abort(self):
        self._abort = True

    def run(self):
        # Stage 1: structural insights from all non-coding engines
        self.stage_changed.emit("insights")

        for idx, (label, eng) in enumerate(self.insight_engines):
            if self._abort:
                self.err.emit("Aborted before all insights completed")
                return

            self.insight_started.emit(idx, label)

            fam = detect_model_family(getattr(eng, "model_path", ""))
            insight_prompt = (
                fam.bos +
                fam.user_prefix +
                self.INSIGHT_PROMPT_TEMPLATE.format(prompt=self.prompt) +
                fam.user_suffix +
                fam.assistant_prefix
            )

            text = self._infer_blocking(
                eng, insight_prompt,
                self.n_predict_insight,
                token_cb=lambda t, i=idx: self.insight_token.emit(i, t)
            )

            if self._abort:
                return
            if text is None:
                self.err.emit(f"Insight stage failed for engine: {label}")
                return

            cleaned = text.strip()
            self._insights.append((label, cleaned))
            self.insight_done.emit(idx, cleaned)

            # bridge old reasoning_done signal for single-engine compat
            if idx == 0:
                self.reasoning_done.emit(cleaned)

        if not self._insights:
            self.err.emit("No structural insights were produced")
            return

        # Stage 2: coding engine receives all insights
        self.stage_changed.emit("coding")

        insights_block = ""
        for label, text in self._insights:
            bar = "=" * max(4, len(label) + 4)
            insights_block += (
                f"[ {label} ]\n"
                f"{text}\n"
                f"{bar}\n\n"
            )

        code_fam = detect_model_family(getattr(self.coding_eng, "model_path", ""))
        sep = "=" * 60
        code_prompt = (
            code_fam.bos +
            code_fam.user_prefix +
            f"You are an expert code generation assistant.\n\n"
            f"{sep}\n"
            f"STRUCTURAL INSIGHTS FROM ANALYSIS MODELS\n"
            f"{sep}\n"
            f"{insights_block}"
            f"{sep}\n"
            f"ORIGINAL REQUEST\n"
            f"{sep}\n"
            f"{self.prompt}\n\n"
            f"Using the structural insights above as your blueprint, generate "
            f"complete, working, well-commented code. Follow the architecture "
            f"and design patterns recommended in the insights. Include all "
            f"necessary imports. Add docstrings and inline comments where useful. "
            f"Handle the edge cases mentioned in the insights." +
            code_fam.user_suffix +
            code_fam.assistant_prefix
        )

        t0 = time.time()
        n_tokens = [0]
        self._infer_blocking(
            self.coding_eng, code_prompt,
            self.n_predict_code,
            token_cb=lambda t: (
                self.coding_token.emit(t),
                n_tokens.__setitem__(0, n_tokens[0] + 1)
            )
        )
        elapsed = time.time() - t0
        tps = n_tokens[0] / elapsed if elapsed > 0 else 0.0
        self.coding_done.emit(tps)

    def _infer_blocking(self, eng, prompt: str,
                        n_predict: int, token_cb=None):
        """Blocking inference that calls token_cb for each token."""
        if eng.mode == "server":
            return self._infer_server(eng, prompt, n_predict, token_cb)
        return self._infer_cli(eng, prompt, n_predict, token_cb)

    def _infer_server(self, eng, prompt: str,
                      n_predict: int, token_cb=None):
        import http.client
        fam = detect_model_family(getattr(eng, "model_path", ""))
        try:
            conn = http.client.HTTPConnection("127.0.0.1", eng.server_port, timeout=600)
            body = json.dumps({
                "prompt":         prompt,
                "n_predict":      n_predict,
                "stream":         True,
                "temperature":    0.7,
                "top_p":          0.9,
                "repeat_penalty": 1.1,
                "stop":           fam.stop_tokens,
            })
            conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200:
                return None
            result = []
            buf = b""
            while not self._abort:
                b = r.read(1)
                if not b:
                    break
                buf += b
                if b == b"\n":
                    line = buf.decode("utf-8", errors="replace").strip()
                    buf  = b""
                    if line.startswith("data: "):
                        try:
                            d = json.loads(line[6:])
                            if d.get("stop"):
                                break
                            c = d.get("content", "")
                            if c:
                                result.append(c)
                                if token_cb:
                                    token_cb(c)
                        except json.JSONDecodeError:
                            pass
            return "".join(result)
        except Exception:
            return None

    def _infer_cli(self, eng, prompt: str,
                   n_predict: int, token_cb=None):
        try:
            proc = subprocess.Popen(
                [LLAMA_CLI, "-m", eng.model_path,
                 "-t", str(DEFAULT_THREADS),
                 "--ctx-size", str(getattr(eng, "ctx_value", DEFAULT_CTX)),
                 "-n", str(n_predict),
                 "--no-display-prompt", "--no-escape",
                 "-p", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                bufsize=0
            )
            result = []
            while not self._abort:
                b = proc.stdout.read(1)
                if not b:
                    break
                c = b.decode("utf-8", errors="replace")
                result.append(c)
                if token_cb:
                    token_cb(c)
            proc.terminate()
            return "".join(result)
        except Exception:
            return None