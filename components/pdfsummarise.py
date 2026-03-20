from imports.import_global import List, Optional, QThread, pyqtSignal, subprocess, time, datetime, json
from components.components_global import detect_model_family, load_paused_job, save_paused_job, delete_paused_job
from Model.model_global import MODE_SECTION_INSTRUCTIONS, MODE_FINAL_INSTRUCTIONS
from GlobalConfig.config_global import LLAMA_CLI, DEFAULT_CTX, DEFAULT_THREADS, APP_CONFIG, simple_hash

class ChunkedSummaryWorker(QThread):
    section_done  = pyqtSignal(int, int, str, str)
    final_done    = pyqtSignal(str)
    progress      = pyqtSignal(str)
    err           = pyqtSignal(str)
    pause_suggest = pyqtSignal(str)

    def __init__(self, engine, text: str, filename: str = "",
                 engine2=None, resume_job_id: str = "",
                 session_id: str = "", summary_mode: str = "summary"):
        super().__init__()
        self.engine        = engine
        self.engine2       = engine2
        self.text          = text
        self.filename      = filename
        self.resume_job_id = resume_job_id
        self.session_id    = session_id
        self.summary_mode  = summary_mode
        self._abort        = False
        self._pause        = False
        self._pending_txt  = ""
        self.job_id        = resume_job_id or f"sum_{simple_hash(filename + text[:32])}_{int(time.time())}"

    def abort(self):
        self._abort = True

    def request_pause(self):
        self._pause = True

    def run(self):
        cfg = APP_CONFIG
        CHUNK_CHARS  = int(cfg["summary_chunk_chars"])
        CTX_CARRY    = int(cfg["summary_ctx_carry"])
        N_PRED_SECT  = int(cfg["summary_n_pred_sect"])
        N_PRED_FINAL = int(cfg["summary_n_pred_final"])
        PAUSE_THRESH = int(cfg["pause_after_chunks"])

        mode = getattr(self, "summary_mode", "summary")
        sect_instruction  = MODE_SECTION_INSTRUCTIONS.get(mode, MODE_SECTION_INSTRUCTIONS["summary"])
        final_instruction = MODE_FINAL_INSTRUCTIONS.get(mode, MODE_FINAL_INSTRUCTIONS["summary"])

        chunks    = self._split(self.text, CHUNK_CHARS)
        total     = len(chunks)
        start_idx = 0
        section_summaries: List[str] = []
        running_ctx = ""

        if self.resume_job_id:
            state = load_paused_job(self.resume_job_id)
            if state:
                start_idx         = state.get("next_chunk", 0)
                section_summaries = state.get("summaries", [])
                running_ctx       = state.get("running_ctx", "")
                self.progress.emit(f"Resuming from chunk {start_idx + 1} / {total}…")

        fam = detect_model_family(getattr(self.engine, "model_path", ""))

        for i in range(start_idx, total):
            if self._abort:
                self.err.emit("Aborted by user."); return

            if self._pause:
                state = {
                    "job_id": self.job_id, "filename": self.filename,
                    "session_id": self.session_id, "total": total,
                    "next_chunk": i, "summaries": section_summaries,
                    "running_ctx": running_ctx, "raw_text": self.text,
                    "model_path": getattr(self.engine, "model_path", ""),
                    "paused_at": datetime.now().isoformat(), "summary_mode": mode,
                }
                save_paused_job(self.job_id, state)
                self.progress.emit(f"⏸  Paused after chunk {i} / {total}. State saved to disk.")
                self.err.emit(f"__PAUSED__:{self.job_id}")
                return

            if i > 0 and i == PAUSE_THRESH and (total - i) > PAUSE_THRESH:
                self.pause_suggest.emit(self.job_id)

            chunk = chunks[i]
            self.progress.emit(f"Summarising section {i+1} / {total}…")
            ctx_block = (f"Running context from previous sections:\n{running_ctx}\n\n" if running_ctx else "")
            prompt = (
                fam.bos + fam.user_prefix +
                f"You are analysing a long document section by section.\n"
                f"File: '{self.filename}'  |  Section {i+1} of {total}\n"
                f"Mode: {mode.upper()}\n\n"
                f"{ctx_block}"
                f"Current section:\n{chunk}\n\n"
                f"{sect_instruction}" +
                fam.user_suffix + fam.assistant_prefix
            )
            summary = self._infer(prompt, N_PRED_SECT)
            if summary is None:
                self.err.emit(f"Inference failed on section {i+1}"); return
            summary = summary.strip()
            section_summaries.append(f"[Section {i+1} of {total}]\n{summary}")
            running_ctx = summary[-CTX_CARRY:]
            self.section_done.emit(i + 1, total, chunk, summary)

            if (i + 1) % 3 == 0:
                save_paused_job(self.job_id + "_autosave", {
                    "job_id": self.job_id, "filename": self.filename,
                    "session_id": self.session_id, "total": total,
                    "next_chunk": i + 1, "summaries": section_summaries,
                    "running_ctx": running_ctx, "raw_text": self.text,
                    "model_path": getattr(self.engine, "model_path", ""),
                    "paused_at": datetime.now().isoformat(), "summary_mode": mode,
                })

        if self._abort: return

        fin_eng   = self.engine2 if (self.engine2 and self.engine2.is_loaded) else self.engine
        fin_label = "reasoning model" if fin_eng is self.engine2 else "primary model"
        self.progress.emit(f"Running final consolidation pass ({fin_label})…")

        fin_fam   = detect_model_family(getattr(fin_eng, "model_path", ""))
        all_sects = "\n\n".join(section_summaries)
        final_prompt = (
            fin_fam.bos + fin_fam.user_prefix +
            f"You have finished reading all {total} sections of '{self.filename}'.\n"
            f"Mode: {mode.upper()}\n\n"
            f"Section-by-section notes:\n{all_sects}\n\n"
            f"{final_instruction}" +
            fin_fam.user_suffix + fin_fam.assistant_prefix
        )

        final = self._infer_with(fin_eng, final_prompt, N_PRED_FINAL)
        if final is None:
            self.progress.emit("⚠️ Final pass failed on secondary engine — retrying with primary…")
            final = self._infer_with(self.engine, final_prompt, N_PRED_FINAL)
        if final is None:
            self.progress.emit("⚠️ Final pass failed — using section summaries as fallback.")
            final = f"[Auto-fallback — final consolidation failed]\n\n" + all_sects

        delete_paused_job(self.job_id + "_autosave")
        delete_paused_job(self.job_id)
        self.final_done.emit(final.strip())

    def _split(self, text: str, chunk_chars: int) -> List[str]:
        chunks: List[str] = []
        while text:
            if len(text) <= chunk_chars:
                chunks.append(text.strip()); break
            cut = text.rfind("\n\n", 0, chunk_chars)
            if cut < 200:
                cut = text.rfind("\n", 0, chunk_chars)
            if cut < 200:
                cut = chunk_chars
            chunks.append(text[:cut].strip())
            text = text[cut:].lstrip()
        return [c for c in chunks if c]

    def _infer(self, prompt: str, n_predict: int) -> Optional[str]:
        return self._infer_with(self.engine, prompt, n_predict)

    def _infer_with(self, eng, prompt: str, n_predict: int) -> Optional[str]:
        if eng.mode == "server":
            return self._infer_server(prompt, n_predict, eng.server_port)
        return self.infer_cli(prompt, n_predict, eng.model_path,
                              getattr(eng, "ctx_value", DEFAULT_CTX))

    def _infer_server(self, prompt: str, n_predict: int, port: int = 0) -> Optional[str]:
        import http.client
        fam = detect_model_family(getattr(self.engine, "model_path", ""))
        if not port: port = self.engine.server_port
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
            body = json.dumps({
                "prompt": prompt, "n_predict": n_predict,
                "stream": False, "temperature": 0.3, "top_p": 0.9,
                "repeat_penalty": 1.15, "stop": fam.stop_tokens,
            })
            conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200: return None
            d = json.loads(r.read().decode("utf-8", errors="replace"))
            return d.get("content", "")
        except Exception:
            return None

    def infer_cli(self, prompt: str, n_predict: int,
                  model_path: str = "", ctx: int = 0) -> Optional[str]:
        if not ctx: ctx = DEFAULT_CTX
        if not model_path: model_path = self.engine.model_path
        try:
            result = subprocess.run(
                [LLAMA_CLI, "-m", model_path, "-t", str(DEFAULT_THREADS),
                 "--ctx-size", str(ctx), "-n", str(n_predict),
                 "--no-display-prompt", "--no-escape",
                 "--temp", "0.3", "--repeat-penalty", "1.15", "-p", prompt],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, timeout=300,
            )
            return result.stdout.decode("utf-8", errors="replace")
        except Exception:
            return None