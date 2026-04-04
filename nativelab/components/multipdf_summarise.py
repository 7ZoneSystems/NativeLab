from imports.import_global import List, Tuple, Dict, Optional, Path, QThread, pyqtSignal, subprocess, time, datetime, json
from GlobalConfig.config_global import simple_hash, APP_CONFIG, REF_CACHE_DIR, LLAMA_CLI, DEFAULT_THREADS, DEFAULT_CTX
from GlobalConfig.hardwareUtil import RamWatchdog, get_ref_store
from Model.model_global import detect_model_family
from .jobhandler import save_paused_job, load_paused_job, delete_paused_job

class MultiPdfSummaryWorker(QThread):
    file_started  = pyqtSignal(int, str, int)
    file_progress = pyqtSignal(int, str)
    file_done     = pyqtSignal(int, str)
    section_done  = pyqtSignal(int, int, str, str)
    ram_warning   = pyqtSignal(str)
    final_done    = pyqtSignal(str)
    progress      = pyqtSignal(str)
    err           = pyqtSignal(str)
    pause_suggest = pyqtSignal(str)

    def __init__(self, engine, pdf_texts: List[Tuple[str, str]],
                 session_id: str, engine2=None, resume_job_id: str = ""):
        super().__init__()
        self.engine        = engine
        self.engine2       = engine2
        self.pdf_texts     = pdf_texts
        self.session_id    = session_id
        self.resume_job_id = resume_job_id
        self._abort        = False
        self._pause        = False
        self._disk_summaries: Dict[str, Path] = {}
        names_key = "_".join(Path(fn).stem for fn, _ in pdf_texts)[:40]
        self.job_id = resume_job_id or f"mpdf_{simple_hash(names_key)}_{int(time.time())}"

    def abort(self):
        self._abort = True

    def request_pause(self):
        self._pause = True

    def save_state(self, next_fi: int, next_ci: int,
                   file_summaries_so_far: List[str],
                   running_ctx: str,
                   completed_file_summaries: List[Tuple[str, str]]):
        disk_paths = {fn: str(p) for fn, p in self._disk_summaries.items()}
        state = {
            "job_id":                self.job_id,
            "session_id":            self.session_id,
            "filename":              f"{len(self.pdf_texts)} PDFs",
            "total":                 len(self.pdf_texts),
            "next_fi":               next_fi,
            "next_ci":               next_ci,
            "next_chunk":            next_ci,
            "file_summaries_so_far": file_summaries_so_far,
            "running_ctx":           running_ctx,
            "completed_files":       completed_file_summaries,
            "disk_summaries":        disk_paths,
            "pdf_texts":             [[fn, txt] for fn, txt in self.pdf_texts],
            "model_path":            getattr(self.engine, "model_path", ""),
            "paused_at":             datetime.now().isoformat(),
        }
        save_paused_job(self.job_id, state)

    def run(self):
        cfg          = APP_CONFIG
        CHUNK_CHARS  = int(cfg["summary_chunk_chars"])
        CTX_CARRY    = int(cfg["summary_ctx_carry"])
        N_PRED_SECT  = int(cfg["multipdf_n_pred_sect"])
        N_PRED_FINAL = int(cfg["multipdf_n_pred_final"])
        PAUSE_THRESH = int(cfg["pause_after_chunks"])

        start_fi   = 0
        start_ci   = 0
        completed_file_summaries: List[Tuple[str, str]] = []
        file_summaries_so_far: List[str] = []
        running_ctx = ""

        if self.resume_job_id:
            state = load_paused_job(self.resume_job_id)
            if state:
                start_fi  = state.get("next_fi", 0)
                start_ci  = state.get("next_ci", 0)
                completed_file_summaries = [tuple(x) for x in state.get("completed_files", [])]
                file_summaries_so_far    = state.get("file_summaries_so_far", [])
                running_ctx              = state.get("running_ctx", "")
                for fn, sp in state.get("disk_summaries", {}).items():
                    p = Path(sp)
                    if p.exists():
                        self._disk_summaries[fn] = p
                self.progress.emit(f"Resuming multi-PDF job from file {start_fi+1}, chunk {start_ci+1}…")

        total_chunks_done = 0

        for fi, (filename, text) in enumerate(self.pdf_texts):
            if fi < start_fi:
                continue
            if self._abort:
                self.err.emit("Aborted"); return

            chunks   = self._split(text, CHUNK_CHARS)
            n_chunks = len(chunks)
            self.file_started.emit(fi, filename, n_chunks)
            self.progress.emit(f"Processing '{filename}' — {n_chunks} chunks…")

            spilled = RamWatchdog.check_and_spill(self.session_id)
            if spilled:
                self.ram_warning.emit(f"⚠️ Low RAM before '{filename}' — spilled cache to disk.")

            ci_start = start_ci if fi == start_fi else 0
            if fi > start_fi:
                file_summaries_so_far = []
            fam = detect_model_family(getattr(self.engine, "model_path", ""))

            for i in range(ci_start, n_chunks):
                if self._abort:
                    return
                if self._pause:
                    self.save_state(fi, i, file_summaries_so_far, running_ctx, list(completed_file_summaries))
                    self.progress.emit(f"⏸  Paused at file {fi+1}, chunk {i+1}. State saved.")
                    self.err.emit(f"__PAUSED__:{self.job_id}")
                    return

                total_chunks_done += 1
                if total_chunks_done == PAUSE_THRESH and PAUSE_THRESH > 0:
                    remaining = sum(
                        len(self._split(t, CHUNK_CHARS)) - (i+1 if fidx == fi else 0)
                        for fidx, (_, t) in enumerate(self.pdf_texts) if fidx >= fi
                    )
                    if remaining > PAUSE_THRESH:
                        self.pause_suggest.emit(self.job_id)

                if i % 5 == 0:
                    spilled = RamWatchdog.check_and_spill(self.session_id)
                    if spilled:
                        self.ram_warning.emit(f"⚠️ RAM low at chunk {i+1} of '{filename}'.")

                self.file_progress.emit(fi, f"Chunk {i+1}/{n_chunks}")
                ctx_block = (f"Context from previous chunks:\n{running_ctx}\n\n" if running_ctx else "")
                prompt = (
                    fam.bos + fam.user_prefix +
                    f"Summarise this document chunk. File: '{filename}' | "
                    f"Chunk {i+1}/{n_chunks}\n\n{ctx_block}"
                    f"Chunk:\n{chunks[i]}\n\n"
                    f"Summarise clearly, keep all key facts and data." +
                    fam.user_suffix + fam.assistant_prefix
                )
                summary = self._infer(prompt, N_PRED_SECT)
                if summary is None:
                    self.err.emit(f"Inference failed at chunk {i+1} of '{filename}'"); return
                summary = summary.strip()
                file_summaries_so_far.append(f"[Chunk {i+1}/{n_chunks}]\n{summary}")
                running_ctx = summary[-CTX_CARRY:]
                self.section_done.emit(i + 1, n_chunks, chunks[i], summary)

                if (i + 1) % 3 == 0:
                    self.save_state(fi, i + 1, file_summaries_so_far, running_ctx, list(completed_file_summaries))

            all_chunks_text = "\n\n".join(file_summaries_so_far)
            consolidate_prompt = (
                fam.bos + fam.user_prefix +
                f"You have summarised all {n_chunks} chunks of '{filename}'.\n\n"
                f"Chunk summaries:\n{all_chunks_text}\n\n"
                f"Write a single coherent summary of this entire document." +
                fam.user_suffix + fam.assistant_prefix
            )
            file_summary = self._infer(consolidate_prompt, N_PRED_FINAL)
            if file_summary is None:
                file_summary = "\n".join(file_summaries_so_far)
            file_summary = file_summary.strip()

            disk_path = REF_CACHE_DIR / f"mpdf_{simple_hash(filename)}_{fi}.txt"
            disk_path.write_text(file_summary, encoding="utf-8")
            self._disk_summaries[filename] = disk_path
            completed_file_summaries.append((filename, file_summary))
            self.file_done.emit(fi, file_summary)

            file_summaries_so_far = []
            running_ctx = ""
            start_ci = 0

        if self._abort:
            return

        self.progress.emit("♻️  Reactive reload before final consolidation…")
        get_ref_store(self.session_id).reactive_reload(" ".join(fn for fn, _ in self.pdf_texts))

        self.progress.emit("📝  Final cross-document consolidation…")
        fin_eng = self.engine2 if (self.engine2 and self.engine2.is_loaded) else self.engine
        fin_fam = detect_model_family(getattr(fin_eng, "model_path", ""))

        all_summaries_text = ""
        for filename, disk_path in self._disk_summaries.items():
            try:
                txt = disk_path.read_text(encoding="utf-8")
            except Exception:
                txt = dict(completed_file_summaries).get(filename, "")
            all_summaries_text += f"\n\n=== {filename} ===\n{txt}"

        final_prompt = (
            fin_fam.bos + fin_fam.user_prefix +
            f"You have received summaries of {len(self.pdf_texts)} documents.\n\n"
            f"Per-document summaries:\n{all_summaries_text}\n\n"
            f"Write a final consolidated summary covering all documents, "
            f"including key themes, differences, and connections between them." +
            fin_fam.user_suffix + fin_fam.assistant_prefix
        )
        final = self._infer_with(fin_eng, final_prompt, N_PRED_FINAL + 400)
        if final is None:
            final = all_summaries_text
        self.final_done.emit(final.strip())

        delete_paused_job(self.job_id)
        for dp in self._disk_summaries.values():
            try: dp.unlink()
            except Exception: pass

    def _split(self, text: str, chunk_chars: int) -> List[str]:
        chunks = []
        while text:
            if len(text) <= chunk_chars:
                chunks.append(text.strip()); break
            cut = text.rfind("\n\n", 0, chunk_chars)
            if cut < 200: cut = text.rfind("\n", 0, chunk_chars)
            if cut < 200: cut = chunk_chars
            chunks.append(text[:cut].strip())
            text = text[cut:].lstrip()
        return [c for c in chunks if c]

    def _infer(self, prompt: str, n_predict: int) -> Optional[str]:
        return self._infer_with(self.engine, prompt, n_predict)

    def _infer_with(self, eng, prompt: str, n_predict: int) -> Optional[str]:
        if eng.mode == "server":
            return self._infer_server(eng, prompt, n_predict)
        return self.infer_cli(eng, prompt, n_predict)

    def _infer_server(self, eng, prompt: str, n_predict: int) -> Optional[str]:
        import http.client
        import threading
        fam = detect_model_family(getattr(eng, "model_path", ""))
        
        warned = False
        def _warn_if_slow():
            time.sleep(300)  # 5 minutes
            nonlocal warned
            warned = True
            self.progress.emit("⚠️ Response is taking longer than usual. You may retry if needed.")

        t = threading.Thread(target=_warn_if_slow, daemon=True)
        t.start()

        try:
            conn = http.client.HTTPConnection("127.0.0.1", eng.server_port, timeout=None)
            body = json.dumps({
                "prompt": prompt, "n_predict": n_predict, "stream": False,
                "temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.15,
                "stop": fam.stop_tokens,
            })
            conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
            r = conn.getresponse()
            if r.status != 200: return None
            d = json.loads(r.read().decode("utf-8", errors="replace"))
            return d.get("content", "")
        except Exception as e:
            self.err.emit(f"Server inference error: {e}")
            return None

    def infer_cli(self, eng, prompt: str, n_predict: int) -> Optional[str]:
        import threading
        
        warned = False
        def _warn_if_slow():
            time.sleep(300)  # 5 minutes
            nonlocal warned
            warned = True
            self.progress.emit("⚠️ Response is taking longer than usual. You may retry if needed.")

        t = threading.Thread(target=_warn_if_slow, daemon=True)
        t.start()

        try:
            result = subprocess.run(
                [LLAMA_CLI, "-m", eng.model_path, "-t", str(DEFAULT_THREADS),
                "--ctx-size", str(getattr(eng, "ctx_value", DEFAULT_CTX)),
                "-n", str(n_predict), "--no-display-prompt", "--no-escape",
                "--temp", "0.3", "--repeat-penalty", "1.15", "-p", prompt],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, timeout=None,
            )
            return result.stdout.decode("utf-8", errors="replace")
        except Exception as e:
            self.err.emit(f"CLI inference error: {e}")
            return None