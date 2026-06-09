"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *


class DocumentsMixin:
    def _load_pdf(self):
        if not HAS_PDF:
            QMessageBox.warning(self, "Missing Dependency",
                                "Install pypdf:  pip install pypdf"); return
        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Model Not Ready",
                                "Wait for the model to finish loading."); return

        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if not path: return

        reader   = PdfReader(path)
        n_pages  = len(reader.pages)
        filename = Path(path).name
        self._log("INFO", f"Reading PDF: {filename}  ({n_pages} pages)")

        text = ""
        for i, page in enumerate(reader.pages):
            text += page.extract_text() or ""
            if i % 10 == 0:
                self._log("INFO", f"  …extracted page {i+1}/{n_pages}")

        if not text.strip():
            QMessageBox.warning(self, "Empty PDF", "No text could be extracted."); return

        ctx_chars   = getattr(self.engine, "ctx_value", DEFAULT_CTX()) * 3
        DIRECT_LIMIT = min(ctx_chars, 6000)

        if len(text) <= DIRECT_LIMIT:
            self.input_bar.input.setPlainText(
                f"The following is extracted from '{filename}':\n\n{text}"
                "\n\nPlease summarise the key points.")
            self._log("INFO", "Short document - loaded directly into prompt.")
            return

        if not self.active: self._new_session()
        self.active.title = f"Summary: {filename}"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        mode = self.input_bar.summary_mode
        mode_label = {"summary": "Summary", "logical": "Logical Analysis", "advice": "Advisory"}.get(mode, "Summary")

        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"Starting **{mode_label}** of **{filename}** "
            f"({n_pages} pages, {len(text):,} chars)…\n", ts)

        self.input_bar.set_generating(True)
        self.input_bar.input.setEnabled(False)
        self.input_bar.input.setPlaceholderText(
            "Summarization in progress - pause or abort above before typing...")
        self._summarizing_active = True
        self._set_engine_status("Summarising...", "loading")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        # Add pause banner to chat
        self._pause_banner = self.chat_area.add_pause_banner(
            on_pause_cb=self._on_pause_summary,
            on_abort_cb=self._on_stop
        )

        if self._summary_worker:
            if not stop_worker(self._summary_worker, LONG_TIMEOUT_MS):
                self._log("WARN", "Previous summary worker is still stopping.")
                return

        from math import ceil
        estimated_chunks = ceil(len(text) / int(APP_CONFIG["summary_chunk_chars"]))
        self._thinking_block = self.chat_area.add_thinking_block(estimated_chunks)

        self._summary_worker = ChunkedSummaryWorker(
            self.engine, text, filename,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None),
            session_id=self.active.id if self.active else "",
            summary_mode=mode)
        self._summary_worker.progress.connect(self._on_summary_progress)
        self._summary_worker.section_done.connect(self._on_section_done)
        self._summary_worker.final_done.connect(self._on_summary_final)
        self._summary_worker.err.connect(self._on_summary_err_or_pause)
        self._summary_worker.pause_suggest.connect(self._on_pause_suggest)
        self._summary_worker.start()
        self._summary_session_id = self.active.id if self.active else ""
        self._busy_session_id    = self._summary_session_id
        self._refresh_sidebar()

    def _on_pause_summary(self):
        """Pause the active summary worker gracefully."""
        if self._summary_worker and hasattr(self._summary_worker, "request_pause"):
            self._summary_worker.request_pause()
            self._set_engine_status("Pausing...", "warn")
            self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")
            if self._pause_banner:
                try:
                    self._pause_banner.status_lbl.setText("Pausing after current chunk…")
                except RuntimeError:
                    pass
        elif self._multi_pdf_worker and hasattr(self._multi_pdf_worker, "request_pause"):
            self._multi_pdf_worker.request_pause()
            self._set_engine_status("Pausing...", "warn")
            self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")

    def _on_summary_progress(self, msg: str):
        self._log("INFO", msg)
        self._set_engine_status(msg, "loading")
        if "final consolidation" in msg.lower():
            try:
                tb = getattr(self, "_thinking_block", None)
                if tb is not None: tb.add_phase("Final consolidation pass")
            except RuntimeError:
                pass

    def _on_section_done(self, num: int, total: int, chunk_text: str, summary: str):
        self._log("INFO", f"Section {num}/{total} done ({len(summary)} chars)")
        try:
            tb = getattr(self, "_thinking_block", None)
            if tb is not None: tb.add_section(num, total, chunk_text, summary)
        except RuntimeError:
            pass
        try:
            bubble = getattr(self, "_summary_bubble", None)
            if bubble is not None:
                bubble.append_text(f"\nSection {num}/{total} summarised.\n")
        except RuntimeError:
            pass
        self.chat_area._scroll_bottom()

    def _on_summary_final(self, final: str):
        self._log("INFO", f"Final summary ready ({len(final)} chars)")

        # Remove pause banner
        self.chat_area.remove_pause_banner()
        self._pause_banner = None
        self._summarizing_active = False
        self.input_bar.input.setEnabled(True)
        self.input_bar.input.setPlaceholderText(
            "Type a message…  (Enter = send · Shift+Enter = newline)")

        try:
            bubble = getattr(self, "_summary_bubble", None)
            if bubble is not None:
                bubble._flush_timer.stop()
                bubble._flush_pending()
                bubble.append_text("\n\n---\n**Final Summary:**\n\n" + final)
                QTimer.singleShot(80, bubble._flush_pending)
        except RuntimeError:
            pass
        finally:
            self._summary_bubble = None
        try:
            tb = getattr(self, "_thinking_block", None)
            if tb is not None: tb.mark_done()
        except RuntimeError:
            pass
        finally:
            self._thinking_block = None

        final_text = final
        save_sid     = self._summary_session_id or (self.active.id if self.active else "")
        def _persist():
            try:
                sess = self.sessions.get(save_sid)
                if sess:
                    sess.add_message("assistant", f"**Document Summary**\n\n{final_text}")
                    sess.save()
            except Exception as e:
                self._log("WARN", f"Could not save final summary: {e}")
            self._summary_worker    = None
            self._summary_session_id = ""
            self._busy_session_id   = ""
            self.input_bar.set_generating(False)
            self._set_engine_status_from_engine(self.engine)
            self._update_ctx_bar()
            self._refresh_sidebar()
        QTimer.singleShot(120, _persist)

    def _on_multi_pdf_ram_warning(self, msg: str):
        self._log("WARN", msg)
        bubble = getattr(self, "_summary_bubble", None)
        if bubble:
            try:
                bubble.append_text(f"\n{msg}\n")
            except RuntimeError:
                pass

    def _start_multi_pdf(self, paths: List[str]):
        """Start multi-PDF summarization with adaptive RAM watchdog."""
        if not HAS_PDF:
            QMessageBox.warning(self, "Missing Dep", "Install pypdf: pip install pypdf"); return
        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Not Ready", "Wait for model to load."); return

        pdf_texts: List[Tuple[str, str]] = []
        for path in paths:
            try:
                reader = PdfReader(path)
                text   = "\n".join(p.extract_text() or "" for p in reader.pages)
                if text.strip():
                    pdf_texts.append((Path(path).name, text))
                    self._log("INFO", f"Loaded: {Path(path).name} ({len(text):,} chars)")
            except Exception as e:
                self._log("WARN", f"Could not read {path}: {e}")

        if not pdf_texts:
            QMessageBox.warning(self, "No Content", "No readable PDFs found."); return

        if not self.active: self._new_session()
        self.active.title = f"Multi-PDF: {len(pdf_texts)} docs"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"Starting multi-PDF summarization:\n"
            + "\n".join(f"  • {fn} ({len(t):,} chars)" for fn, t in pdf_texts)
            + "\n\nProcessing...\n", ts)

        from math import ceil
        total_chunks = sum(
            ceil(len(t) / int(APP_CONFIG["summary_chunk_chars"])) for _, t in pdf_texts)
        self._thinking_block = self.chat_area.add_thinking_block(total_chunks)

        self.input_bar.set_generating(True)
        self._set_engine_status("Multi-PDF...", "loading")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        sid = self.active.id if self.active else "default"
        self._multi_pdf_worker = MultiPdfSummaryWorker(
            self.engine, pdf_texts, sid,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None))
        self._multi_pdf_worker.file_started.connect(
            lambda fi, fn, nc: self._log("INFO", f"PDF {fi+1}: {fn} ({nc} chunks)"))
        self._multi_pdf_worker.file_progress.connect(
            lambda fi, msg: self._set_engine_status(f"PDF {fi+1}: {msg}", "loading"))
        self._multi_pdf_worker.ram_warning.connect(self._on_multi_pdf_ram_warning)
        self._multi_pdf_worker.section_done.connect(self._on_section_done)
        self._multi_pdf_worker.file_done.connect(
            lambda fi, s: (
                self._log("INFO", f"File {fi+1} summary done ({len(s)} chars)"),
                self._summary_bubble and self._summary_bubble.append_text(
                    f"\nDocument {fi+1} summarised.\n")))
        self._multi_pdf_worker.progress.connect(self._on_summary_progress)
        self._multi_pdf_worker.final_done.connect(self._on_multi_pdf_final)
        self._multi_pdf_worker.err.connect(self._on_summary_err_or_pause)
        self._multi_pdf_worker.pause_suggest.connect(self._on_pause_suggest)
        self._multi_pdf_worker.start()

    def _on_multi_pdf_final(self, final: str):
        self._log("INFO", f"Multi-PDF final summary ready ({len(final)} chars)")
        try:
            bubble = getattr(self, "_summary_bubble", None)
            if bubble:
                bubble.append_text("\n\n---\n**Final Multi-Document Summary:**\n\n" + final)
                QTimer.singleShot(80, bubble._flush_pending)
        except RuntimeError:
            pass
        self._summary_bubble = None
        try:
            tb = getattr(self, "_thinking_block", None)
            if tb: tb.mark_done()
        except RuntimeError:
            pass
        self._thinking_block = None

        if self.active:
            self.active.add_message("assistant",
                f"**Multi-Document Summary**\n\n{final}")
            self.active.save()

        self._multi_pdf_worker = None
        self.input_bar.set_generating(False)
        self._set_engine_status_from_engine(self.engine)
        self._update_ctx_bar()

    def _resume_paused_job(self):
        state = self.config_tab.get_selected_job_state()
        if not state:
            QMessageBox.warning(self, "No Job Selected", "Select a paused job first.")
            return
        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Not Ready", "Wait for model to load."); return

        raw_text = state.get("raw_text", "")        
        filename   = state.get("filename", "unknown")
        job_id     = state.get("job_id", "")
        session_id = state.get("session_id", "")
        pdf_texts  = state.get("pdf_texts", None)   # present for multi-PDF jobs

        # ── Multi-PDF resume ──────────────────────────────────────────────────
        if pdf_texts is not None:
            if not self.engine.is_loaded:
                QMessageBox.warning(self, "Not Ready", "Wait for model to load.")
                return
            if not self.active: self._new_session()
            self.active.title = f"Resume Multi-PDF: {filename}"
            self.active.save(); self._refresh_sidebar()
            ts = datetime.now().strftime("%H:%M")
            next_fi = state.get("next_fi", 0)
            next_ci = state.get("next_ci", 0)
            total   = state.get("total", len(pdf_texts))
            self._summary_bubble = self.chat_area.add_message(
                "assistant",
                f"Resuming multi-PDF job from file {next_fi+1}/{total}, "
                f"chunk {next_ci+1}…\n", ts)
            from math import ceil
            est = sum(
                ceil(len(t) / int(APP_CONFIG["summary_chunk_chars"]))
                for _, t in pdf_texts[next_fi:])
            self._thinking_block = self.chat_area.add_thinking_block(max(est, 1))
            self.input_bar.set_generating(True)
            self._set_engine_status("Resuming Multi-PDF...", "loading")
            self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")
            sid = self.active.id if self.active else "default"
            self._multi_pdf_worker = MultiPdfSummaryWorker(
                self.engine,
                [(fn, txt) for fn, txt in pdf_texts],
                sid,
                engine2=getattr(self, "summarization_engine", None) or
                        getattr(self, "reasoning_engine", None),
                resume_job_id=job_id)
            self._multi_pdf_worker.file_started.connect(
                lambda fi, fn, nc: self._log("INFO", f"PDF {fi+1}: {fn} ({nc} chunks)"))
            self._multi_pdf_worker.file_progress.connect(
                lambda fi, msg: self._set_engine_status(f"PDF {fi+1}: {msg}", "loading"))
            self._multi_pdf_worker.ram_warning.connect(
                lambda msg: self._log("WARN", msg))
            self._multi_pdf_worker.section_done.connect(self._on_section_done)
            self._multi_pdf_worker.file_done.connect(
                lambda fi, s: self._log("INFO", f"File {fi+1} done ({len(s)} chars)"))
            self._multi_pdf_worker.progress.connect(self._on_summary_progress)
            self._multi_pdf_worker.final_done.connect(self._on_multi_pdf_final)
            self._multi_pdf_worker.err.connect(self._on_summary_err_or_pause)
            self._multi_pdf_worker.pause_suggest.connect(self._on_pause_suggest)
            self._multi_pdf_worker.start()
            self.config_tab.refresh_paused_jobs()
            self._log("INFO", f"Resumed multi-PDF job: {job_id}")
            return

        # ── Single-PDF resume (original path) ─────────────────────────────────
        if not raw_text:
            QMessageBox.warning(self, "No Data", "Paused job has no raw text saved.")
            return

        if not self.active: self._new_session()
        self.active.title = f"Resume: {filename}"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        next_chunk = state.get("next_chunk", 0)
        total      = state.get("total", "?")
        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"Resuming summarization of **{filename}** "
            f"from chunk {next_chunk}/{total}…\n", ts)

        from math import ceil
        est_remaining = (total - next_chunk) if isinstance(total, int) else 5
        self._thinking_block = self.chat_area.add_thinking_block(
            max(est_remaining, 1))

        self.input_bar.set_generating(True)
        self._set_engine_status(f"Resuming '{filename}'...", "loading")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        self._summary_worker = ChunkedSummaryWorker(
            self.engine, raw_text, filename,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None),
            resume_job_id=job_id,
            session_id=session_id or (self.active.id if self.active else ""),
            summary_mode=state.get("summary_mode", "summary"))
        self._summary_worker.progress.connect(self._on_summary_progress)
        self._summary_worker.section_done.connect(self._on_section_done)
        self._summary_worker.final_done.connect(self._on_summary_final)
        self._summary_worker.err.connect(self._on_summary_err_or_pause)
        self._summary_worker.pause_suggest.connect(self._on_pause_suggest)
        self._summary_worker.start()
        self.config_tab.refresh_paused_jobs()
        self._log("INFO", f"Resumed paused job: {job_id}")

    def _on_pause_suggest(self, job_id: str):
        """Show pause banner in chat when auto-pause threshold is hit."""
        ts = datetime.now().strftime("%H:%M")
        banner = self.chat_area.add_message(
            "assistant",
            f"**Pause available** - The summarization has processed "
            f"{APP_CONFIG['pause_after_chunks']} chunks. You can pause & save "
            f"state now to resume later, or continue processing.\n\n"
            f"Click **Pause** in the stop button area to pause, "
            f"or ignore this to keep going.", ts)

    def _on_summary_err_or_pause(self, msg: str):
        if msg.startswith("__PAUSED__:"):
            job_id = msg.split(":", 1)[1]
            self._log("INFO", f"Job paused: {job_id}")
            self.chat_area.remove_pause_banner()
            self._pause_banner = None
            self._summarizing_active = False
            self.input_bar.input.setEnabled(True)
            self.input_bar.input.setPlaceholderText(
                "Type a message…  (Enter = send · Shift+Enter = newline)")
            self.input_bar.set_generating(False)
            self._set_engine_status("Paused - state saved", "warn")
            self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")
            self._summary_worker = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()
            return
        self._on_summary_err(msg)

    def _on_summary_err(self, msg: str):
        self._log("ERROR", f"Summary pipeline error: {msg}")
        self.chat_area.remove_pause_banner()
        self._pause_banner = None
        self._summarizing_active = False
        self.input_bar.input.setEnabled(True)
        self.input_bar.input.setPlaceholderText(
            "Type a message…  (Enter = send · Shift+Enter = newline)")
        if hasattr(self, "_summary_bubble") and self._summary_bubble:
            self._summary_bubble.append_text(f"\n\nError: {msg}")
        self._summary_worker = None
        self._summary_bubble = None
        self.input_bar.set_generating(False)
        self._set_engine_status("Summary failed", "err")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")

    def _export_active(self, fmt: str):
        if self.active: self._export_session(self.active.id, fmt)

    def _export_session(self, sid: str, fmt: str):
        s = self.sessions.get(sid)
        if not s: return
        ext_map = {"json": "JSON (*.json)", "md": "Markdown (*.md)", "txt": "Plain Text (*.txt)"}
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Session", f"{s.id}.{fmt}", ext_map.get(fmt, "*"))
        if not path: return
        content = {"json": s.to_json, "md": s.to_markdown, "txt": s.to_txt}.get(fmt, s.to_json)()
        Path(path).write_text(content, encoding="utf-8")
        self._log("INFO", f"Exported to {path}")
