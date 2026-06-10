"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *
from nativelab.UI.llm_error_dialog import show_llm_error_dialog


class ChatPipelineMixin:
    def _on_pipeline_from_chat(self):
        """Run a saved pipeline from the chat window. Output rendered as structured chat bubbles."""
        saved = list_saved_pipelines()
        if not saved:
            QMessageBox.information(
                self, "No Saved Pipelines",
                "No saved pipelines found.\n\n"
                "Build and save a pipeline in Dev > Pipeline first.")
            return

        pipeline_name, ok = QInputDialog.getItem(
            self, "Select Pipeline",
            "Choose a pipeline to run on your current input:",
            saved, 0, False)
        if not ok:
            return

        text = self.input_bar.input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Input",
                                "Type a message in the chat input first."); return
        ref_ctx = ""
        try:
            ref_ctx = self.chat_module.ref_panel.get_context_for(text)
        except Exception:
            ref_ctx = ""
        pipeline_text = (
            f"Reference material for this pipeline run:\n\n{ref_ctx}\n\n"
            f"User request:\n{text}"
            if ref_ctx else text
        )

        try:
            blocks, conns = load_pipeline(pipeline_name)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return

        active_pipe_eng = self._active_engine_for(text)
        if not active_pipe_eng or not active_pipe_eng.is_loaded:
            QMessageBox.warning(self, "Engine Not Ready",
                                "Wait for the model to finish loading."); return

        ts = datetime.now().strftime("%H:%M")
        self.chat_area.add_message("user", text, ts)
        self.input_bar.input.clear()

        self.chat_area.add_message(
            "assistant",
            f"**Running pipeline: {pipeline_name}**\n\n"
            f"_{len(blocks)} blocks · processing…_",
            ts)

        self._chat_pipeline_worker = PipelineExecutionWorker(
            blocks, conns, pipeline_text, active_pipe_eng)
        self._chat_pipeline_worker.step_started.connect(
            lambda bid, lbl: self._chat_pipeline_step(lbl))
        self._chat_pipeline_worker.intermediate_live.connect(
            self._chat_pipeline_intermediate)
        self._chat_pipeline_worker.pipeline_done.connect(
            self._chat_pipeline_done)
        self._chat_pipeline_worker.err.connect(self._on_chat_pipeline_exec_err)
        self._chat_pipeline_worker.log_msg.connect(
            lambda m: self._log("INFO", m))
        self._chat_pipeline_worker.start()

    def _on_chat_pipeline_exec_err(self, msg: str):
        self._log("ERROR", f"Chat pipeline error: {msg}")
        notice = show_llm_error_dialog(self, msg, source="Chat pipeline")
        self.chat_area.add_message(
            "assistant",
            f"{notice.title}\n\n{notice.user_message}",
            datetime.now().strftime("%H:%M"))
        self._chat_pipeline_worker = None

    def _chat_pipeline_step(self, label: str):
        ts = datetime.now().strftime("%H:%M")
        self.chat_area.add_message("system_note",
                                   f"Processing block: **{label}**...", ts)

    def _chat_pipeline_intermediate(self, bid: int, label: str, text: str):
        ts = datetime.now().strftime("%H:%M")
        self.chat_area.add_message(
            "pipeline_intermediate",
            f"**{label}** - intermediate output\n\n{text}", ts)

    def _chat_pipeline_done(self, payload: str):
        import json as _json
        try:
            data   = _json.loads(payload)
            final  = data.get("text", payload)
            sender = data.get("sender", "")
        except Exception:
            final  = payload
            sender = ""
        ts     = datetime.now().strftime("%H:%M")
        header = f"**Output** _(from: {sender})_\n\n" if sender else "**Output**\n\n"
        self.chat_area.add_message("assistant", header + final, ts)
        self._chat_pipeline_worker = None

    def _on_send_with_refs(self, text: str, ref_ctx: str, ref_images=None):
        """Called by ChatModule - injects reference context into prompt."""
        self._pending_ref_ctx = ref_ctx
        self._pending_ref_images = list(ref_images or [])
        self._on_send(text)

    def _llama_image_data_from_parts(self, ref_images: list) -> list:
        image_data = []
        for part in ref_images or []:
            data = ""
            if isinstance(part, dict) and part.get("data"):
                data = part.get("data", "")
            elif isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if ";base64," in url:
                    data = url.split(";base64,", 1)[1]
            elif isinstance(part, dict) and part.get("type") == "image":
                src = part.get("source") or {}
                if src.get("type") == "base64":
                    data = src.get("data", "")
            if data:
                image_data.append({"id": int(part.get("id", len(image_data) + 1)), "data": data})
        return image_data

    def _on_send(self, text: str):
        if not self.active:    self._new_session()
        active_eng = self._active_engine_for(text)
        if not active_eng or not active_eng.is_loaded:
            self._deferred_send = (
                text,
                getattr(self, "_pending_ref_ctx", ""),
                list(getattr(self, "_pending_ref_images", [])),
            )
            self._set_engine_status("Loading selected model for first message...", "loading")
            loader = getattr(self, "_loader", None)
            api_loader = getattr(self, "_api_loader", None)
            if not ((loader and loader.isRunning()) or (api_loader and api_loader.isRunning())):
                self._start_model_load()
            self._log("INFO", "Loading selected model before sending first message.")
            return
        # Block sends while ANY session is generating
        if self._busy_session_id:
            current_sid = self.active.id if self.active else ""
            if self._busy_session_id != current_sid:
                busy_sess = self.sessions.get(self._busy_session_id)
                busy_title = busy_sess.title if busy_sess else self._busy_session_id
                QMessageBox.warning(
                    self, "Processing in Progress",
                    f'Chat "{busy_title}" is currently generating a response.\n\n'
                    f"Please wait for it to finish before sending a new message.")
            else:
                QMessageBox.warning(
                    self, "Processing in Progress",
                    "This chat is already generating a response.\n"
                    "Please wait for it to finish.")
            return
        # Block new messages while summarization is active
        if getattr(self, "_summarizing_active", False):
            QMessageBox.warning(
                self, "Summarization Active",
                "A summarization job is in progress.\n\n"
                "Please pause or abort it from the chat panel before sending new messages.\n"
                "You can resume paused jobs from the top-right settings button.")
            return
        ts = datetime.now().strftime("%H:%M")
        if not self.active.messages:
            self.active.title = text[:40].replace("\n", " ")
            self.active.save()

        self.active.add_message("user", text)
        self.active.save()
        self.chat_area.add_message("user", text, ts)
        self._update_ctx_bar()

        ref_ctx_for_turn = getattr(self, "_pending_ref_ctx", "")
        pipeline_text = (
            f"Reference material for this turn:\n\n{ref_ctx_for_turn}\n\n"
            f"User request:\n{text}"
            if ref_ctx_for_turn else text
        )
        skill_ctx = self._active_skill_context()
        if skill_ctx:
            pipeline_text = (
                f"{skill_ctx}\n\n"
                "Apply relevant active skills while answering.\n\n"
                f"{pipeline_text}"
            )

        # ── Pipeline mode ─────────────────────────────────────────────────────
        if self._can_use_pipeline(text):
            self._pending_ref_ctx = ""
            self._pending_ref_images = []
            self._start_pipeline(pipeline_text, ts)
            return

        # ── Normal / single-engine mode ───────────────────────────────────────
        is_coding  = active_eng is self.coding_engine
        eng_label  = "Coding engine" if is_coding else engine_status(active_eng).status_text

        ctx_chars = getattr(active_eng, "ctx_value", DEFAULT_CTX()) * 4
        prompt    = self.active.build_prompt(
            model_path=active_eng.model_path,
            max_chars=ctx_chars
        )
        if skill_ctx:
            fam = detect_model_family(model_ref_payload(active_eng.model_path) or active_eng.model_path)
            skill_block = (
                f"{fam.bos}{fam.user_prefix}"
                f"{skill_ctx}\n\n"
                "Apply relevant active skills while answering."
                f"{fam.user_suffix}{fam.assistant_prefix}"
            )
            prompt = skill_block + "\n" + prompt
        # Inject reference context if available
        ref_ctx = getattr(self, "_pending_ref_ctx", "")
        ref_images = list(getattr(self, "_pending_ref_images", []))
        if ref_ctx:
            fam = detect_model_family(model_ref_payload(active_eng.model_path) or active_eng.model_path)
            ref_block = (
                f"{fam.bos}{fam.user_prefix}"
                f"The following reference material is provided for context:\n\n"
                f"{ref_ctx}\n\n"
                f"Use this reference when answering the user's question."
                f"{fam.user_suffix}{fam.assistant_prefix}"
            )
            prompt = ref_block + "\n" + prompt
            self._pending_ref_ctx = ""
        if ref_images and not isinstance(active_eng, ApiEngine):
            llama_images = self._llama_image_data_from_parts(ref_images)
            if active_eng.mode in ("server", "ollama", "hf_transformers") and llama_images and hasattr(active_eng, "set_images"):
                active_eng.set_images(llama_images)
                prompt = (
                    "The user attached image reference(s). Inspect the attached image data "
                    "together with the text reference metadata when answering.\n\n" + prompt
                )
            else:
                prompt = (
                    "The user attached image reference(s), but the active local model is not "
                    "running in a vision-capable backend. Use any image filenames listed in "
                    "the reference context as metadata only.\n\n" + prompt
                )

        cfg_pred = 0 if isinstance(active_eng, ApiEngine) else DEFAULT_N_PRED
        if active_eng.model_path and not isinstance(active_eng, ApiEngine):
            cfg_pred = get_model_registry().get_config(active_eng.model_path).n_predict

        self._stream_w = self.chat_area.add_message(
            "assistant", "", ts, tag="Coding" if is_coding else "")
        self._log("INFO", f"Prompt ≈ {len(prompt)} chars · engine: {eng_label}")

        # For API engines, pass the full structured message history
        if isinstance(active_eng, ApiEngine):
            api_msgs = [{"role": m.role, "content": m.content}
                        for m in self.active.messages[-60:]]
            if skill_ctx:
                api_msgs = [{"role": "system", "content": skill_ctx}] + api_msgs
            if api_msgs:
                last_user_idx = next(
                    (i for i in range(len(api_msgs) - 1, -1, -1)
                     if api_msgs[i].get("role") == "user"),
                    -1)
                if last_user_idx >= 0 and (ref_ctx or ref_images):
                    user_text = str(api_msgs[last_user_idx].get("content", ""))
                    if ref_ctx:
                        user_text = (
                            "Reference material for this turn:\n\n"
                            f"{ref_ctx}\n\n"
                            "Use the reference material when answering.\n\n"
                            f"User request:\n{user_text}"
                        )
                    if ref_images:
                        if active_eng._config and active_eng._config.api_format == "anthropic":
                            api_msgs[last_user_idx]["content"] = (
                                [{"type": "text", "text": user_text}] + ref_images
                            )
                        else:
                            api_msgs[last_user_idx]["content"] = (
                                [{"type": "text", "text": user_text}] + ref_images
                            )
                    else:
                        api_msgs[last_user_idx]["content"] = user_text
            active_eng.set_messages(api_msgs)
            context_meter.report_messages(
                source="Chat",
                engine=active_eng,
                messages=api_msgs,
                n_predict=cfg_pred,
            )
        else:
            context_meter.report_prompt(
                source="Coding" if is_coding else "Chat",
                engine=active_eng,
                prompt=prompt,
                n_predict=cfg_pred,
            )
        self._pending_ref_images = []

        self._worker = active_eng.create_worker(
            prompt, n_predict=cfg_pred, model_path=active_eng.model_path)
        self._worker.token.connect(self._on_token)
        self._worker.done.connect(self._on_done)
        self._worker.err.connect(self._on_err)
        self._worker.start()

        self._stream_buffer     = ""
        self._stream_session_id = self.active.id if self.active else ""
        self._busy_session_id   = self._stream_session_id
        self._refresh_sidebar()

        self.input_bar.set_generating(True)
        lbl_txt = "Coding..." if is_coding else "Generating..."
        self._set_engine_status(lbl_txt, "loading")
        self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")

    def _collect_insight_engines(self):
        """Return list of (label, engine) for all active non-coding loaded engines."""
        candidates = [
            ("Reasoning",     self.reasoning_engine),
            ("Summarization", self.summarization_engine),
            ("Secondary",     self.secondary_engine),
        ]
        # Also include primary engine if it's not the coding engine
        if self.engine and self.engine.is_loaded and self.engine is not self.coding_engine:
            candidates.append(("Primary", self.engine))

        return [
            (label, eng)
            for label, eng in candidates
            if eng is not None and eng.is_loaded and eng is not self.coding_engine
        ]

    def _start_pipeline(self, text: str, ts: str):
        # Ensure all engines are in server mode before starting pipeline
        # to avoid CLI prompt echo glitch
        engines_to_check = []
        if self.coding_engine and self.coding_engine.is_loaded:
            engines_to_check.append(("coding", self.coding_engine))
        for role in ("reasoning", "summarization", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng and eng.is_loaded:
                engines_to_check.append((role, eng))

        for role, eng in engines_to_check:
            if eng.mode != "server":
                self._log("WARN", f"{role} engine in CLI mode - attempting server upgrade…")
                ok = eng.ensure_server_or_reload(log_cb=self._log)
                if not ok:
                    self._log("ERROR",
                        f"{role} engine could not start server mode - aborting pipeline")
                    self.chat_area.add_message(
                        "assistant",
                        f"Pipeline aborted: **{role}** engine could not start in server mode.\n"
                        f"Try reloading the model from the Models tab.",
                        ts)
                    self.input_bar.set_generating(False)
                    return

        insight_engines = self._collect_insight_engines()

        if not insight_engines:
            # Fallback: no insight engines, just run coding engine directly
            self._log("WARN", "No insight engines available - running coding engine directly")
            active_eng = self.coding_engine
            ctx_chars  = getattr(active_eng, "ctx_value", DEFAULT_CTX()) * 4
            prompt     = self.active.build_prompt(model_path=active_eng.model_path, max_chars=ctx_chars)
            cfg_pred   = get_model_registry().get_config(active_eng.model_path).n_predict
            self._stream_w = self.chat_area.add_message("assistant", "", ts, tag="Coding")
            context_meter.report_prompt(
                source="Pipeline",
                engine=active_eng,
                prompt=prompt,
                n_predict=cfg_pred,
            )
            self._worker = active_eng.create_worker(prompt, n_predict=cfg_pred, model_path=active_eng.model_path)
            self._worker.token.connect(self._on_token)
            self._worker.done.connect(self._on_done)
            self._worker.err.connect(self._on_err)
            self._worker.start()
            self.input_bar.set_generating(True)
            return

        n_engines = len(insight_engines)
        self._log("INFO", f"Pipeline: {n_engines} insight engine(s) -> coding")
        self._set_engine_status("Structural Insights...", "loading")
        self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")
        self.input_bar.set_generating(True)

        # Create one bubble per insight engine + divider + coding bubble
        self._pipeline_insight_widgets = []
        for label, eng in insight_engines:
            w = self.chat_area.add_message("assistant", "", ts, tag=label)
            self._pipeline_insight_widgets.append(w)

        self.chat_area.add_pipeline_divider(
            f"{n_engines} model(s) analysed → Coding model generating"
        )
        self._pipeline_reason_w = self._pipeline_insight_widgets[0] if self._pipeline_insight_widgets else None
        self._pipeline_code_w   = self.chat_area.add_message("assistant", "", ts, tag="Coding")

        insight_np = max(
            (get_model_registry().get_config(eng.model_path).n_predict for _, eng in insight_engines),
            default=512
        )
        code_np = get_model_registry().get_config(self.coding_engine.model_path).n_predict

        self._pipeline_worker = PipelineWorker(
            insight_engines, self.coding_engine, text,
            n_predict_insight=insight_np, n_predict_code=max(code_np, 1024)
        )
        self._pipeline_worker.insight_started.connect(self._on_pipeline_insight_started)
        self._pipeline_worker.insight_token.connect(self._on_pipeline_insight_token)
        self._pipeline_worker.insight_done.connect(self._on_pipeline_insight_done)
        self._pipeline_worker.coding_token.connect(self._on_pipeline_code_token)
        self._pipeline_worker.coding_done.connect(self._on_pipeline_done)
        self._pipeline_worker.stage_changed.connect(self._on_pipeline_stage)
        self._pipeline_worker.err.connect(self._on_pipeline_err)
        self._pipeline_worker.start()
        self._busy_session_id = self.active.id if self.active else ""
        self._refresh_sidebar()

    def _on_pipeline_insight_started(self, idx: int, label: str):
        self._set_engine_status(f"{label} analysing...", "loading")
        self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")

    def _on_pipeline_insight_token(self, idx: int, token: str):
        context_meter.append_output(token)
        widgets = getattr(self, "_pipeline_insight_widgets", [])
        if idx < len(widgets) and widgets[idx]:
            try:
                widgets[idx].append_text(token)
            except RuntimeError:
                pass
        self.chat_area._scroll_bottom()

    def _on_pipeline_insight_done(self, idx: int, full_text: str):
        widgets = getattr(self, "_pipeline_insight_widgets", [])
        if idx < len(widgets) and widgets[idx]:
            try:
                widgets[idx].finalize()
            except RuntimeError:
                pass

    def _on_pipeline_reason_token(self, text: str):
        self._on_pipeline_insight_token(0, text)

    def _on_pipeline_reason_done(self, full_text: str):
        self._on_pipeline_insight_done(0, full_text)

    def _on_pipeline_stage(self, stage: str):
        if stage == "coding":
            self._set_engine_status("Coding...", "loading")
            self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        elif stage == "insights":
            self._set_engine_status("Structural Insights...", "loading")
            self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")

    def _on_pipeline_code_token(self, text: str):
        context_meter.append_output(text)
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.append_text(text)
            except RuntimeError:
                pass
        self.chat_area._scroll_bottom()

    def _on_pipeline_done(self, tps: float):
        self.tps_lbl.setText(f"{tps:.1f} tok/s")
        self._set_engine_status_from_engine(self.engine)
        self.input_bar.set_generating(False)

        # Collect all insight texts for session save
        insight_parts = []
        for w in getattr(self, "_pipeline_insight_widgets", []):
            if w:
                try:
                    w.finalize()
                    t = w.full_text.strip()
                    if t:
                        insight_parts.append(t)
                except RuntimeError:
                    pass

        code_text = ""
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.finalize()
                code_text = self._pipeline_code_w.full_text.strip()
            except RuntimeError:
                pass

        if self.active and (insight_parts or code_text):
            insights_joined = "\n\n---\n\n".join(
                f"**[Structural Insight {i+1}]**\n\n{t}"
                for i, t in enumerate(insight_parts)
            )
            self.active.add_message("assistant",
                f"{insights_joined}\n\n---\n\n**[Code Output]**\n\n{code_text}")
            self.active.save()

        self._pipeline_insight_widgets = []
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None
        self._pipeline_worker   = None
        self._busy_session_id   = ""
        self._update_ctx_bar()
        self._refresh_sidebar()

    def _on_pipeline_err(self, msg: str):
        self._log("ERROR", f"Pipeline error: {msg}")
        notice = show_llm_error_dialog(self, msg, source="Chat pipeline")
        self._set_engine_status("Pipeline Error", "err")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.append_text(f"\n\n{notice.title}\n\n{notice.user_message}")
            except RuntimeError:
                pass
        self._pipeline_insight_widgets = []
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None
        self._pipeline_worker   = None
        self._busy_session_id   = ""
        self._refresh_sidebar()

    def _on_token(self, text: str):
        context_meter.append_output(text)
        self._stream_buffer += text          # shadow-buffer; always kept
        if not self._stream_w: return
        try:
            self._stream_w.append_text(text)
        except RuntimeError:
            self._stream_w = None            # widget gone; stop future writes
            return
        self.chat_area._scroll_bottom()

    def _on_done(self, tps: float):
        self.tps_lbl.setText(f"{tps:.1f} tok/s")
        self._set_engine_status_from_engine(self._active_engine_for(""))
        self.input_bar.set_generating(False)
        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass
        self._save_streamed()
        self._busy_session_id = ""
        self._refresh_sidebar()

    def _on_err(self, msg: str):
        self._log("ERROR", msg)
        notice = show_llm_error_dialog(self, msg, source="Chat")
        self._set_engine_status("Error", "err")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._stream_w:
            try: self._stream_w.append_text(f"\n\n{notice.title}\n\n{notice.user_message}")
            except RuntimeError: pass
        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass
        self._save_streamed()
        self._busy_session_id = ""
        self._refresh_sidebar()

    def _on_stop(self):
        if self._pipeline_worker:
            if not stop_worker(self._pipeline_worker, LONG_TIMEOUT_MS):
                self._log("WARN", "Pipeline worker is still stopping.")
                return
            self._pipeline_worker = None
            for w in (self._pipeline_reason_w, self._pipeline_code_w):
                if w:
                    try: w.finalize()
                    except RuntimeError: pass
            self._pipeline_reason_w = None
            self._pipeline_code_w   = None

        if self._worker:
            if not stop_worker(self._worker, LONG_TIMEOUT_MS):
                self._log("WARN", "Generation worker is still stopping.")
                return
            self._worker = None

        if self._summary_worker:
            stopped = stop_worker(
                self._summary_worker,
                LONG_TIMEOUT_MS,
                abort=False,
                cancel=False,
                request_pause=True,
            )
            if not stopped:
                stopped = stop_worker(self._summary_worker, LONG_TIMEOUT_MS)
            if not stopped:
                self._log("WARN", "Summary worker is still stopping.")
                return
            self._summary_worker = None
            if hasattr(self, "_summary_bubble") and self._summary_bubble:
                self._summary_bubble.append_text(
                    "\n\nPaused & saved to disk. Resume from the top-right settings button.")
                self._summary_bubble = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()

        if self._multi_pdf_worker:
            stopped = stop_worker(
                self._multi_pdf_worker,
                LONG_TIMEOUT_MS,
                abort=False,
                cancel=False,
                request_pause=True,
            )
            if not stopped:
                stopped = stop_worker(self._multi_pdf_worker, LONG_TIMEOUT_MS)
            if not stopped:
                self._log("WARN", "Multi-PDF worker is still stopping.")
                return
            self._multi_pdf_worker = None
            if hasattr(self, "_summary_bubble") and self._summary_bubble:
                self._summary_bubble.append_text(
                    "\n\nMulti-PDF paused & saved. Resume from the top-right settings button.")
                self._summary_bubble = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()

        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass

        self.input_bar.set_generating(False)
        self._set_engine_status_from_engine(self.engine)
        self._log("INFO", "Generation stopped by user.")
        self._save_streamed(suffix=" [stopped]")
        self._busy_session_id  = ""
        self._stream_session_id = ""
        self._stream_buffer     = ""
        self._summary_session_id = ""
        self._refresh_sidebar()

    def _save_streamed(self, suffix: str = ""):
        # Try to flush the live widget first; fall back to the shadow buffer
        content = ""
        if self._stream_w:
            try:
                self._stream_w._flush_timer.stop()
                self._stream_w._flush_pending()
                content = self._stream_w.full_text.strip()
            except RuntimeError:
                self._stream_w = None
        if not content:
            content = self._stream_buffer.strip()

        # Save to the session that STARTED the stream, not necessarily self.active
        save_sid = self._stream_session_id or (self.active.id if self.active else "")
        if content and save_sid:
            sess = self.sessions.get(save_sid)
            if sess:
                sess.add_message("assistant", content + suffix)
                sess.save()

        self._stream_w         = None
        self._worker           = None
        self._stream_buffer    = ""
        self._stream_session_id = ""
        self._update_ctx_bar()
