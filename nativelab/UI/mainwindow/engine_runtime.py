"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *
from nativelab.UI.llm_error_dialog import show_llm_error_dialog


class EngineRuntimeMixin:
    _CODING_KEYWORDS = (
        "def ", "class ", "import ", "function ", "```", "debug ",
        "fix bug", "write code", "python ", "javascript", "typescript",
        "rust ", "c++", "c#", "golang", " sql", "regex", "script ",
        "component", "api endpoint", "unit test", "refactor", "stacktrace",
        "code to ", "write a ", "create a ", "build a ", "generate code",
    )

    def _loader_is_running(self, attr: str = "_loader") -> bool:
        loader = getattr(self, attr, None)
        try:
            return bool(loader and loader.isRunning())
        except Exception:
            return False

    def _busy_primary_engine_tasks(self) -> list[str]:
        tasks: list[str] = []
        for attr, label in (
            ("_worker", "chat generation"),
            ("_summary_worker", "document summary"),
            ("_multi_pdf_worker", "multi-PDF summary"),
            ("_pipeline_worker", "pipeline generation"),
            ("_chat_pipeline_worker", "chat pipeline"),
        ):
            worker = getattr(self, attr, None)
            if worker is not None and hasattr(worker, "isRunning"):
                try:
                    if worker.isRunning():
                        tasks.append(label)
                except Exception:
                    pass
        if hasattr(self, "pipeline_tab"):
            worker = getattr(self.pipeline_tab, "_exec_worker", None)
            if worker is not None and hasattr(worker, "isRunning"):
                try:
                    if worker.isRunning():
                        tasks.append("pipeline builder execution")
                except Exception:
                    pass
        if hasattr(self, "labs_tab"):
            busy_panels = getattr(self.labs_tab, "busy_panel_names", None)
            if callable(busy_panels):
                try:
                    tasks.extend(f"lab: {name}" for name in busy_panels())
                except Exception:
                    pass
        return tasks

    def _can_restart_primary_engine(self, action: str) -> bool:
        if self._loader_is_running("_loader") or self._loader_is_running("_api_loader"):
            QMessageBox.warning(
                self,
                "Model Load Active",
                f"Wait for the current model load to finish before {action}."
            )
            return False
        busy = self._busy_primary_engine_tasks()
        if busy:
            QMessageBox.warning(
                self,
                "Engine Busy",
                f"Stop the active task before {action}:\n" + "\n".join(f"- {name}" for name in busy)
            )
            return False
        return True

    def _start_model_load(self):
        model = self.input_bar.selected_model
        if is_api_model_ref(model):
            self._start_api_model_load(model)
            return
        if self._loader_is_running("_loader") or self._loader_is_running("_api_loader"):
            self._log("WARN", "Model load already in progress; ignoring duplicate load request.")
            return
        if not model or not is_model_ref_valid(model):
            model = str(MODELS_DIR / DEFAULT_MODEL)
        if self.engine.is_loaded and getattr(self.engine, "model_path", "") == model:
            self._set_engine_status_from_engine(self.engine)
            return
        if (
            (self.engine and self.engine.is_loaded)
            or (self._api_engine and self._api_engine.is_loaded)
        ) and not self._can_restart_primary_engine("loading another model"):
            return
        if self._api_engine and self._api_engine.is_loaded:
            self._api_engine.shutdown()
        self._set_engine_status("Loading model...", "loading")
        cfg = get_model_registry().get_config(model)
        profile_ctx = max(512, min(MAX_CONTEXT_TOKENS, int(cfg.ctx)))
        self.ctx_slider.blockSignals(True)
        self.ctx_slider.setValue(profile_ctx)
        self.ctx_slider.blockSignals(False)
        self.ctx_input.setText(str(profile_ctx))
        self.current_ctx = profile_ctx
        self.ctx_bar.setRange(0, profile_ctx)
        payload = model_ref_payload(model) or model
        fam   = detect_model_family(payload)
        quant = cfg.quant_type
        vi    = cfg.vision_info
        ql, _ = quant_info(quant)
        vision = f"  ·  VLM: {vi.label}" if vi.is_vision else ""
        self.lbl_family.setText(f"{fam.name}  ·  {quant}  ·  {ql}{vision}")
        self._log(
            "INFO",
            f"Loading model profile: {model_ref_display_name(model)}  "
            f"[{fam.name} / {quant}{vision}] ctx={profile_ctx:,} threads={cfg.threads}",
        )
        self._loader = ModelLoaderThread(self.engine, model, profile_ctx, cfg.threads)
        self._loader.log.connect(self._log)
        self._loader.finished.connect(self._on_model_loaded)
        self._loader.start()

    def _on_model_loaded(self, ok: bool, status: str):
        self.ctx_slider.setEnabled(True)
        current = engine_status(self.engine) if ok else None
        state = current.state if current is not None else "err"
        self._set_engine_status(current.status_text if current is not None else status, state)
        color = C["warn"] if state == "warn" else (C["ok"] if ok else C["err"])
        self.lbl_engine.setStyleSheet(f"color:{color};padding:0 8px;")
        self._log("INFO" if ok else "ERROR", f"Model load: {status}")
        if not ok:
            show_llm_error_dialog(self, status, source="Model loader")
        if hasattr(self, "model_list"):
            self._refresh_model_list()
        # Keep pipeline tab's engine reference up-to-date
        if hasattr(self, "pipeline_tab"):
            self.pipeline_tab.update_engine(self.engine)
        self._notify_labs()
        self._last_context_snapshot = {}
        self._update_ctx_bar()
        self._flush_deferred_send(ok)

    def _unload_chat_model(self):
        if not self._can_restart_primary_engine("unloading the model"):
            return
        if self._api_engine:
            self._api_engine.shutdown()
            self._api_engine = None
        if self.engine:
            self.engine.shutdown()
        self.engine = LlamaEngine()
        self._deferred_send = None
        self.lbl_family.setText("")
        self._set_engine_status("Model not loaded", "idle")
        self._last_context_snapshot = {}
        self._update_ctx_bar()
        self._refresh_model_list()
        if hasattr(self, "pipeline_tab"):
            self.pipeline_tab.update_engine(self.engine)
        self._notify_labs()
        self._log("INFO", "Chat model unloaded.")

    def _flush_deferred_send(self, ok: bool):
        if not ok or not self._deferred_send:
            return
        text, ref_ctx, ref_images = self._deferred_send
        self._deferred_send = None
        QTimer.singleShot(0, lambda: self._on_send_with_refs(text, ref_ctx, ref_images))

    def _reload_model(self):
        if not self._can_restart_primary_engine("reloading the model"):
            return
        self.engine.shutdown()
        QTimer.singleShot(500, self._start_model_load)

    def _is_coding_prompt(self, text: str) -> bool:
        tl = text.lower()
        return self._force_coding_mode or any(k in tl for k in self._CODING_KEYWORDS)

    def _can_use_pipeline(self, text: str) -> bool:
        """Return True if pipeline mode should be used for this prompt."""
        return (
            PARALLEL_PREFS.enabled and
            PARALLEL_PREFS.pipeline_mode and
            self._is_coding_prompt(text) and
            self.reasoning_engine is not None and self.reasoning_engine.is_loaded and
            self.coding_engine is not None and self.coding_engine.is_loaded
        )

    def _active_engine_for(self, text: str):
        if self._is_coding_prompt(text) and \
                self.coding_engine and self.coding_engine.is_loaded:
            return self.coding_engine
        selected = self.input_bar.selected_model if hasattr(self, "input_bar") else ""
        if is_api_model_ref(selected):
            if self._api_engine and self._api_engine.is_loaded and self._api_engine.model_path == selected:
                return self._api_engine
            return None
        if selected and self.engine.is_loaded and getattr(self.engine, "model_path", "") != selected:
            return None
        return self.engine

    def _on_tab_changed(self, idx: int):
        w = self.tabs.widget(idx)
        if not w:
            return
        if hasattr(self, "left_sidebar_stack"):
            if hasattr(self, "dev_tab") and w is self.dev_tab:
                self.left_sidebar_stack.setCurrentWidget(self.dev_sidebar)
            else:
                self.left_sidebar_stack.setCurrentWidget(self.sidebar)
        # Cover exactly the newly visible tab page
        self._tab_overlay.setGeometry(w.geometry())
        self._tab_overlay.raise_()
        self._tab_overlay.show()
        anim = QPropertyAnimation(self._tab_overlay, b"alpha", self)
        anim.setDuration(180)
        anim.setStartValue(220)
        anim.setEndValue(0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self._tab_overlay.hide)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _on_api_model_loaded(self, api_engine: ApiEngine):
        """Called when ApiModelsTab successfully verifies an API model."""
        if self.engine and self.engine.is_loaded:
            if not self._can_restart_primary_engine("switching to the API model"):
                try:
                    api_engine.shutdown()
                except Exception:
                    pass
                return
            self.engine.shutdown()
        self._api_engine = api_engine
        cfg = api_engine._config
        if cfg and hasattr(self, "input_bar"):
            ref = api_model_ref(cfg.name)
            idx = self.input_bar.model_combo.findData(ref)
            if idx == -1:
                self.input_bar.model_combo.addItem(api_model_label(cfg), ref)
                idx = self.input_bar.model_combo.findData(ref)
            self.input_bar.model_combo.blockSignals(True)
            self.input_bar.model_combo.setCurrentIndex(idx)
            self.input_bar.model_combo.blockSignals(False)
            self.input_bar._update_family_badge()
        status = engine_status(api_engine)
        self._set_engine_status(status.status_text, status.state)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self.lbl_family.setText("API  ·  provider output limit")
        self._log("INFO", f"API model loaded: {api_engine.status_text}")
        # Update pipeline tab to use api engine for pipeline blocks
        if hasattr(self, "pipeline_tab"):
            self.pipeline_tab.update_engine(api_engine)
        self._notify_labs()
        self._last_context_snapshot = {}
        self._update_ctx_bar()
        self._flush_deferred_send(True)

    def _start_api_model_load(self, api_ref: str):
        if self._loader_is_running("_loader") or self._loader_is_running("_api_loader"):
            self._log("WARN", "Model load already in progress; ignoring duplicate API load request.")
            return
        cfg = getapi_registry().get_by_ref(api_ref)
        if cfg is None:
            self._set_engine_status("API config missing", "err")
            self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
            self._log("ERROR", f"API config not found: {api_ref}")
            show_llm_error_dialog(self, f"API config not found: {api_ref}", source="API model loader")
            return
        if self._api_engine and self._api_engine.is_loaded and self._api_engine.model_path == api_ref:
            self._flush_deferred_send(True)
            return
        if self.engine and self.engine.is_loaded:
            if not self._can_restart_primary_engine("switching to the API model"):
                return
            self.engine.shutdown()
        self._set_engine_status("Connecting API model...", "loading")
        self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")
        self.lbl_family.setText(f"API  ·  {cfg.provider}  ·  provider output limit")
        self._api_loader = ApiLoaderThread(cfg)
        self._api_loader.finished.connect(self._on_api_combo_load_done)
        self._api_loader.start()

    def _on_api_combo_load_done(self, ok: bool, status: str, engine):
        if ok and engine:
            self._on_api_model_loaded(engine)
            return
        self._set_engine_status(status, "err")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
        self._log("ERROR", status)
        show_llm_error_dialog(self, status, source="API model loader")
        self._notify_labs()
        self._flush_deferred_send(False)
