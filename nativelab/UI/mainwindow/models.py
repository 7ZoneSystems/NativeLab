"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *


class ModelManagementMixin:
    def _auto_load_parallel_engines(self):
        """Load all engines whose roles are in the auto_load list."""
        for role in PARALLEL_PREFS.auto_load_roles:
            models = get_model_registry().all_models()
            for m in models:
                if m.get("role") == role and is_model_ref_valid(m["path"]):
                    self._start_role_engine_load(role, m["path"])
                    break

    def _start_role_engine_load(self, role: str, path: str):
        attr        = f"{role}_engine"
        loader_attr = f"_loader_{role}"

        # Cancel any in-progress loader for this role to avoid double-loading
        old_loader = getattr(self, loader_attr, None)
        if old_loader and old_loader.isRunning():
            self._log("WARN", f"Cancelling previous {role} loader before new one starts")
            if not stop_worker(old_loader, LONG_TIMEOUT_MS, abort=False, cancel=False):
                self._log("WARN", f"{role.capitalize()} loader is still running.")
                return

        # Shutdown any existing engine for this role before creating a fresh one
        old_eng = getattr(self, attr, None)
        if old_eng and old_eng.is_loaded:
            self._log("INFO", f"Shutting down existing {role} engine before reload")
            old_eng.shutdown()

        # Always create a brand-new engine instance to avoid state leakage
        new_eng = LlamaEngine()
        setattr(self, attr, new_eng)

        cfg    = get_model_registry().get_config(path)
        loader = ModelLoaderThread(new_eng, path, cfg.ctx, cfg.threads)
        loader.log.connect(self._log)
        loader.finished.connect(
            lambda ok, st, r=role, n=model_ref_display_name(path):
            self._on_role_engine_loaded(ok, st, r, n, None))
        loader.start()
        setattr(self, loader_attr, loader)

        # Give immediate visual feedback so the user sees something changed
        self._refresh_engine_status()
        self._log("INFO", f"Loading {role} engine: {model_ref_display_name(path)}")

    def _build_models_tab(self) -> QWidget:
        outer = QWidget()
        outer_l = QVBoxLayout()
        outer_l.setContentsMargins(0, 0, 0, 0); outer_l.setSpacing(0)
        outer.setLayout(outer_l)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("chat_scroll")

        w = QWidget()
        w.setObjectName("chat_container")
        root = QVBoxLayout()
        root.setContentsMargins(20, 18, 20, 18); root.setSpacing(0)
        w.setLayout(root); scroll.setWidget(w); outer_l.addWidget(scroll)

        def _section_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color:{C['txt']};font-size:12px;font-weight:bold;"
                              f"letter-spacing:0.5px;padding:0;margin-bottom:2px;")
            return lbl

        def _card(layout) -> QFrame:
            card = QFrame()
            card.setObjectName("tab_card")
            card.setLayout(layout); return card

        # ── header ───────────────────────────────────────────────────────────
        hdr = QLabel("Model Manager")
        set_label_icon(hdr, "models", "Model Manager", 18)
        hdr.setStyleSheet(f"color:{C['txt']};font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        note = QLabel("Add models, assign roles, and configure the reasoning→coding pipeline.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C['txt2']};font-size:11px;margin-bottom:14px;")
        root.addWidget(note)

        # ── MODEL LIBRARY ─────────────────────────────────────────────────────
        root.addWidget(_section_label("MODEL LIBRARY"))
        list_card_l = QVBoxLayout()
        list_card_l.setContentsMargins(0, 0, 0, 0); list_card_l.setSpacing(0)

        # legend
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(10, 8, 10, 6); legend_row.setSpacing(14)
        for role, label in ROLE_ICONS.items():
            pill = QLabel(label)
            set_label_icon(pill, role, label, 14)
            pill.setStyleSheet(f"color:{C['txt2']};font-size:10px;"
                               f"background:{C['bg2']};border-radius:4px;padding:2px 6px;")
            legend_row.addWidget(pill)
        legend_row.addStretch()
        list_card_l.addLayout(legend_row)

        self.model_list = QListWidget()
        self.model_list.setObjectName("model_list")
        self.model_list.setIconSize(icon_size(18))
        self.model_list.setMinimumHeight(150)
        self.model_list.setMaximumHeight(240)
        self.model_list.currentItemChanged.connect(self._on_model_list_select)
        list_card_l.addWidget(self.model_list)

        btn_strip = QHBoxLayout()
        btn_strip.setContentsMargins(10, 8, 10, 8); btn_strip.setSpacing(8)
        self.btn_browse_model = QPushButton("Browse GGUF...")
        self.btn_add_ollama = QPushButton("Add Ollama")
        self.btn_add_hf = QPushButton("Add HF")
        self.btn_load_primary = QPushButton("Load Selected")
        self.btn_load_primary.setObjectName("btn_send")
        self.btn_remove_model = QPushButton("Remove")
        self.btn_remove_model.setObjectName("btn_stop")
        set_button_icon(self.btn_browse_model, "folder-open", "Browse GGUF...")
        set_button_icon(self.btn_add_ollama, "ollama", "Add Ollama")
        set_button_icon(self.btn_add_hf, "huggingface", "Add HF")
        set_button_icon(self.btn_load_primary, "zap", "Load Selected")
        set_button_icon(self.btn_remove_model, "delete", "Remove")
        for b in (self.btn_browse_model, self.btn_add_ollama, self.btn_add_hf, self.btn_load_primary, self.btn_remove_model):
            b.setFixedHeight(30); btn_strip.addWidget(b)
        btn_strip.addStretch()
        list_card_l.addLayout(btn_strip)
        root.addWidget(_card(list_card_l))
        root.addSpacing(14)
        self.btn_browse_model.clicked.connect(self._browse_add_model)
        self.btn_add_ollama.clicked.connect(self._add_ollama_model)
        self.btn_add_hf.clicked.connect(self._add_hf_model)
        self.btn_load_primary.clicked.connect(self._load_selected_as_primary)
        self.btn_remove_model.clicked.connect(self._remove_selected_model)

        # ── PER-MODEL PARAMETERS ──────────────────────────────────────────────
        root.addWidget(_section_label("PER-MODEL PARAMETERS"))
        hint = QLabel("Select a model above to edit its parameters.")
        hint.setStyleSheet(f"color:{C['txt2']};font-size:10px;margin-bottom:6px;")
        root.addWidget(hint)

        cfg_card_l = QVBoxLayout()
        cfg_card_l.setContentsMargins(16, 14, 16, 14); cfg_card_l.setSpacing(10)
        LW = 110

        def _field_row(label_text: str, *widgets, stretch=True) -> QHBoxLayout:
            row = QHBoxLayout(); row.setSpacing(8)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(LW)
            lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
            row.addWidget(lbl)
            for ww in widgets: row.addWidget(ww)
            if stretch: row.addStretch()
            return row

        def _desc_row(label_text: str, widget, desc: str) -> QHBoxLayout:
            row = _field_row(label_text, widget, stretch=False)
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
            row.addWidget(desc_lbl, 1)
            return row

        # Detected family banner (read-only, auto-set)
        self.cfg_family_lbl = QLabel("-")
        self.cfg_family_lbl.setStyleSheet(
            f"color:{C['acc2']};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;"
        )
        cfg_card_l.addLayout(_field_row("Detected Family:", self.cfg_family_lbl))

        # Quant type banner
        self.cfg_quant_lbl = QLabel("-")
        self.cfg_quant_lbl.setStyleSheet(
            f"color:{C['ok']};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;"
        )
        cfg_card_l.addLayout(_field_row("Quant Type:", self.cfg_quant_lbl))

        dv0 = QFrame(); dv0.setFrameShape(QFrame.Shape.HLine)
        dv0.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv0)

        self.cfg_role = QComboBox()
        self.cfg_role.setMinimumWidth(200); self.cfg_role.setFixedHeight(28)
        self.cfg_role.setIconSize(icon_size(16))
        for r in MODEL_ROLES:
            self.cfg_role.addItem(role_icon(r), ROLE_ICONS[r], r)
        cfg_card_l.addLayout(_field_row("Role:", self.cfg_role))

        dv1 = QFrame(); dv1.setFrameShape(QFrame.Shape.HLine)
        dv1.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv1)

        self.cfg_threads = QLineEdit(str(DEFAULT_THREADS()))
        self.cfg_threads.setFixedWidth(64); self.cfg_threads.setFixedHeight(28)
        self.cfg_threads.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Threads:", self.cfg_threads))

        self.cfg_ctx = QLineEdit(str(DEFAULT_CTX()))
        self.cfg_ctx.setFixedWidth(80); self.cfg_ctx.setFixedHeight(28)
        self.cfg_ctx.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Context (tokens):", self.cfg_ctx))

        self.cfg_temp = QLineEdit("0.7")
        self.cfg_temp.setFixedWidth(64); self.cfg_temp.setFixedHeight(28)
        self.cfg_temp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Temperature:", self.cfg_temp))

        self.cfg_topp = QLineEdit("0.9")
        self.cfg_topp.setFixedWidth(64); self.cfg_topp.setFixedHeight(28)
        self.cfg_topp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Top-P:", self.cfg_topp))

        self.cfg_rep = QLineEdit("1.1")
        self.cfg_rep.setFixedWidth(64); self.cfg_rep.setFixedHeight(28)
        self.cfg_rep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Repeat Penalty:", self.cfg_rep))

        self.cfg_npred = QLineEdit(str(DEFAULT_N_PRED))
        self.cfg_npred.setFixedWidth(80); self.cfg_npred.setFixedHeight(28)
        self.cfg_npred.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Max Tokens:", self.cfg_npred))

        self.btn_advanced_params = QPushButton("Advanced")
        self.btn_advanced_params.setCheckable(True)
        self.btn_advanced_params.setFixedHeight(28)
        set_button_icon(self.btn_advanced_params, "square-chevron-right", "Advanced")
        self.btn_advanced_params.clicked.connect(self._toggle_advanced_params)
        cfg_card_l.addWidget(self.btn_advanced_params)

        self.advanced_params_body = QWidget()
        advanced_l = QVBoxLayout(self.advanced_params_body)
        advanced_l.setContentsMargins(10, 8, 10, 8)
        advanced_l.setSpacing(8)

        self.cfg_topk = QLineEdit("40")
        self.cfg_topk.setFixedWidth(64); self.cfg_topk.setFixedHeight(28)
        self.cfg_topk.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cfg_topk.setToolTip("Top-K sampler cutoff. 0 disables top-k where the backend supports it.")
        advanced_l.addLayout(_desc_row("Top-K:", self.cfg_topk, "Limits token choices to the top ranked K tokens; 0 disables it when supported."))

        self.cfg_minp = QLineEdit("0.0")
        self.cfg_minp.setFixedWidth(64); self.cfg_minp.setFixedHeight(28)
        self.cfg_minp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cfg_minp.setToolTip("Min-P sampler cutoff. 0 keeps the backend default.")
        advanced_l.addLayout(_desc_row("Min-P:", self.cfg_minp, "Drops tokens below a relative probability floor; 0 keeps backend default."))

        self.cfg_typicalp = QLineEdit("1.0")
        self.cfg_typicalp.setFixedWidth(64); self.cfg_typicalp.setFixedHeight(28)
        self.cfg_typicalp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cfg_typicalp.setToolTip("Typical-P sampler cutoff. 1 keeps the backend default.")
        advanced_l.addLayout(_desc_row("Typical-P:", self.cfg_typicalp, "Prefers tokens with typical information content; 1 keeps backend default."))

        self.cfg_seed = QLineEdit("-1")
        self.cfg_seed.setFixedWidth(80); self.cfg_seed.setFixedHeight(28)
        self.cfg_seed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cfg_seed.setToolTip("Seed for deterministic sampling. -1 keeps random/default backend behavior.")
        advanced_l.addLayout(_desc_row("Seed:", self.cfg_seed, "Use a fixed non-negative seed for repeatable output; -1 keeps random behavior."))

        self.advanced_params_body.setVisible(False)
        cfg_card_l.addWidget(self.advanced_params_body)

        self.cfg_param_warn = QLabel("")
        self.cfg_param_warn.setWordWrap(True)
        self.cfg_param_warn.setStyleSheet(
            f"color:{C['warn']};font-size:11px;padding:4px 8px;"
            f"background:{palette_rgba(C, 'warn', 0.10)};"
            f"border-radius:5px;border:1px solid {palette_rgba(C, 'warn', 0.28)};"
        )
        self.cfg_param_warn.setVisible(False)
        cfg_card_l.addWidget(self.cfg_param_warn)

        dv2 = QFrame(); dv2.setFrameShape(QFrame.Shape.HLine)
        dv2.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv2)

        save_row = QHBoxLayout(); save_row.setSpacing(8)
        self.btn_save_cfg = QPushButton("Save Parameters")
        set_button_icon(self.btn_save_cfg, "save", "Save Parameters")
        self.btn_save_cfg.setFixedHeight(30)
        self.btn_save_cfg.clicked.connect(self._save_model_config)
        save_row.addWidget(self.btn_save_cfg)
        save_row.addStretch()
        cfg_card_l.addLayout(save_row)
        for wf in (
            self.cfg_ctx, self.cfg_threads, self.cfg_temp, self.cfg_topp,
            self.cfg_topk, self.cfg_minp, self.cfg_typicalp, self.cfg_rep,
            self.cfg_seed, self.cfg_npred,
        ):
            wf.textChanged.connect(self._check_param_warnings)
        root.addWidget(_card(cfg_card_l))
        root.addSpacing(14)

        # ── ACTIVE ENGINES ────────────────────────────────────────────────────
        root.addWidget(_section_label("ACTIVE ENGINES"))
        eng_card_l = QVBoxLayout()
        eng_card_l.setContentsMargins(0, 0, 0, 0); eng_card_l.setSpacing(0)
        self.engine_status_list = QListWidget()
        self.engine_status_list.setFixedHeight(130)
        self.engine_status_list.setObjectName("engine_list")
        self.engine_status_list.setIconSize(icon_size(18))
        eng_card_l.addWidget(self.engine_status_list)
        eng_btn_strip = QHBoxLayout()
        eng_btn_strip.setContentsMargins(10, 8, 10, 8); eng_btn_strip.setSpacing(8)
        self.btn_load_role_engine = QPushButton("Load Engine for Role")
        set_button_icon(self.btn_load_role_engine, "zap", "Load Engine for Role")
        self.btn_load_role_engine.setFixedHeight(30)
        self.btn_load_role_engine.clicked.connect(self._load_engine_for_selected)
        self.btn_unload_all = QPushButton("Unload All")
        set_button_icon(self.btn_unload_all, "eject", "Unload All")
        self.btn_unload_all.setFixedHeight(30)
        self.btn_unload_all.clicked.connect(self._unload_all_engines)
        eng_btn_strip.addWidget(self.btn_load_role_engine)
        eng_btn_strip.addWidget(self.btn_unload_all)
        eng_btn_strip.addStretch()
        eng_card_l.addLayout(eng_btn_strip)
        root.addWidget(_card(eng_card_l))
        root.addSpacing(14)

        # ── PARALLEL LOADING ──────────────────────────────────────────────────
        root.addWidget(_section_label("PARALLEL LOADING & PIPELINE"))
        par_card_l = QVBoxLayout()
        par_card_l.setContentsMargins(0, 0, 0, 0)
        self.parallel_panel = ParallelLoadingDialog()
        self.parallel_panel.settings_changed.connect(self._on_parallel_settings_changed)
        par_card_l.addWidget(self.parallel_panel)
        root.addWidget(_card(par_card_l))
        root.addSpacing(14)

        # hidden compat stubs
        self.reasoning_status      = QLabel()
        self.summary_engine_status = QLabel()
        self.coding_engine_status  = QLabel()
        for lbl in (self.reasoning_status, self.summary_engine_status, self.coding_engine_status):
            lbl.setVisible(False); root.addWidget(lbl)

        root.addStretch()
        self._refresh_model_list()
        return outer

    def _on_parallel_settings_changed(self):
        pipeline_on = bool(
            PARALLEL_PREFS.enabled and
            PARALLEL_PREFS.pipeline_mode and
            self.reasoning_engine is not None and self.reasoning_engine.is_loaded and
            self.coding_engine is not None and self.coding_engine.is_loaded
        )
        self.input_bar.set_pipeline_mode(pipeline_on)
        self._log("INFO",
            f"Parallel settings: enabled={PARALLEL_PREFS.enabled}, "
            f"pipeline={PARALLEL_PREFS.pipeline_mode}, "
            f"auto_roles={PARALLEL_PREFS.auto_load_roles}")

    def _toggle_advanced_params(self):
        expanded = bool(self.btn_advanced_params.isChecked())
        self.advanced_params_body.setVisible(expanded)
        set_button_icon(
            self.btn_advanced_params,
            "square-chevron-down" if expanded else "square-chevron-right",
            "Advanced",
        )

    def _check_param_warnings(self):
        warnings = []
        try:
            ctx = int(self.cfg_ctx.text())
            if ctx > 24576:
                warnings.append(f"Context {ctx:,} tokens is very high")
            elif ctx > 16384:
                warnings.append(f"Context {ctx:,} tokens is high")
        except ValueError:
            pass
        try:
            threads = int(self.cfg_threads.text())
            import multiprocessing
            ncpu = multiprocessing.cpu_count()
            if threads > ncpu:
                warnings.append(f"{threads} threads exceeds {ncpu} logical CPUs")
        except (ValueError, NotImplementedError):
            pass
        try:
            temp = float(self.cfg_temp.text())
            if temp > 1.5:
                warnings.append("Temperature > 1.5")
            elif temp < 0.05:
                warnings.append("Temperature near 0")
        except ValueError:
            pass
        try:
            top_p = float(self.cfg_topp.text())
            if not (0.0 < top_p <= 1.0):
                warnings.append("Top-P should be > 0 and <= 1")
        except ValueError:
            pass
        try:
            top_k = int(self.cfg_topk.text())
            if top_k < 0:
                warnings.append("Top-K cannot be negative")
            elif top_k > 200:
                warnings.append(f"Top-K {top_k} is unusually high")
        except ValueError:
            pass
        try:
            min_p = float(self.cfg_minp.text())
            if not (0.0 <= min_p <= 1.0):
                warnings.append("Min-P should be between 0 and 1")
        except ValueError:
            pass
        try:
            typical_p = float(self.cfg_typicalp.text())
            if not (0.0 < typical_p <= 1.0):
                warnings.append("Typical-P should be > 0 and <= 1")
        except ValueError:
            pass
        try:
            npred = int(self.cfg_npred.text())
            if npred < 0:
                warnings.append("Max tokens cannot be negative")
        except ValueError:
            pass
        if warnings:
            self.cfg_param_warn.setText("\n".join(warnings))
            self.cfg_param_warn.setVisible(True)
        else:
            self.cfg_param_warn.setVisible(False)

    def _on_model_list_select(self, item: Optional[QListWidgetItem], _=None):
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path: return

        # Update family / quant labels
        cfg = get_model_registry().get_config(path)
        payload = model_ref_payload(path) or path
        fam   = detect_model_family(payload)
        quant = cfg.quant_type
        ql, qcolor = quant_info(quant)
        vi = cfg.vision_info
        vision_txt = f"  ·  VLM: {vi.label}" if vi.is_vision else ""
        mmproj = detect_mmproj_for_model(path) if vi.is_vision and not is_external_model_ref(path) else ""
        backend_txt = f"  ·  {cfg.backend}"
        self.cfg_family_lbl.setText(
            f"{fam.name}  (template: {fam.template}){backend_txt}{vision_txt}"
            + (f"  ·  mmproj: {Path(mmproj).name}" if mmproj else "")
        )
        self.cfg_quant_lbl.setText(f"{quant}  ·  {ql}")
        self.cfg_quant_lbl.setStyleSheet(
            f"color:{qcolor};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;")

        idx = self.cfg_role.findData(cfg.role)
        self.cfg_role.setCurrentIndex(max(idx, 0))
        self.cfg_threads.setText(str(cfg.threads))
        if path == getattr(self.engine, "model_path", "") and hasattr(self, "ctx_slider"):
            self.cfg_ctx.setText(str(self.ctx_slider.value()))
        else:
            self.cfg_ctx.setText(str(cfg.ctx))
        self.cfg_temp.setText(str(cfg.temperature))
        self.cfg_topp.setText(str(cfg.top_p))
        self.cfg_topk.setText(str(getattr(cfg, "top_k", 40)))
        self.cfg_minp.setText(str(getattr(cfg, "min_p", 0.0)))
        self.cfg_typicalp.setText(str(getattr(cfg, "typical_p", 1.0)))
        self.cfg_rep.setText(str(cfg.repeat_penalty))
        self.cfg_seed.setText(str(getattr(cfg, "seed", -1)))
        self.cfg_npred.setText(str(cfg.n_predict))
        self._check_param_warnings()

    def _save_model_config(self):
        item = self.model_list.currentItem()
        if not item:
            self._log("WARN", "No model selected"); return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path: return
        try:
            ctx = int(self.cfg_ctx.text()); threads = int(self.cfg_threads.text())
            temp = float(self.cfg_temp.text()); topp = float(self.cfg_topp.text())
            topk = int(self.cfg_topk.text()); minp = float(self.cfg_minp.text())
            typicalp = float(self.cfg_typicalp.text())
            rep = float(self.cfg_rep.text()); seed = int(self.cfg_seed.text())
            npred = int(self.cfg_npred.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Parameter", "One or more fields are invalid."); return
        if not (0.0 < topp <= 1.0 and topk >= 0 and 0.0 <= minp <= 1.0 and 0.0 < typicalp <= 1.0 and rep > 0 and npred >= 0):
            QMessageBox.warning(
                self,
                "Invalid Parameter",
                "Check sampler ranges: Top-P and Typical-P must be > 0 and <= 1; "
                "Min-P must be 0..1; Top-K and Max Tokens cannot be negative; "
                "Repeat Penalty must be positive.",
            )
            return

        dangers = []
        if ctx > 24576:  dangers.append(f"Context = {ctx:,} tokens (very high)")
        if threads > 32: dangers.append(f"Threads = {threads}")
        if temp > 2.0:   dangers.append(f"Temperature = {temp}")
        if topk > 200:   dangers.append(f"Top-K = {topk}")
        if dangers:
            msg = "High-compute parameters:\n\n" + "\n".join(f"  • {d}" for d in dangers) + "\n\nSave?"
            if QMessageBox.warning(self, "Confirm", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return

        payload = model_ref_payload(path) or path
        fam = detect_model_family(payload)
        vi = detect_vision_model(payload)
        cfg = ModelConfig(
            path=path, role=self.cfg_role.currentData() or "general",
            backend=model_ref_backend(path), vision=vi.is_vision,
            threads=threads, ctx=ctx, temperature=temp, top_p=topp,
            top_k=topk, min_p=minp, typical_p=typicalp,
            repeat_penalty=rep, seed=seed, n_predict=npred,
            family=fam.family,
        )
        get_model_registry().set_config(path, cfg)
        self._refresh_model_list()
        self._log("INFO", f"Saved config for {model_ref_display_name(path)}: family={fam.name}, "
                          f"role={cfg.role}, ctx={cfg.ctx}")

        if path == getattr(self.engine, "model_path", ""):
            if ctx != self.ctx_slider.value():
                self.ctx_slider.blockSignals(True)
                self.ctx_slider.setValue(ctx)
                self.ctx_input.setText(str(ctx))
                self.ctx_slider.blockSignals(False)
                self.current_ctx = ctx
                self._ctx_reload_timer.start(500)

    def _load_engine_for_selected(self):
        item = self.model_list.currentItem()
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not is_model_ref_valid(path):
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{path}"); return
        cfg = get_model_registry().get_config(path)
        role = cfg.role

        if PARALLEL_PREFS.enabled and role != "general":
            # Warn if loading a second/third model
            n_loaded = sum(1 for r in ("reasoning","summarization","coding","secondary")
                           if getattr(self, f"{r}_engine", None) and
                           getattr(self, f"{r}_engine").is_loaded)
            if n_loaded >= 1:
                size_mb = get_model_registry().get_config(path).size_mb
                ram_est = max(size_mb * 1.1 / 1000, 1)
                QMessageBox.information(
                    self, "Parallel RAM Usage",
                    f"Loading an additional engine (~{ram_est:.1f} GB).\n"
                    f"Total parallel engines after this: {n_loaded + 2}\n\n"
                    f"Ensure you have sufficient free RAM."
                )

        if role == "general":
            if not self._can_restart_primary_engine("loading the selected model"):
                return
            idx = self.input_bar.model_combo.findData(path)
            if idx == -1:
                self.input_bar.model_combo.addItem(model_ref_display_name(path), path)
                idx = self.input_bar.model_combo.findData(path)
            self.input_bar.model_combo.setCurrentIndex(idx)
            self.engine.shutdown()
            QTimer.singleShot(200, self._start_model_load)
        elif role in ("reasoning", "summarization", "coding", "secondary"):
            # Disable button while loading to prevent rapid re-clicks causing races
            self.btn_load_role_engine.setEnabled(False)
            self.btn_load_role_engine.setText("Loading...")

            attr        = f"{role}_engine"
            loader_attr = f"_loader_{role}"

            # Cancel any in-progress loader for this role
            old_loader = getattr(self, loader_attr, None)
            if old_loader and old_loader.isRunning():
                self._log("WARN", f"Cancelling stale {role} loader")
                if not stop_worker(old_loader, LONG_TIMEOUT_MS, abort=False, cancel=False):
                    self._log("WARN", f"{role.capitalize()} loader is still running.")
                    return

            # Cleanly shut down any already-running engine for this role
            old_eng = getattr(self, attr, None)
            if old_eng and old_eng.is_loaded:
                self._log("INFO", f"Shutting down existing {role} engine")
                old_eng.shutdown()

            new_eng = LlamaEngine()
            setattr(self, attr, new_eng)
            cfg = get_model_registry().get_config(path)

            def _on_loaded_reenable(ok, st, r=role, n=model_ref_display_name(path)):
                self._on_role_engine_loaded(ok, st, r, n, None)
                self.btn_load_role_engine.setEnabled(True)
                set_button_icon(self.btn_load_role_engine, "zap", "Load Engine for Role")

            loader = ModelLoaderThread(new_eng, path, cfg.ctx, cfg.threads)
            loader.log.connect(self._log)
            loader.finished.connect(_on_loaded_reenable)
            loader.start()
            setattr(self, loader_attr, loader)
            self._log("INFO", f"Loading {role} engine: {model_ref_display_name(path)}")

        self._refresh_engine_status()

    def _on_role_engine_loaded(self, ok: bool, status: str, role: str,
                                name: str, lbl):
        color = C["ok"] if ok else C["err"]
        role_label = ROLE_ICONS.get(role, role.capitalize())
        state = "Loaded" if ok else "Failed"
        text  = f"{role_label}:  {state}  {name}"
        if lbl:
            try:
                lbl.setText(text)
                lbl.setStyleSheet(f"color:{color};font-size:11px;")
            except RuntimeError:
                pass
        self._log("INFO" if ok else "ERROR", f"{role} engine: {status}")
        self._refresh_engine_status()
        # Update pipeline badge in input bar
        self._on_parallel_settings_changed()

    def _refresh_engine_status(self):
        self.engine_status_list.clear()
        engines = {"General (primary)": self.engine}
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng:
                engines[role.capitalize()] = eng
        for role_name, eng in engines.items():
            status = engine_status(eng, role=role_name)
            item = QListWidgetItem(status.manager_label)
            item.setIcon(status_icon(status.state))
            item.setForeground(QColor(C["ok"] if status.is_loaded else C["txt2"]))
            self.engine_status_list.addItem(item)

    def _unload_all_engines(self):
        if not self._can_restart_primary_engine("unloading all engines"):
            return
        if QMessageBox.question(
            self, "Unload All Engines",
            "Unload all model engines?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng:
                eng.shutdown()
                setattr(self, f"{role}_engine", None)
        if self._api_engine and self._api_engine.is_loaded:
            self._api_engine.shutdown()
            self._api_engine = None
        self.engine.shutdown()
        self.engine = LlamaEngine()
        self._refresh_engine_status()
        self._on_parallel_settings_changed()
        self._log("INFO", "All engines unloaded.")
        self._notify_labs()

    def _refresh_model_list(self):
        self.model_list.clear()
        active = getattr(self.engine, "model_path", "")
        for m in get_model_registry().all_models():
            backend = m.get("backend", "llama_cpp")
            if backend == "ollama":
                source_label = "Ollama"
            elif backend == "hf_transformers":
                source_label = "HF"
            else:
                source_label = "Custom" if m["source"] == "custom" else "Bundled"
            role_label = ROLE_ICONS.get(m.get("role", "general"), "General")
            ql, qc    = quant_info(m.get("quant", ""))
            vision = f"  VLM: {m.get('vision_label') or 'vision'}" if m.get("vision") else ""
            mmproj = " · mmproj" if m.get("mmproj") else ""
            label = (f"{source_label}  {role_label} [{m.get('role','general'):<14}]  "
                     f"{m['name']}   ({m['size_mb']} MB)  "
                     f"[{m.get('family','?')}·{m.get('quant','?')}·{ql}{vision}{mmproj}]")
            if m["path"] == active: label += "  Loaded"
            item = QListWidgetItem(label)
            if m["path"] == active:
                item.setIcon(status_icon("ok"))
            else:
                item.setIcon(icon("vision") if m.get("vision") else role_icon(m.get("role", "general")))
            item.setData(Qt.ItemDataRole.UserRole, m["path"])
            if m["path"] == active:
                item.setForeground(QColor(C["ok"]))
            self.model_list.addItem(item)
        if hasattr(self, "engine_status_list"):
            self._refresh_engine_status()

    def _sync_input_bar_combo(self):
        cur = self.input_bar.model_combo.currentData()
        self.input_bar.model_combo.blockSignals(True)
        self.input_bar.model_combo.clear()
        for m in get_model_registry().all_models():
            self.input_bar.model_combo.addItem(m["name"], m["path"])
        for cfg in getapi_registry().all():
            self.input_bar.model_combo.addItem(api_model_label(cfg), api_model_ref(cfg.name))
        idx = self.input_bar.model_combo.findData(cur)
        self.input_bar.model_combo.setCurrentIndex(max(idx, 0))
        self.input_bar.model_combo.blockSignals(False)
        self.input_bar._update_family_badge()
        if hasattr(self, "model_list"):
            self._refresh_model_list()

    def _browse_add_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", str(Path.home()),
            "GGUF Models (*.gguf);;All Files (*)")
        if not path: return
        get_model_registry().add(path)
        fam   = detect_model_family(path)
        quant = detect_quant_type(path)
        vi    = detect_vision_model(path)
        ql, _ = quant_info(quant)
        self._refresh_model_list()
        self._sync_input_bar_combo()
        self._log("INFO",
            f"Added model: {Path(path).name}  →  {fam.name}  ·  {quant}  ·  {ql}"
            + (f"  ·  VLM: {vi.label}" if vi.is_vision else ""))

    def _fetch_ollama_model_names(self) -> list:
        import urllib.request
        import urllib.error
        host = normalize_ollama_host(str(APP_CONFIG.get("ollama_host", "http://127.0.0.1:11434")))
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=LONG_TIMEOUT_SECONDS) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
            rows = data.get("models") or []
            return sorted({str(row.get("name") or row.get("model") or "").strip() for row in rows if row.get("name") or row.get("model")})
        except Exception as exc:
            self._log("ERROR", normalize_ollama_exception(exc, host, action="list models from"))
            return []

    def _add_ollama_model(self):
        names = self._fetch_ollama_model_names()
        if names:
            chosen, ok = QInputDialog.getItem(
                self, "Add Ollama Model",
                "Select an installed Ollama model:",
                names, 0, False)
        else:
            chosen, ok = QInputDialog.getText(
                self, "Add Ollama Model",
                "Ollama was not reachable or returned no models.\nEnter model name manually:")
        if not ok or not str(chosen).strip():
            return
        ref = make_ollama_model_ref(str(chosen).strip())
        get_model_registry().add(ref)
        self._refresh_model_list()
        self._sync_input_bar_combo()
        self._log("INFO", f"Added Ollama model: {model_ref_display_name(ref)}")

    def _add_hf_model(self):
        choice, ok = QInputDialog.getItem(
            self, "Add HF Transformers Model",
            "Source type:",
            ["Hugging Face repo id", "Local model directory"],
            0, False)
        if not ok:
            return
        if choice == "Local model directory":
            value = QFileDialog.getExistingDirectory(
                self, "Select HF model directory", str(MODELS_DIR))
        else:
            value, ok = QInputDialog.getText(
                self, "Add HF Transformers Model",
                "Repo id, for example: Qwen/Qwen2.5-0.5B-Instruct")
            if not ok:
                return
        if not str(value).strip():
            return
        ref = make_hf_model_ref(str(value).strip())
        get_model_registry().add(ref)
        self._refresh_model_list()
        self._sync_input_bar_combo()
        self._log("INFO", f"Added HF Transformers model: {model_ref_payload(ref)}")

    def _load_selected_as_primary(self):
        item = self.model_list.currentItem()
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not is_model_ref_valid(path):
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{path}"); return
        if not self._can_restart_primary_engine("loading the selected primary model"):
            return
        idx = self.input_bar.model_combo.findData(path)
        if idx == -1:
            self.input_bar.model_combo.addItem(model_ref_display_name(path), path)
            idx = self.input_bar.model_combo.findData(path)
        self.input_bar.model_combo.setCurrentIndex(idx)
        self.engine.shutdown()
        QTimer.singleShot(200, self._start_model_load)
        self._log("INFO", f"Loading primary model: {model_ref_display_name(path)}")

    def _remove_selected_model(self):
        item = self.model_list.currentItem()
        if not item: return
        get_model_registry().remove(item.data(Qt.ItemDataRole.UserRole))
        self._refresh_model_list()
        self._sync_input_bar_combo()
