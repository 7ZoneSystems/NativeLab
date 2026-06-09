"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *


class UiBuildMixin:
    def _build_ui(self):
        central = QWidget()
        ml = QHBoxLayout()
        ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(2)

        self.sidebar = SessionSidebar()
        self.sidebar.session_selected.connect(self._switch_session)
        self.sidebar.new_session.connect(self._new_session)
        self.sidebar.session_deleted.connect(self._delete_session)
        self.sidebar.session_renamed.connect(self._rename_session)
        self.sidebar.session_exported.connect(
            lambda sid: self._export_session(sid, "md"))
        self.left_sidebar_stack = QStackedWidget()
        self.left_sidebar_stack.setObjectName("session_sidebar")
        self.left_sidebar_stack.addWidget(self.sidebar)
        self.splitter.addWidget(self.left_sidebar_stack)

        self.tabs = QTabWidget()
        self._tab_overlay = FadeOverlay(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.tabBar().installEventFilter(self)

        # ── Chat tab (ChatModule) ──
        self.chat_module = ChatModule(
            session_id=self.active.id if self.active else "default")
        self.chat_area = self.chat_module.chat_area
        self.input_bar = self.chat_module.input_bar
        self.chat_module.send_requested.connect(self._on_send_with_refs)
        self.chat_module.stop_requested.connect(self._on_stop)
        self.chat_module.pdf_requested.connect(self._load_pdf)
        self.chat_module.clear_requested.connect(self._clear_chat)
        self.chat_module.multi_pdf_requested.connect(self._start_multi_pdf)
        self.input_bar.code_btn.toggled.connect(
            lambda chk: setattr(self, "_force_coding_mode", chk))
        self.input_bar.pipeline_run_requested.connect(self._on_pipeline_from_chat)
        self.input_bar.load_model_requested.connect(self._start_model_load)
        self.input_bar.unload_model_requested.connect(self._unload_chat_model)
        self.tabs.addTab(self.chat_module, icon("chat"), "Chat")

        # ── Models tab ──
        self.models_tab = self._build_models_tab()
        self.tabs.addTab(self.models_tab, icon("models"), "Models")

        # ── Config panel (opened from top-right settings button) ──
        self.config_dialog = None
        self._create_settings_pages()

        # ── Dev surfaces ──
        self.pipeline_tab = PipelineBuilderTab(self.engine)
        self.integrations_tab = IntegrationsTab(self._integration_endpoints)
        self.api_server_tab = ApiServerTab(self._lab_endpoints)
        self.log_console = LogConsole()
        self.labs_tab = LabsTab()
        self.mcp_tab = McpTab()
        self.skills_tab = SkillsTab()
        self.skills_tab.skills_changed.connect(lambda: self._log("INFO", "Skill library updated."))

        # ── API Models tab ──
        self.api_tab = ApiModelsTab()
        self.api_tab.api_model_loaded.connect(self._on_api_model_loaded)
        self.tabs.addTab(self.api_tab, icon("api"), "API Models")

        # ── Model Download tab ──
        self.download_tab = ModelDownloadTab()
        self.download_tab.hf_login_requested.connect(self._show_hf_login_tab)
        self.tabs.addTab(self.download_tab, icon("download"), "Download")

        # ── Dev tab ──
        self.dev_tab = self._build_dev_tab()
        self.tabs.addTab(self.dev_tab, icon("code"), "Dev")

        self.splitter.addWidget(self.tabs)
        self.splitter.setSizes([220, 1080])
        self.splitter.setStretchFactor(1, 1)
        ml.addWidget(self.splitter)
        central.setLayout(ml)
        self.setCentralWidget(central)

        self._apply_saved_view_state()
        self._wire_lab_endpoints()

    def _create_settings_pages(self):
        general_sections = [
            ("Memory & RAM", ["ram_watchdog_mb", "max_ram_chunks", "auto_spill_on_start"]),
            ("Reference Engine", ["chunk_index_size", "ref_top_k", "ref_max_context_chars"]),
            ("Model Defaults", ["default_threads", "default_ctx", "default_n_predict"]),
            ("Developer Mode", ["developer_mode"]),
        ]
        docs_sections = [
            ("Summarization", [
                "summary_chunk_chars", "summary_ctx_carry",
                "summary_n_pred_sect", "summary_n_pred_final",
                "pause_after_chunks",
            ]),
            ("Multi-PDF", ["multipdf_n_pred_sect", "multipdf_n_pred_final"]),
        ]
        hf_sections = [
            ("HF Transformers", [
                "hf_transformers_dir", "hf_token", "hf_revision",
                "hf_trust_remote_code", "hf_local_files_only", "hf_use_safetensors",
                "hf_torch_dtype", "hf_device_map", "hf_low_cpu_mem_usage",
                "hf_attn_implementation", "hf_max_memory", "hf_quantization",
            ]),
        ]
        ollama_sections = [
            ("Ollama", ["ollama_host", "ollama_keep_alive"]),
        ]

        self.config_tab = ConfigTab(
            title="General Settings",
            subtitle="Memory, references, defaults, paused jobs, and developer mode.",
            sections=general_sections,
            include_paused_jobs=True,
            icon_name="config",
        )
        self.config_tab.config_changed.connect(self._on_config_changed)
        self.config_tab.btn_resume_job.clicked.connect(self._resume_paused_job)

        self.docs_config_tab = ConfigTab(
            title="Document Settings",
            subtitle="Summarization and multi-PDF processing limits.",
            sections=docs_sections,
            include_paused_jobs=False,
            icon_name="docs",
        )
        self.docs_config_tab.config_changed.connect(self._on_config_changed)

        self.hf_config_tab = ConfigTab(
            title="Hugging Face Settings",
            subtitle="Transformers download, loading, memory, dtype, and quantization options.",
            sections=hf_sections,
            include_paused_jobs=False,
            icon_name="huggingface",
        )
        self.hf_config_tab.config_changed.connect(self._on_config_changed)

        self.ollama_config_tab = ConfigTab(
            title="Ollama Settings",
            subtitle="Connection and model keep-alive settings for an existing Ollama daemon.",
            sections=ollama_sections,
            include_paused_jobs=False,
            icon_name="ollama",
        )
        self.ollama_config_tab.config_changed.connect(self._on_config_changed)

        self.server_tab = ServerTab()
        self.server_tab.config_changed.connect(
            lambda: self._log("INFO", "Server config updated."))

        self.accounts_tab = AccountsTab()
        self.hf_login_tab = self.accounts_tab.hf_login_tab
        self.accounts_tab.auth_changed.connect(self._on_hf_auth_changed)
        self.accounts_tab.data_imported.connect(self._on_data_imported)

        self.appearance_tab = AppearanceTab()
        self.appearance_tab.theme_changed.connect(self._on_appearance_changed)
        self.settings_panel = self._build_settings_panel()

    def _on_data_imported(self):
        try:
            saved = json.loads(APP_CONFIG_FILE.read_text(encoding="utf-8")) if APP_CONFIG_FILE.exists() else {}
            APP_CONFIG.clear()
            APP_CONFIG.update(APP_CONFIG_DEFAULTS)
            APP_CONFIG.update({k: v for k, v in saved.items() if k in APP_CONFIG})
        except Exception as exc:
            self._log("WARN", f"Imported app config could not be reloaded: {exc}")
        try:
            get_model_registry()._load()
            getapi_registry()._load()
        except Exception as exc:
            self._log("WARN", f"Imported model registries could not be reloaded: {exc}")
        active_id = self.active.id if self.active else ""
        self._load_sessions()
        if self.sessions:
            self._switch_session(active_id if active_id in self.sessions else max(self.sessions.values(), key=lambda s: s.id).id)
        else:
            self._new_session()
        if hasattr(self, "model_list"):
            self._refresh_model_list()
        if hasattr(self, "input_bar"):
            self._sync_input_bar_combo()
        for tab_name in ("config_tab", "docs_config_tab", "hf_config_tab", "ollama_config_tab"):
            tab = getattr(self, tab_name, None)
            if hasattr(tab, "refresh_values"):
                tab.refresh_values()
        if hasattr(self, "config_tab"):
            self.config_tab.refresh_paused_jobs()
        if hasattr(self, "server_tab") and hasattr(self.server_tab, "refresh_values"):
            self.server_tab.refresh_values()
        if hasattr(self, "api_tab") and hasattr(self.api_tab, "_refresh_saved"):
            self.api_tab._refresh_saved()
        if hasattr(self, "integrations_tab"):
            self.integrations_tab.refresh()
            if hasattr(self.integrations_tab, "_reload_discord_profiles"):
                self.integrations_tab._reload_discord_profiles()
            if hasattr(self.integrations_tab, "_reload_whatsapp_profiles"):
                self.integrations_tab._reload_whatsapp_profiles()
        if hasattr(self, "skills_tab") and hasattr(self.skills_tab, "_reload"):
            self.skills_tab._reload()
        if hasattr(self, "download_tab"):
            self.download_tab.refresh_hf_auth_state()
        if hasattr(self, "accounts_tab"):
            self.accounts_tab.refresh_state()
        self._log("INFO", "Imported NativeLab data and refreshed visible registries.")

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        root = QHBoxLayout(panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("labs_sidebar")
        sidebar.setFixedWidth(168)
        side_l = QVBoxLayout(sidebar)
        side_l.setContentsMargins(10, 12, 10, 12)
        side_l.setSpacing(8)

        hdr = QLabel("Settings")
        set_label_icon(hdr, "config", "Settings", 18)
        hdr.setObjectName("labs_sidebar_hdr")
        side_l.addWidget(hdr)

        self.settings_nav = QListWidget()
        self.settings_nav.setObjectName("labs_nav")
        self.settings_nav.setIconSize(icon_size(18))
        self.settings_nav.setFrameShape(QFrame.Shape.NoFrame)
        side_l.addWidget(self.settings_nav, 1)

        self.settings_stack = QStackedWidget()
        self._settings_pages = [
            ("General", "config", self.config_tab),
            ("Docs", "docs", self.docs_config_tab),
            ("Hugging Face", "huggingface", self.hf_config_tab),
            ("Ollama", "ollama", self.ollama_config_tab),
            ("Server", "server", self.server_tab),
            ("Appearance", "appearance", self.appearance_tab),
            ("Accounts", "key", self.accounts_tab),
        ]
        for label, icon_name, widget in self._settings_pages:
            item = QListWidgetItem(icon(icon_name), label)
            item.setData(Qt.ItemDataRole.UserRole, label)
            self.settings_nav.addItem(item)
            self.settings_stack.addWidget(widget)

        self.settings_nav.currentRowChanged.connect(self._on_settings_nav_changed)
        root.addWidget(sidebar)
        root.addWidget(self.settings_stack, 1)
        self.settings_nav.setCurrentRow(0)
        self._refresh_settings_nav_colors()
        return panel

    def _on_settings_nav_changed(self, row: int):
        if hasattr(self, "settings_stack") and 0 <= row < self.settings_stack.count():
            self.settings_stack.setCurrentIndex(row)
        self._refresh_settings_nav_colors()

    def _refresh_settings_nav_colors(self):
        if not hasattr(self, "settings_nav"):
            return
        current = self.settings_nav.currentRow()
        for i in range(self.settings_nav.count()):
            item = self.settings_nav.item(i)
            if item:
                item.setForeground(QColor(C["acc"] if i == current else C["txt2"]))

    def _show_settings_page(self, label: str):
        if not hasattr(self, "settings_nav"):
            return
        for i in range(self.settings_nav.count()):
            item = self.settings_nav.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == label:
                self.settings_nav.setCurrentRow(i)
                return

    def _current_settings_page(self) -> str:
        if not hasattr(self, "settings_nav"):
            return "General"
        item = self.settings_nav.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else "General"

    def _rebuild_settings_panel(self):
        old_panel = getattr(self, "settings_panel", None)
        current_page = self._current_settings_page()
        if hasattr(self, "accounts_tab") and hasattr(self.accounts_tab, "shutdown"):
            try:
                self.accounts_tab.shutdown()
            except Exception as exc:
                self._log("WARN", f"Accounts shutdown before rebuild failed: {exc}")
        if self.config_dialog is not None and old_panel is not None:
            layout = self.config_dialog.layout()
            if layout is not None:
                layout.removeWidget(old_panel)
        self._create_settings_pages()
        if self.config_dialog is not None:
            layout = self.config_dialog.layout()
            if layout is not None:
                layout.addWidget(self.settings_panel)
            self._show_settings_page(current_page)
        if old_panel is not None:
            old_panel.setParent(None)
            old_panel.deleteLater()

    def _build_dev_tab(self) -> QWidget:
        self.dev_stack = QStackedWidget()
        self.dev_sidebar = QWidget()
        self.dev_sidebar.setObjectName("labs_sidebar")
        side_l = QVBoxLayout(self.dev_sidebar)
        side_l.setContentsMargins(10, 12, 10, 12)
        side_l.setSpacing(8)

        hdr = QLabel("Dev")
        set_label_icon(hdr, "code", "Dev", 18)
        hdr.setObjectName("labs_sidebar_hdr")
        side_l.addWidget(hdr)

        self.dev_nav = QListWidget()
        self.dev_nav.setObjectName("labs_nav")
        self.dev_nav.setIconSize(icon_size(18))
        self.dev_nav.setFrameShape(QFrame.Shape.NoFrame)
        side_l.addWidget(self.dev_nav, 1)
        self.left_sidebar_stack.addWidget(self.dev_sidebar)

        self._dev_pages = [
            ("Labs", "labs", self.labs_tab),
            ("Logs", "logs", self.log_console),
            ("Integrations", "integrations", self.integrations_tab),
            ("API Server", "server", self.api_server_tab),
            ("Pipeline", "pipeline", self.pipeline_tab),
            ("MCP", "mcp", self.mcp_tab),
            ("Skills", "lightbulb", self.skills_tab),
        ]
        for label, icon_name, widget in self._dev_pages:
            item = QListWidgetItem(icon(icon_name), label)
            item.setData(Qt.ItemDataRole.UserRole, label)
            self.dev_nav.addItem(item)
            self.dev_stack.addWidget(widget)
        self.dev_nav.currentRowChanged.connect(self._on_dev_nav_changed)
        self.dev_nav.setCurrentRow(0)
        self._refresh_dev_nav_colors()
        return self.dev_stack

    def _on_dev_nav_changed(self, row: int):
        if not hasattr(self, "dev_stack"):
            return
        if 0 <= row < self.dev_stack.count():
            self.dev_stack.setCurrentIndex(row)
        self._refresh_dev_nav_colors()

    def _refresh_dev_nav_colors(self):
        if not hasattr(self, "dev_nav"):
            return
        current = self.dev_nav.currentRow()
        for i in range(self.dev_nav.count()):
            item = self.dev_nav.item(i)
            if item:
                item.setForeground(QColor(C["acc"] if i == current else C["txt2"]))

    def _show_dev_page(self, label: str):
        if not self._developer_mode_enabled():
            self._log("WARN", "Enable Developer Mode in Settings > General to use Dev pages.")
            return
        if not hasattr(self, "dev_nav"):
            return
        for i in range(self.dev_nav.count()):
            item = self.dev_nav.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == label:
                self.tabs.setCurrentWidget(self.dev_tab)
                self.dev_nav.setCurrentRow(i)
                return

    def _show_hf_login_tab(self):
        if hasattr(self, "accounts_tab"):
            self._show_config_dialog(default_page="Accounts")
            self.accounts_tab.show_hugging_face()

    def _on_hf_auth_changed(self):
        self._log("INFO", "Hugging Face auth updated.")
        if hasattr(self, "download_tab"):
            self.download_tab.refresh_hf_auth_state()

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction(QAction("New Session\tCtrl+N", self, triggered=self._new_session))
        fm.addSeparator()
        xm = fm.addMenu("Export Current Session")
        xm.addAction(QAction("JSON",     self, triggered=lambda: self._export_active("json")))
        xm.addAction(QAction("Markdown", self, triggered=lambda: self._export_active("md")))
        xm.addAction(QAction("TXT",      self, triggered=lambda: self._export_active("txt")))
        fm.addSeparator()
        fm.addAction(QAction("Quit\tCtrl+Q", self, triggered=self.close))
        vm = mb.addMenu("View")
        vm.addAction(QAction("Toggle Sidebar\tCtrl+B", self, triggered=self._toggle_sidebar))
        vm.addAction(QAction("Toggle Top Bar", self, triggered=self._toggle_topbar))
        vm.addAction(QAction("Choose Visible Tabs", self, triggered=lambda: self._show_tab_visibility_menu()))
        vm.addAction(QAction("Go to Logs\tCtrl+L",    self, triggered=self._goto_logs))
        vm.addAction(QAction("Go to Models\tCtrl+M",  self, triggered=self._goto_models_tab))
        vm.addSeparator()
        # ── Theme toggle ──────────────────────────────────────────────────────
        self._theme_action = QAction(icon("appearance"), "Switch to Dark Theme", self)
        self._theme_action.setCheckable(False)
        self._theme_action.triggered.connect(self._toggle_theme)
        vm.addAction(self._theme_action)
        self._update_theme_action_label()
        mm = mb.addMenu("Model")
        mm.addAction(QAction("Reload Model", self, triggered=self._reload_model))
        self._install_topbar_controls(mb)

    def _install_topbar_controls(self, menu_bar):
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 8, 0)
        row.setSpacing(6)
        self.btn_toggle_sidebar = QPushButton("")
        self.btn_toggle_sidebar.setToolTip("Toggle left sidebar")
        self.btn_toggle_topbar = QPushButton("")
        self.btn_toggle_topbar.setToolTip("Toggle top tab bar")
        self.btn_tab_menu = QPushButton("")
        self.btn_tab_menu.setToolTip("Choose visible tabs")
        self.btn_settings = QPushButton("")
        self.btn_settings.setToolTip("Open app configuration")
        for b in (self.btn_toggle_sidebar, self.btn_toggle_topbar, self.btn_tab_menu, self.btn_settings):
            b.setFixedSize(28, 24)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(b)
        self._apply_topbar_button_styles()
        self.btn_toggle_sidebar.clicked.connect(self._toggle_sidebar)
        self.btn_toggle_topbar.clicked.connect(self._toggle_topbar)
        self.btn_tab_menu.clicked.connect(lambda: self._show_tab_visibility_menu())
        self.btn_settings.clicked.connect(lambda: self._show_config_dialog())
        menu_bar.setCornerWidget(box, Qt.Corner.TopRightCorner)
        self._update_view_toggle_buttons()

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        sb.setFixedHeight(24)
        self.lbl_engine = QLabel("")
        self.lbl_engine.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        self._set_engine_status("Loading...", "loading")
        sb.addWidget(self.lbl_engine)
        sb.addWidget(self._vline())

        # Family badge in status bar
        self.lbl_family = QLabel("")
        self.lbl_family.setStyleSheet(f"color:{C['acc2']};padding:0 6px;font-size:10px;")
        sb.addWidget(self.lbl_family)
        sb.addWidget(self._vline())

        sb.addWidget(QLabel("  Context:"))
        self.ctx_slider = QSlider(Qt.Orientation.Horizontal)
        self.ctx_slider.setRange(512, MAX_CONTEXT_TOKENS)
        self.ctx_slider.setFixedWidth(140)
        self.ctx_slider.blockSignals(True)
        self.ctx_slider.setValue(DEFAULT_CTX())
        self.ctx_slider.blockSignals(False)
        self.ctx_slider.valueChanged.connect(self._on_ctx_changed)
        sb.addWidget(self.ctx_slider)

        self.ctx_input = QLineEdit(str(DEFAULT_CTX())) 
        self.ctx_input.setFixedWidth(60)
        self.ctx_input.setFixedHeight(20)
        self.ctx_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_input.editingFinished.connect(self._on_ctx_input_changed)
        sb.addWidget(self.ctx_input)

        self.ctx_warn = QLabel("")
        self.ctx_warn.setFixedWidth(24)
        self.ctx_warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_warn.setStyleSheet(f"color:{C['warn']};font-weight:bold;")
        sb.addWidget(self.ctx_warn)

        self.ctx_scope_lbl = QLabel("Session")
        self.ctx_scope_lbl.setFixedWidth(64)
        self.ctx_scope_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_scope_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 4px;font-size:10px;")
        sb.addWidget(self.ctx_scope_lbl)

        self.ctx_bar = QProgressBar()
        self.ctx_bar.setRange(0, DEFAULT_CTX())
        self.ctx_bar.setValue(0)
        self.ctx_bar.setFixedWidth(120)
        self.ctx_bar.setFixedHeight(6)
        self.ctx_bar.setTextVisible(False)
        sb.addWidget(self.ctx_bar)

        self.ctx_lbl = QLabel(f"0 / {DEFAULT_CTX()}")
        self.ctx_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_lbl.setMinimumWidth(112)
        self.ctx_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addWidget(self.ctx_lbl)

        sb.addPermanentWidget(self._vline())
        self.tps_lbl = QLabel("- tok/s")
        self.tps_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addPermanentWidget(self.tps_lbl)

        if HAS_PSUTIL:
            sb.addPermanentWidget(self._vline())
            self.ram_lbl = QLabel("RAM: -")
            self.ram_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
            sb.addPermanentWidget(self.ram_lbl)
            self._ram_timer = QTimer(self)
            self._ram_timer.timeout.connect(self._update_ram)
            self._ram_timer.start(2500)

    @staticmethod
    def _vline() -> QFrame:
        f = QFrame(); f.setFrameShape(QFrame.Shape.VLine)
        f.setStyleSheet(f"color:{C['bdr']};"); return f

    def _topbar_button_style(self) -> str:
        return (
            f"QPushButton{{background:transparent;color:{C['txt3']};"
            f"border:1px solid {C['bdr']};border-radius:6px;font-size:13px;"
            f"font-weight:700;padding:0;}}"
            f"QPushButton:hover{{color:{C['acc2']};border-color:{C['acc']};"
            f"background:{palette_rgba(C, 'acc', 0.14)};}}"
        )

    def _apply_topbar_button_styles(self):
        for name in ("btn_toggle_sidebar", "btn_toggle_topbar", "btn_tab_menu", "btn_settings"):
            button = getattr(self, name, None)
            if button is not None:
                button.setStyleSheet(self._topbar_button_style())

    def refresh_theme(self):
        self._apply_topbar_button_styles()
        if hasattr(self, "ctx_slider"):
            value = self.ctx_slider.value()
            color = C["ok"] if value <= 16384 else C["warn"] if value <= 24576 else C["err"]
            self._apply_ctx_slider_style(color)
        for label_name in ("ctx_lbl", "ctx_scope_lbl", "tps_lbl", "ram_lbl"):
            label = getattr(self, label_name, None)
            if label is not None:
                label.setStyleSheet(f"color:{C['txt2']};padding:0 6px;")
        if hasattr(self, "ctx_warn"):
            self.ctx_warn.setStyleSheet(f"color:{C['warn']};font-weight:bold;")
        if hasattr(self, "lbl_family"):
            self.lbl_family.setStyleSheet(f"color:{C['acc2']};padding:0 6px;font-size:10px;")
        if hasattr(self, "settings_nav"):
            self._refresh_settings_nav_colors()
        if hasattr(self, "dev_nav"):
            self._refresh_dev_nav_colors()
