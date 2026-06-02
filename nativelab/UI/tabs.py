from nativelab.imports.import_global import QStackedWidget,QThread,HAS_PSUTIL,json,QProgressBar,Path,QSpinBox,QComboBox,QFileDialog, QSlider, QColorDialog, psutil, Optional, subprocess, Dict, QHBoxLayout, datetime, Qt, pyqtSignal, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QListWidget, QListWidgetItem, QMenu, QInputDialog, QColor, QTextEdit, QFont, QCheckBox, QMessageBox, QScrollArea , QFrame, QTabWidget
from .UI_const import C_DARK, C_LIGHT, CURRENT_THEME, C,set_theme
from .effects import fade_in
from nativelab.core.engine_global import ApiConfig, ApiEngine
from nativelab.Prefrences.prefrence_global import ParallelPrefs, PARALLEL_PREFS
from nativelab.GlobalConfig.config_global import MODELS_DIR,APP_CONFIG, APP_CONFIG_DEFAULTS, CONFIG_FIELD_META, save_app_config, MODEL_ROLES, ROLE_ICONS,LLAMA_CLI_DEFAULT, LLAMA_SERVER_DEFAULT, refresh_binary_paths, LONG_TIMEOUT_SECONDS
from nativelab.components.components_global import list_paused_jobs, delete_paused_job, load_paused_job
from nativelab.Server.server_global import SERVER_CONFIG, detect_gpus, HfSearchWorker, HfDownloadWorker, MCP_CONFIG_FILE
from nativelab.Server.hfdwld import (
    HfSnapshotDownloadWorker,
    HfSnapshotSearchWorker,
    LlamaCppReleaseFetcher,
    LlamaCppDownloadWorker,
    OllamaListWorker,
    OllamaPullWorker,
)
from nativelab.Server.hfauth import (
    HF_TOKEN_SETTINGS_URL,
    HfDirectTokenWorker,
    HfOAuthLoginWorker,
    HfValidateWorker,
    clear_hf_credentials,
    load_hf_credentials,
    mask_hf_token,
)
from nativelab.Server.ollama_helpers import normalize_ollama_host
from nativelab.Model.model_global import ApiRegistry,getapi_registry,detect_quant_type, quant_info, detect_model_family, get_model_registry, API_PROVIDERS, ApiConfig, PROMPT_TEMPLATES, make_hf_model_ref, make_ollama_model_ref, popular_model_presets
from nativelab.labs import LabsTab, LabEndpoints
from nativelab.UI.icons import add_menu_action, icon, icon_size, role_icon, set_button_icon, set_label_icon, set_status_label, status_icon
from nativelab.UI.toggle import ToggleSwitch
CONFIG_SECTIONS = [
    ("Memory & RAM", ["ram_watchdog_mb", "max_ram_chunks", "auto_spill_on_start"]),
    ("Reference Engine", ["chunk_index_size", "ref_top_k", "ref_max_context_chars"]),
    ("Summarization", [
        "summary_chunk_chars", "summary_ctx_carry",
        "summary_n_pred_sect", "summary_n_pred_final",
        "pause_after_chunks",
    ]),
    ("Multi-PDF", ["multipdf_n_pred_sect", "multipdf_n_pred_final"]),
    ("Model Defaults", ["default_threads", "default_ctx", "default_n_predict"]),
    ("HF Transformers", [
        "hf_transformers_dir", "hf_token", "hf_revision",
        "hf_trust_remote_code", "hf_local_files_only", "hf_use_safetensors",
        "hf_torch_dtype", "hf_device_map", "hf_low_cpu_mem_usage",
        "hf_attn_implementation", "hf_max_memory", "hf_quantization",
    ]),
    ("Ollama", ["ollama_host", "ollama_keep_alive"]),
]


class ConfigTab(QWidget):
    """Settings page for one or more app-config sections."""

    config_changed = pyqtSignal()

    def __init__(
        self,
        parent=None,
        *,
        title: str = "App Configuration",
        subtitle: str = "",
        sections: Optional[list] = None,
        include_paused_jobs: bool = True,
        icon_name: str = "config",
    ):
        super().__init__(parent)
        self._fields: Dict[str, QWidget] = {}
        self._title = title
        self._icon_name = icon_name
        self._subtitle = subtitle or (
            "All thresholds and defaults are persisted to app_config.json. "
            "Hover over any field for a full description. Changes take effect immediately."
        )
        self._sections = list(sections) if sections is not None else list(CONFIG_SECTIONS)
        self._include_paused_jobs = include_paused_jobs
        self._build()

    def _build(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        self.setLayout(outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("chat_scroll")

        inner = QWidget()
        inner.setObjectName("chat_container")
        root = QVBoxLayout()
        root.setContentsMargins(22, 18, 22, 22); root.setSpacing(0)
        inner.setLayout(root); scroll.setWidget(inner); outer.addWidget(scroll)

        # Header
        hdr = QLabel(self._title)
        set_label_icon(hdr, self._icon_name, self._title, 18)
        hdr.setStyleSheet("font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        sub = QLabel(self._subtitle)
        sub.setWordWrap(True)
        sub.setObjectName("txt2_small")
        sub.setStyleSheet("margin-bottom:16px;")
        root.addWidget(sub)

        for cat_title, keys in self._sections:
            root.addSpacing(10)
            cat_lbl = QLabel(cat_title)
            cat_lbl.setStyleSheet(
                "font-size:12px;font-weight:bold;"
                f"letter-spacing:0.4px;padding:2px 0;")
            root.addWidget(cat_lbl)

            card = QFrame()
            card.setObjectName("tab_card")
            card_l = QVBoxLayout()
            card_l.setContentsMargins(18, 12, 18, 14); card_l.setSpacing(10)
            card.setLayout(card_l)

            for key in keys:
                meta = CONFIG_FIELD_META.get(key, {})
                val  = APP_CONFIG.get(key, APP_CONFIG_DEFAULTS.get(key, 0))
                row  = self._make_field(key, meta, val)
                card_l.addWidget(row)

            root.addWidget(card)

        if self._include_paused_jobs:
            self._build_paused_jobs_section(root)

        # Save / Reset buttons
        root.addSpacing(14)
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        save_btn = QPushButton("Save All Settings")
        set_button_icon(save_btn, "save", "Save All Settings")
        save_btn.setObjectName("btn_send")
        save_btn.setFixedHeight(34)
        save_btn.clicked.connect(self._save_all)
        reset_btn = QPushButton("Reset to Defaults")
        set_button_icon(reset_btn, "refresh-cw", "Reset to Defaults")
        reset_btn.setFixedHeight(34)
        reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(save_btn); btn_row.addWidget(reset_btn); btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()
        if self._include_paused_jobs:
            self.refresh_paused_jobs()
        dev_widget = self._fields.get("developer_mode")
        if isinstance(dev_widget, QCheckBox):
            dev_widget.toggled.connect(self._save_developer_mode)

    def _build_paused_jobs_section(self, root: QVBoxLayout):
        root.addSpacing(14)
        pj_lbl = QLabel("Paused Summarization Jobs")
        set_label_icon(pj_lbl, "circle-pause", "Paused Summarization Jobs", 16)
        pj_lbl.setStyleSheet(
            "font-size:12px;font-weight:bold;padding:2px 0;")
        root.addWidget(pj_lbl)

        pj_card = QFrame()
        pj_card.setObjectName("tab_card")
        pj_l = QVBoxLayout()
        pj_l.setContentsMargins(14, 10, 14, 12); pj_l.setSpacing(8)
        pj_card.setLayout(pj_l)

        pj_desc = QLabel(
            "Summarization jobs paused mid-way are saved here. "
            "Select a job and click Resume to continue from where it left off.")
        pj_desc.setWordWrap(True)
        pj_desc.setObjectName("txt2_small")
        pj_l.addWidget(pj_desc)

        self.paused_jobs_list = QListWidget()
        self.paused_jobs_list.setFixedHeight(110)
        self.paused_jobs_list.setObjectName("paused_list")
        pj_l.addWidget(self.paused_jobs_list)

        pj_btn_row = QHBoxLayout(); pj_btn_row.setSpacing(8)
        self.btn_resume_job   = QPushButton("Resume Job")
        self.btn_delete_job   = QPushButton("Delete Job")
        self.btn_refresh_jobs = QPushButton("Refresh")
        set_button_icon(self.btn_resume_job, "play", "Resume Job")
        set_button_icon(self.btn_delete_job, "delete", "Delete Job")
        set_button_icon(self.btn_refresh_jobs, "refresh-cw", "Refresh")
        for b in (self.btn_resume_job, self.btn_delete_job, self.btn_refresh_jobs):
            b.setFixedHeight(28); pj_btn_row.addWidget(b)
        pj_btn_row.addStretch()
        self.btn_refresh_jobs.clicked.connect(self.refresh_paused_jobs)
        self.btn_delete_job.clicked.connect(self.delete_paused_job)
        pj_l.addLayout(pj_btn_row)
        root.addWidget(pj_card)

    def _make_field(self, key: str, meta: dict, current_val) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame{background:transparent;border:none;}")
        fl = QVBoxLayout()
        fl.setContentsMargins(0, 0, 0, 2); fl.setSpacing(3)

        top_row = QHBoxLayout(); top_row.setSpacing(10)
        lbl = QLabel(meta.get("label", key))
        lbl.setStyleSheet("font-size:12px;font-weight:600;")
        lbl.setFixedWidth(230)
        top_row.addWidget(lbl)

        ftype = meta.get("type", "int")
        if ftype == "bool":
            widget = ToggleSwitch()
            widget.setChecked(bool(current_val))
        elif ftype == "choice":
            widget = QComboBox()
            widget.setFixedHeight(26)
            widget.setFixedWidth(170)
            for choice in meta.get("choices", []):
                widget.addItem(str(choice), str(choice))
            idx = widget.findData(str(current_val))
            widget.setCurrentIndex(max(idx, 0))
        else:
            widget = QLineEdit(str(current_val))
            widget.setFixedWidth(360 if ftype in ("str", "path", "password") else 90)
            widget.setFixedHeight(26)
            if ftype == "password":
                widget.setEchoMode(QLineEdit.EchoMode.Password)
                widget.setPlaceholderText("Optional token")
            elif ftype == "path":
                widget.setPlaceholderText("Directory path")
            if ftype == "int":
                widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
                mn = meta.get("min", 0); mx = meta.get("max", 99999)
                widget.setToolTip(f"Range: {mn} – {mx}")

        self._fields[key] = widget
        top_row.addWidget(widget)

        # Range hint
        if ftype == "int":
            rng_lbl = QLabel(f"({meta.get('min', 0)} – {meta.get('max', '∞')})")
            rng_lbl.setObjectName("txt2_xs")
            top_row.addWidget(rng_lbl)
        elif ftype == "choice":
            choices = ", ".join(meta.get("choices", []))
            rng_lbl = QLabel(choices)
            rng_lbl.setObjectName("txt2_xs")
            top_row.addWidget(rng_lbl)
        top_row.addStretch()
        fl.addLayout(top_row)

        # Description
        desc = QLabel(meta.get("desc", ""))
        desc.setWordWrap(True)
        desc.setObjectName("txt2_xs")
        desc.setStyleSheet("padding-left:2px;")
        fl.addWidget(desc)

        frame.setLayout(fl)
        return frame

    def _save_all(self):
        for key, widget in self._fields.items():
            meta  = CONFIG_FIELD_META.get(key, {})
            ftype = meta.get("type", "int")
            mn    = meta.get("min", 0)
            mx    = meta.get("max", 999999)
            if ftype == "bool":
                APP_CONFIG[key] = widget.isChecked()
            elif ftype == "choice":
                APP_CONFIG[key] = widget.currentData() or widget.currentText()
            elif ftype in ("str", "path", "password"):
                APP_CONFIG[key] = widget.text().strip()
            else:
                try:
                    v = int(widget.text())
                    v = max(mn, min(mx, v))
                    APP_CONFIG[key] = v
                    widget.setText(str(v))
                except ValueError:
                    widget.setText(str(APP_CONFIG.get(key, APP_CONFIG_DEFAULTS.get(key, 0))))
        save_app_config(APP_CONFIG)
        self.config_changed.emit()
        QMessageBox.information(self, "Saved", "Configuration saved successfully.")

    def _save_developer_mode(self, enabled: bool):
        APP_CONFIG["developer_mode"] = bool(enabled)
        if enabled:
            tab_cfg = dict(APP_CONFIG.get("tab_visibility", {}))
            tab_cfg["Dev"] = True
            APP_CONFIG["tab_visibility"] = tab_cfg
        save_app_config(APP_CONFIG)
        self.config_changed.emit()

    def _reset_defaults(self):
        if QMessageBox.question(
            self, "Reset Defaults", "Reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        for key in self._fields:
            if key in APP_CONFIG_DEFAULTS:
                APP_CONFIG[key] = APP_CONFIG_DEFAULTS[key]
        save_app_config(APP_CONFIG)
        for key, widget in self._fields.items():
            val = APP_CONFIG_DEFAULTS.get(key, 0)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            elif isinstance(widget, QComboBox):
                idx = widget.findData(str(val))
                widget.setCurrentIndex(max(idx, 0))
            else:
                widget.setText(str(val))
        self.config_changed.emit()

    def refresh_paused_jobs(self):
        if not hasattr(self, "paused_jobs_list"):
            return
        self.paused_jobs_list.clear()
        for job in list_paused_jobs():
            if job.get("job_id", "").endswith("_autosave"):
                continue
            jid   = job.get("job_id", "?")
            fname = job.get("filename", "?")
            nc    = job.get("next_chunk", 0)
            tot   = job.get("total", "?")
            ts    = job.get("paused_at", "")[:16]
            item  = QListWidgetItem(
                f"{fname}  -  chunk {nc}/{tot}  -  paused {ts}")
            item.setIcon(icon("circle-pause"))
            item.setData(Qt.ItemDataRole.UserRole, jid)
            item.setForeground(QColor(C["warn"]))
            self.paused_jobs_list.addItem(item)

    def delete_paused_job(self):
        if not hasattr(self, "paused_jobs_list"):
            return
        item = self.paused_jobs_list.currentItem()
        if not item: return
        jid = item.data(Qt.ItemDataRole.UserRole)
        delete_paused_job(jid)
        self.refresh_paused_jobs()

    def get_selected_job_id(self) -> str:
        if not hasattr(self, "paused_jobs_list"):
            return ""
        item = self.paused_jobs_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def get_selected_job_state(self) -> Optional[dict]:
        jid = self.get_selected_job_id()
        return load_paused_job(jid) if jid else None


class HuggingFaceLoginTab(QWidget):
    """Local Hugging Face OAuth/device login surface."""

    auth_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._login_worker: Optional[HfOAuthLoginWorker] = None
        self._token_worker: Optional[HfDirectTokenWorker] = None
        self._validate_worker: Optional[HfValidateWorker] = None
        self._build()
        self.refresh_state()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("chat_scroll")
        inner = QWidget(); inner.setObjectName("chat_container")
        root = QVBoxLayout(inner)
        root.setContentsMargins(22, 18, 22, 22); root.setSpacing(12)
        scroll.setWidget(inner); outer.addWidget(scroll)

        hdr = QLabel("Hugging Face Login")
        set_label_icon(hdr, "huggingface", "Hugging Face Login", 18)
        hdr.setStyleSheet("font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        sub = QLabel(
            "Sign in once to use gated or private Hugging Face repos across GGUF downloads, "
            "Transformers snapshots, and HF model loading. Tokens are stored locally under localllm/cred.")
        sub.setWordWrap(True)
        sub.setObjectName("txt2_small")
        root.addWidget(sub)

        token_card = QFrame(); token_card.setObjectName("tab_card")
        tl = QVBoxLayout(token_card); tl.setContentsMargins(16, 14, 16, 14); tl.setSpacing(10)
        token_title = QLabel("Advanced: Access Token")
        token_title.setStyleSheet("font-size:12px;font-weight:bold;")
        tl.addWidget(token_title)
        token_hint = QLabel(
            "Optional fallback. Paste a Hugging Face read or fine-grained access token "
            "if browser login is unavailable.")
        token_hint.setWordWrap(True); token_hint.setObjectName("txt2_small")
        tl.addWidget(token_hint)

        token_row = QHBoxLayout(); token_row.setSpacing(8)
        token_lbl = QLabel("Token:")
        token_lbl.setFixedWidth(78); token_lbl.setObjectName("txt2")
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setPlaceholderText("hf_...")
        self.hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_edit.setFixedHeight(30)
        self.btn_save_hf_token = QPushButton("Save Token")
        set_button_icon(self.btn_save_hf_token, "save", "Save Token")
        self.btn_save_hf_token.setFixedHeight(30)
        self.btn_save_hf_token.clicked.connect(self._save_direct_token)
        token_row.addWidget(token_lbl)
        token_row.addWidget(self.hf_token_edit, 1)
        token_row.addWidget(self.btn_save_hf_token)
        tl.addLayout(token_row)

        status = QFrame(); status.setObjectName("tab_card")
        st = QVBoxLayout(status); st.setContentsMargins(16, 14, 16, 14); st.setSpacing(8)
        status_title = QLabel("Auth Status")
        status_title.setStyleSheet("font-size:12px;font-weight:bold;")
        st.addWidget(status_title)
        self.hf_auth_status = QLabel("")
        self.hf_auth_status.setWordWrap(True)
        self.hf_auth_status.setStyleSheet("font-size:12px;font-weight:600;")
        st.addWidget(self.hf_auth_status)
        self.hf_auth_details = QLabel("")
        self.hf_auth_details.setWordWrap(True)
        self.hf_auth_details.setObjectName("txt2_small")
        st.addWidget(self.hf_auth_details)

        self.hf_login_message = QLabel("")
        self.hf_login_message.setWordWrap(True)
        self.hf_login_message.setObjectName("txt2_small")
        st.addWidget(self.hf_login_message)

        self.hf_device_code_label = QLabel("")
        self.hf_device_code_label.setWordWrap(True)
        self.hf_device_code_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hf_device_code_label.setStyleSheet(
            f"font-size:18px;font-weight:700;color:{C['acc']};"
            f"background:{C['bg2']};border:1px solid {C['bdr']};"
            "border-radius:6px;padding:8px 10px;"
        )
        self.hf_device_code_label.setVisible(False)
        st.addWidget(self.hf_device_code_label)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self.btn_hf_login = QPushButton("Login with Hugging Face")
        set_button_icon(self.btn_hf_login, "huggingface", "Login with Hugging Face")
        self.btn_hf_validate = QPushButton("Validate")
        set_button_icon(self.btn_hf_validate, "done", "Validate")
        self.btn_hf_logout = QPushButton("Logout")
        set_button_icon(self.btn_hf_logout, "log-out", "Logout")
        self.btn_hf_token_settings = QPushButton("Open Token Settings")
        set_button_icon(self.btn_hf_token_settings, "settings", "Open Token Settings")
        for btn in (self.btn_hf_login, self.btn_hf_validate, self.btn_hf_logout, self.btn_hf_token_settings):
            btn.setFixedHeight(32)
            btn_row.addWidget(btn)
        btn_row.addStretch()
        st.addLayout(btn_row)
        self.btn_hf_login.clicked.connect(self._login)
        self.btn_hf_validate.clicked.connect(self._validate)
        self.btn_hf_logout.clicked.connect(self._logout)
        self.btn_hf_token_settings.clicked.connect(lambda: self._open_url(HF_TOKEN_SETTINGS_URL))
        root.addWidget(status)
        root.addWidget(token_card)
        root.addStretch()

    def _format_ts(self, value) -> str:
        try:
            ts = int(value or 0)
            if ts <= 0:
                return ""
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ""

    def refresh_state(self):
        creds = load_hf_credentials()
        saved_token = str(creds.get("access_token", "") or "").strip()
        fallback_token = str(APP_CONFIG.get("hf_token", "") or "").strip()
        self.hf_token_edit.clear()
        self.hf_token_edit.setPlaceholderText(
            "Saved token present; paste a new token to replace" if saved_token else "hf_...")
        username = str(creds.get("username", "") or "").strip()
        details = []
        if saved_token:
            self.hf_auth_status.setText(f"Signed in{f' as {username}' if username else ''}.")
            details.append(f"Token: {mask_hf_token(saved_token)}")
        elif fallback_token:
            self.hf_auth_status.setText("Using App Configuration token fallback.")
            details.append(f"Token: {mask_hf_token(fallback_token)}")
        else:
            self.hf_auth_status.setText("Signed out. Public repositories only.")
        scope = str(creds.get("scope", "") or "").strip()
        if scope:
            details.append(f"Scope: {scope}")
        client_id = str(creds.get("client_id", "") or "").strip()
        if client_id and not client_id.startswith("hf_"):
            details.append("OAuth app: NativeLab")
        expiry = self._format_ts(creds.get("expires_at"))
        if expiry:
            details.append(f"Expires: {expiry}")
        validated = self._format_ts(creds.get("last_validated_at"))
        if validated:
            details.append(f"Last validated: {validated}")
        last_error = str(creds.get("last_error", "") or "").strip()
        if last_error:
            details.append(f"Last error: {last_error}")
        self.hf_auth_details.setText("\n".join(details) if details else "No saved Hugging Face token.")
        if saved_token:
            self.hf_device_code_label.setVisible(False)
            self.hf_device_code_label.setText("")

    def _save_direct_token(self):
        token = self.hf_token_edit.text().strip()
        if not token:
            self.hf_login_message.setText("Paste a Hugging Face access token first.")
            return
        self.btn_save_hf_token.setEnabled(False)
        self.hf_login_message.setText("Validating Hugging Face access token...")
        self._token_worker = HfDirectTokenWorker(token)
        self._token_worker.done.connect(self._on_direct_token_done)
        self._token_worker.err.connect(self._on_direct_token_err)
        self._token_worker.start()

    def _on_direct_token_done(self, _creds: dict):
        self.btn_save_hf_token.setEnabled(True)
        self.hf_login_message.setText("Hugging Face token saved and validated.")
        self.hf_token_edit.clear()
        self.refresh_state()
        self.auth_changed.emit()

    def _on_direct_token_err(self, msg: str):
        self.btn_save_hf_token.setEnabled(True)
        self.hf_login_message.setText(msg)
        self.refresh_state()

    def _login(self):
        self.btn_hf_login.setEnabled(False)
        self.hf_login_message.setText("Starting Hugging Face login...")
        self.hf_device_code_label.setVisible(False)
        self.hf_device_code_label.setText("")
        self._login_worker = HfOAuthLoginWorker()
        self._login_worker.code_ready.connect(self._on_login_code)
        self._login_worker.status.connect(self.hf_login_message.setText)
        self._login_worker.done.connect(self._on_login_done)
        self._login_worker.err.connect(self._on_login_err)
        self._login_worker.start()

    def _on_login_code(self, user_code: str, url: str):
        if user_code:
            self.hf_device_code_label.setText(f"Enter this code in Hugging Face:\n{user_code}")
            self.hf_device_code_label.setVisible(True)
            msg = "Browser opened for Hugging Face login. Enter the code shown below."
        else:
            self.hf_device_code_label.setVisible(False)
            msg = "Browser opened for Hugging Face login. Follow the browser instructions."
        self.hf_login_message.setText(msg)
        self._open_url(url)

    def _on_login_done(self, _creds: dict):
        self.btn_hf_login.setEnabled(True)
        self.hf_login_message.setText("Hugging Face login saved and validated.")
        self.hf_device_code_label.setVisible(False)
        self.hf_device_code_label.setText("")
        self.refresh_state()
        self.auth_changed.emit()

    def _on_login_err(self, msg: str):
        self.btn_hf_login.setEnabled(True)
        self.hf_login_message.setText(msg)
        self.hf_device_code_label.setVisible(False)
        self.hf_device_code_label.setText("")
        self.refresh_state()

    def _validate(self):
        self.btn_hf_validate.setEnabled(False)
        self.hf_login_message.setText("Validating Hugging Face token...")
        self._validate_worker = HfValidateWorker()
        self._validate_worker.done.connect(self._on_validate_done)
        self._validate_worker.err.connect(self._on_validate_err)
        self._validate_worker.start()

    def _on_validate_done(self, _creds: dict):
        self.btn_hf_validate.setEnabled(True)
        self.hf_login_message.setText("Hugging Face token is valid.")
        self.refresh_state()
        self.auth_changed.emit()

    def _on_validate_err(self, msg: str):
        self.btn_hf_validate.setEnabled(True)
        self.hf_login_message.setText(msg)
        self.refresh_state()

    def _logout(self):
        clear_hf_credentials(keep_client_id=False)
        self.hf_login_message.setText("Signed out.")
        self.hf_device_code_label.setVisible(False)
        self.hf_device_code_label.setText("")
        self.refresh_state()
        self.auth_changed.emit()

    def _open_url(self, url: str):
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as exc:
            self.hf_login_message.setText(f"Open this URL manually: {url}\n{exc}")


class AccountsTab(QWidget):
    """Account integrations grouped by provider."""

    auth_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("accounts_subtabs")
        self.hf_login_tab = HuggingFaceLoginTab()
        self.hf_login_tab.auth_changed.connect(self.auth_changed.emit)
        self.tabs.addTab(self.hf_login_tab, icon("huggingface"), "Hugging Face")
        root.addWidget(self.tabs)

    def show_hugging_face(self):
        self.tabs.setCurrentWidget(self.hf_login_tab)

    def refresh_state(self):
        self.hf_login_tab.refresh_state()

    def refresh_icons(self):
        self.tabs.setTabIcon(0, icon("huggingface"))


class ParallelLoadingDialog(QWidget):
    """Embedded panel (not a popup) in the Models tab for parallel loading config."""

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._prefs = PARALLEL_PREFS
        self._build()

    def _build(self):
        root = QVBoxLayout()
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # ── enable toggle ──────────────────────────────────────────────────────
        top_row = QHBoxLayout(); top_row.setSpacing(10)
        self.chk_enable = ToggleSwitch("Enable Parallel Model Loading")
        self.chk_enable.setStyleSheet("font-weight:600;font-size:12px;")
        self.chk_enable.setChecked(self._prefs.enabled)
        self.chk_enable.toggled.connect(self._on_toggle)
        top_row.addWidget(self.chk_enable)
        top_row.addStretch()
        root.addLayout(top_row)

        # ── warning banner ─────────────────────────────────────────────────────
        self.warn_banner = QLabel(
            "Parallel loading runs multiple model engines simultaneously.\n"
            "Each model consumes its own RAM slice (typically 4–16 GB per model).\n"
            "High CPU/RAM usage is expected. Ensure your system has sufficient memory\n"
            "before enabling. Swap usage may cause severe performance degradation."
        )
        self.warn_banner.setWordWrap(True)
        self.warn_banner.setStyleSheet(
            f"color:{C['warn']};font-size:11px;padding:10px 12px;"
            f"background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.25);"
            f"border-radius:8px;"
        )
        root.addWidget(self.warn_banner)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        root.addWidget(sep)

        # ── auto-load role checkboxes ──────────────────────────────────────────
        auto_lbl = QLabel("Auto-load these roles on startup:")
        auto_lbl.setStyleSheet("font-size:11px;")
        root.addWidget(auto_lbl)

        self._role_checks: Dict[str, QCheckBox] = {}
        role_grid = QHBoxLayout(); role_grid.setSpacing(12)
        for role in MODEL_ROLES:
            if role == "general": continue  # always loaded
            chk = ToggleSwitch(ROLE_ICONS[role])
            chk.setIcon(role_icon(role))
            chk.setIconSize(icon_size(16))
            chk.setChecked(role in self._prefs.auto_load_roles)
            chk.setEnabled(self._prefs.enabled)
            chk.toggled.connect(self._save)
            self._role_checks[role] = chk
            role_grid.addWidget(chk)
        role_grid.addStretch()
        root.addLayout(role_grid)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        root.addWidget(sep2)

        # ── pipeline mode ──────────────────────────────────────────────────────
        pipeline_row = QHBoxLayout(); pipeline_row.setSpacing(10)
        self.chk_pipeline = ToggleSwitch("Enable Reasoning -> Coding Pipeline Mode")
        self.chk_pipeline.setStyleSheet("font-size:12px;")
        self.chk_pipeline.setChecked(self._prefs.pipeline_mode)
        self.chk_pipeline.setEnabled(self._prefs.enabled)
        self.chk_pipeline.toggled.connect(self._save)
        pipeline_row.addWidget(self.chk_pipeline)
        pipeline_row.addStretch()
        root.addLayout(pipeline_row)

        pipeline_desc = QLabel(
            "When enabled and both Reasoning + Coding engines are loaded:\n"
            "  1. Reasoning model analyses your coding request -> produces a detailed plan\n"
            "  2. Coding model receives the plan as context -> generates the code\n"
            "Only activates when the prompt looks like a coding request."
        )
        pipeline_desc.setWordWrap(True)
        pipeline_desc.setStyleSheet("font-size:11px;padding-left:4px;")
        root.addWidget(pipeline_desc)

        # ── RAM estimate helper ────────────────────────────────────────────────
        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        root.addWidget(sep3)

        self.ram_estimate_lbl = QLabel("")
        self.ram_estimate_lbl.setWordWrap(True)
        self.ram_estimate_lbl.setStyleSheet("font-size:10px;")
        root.addWidget(self.ram_estimate_lbl)
        self._update_ram_estimate()

        root.addStretch()
        self.setLayout(root)
        self._update_enabled_state()

    def _on_toggle(self, checked: bool):
        if checked and not self._prefs.warned:
            result = QMessageBox.warning(
                self, "Parallel Loading Warning",
                "Parallel model loading runs multiple GGUF engines simultaneously.\n\n"
                "Each model occupies its own RAM allocation:\n"
                "  • Q4 7B model  ≈  4–5 GB RAM\n"
                "  • Q5 13B model ≈ 9–10 GB RAM\n"
                "  • Q4 70B model ≈ 38–40 GB RAM\n\n"
                "Running 2–3 models simultaneously requires 12–30+ GB of RAM.\n"
                "Insufficient RAM will cause heavy swap usage or system freezes.\n\n"
                "Only proceed if you have sufficient memory.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result != QMessageBox.StandardButton.Yes:
                self.chk_enable.blockSignals(True)
                self.chk_enable.setChecked(False)
                self.chk_enable.blockSignals(False)
                return
            self._prefs.warned = True

        self._prefs.enabled = checked
        self._update_enabled_state()
        self._save()

    def _update_enabled_state(self):
        en = self._prefs.enabled
        for chk in self._role_checks.values():
            chk.setEnabled(en)
        self.chk_pipeline.setEnabled(en)
        self.warn_banner.setVisible(en)

    def _save(self):
        self._prefs.auto_load_roles = [
            r for r, chk in self._role_checks.items() if chk.isChecked()]
        self._prefs.pipeline_mode = self.chk_pipeline.isChecked()
        self._prefs.save()
        self._update_ram_estimate()
        self.settings_changed.emit()

    def _update_ram_estimate(self):
        if not self._prefs.enabled:
            self.ram_estimate_lbl.setText("")
            return
        n_models = 1 + len(self._prefs.auto_load_roles)
        estimate  = n_models * 6   # rough 6 GB average
        avail     = ""
        if HAS_PSUTIL:
            mem   = psutil.virtual_memory()
            avail = f"  |  System RAM: {mem.total // (1024**3)} GB available"
        self.ram_estimate_lbl.setText(
            f"Estimated RAM for {n_models} model(s): ~{estimate} GB{avail}\n"
            f"(Rough average; actual usage depends on model size and quant type)"
        )

    @property
    def prefs(self) -> ParallelPrefs:
        return self._prefs

class AppearanceTab(QWidget):
    """Live theme colour editor - sliders + colour picker per token."""

    theme_changed = pyqtSignal(dict)   # emits updated C_LIGHT dict

    _GROUPS = [
        ("Backgrounds", ["bg0","bg1","bg2","bg3","surface","surface2"]),
        ("Text",        ["txt","txt2","txt3"]),
        ("Accent",      ["acc","acc2","acc_dim","highlight","glow"]),
        ("Bubbles",     ["usr","ast","rsn","cod"]),
        ("Borders",     ["bdr","bdr2"]),
        ("Semantic",    ["ok","warn","err","pipeline"]),
    ]

    _LABELS = {
        "bg0":"Canvas","bg1":"Sidebar/Panel","bg2":"Input Surface",
        "bg3":"Hover/Selection","surface":"Card Surface","surface2":"Focused Input",
        "txt":"Primary Text","txt2":"Secondary Text","txt3":"Muted/Disabled",
        "acc":"Accent Primary","acc2":"Accent Hover","acc_dim":"Accent Tint BG",
        "highlight":"Highlight Tint","glow":"Glow",
        "usr":"User Bubble","ast":"Assistant Bubble","rsn":"Reasoning Bubble","cod":"Code Bubble",
        "bdr":"Border Light","bdr2":"Border Medium",
        "ok":"Success","warn":"Warning","err":"Error","pipeline":"Pipeline Badge",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._palette = dict(C_LIGHT)   # working copy
        self._rows: dict[str, tuple] = {}  # key → (swatch_btn, hex_edit, h, s, l)
        self._building = False
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top toolbar ──────────────────────────────────────────────────────
        bar = QWidget(); bar.setObjectName("appearance_bar")
        bar_row = QHBoxLayout(bar)
        bar_row.setContentsMargins(16, 10, 16, 10)
        bar_row.setSpacing(8)

        lbl = QLabel("Theme Editor")
        set_label_icon(lbl, "appearance", "Theme Editor", 18)
        lbl.setObjectName("appearance_hdr")
        bar_row.addWidget(lbl)
        bar_row.addStretch()

        self.btn_reset = QPushButton("Reset")
        set_button_icon(self.btn_reset, "refresh-cw", "Reset")
        self.btn_reset.setObjectName("appearance_btn")
        self.btn_reset.setToolTip("Reset to current built-in palette")
        self.btn_reset.clicked.connect(self._reset)

        self.btn_save = QPushButton("Save")
        set_button_icon(self.btn_save, "save", "Save")
        self.btn_save.setObjectName("appearance_btn_acc")
        self.btn_save.setToolTip("Save palette to config (persists across restarts)")
        self.btn_save.clicked.connect(self._save)

        bar_row.addWidget(self.btn_reset)
        bar_row.addWidget(self.btn_save)
        root.addWidget(bar)

        # ── Scroll area ───────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setObjectName("chat_scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget(); inner.setObjectName("chat_container")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(16, 12, 16, 20)
        inner_layout.setSpacing(16)

        for group_name, keys in self._GROUPS:
            grp = self._make_group(group_name, keys)
            inner_layout.addWidget(grp)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll)

    def _make_group(self, title: str, keys: list) -> QWidget:
        frame = QFrame(); frame.setObjectName("tab_card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 14)
        lay.setSpacing(6)

        hdr = QLabel(title)
        hdr.setObjectName("appearance_group_hdr")
        lay.addWidget(hdr)

        for key in keys:
            row = self._make_row(key)
            lay.addLayout(row)

        return frame

    def _make_row(self, key: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        # Label
        name_lbl = QLabel(self._LABELS.get(key, key))
        name_lbl.setObjectName("appearance_row_lbl")
        name_lbl.setFixedWidth(130)
        row.addWidget(name_lbl)

        # Colour swatch button
        swatch = QPushButton()
        swatch.setObjectName("appearance_swatch")
        swatch.setFixedSize(32, 24)
        swatch.setToolTip("Click to pick colour")
        swatch.clicked.connect(lambda _, k=key: self._pick_color(k))
        self._apply_swatch_color(swatch, self._palette.get(key, "#ffffff"))
        row.addWidget(swatch)

        # Hex field
        hex_edit = QLineEdit(self._resolve_hex(self._palette.get(key, "#ffffff")))
        hex_edit.setObjectName("appearance_hex")
        hex_edit.setFixedWidth(80)
        hex_edit.setMaxLength(9)
        hex_edit.editingFinished.connect(lambda k=key, e=hex_edit: self._on_hex_edit(k, e))
        row.addWidget(hex_edit)

        # HSL sliders
        slider_box = QHBoxLayout(); slider_box.setSpacing(4)
        sliders = []
        for label, rng in [("H", 360), ("S", 100), ("L", 100)]:
            col_box = QVBoxLayout(); col_box.setSpacing(1)
            sl_lbl = QLabel(label); sl_lbl.setObjectName("appearance_sl_lbl")
            sl_lbl.setFixedWidth(10)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setObjectName("appearance_slider")
            sl.setRange(0, rng)
            sl.setFixedWidth(72)
            col_box.addWidget(sl_lbl)
            col_box.addWidget(sl)
            slider_box.addLayout(col_box)
            sliders.append(sl)

        self._set_sliders_from_hex(sliders, self._palette.get(key, "#ffffff"))

        for i, sl in enumerate(sliders):
            sl.valueChanged.connect(lambda _, k=key, ss=sliders: self._on_slider(k, ss))

        row.addLayout(slider_box)
        row.addStretch()

        self._rows[key] = (swatch, hex_edit, sliders)
        return row

    # ── Colour helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_hex(val: str) -> str:
        """Return #rrggbb for solid values; for rgba keep as-is."""
        if val.startswith("rgba"): return val
        if val.startswith("#"):    return val
        return "#ffffff"

    @staticmethod
    def _apply_swatch_color(btn: QPushButton, val: str):
        if val.startswith("rgba"):
            # parse rgba → solid for swatch display
            try:
                parts = val.replace("rgba(","").replace(")","").split(",")
                r,g,b = int(parts[0]),int(parts[1]),int(parts[2])
                hex_c = f"#{r:02x}{g:02x}{b:02x}"
            except Exception:
                hex_c = "#aaaaaa"
        else:
            hex_c = val
        btn.setStyleSheet(
            f"QPushButton{{background:{hex_c};border:1px solid #00000033;"
            f"border-radius:4px;}}"
            f"QPushButton:hover{{border:2px solid #000000aa;}}")

    @staticmethod
    def _hex_to_hsl(hex_c: str):
        hex_c = hex_c.lstrip("#")
        if len(hex_c) == 3:
            hex_c = "".join(c*2 for c in hex_c)
        r,g,b = int(hex_c[0:2],16)/255, int(hex_c[2:4],16)/255, int(hex_c[4:6],16)/255
        import colorsys
        h,l,s = colorsys.rgb_to_hls(r,g,b)
        return int(h*360), int(s*100), int(l*100)

    @staticmethod
    def _hsl_to_hex(h: int, s: int, l: int) -> str:
        import colorsys
        r,g,b = colorsys.hls_to_rgb(h/360, l/100, s/100)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def _set_sliders_from_hex(self, sliders, val: str):
        if val.startswith("rgba"): return
        try:
            h,s,l = self._hex_to_hsl(self._resolve_hex(val))
            for sl, v in zip(sliders, [h,s,l]):
                sl.blockSignals(True); sl.setValue(v); sl.blockSignals(False)
        except Exception:
            pass

    # ── Event handlers ────────────────────────────────────────────────────────
    def _pick_color(self, key: str):
        current = self._resolve_hex(self._palette.get(key, "#ffffff"))
        init = QColor(current) if current.startswith("#") else QColor("#ffffff")
        color = QColorDialog.getColor(init, self, f"Pick colour - {self._LABELS.get(key,key)}")
        if color.isValid():
            hex_c = color.name()
            self._apply_key(key, hex_c)

    def _on_hex_edit(self, key: str, edit: QLineEdit):
        val = edit.text().strip()
        if not val.startswith("#") and not val.startswith("rgba"):
            val = "#" + val
        self._apply_key(key, val)

    def _on_slider(self, key: str, sliders):
        h,s,l = sliders[0].value(), sliders[1].value(), sliders[2].value()
        hex_c = self._hsl_to_hex(h, s, l)
        self._apply_key(key, hex_c, skip_sliders=True)

    def _apply_key(self, key: str, val: str, skip_sliders=False):
        self._palette[key] = val
        if key in self._rows:
            swatch, hex_edit, sliders = self._rows[key]
            self._apply_swatch_color(swatch, val)
            hex_edit.blockSignals(True)
            hex_edit.setText(self._resolve_hex(val))
            hex_edit.blockSignals(False)
            if not skip_sliders and not val.startswith("rgba"):
                self._set_sliders_from_hex(sliders, val)
        self.theme_changed.emit(dict(self._palette))

    # ── Toolbar actions ───────────────────────────────────────────────────────
    def _reset(self):
        from nativelab.UI import UI_const
        self._palette = dict(UI_const.C_LIGHT if UI_const.CURRENT_THEME == "light" else UI_const.C_DARK)
        self._rebuild_rows()
        self.theme_changed.emit(dict(self._palette))

    def _save(self):
        from nativelab.UI import UI_const
        key = "custom_light_palette" if UI_const.CURRENT_THEME == "light" else "custom_dark_palette"
        APP_CONFIG[key] = dict(self._palette)
        save_app_config(APP_CONFIG)

    def _rebuild_rows(self):
        """Refresh all swatches/hex/sliders after bulk palette change."""
        for key, (swatch, hex_edit, sliders) in self._rows.items():
            val = self._palette.get(key, "#ffffff")
            self._apply_swatch_color(swatch, val)
            hex_edit.blockSignals(True)
            hex_edit.setText(self._resolve_hex(val))
            hex_edit.blockSignals(False)
            self._set_sliders_from_hex(sliders, val)

    def load_palette(self, palette: dict):
        self._palette = dict(palette)
        self._rebuild_rows()

# ═════════════════════════════ SERVER TAB ════════════════════════════════════

class ServerTab(QWidget):
    """
    Server & binary configuration tab.
    Lets the user browse for llama-cli / llama-server binaries,
    configure host / port range, and add extra CLI flags.
    """
    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg = SERVER_CONFIG
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("chat_scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget(); inner.setObjectName("chat_container")
        root = QVBoxLayout(inner)
        root.setContentsMargins(22, 18, 22, 22)
        root.setSpacing(0)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QLabel("Server & Binary Configuration")
        set_label_icon(hdr, "server", "Server & Binary Configuration", 18)
        hdr.setStyleSheet(
            "font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)

        os_lbl = QLabel(
            f"Detected OS: <b>{self._cfg.detected_os}</b>   "
            f"(looking for <code>{self._cfg.default_cli_name}</code> / "
            f"<code>{self._cfg.default_server_name}</code>)")
        os_lbl.setTextFormat(Qt.TextFormat.RichText)
        os_lbl.setWordWrap(True)
        os_lbl.setObjectName("txt2_small")
        os_lbl.setStyleSheet("margin-bottom:14px;")
        root.addWidget(os_lbl)

        # ── Binary paths card ────────────────────────────────────────────────
        root.addWidget(self._section("BINARY PATHS"))
        bin_card = self._card()
        bin_l = QVBoxLayout(bin_card)
        bin_l.setContentsMargins(16, 14, 16, 14)
        bin_l.setSpacing(10)

        # llama-cli row
        cli_row = QHBoxLayout(); cli_row.setSpacing(8)
        cli_lbl = QLabel("llama-cli path:")
        cli_lbl.setFixedWidth(130)
        cli_lbl.setObjectName("txt2")
        self.cli_edit = QLineEdit()
        self.cli_edit.setPlaceholderText(
            f"Leave blank to use built-in default  ({LLAMA_CLI_DEFAULT})")
        self.cli_edit.setText(self._cfg.cli_path)
        self.cli_edit.setMinimumWidth(300)
        self._cli_status = QLabel("")
        self._cli_status.setFixedWidth(18)
        btn_cli = QPushButton("Browse…")
        btn_cli.setFixedWidth(80)
        btn_cli.setFixedHeight(28)
        btn_cli.clicked.connect(self._browse_cli)
        self.cli_edit.textChanged.connect(self._update_cli_status)
        cli_row.addWidget(cli_lbl)
        cli_row.addWidget(self.cli_edit, 1)
        cli_row.addWidget(self._cli_status)
        cli_row.addWidget(btn_cli)
        bin_l.addLayout(cli_row)

        # llama-server row
        srv_row = QHBoxLayout(); srv_row.setSpacing(8)
        srv_lbl = QLabel("llama-server path:")
        srv_lbl.setFixedWidth(130)
        srv_lbl.setObjectName("txt2")
        self.srv_edit = QLineEdit()
        self.srv_edit.setPlaceholderText(
            f"Leave blank to use built-in default  ({LLAMA_SERVER_DEFAULT})")
        self.srv_edit.setText(self._cfg.server_path)
        self.srv_edit.setMinimumWidth(300)
        self._srv_status = QLabel("")
        self._srv_status.setFixedWidth(18)
        btn_srv = QPushButton("Browse…")
        btn_srv.setFixedWidth(80)
        btn_srv.setFixedHeight(28)
        btn_srv.clicked.connect(self._browse_srv)
        self.srv_edit.textChanged.connect(self._update_srv_status)
        srv_row.addWidget(srv_lbl)
        srv_row.addWidget(self.srv_edit, 1)
        srv_row.addWidget(self._srv_status)
        srv_row.addWidget(btn_srv)
        bin_l.addLayout(srv_row)

        # Resolved paths display
        self._resolved_lbl = QLabel("")
        self._resolved_lbl.setWordWrap(True)
        self._resolved_lbl.setObjectName("resolved_box")
        bin_l.addWidget(self._resolved_lbl)
        self._refresh_resolved()

        root.addWidget(bin_card)
        root.addSpacing(14)

        # ── Server settings card ──────────────────────────────────────────────
        root.addWidget(self._section("SERVER SETTINGS"))
        srv_card = self._card()
        srv_l = QVBoxLayout(srv_card)
        srv_l.setContentsMargins(16, 14, 16, 14)
        srv_l.setSpacing(10)

        def _row(label_text, widget, hint=""):
            r = QHBoxLayout(); r.setSpacing(8)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(160)
            lbl.setObjectName("txt2")
            r.addWidget(lbl)
            r.addWidget(widget)
            if hint:
                hl = QLabel(hint)
                hl.setObjectName("txt3_xs")
                r.addWidget(hl)
            r.addStretch()
            return r

        self.host_edit = QLineEdit(self._cfg.host)
        self.host_edit.setFixedWidth(160)
        self.host_edit.setToolTip(
            "Bind host for llama-server.\n127.0.0.1 = localhost only (recommended).")
        srv_l.addLayout(_row("Bind Host:", self.host_edit,
                             "127.0.0.1 = local only"))

        self.port_lo_edit = QLineEdit(str(self._cfg.port_range_lo))
        self.port_lo_edit.setFixedWidth(80)
        self.port_lo_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.port_hi_edit = QLineEdit(str(self._cfg.port_range_hi))
        self.port_hi_edit.setFixedWidth(80)
        self.port_hi_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        port_row = QHBoxLayout(); port_row.setSpacing(8)
        port_lbl = QLabel("Port Range:")
        port_lbl.setFixedWidth(160)
        port_lbl.setObjectName("txt2")
        dash = QLabel("–")
        dash.setObjectName("txt2")
        port_row.addWidget(port_lbl)
        port_row.addWidget(self.port_lo_edit)
        port_row.addWidget(dash)
        port_row.addWidget(self.port_hi_edit)
        port_row.addWidget(
            self._hint("First free port in this range is used for each server instance"))
        port_row.addStretch()
        srv_l.addLayout(port_row)

        root.addWidget(srv_card)
        root.addSpacing(14)

        # ── GPU Acceleration card ─────────────────────────────────────────────
        root.addWidget(self._section("GPU ACCELERATION"))
        gpu_card = self._card()
        gpu_l = QVBoxLayout(gpu_card)
        gpu_l.setContentsMargins(16, 14, 16, 14)
        gpu_l.setSpacing(10)

        # Detect GPUs once at build time
        self._detected_gpus = detect_gpus()
        gpu_type = self._detected_gpus[0]["type"] if self._detected_gpus else "none"
        n_gpus   = len(self._detected_gpus)

        # Backend badge
        if gpu_type == "cuda":
            badge_txt = f"NVIDIA CUDA detected  ·  {n_gpus} GPU(s)"
            badge_col = C["ok"]
        elif gpu_type == "metal":
            badge_txt = f"Apple Metal detected  ·  {self._detected_gpus[0]['name']}"
            badge_col = C["ok"]
        elif gpu_type == "vulkan":
            badge_txt = f"Vulkan GPU detected  ·  {n_gpus} device(s)"
            badge_col = C["warn"]
        else:
            badge_txt = "No GPU detected - CPU-only mode"
            badge_col = C["txt2"]
        gpu_badge = QLabel(badge_txt)
        gpu_badge.setObjectName("gpu_badge")
        gpu_badge.setProperty("state", gpu_type)
        gpu_l.addWidget(gpu_badge)

        # GPU list (NVIDIA only shows VRAM)
        if self._detected_gpus:
            for g in self._detected_gpus:
                vram_s = f"  -  {g['vram_mb']} MB VRAM" if g["vram_mb"] else ""
                gl = QLabel(f"  [{g['idx']}]  {g['name']}{vram_s}")
                gl.setObjectName("txt2_xs")
                gpu_l.addWidget(gl)

        sep_gpu = QFrame(); sep_gpu.setFrameShape(QFrame.Shape.HLine)
        sep_gpu.setStyleSheet(
            f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        gpu_l.addWidget(sep_gpu)

        # Enable GPU checkbox
        enable_row = QHBoxLayout(); enable_row.setSpacing(10)
        self.chk_gpu = ToggleSwitch("Enable GPU offloading  (--ngl flag)")
        self.chk_gpu.setStyleSheet("font-weight:600;font-size:12px;")
        self.chk_gpu.setChecked(self._cfg.enable_gpu)
        self.chk_gpu.setEnabled(bool(self._detected_gpus))
        enable_row.addWidget(self.chk_gpu); enable_row.addStretch()
        gpu_l.addLayout(enable_row)

        # n_gpu_layers row
        ngl_row = QHBoxLayout(); ngl_row.setSpacing(8)
        ngl_lbl = QLabel("GPU layers  (--ngl):")
        ngl_lbl.setFixedWidth(160)
        ngl_lbl.setObjectName("txt2")
        self.spin_ngl = QSpinBox()
        self.spin_ngl.setRange(-1, 999); self.spin_ngl.setValue(self._cfg.ngl)
        self.spin_ngl.setFixedWidth(80); self.spin_ngl.setFixedHeight(28)
        self.spin_ngl.setSpecialValueText("All  (-1)")
        self.spin_ngl.setToolTip(
            "-1 = offload all layers to GPU (fastest)\n"
            "0  = CPU only\n"
            "N  = offload first N transformer layers")
        ngl_row.addWidget(ngl_lbl); ngl_row.addWidget(self.spin_ngl)
        ngl_row.addWidget(self._hint(
            "Set to -1 to offload all layers.  Lower if VRAM is limited."))
        ngl_row.addStretch()
        gpu_l.addLayout(ngl_row)

        # Main GPU row (multi-GPU NVIDIA only)
        mgpu_row = QHBoxLayout(); mgpu_row.setSpacing(8)
        mgpu_lbl = QLabel("Main GPU  (--main-gpu):")
        mgpu_lbl.setFixedWidth(160)
        mgpu_lbl.setObjectName("txt2")
        self.combo_main_gpu = QComboBox()
        self.combo_main_gpu.setFixedHeight(28); self.combo_main_gpu.setFixedWidth(220)
        if self._detected_gpus:
            for g in self._detected_gpus:
                self.combo_main_gpu.addItem(
                    f"[{g['idx']}]  {g['name']}", g["idx"])
            idx = self.combo_main_gpu.findData(self._cfg.main_gpu)
            self.combo_main_gpu.setCurrentIndex(max(idx, 0))
        else:
            self.combo_main_gpu.addItem("No GPU detected", 0)
        mgpu_row.addWidget(mgpu_lbl); mgpu_row.addWidget(self.combo_main_gpu)
        mgpu_row.addStretch()
        gpu_l.addLayout(mgpu_row)

        # Tensor split row (multi-GPU)
        ts_row = QHBoxLayout(); ts_row.setSpacing(8)
        ts_lbl = QLabel("Tensor split  (--ts):")
        ts_lbl.setFixedWidth(160)
        ts_lbl.setObjectName("txt2")
        self.ts_edit = QLineEdit(self._cfg.tensor_split)
        self.ts_edit.setFixedHeight(28)
        self.ts_edit.setPlaceholderText(
            "e.g. 0.6,0.4 for two GPUs  (leave blank for single GPU)")
        self.ts_edit.setToolTip(
            "Comma-separated fractions summing to 1.0.\n"
            "Used with multi-GPU NVIDIA setups only.\n"
            "Example: '0.5,0.5' splits evenly across 2 GPUs.")
        ts_row.addWidget(ts_lbl); ts_row.addWidget(self.ts_edit, 1)
        gpu_l.addLayout(ts_row)

        if n_gpus <= 1:
            self.ts_edit.setEnabled(False)
            self.ts_edit.setPlaceholderText("Multi-GPU only")

        root.addWidget(gpu_card)
        root.addSpacing(14)

        # ── Extra flags card ──────────────────────────────────────────────────
        root.addWidget(self._section("EXTRA LAUNCH FLAGS"))
        flag_card = self._card()
        flag_l = QVBoxLayout(flag_card)
        flag_l.setContentsMargins(16, 14, 16, 14)
        flag_l.setSpacing(10)

        flag_note = QLabel(
            "Additional flags appended to every llama-cli or llama-server launch.\n"
            "Separate multiple flags with spaces.  Example: --numa --no-mmap")
        flag_note.setWordWrap(True)
        flag_note.setObjectName("txt2_small")
        flag_l.addWidget(flag_note)

        cli_flag_row = QHBoxLayout(); cli_flag_row.setSpacing(8)
        cfl = QLabel("llama-cli extra flags:")
        cfl.setFixedWidth(160)
        cfl.setObjectName("txt2")
        self.extra_cli_edit = QLineEdit(self._cfg.extra_cli_args)
        self.extra_cli_edit.setPlaceholderText("e.g.  --numa  --no-mmap")
        cli_flag_row.addWidget(cfl)
        cli_flag_row.addWidget(self.extra_cli_edit, 1)
        flag_l.addLayout(cli_flag_row)

        srv_flag_row = QHBoxLayout(); srv_flag_row.setSpacing(8)
        sfl = QLabel("llama-server extra flags:")
        sfl.setFixedWidth(160)
        sfl.setObjectName("txt2")
        self.extra_srv_edit = QLineEdit(self._cfg.extra_server_args)
        self.extra_srv_edit.setPlaceholderText("e.g.  --numa  --flash-attn")
        srv_flag_row.addWidget(sfl)
        srv_flag_row.addWidget(self.extra_srv_edit, 1)
        flag_l.addLayout(srv_flag_row)

        root.addWidget(flag_card)
        root.addSpacing(14)

        # ── Quick-test card ───────────────────────────────────────────────────
        root.addWidget(self._section("BINARY TEST"))
        test_card = self._card()
        test_l = QVBoxLayout(test_card)
        test_l.setContentsMargins(16, 14, 16, 14)
        test_l.setSpacing(8)

        test_note = QLabel(
            "Click Test to run the binary with --version and verify it works.")
        test_note.setObjectName("txt2_small")
        test_l.addWidget(test_note)

        test_row = QHBoxLayout(); test_row.setSpacing(8)
        self.btn_test_cli = QPushButton("Test llama-cli")
        set_button_icon(self.btn_test_cli, "test-tube", "Test llama-cli")
        self.btn_test_cli.setFixedHeight(28)
        self.btn_test_cli.clicked.connect(lambda: self._test_binary("cli"))
        self.btn_test_srv = QPushButton("Test llama-server")
        set_button_icon(self.btn_test_srv, "test-tube", "Test llama-server")
        self.btn_test_srv.setFixedHeight(28)
        self.btn_test_srv.clicked.connect(lambda: self._test_binary("server"))
        test_row.addWidget(self.btn_test_cli)
        test_row.addWidget(self.btn_test_srv)
        test_row.addStretch()
        test_l.addLayout(test_row)

        self.test_output = QTextEdit()
        self.test_output.setReadOnly(True)
        self.test_output.setFixedHeight(90)
        self.test_output.setFont(QFont("Consolas", 10))
        self.test_output.setObjectName("log_te")
        test_l.addWidget(self.test_output)

        root.addWidget(test_card)
        root.addSpacing(14)

        # ── Save / Reset ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        save_btn = QPushButton("Save Server Settings")
        set_button_icon(save_btn, "save", "Save Server Settings")
        save_btn.setObjectName("btn_send")
        save_btn.setFixedHeight(34)
        save_btn.clicked.connect(self._save)
        reset_btn = QPushButton("Reset to Defaults")
        set_button_icon(reset_btn, "refresh-cw", "Reset to Defaults")
        reset_btn.setFixedHeight(34)
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)
        root.addStretch()

        self._update_cli_status()
        self._update_srv_status()

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size:12px;font-weight:bold;"
            f"letter-spacing:0.5px;padding:0;margin-bottom:2px;")
        return lbl

    @staticmethod
    def _card() -> QFrame:
        f = QFrame(); f.setObjectName("tab_card"); return f

    @staticmethod
    def _hint(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("txt3_xs")
        return lbl

    def _update_cli_status(self):
        path = self.cli_edit.text().strip()
        if not path:
            resolved = LLAMA_CLI_DEFAULT
        else:
            resolved = path
        ok = Path(resolved).exists()
        set_status_label(self._cli_status, "", "ok" if ok else "err", 15)
        self._cli_status.setToolTip(
            f"{'Found' if ok else 'Not found'}: {resolved}")
        self._refresh_resolved()

    def _update_srv_status(self):
        path = self.srv_edit.text().strip()
        if not path:
            resolved = LLAMA_SERVER_DEFAULT
        else:
            resolved = path
        ok = Path(resolved).exists()
        set_status_label(self._srv_status, "", "ok" if ok else "err", 15)
        self._srv_status.setToolTip(
            f"{'Found' if ok else 'Not found'}: {resolved}")
        self._refresh_resolved()

    def _refresh_resolved(self):
        cli_path  = self.cli_edit.text().strip() or LLAMA_CLI_DEFAULT
        srv_path  = self.srv_edit.text().strip() or LLAMA_SERVER_DEFAULT
        cli_ok    = "OK" if Path(cli_path).exists() else "NO"
        srv_ok    = "OK" if Path(srv_path).exists() else "NO"
        self._resolved_lbl.setText(
            f"Resolved  llama-cli:     {cli_ok}  {cli_path}\n"
            f"Resolved  llama-server:  {srv_ok}  {srv_path}")

    def _browse_cli(self):
        _ext = ".exe" if self._cfg.detected_os == "Windows" else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select llama-cli binary",
            str(Path(self.cli_edit.text()).parent
                if self.cli_edit.text() else Path.home()),
            f"llama-cli{_ext} (*{_ext});;All Files (*)")
        if path:
            self.cli_edit.setText(path)

    def _browse_srv(self):
        _ext = ".exe" if self._cfg.detected_os == "Windows" else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select llama-server binary",
            str(Path(self.srv_edit.text()).parent
                if self.srv_edit.text() else Path.home()),
            f"llama-server{_ext} (*{_ext});;All Files (*)")
        if path:
            self.srv_edit.setText(path)

    def _test_binary(self, which: str):
        if which == "cli":
            path = self.cli_edit.text().strip() or LLAMA_CLI_DEFAULT
        else:
            path = self.srv_edit.text().strip() or LLAMA_SERVER_DEFAULT

        if not Path(path).exists():
            self.test_output.setPlainText(f"Binary not found:\n{path}")
            return

        try:
            result = subprocess.run(
                [path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=LONG_TIMEOUT_SECONDS)
            out = result.stdout.decode("utf-8", errors="replace").strip()
            self.test_output.setPlainText(
                f"{Path(path).name} --version\n\n{out or '(no output)'}")
        except subprocess.TimeoutExpired:
            self.test_output.setPlainText(f"Timed out after {LONG_TIMEOUT_SECONDS}s:\n{path}")
        except Exception as e:
            self.test_output.setPlainText(f"Error:\n{e}")

    def _save(self):
        try:
            lo = int(self.port_lo_edit.text())
            hi = int(self.port_hi_edit.text())
            assert 1024 <= lo < hi <= 65535
        except Exception:
            QMessageBox.warning(
                self, "Invalid Port Range",
                "Port range must be two integers between 1024 and 65535,\n"
                "with the low value smaller than the high value.")
            return

        self._cfg.cli_path          = self.cli_edit.text().strip()
        self._cfg.server_path       = self.srv_edit.text().strip()
        self._cfg.host              = self.host_edit.text().strip() or "127.0.0.1"
        self._cfg.port_range_lo     = lo
        self._cfg.port_range_hi     = hi
        self._cfg.extra_cli_args    = self.extra_cli_edit.text().strip()
        # ── GPU settings ──────────────────────────────────────────────────────
        self._cfg.enable_gpu   = self.chk_gpu.isChecked()
        self._cfg.ngl          = self.spin_ngl.value()
        self._cfg.main_gpu     = self.combo_main_gpu.currentData() or 0
        self._cfg.tensor_split = self.ts_edit.text().strip()
        # Build GPU flags and prepend to extra_server_args
        _base_srv = self.extra_srv_edit.text().strip()
        # Strip any previously auto-injected GPU flags
        import re as _re
        _base_srv = _re.sub(
            r'\s*(?:--ngl\s+-?\d+|--main-gpu\s+\d+|--tensor-split\s+\S+)',
            "", _base_srv).strip()
        if self._cfg.enable_gpu and self._cfg.ngl != 0:
            _gpu_flags = f"--ngl {self._cfg.ngl}"
            if self._cfg.main_gpu:
                _gpu_flags += f" --main-gpu {self._cfg.main_gpu}"
            if self._cfg.tensor_split:
                _gpu_flags += f" --tensor-split {self._cfg.tensor_split}"
            _base_srv = (_gpu_flags + ("  " + _base_srv if _base_srv else "")).strip()
        self._cfg.extra_server_args = _base_srv
        self._cfg.save()
        refresh_binary_paths()
        self._refresh_resolved()
        self.config_changed.emit()
        QMessageBox.information(
            self, "Saved",
            "Server settings saved.\n\n"
            "Reload your model for the new binary paths to take effect.")

    def _reset(self):
        if QMessageBox.question(
            self, "Reset", "Reset all server settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        self.cli_edit.setText("")
        self.srv_edit.setText("")
        self.host_edit.setText("127.0.0.1")
        self.port_lo_edit.setText("8600")
        self.port_hi_edit.setText("8700")
        self.extra_cli_edit.setText("")
        self.extra_srv_edit.setText("")
        self._cfg.cli_path = ""
        self._cfg.server_path = ""
        self._cfg.host = "127.0.0.1"
        self._cfg.port_range_lo = 8600
        self._cfg.port_range_hi = 8700
        self._cfg.extra_cli_args = ""
        self._cfg.extra_server_args = ""
        self._cfg.enable_gpu   = False
        self._cfg.ngl          = -1
        self._cfg.main_gpu     = 0
        self._cfg.tensor_split = ""
        self.chk_gpu.setChecked(False)
        self.spin_ngl.setValue(-1)
        self.combo_main_gpu.setCurrentIndex(0)
        self.ts_edit.setText("")
        self._cfg.save()
        refresh_binary_paths()
        self._refresh_resolved()
        self._update_cli_status()
        self._update_srv_status()

class ModelDownloadTab(QWidget):
    """Model and runtime downloader tab."""

    hf_login_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._search_worker: Optional[HfSearchWorker] = None
        self._dl_worker:     Optional[HfDownloadWorker] = None
        self._files: list = []
        self._llama_fetcher:  Optional[LlamaCppReleaseFetcher] = None
        self._llama_dl:       Optional[LlamaCppDownloadWorker] = None
        self._llama_releases: list = []
        self._hf_snapshot_search: Optional[HfSnapshotSearchWorker] = None
        self._hf_snapshot_dl: Optional[HfSnapshotDownloadWorker] = None
        self._hf_snapshot_files: list = []
        self._ollama_list_worker: Optional[OllamaListWorker] = None
        self._ollama_pull_worker: Optional[OllamaPullWorker] = None
        self._build()
    # ── build ─────────────────────────────────────────────────────────────────
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("chat_scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); inner.setObjectName("chat_container")
        root  = QVBoxLayout(inner)
        root.setContentsMargins(22, 18, 22, 22); root.setSpacing(0)
        scroll.setWidget(inner); outer.addWidget(scroll)

        # Header
        hdr = QLabel("Model & Runtime Downloads")
        hdr.setStyleSheet(
            "font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        sub = QLabel(
            "Download GGUF models, full HF Transformers snapshots, Ollama models, "
            "and llama.cpp runtime binaries. Network access required.")
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"color:{C['txt2']};font-size:11px;margin-bottom:14px;")
        root.addWidget(sub)

        auth_card = self._card()
        auth_l = QHBoxLayout(auth_card)
        auth_l.setContentsMargins(14, 10, 14, 10); auth_l.setSpacing(8)
        self.hf_auth_download_label = QLabel("")
        self.hf_auth_download_label.setObjectName("txt2_small")
        self.hf_auth_download_label.setWordWrap(True)
        self.btn_hf_login_shortcut = QPushButton("Accounts")
        set_button_icon(self.btn_hf_login_shortcut, "huggingface", "Accounts")
        self.btn_hf_login_shortcut.setFixedHeight(30)
        self.btn_hf_login_shortcut.clicked.connect(lambda _=False: self.hf_login_requested.emit())
        auth_l.addWidget(self.hf_auth_download_label, 1)
        auth_l.addWidget(self.btn_hf_login_shortcut)
        root.addWidget(auth_card)
        root.addSpacing(14)

        # ── SEARCH ───────────────────────────────────────────────────────────
        root.addWidget(self._section("SEARCH REPOSITORY"))
        sc = self._card(); sl = QVBoxLayout(sc)
        sl.setContentsMargins(16, 14, 16, 14); sl.setSpacing(10)

        hint = QLabel(
            "Examples:  TheBloke/Mistral-7B-Instruct-v0.2-GGUF  "
            "bartowski/Llama-3.2-3B-Instruct-GGUF  "
            "lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        sl.addWidget(hint)

        preset_row = QHBoxLayout(); preset_row.setSpacing(8)
        preset_lbl = QLabel("Popular:")
        preset_lbl.setFixedWidth(60); preset_lbl.setObjectName("txt2")
        self.gguf_preset_combo = QComboBox()
        self.gguf_preset_combo.setFixedHeight(30)
        self.gguf_preset_combo.addItem("Choose a popular GGUF repo...", None)
        for preset in popular_model_presets("gguf"):
            self.gguf_preset_combo.addItem(preset.get("label", preset.get("repo", "")), preset)
        self.gguf_preset_combo.currentIndexChanged.connect(self._on_gguf_preset_selected)
        preset_row.addWidget(preset_lbl)
        preset_row.addWidget(self.gguf_preset_combo, 1)
        sl.addLayout(preset_row)

        row1 = QHBoxLayout(); row1.setSpacing(8)
        self.repo_edit = QLineEdit()
        self.repo_edit.setPlaceholderText(
            "Enter HuggingFace repo ID  (e.g. TheBloke/Mistral-7B-Instruct-v0.2-GGUF)")
        self.repo_edit.setFixedHeight(30)
        self.btn_search = QPushButton("Search")
        self.btn_search.setObjectName("btn_send")
        self.btn_search.setFixedHeight(30); self.btn_search.setFixedWidth(100)
        self.btn_search.clicked.connect(self._do_search)
        self.repo_edit.returnPressed.connect(self._do_search)
        row1.addWidget(self.repo_edit, 1); row1.addWidget(self.btn_search)
        sl.addLayout(row1)

        self.search_status = QLabel("")
        self.search_status.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        sl.addWidget(self.search_status)
        root.addWidget(sc); root.addSpacing(14)

        # ── RESULTS ──────────────────────────────────────────────────────────
        root.addWidget(self._section("AVAILABLE FILES"))
        rc = self._card(); rl = QVBoxLayout(rc)
        rl.setContentsMargins(16, 14, 16, 14); rl.setSpacing(10)

        self.results_list = QListWidget()
        self.results_list.setObjectName("model_list")
        self.results_list.setMinimumHeight(200)
        self.results_list.currentItemChanged.connect(self._on_file_selected)
        rl.addWidget(self.results_list)

        self.file_info = QLabel("Select a file above to see details.")
        self.file_info.setWordWrap(True)
        self.file_info.setStyleSheet(
            f"color:{C['txt2']};font-size:10px;"
            f"background:{C['bg2']};border-radius:5px;padding:6px 8px;")
        rl.addWidget(self.file_info)
        root.addWidget(rc); root.addSpacing(14)

        # ── DOWNLOAD ─────────────────────────────────────────────────────────
        root.addWidget(self._section("DOWNLOAD"))
        dc = self._card(); dl_l = QVBoxLayout(dc)
        dl_l.setContentsMargins(16, 14, 16, 14); dl_l.setSpacing(10)

        dest_row = QHBoxLayout(); dest_row.setSpacing(8)
        dest_lbl = QLabel("Save to:")
        dest_lbl.setFixedWidth(60)
        dest_lbl.setObjectName("txt2")
        self.dest_edit = QLineEdit(str(MODELS_DIR.resolve()))
        self.dest_edit.setReadOnly(True)
        btn_dest = QPushButton("Browse")
        btn_dest.setFixedHeight(28); btn_dest.setFixedWidth(80)
        btn_dest.clicked.connect(self._browse_dest)
        dest_row.addWidget(dest_lbl)
        dest_row.addWidget(self.dest_edit, 1)
        dest_row.addWidget(btn_dest)
        dl_l.addLayout(dest_row)

        self.dl_progress = QProgressBar()
        self.dl_progress.setRange(0, 100); self.dl_progress.setValue(0)
        self.dl_progress.setFixedHeight(10); self.dl_progress.setTextVisible(False)
        self.dl_progress.setStyleSheet(
            f"QProgressBar{{background:{C['bg2']};border:1px solid {C['bdr']};"
            f"border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{C['acc']};border-radius:4px;}}")
        dl_l.addWidget(self.dl_progress)

        self.dl_status = QLabel("No download in progress.")
        self.dl_status.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        dl_l.addWidget(self.dl_status)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)

        self.btn_download = QPushButton("Download Selected")
        self.btn_download.setObjectName("btn_send")
        self.btn_download.setFixedHeight(32)
        self.btn_download.setEnabled(False)
        self.btn_download.clicked.connect(self._start_download)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setObjectName("btn_stop")
        self.btn_pause.setFixedHeight(32)
        self.btn_pause.setFixedWidth(80)
        self.btn_pause.setVisible(False)
        self.btn_pause.clicked.connect(self._toggle_pause)

        # Stops download but keeps .part file - can be resumed next session
        self.btn_abort = QPushButton("Cancel")
        self.btn_abort.setObjectName("btn_stop")
        self.btn_abort.setFixedHeight(32)
        self.btn_abort.setFixedWidth(80)
        self.btn_abort.setVisible(False)
        self.btn_abort.setToolTip("Stop download and keep progress for resuming later")
        self.btn_abort.clicked.connect(self._abort_download)

        # Stops download AND wipes the .part file entirely
        self.btn_abort_delete = QPushButton("Cancel & Delete")
        self.btn_abort_delete.setObjectName("btn_stop")
        self.btn_abort_delete.setFixedHeight(32)
        self.btn_abort_delete.setFixedWidth(110)
        self.btn_abort_delete.setVisible(False)
        self.btn_abort_delete.setToolTip("Stop download and delete the partial file")
        self.btn_abort_delete.clicked.connect(self._abort_and_delete)

        btn_row.addWidget(self.btn_download)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_abort)
        btn_row.addWidget(self.btn_abort_delete)
        btn_row.addStretch()
        dl_l.addLayout(btn_row)
        root.addWidget(dc)
        root.addSpacing(18)

        # ── HF TRANSFORMERS SNAPSHOTS ────────────────────────────────────────
        root.addWidget(self._section("HF TRANSFORMERS SNAPSHOT"))
        hfc = self._card(); hfl = QVBoxLayout(hfc)
        hfl.setContentsMargins(16, 14, 16, 14); hfl.setSpacing(10)

        hf_note = QLabel(
            "Download a full Hugging Face Transformers runtime snapshot into "
            "localllm/hf_transformers and register it as an hf:<folder> model. "
            "Install optional dependencies with: pip install -e \".[hf]\"")
        hf_note.setWordWrap(True)
        hf_note.setObjectName("txt2_small")
        hfl.addWidget(hf_note)

        hf_preset_row = QHBoxLayout(); hf_preset_row.setSpacing(8)
        hf_preset_lbl = QLabel("Popular:")
        hf_preset_lbl.setFixedWidth(60); hf_preset_lbl.setObjectName("txt2")
        self.hf_preset_combo = QComboBox()
        self.hf_preset_combo.setFixedHeight(30)
        self.hf_preset_combo.addItem("Choose a popular HF Transformers repo...", None)
        for preset in popular_model_presets("hf_transformers"):
            self.hf_preset_combo.addItem(preset.get("label", preset.get("repo", "")), preset)
        self.hf_preset_combo.currentIndexChanged.connect(self._on_hf_preset_selected)
        hf_preset_row.addWidget(hf_preset_lbl)
        hf_preset_row.addWidget(self.hf_preset_combo, 1)
        hfl.addLayout(hf_preset_row)

        hf_repo_row = QHBoxLayout(); hf_repo_row.setSpacing(8)
        hf_repo_lbl = QLabel("Repo:")
        hf_repo_lbl.setFixedWidth(60); hf_repo_lbl.setObjectName("txt2")
        self.hf_repo_edit = QLineEdit()
        self.hf_repo_edit.setPlaceholderText("e.g. Qwen/Qwen2.5-0.5B-Instruct")
        self.hf_repo_edit.setFixedHeight(30)
        self.hf_revision_edit = QLineEdit(str(APP_CONFIG.get("hf_revision", "main")))
        self.hf_revision_edit.setFixedWidth(120)
        self.hf_revision_edit.setFixedHeight(30)
        self.hf_revision_edit.setPlaceholderText("main")
        self.btn_hf_snapshot_search = QPushButton("Inspect")
        set_button_icon(self.btn_hf_snapshot_search, "huggingface", "Inspect")
        self.btn_hf_snapshot_search.setObjectName("btn_send")
        self.btn_hf_snapshot_search.setFixedHeight(30)
        self.btn_hf_snapshot_search.setFixedWidth(92)
        self.btn_hf_snapshot_search.clicked.connect(self._hf_snapshot_search_repo)
        self.hf_repo_edit.returnPressed.connect(self._hf_snapshot_search_repo)
        hf_repo_row.addWidget(hf_repo_lbl)
        hf_repo_row.addWidget(self.hf_repo_edit, 1)
        hf_repo_row.addWidget(QLabel("Revision:"))
        hf_repo_row.addWidget(self.hf_revision_edit)
        hf_repo_row.addWidget(self.btn_hf_snapshot_search)
        hfl.addLayout(hf_repo_row)

        hf_dest_row = QHBoxLayout(); hf_dest_row.setSpacing(8)
        hf_dest_lbl = QLabel("Save to:")
        hf_dest_lbl.setFixedWidth(60); hf_dest_lbl.setObjectName("txt2")
        self.hf_dest_edit = QLineEdit(str(Path(APP_CONFIG.get("hf_transformers_dir", "localllm/hf_transformers")).expanduser()))
        self.hf_dest_edit.setReadOnly(True)
        self.hf_dest_edit.setFixedHeight(28)
        btn_hf_dest = QPushButton("Browse")
        btn_hf_dest.setFixedHeight(28); btn_hf_dest.setFixedWidth(80)
        btn_hf_dest.clicked.connect(self._browse_hf_dest)
        hf_dest_row.addWidget(hf_dest_lbl)
        hf_dest_row.addWidget(self.hf_dest_edit, 1)
        hf_dest_row.addWidget(btn_hf_dest)
        hfl.addLayout(hf_dest_row)

        self.hf_snapshot_list = QListWidget()
        self.hf_snapshot_list.setObjectName("model_list")
        self.hf_snapshot_list.setMinimumHeight(150)
        hfl.addWidget(self.hf_snapshot_list)

        self.hf_snapshot_status = QLabel("No HF snapshot inspected.")
        self.hf_snapshot_status.setWordWrap(True)
        self.hf_snapshot_status.setObjectName("txt2_small")
        hfl.addWidget(self.hf_snapshot_status)

        self.hf_snapshot_progress = QProgressBar()
        self.hf_snapshot_progress.setRange(0, 100); self.hf_snapshot_progress.setValue(0)
        self.hf_snapshot_progress.setFixedHeight(10); self.hf_snapshot_progress.setTextVisible(False)
        self.hf_snapshot_progress.setStyleSheet(
            f"QProgressBar{{background:{C['bg2']};border:1px solid {C['bdr']};border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{C['acc2']};border-radius:4px;}}")
        hfl.addWidget(self.hf_snapshot_progress)

        hf_btn_row = QHBoxLayout(); hf_btn_row.setSpacing(8)
        self.btn_hf_snapshot_download = QPushButton("Download Snapshot")
        set_button_icon(self.btn_hf_snapshot_download, "huggingface", "Download Snapshot")
        self.btn_hf_snapshot_download.setObjectName("btn_send")
        self.btn_hf_snapshot_download.setFixedHeight(32)
        self.btn_hf_snapshot_download.setEnabled(False)
        self.btn_hf_snapshot_download.clicked.connect(self._hf_snapshot_start_download)
        self.btn_hf_snapshot_pause = QPushButton("Pause")
        self.btn_hf_snapshot_pause.setObjectName("btn_stop")
        self.btn_hf_snapshot_pause.setFixedHeight(32)
        self.btn_hf_snapshot_pause.setFixedWidth(80)
        self.btn_hf_snapshot_pause.setVisible(False)
        self.btn_hf_snapshot_pause.clicked.connect(self._hf_snapshot_toggle_pause)
        self.btn_hf_snapshot_cancel = QPushButton("Cancel")
        self.btn_hf_snapshot_cancel.setObjectName("btn_stop")
        self.btn_hf_snapshot_cancel.setFixedHeight(32)
        self.btn_hf_snapshot_cancel.setFixedWidth(80)
        self.btn_hf_snapshot_cancel.setVisible(False)
        self.btn_hf_snapshot_cancel.clicked.connect(lambda: self._hf_snapshot_abort(False))
        self.btn_hf_snapshot_delete = QPushButton("Cancel & Delete")
        self.btn_hf_snapshot_delete.setObjectName("btn_stop")
        self.btn_hf_snapshot_delete.setFixedHeight(32)
        self.btn_hf_snapshot_delete.setFixedWidth(120)
        self.btn_hf_snapshot_delete.setVisible(False)
        self.btn_hf_snapshot_delete.clicked.connect(lambda: self._hf_snapshot_abort(True))
        hf_btn_row.addWidget(self.btn_hf_snapshot_download)
        hf_btn_row.addWidget(self.btn_hf_snapshot_pause)
        hf_btn_row.addWidget(self.btn_hf_snapshot_cancel)
        hf_btn_row.addWidget(self.btn_hf_snapshot_delete)
        hf_btn_row.addStretch()
        hfl.addLayout(hf_btn_row)
        root.addWidget(hfc); root.addSpacing(18)

        # ── OLLAMA MODEL PULL ────────────────────────────────────────────────
        root.addWidget(self._section("OLLAMA MODEL PULL"))
        oc = self._card(); ol = QVBoxLayout(oc)
        ol.setContentsMargins(16, 14, 16, 14); ol.setSpacing(10)

        ollama_note = QLabel(
            "Uses an already-running Ollama daemon. Pulling a model here registers "
            "it as ollama:<model> after completion.")
        ollama_note.setWordWrap(True)
        ollama_note.setObjectName("txt2_small")
        ol.addWidget(ollama_note)

        ollama_preset_row = QHBoxLayout(); ollama_preset_row.setSpacing(8)
        ollama_preset_lbl = QLabel("Popular:")
        ollama_preset_lbl.setFixedWidth(60); ollama_preset_lbl.setObjectName("txt2")
        self.ollama_preset_combo = QComboBox()
        self.ollama_preset_combo.setFixedHeight(30)
        self.ollama_preset_combo.addItem("Choose a popular Ollama model...", None)
        for preset in popular_model_presets("ollama"):
            self.ollama_preset_combo.addItem(preset.get("label", preset.get("model", "")), preset)
        self.ollama_preset_combo.currentIndexChanged.connect(self._on_ollama_preset_selected)
        ollama_preset_row.addWidget(ollama_preset_lbl)
        ollama_preset_row.addWidget(self.ollama_preset_combo, 1)
        ol.addLayout(ollama_preset_row)

        ollama_host_row = QHBoxLayout(); ollama_host_row.setSpacing(8)
        host_lbl = QLabel("Host:")
        host_lbl.setFixedWidth(60); host_lbl.setObjectName("txt2")
        self.ollama_host_edit = QLineEdit(str(APP_CONFIG.get("ollama_host", "http://127.0.0.1:11434")))
        self.ollama_host_edit.setFixedHeight(30)
        self.btn_ollama_refresh = QPushButton("Refresh")
        set_button_icon(self.btn_ollama_refresh, "ollama", "Refresh")
        self.btn_ollama_refresh.setFixedHeight(30)
        self.btn_ollama_refresh.setFixedWidth(92)
        self.btn_ollama_refresh.clicked.connect(self._ollama_refresh)
        ollama_host_row.addWidget(host_lbl)
        ollama_host_row.addWidget(self.ollama_host_edit, 1)
        ollama_host_row.addWidget(self.btn_ollama_refresh)
        ol.addLayout(ollama_host_row)

        pull_row = QHBoxLayout(); pull_row.setSpacing(8)
        pull_lbl = QLabel("Model:")
        pull_lbl.setFixedWidth(60); pull_lbl.setObjectName("txt2")
        self.ollama_model_edit = QLineEdit()
        self.ollama_model_edit.setPlaceholderText("e.g. llama3.2:3b or qwen2.5:7b")
        self.ollama_model_edit.setFixedHeight(30)
        self.btn_ollama_pull = QPushButton("Pull")
        set_button_icon(self.btn_ollama_pull, "ollama", "Pull")
        self.btn_ollama_pull.setObjectName("btn_send")
        self.btn_ollama_pull.setFixedHeight(30)
        self.btn_ollama_pull.setFixedWidth(92)
        self.btn_ollama_pull.clicked.connect(self._ollama_pull)
        self.ollama_model_edit.returnPressed.connect(self._ollama_pull)
        pull_row.addWidget(pull_lbl)
        pull_row.addWidget(self.ollama_model_edit, 1)
        pull_row.addWidget(self.btn_ollama_pull)
        ol.addLayout(pull_row)

        self.ollama_list = QListWidget()
        self.ollama_list.setObjectName("model_list")
        self.ollama_list.setMinimumHeight(110)
        self.ollama_list.itemClicked.connect(lambda item: self.ollama_model_edit.setText(item.data(Qt.ItemDataRole.UserRole) or ""))
        ol.addWidget(self.ollama_list)

        self.ollama_progress = QProgressBar()
        self.ollama_progress.setRange(0, 100); self.ollama_progress.setValue(0)
        self.ollama_progress.setFixedHeight(10); self.ollama_progress.setTextVisible(False)
        self.ollama_progress.setStyleSheet(
            f"QProgressBar{{background:{C['bg2']};border:1px solid {C['bdr']};border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{C['ok']};border-radius:4px;}}")
        ol.addWidget(self.ollama_progress)

        ollama_status_row = QHBoxLayout(); ollama_status_row.setSpacing(8)
        self.ollama_status = QLabel("Refresh to list installed Ollama models.")
        self.ollama_status.setObjectName("txt2_small")
        self.ollama_status.setWordWrap(True)
        self.btn_ollama_cancel = QPushButton("Cancel Pull")
        self.btn_ollama_cancel.setObjectName("btn_stop")
        self.btn_ollama_cancel.setFixedHeight(30)
        self.btn_ollama_cancel.setVisible(False)
        self.btn_ollama_cancel.clicked.connect(self._ollama_abort_pull)
        ollama_status_row.addWidget(self.ollama_status, 1)
        ollama_status_row.addWidget(self.btn_ollama_cancel)
        ol.addLayout(ollama_status_row)
        root.addWidget(oc)
        root.addSpacing(18)

        # ── LLAMA.CPP BINARIES ────────────────────────────────────────────────
        root.addWidget(self._section("LLAMA.CPP RUNTIME"))
        lc = self._card(); ll = QVBoxLayout(lc)
        ll.setContentsMargins(16, 14, 16, 14); ll.setSpacing(10)

        lcpp_note = QLabel(
            "Download llama.cpp prebuilt binaries directly from GitHub releases.\n"
            "They will be installed to ./llama/bin/ in your launch directory and\n"
            "picked up automatically by the Server tab.")
        lcpp_note.setWordWrap(True)
        lcpp_note.setObjectName("txt2_small")
        ll.addWidget(lcpp_note)

        fetch_row = QHBoxLayout(); fetch_row.setSpacing(8)
        self.btn_fetch_llama = QPushButton("Fetch Latest Releases")
        set_button_icon(self.btn_fetch_llama, "search", "Fetch Latest Releases")
        self.btn_fetch_llama.setObjectName("btn_send")
        self.btn_fetch_llama.setFixedHeight(30)
        self.btn_fetch_llama.clicked.connect(self._fetch_llama_releases)
        self.llama_fetch_status = QLabel("")
        self.llama_fetch_status.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        fetch_row.addWidget(self.btn_fetch_llama)
        fetch_row.addWidget(self.llama_fetch_status)
        fetch_row.addStretch()
        ll.addLayout(fetch_row)

        # Release picker
        rel_row = QHBoxLayout(); rel_row.setSpacing(8)
        rel_lbl = QLabel("Release:")
        rel_lbl.setFixedWidth(60)
        rel_lbl.setObjectName("txt2")
        self.combo_llama_release = QComboBox()
        self.combo_llama_release.setFixedHeight(28)
        self.combo_llama_release.setEnabled(False)
        self.combo_llama_release.currentIndexChanged.connect(self._on_llama_release_changed)
        rel_row.addWidget(rel_lbl)
        rel_row.addWidget(self.combo_llama_release, 1)
        ll.addLayout(rel_row)

        # Asset picker
        asset_row = QHBoxLayout(); asset_row.setSpacing(8)
        asset_lbl = QLabel("Build:")
        asset_lbl.setFixedWidth(60)
        asset_lbl.setObjectName("txt2")
        self.combo_llama_asset = QComboBox()
        self.combo_llama_asset.setFixedHeight(28)
        self.combo_llama_asset.setEnabled(False)
        asset_row.addWidget(asset_lbl)
        asset_row.addWidget(self.combo_llama_asset, 1)
        ll.addLayout(asset_row)

        # Progress + status
        self.llama_progress = QProgressBar()
        self.llama_progress.setRange(0, 100); self.llama_progress.setValue(0)
        self.llama_progress.setFixedHeight(10); self.llama_progress.setTextVisible(False)
        self.llama_progress.setStyleSheet(
            f"QProgressBar{{background:{C['bg2']};border:1px solid {C['bdr']};"
            f"border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{C['ok']};border-radius:4px;}}")
        ll.addWidget(self.llama_progress)

        self.llama_dl_status = QLabel("Not installed.")
        self.llama_dl_status.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        ll.addWidget(self.llama_dl_status)

        # Check current install
        _llama_bin = Path("./llama/bin")
        _has_server = any(_llama_bin.glob("llama-server*")) if _llama_bin.exists() else False
        if _has_server:
            self.llama_dl_status.setText(f"llama.cpp already installed at {_llama_bin.resolve()}")

        llama_btn_row = QHBoxLayout(); llama_btn_row.setSpacing(8)
        self.btn_install_llama = QPushButton("⬇️  Download & Install")
        self.btn_install_llama.setObjectName("btn_send")
        self.btn_install_llama.setFixedHeight(32)
        self.btn_install_llama.setEnabled(False)
        self.btn_install_llama.clicked.connect(self._install_llama)
        self.btn_abort_llama = QPushButton("Cancel")
        self.btn_abort_llama.setObjectName("btn_stop")
        self.btn_abort_llama.setFixedHeight(32)
        self.btn_abort_llama.setFixedWidth(80)
        self.btn_abort_llama.setVisible(False)
        self.btn_abort_llama.clicked.connect(self._abort_llama)
        llama_btn_row.addWidget(self.btn_install_llama)
        llama_btn_row.addWidget(self.btn_abort_llama)
        llama_btn_row.addStretch()
        ll.addLayout(llama_btn_row)
        root.addWidget(lc); root.addStretch()
        self.refresh_hf_auth_state()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size:12px;font-weight:bold;"
            f"letter-spacing:0.5px;padding:0;margin-bottom:2px;")
        return lbl

    @staticmethod
    def _card() -> QFrame:
        f = QFrame(); f.setObjectName("tab_card"); return f

    @staticmethod
    def _fmt_bytes(b: int) -> str:
        return f"{b/1e9:.2f} GB" if b >= 1e9 else f"{b/1e6:.1f} MB"

    def refresh_hf_auth_state(self):
        creds = load_hf_credentials()
        saved_token = str(creds.get("access_token", "") or "").strip()
        fallback_token = str(APP_CONFIG.get("hf_token", "") or "").strip()
        username = str(creds.get("username", "") or "").strip()
        last_error = str(creds.get("last_error", "") or "").strip()
        if saved_token:
            text = f"Hugging Face: signed in{f' as {username}' if username else ''} ({mask_hf_token(saved_token)})."
        elif fallback_token:
            text = f"Hugging Face: using App Configuration token fallback ({mask_hf_token(fallback_token)})."
        else:
            text = "Hugging Face: signed out. Public repositories only."
        if last_error:
            text += f" Last error: {last_error}"
        self.hf_auth_download_label.setText(text)

    def _reset_dl_ui(self, status: str = "No download in progress."):
        self.btn_download.setVisible(True)
        self.btn_download.setEnabled(True)
        self.btn_pause.setVisible(False)
        self.btn_pause.setText("Pause")
        self.btn_abort.setVisible(False)
        self.btn_abort_delete.setVisible(False)
        self.dl_status.setText(status)

    def _on_gguf_preset_selected(self, _idx: int):
        preset = self.gguf_preset_combo.currentData()
        if not preset:
            return
        repo = str(preset.get("repo", "")).strip()
        if not repo:
            return
        self.repo_edit.setText(repo)
        self.search_status.setText(
            f"Preset selected: {preset.get('size', '')} {preset.get('task', '')}. Click Search to list quants.")

    def _on_hf_preset_selected(self, _idx: int):
        preset = self.hf_preset_combo.currentData()
        if not preset:
            return
        repo = str(preset.get("repo", "")).strip()
        if not repo:
            return
        self.hf_repo_edit.setText(repo)
        self.hf_revision_edit.setText(str(preset.get("revision", "main") or "main"))
        suffix = "vision model" if preset.get("vision") else "text model"
        self.hf_snapshot_status.setText(f"Preset selected: {suffix}. Click Inspect to list snapshot files.")

    def _on_ollama_preset_selected(self, _idx: int):
        preset = self.ollama_preset_combo.currentData()
        if not preset:
            return
        model = str(preset.get("model", "")).strip()
        if not model:
            return
        self.ollama_model_edit.setText(model)
        suffix = "vision model" if preset.get("vision") else str(preset.get("task", "model"))
        self.ollama_status.setText(f"Preset selected: {model} ({preset.get('size', '')}, {suffix}). Click Pull.")

    # ── actions ───────────────────────────────────────────────────────────────
    def _do_search(self):
        repo = self.repo_edit.text().strip()
        if not repo:
            self.search_status.setText("[WARN]  Enter a repo ID first."); return
        self.search_status.setText("Querying HuggingFace API...")
        self.btn_search.setEnabled(False)
        self.results_list.clear(); self._files = []
        self.btn_download.setEnabled(False)
        if self._search_worker:
            try: self._search_worker.quit()
            except Exception: pass
        self._search_worker = HfSearchWorker(repo)
        self._search_worker.results_ready.connect(self._on_results)
        self._search_worker.err.connect(self._on_search_err)
        self._search_worker.start()

    def _on_results(self, files: list):
        self.btn_search.setEnabled(True)
        self._files = files
        self.results_list.clear()
        if not files:
            self.search_status.setText("[WARN]  No GGUF files found in this repo.")
            return
        self.search_status.setText(f"[OK]  Found {len(files)} GGUF file(s).")
        for fdata in files:
            fname  = fdata.get("rfilename", "?")
            size   = fdata.get("size", 0)
            size_s = (f"{size/1e9:.2f} GB" if size > 1e9 else
                      f"{size/1e6:.0f} MB"  if size      else "")
            quant  = detect_quant_type(fname)
            ql, qc = quant_info(quant)
            item   = QListWidgetItem(
                f"  {fname}   [{quant} · {ql}]  {size_s}")
            item.setData(Qt.ItemDataRole.UserRole, fdata)
            item.setForeground(QColor(qc))
            self.results_list.addItem(item)

    def _on_search_err(self, msg: str):
        self.btn_search.setEnabled(True)
        self.refresh_hf_auth_state()
        self.search_status.setText(f"[FAIL]  Error: {msg}")

    def _on_file_selected(self, item, _=None):
        if not item:
            self.btn_download.setEnabled(False); return
        fdata = item.data(Qt.ItemDataRole.UserRole)
        if not fdata:
            self.btn_download.setEnabled(False); return
        fname  = fdata.get("rfilename", "?")
        size   = fdata.get("size", 0)
        size_s = (f"{size/1e9:.2f} GB" if size > 1e9 else
                  f"{size/1e6:.0f} MB"  if size      else "unknown size")
        quant  = detect_quant_type(fname)
        ql, _  = quant_info(quant)
        fam    = detect_model_family(fname)
        self.file_info.setText(
            f"File: {fname}\n"
            f"Size: {size_s}   ·   Quant: {quant} ({ql})\n"
            f"Detected Family: {fam.name}   ·   Template: {fam.template}")
        self.btn_download.setEnabled(True)

    def _browse_dest(self):
        p = QFileDialog.getExistingDirectory(
            self, "Select Download Folder", self.dest_edit.text())
        if p: self.dest_edit.setText(p)

    def _start_download(self):
        item = self.results_list.currentItem()
        if not item: return
        fdata = item.data(Qt.ItemDataRole.UserRole)
        if not fdata: return
        fname = fdata.get("rfilename", "")
        repo  = self.repo_edit.text().strip()
        dest  = Path(self.dest_edit.text())

        self.btn_download.setVisible(False)
        self.btn_pause.setVisible(True)
        self.btn_pause.setText("Pause")
        self.btn_abort.setVisible(True)
        self.btn_abort_delete.setVisible(True)
        self.dl_progress.setValue(0)
        self.dl_status.setText(f"Downloading  {fname}...")

        self._dl_worker = HfDownloadWorker(
            repo, fname, dest,
            expected_size=fdata.get("size", 0)
        )
        self._dl_worker.progress.connect(self._on_dl_progress)
        self._dl_worker.done.connect(self._on_dl_done)
        self._dl_worker.err.connect(self._on_dl_err)
        self._dl_worker.paused.connect(self._on_dl_paused)
        self._dl_worker.start()

    def _toggle_pause(self):
        if not self._dl_worker: return
        if self._dl_worker.is_paused():
            self._dl_worker.resume()
        else:
            self._dl_worker.pause()

    def _on_dl_paused(self, is_paused: bool):
        if is_paused:
            self.btn_pause.setText("Resume")
            current = self.dl_status.text().replace("  [paused]", "")
            self.dl_status.setText(current + "  [paused]")
        else:
            self.btn_pause.setText("Pause")
            self.dl_status.setText(
                self.dl_status.text().replace("  [paused]", ""))

    def _abort_download(self):
        """Stop download, keep .part file so it can be resumed next session."""
        if self._dl_worker:
            self._dl_worker.abort(delete_part=False)
        self.dl_progress.setValue(0)
        self._reset_dl_ui("Download stopped. Progress saved - restart to resume.")

    def _abort_and_delete(self):
        """Stop download and wipe the partial file entirely."""
        if self._dl_worker:
            self._dl_worker.abort(delete_part=True)
        self.dl_progress.setValue(0)
        self._reset_dl_ui("Download cancelled and partial file deleted.")

    def _on_dl_progress(self, done: int, total: int):
        if total > 0:
            pct = int(done * 100 / total)
            self.dl_progress.setValue(pct)
            self.dl_status.setText(
                f"{self._fmt_bytes(done)} / {self._fmt_bytes(total)}  ({pct}%)")
        else:
            self.dl_status.setText(f"{self._fmt_bytes(done)} downloaded...")

    def _on_dl_done(self, path: str):
        self.dl_progress.setValue(100)
        self._reset_dl_ui(f"[OK]  Saved to:  {path}")
        get_model_registry().add(path)
        QMessageBox.information(
            self, "Download Complete",
            f"Model saved to:\n{path}\n\n"
            "It has been added to your model library.")

    def _on_dl_err(self, msg: str):
        self.dl_progress.setValue(0)
        self.refresh_hf_auth_state()
        self._reset_dl_ui(f"[FAIL]  Error: {msg}")

    # ── HF Transformers snapshot downloader ─────────────────────────────────

    def _browse_hf_dest(self):
        p = QFileDialog.getExistingDirectory(
            self, "Select HF Snapshot Folder", self.hf_dest_edit.text())
        if p:
            self.hf_dest_edit.setText(p)
            APP_CONFIG["hf_transformers_dir"] = p
            save_app_config(APP_CONFIG)

    def _hf_snapshot_search_repo(self):
        repo = self.hf_repo_edit.text().strip()
        if not repo:
            self.hf_snapshot_status.setText("[WARN] Enter a Hugging Face repo ID first.")
            return
        revision = self.hf_revision_edit.text().strip() or "main"
        APP_CONFIG["hf_revision"] = revision
        APP_CONFIG["hf_transformers_dir"] = self.hf_dest_edit.text().strip()
        save_app_config(APP_CONFIG)
        self.btn_hf_snapshot_search.setEnabled(False)
        self.btn_hf_snapshot_download.setEnabled(False)
        self.hf_snapshot_list.clear()
        self._hf_snapshot_files = []
        self.hf_snapshot_status.setText("Inspecting Hugging Face repo...")
        self._hf_snapshot_search = HfSnapshotSearchWorker(repo, revision)
        self._hf_snapshot_search.results_ready.connect(self._hf_snapshot_on_results)
        self._hf_snapshot_search.err.connect(self._hf_snapshot_on_search_err)
        self._hf_snapshot_search.start()

    def _hf_snapshot_on_results(self, payload: dict):
        self.btn_hf_snapshot_search.setEnabled(True)
        self._hf_snapshot_files = payload.get("files") or []
        self.hf_snapshot_list.clear()
        if not self._hf_snapshot_files:
            self.hf_snapshot_status.setText("[WARN] No Transformers runtime files found in this repo.")
            return
        for row in self._hf_snapshot_files:
            size = int(row.get("size") or 0)
            item = QListWidgetItem(f"{row.get('name', '')}   {self._fmt_bytes(size) if size else ''}")
            item.setData(Qt.ItemDataRole.UserRole, row)
            self.hf_snapshot_list.addItem(item)
        total = int(payload.get("total_size") or 0)
        self.hf_snapshot_status.setText(
            f"Found {len(self._hf_snapshot_files)} runtime file(s). "
            f"Total: {self._fmt_bytes(total) if total else 'unknown size'}")
        self.btn_hf_snapshot_download.setEnabled(True)

    def _hf_snapshot_on_search_err(self, msg: str):
        self.btn_hf_snapshot_search.setEnabled(True)
        self.refresh_hf_auth_state()
        self.hf_snapshot_status.setText(f"[FAIL] Error: {msg}")

    def _hf_snapshot_start_download(self):
        repo = self.hf_repo_edit.text().strip()
        if not repo or not self._hf_snapshot_files:
            return
        dest_root = Path(self.hf_dest_edit.text().strip() or "localllm/hf_transformers")
        revision = self.hf_revision_edit.text().strip() or "main"
        APP_CONFIG["hf_transformers_dir"] = str(dest_root)
        APP_CONFIG["hf_revision"] = revision
        save_app_config(APP_CONFIG)
        self.btn_hf_snapshot_download.setVisible(False)
        self.btn_hf_snapshot_pause.setVisible(True)
        self.btn_hf_snapshot_cancel.setVisible(True)
        self.btn_hf_snapshot_delete.setVisible(True)
        self.hf_snapshot_progress.setValue(0)
        self.hf_snapshot_status.setText("Starting HF snapshot download...")
        self._hf_snapshot_dl = HfSnapshotDownloadWorker(
            repo, revision, self._hf_snapshot_files, dest_root)
        self._hf_snapshot_dl.progress.connect(self._hf_snapshot_on_progress)
        self._hf_snapshot_dl.status.connect(lambda m: self.hf_snapshot_status.setText(m))
        self._hf_snapshot_dl.done.connect(self._hf_snapshot_on_done)
        self._hf_snapshot_dl.err.connect(self._hf_snapshot_on_err)
        self._hf_snapshot_dl.paused.connect(self._hf_snapshot_on_paused)
        self._hf_snapshot_dl.start()

    def _hf_snapshot_reset_download_ui(self, text: str):
        self.btn_hf_snapshot_download.setVisible(True)
        self.btn_hf_snapshot_download.setEnabled(bool(self._hf_snapshot_files))
        self.btn_hf_snapshot_pause.setVisible(False)
        self.btn_hf_snapshot_pause.setText("Pause")
        self.btn_hf_snapshot_cancel.setVisible(False)
        self.btn_hf_snapshot_delete.setVisible(False)
        self.hf_snapshot_status.setText(text)

    def _hf_snapshot_toggle_pause(self):
        if not self._hf_snapshot_dl:
            return
        if self._hf_snapshot_dl.is_paused():
            self._hf_snapshot_dl.resume()
        else:
            self._hf_snapshot_dl.pause()

    def _hf_snapshot_on_paused(self, paused: bool):
        self.btn_hf_snapshot_pause.setText("Resume" if paused else "Pause")

    def _hf_snapshot_abort(self, delete_part: bool):
        if self._hf_snapshot_dl:
            self._hf_snapshot_dl.abort(delete_part=delete_part)
        self.hf_snapshot_progress.setValue(0)
        self._hf_snapshot_reset_download_ui(
            "HF snapshot cancelled and partial files deleted." if delete_part
            else "HF snapshot stopped. Partial files were kept for resume.")

    def _hf_snapshot_on_progress(self, done: int, total: int, current: str):
        if total > 0:
            pct = max(0, min(100, int(done * 100 / total)))
            self.hf_snapshot_progress.setValue(pct)
            self.hf_snapshot_status.setText(
                f"{self._fmt_bytes(done)} / {self._fmt_bytes(total)} ({pct}%)  -  {current}")
        else:
            self.hf_snapshot_status.setText(f"{self._fmt_bytes(done)} downloaded  -  {current}")

    def _hf_snapshot_on_done(self, path: str):
        self.hf_snapshot_progress.setValue(100)
        ref = make_hf_model_ref(path)
        get_model_registry().add(ref)
        self._hf_snapshot_dl = None
        self._hf_snapshot_reset_download_ui(f"[OK] HF snapshot saved and registered: {ref}")
        QMessageBox.information(
            self, "HF Snapshot Complete",
            f"Snapshot saved to:\n{path}\n\nRegistered as:\n{ref}")

    def _hf_snapshot_on_err(self, msg: str):
        self.hf_snapshot_progress.setValue(0)
        self._hf_snapshot_dl = None
        self.refresh_hf_auth_state()
        self._hf_snapshot_reset_download_ui(f"[FAIL] Error: {msg}")

    # ── Ollama puller ────────────────────────────────────────────────────────

    def _ollama_save_host(self) -> str:
        host = normalize_ollama_host(self.ollama_host_edit.text())
        self.ollama_host_edit.setText(host)
        APP_CONFIG["ollama_host"] = host
        save_app_config(APP_CONFIG)
        return host

    def _ollama_refresh(self):
        host = self._ollama_save_host()
        self.btn_ollama_refresh.setEnabled(False)
        self.ollama_status.setText("Connecting to Ollama...")
        self.ollama_list.clear()
        self._ollama_list_worker = OllamaListWorker(host)
        self._ollama_list_worker.results_ready.connect(self._ollama_on_list)
        self._ollama_list_worker.err.connect(self._ollama_on_list_err)
        self._ollama_list_worker.start()

    def _ollama_on_list(self, rows: list):
        self.btn_ollama_refresh.setEnabled(True)
        self.ollama_list.clear()
        for row in rows:
            name = str(row.get("name") or row.get("model") or "")
            if not name:
                continue
            size = int(row.get("size") or 0)
            item = QListWidgetItem(f"{name}   {self._fmt_bytes(size) if size else ''}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.ollama_list.addItem(item)
        self.ollama_status.setText(f"Found {self.ollama_list.count()} installed Ollama model(s).")

    def _ollama_on_list_err(self, msg: str):
        self.btn_ollama_refresh.setEnabled(True)
        self.ollama_status.setText(f"[FAIL] Ollama connection error: {msg}")

    def _ollama_pull(self):
        model = self.ollama_model_edit.text().strip()
        if not model:
            self.ollama_status.setText("[WARN] Enter an Ollama model name first.")
            return
        host = self._ollama_save_host()
        self.btn_ollama_pull.setEnabled(False)
        self.btn_ollama_cancel.setVisible(True)
        self.ollama_progress.setValue(0)
        self.ollama_status.setText(f"Pulling {model}...")
        self._ollama_pull_worker = OllamaPullWorker(host, model)
        self._ollama_pull_worker.progress.connect(self._ollama_on_pull_progress)
        self._ollama_pull_worker.status.connect(lambda m: self.ollama_status.setText(m))
        self._ollama_pull_worker.done.connect(self._ollama_on_pull_done)
        self._ollama_pull_worker.err.connect(self._ollama_on_pull_err)
        self._ollama_pull_worker.start()

    def _ollama_abort_pull(self):
        if self._ollama_pull_worker:
            self._ollama_pull_worker.abort()
        self.btn_ollama_pull.setEnabled(True)
        self.btn_ollama_cancel.setVisible(False)
        self.ollama_progress.setValue(0)
        self.ollama_status.setText("Ollama pull cancelled.")

    def _ollama_on_pull_progress(self, done: int, total: int, status: str):
        if total > 0:
            pct = max(0, min(100, int(done * 100 / total)))
            self.ollama_progress.setValue(pct)
            self.ollama_status.setText(
                f"{status}  {self._fmt_bytes(done)} / {self._fmt_bytes(total)} ({pct}%)")
        elif status:
            self.ollama_status.setText(status)

    def _ollama_on_pull_done(self, model: str):
        ref = make_ollama_model_ref(model)
        get_model_registry().add(ref)
        self.btn_ollama_pull.setEnabled(True)
        self.btn_ollama_cancel.setVisible(False)
        self.ollama_progress.setValue(100)
        self.ollama_status.setText(f"[OK] Pulled and registered: {ref}")
        self._ollama_pull_worker = None
        self._ollama_refresh()

    def _ollama_on_pull_err(self, msg: str):
        self.btn_ollama_pull.setEnabled(True)
        self.btn_ollama_cancel.setVisible(False)
        self.ollama_progress.setValue(0)
        self.ollama_status.setText(f"[FAIL] Ollama pull error: {msg}")
        self._ollama_pull_worker = None

    # ── llama.cpp downloader ──────────────────────────────────────────────────

    def _fetch_llama_releases(self):
        self.btn_fetch_llama.setEnabled(False)
        self.llama_fetch_status.setText("Fetching releases from GitHub…")
        self.combo_llama_release.setEnabled(False)
        self.combo_llama_asset.setEnabled(False)
        self.btn_install_llama.setEnabled(False)
        self._llama_releases = []
        if self._llama_fetcher:
            try: self._llama_fetcher.quit()
            except Exception: pass
        self._llama_fetcher = LlamaCppReleaseFetcher()
        self._llama_fetcher.results_ready.connect(self._on_llama_releases)
        self._llama_fetcher.err.connect(self._on_llama_fetch_err)
        self._llama_fetcher.start()

    def _on_llama_releases(self, releases: list):
        self.btn_fetch_llama.setEnabled(True)
        self._llama_releases = releases
        self.combo_llama_release.blockSignals(True)
        self.combo_llama_release.clear()
        for rel in releases:
            self.combo_llama_release.addItem(rel["tag"])
        self.combo_llama_release.blockSignals(False)
        self.combo_llama_release.setEnabled(True)
        if releases:
            self.llama_fetch_status.setText(f"Found {len(releases)} release(s).")
            self._on_llama_release_changed(0)
        else:
            self.llama_fetch_status.setText("No compatible releases found for your platform.")

    def _on_llama_fetch_err(self, msg: str):
        self.btn_fetch_llama.setEnabled(True)
        self.llama_fetch_status.setText(f"Error: {msg}")

    def _on_llama_release_changed(self, idx: int):
        self.combo_llama_asset.clear()
        if not self._llama_releases or idx >= len(self._llama_releases):
            return
        assets = self._llama_releases[idx]["assets"]
        for a in assets:
            size_s = f"{a['size']/1e6:.0f} MB" if a["size"] else ""
            self.combo_llama_asset.addItem(f"{a['name']}  {size_s}", a)
        self.combo_llama_asset.setEnabled(True)
        self.btn_install_llama.setEnabled(True)

    def _install_llama(self):
        idx = self.combo_llama_asset.currentIndex()
        if idx < 0: return
        asset = self.combo_llama_asset.itemData(idx)
        if not asset: return

        self.btn_install_llama.setEnabled(False)
        self.btn_abort_llama.setVisible(True)
        self.llama_progress.setValue(0)
        self.llama_dl_status.setText(f"Starting download of {asset['name']}…")

        dest = Path("./llama")
        dest.mkdir(exist_ok=True)
        self._llama_dl = LlamaCppDownloadWorker(
            url=asset["url"], filename=asset["name"],
            dest_dir=dest, expected_size=asset.get("size", 0))
        self._llama_dl.progress.connect(self._on_llama_progress)
        self._llama_dl.status.connect(lambda m: self.llama_dl_status.setText(m))
        self._llama_dl.done.connect(self._on_llama_done)
        self._llama_dl.err.connect(self._on_llama_err)
        self._llama_dl.start()

    def _abort_llama(self):
        if self._llama_dl:
            self._llama_dl.abort()
        self.btn_abort_llama.setVisible(False)
        self.btn_install_llama.setEnabled(True)
        self.llama_dl_status.setText("Download cancelled.")
        self.llama_progress.setValue(0)

    def _on_llama_progress(self, done: int, total: int):
        if total > 0:
            self.llama_progress.setValue(int(done * 100 / total))
            self.llama_dl_status.setText(
                f"{done/1e6:.1f} MB / {total/1e6:.1f} MB  ({int(done*100/total)}%)")

    def _on_llama_done(self, path: str):
        self.llama_progress.setValue(100)
        self.btn_abort_llama.setVisible(False)
        self.btn_install_llama.setEnabled(True)
        self.llama_dl_status.setText(f"Installed to {path}")
        self._llama_dl = None
        QMessageBox.information(
            self, "llama.cpp Installed",
            f"Binaries installed to:\n{path}\n\n"
            "Go to the Server tab, leave the binary paths blank,\n"
            "and reload your model - they will be picked up automatically.")

    def _on_llama_err(self, msg: str):
        self.llama_progress.setValue(0)
        self.btn_abort_llama.setVisible(False)
        self.btn_install_llama.setEnabled(True)
        self.llama_dl_status.setText(f"Error: {msg}")
        self._llama_dl = None
                
class McpTab(QWidget):
    """
    MCP (Model Context Protocol) server management.
    Configure and launch stdio / SSE MCP servers.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._servers: list = self._load_servers()
        self._procs:   dict = {}   # name → subprocess.Popen
        self._build()

    # ── persistence ───────────────────────────────────────────────────────────
    def _load_servers(self) -> list:
        if MCP_CONFIG_FILE.exists():
            try:
                return json.loads(MCP_CONFIG_FILE.read_text()).get("servers", [])
            except Exception:
                pass
        return []

    def _save_servers(self):
        MCP_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        MCP_CONFIG_FILE.write_text(
            json.dumps({"servers": self._servers}, indent=2))

    # ── build ─────────────────────────────────────────────────────────────────
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("chat_scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); inner.setObjectName("chat_container")
        root  = QVBoxLayout(inner)
        root.setContentsMargins(22, 18, 22, 22); root.setSpacing(0)
        scroll.setWidget(inner); outer.addWidget(scroll)

        # Header
        hdr = QLabel("MCP - Model Context Protocol")
        set_label_icon(hdr, "mcp", "MCP - Model Context Protocol", 18)
        hdr.setStyleSheet(
            "font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        sub = QLabel(
            "MCP connects your LLM to external tools and data sources "
            "(filesystem, web search, databases, APIs…) via standardised servers. "
            "Add stdio or SSE transports below. Stdio servers launch as child processes.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{C['txt2']};font-size:11px;margin-bottom:14px;")
        root.addWidget(sub)

        # ── SERVER LIST ───────────────────────────────────────────────────────
        root.addWidget(self._section("CONFIGURED SERVERS"))
        lc = self._card(); ll = QVBoxLayout(lc)
        ll.setContentsMargins(16, 14, 16, 14); ll.setSpacing(8)

        self.server_list = QListWidget()
        self.server_list.setObjectName("model_list")
        self.server_list.setMinimumHeight(160)
        self.server_list.currentItemChanged.connect(self._on_srv_selected)
        ll.addWidget(self.server_list)

        lb_row = QHBoxLayout(); lb_row.setSpacing(8)
        self.btn_start = QPushButton("Start")
        set_button_icon(self.btn_start, "play", "Start")
        self.btn_start.setObjectName("btn_send"); self.btn_start.setFixedHeight(28)
        self.btn_start.setEnabled(False); self.btn_start.clicked.connect(self._start_server)
        self.btn_stop = QPushButton("Stop")
        set_button_icon(self.btn_stop, "stop-circle", "Stop")
        self.btn_stop.setObjectName("btn_stop"); self.btn_stop.setFixedHeight(28)
        self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._stop_server)
        self.btn_del_srv = QPushButton("Remove")
        set_button_icon(self.btn_del_srv, "delete", "Remove")
        self.btn_del_srv.setFixedHeight(28); self.btn_del_srv.setEnabled(False)
        self.btn_del_srv.clicked.connect(self._remove_server)
        for b in (self.btn_start, self.btn_stop, self.btn_del_srv):
            lb_row.addWidget(b)
        lb_row.addStretch()
        ll.addLayout(lb_row)
        root.addWidget(lc); root.addSpacing(14)

        # ── ADD SERVER ────────────────────────────────────────────────────────
        root.addWidget(self._section("ADD SERVER"))
        ac = self._card(); al = QVBoxLayout(ac)
        al.setContentsMargins(16, 14, 16, 14); al.setSpacing(10)

        help_txt = QLabel(
            "stdio example:  npx -y @modelcontextprotocol/server-filesystem  ~/Documents\n"
            "SSE example:    http://localhost:8080/sse")
        help_txt.setWordWrap(True)
        help_txt.setStyleSheet(f"color:{C['txt2']};font-size:10px;"
                               f"background:{C['bg2']};border-radius:5px;padding:6px 8px;")
        al.addWidget(help_txt)

        def _lrow(label: str, widget, width=100):
            r = QHBoxLayout(); r.setSpacing(8)
            l = QLabel(label); l.setFixedWidth(width)
            l.setObjectName("txt2")
            r.addWidget(l); r.addWidget(widget, 1); return r

        self.add_name = QLineEdit()
        self.add_name.setPlaceholderText("Display name")
        self.add_name.setFixedHeight(28)
        al.addLayout(_lrow("Name:", self.add_name))

        self.add_transport = QComboBox()
        self.add_transport.addItem("stdio  (local process)", "stdio")
        self.add_transport.addItem("sse    (HTTP SSE endpoint)", "sse")
        self.add_transport.setFixedHeight(28)
        al.addLayout(_lrow("Transport:", self.add_transport))

        self.add_cmd = QLineEdit()
        self.add_cmd.setPlaceholderText(
            "Command or URL - e.g.  npx -y @mcp/server-filesystem /path")
        self.add_cmd.setFixedHeight(28)
        al.addLayout(_lrow("Command / URL:", self.add_cmd))

        self.add_desc = QLineEdit()
        self.add_desc.setPlaceholderText("Optional description")
        self.add_desc.setFixedHeight(28)
        al.addLayout(_lrow("Description:", self.add_desc))

        btn_add = QPushButton("Add Server")
        set_button_icon(btn_add, "plus", "Add Server")
        btn_add.setObjectName("btn_send"); btn_add.setFixedHeight(30)
        btn_add.clicked.connect(self._add_server)
        al.addWidget(btn_add)
        root.addWidget(ac); root.addSpacing(14)

        # ── SERVER LOG ────────────────────────────────────────────────────────
        root.addWidget(self._section("SERVER LOG"))
        log_c = self._card(); log_l = QVBoxLayout(log_c)
        log_l.setContentsMargins(16, 14, 16, 14)
        self.mcp_log = QTextEdit()
        self.mcp_log.setReadOnly(True)
        self.mcp_log.setObjectName("log_te")
        self.mcp_log.setFixedHeight(130)
        self.mcp_log.setFont(QFont("Consolas", 10))
        log_l.addWidget(self.mcp_log)
        root.addWidget(log_c); root.addStretch()

        self._refresh_list()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size:12px;font-weight:bold;"
            f"letter-spacing:0.5px;padding:0;margin-bottom:2px;")
        return lbl

    @staticmethod
    def _card() -> QFrame:
        f = QFrame(); f.setObjectName("tab_card"); return f

    def _mcp_log_msg(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.mcp_log.append(f"[{ts}]  {msg}")

    def _refresh_list(self):
        self.server_list.clear()
        for s in self._servers:
            running = (s["name"] in self._procs and
                       self._procs[s["name"]].poll() is None)
            label = (f"  {s['name']}"
                     f"   [{s['transport']}]"
                     f"   {s.get('desc', '') or s['cmd'][:50]}")
            item = QListWidgetItem(label)
            item.setIcon(status_icon("ok" if running else "idle"))
            item.setData(Qt.ItemDataRole.UserRole, s["name"])
            item.setForeground(QColor(C["ok"] if running else C["txt"]))
            self.server_list.addItem(item)

    def _on_srv_selected(self, item, _=None):
        ok = item is not None
        self.btn_del_srv.setEnabled(ok)
        if ok:
            name    = item.data(Qt.ItemDataRole.UserRole)
            running = (name in self._procs and
                       self._procs[name].poll() is None)
            self.btn_start.setEnabled(not running)
            self.btn_stop.setEnabled(running)
        else:
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(False)

    def _add_server(self):
        name = self.add_name.text().strip()
        cmd  = self.add_cmd.text().strip()
        if not name or not cmd:
            QMessageBox.warning(self, "Missing Fields",
                                "Name and Command/URL are required."); return
        if any(s["name"] == name for s in self._servers):
            QMessageBox.warning(self, "Duplicate",
                                f'A server named "{name}" already exists.'); return
        self._servers.append({
            "name":      name,
            "transport": self.add_transport.currentData(),
            "cmd":       cmd,
            "desc":      self.add_desc.text().strip(),
        })
        self._save_servers()
        self.add_name.clear(); self.add_cmd.clear(); self.add_desc.clear()
        self._refresh_list()
        self._mcp_log_msg(f"Added server: {name}")

    def _remove_server(self):
        item = self.server_list.currentItem()
        if not item: return
        name = item.data(Qt.ItemDataRole.UserRole)
        self._stop_by_name(name)
        self._servers = [s for s in self._servers if s["name"] != name]
        self._save_servers(); self._refresh_list()

    def _start_server(self):
        item = self.server_list.currentItem()
        if not item: return
        name = item.data(Qt.ItemDataRole.UserRole)
        srv  = next((s for s in self._servers if s["name"] == name), None)
        if not srv: return
        if srv["transport"] == "sse":
            self._mcp_log_msg(
                f"SSE server '{name}' - connect at: {srv['cmd']}")
            self._refresh_list(); return
        try:
            proc = subprocess.Popen(
                srv["cmd"], shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self._procs[name] = proc
            self._mcp_log_msg(f"Started '{name}'  (PID {proc.pid})")
        except Exception as e:
            self._mcp_log_msg(f"Failed to start '{name}': {e}")
        self._refresh_list(); self._on_srv_selected(item)

    def _stop_server(self):
        item = self.server_list.currentItem()
        if not item: return
        name = item.data(Qt.ItemDataRole.UserRole)
        self._stop_by_name(name)
        self._refresh_list(); self._on_srv_selected(item)

    def _stop_by_name(self, name: str):
        proc = self._procs.get(name)
        if proc:
            try: proc.terminate(); proc.wait(LONG_TIMEOUT_SECONDS)
            except Exception:
                try: proc.kill()
                except Exception: pass
            del self._procs[name]
            self._mcp_log_msg(f"Stopped '{name}'")
API_REGISTRY = ApiRegistry()

class ApiModelsTab(QWidget):
    
    """Tab for connecting to cloud/local API models. Once verified, treated as a normal engine."""
    api_model_loaded = pyqtSignal(object)   # emits ApiEngine

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tester = None
        self._build()
        self._refresh_saved()

    @staticmethod
    def _sec_lbl(text: str) -> QLabel:
        lb = QLabel(text)
        lb.setObjectName("sec_lbl")
        return lb

    @staticmethod
    def _inp(placeholder: str = "", pw: bool = False) -> QLineEdit:
        e = QLineEdit()
        e.setPlaceholderText(placeholder)
        e.setFixedHeight(32)
        e.setObjectName("input")
        if pw:
            e.setEchoMode(QLineEdit.EchoMode.Password)
        return e

    @staticmethod
    def _combo_style() -> str:
        # No longer used - combos are styled via QSS objectName "combo"
        return ""

    @staticmethod
    def _action_btn(label: str, color: str) -> QPushButton:
        b = QPushButton(label)
        b.setFixedHeight(34)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setObjectName("outline_btn")
        # Store the semantic color role as a dynamic property so QSS can target it
        b.setProperty("btn_color", color)
        return b

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(14)

        hdr = QLabel("API Models")
        set_label_icon(hdr, "api", "API Models", 18)
        hdr.setStyleSheet("font-size:15px;font-weight:700;")
        root.addWidget(hdr)

        sub = QLabel("Connect to any OpenAI-compatible or Anthropic endpoint. "
                     "Once verified, the API model is treated exactly like a local model.")
        sub.setWordWrap(True)
        sub.setObjectName("txt2_small")
        root.addWidget(sub)

        # ── Connection card (scrollable so custom-format fields never overlap) ─
        card_scroll = QScrollArea()
        card_scroll.setWidgetResizable(True)
        card_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        card_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        card_scroll.setObjectName("card_scroll")

        card = QWidget()
        card.setObjectName("card_inner")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(18, 16, 18, 16)
        cl.setSpacing(12)
        cl.addWidget(self._sec_lbl("NEW CONNECTION"))

        # Provider + Model row
        r1 = QHBoxLayout(); r1.setSpacing(12)

        lp = QVBoxLayout(); lp.setSpacing(4)
        lp.addWidget(QLabel("Provider"))
        self.combo_provider = QComboBox()
        self.combo_provider.setFixedHeight(32)
        self.combo_provider.setObjectName("combo")
        for p in API_PROVIDERS:
            self.combo_provider.addItem(p)
        self.combo_provider.currentTextChanged.connect(self._on_provider_changed)
        lp.addWidget(self.combo_provider)
        r1.addLayout(lp, 1)

        lm = QVBoxLayout(); lm.setSpacing(4)
        lm.addWidget(QLabel("Model"))
        self.combo_model = QComboBox()
        self.combo_model.setFixedHeight(32)
        self.combo_model.setEditable(True)
        self.combo_model.setObjectName("combo")
        lm.addWidget(self.combo_model)
        r1.addLayout(lm, 2)
        cl.addLayout(r1)

        # API Key
        kl = QVBoxLayout(); kl.setSpacing(4)
        kl.addWidget(QLabel("API Key"))
        self.inp_key = self._inp("sk-…  (leave blank for Ollama / no-auth endpoints)", pw=True)
        kl.addWidget(self.inp_key)
        cl.addLayout(kl)

        # Base URL + context hint row
        r3 = QHBoxLayout(); r3.setSpacing(12)
        ul = QVBoxLayout(); ul.setSpacing(4)
        ul.addWidget(QLabel("Base URL"))
        self.inp_url = self._inp("https://api.openai.com/v1")
        ul.addWidget(self.inp_url)
        r3.addLayout(ul, 3)
        tl = QVBoxLayout(); tl.setSpacing(4)
        tl.addWidget(QLabel("Context Hint"))
        self.inp_tokens = self._inp("8192")
        self.inp_tokens.setFixedWidth(96)
        tl.addWidget(self.inp_tokens)
        r3.addLayout(tl)
        cl.addLayout(r3)

        # Config name + custom provider name
        r4 = QHBoxLayout(); r4.setSpacing(12)
        nl = QVBoxLayout(); nl.setSpacing(4)
        nl.addWidget(QLabel("Config Name  (for saving)"))
        self.inp_name = self._inp("e.g. My GPT-4o")
        nl.addWidget(self.inp_name)
        r4.addLayout(nl, 2)
        pnl = QVBoxLayout(); pnl.setSpacing(4)
        pnl.addWidget(QLabel("Custom Provider Label  (optional)"))
        self.inp_custom_provider = self._inp("e.g. My Company API")
        pnl.addWidget(self.inp_custom_provider)
        r4.addLayout(pnl, 2)
        cl.addLayout(r4)

        # ── Prompt format section ─────────────────────────────────────────────
        sep_pf = QFrame()
        sep_pf.setFrameShape(QFrame.Shape.HLine)
        cl.addWidget(sep_pf)

        pf_hdr = QHBoxLayout()
        pf_lbl = self._sec_lbl("PROMPT FORMAT")
        self.chk_custom_prompt = ToggleSwitch("Use custom format")
        self.chk_custom_prompt.toggled.connect(self._toggle_prompt_format)
        pf_hdr.addWidget(pf_lbl)
        pf_hdr.addStretch()
        pf_hdr.addWidget(self.chk_custom_prompt)
        cl.addLayout(pf_hdr)

        # Template picker row
        self.prompt_format_widget = QWidget()
        pfw = QVBoxLayout(self.prompt_format_widget)
        pfw.setContentsMargins(0, 0, 0, 0)
        pfw.setSpacing(10)

        tr = QHBoxLayout(); tr.setSpacing(12)
        tl2 = QVBoxLayout(); tl2.setSpacing(4)
        tl2.addWidget(QLabel("Template Preset"))
        self.combo_template = QComboBox()
        self.combo_template.setFixedHeight(32)
        self.combo_template.setObjectName("combo")
        for key, val in PROMPT_TEMPLATES.items():
            self.combo_template.addItem(val["label"], key)
        self.combo_template.currentIndexChanged.connect(self._on_template_changed)
        tl2.addWidget(self.combo_template)
        tr.addLayout(tl2, 1)
        pfw.addLayout(tr)

        # System prompt
        spl = QVBoxLayout(); spl.setSpacing(4)
        spl.addWidget(QLabel("System Prompt"))
        self.inp_system = QTextEdit()
        self.inp_system.setFixedHeight(64)
        self.inp_system.setPlaceholderText("System prompt / instruction prefix (optional)")
        spl.addWidget(self.inp_system)
        pfw.addLayout(spl)

        # Prefix / suffix rows
        pfx_row = QHBoxLayout(); pfx_row.setSpacing(12)
        upl = QVBoxLayout(); upl.setSpacing(4)
        upl.addWidget(QLabel("User Prefix"))
        self.inp_user_prefix = self._inp(r"e.g. [INST] ")
        upl.addWidget(self.inp_user_prefix)
        pfx_row.addLayout(upl)
        usl = QVBoxLayout(); usl.setSpacing(4)
        usl.addWidget(QLabel("User Suffix"))
        self.inp_user_suffix = self._inp(r"e.g.  [/INST]")
        usl.addWidget(self.inp_user_suffix)
        pfx_row.addLayout(usl)
        apl = QVBoxLayout(); apl.setSpacing(4)
        apl.addWidget(QLabel("Assistant Prefix"))
        self.inp_asst_prefix = self._inp(r"e.g. <|assistant|>\n")
        apl.addWidget(self.inp_asst_prefix)
        pfx_row.addLayout(apl)
        pfw.addLayout(pfx_row)

        self.prompt_format_widget.setVisible(False)
        cl.addWidget(self.prompt_format_widget)
        cl.addStretch()

        card_scroll.setWidget(card)

        # Buttons + status
        br = QHBoxLayout(); br.setSpacing(10)
        self.btn_test = self._action_btn("Test & Load", "ok")
        self.btn_save = self._action_btn("Save Config",  "acc")
        set_button_icon(self.btn_test, "zap", "Test & Load")
        set_button_icon(self.btn_save, "save", "Save Config")
        self.btn_test.clicked.connect(self._test_and_load)
        self.btn_save.clicked.connect(self._save_config)
        br.addWidget(self.btn_test)
        br.addWidget(self.btn_save)
        br.addStretch()
        cl.addLayout(br)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("api_status")
        set_status_label(self.lbl_status, "Not connected", "idle")
        cl.addWidget(self.lbl_status)
        root.addWidget(card_scroll, 2)

        # ── Saved configs ─────────────────────────────────────────────────────
        root.addWidget(self._sec_lbl("SAVED CONFIGS"))

        saved_scroll = QScrollArea()
        saved_scroll.setWidgetResizable(True)
        saved_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.saved_container = QWidget()
        self.saved_vbox = QVBoxLayout(self.saved_container)
        self.saved_vbox.setContentsMargins(0, 4, 0, 4)
        self.saved_vbox.setSpacing(6)
        self.saved_vbox.addStretch()
        saved_scroll.setWidget(self.saved_container)
        root.addWidget(saved_scroll, 1)

        self._on_provider_changed(self.combo_provider.currentText())

    def _toggle_prompt_format(self, checked: bool):
        self.prompt_format_widget.setVisible(checked)
        if checked:
            fade_in(self.prompt_format_widget, 180)

    def _on_template_changed(self, _idx: int):
        key = self.combo_template.currentData()
        if not key or key == "custom":
            return
        t = PROMPT_TEMPLATES[key]
        self.inp_system.setPlainText(t["system"])
        self.inp_user_prefix.setText(t["user_prefix"])
        self.inp_user_suffix.setText(t["user_suffix"])
        self.inp_asst_prefix.setText(t["assistant_prefix"])

    def _on_provider_changed(self, provider: str):
        info = API_PROVIDERS.get(provider, {})
        self.combo_model.clear()
        for m in info.get("models", []):
            self.combo_model.addItem(m)
        self.inp_url.setText(info.get("base_url", ""))

    def _collect_config(self) -> ApiConfig:
        provider   = self.combo_provider.currentText()
        model_id   = self.combo_model.currentText().strip()
        api_key    = self.inp_key.text().strip()
        base_url   = self.inp_url.text().strip()
        api_format = API_PROVIDERS.get(provider, {}).get("format", "openai")
        try:    max_tok = int(self.inp_tokens.text())
        except: max_tok = 8192
        custom_prov = self.inp_custom_provider.text().strip()
        name = self.inp_name.text().strip() or f"{custom_prov or provider} {model_id}"

        use_custom = self.chk_custom_prompt.isChecked()
        tmpl_key   = self.combo_template.currentData() or "default"
        return ApiConfig(
            name=name, provider=provider, model_id=model_id,
            api_key=api_key, base_url=base_url,
            api_format=api_format, max_tokens=max_tok,
            custom_provider_name=custom_prov,
            use_custom_prompt=use_custom,
            prompt_template=tmpl_key,
            system_prompt=(self.inp_system.toPlainText().strip() if use_custom else ""),
            user_prefix=(self.inp_user_prefix.text() if use_custom else ""),
            user_suffix=(self.inp_user_suffix.text() if use_custom else ""),
            assistant_prefix=(self.inp_asst_prefix.text() if use_custom else ""),
        )

    def _run_test(self, cfg: ApiConfig):
        self.btn_test.setEnabled(False)
        set_status_label(self.lbl_status, "Testing connection...", "loading")
        self.lbl_status.setProperty("state", "warn")
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)

        class _T(QThread):
            finished = pyqtSignal(bool, str, object)
            def __init__(self, c): super().__init__(); self.cfg = c
            def run(self):
                eng = ApiEngine()
                ok  = eng.load(self.cfg)
                self.finished.emit(ok, eng.status_text if ok else
                                   "Connection failed - check key / URL / model ID", eng if ok else None)

        self._tester = _T(cfg)
        self._tester.finished.connect(self._on_test_done)
        self._tester.start()

    def _test_and_load(self):
        cfg = self._collect_config()
        if not cfg.model_id:
            set_status_label(self.lbl_status, "Select or enter a model ID first", "warn")
            self.lbl_status.setProperty("state", "warn")
            self.lbl_status.style().unpolish(self.lbl_status)
            self.lbl_status.style().polish(self.lbl_status)
            return
        self._run_test(cfg)

    def _on_test_done(self, ok: bool, msg: str, engine):
        self.btn_test.setEnabled(True)
        state = "ok" if ok else "err"
        prefix = "OK" if ok else "Error"
        set_status_label(self.lbl_status, f"{prefix}: {msg}", state)
        self.lbl_status.setProperty("state", state)
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)
        if ok:
            self.api_model_loaded.emit(engine)
        self._tester = None

    def _save_config(self):
        cfg = self._collect_config()
        if not cfg.model_id:
            return
        getapi_registry().add(cfg)
        self._refresh_saved()

    def _refresh_saved(self):
        while self.saved_vbox.count() > 1:
            item = self.saved_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for cfg in getapi_registry().all():
            self._add_saved_card(cfg)

    def _add_saved_card(self, cfg: ApiConfig):
        card = QFrame()
        card.setObjectName("card")
        cl = QHBoxLayout(card)
        cl.setContentsMargins(14, 10, 14, 10)
        cl.setSpacing(10)

        il = QVBoxLayout(); il.setSpacing(2)
        t = QLabel(f"<b>{cfg.name}</b>")
        t.setTextFormat(Qt.TextFormat.RichText)
        t.setStyleSheet("font-size:12px;")
        prov_display = getattr(cfg, "custom_provider_name", "") or cfg.provider
        fmt_badge    = "  ·  custom fmt" if getattr(cfg, "use_custom_prompt", False) else ""
        s = QLabel(f"{prov_display}  ·  {cfg.model_id}{fmt_badge}")
        s.setObjectName("txt2_small")
        il.addWidget(t); il.addWidget(s)
        cl.addLayout(il, 1)

        def _sb(label, role):
            b = QPushButton(label)
            b.setFixedHeight(28)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setObjectName("outline_btn")
            b.setProperty("btn_color", role)
            return b

        bl = _sb("Load", "ok")
        bd = _sb("", "err")
        set_button_icon(bl, "play", "Load")
        set_button_icon(bd, "delete", "", 15)
        bl.clicked.connect(lambda _, c=cfg: self._load_saved(c))
        bd.clicked.connect(lambda _, c=cfg: self._delete_saved(c))
        cl.addWidget(bl); cl.addWidget(bd)
        self.saved_vbox.insertWidget(self.saved_vbox.count() - 1, card)
        fade_in(card, 200)

    def _load_saved(self, cfg: ApiConfig):
        idx = self.combo_provider.findText(cfg.provider)
        if idx >= 0: self.combo_provider.setCurrentIndex(idx)
        self.inp_url.setText(cfg.base_url)
        self.inp_key.setText(cfg.api_key)
        self.inp_tokens.setText(str(cfg.max_tokens))
        self.inp_name.setText(cfg.name)
        self.inp_custom_provider.setText(getattr(cfg, "custom_provider_name", ""))
        mi = self.combo_model.findText(cfg.model_id)
        if mi < 0: self.combo_model.addItem(cfg.model_id)
        self.combo_model.setCurrentText(cfg.model_id)

        use_custom = getattr(cfg, "use_custom_prompt", False)
        self.chk_custom_prompt.setChecked(use_custom)
        if use_custom:
            tmpl_key = getattr(cfg, "prompt_template", "custom")
            ti = self.combo_template.findData(tmpl_key)
            if ti >= 0: self.combo_template.setCurrentIndex(ti)
            self.inp_system.setPlainText(getattr(cfg, "system_prompt", ""))
            self.inp_user_prefix.setText(getattr(cfg, "user_prefix", ""))
            self.inp_user_suffix.setText(getattr(cfg, "user_suffix", ""))
            self.inp_asst_prefix.setText(getattr(cfg, "assistant_prefix", ""))
        self._run_test(cfg)

    def _delete_saved(self, cfg: ApiConfig):
        ans = QMessageBox.question(
            self, "Delete Config", f"Delete '{cfg.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Yes:
            getapi_registry().remove(cfg.name)
            self._refresh_saved()
