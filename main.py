"""
Native Lab Pro v2 — Local LLM Desktop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v2 New Features:
  · Full GGUF quant format detection (Q2_K → F32, IQ* imatrix quants)
  · Auto model-family detection from filename → correct prompt template
    (DeepSeek, Mistral, LLaMA-2/3, Phi, Qwen/ChatML, Gemma, CodeLlama,
     Falcon, Vicuna, OpenChat, Neural-Chat, Starling, Yi, Command-R…)
  · Parallel model loading toggle with CPU/RAM warnings
  · Pipeline mode: Reasoning → Coding chain
    (reasoning model summarises intent → coding model generates code)
  · Python/code snippet copy buttons inside chat bubbles
  · All existing v1 features preserved
"""
#import 
from imports.import_global import *
from GlobalConfig.config_global import *
from Model.model_global import *
from UI.UI_global import *
from Prefrences.prefrence_global import *
from Server.server_global import *
from core.streamer_global import *
from components.components_global import *
from core.engine_global import *
from codeparser.codeparser_global import *

class ModelLoaderThread(QThread):
    finished = pyqtSignal(bool, str)
    log      = pyqtSignal(str, str)

    def __init__(self, engine: "LlamaEngine", model_path: str, ctx: int):
        super().__init__()
        self.engine     = engine
        self.model_path = model_path
        self.ctx        = ctx

    def run(self):
        ok = self.engine.load(self.model_path, ctx=self.ctx)
        self.finished.emit(ok, self.engine.status_text)


class ConfigTab(QWidget):
    """Full configuration tab — all thresholds with descriptions."""

    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields: Dict[str, QWidget] = {}
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
        hdr = QLabel("⚙️  App Configuration")
        hdr.setStyleSheet(f"color:{C['txt']};font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        sub = QLabel(
            "All thresholds and defaults are persisted to app_config.json. "
            "Hover over any field for a full description. Changes take effect immediately.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{C['txt2']};font-size:11px;margin-bottom:16px;")
        root.addWidget(sub)

        # Group fields by category
        categories = [
            ("🧠  Memory & RAM",   ["ram_watchdog_mb", "max_ram_chunks", "auto_spill_on_start"]),
            ("📄  Reference Engine", ["chunk_index_size", "ref_top_k", "ref_max_context_chars"]),
            ("📝  Summarization",   ["summary_chunk_chars", "summary_ctx_carry",
                                     "summary_n_pred_sect", "summary_n_pred_final",
                                     "pause_after_chunks"]),
            ("📚  Multi-PDF",       ["multipdf_n_pred_sect", "multipdf_n_pred_final"]),
            ("⚡  Model Defaults",  ["default_threads", "default_ctx", "default_n_predict"]),
        ]

        for cat_title, keys in categories:
            root.addSpacing(10)
            cat_lbl = QLabel(cat_title)
            cat_lbl.setStyleSheet(
                f"color:{C['txt']};font-size:12px;font-weight:bold;"
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

        # Paused jobs section
        root.addSpacing(14)
        pj_lbl = QLabel("⏸  Paused Summarization Jobs")
        pj_lbl.setStyleSheet(
            f"color:{C['txt']};font-size:12px;font-weight:bold;padding:2px 0;")
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
        pj_desc.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        pj_l.addWidget(pj_desc)

        self.paused_jobs_list = QListWidget()
        self.paused_jobs_list.setFixedHeight(110)
        self.paused_jobs_list.setObjectName("paused_list")
        pj_l.addWidget(self.paused_jobs_list)

        pj_btn_row = QHBoxLayout(); pj_btn_row.setSpacing(8)
        self.btn_resume_job   = QPushButton("▶  Resume Job")
        self.btn_delete_job   = QPushButton("🗑  Delete Job")
        self.btn_refresh_jobs = QPushButton("↻  Refresh")
        for b in (self.btn_resume_job, self.btn_delete_job, self.btn_refresh_jobs):
            b.setFixedHeight(28); pj_btn_row.addWidget(b)
        pj_btn_row.addStretch()
        self.btn_refresh_jobs.clicked.connect(self.refresh_paused_jobs)
        self.btn_delete_job.clicked.connect(self.delete_paused_job)
        pj_l.addLayout(pj_btn_row)
        root.addWidget(pj_card)

        # Save / Reset buttons
        root.addSpacing(14)
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        save_btn = QPushButton("💾  Save All Settings")
        save_btn.setObjectName("btn_send")
        save_btn.setFixedHeight(34)
        save_btn.clicked.connect(self._save_all)
        reset_btn = QPushButton("↺  Reset to Defaults")
        reset_btn.setFixedHeight(34)
        reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(save_btn); btn_row.addWidget(reset_btn); btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()
        self.refresh_paused_jobs()

    def _make_field(self, key: str, meta: dict, current_val) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame{background:transparent;border:none;}")
        fl = QVBoxLayout()
        fl.setContentsMargins(0, 0, 0, 2); fl.setSpacing(3)

        top_row = QHBoxLayout(); top_row.setSpacing(10)
        lbl = QLabel(meta.get("label", key))
        lbl.setStyleSheet(f"color:{C['txt']};font-size:12px;font-weight:600;")
        lbl.setFixedWidth(230)
        top_row.addWidget(lbl)

        ftype = meta.get("type", "int")
        if ftype == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(current_val))
        else:
            widget = QLineEdit(str(current_val))
            widget.setFixedWidth(90)
            widget.setFixedHeight(26)
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            mn = meta.get("min", 0); mx = meta.get("max", 99999)
            widget.setToolTip(f"Range: {mn} – {mx}")

        self._fields[key] = widget
        top_row.addWidget(widget)

        # Range hint
        if ftype != "bool":
            rng_lbl = QLabel(f"({meta.get('min', 0)} – {meta.get('max', '∞')})")
            rng_lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
            top_row.addWidget(rng_lbl)
        top_row.addStretch()
        fl.addLayout(top_row)

        # Description
        desc = QLabel(meta.get("desc", ""))
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color:{C['txt2']};font-size:10px;padding-left:2px;"
            f"line-height:1.5;")
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

    def _reset_defaults(self):
        if QMessageBox.question(
            self, "Reset Defaults", "Reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        APP_CONFIG.update(APP_CONFIG_DEFAULTS)
        save_app_config(APP_CONFIG)
        for key, widget in self._fields.items():
            val = APP_CONFIG_DEFAULTS.get(key, 0)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            else:
                widget.setText(str(val))
        self.config_changed.emit()

    def refresh_paused_jobs(self):
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
                f"⏸  {fname}  —  chunk {nc}/{tot}  —  paused {ts}")
            item.setData(Qt.ItemDataRole.UserRole, jid)
            item.setForeground(QColor(C["warn"]))
            self.paused_jobs_list.addItem(item)

    def delete_paused_job(self):
        item = self.paused_jobs_list.currentItem()
        if not item: return
        jid = item.data(Qt.ItemDataRole.UserRole)
        delete_paused_job(jid)
        self.refresh_paused_jobs()

    def get_selected_job_id(self) -> str:
        item = self.paused_jobs_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def get_selected_job_state(self) -> Optional[dict]:
        jid = self.get_selected_job_id()
        return load_paused_job(jid) if jid else None

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
        self.chk_enable = QCheckBox("Enable Parallel Model Loading")
        self.chk_enable.setStyleSheet(f"color:{C['txt']};font-weight:600;font-size:12px;")
        self.chk_enable.setChecked(self._prefs.enabled)
        self.chk_enable.toggled.connect(self._on_toggle)
        top_row.addWidget(self.chk_enable)
        top_row.addStretch()
        root.addLayout(top_row)

        # ── warning banner ─────────────────────────────────────────────────────
        self.warn_banner = QLabel(
            "⚠️  Parallel loading runs multiple model engines simultaneously.\n"
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
        auto_lbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        root.addWidget(auto_lbl)

        self._role_checks: Dict[str, QCheckBox] = {}
        role_grid = QHBoxLayout(); role_grid.setSpacing(12)
        for role in MODEL_ROLES:
            if role == "general": continue  # always loaded
            chk = QCheckBox(f"{ROLE_ICONS[role]} {role.capitalize()}")
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
        self.chk_pipeline = QCheckBox("🔗  Enable Reasoning → Coding Pipeline Mode")
        self.chk_pipeline.setStyleSheet(f"color:{C['txt']};font-size:12px;")
        self.chk_pipeline.setChecked(self._prefs.pipeline_mode)
        self.chk_pipeline.setEnabled(self._prefs.enabled)
        self.chk_pipeline.toggled.connect(self._save)
        pipeline_row.addWidget(self.chk_pipeline)
        pipeline_row.addStretch()
        root.addLayout(pipeline_row)

        pipeline_desc = QLabel(
            "When enabled and both Reasoning + Coding engines are loaded:\n"
            "  1. 🧠 Reasoning model analyses your coding request → produces a detailed plan\n"
            "  2. 💻 Coding model receives the plan as context → generates the code\n"
            "Only activates when the prompt looks like a coding request."
        )
        pipeline_desc.setWordWrap(True)
        pipeline_desc.setStyleSheet(f"color:{C['txt2']};font-size:11px;padding-left:4px;")
        root.addWidget(pipeline_desc)

        # ── RAM estimate helper ────────────────────────────────────────────────
        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        root.addWidget(sep3)

        self.ram_estimate_lbl = QLabel("")
        self.ram_estimate_lbl.setWordWrap(True)
        self.ram_estimate_lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        root.addWidget(self.ram_estimate_lbl)
        self._update_ram_estimate()

        root.addStretch()
        self.setLayout(root)
        self._update_enabled_state()

    def _on_toggle(self, checked: bool):
        if checked and not self._prefs.warned:
            result = QMessageBox.warning(
                self, "⚠️ Parallel Loading Warning",
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
    """Live theme colour editor — sliders + colour picker per token."""

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

        lbl = QLabel("🎨  Theme Editor")
        lbl.setObjectName("appearance_hdr")
        bar_row.addWidget(lbl)
        bar_row.addStretch()

        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.setObjectName("appearance_btn")
        self.btn_reset.setToolTip("Reset to current built-in palette")
        self.btn_reset.clicked.connect(self._reset)

        self.btn_save = QPushButton("💾  Save")
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
        color = QColorDialog.getColor(init, self, f"Pick colour — {self._LABELS.get(key,key)}")
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
        self._palette = dict(C_LIGHT)
        self._rebuild_rows()
        self.theme_changed.emit(dict(self._palette))

    def _save(self):
        key = "custom_light_palette" if CURRENT_THEME == "light" else "custom_dark_palette"
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
        hdr = QLabel("🖥️  Server & Binary Configuration")
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)

        os_lbl = QLabel(
            f"Detected OS: <b>{self._cfg.detected_os}</b>   "
            f"(looking for <code>{self._cfg.default_cli_name}</code> / "
            f"<code>{self._cfg.default_server_name}</code>)")
        os_lbl.setTextFormat(Qt.TextFormat.RichText)
        os_lbl.setWordWrap(True)
        os_lbl.setStyleSheet(
            f"color:{C['txt2']};font-size:11px;margin-bottom:14px;")
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
        cli_lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
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
        srv_lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
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
        self._resolved_lbl.setStyleSheet(
            f"color:{C['txt2']};font-size:10px;padding:6px 8px;"
            f"background:{C['bg2']};border-radius:5px;")
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
            lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
            r.addWidget(lbl)
            r.addWidget(widget)
            if hint:
                hl = QLabel(hint)
                hl.setStyleSheet(f"color:{C['txt3']};font-size:10px;")
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
        port_lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
        dash = QLabel("–")
        dash.setStyleSheet(f"color:{C['txt2']};")
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
            badge_txt = f"🟢  NVIDIA CUDA detected  ·  {n_gpus} GPU(s)"
            badge_col = C["ok"]
        elif gpu_type == "metal":
            badge_txt = f"🟢  Apple Metal detected  ·  {self._detected_gpus[0]['name']}"
            badge_col = C["ok"]
        elif gpu_type == "vulkan":
            badge_txt = f"🟡  Vulkan GPU detected  ·  {n_gpus} device(s)"
            badge_col = C["warn"]
        else:
            badge_txt = "⚪  No GPU detected — CPU-only mode"
            badge_col = C["txt2"]
        gpu_badge = QLabel(badge_txt)
        gpu_badge.setStyleSheet(
            f"color:{badge_col};font-size:11px;"
            f"background:{C['bg2']};border-radius:5px;padding:5px 10px;")
        gpu_l.addWidget(gpu_badge)

        # GPU list (NVIDIA only shows VRAM)
        if self._detected_gpus:
            for g in self._detected_gpus:
                vram_s = f"  —  {g['vram_mb']} MB VRAM" if g["vram_mb"] else ""
                gl = QLabel(f"  [{g['idx']}]  {g['name']}{vram_s}")
                gl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
                gpu_l.addWidget(gl)

        sep_gpu = QFrame(); sep_gpu.setFrameShape(QFrame.Shape.HLine)
        sep_gpu.setStyleSheet(
            f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        gpu_l.addWidget(sep_gpu)

        # Enable GPU checkbox
        enable_row = QHBoxLayout(); enable_row.setSpacing(10)
        self.chk_gpu = QCheckBox("Enable GPU offloading  (--ngl flag)")
        self.chk_gpu.setStyleSheet(f"color:{C['txt']};font-weight:600;font-size:12px;")
        self.chk_gpu.setChecked(self._cfg.enable_gpu)
        self.chk_gpu.setEnabled(bool(self._detected_gpus))
        enable_row.addWidget(self.chk_gpu); enable_row.addStretch()
        gpu_l.addLayout(enable_row)

        # n_gpu_layers row
        ngl_row = QHBoxLayout(); ngl_row.setSpacing(8)
        ngl_lbl = QLabel("GPU layers  (--ngl):")
        ngl_lbl.setFixedWidth(160)
        ngl_lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
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
        mgpu_lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
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
        ts_lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
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
        flag_note.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        flag_l.addWidget(flag_note)

        cli_flag_row = QHBoxLayout(); cli_flag_row.setSpacing(8)
        cfl = QLabel("llama-cli extra flags:")
        cfl.setFixedWidth(160)
        cfl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
        self.extra_cli_edit = QLineEdit(self._cfg.extra_cli_args)
        self.extra_cli_edit.setPlaceholderText("e.g.  --numa  --no-mmap")
        cli_flag_row.addWidget(cfl)
        cli_flag_row.addWidget(self.extra_cli_edit, 1)
        flag_l.addLayout(cli_flag_row)

        srv_flag_row = QHBoxLayout(); srv_flag_row.setSpacing(8)
        sfl = QLabel("llama-server extra flags:")
        sfl.setFixedWidth(160)
        sfl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
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
        test_note.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        test_l.addWidget(test_note)

        test_row = QHBoxLayout(); test_row.setSpacing(8)
        self.btn_test_cli = QPushButton("🧪  Test llama-cli")
        self.btn_test_cli.setFixedHeight(28)
        self.btn_test_cli.clicked.connect(lambda: self._test_binary("cli"))
        self.btn_test_srv = QPushButton("🧪  Test llama-server")
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
        save_btn = QPushButton("💾  Save Server Settings")
        save_btn.setObjectName("btn_send")
        save_btn.setFixedHeight(34)
        save_btn.clicked.connect(self._save)
        reset_btn = QPushButton("↺  Reset to Defaults")
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
            f"color:{C['txt']};font-size:12px;font-weight:bold;"
            f"letter-spacing:0.5px;padding:0;margin-bottom:2px;")
        return lbl

    @staticmethod
    def _card() -> QFrame:
        f = QFrame(); f.setObjectName("tab_card"); return f

    @staticmethod
    def _hint(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{C['txt3']};font-size:10px;")
        return lbl

    def _update_cli_status(self):
        path = self.cli_edit.text().strip()
        if not path:
            resolved = LLAMA_CLI_DEFAULT
        else:
            resolved = path
        ok = Path(resolved).exists()
        self._cli_status.setText("✅" if ok else "❌")
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
        self._srv_status.setText("✅" if ok else "❌")
        self._srv_status.setToolTip(
            f"{'Found' if ok else 'Not found'}: {resolved}")
        self._refresh_resolved()

    def _refresh_resolved(self):
        cli_path  = self.cli_edit.text().strip() or LLAMA_CLI_DEFAULT
        srv_path  = self.srv_edit.text().strip() or LLAMA_SERVER_DEFAULT
        cli_ok    = "✅" if Path(cli_path).exists() else "❌"
        srv_ok    = "✅" if Path(srv_path).exists() else "❌"
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
            self.test_output.setPlainText(f"❌  Binary not found:\n{path}")
            return

        try:
            result = subprocess.run(
                [path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=8)
            out = result.stdout.decode("utf-8", errors="replace").strip()
            self.test_output.setPlainText(
                f"✅  {Path(path).name} --version\n\n{out or '(no output)'}")
        except subprocess.TimeoutExpired:
            self.test_output.setPlainText(f"⚠️  Timed out after 8s:\n{path}")
        except Exception as e:
            self.test_output.setPlainText(f"❌  Error:\n{e}")

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


# ═════════════════════════════ HF DOWNLOAD WORKERS ═══════════════════════════

class HfSearchWorker(QThread):
    """Queries the HuggingFace API for GGUF siblings in a repo."""
    results_ready = pyqtSignal(list)
    err           = pyqtSignal(str)

    def __init__(self, repo_id: str):
        super().__init__()
        self._repo = repo_id.strip().strip("/")

    def run(self):
        import urllib.request as _ur, json as _j
        url = f"https://huggingface.co/api/models/{self._repo}"
        try:
            req = _ur.Request(url, headers={"User-Agent": "NativeLabPro/2"})
            with _ur.urlopen(req, timeout=15) as r:
                data = _j.loads(r.read())
            siblings = data.get("siblings", [])
            gguf = [s for s in siblings
                    if s.get("rfilename", "").lower().endswith(".gguf")]
            self.results_ready.emit(gguf)
        except Exception as e:
            self.err.emit(str(e))


class HfDownloadWorker(QThread):
    """Downloads a single GGUF file from HuggingFace."""
    progress = pyqtSignal(int, int)   # bytes_done, bytes_total
    done     = pyqtSignal(str)        # final save path
    err      = pyqtSignal(str)

    def __init__(self, repo_id: str, filename: str, dest_dir: Path):
        super().__init__()
        self._repo     = repo_id.strip().strip("/")
        self._filename = filename
        self._dest_dir = dest_dir
        self._abort    = False

    def abort(self):
        self._abort = True

    def run(self):
        import urllib.request as _ur
        url  = f"https://huggingface.co/{self._repo}/resolve/main/{self._filename}"
        dest = self._dest_dir / self._filename
        self._dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            req = _ur.Request(url, headers={"User-Agent": "NativeLabPro/2"})
            with _ur.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length", 0))
                done  = 0
                CHUNK = 262144   # 256 KB
                with open(dest, "wb") as f:
                    while True:
                        if self._abort:
                            try: dest.unlink()
                            except Exception: pass
                            return
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        self.progress.emit(done, total)
            self.done.emit(str(dest))
        except Exception as e:
            try: dest.unlink()
            except Exception: pass
            self.err.emit(str(e))


# ═════════════════════════════ MODEL DOWNLOAD TAB ════════════════════════════

class ModelDownloadTab(QWidget):
    """HuggingFace GGUF Model Downloader tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._search_worker: Optional[HfSearchWorker] = None
        self._dl_worker:     Optional[HfDownloadWorker] = None
        self._files: list = []
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
        hdr = QLabel("⬇️  HuggingFace GGUF Downloader")
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:16px;font-weight:bold;margin-bottom:4px;")
        root.addWidget(hdr)
        sub = QLabel(
            "Enter any HuggingFace repo ID to browse its GGUF files and download "
            "them straight to your models folder.  Network access required.")
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"color:{C['txt2']};font-size:11px;margin-bottom:14px;")
        root.addWidget(sub)

        # ── SEARCH ───────────────────────────────────────────────────────────
        root.addWidget(self._section("SEARCH REPOSITORY"))
        sc = self._card(); sl = QVBoxLayout(sc)
        sl.setContentsMargins(16, 14, 16, 14); sl.setSpacing(10)

        hint = QLabel(
            "Examples:  TheBloke/Mistral-7B-Instruct-v0.2-GGUF  ·  "
            "bartowski/Llama-3.2-3B-Instruct-GGUF  ·  "
            "lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        sl.addWidget(hint)

        row1 = QHBoxLayout(); row1.setSpacing(8)
        self.repo_edit = QLineEdit()
        self.repo_edit.setPlaceholderText(
            "Enter HuggingFace repo ID  (e.g. TheBloke/Mistral-7B-Instruct-v0.2-GGUF)")
        self.repo_edit.setFixedHeight(30)
        self.btn_search = QPushButton("🔍  Search")
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
        dest_lbl.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
        self.dest_edit = QLineEdit(str(MODELS_DIR.resolve()))
        self.dest_edit.setReadOnly(True)
        btn_dest = QPushButton("Browse…")
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
        self.btn_download = QPushButton("⬇️  Download Selected")
        self.btn_download.setObjectName("btn_send")
        self.btn_download.setFixedHeight(32); self.btn_download.setEnabled(False)
        self.btn_download.clicked.connect(self._start_download)
        self.btn_abort = QPushButton("⏹  Cancel")
        self.btn_abort.setObjectName("btn_stop")
        self.btn_abort.setFixedHeight(32); self.btn_abort.setVisible(False)
        self.btn_abort.clicked.connect(self._abort_download)
        btn_row.addWidget(self.btn_download); btn_row.addWidget(self.btn_abort)
        btn_row.addStretch()
        dl_l.addLayout(btn_row)
        root.addWidget(dc); root.addStretch()

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{C['txt']};font-size:12px;font-weight:bold;"
            f"letter-spacing:0.5px;padding:0;margin-bottom:2px;")
        return lbl

    @staticmethod
    def _card() -> QFrame:
        f = QFrame(); f.setObjectName("tab_card"); return f

    # ── actions ───────────────────────────────────────────────────────────────
    def _do_search(self):
        repo = self.repo_edit.text().strip()
        if not repo:
            self.search_status.setText("⚠️  Enter a repo ID first."); return
        self.search_status.setText("🔍  Querying HuggingFace API…")
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
            self.search_status.setText("⚠️  No GGUF files found in this repo."); return
        self.search_status.setText(f"✅  Found {len(files)} GGUF file(s).")
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
        self.search_status.setText(f"❌  Error: {msg}")

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
        self.btn_abort.setVisible(True)
        self.dl_progress.setValue(0)
        self.dl_status.setText(f"Downloading  {fname}…")
        self._dl_worker = HfDownloadWorker(repo, fname, dest)
        self._dl_worker.progress.connect(self._on_dl_progress)
        self._dl_worker.done.connect(self._on_dl_done)
        self._dl_worker.err.connect(self._on_dl_err)
        self._dl_worker.start()

    def _abort_download(self):
        if self._dl_worker:
            self._dl_worker.abort()
        self.btn_download.setVisible(True); self.btn_abort.setVisible(False)
        self.dl_status.setText("Download cancelled."); self.dl_progress.setValue(0)

    def _on_dl_progress(self, done: int, total: int):
        if total > 0:
            pct = int(done * 100 / total)
            self.dl_progress.setValue(pct)
            self.dl_status.setText(
                f"⬇️  {done/1e6:.1f} / {total/1e6:.1f} MB  ({pct}%)")
        else:
            self.dl_status.setText(f"⬇️  {done/1e6:.1f} MB downloaded…")

    def _on_dl_done(self, path: str):
        self.btn_download.setVisible(True); self.btn_abort.setVisible(False)
        self.dl_progress.setValue(100)
        self.dl_status.setText(f"✅  Saved to:  {path}")
        get_model_registry().add(path)
        QMessageBox.information(
            self, "Download Complete",
            f"Model saved to:\n{path}\n\n"
            "It has been added to your model library.")

    def _on_dl_err(self, msg: str):
        self.btn_download.setVisible(True); self.btn_abort.setVisible(False)
        self.dl_progress.setValue(0)
        self.dl_status.setText(f"❌  Error: {msg}")


# ═════════════════════════════ MCP TAB ═══════════════════════════════════════

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
        hdr = QLabel("🔌  MCP — Model Context Protocol")
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:16px;font-weight:bold;margin-bottom:4px;")
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
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("btn_send"); self.btn_start.setFixedHeight(28)
        self.btn_start.setEnabled(False); self.btn_start.clicked.connect(self._start_server)
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setObjectName("btn_stop"); self.btn_stop.setFixedHeight(28)
        self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._stop_server)
        self.btn_del_srv = QPushButton("🗑  Remove")
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
            l.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
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
            "Command or URL — e.g.  npx -y @mcp/server-filesystem /path")
        self.add_cmd.setFixedHeight(28)
        al.addLayout(_lrow("Command / URL:", self.add_cmd))

        self.add_desc = QLineEdit()
        self.add_desc.setPlaceholderText("Optional description")
        self.add_desc.setFixedHeight(28)
        al.addLayout(_lrow("Description:", self.add_desc))

        btn_add = QPushButton("＋  Add Server")
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
            f"color:{C['txt']};font-size:12px;font-weight:bold;"
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
            icon = "🟢" if running else "⚪"
            label = (f"  {icon}  {s['name']}"
                     f"   [{s['transport']}]"
                     f"   {s.get('desc', '') or s['cmd'][:50]}")
            item = QListWidgetItem(label)
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
                f"SSE server '{name}' — connect at: {srv['cmd']}")
            self._refresh_list(); return
        try:
            proc = subprocess.Popen(
                srv["cmd"], shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self._procs[name] = proc
            self._mcp_log_msg(f"✅  Started '{name}'  (PID {proc.pid})")
        except Exception as e:
            self._mcp_log_msg(f"❌  Failed to start '{name}': {e}")
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
            try: proc.terminate(); proc.wait(3)
            except Exception:
                try: proc.kill()
                except Exception: pass
            del self._procs[name]
            self._mcp_log_msg(f"⏹  Stopped '{name}'")


# ═════════════════════════════ PIPELINE BUILDER ══════════════════════════════

import math as _math

class PipelineBlockType:
    INPUT        = "input"
    OUTPUT       = "output"
    MODEL        = "model"
    INTERMEDIATE = "intermediate"
    REFERENCE    = "reference"    # injects a reference text snippet before context
    KNOWLEDGE    = "knowledge"    # prepends a knowledge-base chunk before context
    PDF_SUMMARY  = "pdf_summary"  # extracts / summarises a PDF and prepends it
    # ── Logic blocks ─────────────────────────────────────────────────────────
    IF_ELSE      = "if_else"      # condition → True port or False port
    SWITCH       = "switch"       # match value → one of N labelled ports
    FILTER       = "filter"       # pass-through or drop based on condition
    TRANSFORM    = "transform"    # deterministic text transform (prefix/suffix/regex)
    MERGE        = "merge"        # combine multiple incoming contexts into one
    SPLIT        = "split"        # broadcast same text to all outgoing connections
    CUSTOM_CODE  = "custom_code"  # user-written Python executed at runtime
    # ── LLM Logic blocks — conditions evaluated by an attached LLM ───────────
    LLM_IF       = "llm_if"       # LLM answers YES/NO → TRUE/FALSE routing
    LLM_SWITCH   = "llm_switch"   # LLM classifies into one of N user-defined labels
    LLM_FILTER   = "llm_filter"   # LLM decides pass/drop in plain English
    LLM_TRANSFORM= "llm_transform"# LLM rewrites/transforms text per instruction
    LLM_SCORE    = "llm_score"    # LLM scores 1–10 → routes to low/mid/high port

# Runtime PyPDF2 guard — PDF blocks show a friendly error if not installed
try:
    import PyPDF2 as _pypdf2_check
    HAS_PDF = True
except ImportError:
    HAS_PDF = False


@dataclass
class PipelineConnection:
    from_block_id: int
    from_port:     str   # "N" | "S" | "E" | "W"
    to_block_id:   int
    to_port:       str
    is_loop:       bool = False
    loop_times:    int  = 1


class PipelineBlock:
    _id_counter = 0

    def __init__(self, btype: str, x: int, y: int,
                 model_path: str = "", role: str = "general", label: str = ""):
        PipelineBlock._id_counter += 1
        self.bid        = PipelineBlock._id_counter
        self.btype      = btype
        self.x          = x
        self.y          = y
        self.w          = 148
        self.h          = 76
        self.model_path = model_path
        self.role       = role
        self.label      = label or self._default_label()
        self.selected   = False
        self.metadata:  dict = {}   # stores ref_text / knowledge_text / pdf_path

    def _default_label(self) -> str:
        if self.btype == PipelineBlockType.INPUT:        return "Input"
        if self.btype == PipelineBlockType.OUTPUT:       return "Output"
        if self.btype == PipelineBlockType.INTERMEDIATE: return "Intermediate"
        if self.btype == PipelineBlockType.REFERENCE:    return "Reference"
        if self.btype == PipelineBlockType.KNOWLEDGE:    return "Knowledge"
        if self.btype == PipelineBlockType.PDF_SUMMARY:  return "PDF"
        if self.model_path:
            return Path(self.model_path).stem[:18]
        return "Model"

    def rect(self):
        return QRect(self.x, self.y, self.w, self.h)

    def port_pos(self, port: str) -> tuple:
        cx = self.x + self.w // 2
        cy = self.y + self.h // 2
        return {
            "N": (cx, self.y),
            "S": (cx, self.y + self.h),
            "W": (self.x, cy),
            "E": (self.x + self.w, cy),
        }.get(port, (cx, cy))

    def port_at(self, px: int, py: int, radius: int = 11) -> Optional[str]:
        for port in ("N", "S", "E", "W"):
            bx, by = self.port_pos(port)
            if abs(px - bx) <= radius and abs(py - by) <= radius:
                return port
        return None

    def contains(self, px: int, py: int) -> bool:
        return self.rect().contains(px, py)


# ── Pipeline persistence helpers ──────────────────────────────────────────────

PIPELINES_DIR = Path.home() / ".native_lab" / "pipelines"
PIPELINES_DIR.mkdir(parents=True, exist_ok=True)


def _pipeline_to_dict(blocks: list, connections: list) -> dict:
    """Serialise a pipeline to a JSON-safe dict."""
    return {
        "version": 2,
        "blocks": [
            {
                "bid":        b.bid,
                "btype":      b.btype,
                "x":          b.x,
                "y":          b.y,
                "w":          b.w,
                "h":          b.h,
                "model_path": b.model_path,
                "role":       b.role,
                "label":      b.label,
                "metadata":   b.metadata,
            }
            for b in blocks
        ],
        "connections": [
            {
                "from_block_id": c.from_block_id,
                "from_port":     c.from_port,
                "to_block_id":   c.to_block_id,
                "to_port":       c.to_port,
                "is_loop":       c.is_loop,
                "loop_times":    c.loop_times,
            }
            for c in connections
        ],
    }


def _pipeline_from_dict(data: dict):
    """Deserialise blocks + connections from a saved dict. Returns (blocks, connections)."""
    blocks = []
    id_map: Dict[int, int] = {}   # old bid → new bid (counter is global)
    for bd in data.get("blocks", []):
        b            = PipelineBlock(bd["btype"], bd["x"], bd["y"],
                                     bd.get("model_path", ""),
                                     bd.get("role", "general"),
                                     bd.get("label", ""))
        b.metadata   = bd.get("metadata", {})
        id_map[bd["bid"]] = b.bid
        blocks.append(b)

    connections = []
    for cd in data.get("connections", []):
        connections.append(PipelineConnection(
            from_block_id=id_map.get(cd["from_block_id"], cd["from_block_id"]),
            from_port=cd["from_port"],
            to_block_id=id_map.get(cd["to_block_id"],   cd["to_block_id"]),
            to_port=cd["to_port"],
            is_loop=cd.get("is_loop", False),
            loop_times=cd.get("loop_times", 1),
        ))
    return blocks, connections


def list_saved_pipelines() -> List[str]:
    """Return sorted list of saved pipeline names (without .json)."""
    return sorted(p.stem for p in PIPELINES_DIR.glob("*.json"))


def save_pipeline(name: str, blocks: list, connections: list):
    path = PIPELINES_DIR / f"{name}.json"
    path.write_text(
        json.dumps(_pipeline_to_dict(blocks, connections), indent=2),
        encoding="utf-8")


def load_pipeline(name: str):
    path = PIPELINES_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return _pipeline_from_dict(data)


class PipelineCanvas(QWidget):
    """Interactive drag-and-drop pipeline canvas with curved arrows."""

    blocks_changed = pyqtSignal()
    PORT_R = 6
    GRID   = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1400, 900)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setAcceptDrops(True)   # accept drags from model list

        self.blocks:      List[PipelineBlock]      = []
        self.connections: List[PipelineConnection] = []

        self._drag_block:     Optional[PipelineBlock] = None
        self._drag_offset:    tuple = (0, 0)
        self._connect_from:   Optional[tuple] = None   # (block, port)
        self._connect_preview:Optional[tuple] = None
        self._hover_block:    Optional[PipelineBlock] = None
        self._hover_port:     Optional[str]           = None
        self._selected:       Optional[PipelineBlock] = None
        self._drop_preview:   Optional[tuple]         = None  # (x, y) ghost position

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._ctx_menu)

    # ── drag-and-drop from sidebar ────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            px = int(event.position().x())
            py = int(event.position().y())
            self._drop_preview = (self._snap(px), self._snap(py))
            self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_preview = None
        self.update()

    def dropEvent(self, event):
        self._drop_preview = None
        if not event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            event.ignore()
            return
        px = int(event.position().x())
        py = int(event.position().y())
        # Decode the QListWidget MIME data to get model path + role
        _data = event.mimeData().data("application/x-qabstractitemmodeldatalist")
        _stream = QDataStream(_data, QIODevice.OpenModeFlag.ReadOnly)
        model_path = ""
        role = "general"
        try:
            while not _stream.atEnd():
                _row = _stream.readInt32()
                _col = _stream.readInt32()
                _n   = _stream.readInt32()
                for _ in range(_n):
                    _role_key = _stream.readInt32()
                    _var = QVariant()
                    _stream >> _var
                    if _role_key == Qt.ItemDataRole.UserRole:
                        model_path = str(_var.value()) if _var.isValid() else ""
                    elif _role_key == Qt.ItemDataRole.UserRole + 1:
                        role = str(_var.value()) if _var.isValid() else "general"
        except Exception:
            pass  # fallback: no path decoded — block will show unconfigured

        drop_x = self._snap(px - 80)
        drop_y = self._snap(py - 30)
        b = self.add_block(PipelineBlockType.MODEL, x=drop_x, y=drop_y,
                           model_path=model_path, role=role or "general")
        if model_path:
            b.label = Path(model_path).stem[:18]
        self._selected = b
        self.update()
        event.acceptProposedAction()

    # ── block management ──────────────────────────────────────────────────────

    def add_block(self, btype: str, x: int = 100, y: int = 100,
                  model_path: str = "", role: str = "general") -> PipelineBlock:
        b = PipelineBlock(btype, x, y, model_path, role)
        self.blocks.append(b)
        self.update()
        self.blocks_changed.emit()
        return b

    def remove_block(self, b: PipelineBlock):
        bid = b.bid
        self.blocks      = [bl for bl in self.blocks      if bl.bid != bid]
        self.connections = [c  for c  in self.connections
                            if c.from_block_id != bid and c.to_block_id != bid]
        if self._selected is b:
            self._selected = None
        self.update()
        self.blocks_changed.emit()

    def clear_all(self):
        self.blocks.clear()
        self.connections.clear()
        self._selected     = None
        self._drag_block   = None
        self._connect_from = None
        PipelineBlock._id_counter = 0
        self.update()
        self.blocks_changed.emit()

    def _snap(self, v: int) -> int:
        return round(v / self.GRID) * self.GRID

    def _block_by_id(self, bid: int) -> Optional[PipelineBlock]:
        for b in self.blocks:
            if b.bid == bid:
                return b
        return None

    # ── painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event):

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ── background ───────────────────────────────────────────────────────
        p.fillRect(self.rect(), QColor(C["bg0"]))

        # ── grid dots ────────────────────────────────────────────────────────
        grid_c = QColor(C["bdr"])
        grid_c.setAlpha(80)
        p.setPen(QPen(grid_c, 1))
        for gx in range(0, self.width(), self.GRID):
            p.drawLine(gx, 0, gx, self.height())
        for gy in range(0, self.height(), self.GRID):
            p.drawLine(0, gy, self.width(), gy)

        # ── connections ───────────────────────────────────────────────────────
        for conn in self.connections:
            fb = self._block_by_id(conn.from_block_id)
            tb = self._block_by_id(conn.to_block_id)
            if fb and tb:
                self._draw_arrow(p, fb, conn.from_port,
                                 tb, conn.to_port,
                                 conn.is_loop, conn.loop_times)

        # ── preview arrow ─────────────────────────────────────────────────────
        if self._connect_from and self._connect_preview:
            b, port = self._connect_from
            sx, sy  = b.port_pos(port)
            ex, ey  = self._connect_preview
            pen = QPen(QColor(C["acc"]), 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(self._make_path(sx, sy, ex, ey, "E", "W"))

        # ── blocks ────────────────────────────────────────────────────────────
        self._draw_drop_ghost(p)

        # ── blocks ────────────────────────────────────────────────────────────
        for b in self.blocks:
            self._draw_block(p, b)

        p.end()

    # ─ colour table ────────────────────────────────────────────────────────────
    _BLOCK_COLORS = {
        PipelineBlockType.INPUT:        lambda: (C["ok"],       C["bg2"]),
        PipelineBlockType.OUTPUT:       lambda: (C["err"],      C["bg2"]),
        PipelineBlockType.INTERMEDIATE: lambda: (C["warn"],     C["bg2"]),
        PipelineBlockType.MODEL:        lambda: (C["acc"],      C["bg1"]),
        PipelineBlockType.REFERENCE:    lambda: (C["acc"],      C["bg2"]),
        PipelineBlockType.KNOWLEDGE:    lambda: (C["acc2"],     C["bg2"]),
        PipelineBlockType.PDF_SUMMARY:  lambda: (C["pipeline"], C["bg2"]),
        # Logic blocks — distinct colour family
        PipelineBlockType.IF_ELSE:      lambda: ("#f59e0b",     C["bg2"]),
        PipelineBlockType.SWITCH:       lambda: ("#f97316",     C["bg2"]),
        PipelineBlockType.FILTER:       lambda: ("#84cc16",     C["bg2"]),
        PipelineBlockType.TRANSFORM:    lambda: ("#06b6d4",     C["bg2"]),
        PipelineBlockType.MERGE:        lambda: ("#8b5cf6",     C["bg2"]),
        PipelineBlockType.SPLIT:        lambda: ("#ec4899",     C["bg2"]),
        PipelineBlockType.CUSTOM_CODE:  lambda: ("#10b981",     C["bg1"]),
        # LLM logic blocks — warmer violet family to distinguish from code logic
        PipelineBlockType.LLM_IF:        lambda: ("#a855f7",    C["bg1"]),
        PipelineBlockType.LLM_SWITCH:    lambda: ("#7c3aed",    C["bg1"]),
        PipelineBlockType.LLM_FILTER:    lambda: ("#6366f1",    C["bg1"]),
        PipelineBlockType.LLM_TRANSFORM: lambda: ("#0ea5e9",    C["bg1"]),
        PipelineBlockType.LLM_SCORE:     lambda: ("#d946ef",    C["bg1"]),
    }
    _BLOCK_ICONS = {
        PipelineBlockType.INPUT:        "▶ INPUT",
        PipelineBlockType.OUTPUT:       "■ OUTPUT",
        PipelineBlockType.INTERMEDIATE: "◈ INTER.",
        PipelineBlockType.MODEL:        None,          # filled dynamically
        PipelineBlockType.REFERENCE:    "📎 REF.",
        PipelineBlockType.KNOWLEDGE:    "💡 KNOW.",
        PipelineBlockType.PDF_SUMMARY:  "📄 PDF",
        PipelineBlockType.IF_ELSE:      "⑂ IF/ELSE",
        PipelineBlockType.SWITCH:       "⑃ SWITCH",
        PipelineBlockType.FILTER:       "⊘ FILTER",
        PipelineBlockType.TRANSFORM:    "⟲ TRANSFORM",
        PipelineBlockType.MERGE:        "⊕ MERGE",
        PipelineBlockType.SPLIT:        "⑁ SPLIT",
        PipelineBlockType.CUSTOM_CODE:  "⌥ CODE",
        PipelineBlockType.LLM_IF:        "🧠 LLM-IF",
        PipelineBlockType.LLM_SWITCH:    "🧠 LLM-SW",
        PipelineBlockType.LLM_FILTER:    "🧠 LLM-FL",
        PipelineBlockType.LLM_TRANSFORM: "🧠 LLM-TX",
        PipelineBlockType.LLM_SCORE:     "🧠 LLM-SC",
    }

    def _draw_block(self, p, b: PipelineBlock):      

        fn = self._BLOCK_COLORS.get(b.btype, self._BLOCK_COLORS[PipelineBlockType.MODEL])
        border_c, fill_c = fn()

        border = QColor(border_c)
        fill   = QColor(fill_c)
        sel    = b is self._selected

        # shadow
        shadow = QColor(0, 0, 0, 50)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(shadow))
        p.drawRoundedRect(b.x + 4, b.y + 4, b.w, b.h, 10, 10)

        # body
        p.setBrush(QBrush(fill))
        p.setPen(QPen(border, 2.5 if sel else 1.5))
        p.drawRoundedRect(b.x, b.y, b.w, b.h, 10, 10)

        # header strip
        hdr_c = QColor(border_c); hdr_c.setAlpha(55)
        p.setBrush(QBrush(hdr_c))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(b.x + 1, b.y + 1, b.w - 2, 20, 9, 9)
        p.drawRect(b.x + 1, b.y + 10, b.w - 2, 11)

        # header text
        p.setPen(QColor(border_c))
        p.setFont(QFont("Inter", 7, QFont.Weight.Bold))
        if b.btype == PipelineBlockType.MODEL:
            tag = f"⚡ {b.role.upper()}"
        else:
            tag = self._BLOCK_ICONS.get(b.btype, "BLOCK")
        p.drawText(b.x, b.y + 1, b.w, 20,
                   Qt.AlignmentFlag.AlignCenter, tag)

        # name label
        p.setPen(QColor(C["txt"]))
        p.setFont(QFont("Inter", 9))
        p.drawText(b.x + 4, b.y + 24, b.w - 8, b.h - 38,
                   Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                   b.label)

        # model badge (quant + family)
        if b.btype == PipelineBlockType.MODEL and b.model_path:
            fam   = detect_model_family(b.model_path)
            quant = detect_quant_type(b.model_path)
            badge = f"{fam.name[:8]}·{quant}"
            p.setPen(QColor(C["txt3"]))
            p.setFont(QFont("Consolas", 7))
            p.drawText(b.x, b.y + b.h - 17, b.w, 15,
                       Qt.AlignmentFlag.AlignCenter, badge)

        # port dots
        for port in ("N", "S", "E", "W"):
            px, py     = b.port_pos(port)
            is_hovered = (b is self._hover_block and port == self._hover_port)
            dot_c      = QColor(border_c)
            r          = self.PORT_R + 3 if is_hovered else self.PORT_R
            p.setBrush(QBrush(QColor(C["bg0"])))
            p.setPen(QPen(dot_c, 2))
            p.drawEllipse(px - r, py - r, r * 2, r * 2)
            if is_hovered:
                p.setBrush(QBrush(dot_c))
                p.setPen(Qt.PenStyle.NoPen)
                inner = r - 3
                p.drawEllipse(px - inner, py - inner, inner * 2, inner * 2)

    def _make_path(self, sx, sy, ex, ey, fport="E", tport="W"):
        """Cubic bezier path that adapts control handles to port orientation."""
        path = QPainterPath(QPointF(sx, sy))
        dx   = abs(ex - sx)
        dy   = abs(ey - sy)
        off  = max(60, dx // 2, dy // 2, 40)

        PORT_VEC = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
        fx, fy = PORT_VEC.get(fport, (1, 0))
        tx, ty = PORT_VEC.get(tport, (-1, 0))

        c1x = sx + fx * off
        c1y = sy + fy * off
        c2x = ex + tx * off
        c2y = ey + ty * off

        path.cubicTo(QPointF(c1x, c1y), QPointF(c2x, c2y), QPointF(ex, ey))
        return path

    def _draw_drop_ghost(self, p):
        """Draw a translucent ghost block at the current drag-over position."""
        if not self._drop_preview:
            return
        gx, gy = self._drop_preview
        gw, gh = 140, 48
        ghost_bg = QColor(C["acc"]); ghost_bg.setAlpha(40)
        ghost_bd = QColor(C["acc"]); ghost_bd.setAlpha(160)
        p.setBrush(QBrush(ghost_bg))
        p.setPen(QPen(ghost_bd, 2, Qt.PenStyle.DashLine))
        p.drawRoundedRect(gx, gy, gw, gh, 8, 8)
        p.setPen(QColor(C["acc"]))
        p.setFont(QFont("Inter", 9))
        p.drawText(gx, gy, gw, gh, Qt.AlignmentFlag.AlignCenter, "⚡ Drop here")

    def _draw_arrow(self, p, fb, fport, tb, tport, is_loop, loop_times):        
        sx, sy = fb.port_pos(fport)
        ex, ey = tb.port_pos(tport)

        color = QColor(C["pipeline"] if is_loop else C["acc2"])
        pen   = QPen(color, 2)
        if is_loop:
            pen.setStyle(Qt.PenStyle.DashDotLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        path = self._make_path(sx, sy, ex, ey, fport, tport)
        p.drawPath(path)

        # arrowhead
        t1 = path.pointAtPercent(0.97)
        t2 = path.pointAtPercent(1.0)
        adx = t2.x() - t1.x(); ady = t2.y() - t1.y()
        length = (adx ** 2 + ady ** 2) ** 0.5
        if length > 0.001:
            adx /= length; ady /= length
            sz = 10
            p1 = QPointF(t2.x() - sz * adx + sz * 0.5 * ady,
                         t2.y() - sz * ady - sz * 0.5 * adx)
            p2 = QPointF(t2.x() - sz * adx - sz * 0.5 * ady,
                         t2.y() - sz * ady + sz * 0.5 * adx)
            p.setBrush(QBrush(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([t2, p1, p2]))

        # loop badge
        if is_loop and loop_times > 1:
            mid = path.pointAtPercent(0.5)
            badge_bg = QColor(C["warn"]); badge_bg.setAlpha(180)
            p.setBrush(QBrush(badge_bg))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(int(mid.x()) - 14, int(mid.y()) - 10, 28, 20, 5, 5)
            p.setPen(QColor(C["bg0"]))
            p.setFont(QFont("Inter", 8, QFont.Weight.Bold))
            p.drawText(int(mid.x()) - 14, int(mid.y()) - 10, 28, 20,
                       Qt.AlignmentFlag.AlignCenter, f"×{loop_times}")

        # branch label badge (IF_ELSE / SWITCH fan-out arms)
        branch = getattr(self.connections[0] if self.connections else object(),
                         "branch_label", None)
        # Retrieve the actual connection object for this specific arrow
        _conn_obj = next(
            (c for c in self.connections
             if c.from_block_id == fb.bid and c.from_port == fport
             and c.to_block_id == tb.bid),
            None)
        branch = getattr(_conn_obj, "branch_label", "") if _conn_obj else ""
        if branch:
            mid = path.pointAtPercent(0.35)
            branch_col = {"TRUE": "#22c55e", "FALSE": "#ef4444"}.get(
                branch, C["pipeline"])
            badge_bg2 = QColor(branch_col); badge_bg2.setAlpha(200)
            w_b = max(len(branch) * 7 + 10, 36)
            p.setBrush(QBrush(badge_bg2))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(
                int(mid.x()) - w_b // 2, int(mid.y()) - 9, w_b, 18, 5, 5)
            p.setPen(QColor("#ffffff"))
            p.setFont(QFont("Inter", 7, QFont.Weight.Bold))
            p.drawText(int(mid.x()) - w_b // 2, int(mid.y()) - 9, w_b, 18,
                       Qt.AlignmentFlag.AlignCenter, branch)

    # ── mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        px = int(event.position().x())
        py = int(event.position().y())

        if event.button() == Qt.MouseButton.LeftButton:
            # Port hit? → start arrow draw
            for b in reversed(self.blocks):
                port = b.port_at(px, py)
                if port:
                    self._connect_from    = (b, port)
                    self._connect_preview = (px, py)
                    return
            # Block hit? → select + drag
            for b in reversed(self.blocks):
                if b.contains(px, py):
                    self._drag_block  = b
                    self._drag_offset = (px - b.x, py - b.y)
                    self._selected    = b
                    self.update()
                    return
            self._selected = None
            self.update()

    def mouseMoveEvent(self, event):
        px = int(event.position().x())
        py = int(event.position().y())

        if self._drag_block:
            ox, oy = self._drag_offset
            self._drag_block.x = self._snap(px - ox)
            self._drag_block.y = self._snap(py - oy)
            self.update()
            return

        if self._connect_from:
            self._connect_preview = (px, py)
            self.update()
            return

        # Hover
        self._hover_block = None
        self._hover_port  = None
        for b in reversed(self.blocks):
            port = b.port_at(px, py)
            if port:
                self._hover_block = b
                self._hover_port  = port
                self.setCursor(Qt.CursorShape.CrossCursor)
                self.update()
                return
            if b.contains(px, py):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                self.update()
                return
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, event):
        px = int(event.position().x())
        py = int(event.position().y())

        if self._drag_block:
            self._drag_block = None
            self.blocks_changed.emit()
            self.update()
            return

        if self._connect_from and event.button() == Qt.MouseButton.LeftButton:
            from_b, from_port   = self._connect_from
            self._connect_from    = None
            self._connect_preview = None
            for b in reversed(self.blocks):
                if b is from_b:
                    continue
                port = b.port_at(px, py)
                if port:
                    self._try_connect(from_b, from_port, b, port)
                    self.update()
                    return
            self.update()

    # Logic block types that may fan-out to multiple targets
    _LOGIC_BTYPES = {
        PipelineBlockType.IF_ELSE,
        PipelineBlockType.SWITCH,
        PipelineBlockType.SPLIT,
        PipelineBlockType.MERGE,
        PipelineBlockType.FILTER,
        PipelineBlockType.TRANSFORM,
        PipelineBlockType.CUSTOM_CODE,
        PipelineBlockType.LLM_IF,
        PipelineBlockType.LLM_SWITCH,
        PipelineBlockType.LLM_FILTER,
        PipelineBlockType.LLM_TRANSFORM,
        PipelineBlockType.LLM_SCORE,
    }

    def _try_connect(self, fb: PipelineBlock, fport: str,
                     tb: PipelineBlock, tport: str):
        # Prevent self-loop via single click
        if fb.bid == tb.bid:
            return

        # Rule: model → model requires an intermediate in between
        # (except when source is a logic block)
        if (fb.btype == PipelineBlockType.MODEL and
                tb.btype == PipelineBlockType.MODEL and
                fb.btype not in self._LOGIC_BTYPES):
            QMessageBox.warning(
                self, "Connection Not Allowed",
                "Direct model-to-model connections are not allowed.\n\n"
                "You must place an ◈ Intermediate Output block between two model blocks.\n"
                "This enforces explicit output capture between pipeline stages.")
            return

        # Prevent exact duplicate connections (same from_bid, same from_port, same to_bid)
        already = any(
            c.from_block_id == fb.bid and c.from_port == fport
            and c.to_block_id == tb.bid
            for c in self.connections)
        if already:
            return

        is_loop    = self._would_form_loop(fb.bid, tb.bid)
        loop_times = 1
        if is_loop:
            val, ok = QInputDialog.getInt(
                self, "Loop Configuration",
                f"How many times should this edge loop?\n"
                f"(The output of '{tb.label}' feeds back into '{fb.label}')",
                value=3, min=2, max=999)
            if not ok:
                return
            loop_times = val

        # For non-logic source blocks: enforce single outgoing connection per port
        # For logic blocks: allow fan-out (multiple arrows from same port)
        if fb.btype not in self._LOGIC_BTYPES:
            self.connections = [
                c for c in self.connections
                if not (c.from_block_id == fb.bid and c.from_port == fport)]

        # For IF_ELSE — label the port so we know which branch this is
        branch_label = ""
        if fb.btype == PipelineBlockType.IF_ELSE:
            # E port = True branch, W port = False branch, S = either
            branch_map = {"E": "TRUE", "W": "FALSE", "S": "ELSE", "N": "PASS"}
            branch_label = branch_map.get(fport, fport)
            # Store branch on connection metadata via a subclass attr we'll inject
        elif fb.btype == PipelineBlockType.SWITCH:
            branch_label, ok = QInputDialog.getText(
                self, "Switch Branch Label",
                f"Label for this output arm of SWITCH '{fb.label}':\n"
                "(e.g. 'case_a', 'default')")
            if not ok:
                branch_label = fport

        conn = PipelineConnection(
            from_block_id=fb.bid, from_port=fport,
            to_block_id=tb.bid,   to_port=tport,
            is_loop=is_loop, loop_times=loop_times)
        conn.branch_label = branch_label   # dynamic attribute — stored at runtime
        self.connections.append(conn)
        self.update()
        self.blocks_changed.emit()

    def _configure_logic_block(self, b: "PipelineBlock"):
        """Open configuration dialog for logic / code blocks."""
        _LOGIC_BTYPES = {
            PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
            PipelineBlockType.FILTER,  PipelineBlockType.TRANSFORM,
            PipelineBlockType.MERGE,   PipelineBlockType.SPLIT,
            PipelineBlockType.CUSTOM_CODE,
        }
        if b.btype not in _LOGIC_BTYPES:
            return

        if b.btype == PipelineBlockType.IF_ELSE:
            cur = b.metadata.get("condition", "")
            cond, ok = QInputDialog.getText(
                self, f"IF / ELSE — '{b.label}'",
                "Python condition evaluated on the incoming text.\n"
                "Variable  'text'  is the incoming string.\n\n"
                "Examples:\n"
                "  len(text) > 500\n"
                "  'error' in text.lower()\n"
                "  text.startswith('FAIL')\n\n"
                "TRUE result → E port    FALSE result → W port",
                text=cur)
            if ok:
                b.metadata["condition"] = cond.strip()
                b.label = f"⑂ {cond.strip()[:18]}" if cond.strip() else "IF/ELSE"

        elif b.btype == PipelineBlockType.SWITCH:
            cur = b.metadata.get("switch_expr", "")
            expr, ok = QInputDialog.getText(
                self, f"SWITCH — '{b.label}'",
                "Python expression returning a string key.\n"
                "Variable  'text'  is the incoming string.\n\n"
                "Example:  'long' if len(text) > 300 else 'short'\n\n"
                "Connect outgoing arrows and label them with matching keys.",
                text=cur)
            if ok:
                b.metadata["switch_expr"] = expr.strip()
                b.label = f"⑃ {expr.strip()[:18]}" if expr.strip() else "SWITCH"

        elif b.btype == PipelineBlockType.FILTER:
            cur_cond  = b.metadata.get("filter_cond", "True")
            cur_mode  = b.metadata.get("filter_mode", "pass")
            cond, ok = QInputDialog.getText(
                self, f"FILTER — '{b.label}'",
                "Python condition. 'text' = incoming string.\n"
                "If TRUE → text passes to next block.\n"
                "If FALSE → pipeline stops (text is dropped).\n\n"
                "Example:  len(text.strip()) > 10",
                text=cur_cond)
            if ok:
                b.metadata["filter_cond"] = cond.strip() or "True"
                b.label = f"⊘ {cond.strip()[:18]}" if cond.strip() else "FILTER"

        elif b.btype == PipelineBlockType.TRANSFORM:
            items = [
                "prefix    — prepend fixed text",
                "suffix    — append fixed text",
                "replace   — find & replace substring",
                "upper     — convert to uppercase",
                "lower     — convert to lowercase",
                "strip     — strip leading/trailing whitespace",
                "truncate  — limit to N characters",
            ]
            cur_type = b.metadata.get("transform_type", "prefix")
            cur_idx  = next((i for i, s in enumerate(items) if cur_type in s), 0)
            ttype, ok = QInputDialog.getItem(
                self, f"TRANSFORM type — '{b.label}'",
                "Choose the transformation:", items, cur_idx, False)
            if not ok:
                return
            key = ttype.split()[0]
            b.metadata["transform_type"] = key
            if key in ("prefix", "suffix"):
                val, ok2 = QInputDialog.getMultiLineText(
                    self, f"TRANSFORM — {key}",
                    "Text to prepend/append:", b.metadata.get("transform_val", ""))
                if ok2: b.metadata["transform_val"] = val
            elif key == "replace":
                find_s, ok2 = QInputDialog.getText(
                    self, "TRANSFORM — find",
                    "Find this text:", text=b.metadata.get("transform_find", ""))
                if ok2: b.metadata["transform_find"] = find_s
                repl_s, ok3 = QInputDialog.getText(
                    self, "TRANSFORM — replace with",
                    "Replace with:", text=b.metadata.get("transform_repl", ""))
                if ok3: b.metadata["transform_repl"] = repl_s
            elif key == "truncate":
                n, ok2 = QInputDialog.getInt(
                    self, "TRANSFORM — truncate",
                    "Maximum characters:", value=int(b.metadata.get("transform_val", 500)),
                    min=1, max=999999)
                if ok2: b.metadata["transform_val"] = n
            b.label = f"⟲ {key.capitalize()}"

        elif b.btype == PipelineBlockType.MERGE:
            items = ["concat  — join with separator", "prepend  — put newest first",
                     "append  — put newest last", "json    — wrap all as JSON array"]
            cur  = b.metadata.get("merge_mode", "concat")
            mode, ok = QInputDialog.getItem(
                self, f"MERGE mode — '{b.label}'", "How to merge inputs:", items,
                next((i for i, s in enumerate(items) if cur in s), 0), False)
            if ok:
                b.metadata["merge_mode"] = mode.split()[0]
                if mode.split()[0] == "concat":
                    sep, ok2 = QInputDialog.getText(
                        self, "MERGE — separator",
                        "Separator between merged texts:", text=b.metadata.get("merge_sep", "\n\n---\n\n"))
                    if ok2: b.metadata["merge_sep"] = sep
                b.label = f"⊕ Merge/{mode.split()[0]}"

        elif b.btype == PipelineBlockType.SPLIT:
            n, ok = QInputDialog.getInt(
                self, f"SPLIT — '{b.label}'",
                "SPLIT broadcasts the same text to ALL outgoing connections.\n"
                "No configuration needed — just draw multiple output arrows.\n\n"
                "Confirm / view outgoing port count:", value=1, min=1, max=20)
            # No actual setting needed — split just fans out
            b.label = "⑁ Split"

        elif b.btype == PipelineBlockType.CUSTOM_CODE:
            dlg = _CodeEditorDialog(b, parent=self)
            dlg.exec()

        self.update()

    def _configure_llm_logic_block(self, b: "PipelineBlock"):
        """Open the LLM-Logic configuration dialog."""
        _LLM_TYPES = {
            PipelineBlockType.LLM_IF,
            PipelineBlockType.LLM_SWITCH,
            PipelineBlockType.LLM_FILTER,
            PipelineBlockType.LLM_TRANSFORM,
            PipelineBlockType.LLM_SCORE,
        }
        if b.btype not in _LLM_TYPES:
            return
        dlg = _LlmLogicEditorDialog(b, parent=self)
        dlg.exec()
        self.update()

    def _configure_context_block(self, b: "PipelineBlock"):
        """Configure a reference / knowledge / PDF block inline."""
        if b.btype == PipelineBlockType.REFERENCE:
            choice, ok = QInputDialog.getItem(
                self, f"Configure '{b.label}'", "Source:",
                ["Type / paste text", "Load from file"], 0, False)
            if not ok:
                return
            if choice.startswith("Type"):
                text, ok2 = QInputDialog.getMultiLineText(
                    self, "Reference Text", "Paste the reference text:")
                if ok2 and text.strip():
                    b.metadata["ref_text"] = text.strip()
                    name, ok3 = QInputDialog.getText(
                        self, "Name", "Short name:", text=b.label)
                    if ok3 and name.strip():
                        b.metadata["ref_name"] = name.strip()
                        b.label = name.strip()[:22]
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, "Select File", str(Path.home()),
                    "Text/Code (*.txt *.md *.py *.js *.ts *.json *.yaml *.rst *.html);;All (*)")
                if path:
                    try:
                        text = Path(path).read_text(encoding="utf-8", errors="replace")
                        b.metadata["ref_text"] = text
                        b.metadata["ref_name"] = Path(path).name
                        b.label = Path(path).name[:22]
                    except Exception as e:
                        QMessageBox.warning(self, "Read Error", str(e))
            self.update()

        elif b.btype == PipelineBlockType.KNOWLEDGE:
            text, ok = QInputDialog.getMultiLineText(
                self, "Knowledge Base", f"Enter knowledge text for '{b.label}':")
            if ok and text.strip():
                b.metadata["knowledge_text"] = text.strip()
                name, ok2 = QInputDialog.getText(
                    self, "Name", "Short name:", text=b.label)
                if ok2 and name.strip():
                    b.label = name.strip()[:22]
            self.update()

        elif b.btype == PipelineBlockType.PDF_SUMMARY:
            if not HAS_PDF:
                QMessageBox.warning(
                    self, "Missing Dependency",
                    "PyPDF2 is required.\n\n  pip install PyPDF2")
                return
            path, _ = QFileDialog.getOpenFileName(
                self, "Select PDF", str(Path.home()), "PDF (*.pdf)")
            if path:
                b.metadata["pdf_path"] = path
                b.label = Path(path).name[:22]

            # Role of this PDF relative to whatever arrived before it
            current_role = b.metadata.get("pdf_role", "reference")
            role, ok = QInputDialog.getItem(
                self,
                f"PDF Role — '{b.label}'",
                "How should this PDF relate to the prior block's output?\n\n"
                "  reference  →  prior output = MAIN,  PDF = REFERENCE\n"
                "  main       →  PDF = MAIN,  prior output = REFERENCE\n"
                "  (if there is no prior block, that side is simply omitted)",
                ["reference  —  prior is main, PDF is supporting context",
                 "main       —  PDF is primary, prior output is supporting context"],
                0 if current_role == "reference" else 1,
                False)
            if ok:
                b.metadata["pdf_role"] = "reference" if role.startswith("reference") else "main"
            self.update()

        elif b.btype == PipelineBlockType.INTERMEDIATE:
            # Show current settings as defaults
            current_prompt   = b.metadata.get("inter_prompt", "")
            current_position = b.metadata.get("inter_position", "above")

            # Step 1 — position choice
            position, ok = QInputDialog.getItem(
                self,
                f"Configure ◈ '{b.label}'",
                "Place your custom prompt:",
                ["Above incoming text  (prompt → model output)",
                 "Below incoming text  (model output → prompt)"],
                0 if current_position == "above" else 1,
                False)
            if not ok:
                return
            b.metadata["inter_position"] = (
                "above" if position.startswith("Above") else "below")

            # Step 2 — prompt text
            prompt_text, ok2 = QInputDialog.getMultiLineText(
                self,
                f"Configure ◈ '{b.label}'",
                "Custom prompt / instruction to inject at this stage:\n"
                "(Leave blank to clear and pass context through unchanged.)",
                current_prompt)
            if not ok2:
                return
            b.metadata["inter_prompt"] = prompt_text.strip()

            # Update block label so it's obvious it's configured
            if prompt_text.strip():
                preview = prompt_text.strip()[:16].replace("\n", " ")
                b.label = f"◈ {preview}…" if len(prompt_text.strip()) > 16 else f"◈ {preview}"
            else:
                b.label = "Intermediate"
            self.update()

    def _would_form_loop(self, from_bid: int, to_bid: int) -> bool:
        """Return True if to_bid can already reach from_bid (i.e. adding this edge creates a cycle)."""
        visited: set = set()
        queue = [to_bid]
        while queue:
            cur = queue.pop()
            if cur == from_bid:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            for c in self.connections:
                if c.from_block_id == cur:
                    queue.append(c.to_block_id)
        return False

    def _ctx_menu(self, pos):
        px, py = pos.x(), pos.y()
        menu   = QMenu(self)
        target = next((b for b in reversed(self.blocks)
                       if b.contains(px, py)), None)

        if target:
            act_del = menu.addAction(f"🗑  Delete '{target.label}'")
            act_ren = menu.addAction("✏️  Rename block")
            act_role = None
            act_cfg  = None
            if target.btype == PipelineBlockType.MODEL:
                act_role = menu.addAction("🔧  Change Role")
            _CONFIGURABLE = {
                PipelineBlockType.REFERENCE, PipelineBlockType.KNOWLEDGE,
                PipelineBlockType.PDF_SUMMARY, PipelineBlockType.INTERMEDIATE,
                PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
                PipelineBlockType.FILTER, PipelineBlockType.TRANSFORM,
                PipelineBlockType.MERGE, PipelineBlockType.SPLIT,
                PipelineBlockType.CUSTOM_CODE,
                PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
                PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
                PipelineBlockType.LLM_SCORE,
            }
            if target.btype in _CONFIGURABLE:
                act_cfg = menu.addAction("⚙️  Configure block…")
            menu.addSeparator()

        act_clr = menu.addAction("🗑  Clear All Blocks & Connections")
        chosen  = menu.exec(self.mapToGlobal(pos))
        if not chosen:
            return

        if target:
            if chosen == act_del:
                self.remove_block(target)
                return
            elif chosen == act_ren:
                name, ok = QInputDialog.getText(
                    self, "Rename Block", "New label:", text=target.label)
                if ok and name.strip():
                    target.label = name.strip()
                    self.update()
                return
            elif act_role and chosen == act_role:
                r, ok = QInputDialog.getItem(
                    self, "Select Role", "Role:", MODEL_ROLES, 0, False)
                if ok:
                    target.role = r
                    self.update()
                return
            elif act_cfg and chosen == act_cfg:
                _LOGIC_TYPES = {
                    PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
                    PipelineBlockType.FILTER, PipelineBlockType.TRANSFORM,
                    PipelineBlockType.MERGE, PipelineBlockType.SPLIT,
                    PipelineBlockType.CUSTOM_CODE,
                }
                _LLM_LOGIC_TYPES = {
                    PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
                    PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
                    PipelineBlockType.LLM_SCORE,
                }
                if target.btype in _LLM_LOGIC_TYPES:
                    self._configure_llm_logic_block(target)
                elif target.btype in _LOGIC_TYPES:
                    self._configure_logic_block(target)
                else:
                    self._configure_context_block(target)
                return

        if chosen == act_clr:
            if QMessageBox.question(
                self, "Clear Pipeline",
                "Remove ALL blocks and connections?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                self.clear_all()


class PipelineExecutionWorker(QThread):
    """
    Sequentially executes a validated pipeline graph.
    SERVER MODE ONLY — refuses to run if model cannot be started as a server.
    Retries server startup up to _SERVER_RETRIES times before aborting.
    """
    step_started      = pyqtSignal(int, str)     # (block_id, label)
    step_token        = pyqtSignal(int, str)     # (block_id, token)
    step_done         = pyqtSignal(int, str)     # (block_id, full_text)
    intermediate_live = pyqtSignal(int, str, str)# (block_id, label, text)
    pipeline_done     = pyqtSignal(str)
    err               = pyqtSignal(str)
    log_msg           = pyqtSignal(str)

    _SERVER_RETRIES   = 3
    _SERVER_RETRY_S   = 6

    def __init__(self, blocks: List[PipelineBlock],
                 connections: List[PipelineConnection],
                 input_text: str,
                 primary_engine: "LlamaEngine"):
        super().__init__()
        self.blocks         = {b.bid: b for b in blocks}
        self.connections    = connections
        self.input_text     = input_text
        self.primary_engine = primary_engine
        self._abort         = False

    def abort(self):
        self._abort = True

    # ── adjacency helpers ─────────────────────────────────────────────────────

    def _adjacency(self) -> Dict[int, List[PipelineConnection]]:
        adj: Dict[int, List[PipelineConnection]] = {}
        for c in self.connections:
            adj.setdefault(c.from_block_id, []).append(c)
        return adj

    def run(self):
        adj = self._adjacency()
        inp = next((b for b in self.blocks.values()
                    if b.btype == PipelineBlockType.INPUT), None)
        if not inp:
            self.err.emit("No INPUT block found."); return

        current_text = self.input_text
        visit_counts: Dict[str, int] = {}
        queue = [(inp.bid, self.input_text)]

        while queue and not self._abort:
            bid, context = queue.pop(0)
            b = self.blocks.get(bid)
            if not b:
                continue

            if b.btype == PipelineBlockType.INPUT:
                current_text = context

            elif b.btype == PipelineBlockType.INTERMEDIATE:
                self.step_started.emit(bid, b.label)
                inter_prompt   = b.metadata.get("inter_prompt", "").strip()
                inter_position = b.metadata.get("inter_position", "above")

                if inter_prompt:
                    if inter_position == "above":
                        current_text = f"{inter_prompt}\n\n{context}"
                    else:
                        current_text = f"{context}\n\n{inter_prompt}"
                    self.log_msg.emit(
                        f"◈  '{b.label}' injected prompt "
                        f"({'above' if inter_position == 'above' else 'below'} output) "
                        f"— {len(current_text):,} chars total")
                else:
                    current_text = context

                self.intermediate_live.emit(bid, b.label, current_text)
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.OUTPUT:
                # Find which block sent us this context so we can label it
                sender_label = ""
                for conn in self.connections:
                    if conn.to_block_id == bid:
                        sender = self.blocks.get(conn.from_block_id)
                        if sender:
                            sender_label = sender.label
                        break

                current_text = context
                # Emit step_done with the raw context (no intermediate noise)
                self.step_done.emit(bid, context)
                # pipeline_done carries (final_text, sender_label) encoded together
                # We encode as JSON so the tab can split them cleanly
                import json as _json
                self.pipeline_done.emit(_json.dumps({
                    "text":   context,
                    "sender": sender_label,
                }))
                return

            elif b.btype == PipelineBlockType.MODEL:
                self.step_started.emit(bid, b.label)
                result = self._run_model(b, context)
                if result is None:
                    if not self._abort:
                        self.err.emit(f"Model block '{b.label}' returned no output.")
                    return
                current_text = result
                self.step_done.emit(bid, result)

            elif b.btype == PipelineBlockType.REFERENCE:
                self.step_started.emit(bid, b.label)
                current_text = self._run_reference(b, context)
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.KNOWLEDGE:
                self.step_started.emit(bid, b.label)
                current_text = self._run_knowledge(b, context)
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.PDF_SUMMARY:
                self.step_started.emit(bid, b.label)
                result = self._run_pdf(b, context)
                if result is None:
                    if not self._abort:
                        self.err.emit(f"PDF block '{b.label}' failed.")
                    return
                current_text = result
                self.step_done.emit(bid, result)

            # ── Logic block execution ─────────────────────────────────────────
            elif b.btype == PipelineBlockType.IF_ELSE:
                self.step_started.emit(bid, b.label)
                cond = b.metadata.get("condition", "True")
                _log_lines: list = []
                _safe_ns = {
                    "text": context,
                    "log": lambda m, _ll=_log_lines: _ll.append(str(m)),
                    "__builtins__": {
                        "len": len, "str": str, "int": int, "float": float,
                        "bool": bool, "list": list, "dict": dict, "any": any,
                        "all": all, "min": min, "max": max, "abs": abs,
                        "True": True, "False": False, "None": None,
                        "isinstance": isinstance,
                    },
                }
                try:
                    _result = eval(compile(cond, "<if_else>", "eval"), _safe_ns)
                    branch_taken = "TRUE" if _result else "FALSE"
                except Exception as _e:
                    self.log_msg.emit(
                        f"⚠️  IF/ELSE '{b.label}' condition error: {_e} → defaulting FALSE")
                    branch_taken = "FALSE"
                self.log_msg.emit(
                    f"⑂  IF/ELSE '{b.label}': condition='{cond[:40]}' → {branch_taken}")
                current_text = context
                self.step_done.emit(bid, current_text)
                # Only enqueue the arm matching branch_taken
                for conn in adj.get(bid, []):
                    bl = getattr(conn, "branch_label", "")
                    # E=TRUE, W=FALSE; if no label treat all as pass-through
                    take = (not bl) or (bl == branch_taken) or (
                        bl == "TRUE" and branch_taken == "TRUE") or (
                        bl == "FALSE" and branch_taken == "FALSE")
                    if take:
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            queue.append((conn.to_block_id, current_text))
                continue   # skip generic enqueue below

            elif b.btype == PipelineBlockType.SWITCH:
                self.step_started.emit(bid, b.label)
                expr = b.metadata.get("switch_expr", "''")
                _safe_ns = {
                    "text": context,
                    "__builtins__": {
                        "len": len, "str": str, "int": int, "float": float,
                        "bool": bool, "True": True, "False": False, "None": None,
                    },
                }
                try:
                    switch_key = str(eval(compile(expr, "<switch>", "eval"), _safe_ns))
                except Exception as _e:
                    self.log_msg.emit(f"⚠️  SWITCH '{b.label}' expr error: {_e}")
                    switch_key = "__error__"
                self.log_msg.emit(
                    f"⑃  SWITCH '{b.label}': key='{switch_key}'")
                current_text = context
                self.step_done.emit(bid, current_text)
                _matched = False
                for conn in adj.get(bid, []):
                    bl = getattr(conn, "branch_label", "")
                    if bl == switch_key or bl == "default" or not bl:
                        _matched = True
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            queue.append((conn.to_block_id, current_text))
                if not _matched:
                    self.log_msg.emit(
                        f"⚠️  SWITCH '{b.label}': no arm matched key '{switch_key}' — dropped.")
                continue

            elif b.btype == PipelineBlockType.FILTER:
                self.step_started.emit(bid, b.label)
                cond = b.metadata.get("filter_cond", "True")
                _safe_ns = {
                    "text": context,
                    "__builtins__": {
                        "len": len, "str": str, "int": int, "bool": bool,
                        "any": any, "all": all, "True": True, "False": False,
                        "None": None, "isinstance": isinstance,
                    },
                }
                try:
                    _pass = bool(eval(compile(cond, "<filter>", "eval"), _safe_ns))
                except Exception as _e:
                    self.log_msg.emit(f"⚠️  FILTER error: {_e} → dropping")
                    _pass = False
                if _pass:
                    self.log_msg.emit(f"⊘  FILTER '{b.label}': PASSED")
                    current_text = context
                    self.step_done.emit(bid, current_text)
                else:
                    self.log_msg.emit(f"⊘  FILTER '{b.label}': DROPPED — pipeline stopped here.")
                    self.pipeline_done.emit(
                        __import__("json").dumps(
                            {"text": f"[FILTER DROPPED]\n\nCondition '{cond}' was False.\nOriginal text:\n{context}",
                             "sender": b.label}))
                    return

            elif b.btype == PipelineBlockType.TRANSFORM:
                self.step_started.emit(bid, b.label)
                ttype = b.metadata.get("transform_type", "prefix")
                val   = b.metadata.get("transform_val", "")
                try:
                    if ttype == "prefix":
                        current_text = f"{val}\n{context}"
                    elif ttype == "suffix":
                        current_text = f"{context}\n{val}"
                    elif ttype == "replace":
                        find_s = b.metadata.get("transform_find", "")
                        repl_s = b.metadata.get("transform_repl", "")
                        current_text = context.replace(find_s, repl_s) if find_s else context
                    elif ttype == "upper":
                        current_text = context.upper()
                    elif ttype == "lower":
                        current_text = context.lower()
                    elif ttype == "strip":
                        current_text = context.strip()
                    elif ttype == "truncate":
                        n = int(val) if val else 500
                        current_text = context[:n]
                    else:
                        current_text = context
                except Exception as _e:
                    self.log_msg.emit(f"⚠️  TRANSFORM error: {_e} — passing unchanged")
                    current_text = context
                self.log_msg.emit(
                    f"⟲  TRANSFORM '{b.label}': {ttype} → {len(current_text):,} chars")
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.MERGE:
                # Collect all contexts that arrived at this block this pass
                self.step_started.emit(bid, b.label)
                # The queue may have multiple entries for this bid; drain them
                _extras = [context]
                _new_queue = []
                for _qbid, _qctx in queue:
                    if _qbid == bid:
                        _extras.append(_qctx)
                    else:
                        _new_queue.append((_qbid, _qctx))
                queue = _new_queue
                mode = b.metadata.get("merge_mode", "concat")
                sep  = b.metadata.get("merge_sep", "\n\n---\n\n")
                try:
                    if mode == "concat":
                        current_text = sep.join(_extras)
                    elif mode == "prepend":
                        current_text = sep.join(reversed(_extras))
                    elif mode == "append":
                        current_text = sep.join(_extras)
                    elif mode == "json":
                        current_text = __import__("json").dumps(_extras, indent=2)
                    else:
                        current_text = sep.join(_extras)
                except Exception as _e:
                    current_text = sep.join(_extras)
                self.log_msg.emit(
                    f"⊕  MERGE '{b.label}': {len(_extras)} inputs → {len(current_text):,} chars")
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.SPLIT:
                self.step_started.emit(bid, b.label)
                current_text = context
                self.log_msg.emit(
                    f"⑁  SPLIT '{b.label}': broadcasting to {len(adj.get(bid,[]))} outputs")
                self.step_done.emit(bid, current_text)
                # Enqueue ALL outgoing connections (fan-out)
                for conn in adj.get(bid, []):
                    key = f"{conn.from_block_id}->{conn.to_block_id}"
                    visits = visit_counts.get(key, 0)
                    limit  = conn.loop_times if conn.is_loop else 1
                    if visits < limit:
                        visit_counts[key] = visits + 1
                        queue.append((conn.to_block_id, current_text))
                continue

            # ── LLM Logic blocks ──────────────────────────────────────────────
            elif b.btype == PipelineBlockType.LLM_IF:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction", "Does this text seem positive?")
                system = (
                    "You are a strict decision-making assistant. "
                    "Read the user's text and the condition carefully. "
                    "Respond with ONLY one word: YES or NO. "
                    "Do not add any explanation, punctuation, or extra words."
                )
                user_prompt = (
                    f"CONDITION: {instr}\n\n"
                    f"TEXT TO EVALUATE:\n{context[:3000]}\n\n"
                    f"Answer YES or NO:"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                # Parse YES/NO robustly
                _answer_up = raw.strip().upper().split()[0] if raw.strip() else "NO"
                branch_taken = "TRUE" if _answer_up in ("YES", "Y", "TRUE", "1", "PASS", "POSITIVE") else "FALSE"
                self.log_msg.emit(
                    f"🧠  LLM-IF '{b.label}': raw='{raw[:60]}' → {branch_taken}")
                current_text = context
                self.step_done.emit(bid, current_text)
                for conn in adj.get(bid, []):
                    bl = getattr(conn, "branch_label", "")
                    take = (not bl) or (bl == branch_taken) or (
                        bl == "TRUE" and branch_taken == "TRUE") or (
                        bl == "FALSE" and branch_taken == "FALSE")
                    if take:
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            queue.append((conn.to_block_id, current_text))
                continue

            elif b.btype == PipelineBlockType.LLM_SWITCH:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction", "")
                # Collect arm labels from outgoing connections
                arm_labels = list({
                    getattr(c, "branch_label", "")
                    for c in adj.get(bid, [])
                    if getattr(c, "branch_label", "")
                })
                arms_str = ", ".join(arm_labels) if arm_labels else "(no labels set)"
                system = (
                    "You are a strict classification assistant. "
                    "Read the user's text and task, then respond with ONLY the "
                    "exact category label from the provided list. "
                    "Do not add punctuation, explanation, or any other text. "
                    "If no category fits, respond with: other"
                )
                user_prompt = (
                    f"CLASSIFICATION TASK: {instr}\n"
                    f"VALID CATEGORIES (respond with exactly one): {arms_str}\n\n"
                    f"TEXT TO CLASSIFY:\n{context[:3000]}\n\n"
                    f"Your answer (one word/phrase only):"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                switch_key = raw.strip().split("\n")[0].strip().lower()
                # Normalise to closest arm label (case-insensitive)
                _match = next(
                    (lbl for lbl in arm_labels if lbl.lower() == switch_key),
                    None)
                if _match is None:
                    # Fuzzy: substring match
                    _match = next(
                        (lbl for lbl in arm_labels if lbl.lower() in switch_key
                         or switch_key in lbl.lower()), "default")
                self.log_msg.emit(
                    f"🧠  LLM-SWITCH '{b.label}': classified as '{_match}' (raw: '{raw[:60]}')")
                current_text = context
                self.step_done.emit(bid, current_text)
                _matched_any = False
                for conn in adj.get(bid, []):
                    bl = getattr(conn, "branch_label", "")
                    if bl.lower() == _match.lower() or bl == "default" or not bl:
                        _matched_any = True
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            queue.append((conn.to_block_id, current_text))
                if not _matched_any:
                    self.log_msg.emit(
                        f"⚠️  LLM-SWITCH '{b.label}': no arm matched '{_match}' — dropped")
                continue

            elif b.btype == PipelineBlockType.LLM_FILTER:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction", "Pass all text")
                system = (
                    "You are a strict content filter. "
                    "Read the condition and the text. "
                    "Respond with ONLY: PASS or STOP. "
                    "If the condition is met and the text should continue, say PASS. "
                    "If the condition is NOT met and the text should be dropped, say STOP. "
                    "Do not add any explanation."
                )
                user_prompt = (
                    f"PASS CONDITION: {instr}\n\n"
                    f"TEXT:\n{context[:3000]}\n\n"
                    f"Should this text pass? Answer PASS or STOP:"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                _ans = raw.strip().upper().split()[0] if raw.strip() else "STOP"
                _pass = _ans in ("PASS", "YES", "Y", "TRUE", "1", "ALLOW", "OK")
                if _pass:
                    self.log_msg.emit(f"🧠  LLM-FILTER '{b.label}': PASSED")
                    current_text = context
                    self.step_done.emit(bid, current_text)
                else:
                    self.log_msg.emit(
                        f"🧠  LLM-FILTER '{b.label}': STOPPED — model said: {raw[:80]}")
                    self.pipeline_done.emit(
                        __import__("json").dumps({
                            "text": (
                                f"[LLM FILTER STOPPED]\n\n"
                                f"Block: {b.label}\n"
                                f"Condition: {instr}\n"
                                f"Model decision: {raw[:200]}\n\n"
                                f"Original text:\n{context}"
                            ),
                            "sender": b.label,
                        }))
                    return

            elif b.btype == PipelineBlockType.LLM_TRANSFORM:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction", "Pass the text through unchanged.")
                max_tok = int(b.metadata.get("llm_max_tokens", 512))
                system = (
                    "You are a precise text transformation assistant. "
                    "Follow the user's instruction exactly. "
                    "Output ONLY the transformed text — no preamble, no explanation, "
                    "no 'Here is the result:', just the transformed content itself."
                )
                user_prompt = (
                    f"INSTRUCTION: {instr}\n\n"
                    f"TEXT TO TRANSFORM:\n{context}"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                # Strip common preamble patterns the model might add
                for _strip in ("Here is", "Here's", "Result:", "Output:", "Transformed:"):
                    if raw.lower().startswith(_strip.lower()):
                        raw = raw[len(_strip):].lstrip(": \n")
                        break
                current_text = raw
                self.log_msg.emit(
                    f"🧠  LLM-TRANSFORM '{b.label}': {len(context):,} → {len(current_text):,} chars")
                self.step_done.emit(bid, current_text)

            elif b.btype == PipelineBlockType.LLM_SCORE:
                self.step_started.emit(bid, b.label)
                instr = b.metadata.get("llm_instruction",
                                       "Rate the quality of this text (1=poor, 10=excellent)")
                system = (
                    "You are a precise scoring assistant. "
                    "Read the criterion and the text. "
                    "Respond with ONLY a single integer from 1 to 10. "
                    "No explanation, no punctuation, just the number."
                )
                user_prompt = (
                    f"SCORING CRITERION: {instr}\n\n"
                    f"TEXT TO SCORE:\n{context[:3000]}\n\n"
                    f"Score (1–10, integer only):"
                )
                raw = self._llm_block_call(b, context, system, user_prompt)
                if raw is None:
                    return
                # Parse score robustly
                try:
                    import re as _re2
                    _nums = _re2.findall(r'\b([1-9]|10)\b', raw)
                    score = int(_nums[0]) if _nums else 5
                except Exception:
                    score = 5
                score = max(1, min(10, score))
                if score <= 3:
                    band = "LOW"; arm_target = "E"
                elif score <= 7:
                    band = "MID"; arm_target = "S"
                else:
                    band = "HIGH"; arm_target = "W"
                self.log_msg.emit(
                    f"🧠  LLM-SCORE '{b.label}': score={score}/10 → {band} (model: '{raw[:40]}')")
                current_text = context
                self.step_done.emit(bid, current_text)
                # Route by band or 'score' label (raw score string)
                for conn in adj.get(bid, []):
                    bl = getattr(conn, "branch_label", "")
                    take = (
                        not bl                               # unlabelled → always
                        or bl.upper() == band                # LOW / MID / HIGH
                        or bl == arm_target                  # E / S / W
                        or bl.lower() == "score"             # raw score as text
                    )
                    if take:
                        key = f"{conn.from_block_id}->{conn.to_block_id}"
                        visits = visit_counts.get(key, 0)
                        limit  = conn.loop_times if conn.is_loop else 1
                        if visits < limit:
                            visit_counts[key] = visits + 1
                            # 'score' label receives the numeric string
                            out_txt = str(score) if bl.lower() == "score" else current_text
                            queue.append((conn.to_block_id, out_txt))
                continue

            elif b.btype == PipelineBlockType.CUSTOM_CODE:
                self.step_started.emit(bid, b.label)
                code = b.metadata.get("custom_code", "result = text")
                _log_lines: list = []
                _safe_builtins = {
                    "len": len, "str": str, "int": int, "float": float,
                    "bool": bool, "list": list, "dict": dict, "tuple": tuple,
                    "range": range, "enumerate": enumerate, "zip": zip,
                    "map": map, "filter": filter, "sorted": sorted,
                    "min": min, "max": max, "sum": sum, "abs": abs,
                    "round": round, "isinstance": isinstance, "hasattr": hasattr,
                    "getattr": getattr, "repr": repr, "type": type,
                    "print": lambda *a: _log_lines.append(" ".join(str(x) for x in a)),
                    "True": True, "False": False, "None": None,
                }
                _ns = {
                    "text":     context,
                    "result":   context,      # default: pass-through
                    "metadata": dict(b.metadata),
                    "log":      lambda m, _ll=_log_lines: _ll.append(str(m)),
                    "__builtins__": _safe_builtins,
                }
                try:
                    exec(compile(code, "<custom_code>", "exec"), _ns)
                    current_text = str(_ns.get("result", context))
                    for _line in _log_lines:
                        self.log_msg.emit(f"  ⌥  {_line}")
                    self.log_msg.emit(
                        f"⌥  CUSTOM_CODE '{b.label}': → {len(current_text):,} chars")
                except Exception as _e:
                    self.log_msg.emit(f"❌  CUSTOM_CODE '{b.label}' runtime error: {_e}")
                    self.err.emit(
                        f"Custom Code block '{b.label}' raised an exception:\n\n{_e}")
                    return
                self.step_done.emit(bid, current_text)

            # Enqueue outgoing edges
            for conn in adj.get(bid, []):
                key    = f"{conn.from_block_id}->{conn.to_block_id}"
                visits = visit_counts.get(key, 0)
                limit  = conn.loop_times if conn.is_loop else 1
                if visits < limit:
                    visit_counts[key] = visits + 1
                    queue.append((conn.to_block_id, current_text))

        if not self._abort:
            self.pipeline_done.emit(current_text)

    # ── server-mode guarantee with retries ────────────────────────────────────

    def _ensure_server(self, eng: "LlamaEngine", target: str) -> bool:
        """
        Ensure eng is running in server mode for target.
        Loads the model if necessary, then retries server startup up to
        _SERVER_RETRIES times. Returns True only when confirmed server mode.
        """
        if not eng.is_loaded or eng.model_path != target:
            self.log_msg.emit(f"🔄  Loading model: {Path(target).name}")
            if eng.is_loaded:
                eng.shutdown()
            ok = eng.load(target, log_cb=lambda m: self.log_msg.emit(m))
            if not ok:
                self.err.emit(f"❌  Could not load model: {Path(target).name}")
                return False
            self.log_msg.emit(f"✅  Model loaded.")

        if eng.mode == "server":
            self.log_msg.emit(f"🟢  Server already running on port {eng.server_port}")
            return True

        for attempt in range(1, self._SERVER_RETRIES + 1):
            self.log_msg.emit(
                f"⚡  Starting server (attempt {attempt}/{self._SERVER_RETRIES})…")
            ok = eng.ensure_server(log_cb=lambda m: self.log_msg.emit(m))
            if ok and eng.mode == "server":
                self.log_msg.emit(
                    f"🟢  Server mode confirmed — port {eng.server_port}")
                return True
            if attempt < self._SERVER_RETRIES:
                self.log_msg.emit(
                    f"⏳  Not ready — waiting {self._SERVER_RETRY_S}s before retry…")
                for _ in range(self._SERVER_RETRY_S * 10):
                    if self._abort:
                        return False
                    time.sleep(0.1)

        self.err.emit(
            f"❌  '{Path(target).name}' could not start in SERVER mode "
            f"after {self._SERVER_RETRIES} attempts.\n\n"
            f"Pipeline requires llama-server (not llama-cli).\n"
            f"Check that llama-server binary is present and the model path is valid.\n"
            f"See the Logs tab for details.")
        return False
    
    def _run_model(self, b: PipelineBlock, context: str) -> Optional[str]:
        target = b.model_path
        if not target or not Path(target).exists():
            self.err.emit(f"Model file not found: {target}"); return None

        eng = self.primary_engine
        if not eng:
            self.err.emit("No engine available."); return None

        # ── enforce server mode (with retries) ──────────────────────────────
        if not self._ensure_server(eng, target):
            return None   # error already emitted by _ensure_server

        fam = detect_model_family(target)
        cfg = get_model_registry().get_config(target)

        ROLE_SYSTEM = {
            "general":       "You are a helpful assistant.",
            "reasoning":     "You are a careful analytical reasoning assistant. Think step by step.",
            "summarization": "You are an expert summarization assistant. Be clear and concise.",
            "coding":        "You are an expert software engineer. Write clean, well-commented code.",
            "secondary":     "You are a versatile general-purpose assistant.",
        }
        sys_msg = ROLE_SYSTEM.get(b.role, ROLE_SYSTEM["general"])

        prompt = (
            fam.bos
            + fam.system_prefix + sys_msg + fam.system_suffix
            + fam.user_prefix + context + fam.user_suffix
            + fam.assistant_prefix
        )

        import http.client
        tokens: List[str] = []
        try:
            conn_h = http.client.HTTPConnection(
                "127.0.0.1", eng.server_port, timeout=600)
            body = json.dumps({
                "prompt": prompt, "n_predict": cfg.n_predict,
                "stream": True, "temperature": cfg.temperature,
                "top_p": cfg.top_p, "repeat_penalty": cfg.repeat_penalty,
                "stop": fam.stop_tokens,
            })
            conn_h.request("POST", "/completion", body,
                           {"Content-Type": "application/json"})
            r = conn_h.getresponse()
            if r.status != 200:
                self.err.emit(f"Server HTTP {r.status}"); return None
            buf = b""
            while not self._abort:
                byte = r.read(1)
                if not byte:
                    break
                buf += byte
                if byte == b"\n":
                    line = buf.decode("utf-8", errors="replace").strip()
                    buf  = b""
                    if line.startswith("data: "):
                        try:
                            d = json.loads(line[6:])
                            if d.get("stop"):
                                break
                            tok = d.get("content", "")
                            if tok:
                                tokens.append(tok)
                                self.step_token.emit(b.bid, tok)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            self.err.emit(f"Server error during model block: {e}"); return None

        return "".join(tokens).strip()

    # ── LLM query helper (synchronous, short answer) ──────────────────────────

    def _llm_query_sync(self, model_path: str, system_prompt: str,
                        user_prompt: str, max_tokens: int = 64,
                        temperature: float = 0.1) -> Optional[str]:
        """
        Call llama-server synchronously and return the full response text.
        Ensures the server is running for model_path before calling.
        Returns None on failure (error already logged to log_msg).
        """
        eng = self.primary_engine
        if not eng:
            self.log_msg.emit("❌  No primary engine available for LLM logic block.")
            return None
        if not self._ensure_server(eng, model_path):
            return None

        fam = detect_model_family(model_path)
        prompt = (
            fam.bos
            + fam.system_prefix + system_prompt + fam.system_suffix
            + fam.user_prefix   + user_prompt   + fam.user_suffix
            + fam.assistant_prefix
        )

        import http.client as _hc
        try:
            ch = _hc.HTTPConnection("127.0.0.1", eng.server_port, timeout=120)
            body = json.dumps({
                "prompt":        prompt,
                "n_predict":     max_tokens,
                "stream":        False,
                "temperature":   temperature,
                "stop":          fam.stop_tokens,
            })
            ch.request("POST", "/completion", body,
                       {"Content-Type": "application/json"})
            resp = ch.getresponse()
            if resp.status != 200:
                self.log_msg.emit(f"❌  LLM query HTTP {resp.status}")
                return None
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return data.get("content", "").strip()
        except Exception as _e:
            self.log_msg.emit(f"❌  LLM query error: {_e}")
            return None

    def _llm_block_call(self, b: "PipelineBlock", context: str,
                        system: str, user_prompt: str) -> Optional[str]:
        """Wrapper around _llm_query_sync that reads block metadata for settings."""
        meta       = b.metadata
        model_path = b.model_path or meta.get("llm_model_path", "")
        max_tok    = int(meta.get("llm_max_tokens", 64))
        temp       = float(meta.get("llm_temp", 0.1))
        show_r     = bool(meta.get("llm_show_reasoning", True))
        passthru   = bool(meta.get("llm_passthrough_on_err", False))

        if not model_path or not Path(model_path).exists():
            msg = f"❌  LLM block '{b.label}': model not found — {model_path}"
            self.log_msg.emit(msg)
            if passthru:
                self.log_msg.emit(f"⚠️  '{b.label}': passthrough-on-error → continuing unchanged")
                return context
            self.err.emit(msg)
            return None

        result = self._llm_query_sync(model_path, system, user_prompt, max_tok, temp)

        if result is None:
            if passthru:
                self.log_msg.emit(
                    f"⚠️  '{b.label}': LLM call failed, passthrough-on-error → continuing unchanged")
                return context
            self.err.emit(
                f"LLM logic block '{b.label}' failed to get a response from the model.\n"
                f"Check that the model is valid and the server can start.")
            return None

        if show_r:
            self.log_msg.emit(f"  🧠  [{b.label}] model said: {result[:120]}")
        return result

    # ── context injection blocks ──────────────────────────────────────────────

    def _run_reference(self, b: PipelineBlock, context: str) -> str:
        text = b.metadata.get("ref_text", "")
        if not text:
            self.log_msg.emit(f"⚠️  Reference '{b.label}' has no text — passing unchanged.")
            return context
        name = b.metadata.get("ref_name", b.label)
        self.log_msg.emit(f"📎  Injecting reference '{name}' ({len(text):,} chars)")
        return (
            f"[REFERENCE: {name}]\n"
            f"{text[:4000]}"
            + ("…\n[truncated]" if len(text) > 4000 else "")
            + f"\n[/REFERENCE]\n\n{context}"
        )

    def _run_knowledge(self, b: PipelineBlock, context: str) -> str:
        text = b.metadata.get("knowledge_text", "")
        if not text:
            self.log_msg.emit(f"⚠️  Knowledge '{b.label}' has no text — passing unchanged.")
            return context
        self.log_msg.emit(f"💡  Injecting knowledge ({len(text):,} chars)")
        return (
            f"Knowledge Base:\n"
            f"{text[:3000]}"
            + ("…\n[truncated]" if len(text) > 3000 else "")
            + f"\n\n---\n\n{context}"
        )

    def _run_pdf(self, b: PipelineBlock, context: str) -> Optional[str]:
        pdf_path = b.metadata.get("pdf_path", "")
        if not pdf_path or not Path(pdf_path).exists():
            self.err.emit(f"PDF block '{b.label}': file not found → {pdf_path}")
            return None
        if not HAS_PDF:
            self.err.emit("PyPDF2 not installed. Run: pip install PyPDF2")
            return None
        try:
            from PyPDF2 import PdfReader
            reader   = PdfReader(pdf_path)
            pdf_text = "\n".join(pg.extract_text() or "" for pg in reader.pages)
        except Exception as e:
            self.err.emit(f"PDF read error: {e}"); return None
        if not pdf_text.strip():
            self.err.emit(f"PDF block '{b.label}': no text extracted."); return None

        fname    = Path(pdf_path).name
        pdf_role = b.metadata.get("pdf_role", "reference")   # "reference" | "main"
        self.log_msg.emit(
            f"📄  PDF loaded: {fname} ({len(pdf_text):,} chars) "
            f"— role: {pdf_role.upper()}")

        # ── Summarise large PDFs ──────────────────────────────────────────────
        LIMIT = 4500
        if len(pdf_text) > LIMIT:
            pdf_text = self._summarise_pdf(b, fname, pdf_text)
            if pdf_text is None:
                return None

        # ── Compose output based on role ──────────────────────────────────────
        #
        #  pdf_role == "reference":
        #    prior context  → MAIN TEXT
        #    pdf_text       → REFERENCE (appended below)
        #
        #  pdf_role == "main":
        #    pdf_text       → MAIN TEXT
        #    prior context  → REFERENCE (appended below)
        #
        #  If either side is empty/None, it is simply omitted.

        prior = context.strip() if context else ""
        pdf   = pdf_text.strip()

        if pdf_role == "reference":
            main_text = prior
            ref_label = f"[PDF REFERENCE: {fname}]"
            ref_text  = pdf
        else:  # "main"
            main_text = pdf
            ref_label = "[PRIOR CONTEXT REFERENCE]"
            ref_text  = prior

        parts = []
        if main_text:
            parts.append(main_text)
        if ref_text:
            parts.append(f"{ref_label}\n{ref_text}\n[/REFERENCE]")

        result = "\n\n".join(parts)
        self.log_msg.emit(
            f"✅  PDF block assembled: {len(result):,} chars "
            f"({'PDF=main, prior=ref' if pdf_role == 'main' else 'prior=main, PDF=ref'})")
        return result

    def _summarise_pdf(self, b: "PipelineBlock", fname: str,
                       pdf_text: str) -> Optional[str]:
        """Chunk-summarise a large PDF via the server. Returns summary string or None."""
        eng   = self.primary_engine
        CHUNK = 3000
        chunks, text = [], pdf_text
        while text:
            if len(text) <= CHUNK:
                chunks.append(text.strip()); break
            cut = text.rfind("\n\n", 0, CHUNK)
            if cut < 200: cut = CHUNK
            chunks.append(text[:cut].strip())
            text = text[cut:].lstrip()

        fam       = detect_model_family(getattr(eng, "model_path", ""))
        summaries = []
        for i, chunk in enumerate(chunks):
            if self._abort:
                return None
            self.log_msg.emit(f"  📄  Summarising chunk {i+1}/{len(chunks)}…")
            prompt = (
                fam.bos + fam.user_prefix +
                f"Summarise this document section concisely. "
                f"File: '{fname}' — Section {i+1}/{len(chunks)}\n\n{chunk}" +
                fam.user_suffix + fam.assistant_prefix
            )
            if eng and eng.mode == "server":
                import http.client
                try:
                    ch = http.client.HTTPConnection(
                        "127.0.0.1", eng.server_port, timeout=300)
                    ch.request("POST", "/completion",
                               json.dumps({"prompt": prompt, "n_predict": 400,
                                           "stream": False, "temperature": 0.3,
                                           "stop": fam.stop_tokens}),
                               {"Content-Type": "application/json"})
                    resp = ch.getresponse()
                    if resp.status == 200:
                        d = json.loads(resp.read().decode("utf-8", errors="replace"))
                        summaries.append(f"[§{i+1}] {d.get('content','').strip()}")
                        continue
                except Exception:
                    pass
            summaries.append(f"[§{i+1}] {chunk[:600]}…")

        summary = "\n\n".join(summaries)
        self.log_msg.emit(f"✅  PDF summarised: {len(summary):,} chars")
        return summary
    
class PipelineOutputRenderer(QTextEdit):
    """
    Read-only rich output pane used for pipeline intermediate and final output.
    Parses the same tokens as the main chat window:
      - ```lang ... ```  → syntax-highlighted code block (Consolas, bg panel)
      - **bold**         → bold
      - `inline code`    → monospace highlighted span
      - Plain text lines → normal paragraph
    Accepts either streaming token-by-token calls (append_token) or
    a full set_content(text) replacement.
    """

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setObjectName("chat_te")
        self._raw = ""          # accumulated raw text
        self._placeholder_text = placeholder
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(120)   # debounce re-renders during streaming
        self._render_timer.timeout.connect(self._render)
        if placeholder:
            self.setPlaceholderText(placeholder)

    # ── public API ────────────────────────────────────────────────────────────

    def append_token(self, token: str):
        self._raw += token
        self._render_timer.start()   # debounce

    def set_content(self, text: str):
        self._raw = text
        self._render()

    def clear_content(self):
        self._raw = ""
        self.clear()

    def raw_text(self) -> str:
        return self._raw

    # ── renderer ──────────────────────────────────────────────────────────────

    def _render(self):
        html  = self._to_html(self._raw)
        cur   = self.verticalScrollBar().value()
        at_bt = cur >= self.verticalScrollBar().maximum() - 40
        self.setHtml(html)
        if at_bt:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum())
        else:
            self.verticalScrollBar().setValue(cur)

    def _to_html(self, text: str) -> str:
        """Convert raw pipeline text to HTML using the same rules as the chat window."""
        CODE_BG   = C.get("bg1", "#1e1e2e")
        CODE_FG   = C.get("acc2", "#a6e3a1")
        INLINE_BG = C.get("bg2", "#313244")
        INLINE_FG = C.get("acc",  "#cba6f7")
        TXT       = C.get("txt",  "#cdd6f4")
        TXT2      = C.get("txt2", "#a6adc8")

        body_css = (
            f"font-family:'Inter',sans-serif;font-size:12px;"
            f"color:{TXT};background:transparent;"
            f"margin:0;padding:4px 8px;line-height:1.65;"
        )
        code_css = (
            f"font-family:'Consolas','Fira Code',monospace;font-size:11px;"
            f"color:{CODE_FG};background:{CODE_BG};"
            f"display:block;padding:10px 14px;margin:6px 0;"
            f"border-radius:6px;border-left:3px solid {CODE_FG};"
            f"white-space:pre;"
        )
        inline_css = (
            f"font-family:'Consolas','Fira Code',monospace;font-size:11px;"
            f"color:{INLINE_FG};background:{INLINE_BG};"
            f"padding:1px 4px;border-radius:3px;"
        )

        import re, html as _html

        # Split into code-block segments and normal text segments
        segments   = re.split(r'(```(?:[a-zA-Z0-9+\-]*)\n[\s\S]*?```)', text)
        html_parts = []

        for seg in segments:
            cb = re.match(r'```([a-zA-Z0-9+\-]*)\n([\s\S]*?)```', seg)
            if cb:
                lang  = cb.group(1) or "text"
                code  = _html.escape(cb.group(2).rstrip())
                html_parts.append(
                    f'<div style="{code_css}">'
                    f'<span style="font-size:9px;color:{TXT2};'
                    f'font-family:Inter,sans-serif;">{lang}</span><br>'
                    f'{code}</div>')
            else:
                # Process inline elements line-by-line
                lines = seg.split("\n")
                para_lines = []
                for line in lines:
                    # Escape HTML first
                    safe = _html.escape(line)
                    # **bold**
                    safe = re.sub(
                        r'\*\*(.+?)\*\*',
                        r'<b>\1</b>', safe)
                    # `inline code`
                    safe = re.sub(
                        r'`([^`]+)`',
                        f'<span style="{inline_css}">\\1</span>', safe)
                    # Headers: ### ## #
                    hm = re.match(r'^(#{1,3})\s+(.*)', safe)
                    if hm:
                        lvl  = len(hm.group(1))
                        size = {1: "16px", 2: "14px", 3: "13px"}.get(lvl, "13px")
                        safe = (f'<span style="font-size:{size};font-weight:700;'
                                f'color:{TXT};">{hm.group(2)}</span>')
                    # Bullet lists: - or *
                    elif re.match(r'^[\-\*]\s+', safe):
                        safe = "• " + safe[2:].lstrip()
                    para_lines.append(safe)

                # Group into paragraphs separated by blank lines
                combined = "<br>".join(para_lines)
                if combined.strip():
                    html_parts.append(
                        f'<p style="margin:2px 0;color:{TXT};">{combined}</p>')

        return (
            f'<html><body style="{body_css}">'
            + "".join(html_parts)
            + "</body></html>"
        )

# ═════════════════════════════ LLM LOGIC EDITOR DIALOG ═══════════════════════

class _LlmLogicEditorDialog(QDialog if hasattr(__builtins__, '__import__') else object):
    """
    Configuration dialog for LLM-backed logic blocks.
    The user writes conditions and instructions in plain English.
    The attached model evaluates them at runtime against the incoming text.
    """

    # Per-type documentation shown in the dialog
    _TYPE_INFO = {
        "llm_if": {
            "icon":  "🧠 LLM IF / ELSE",
            "about": (
                "The model reads the incoming text and your condition, then answers "
                "YES or NO. YES routes to the TRUE (E) port, NO to the FALSE (W) port.\n\n"
                "Write your condition as a plain English question or statement.\n"
                "The model will always respond with a single word: YES or NO."
            ),
            "label":       "Condition (plain English):",
            "placeholder": "e.g.  Does this text contain a complaint or negative sentiment?\n"
                           "e.g.  Is the answer longer than a short paragraph?\n"
                           "e.g.  Does the user seem confused or ask a follow-up question?",
            "branch_hint": "TRUE (E port) → YES    FALSE (W port) → NO",
        },
        "llm_switch": {
            "icon":  "🧠 LLM SWITCH",
            "about": (
                "The model classifies the incoming text into one of your defined categories. "
                "Draw outgoing arrows and label each with the exact category name. "
                "The model picks the best match and only that arm is followed."
            ),
            "label":       "Classification task + categories:",
            "placeholder": "e.g.  Classify this text as one of: positive, negative, neutral\n"
                           "e.g.  What language is this in? english, french, spanish, other\n"
                           "e.g.  Is this a question, a complaint, a compliment, or other?",
            "branch_hint": "Connect outgoing arrows labelled with each category name.",
        },
        "llm_filter": {
            "icon":  "🧠 LLM FILTER",
            "about": (
                "The model decides whether the incoming text should continue through "
                "the pipeline (PASS) or be dropped (STOP). "
                "If dropped, the pipeline ends with a clear reason message."
            ),
            "label":       "Pass condition (plain English):",
            "placeholder": "e.g.  Only pass if this is a genuine technical question\n"
                           "e.g.  Pass only if the text contains a clear action item\n"
                           "e.g.  Allow through only if the topic is related to software",
            "branch_hint": "PASS → continues    STOP → pipeline ends with reason",
        },
        "llm_transform": {
            "icon":  "🧠 LLM TRANSFORM",
            "about": (
                "The model rewrites or transforms the incoming text according to your "
                "instruction. The result replaces the context for all downstream blocks. "
                "Be specific — the model will follow your instruction precisely."
            ),
            "label":       "Transformation instruction:",
            "placeholder": "e.g.  Summarise this in three bullet points\n"
                           "e.g.  Rewrite in a formal professional tone\n"
                           "e.g.  Extract only the action items as a numbered list\n"
                           "e.g.  Translate to Spanish",
            "branch_hint": "Single output — transformed text flows to all connected blocks.",
        },
        "llm_score": {
            "icon":  "🧠 LLM SCORE",
            "about": (
                "The model scores the incoming text from 1 to 10 on your criterion. "
                "Route the result by score band:\n"
                "  LOW  (1–3)   → E port\n"
                "  MID  (4–7)   → S port\n"
                "  HIGH (8–10)  → W port\n"
                "You can also connect a 'score' label to receive the raw score as text."
            ),
            "label":       "Scoring criterion:",
            "placeholder": "e.g.  Rate the clarity of this explanation (1=very unclear, 10=crystal clear)\n"
                           "e.g.  Score the sentiment positivity (1=very negative, 10=very positive)\n"
                           "e.g.  Rate the technical complexity (1=very simple, 10=expert-level)",
            "branch_hint": "LOW (E) 1–3    MID (S) 4–7    HIGH (W) 8–10    score label = raw number",
        },
    }

    def __init__(self, block: "PipelineBlock", parent=None):
        try:
            from PyQt6.QtWidgets import QDialog
            super().__init__(parent)
        except Exception:
            return
        self._block = block
        info = self._TYPE_INFO.get(block.btype, {})
        self.setWindowTitle(f"{info.get('icon','🧠')} — {block.label}")
        self.setMinimumSize(640, 480)
        self.resize(700, 530)
        self._build(info)

    def _build(self, info: dict):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ── Header ───────────────────────────────────────────────────────────
        hdr = QLabel(info.get("icon", "🧠  LLM Logic Block"))
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:14px;font-weight:bold;")
        root.addWidget(hdr)

        about = QLabel(info.get("about", ""))
        about.setWordWrap(True)
        about.setStyleSheet(
            f"color:{C['txt2']};font-size:11px;"
            f"background:{C['bg2']};border-radius:6px;padding:10px 12px;"
            f"border-left:3px solid {C['acc']};")
        root.addWidget(about)

        # ── Branch routing hint ───────────────────────────────────────────────
        hint_lbl = QLabel(f"📌  {info.get('branch_hint','')}")
        hint_lbl.setStyleSheet(
            f"color:{C['ok']};font-size:10px;font-weight:600;"
            f"padding:4px 8px;background:{C['bg2']};border-radius:4px;")
        hint_lbl.setWordWrap(True)
        root.addWidget(hint_lbl)

        # ── Model selector ────────────────────────────────────────────────────
        model_lbl = QLabel("MODEL  (required — evaluated at runtime):")
        model_lbl.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(model_lbl)

        model_row = QHBoxLayout(); model_row.setSpacing(8)
        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(30)
        # Populate from registry
        _models = get_model_registry().all_models()
        _cur_path = self._block.model_path or self._block.metadata.get("llm_model_path", "")
        _sel_idx = 0
        for _i, _m in enumerate(_models):
            self.model_combo.addItem(
                f"{ROLE_ICONS.get(_m.get('role','general'),'💬')}  {_m['name']}",
                _m["path"])
            if _m["path"] == _cur_path:
                _sel_idx = _i
        if _models:
            self.model_combo.setCurrentIndex(_sel_idx)
        else:
            self.model_combo.addItem("⚠️  No models registered — add one in Models tab", "")

        btn_browse_model = QPushButton("Browse…")
        btn_browse_model.setFixedHeight(30); btn_browse_model.setFixedWidth(80)
        btn_browse_model.clicked.connect(self._browse_model)

        self.model_status = QLabel("")
        self.model_status.setFixedWidth(18)
        self.model_combo.currentIndexChanged.connect(self._update_model_status)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.model_status)
        model_row.addWidget(btn_browse_model)
        root.addLayout(model_row)
        self._update_model_status()

        # ── Condition / instruction editor ────────────────────────────────────
        instr_lbl = QLabel(info.get("label", "Instruction:"))
        instr_lbl.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(instr_lbl)

        self.instr_edit = QTextEdit()
        self.instr_edit.setFont(QFont("Inter", 12))
        self.instr_edit.setMaximumHeight(130)
        self.instr_edit.setPlaceholderText(
            info.get("placeholder", "Enter your instruction in plain English…"))
        self.instr_edit.setPlainText(
            self._block.metadata.get("llm_instruction", ""))
        root.addWidget(self.instr_edit)

        # ── Advanced settings (collapsible) ───────────────────────────────────
        adv_hdr = QLabel("▸  ADVANCED SETTINGS  (click to expand)")
        adv_hdr.setStyleSheet(
            f"color:{C['txt2']};font-size:10px;font-weight:600;"
            f"padding:4px 2px;")
        adv_hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        root.addWidget(adv_hdr)

        self._adv_frame = QFrame()
        self._adv_frame.setObjectName("tab_card")
        self._adv_frame.setVisible(False)
        adv_l = QVBoxLayout(self._adv_frame)
        adv_l.setContentsMargins(12, 8, 12, 10); adv_l.setSpacing(8)

        def _adv_row(lbl_text, widget):
            r = QHBoxLayout(); r.setSpacing(8)
            l = QLabel(lbl_text); l.setFixedWidth(180)
            l.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
            r.addWidget(l); r.addWidget(widget); r.addStretch()
            adv_l.addLayout(r)

        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(8, 512)
        self.spin_max_tokens.setValue(int(self._block.metadata.get("llm_max_tokens", 64)))
        self.spin_max_tokens.setFixedHeight(26); self.spin_max_tokens.setFixedWidth(80)
        self.spin_max_tokens.setToolTip(
            "Max tokens the model generates for its answer.\n"
            "Keep small for routing blocks (16–64), larger for transform (128–512).")
        _adv_row("Max response tokens:", self.spin_max_tokens)

        self.spin_temp = QSpinBox()
        self.spin_temp.setRange(0, 100)
        self.spin_temp.setValue(int(float(self._block.metadata.get("llm_temp", 0.1)) * 100))
        self.spin_temp.setFixedHeight(26); self.spin_temp.setFixedWidth(80)
        self.spin_temp.setSuffix("  (÷100)")
        self.spin_temp.setToolTip(
            "Temperature for the LLM response (0–100 maps to 0.00–1.00).\n"
            "Use 0–15 for routing decisions, 30–60 for creative transforms.")
        _adv_row("Temperature (×100):", self.spin_temp)

        self.check_show_reasoning = QCheckBox("Show model reasoning in log")
        self.check_show_reasoning.setChecked(
            bool(self._block.metadata.get("llm_show_reasoning", True)))
        self.check_show_reasoning.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        adv_l.addWidget(self.check_show_reasoning)

        self.check_passthrough_on_err = QCheckBox(
            "Pass text through unchanged if model call fails (instead of stopping pipeline)")
        self.check_passthrough_on_err.setChecked(
            bool(self._block.metadata.get("llm_passthrough_on_err", False)))
        self.check_passthrough_on_err.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        adv_l.addWidget(self.check_passthrough_on_err)

        root.addWidget(self._adv_frame)
        adv_hdr.mousePressEvent = lambda _: self._toggle_adv(adv_hdr)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        btn_save = QPushButton("💾  Save & Close")
        btn_save.setObjectName("btn_send"); btn_save.setFixedHeight(32)
        btn_save.clicked.connect(self._save_and_close)
        btn_cancel = QPushButton("✕  Cancel")
        btn_cancel.setFixedHeight(32)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

    def _toggle_adv(self, hdr_lbl: QLabel):
        vis = not self._adv_frame.isVisible()
        self._adv_frame.setVisible(vis)
        hdr_lbl.setText(
            ("▾  ADVANCED SETTINGS  (click to collapse)"
             if vis else "▸  ADVANCED SETTINGS  (click to expand)"))

    def _update_model_status(self):
        path = self.model_combo.currentData() or ""
        ok = bool(path) and Path(path).exists()
        self.model_status.setText("✅" if ok else "❌")
        self.model_status.setToolTip(path if path else "No model selected")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", str(Path.home()),
            "GGUF Models (*.gguf);;All Files (*)")
        if path:
            # Add to registry if not already there
            get_model_registry().add(path)
            # Refresh combo
            already = self.model_combo.findData(path)
            if already == -1:
                fam = detect_model_family(path)
                self.model_combo.addItem(f"💬  {Path(path).name}", path)
                already = self.model_combo.count() - 1
            self.model_combo.setCurrentIndex(already)

    def _save_and_close(self):
        instr = self.instr_edit.toPlainText().strip()
        if not instr:
            QMessageBox.warning(self, "Missing Instruction",
                                "Please enter a condition or instruction."); return
        model_path = self.model_combo.currentData() or ""
        if not model_path or not Path(model_path).exists():
            ans = QMessageBox.question(
                self, "No Model Selected",
                "No valid model is selected. The block will fail at runtime.\n\n"
                "Save anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return

        self._block.model_path = model_path
        self._block.metadata["llm_instruction"]      = instr
        self._block.metadata["llm_model_path"]       = model_path
        self._block.metadata["llm_max_tokens"]       = self.spin_max_tokens.value()
        self._block.metadata["llm_temp"]             = round(self.spin_temp.value() / 100, 2)
        self._block.metadata["llm_show_reasoning"]   = self.check_show_reasoning.isChecked()
        self._block.metadata["llm_passthrough_on_err"] = self.check_passthrough_on_err.isChecked()

        # Build a readable label from the instruction
        _preview = instr.replace("\n", " ").strip()[:22]
        _icon_map = {
            "llm_if": "🧠⑂", "llm_switch": "🧠⑃",
            "llm_filter": "🧠⊘", "llm_transform": "🧠⟲", "llm_score": "🧠★",
        }
        _icon = _icon_map.get(self._block.btype, "🧠")
        self._block.label = f"{_icon} {_preview}"
        self.accept()


# ═════════════════════════════ CODE EDITOR DIALOG ════════════════════════════

class _CodeEditorDialog(QDialog if hasattr(__builtins__, '__import__') else object):
    """
    Full code editor for CUSTOM_CODE pipeline blocks.
    Shows available variables, validates syntax live, saves to block.metadata.
    """

    # Available variable documentation shown to user
    _VAR_DOCS = [
        ("text",    "str",  "The incoming context string from the previous block."),
        ("result",  "str",  "Write your output here — the pipeline continues with this."),
        ("metadata","dict", "Block metadata dict (read/write — persists across runs)."),
        ("log",     "fn",   "log('message') — writes to the pipeline execution log."),
    ]

    _TEMPLATE = """\
# CUSTOM_CODE block
# Variables available:
#   text     → incoming context string (read-only)
#   result   → set this to your output (default: text unchanged)
#   metadata → block metadata dict (persists across runs)
#   log(msg) → write to pipeline log
#
# Example: count words
result = text   # default: pass through unchanged

word_count = len(text.split())
log(f"Word count: {word_count}")

# You can also modify result:
# result = text.upper()
# result = f"[Processed]\\n{text}"
"""

    def __init__(self, block: "PipelineBlock", parent=None):
        try:            
            super().__init__(parent)
        except Exception:
            return
        self._block = block
        self.setWindowTitle(f"⌥ Code Editor — {block.label}")
        self.setMinimumSize(760, 560)
        self.resize(820, 620)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # ── Header ───────────────────────────────────────────────────────────
        hdr = QLabel("⌥  Custom Code Block Editor")
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:14px;font-weight:bold;")
        root.addWidget(hdr)
        desc = QLabel(
            "Write Python that runs inline during pipeline execution. "
            "Your code receives <b>text</b> (incoming context) and must set "
            "<b>result</b> (output sent to next block). "
            "Code runs in a sandboxed exec() with no network or disk access.")
        desc.setWordWrap(True)
        desc.setTextFormat(Qt.TextFormat.RichText)
        desc.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        root.addWidget(desc)

        # ── Available variables table ─────────────────────────────────────────
        vars_hdr = QLabel("AVAILABLE VARIABLES")
        vars_hdr.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(vars_hdr)

        vars_frame = QFrame(); vars_frame.setObjectName("tab_card")
        vars_l = QHBoxLayout(vars_frame)
        vars_l.setContentsMargins(10, 8, 10, 8); vars_l.setSpacing(16)
        for name, typ, doc in self._VAR_DOCS:
            col = QVBoxLayout(); col.setSpacing(2)
            n_lbl = QLabel(f"<b>{name}</b>  <span style='color:{C['txt3']}'>{typ}</span>")
            n_lbl.setTextFormat(Qt.TextFormat.RichText)
            n_lbl.setStyleSheet(
                f"font-family:Consolas,monospace;font-size:12px;color:{C['acc2']};")
            d_lbl = QLabel(doc)
            d_lbl.setWordWrap(True)
            d_lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
            col.addWidget(n_lbl); col.addWidget(d_lbl)
            vars_l.addLayout(col, 1)
        root.addWidget(vars_frame)

        # ── Code editor ───────────────────────────────────────────────────────
        code_hdr = QLabel("PYTHON CODE")
        code_hdr.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1px;")
        root.addWidget(code_hdr)

        self.editor = QTextEdit()
        _code_font = QFont("Consolas", 11)
        _code_font.setPointSize(max(1, _code_font.pointSize()))
        self.editor.setFont(_code_font)
        self.editor.setObjectName("log_te")
        try:
            self.editor.setTabStopDistance(28.0)
        except Exception:
            pass
        self.editor.setPlaceholderText(
            "# Write Python here. Set 'result' to your output string.")
        saved_code = self._block.metadata.get("custom_code", "")
        self.editor.setPlainText(saved_code if saved_code else self._TEMPLATE)
        self.editor.textChanged.connect(self._on_edit)
        root.addWidget(self.editor, 1)

        # ── Syntax status ─────────────────────────────────────────────────────
        self.syntax_lbl = QLabel("✅  Syntax OK")
        self.syntax_lbl.setStyleSheet(f"color:{C['ok']};font-size:11px;")
        root.addWidget(self.syntax_lbl)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self.btn_test = QPushButton("🧪  Test with sample text…")
        self.btn_test.setFixedHeight(30)
        self.btn_test.clicked.connect(self._run_test)
        btn_ok = QPushButton("💾  Save & Close")
        btn_ok.setObjectName("btn_send")
        btn_ok.setFixedHeight(30)
        btn_ok.clicked.connect(self._save_and_close)
        btn_cancel = QPushButton("✕  Cancel")
        btn_cancel.setFixedHeight(30)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_test)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        root.addLayout(btn_row)

        # Run initial syntax check
        self._on_edit()

    def _on_edit(self):
        code = self.editor.toPlainText()
        try:
            compile(code, "<custom_code>", "exec")
            self.syntax_lbl.setText("✅  Syntax OK")
            self.syntax_lbl.setStyleSheet(f"color:{C['ok']};font-size:11px;")
        except SyntaxError as e:
            self.syntax_lbl.setText(
                f"❌  Syntax error  line {e.lineno}: {e.msg}")
            self.syntax_lbl.setStyleSheet(f"color:{C['err']};font-size:11px;")

    def _run_test(self):
        code = self.editor.toPlainText()
        sample, ok = QInputDialog.getMultiLineText(
            self, "Test Code",
            "Enter sample text to pass as 'text':",
            "Hello from the pipeline test runner.")
        if not ok:
            return
        log_lines = []
        ns = {
            "text": sample,
            "result": sample,
            "metadata": dict(self._block.metadata),
            "log": lambda m: log_lines.append(str(m)),
            "__builtins__": {"len": len, "str": str, "int": int, "float": float,
                             "bool": bool, "list": list, "dict": dict, "tuple": tuple,
                             "range": range, "enumerate": enumerate, "zip": zip,
                             "map": map, "filter": filter, "sorted": sorted,
                             "min": min, "max": max, "sum": sum,
                             "abs": abs, "round": round, "print": lambda *a: log_lines.append(" ".join(str(x) for x in a)),
                             "isinstance": isinstance, "hasattr": hasattr,
                             "getattr": getattr, "setattr": setattr,
                             "repr": repr, "type": type,
                             "True": True, "False": False, "None": None},
        }
        try:
            exec(compile(code, "<custom_code>", "exec"), ns)
            out = str(ns.get("result", sample))
            log_s = "\n".join(log_lines) if log_lines else "(no log output)"
            QMessageBox.information(
                self, "✅  Test Result",
                f"Log output:\n{log_s}\n\n"
                f"result  ({len(out)} chars):\n{out[:600]}"
                + ("…" if len(out) > 600 else ""))
        except Exception as e:
            QMessageBox.critical(self, "❌  Runtime Error", str(e))

    def _save_and_close(self):
        code = self.editor.toPlainText()
        try:
            compile(code, "<custom_code>", "exec")
        except SyntaxError as e:
            QMessageBox.warning(
                self, "Syntax Error",
                f"Fix the syntax error before saving:\n\nLine {e.lineno}: {e.msg}")
            return
        self._block.metadata["custom_code"] = code
        preview = code.strip().splitlines()[0] if code.strip() else "code"
        # Strip comment lines for label
        for ln in code.strip().splitlines():
            stripped = ln.strip()
            if stripped and not stripped.startswith("#"):
                preview = stripped[:20]
                break
        self._block.label = f"⌥ {preview}"
        self.accept()

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Manual — rendered inside _show_manual dialog
# ─────────────────────────────────────────────────────────────────────────────
def _make_manual_html() -> str:
    BG   = C.get("bg0","#1e1e2e"); TXT  = C.get("txt","#cdd6f4")
    TXT2 = C.get("txt2","#a6adc8"); ACC  = C.get("acc","#cba6f7")
    ACC2 = C.get("acc2","#a6e3a1"); WARN = C.get("warn","#f9e2af")
    BG2  = C.get("bg2","#313244"); ERR  = C.get("err","#f38ba8")
    OK   = C.get("ok","#a6e3a1");  PL   = C.get("pipeline","#89b4fa")

    def h1(t): return (f'<h1 style="color:{TXT};font-size:17px;font-weight:800;'
                       f'margin:0 0 4px;letter-spacing:0.5px;">{t}</h1>')
    def h2(t): return (f'<h2 style="color:{ACC};font-size:13px;font-weight:700;'
                       f'margin:22px 0 6px;border-bottom:1px solid {BG2};'
                       f'padding-bottom:5px;">{t}</h2>')
    def h3(t): return (f'<h3 style="color:{ACC2};font-size:12px;font-weight:700;'
                       f'margin:12px 0 3px;">{t}</h3>')
    def p(t):  return (f'<p style="color:{TXT2};font-size:11px;margin:3px 0 9px;'
                       f'line-height:1.7;">{t}</p>')
    def note(t): return (f'<p style="color:{WARN};font-size:10.5px;margin:4px 0 10px;'
                         f'background:{BG2};border-left:3px solid {WARN};'
                         f'padding:6px 10px;border-radius:0 5px 5px 0;">'
                         f'⚠️&nbsp; {t}</p>')
    def tip(t):  return (f'<p style="color:{OK};font-size:10.5px;margin:4px 0 10px;'
                         f'background:{BG2};border-left:3px solid {OK};'
                         f'padding:6px 10px;border-radius:0 5px 5px 0;">'
                         f'💡&nbsp; {t}</p>')
    def code(t): return (f'<code style="font-family:Consolas,monospace;font-size:10.5px;'
                         f'color:{ACC2};background:{BG2};padding:1px 5px;'
                         f'border-radius:3px;">{t}</code>')
    def badge(label, col): return (
        f'<span style="background:transparent;color:{col};border:1px solid {col};'
        f'border-radius:9px;padding:2px 8px;font-size:10px;font-weight:700;'
        f'white-space:nowrap;">{label}</span> ')
    def kbd(k): return (f'<span style="background:{BG2};color:{TXT};border:1px solid {ACC};'
                        f'border-radius:3px;padding:1px 6px;font-size:10px;'
                        f'font-family:Consolas,monospace;">{k}</span>')

    def port_table(*rows):
        hdr = (f'<tr><th style="color:{TXT};font-size:10px;font-weight:700;'
               f'padding:0 12px 5px 0;text-align:left;border-bottom:1px solid {BG2};">Port</th>'
               f'<th style="color:{TXT};font-size:10px;font-weight:700;'
               f'padding:0 12px 5px 0;text-align:left;border-bottom:1px solid {BG2};">Direction</th>'
               f'<th style="color:{TXT};font-size:10px;font-weight:700;'
               f'padding:0 0 5px;text-align:left;border-bottom:1px solid {BG2};">Notes</th></tr>')
        body = "".join(
            f'<tr><td style="color:{ACC2};font-size:11px;padding:4px 12px 4px 0;'
            f'font-family:Consolas,monospace;font-weight:700;">{r[0]}</td>'
            f'<td style="color:{OK if "in" in r[1].lower() else ERR};font-size:11px;'
            f'padding:4px 12px 4px 0;">{r[1]}</td>'
            f'<td style="color:{TXT2};font-size:11px;padding:4px 0;">{r[2]}</td></tr>'
            for r in rows)
        return (f'<table cellspacing="0" cellpadding="0" '
                f'style="width:100%;margin:8px 0 14px;">{hdr}{body}</table>')

    def example_box(title, body_html):
        return (f'<div style="background:{BG2};border-radius:6px;'
                f'padding:10px 14px;margin:6px 0 14px;">'
                f'<p style="color:{ACC};font-size:10px;font-weight:700;'
                f'margin:0 0 6px;letter-spacing:0.8px;">{title}</p>'
                f'{body_html}</div>')

    return f"""<html><body style="background:{BG};color:{TXT};
font-family:Inter,sans-serif;padding:20px 26px 30px;margin:0;line-height:1.5;">

{h1("🔗  NativeLab Pipeline Builder — Full Manual")}
{p(f'Version 2 &nbsp;·&nbsp; All pipeline data stored in {code("./localllm/pipelines/")} as JSON')}

<hr style="border:none;border-top:1px solid {BG2};margin:14px 0 6px;">

{h2("📌  Quick Start — 5 Steps to First Run")}
{p(f'1. <b>Add an ▶ Input block</b> — sidebar left, section <i>Flow Blocks</i>.<br>'
   f'2. <b>Add a ⚡ Model block</b> — drag a model from the sidebar list onto the canvas (ghost appears while hovering) or double-click it.<br>'
   f'3. <b>Add an ■ Output block</b> — same sidebar section.<br>'
   f'4. <b>Draw connections</b> — click a port dot on one block and drag to a port dot on another.<br>'
   f'5. Type text in <i>Input text</i> on the right panel and press <b>▶ Run Pipeline</b>.')}
{tip("Start simple: Input → Model → Output. Add logic blocks only after you confirm the basic run works.")}

{h2("🎨  Canvas Controls")}
{p(f'{kbd("Drag block")} Move any block freely — it snaps to the grid automatically.<br>'
   f'{kbd("Click port dot")} Start drawing a connection arrow.<br>'
   f'{kbd("Drag to port dot")} Complete a connection — creates a curved Bezier arrow.<br>'
   f'{kbd("Right-click block")} Context menu: Delete, Rename, Change Role, Configure.<br>'
   f'{kbd("Right-click canvas")} Clear all blocks and connections.<br>'
   f'{kbd("Pill bar")} The horizontal strip above the canvas shows all model/logic blocks as clickable pills — click one to select it on canvas.')}
{tip("The canvas is larger than the visible area. Use the scrollbars to pan around. Make big pipelines by spreading blocks far apart.")}

{h2("🔵  Port Dots  ( N · S · E · W )")}
{p("Every block has four port dots sitting on its four edges — North (top), South (bottom), East (right), West (left). "
   "Click any dot and drag to any dot on another block to create an arrow. "
   "The arrow curves automatically and adapts its control handles to the port direction.")}
{port_table(
    ("E  →", "Output (default)", "Text leaves the block here — connect to the next block's input"),
    ("W  ←", "Input (default)",  "Text arrives at the block here — connect from the previous block's output"),
    ("N  ↑", "Alt input/output", "Use for branch merges or alternate exits on logic blocks"),
    ("S  ↓", "Alt input/output", "Use for mid-score routing (LLM SCORE) or alternate splits"),
)}
{note("For normal flow blocks (Input, Model, Output, Intermediate) only one arrow per port is allowed. "
     "For all logic blocks and LLM logic blocks multiple arrows can fan out from the same port — this is how branching works.")}
{tip("You can draw arrows in any direction. Input → E is just a convention. The execution engine follows the arrows regardless of port direction.")}

{h2("🟦  Flow Blocks")}

{h3(badge("▶ INPUT", OK) + " Input Block")}
{p("The mandatory starting point. The text you type in the <i>Input text</i> box on the right panel is injected here as the initial context. "
   "You only ever need one. Every pipeline must have exactly one Input block.")}
{port_table(
    ("E  →", "Output", "Sends the raw input text to the first connected block"),
    ("S  ↓", "Output", "Alternative output — use when fanning out to multiple first blocks"),
)}
{example_box("EXAMPLE PIPELINE: Summarise + translate",
    p(f'▶ Input → ⚡ Summariser model → ◈ Intermediate ("Now translate the above to French") → ⚡ Translator model → ■ Output'))}

{h3(badge("◈ INTERMEDIATE", WARN) + " Intermediate Block")}
{p("A pure prompt-injection node — no model is called. It takes the arriving context and wraps your custom instruction around it before passing it on. "
   "Useful for steering the next model without breaking the data flow.")}
{p(f'<b>Right-click → Configure block…</b> to set:<br>'
   f'&nbsp;&nbsp;• <b>Prompt position</b>: above (prompt → model output) or below (model output → prompt)<br>'
   f'&nbsp;&nbsp;• <b>Prompt text</b>: any instruction, e.g. <i>"Now rewrite the above more concisely."</i>')}
{port_table(
    ("W  ←", "Input",  "Receives context from the previous block"),
    ("E  →", "Output", "Sends the wrapped context to the next block"),
)}
{tip("During execution each Intermediate block gets its own live tab in the right panel output area — you can watch the injected context build up in real time.")}
{note("Leaving the prompt blank makes the Intermediate block a transparent pass-through — context flows through unchanged. Useful as a visual separator.")}

{h3(badge("■ OUTPUT", ERR) + " Output Block")}
{p("The terminal node. Whatever context arrives here is shown in the <b>■ Output</b> tab on the right panel. "
   "The pipeline stops when it reaches an Output block. You can have multiple Output blocks — each one terminates its own branch independently.")}
{port_table(
    ("W  ←", "Input",  "Receives the final context — this becomes the displayed output"),
    ("N  ←", "Input",  "Alternative input port for pipelines where the final arrow comes from above"),
)}

{h2("⚡  Model Block")}
{p("Represents a loaded GGUF model. When the pipeline reaches a Model block, NativeLab starts the model in server mode (or reuses it if already running) and sends the current context as a prompt. The response becomes the new context.")}
{p(f'<b>How to add:</b><br>'
   f'&nbsp;&nbsp;• <b>Drag</b> a model from the sidebar list onto the canvas — a ghost block shows the drop position.<br>'
   f'&nbsp;&nbsp;• <b>Double-click</b> a model in the sidebar list to add it at a default position.')}
{p(f'<b>Right-click options:</b><br>'
   f'&nbsp;&nbsp;• <b>Change Role</b> — sets the system prompt used for this block:<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;{code("general")} You are a helpful assistant.<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;{code("reasoning")} Think step by step.<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;{code("summarization")} Be clear and concise.<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;{code("coding")} Write clean, well-commented code.<br>'
   f'&nbsp;&nbsp;• <b>Rename block</b> — give it a human-readable name.<br>'
   f'&nbsp;&nbsp;• <b>Delete</b> — removes the block and all its connections.')}
{port_table(
    ("W  ←", "Input",  "Receives the context to use as the prompt"),
    ("E  →", "Output", "Sends the model response to the next block"),
    ("N  ←", "Input",  "Alternate input — use when connecting from above"),
    ("S  →", "Output", "Alternate output — use when chaining downward"),
)}
{note("Direct Model → Model connections are blocked. You must place an ◈ Intermediate block between two models. "
     "This forces you to explicitly define what the second model should do with the first model's output.")}
{tip("Use different roles on different model blocks in the same pipeline. A 'reasoning' model can analyse, then an 'summarization' model can compress the result.")}

{h2("📎  Context Blocks")}
{p("Context blocks inject fixed text into the pipeline without calling a model. They read configuration you set at design time, not at runtime.")}

{h3(badge("📎 REFERENCE", ACC) + " Reference Block")}
{p("Pastes a fixed reference text <b>before</b> the incoming context. Good for injecting background documents, company info, templates, or examples a model should work with.")}
{p(f'<b>Right-click → Configure block…</b> then choose:<br>'
   f'&nbsp;&nbsp;• <b>Type / paste text</b> — opens a multiline input dialog.<br>'
   f'&nbsp;&nbsp;• <b>Load from file</b> — reads any .txt, .md, .py, .json, .yaml file. '
   f'Content is truncated to 4,000 characters with a [truncated] marker if longer.')}
{tip("Name your reference block descriptively (e.g. 'Company FAQ'). The injected text is wrapped in [REFERENCE: name] tags so the model can distinguish it from the user content.")}

{h3(badge("💡 KNOWLEDGE", ACC2) + " Knowledge Block")}
{p("Identical to Reference but labelled as a <i>knowledge base chunk</i> — the injected section is prefixed with <i>Knowledge Base:</i> instead of REFERENCE. "
   "Useful when you want to semantically distinguish instructional reference docs from factual knowledge.")}
{p(f'<b>Right-click → Configure block…</b> — opens a multiline text input. Truncated to 3,000 characters.')}

{h3(badge("📄 PDF SUMMARY", PL) + " PDF Summary Block")}
{p("Loads a PDF file, extracts text from all pages, and injects it. If the PDF exceeds 4,500 characters it is automatically chunk-summarised by the primary engine model before injection.")}
{p(f'<b>Right-click → Configure block…</b> to:<br>'
   f'&nbsp;&nbsp;1. Select the PDF file.<br>'
   f'&nbsp;&nbsp;2. Set the <b>role</b> of the PDF:<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;• <b>reference</b> — prior context is MAIN, PDF is supporting reference.<br>'
   f'&nbsp;&nbsp;&nbsp;&nbsp;• <b>main</b> — PDF is the MAIN content, prior context is supporting reference.')}
{note("PyPDF2 must be installed: pip install PyPDF2. If it is missing the block will show a warning when you try to configure it.")}

{h2("⑂  Logic Blocks  ( Python conditions )")}
{p(f'Logic blocks evaluate Python expressions or perform text operations at runtime. '
   f'The variable {code("text")} always holds the incoming context string. '
   f'These run instantly with no model call — use them for fast deterministic branching.')}

{h3(badge("⑂ IF / ELSE", "#f59e0b") + " IF / ELSE")}
{p("Evaluates a Python boolean expression against the incoming text and routes to one of two output arms.")}
{port_table(
    ("E  →", "Output — TRUE branch",  "Followed when the condition evaluates to True / YES"),
    ("W  →", "Output — FALSE branch", "Followed when the condition evaluates to False / NO"),
    ("W  ←", "Input",                 "Receives incoming context"),
)}
{p(f'<b>Configure:</b> Right-click → Configure block… and type a Python expression.<br>'
   f'<b>Draw two arrows</b> from this block, then label each one {code("TRUE")} and {code("FALSE")} when prompted.')}
{example_box("CONDITION EXAMPLES",
    p(f'{code("len(text) > 500")} — route long responses to a summariser<br>'
      f'{code("\'error\' in text.lower()")} — catch error messages<br>'
      f'{code("text.strip().startswith(\'```\'")} — detect code output<br>'
      f'{code("len(text.split()) < 20")} — detect very short answers<br>'
      f'{code("\'yes\' in text.lower()[:50]")} — check if model said yes'))}
{note("The expression has access to: len, str, int, float, bool, list, dict, any, all, min, max, abs, isinstance. No file or network access.")}

{h3(badge("⑃ SWITCH", "#f97316") + " SWITCH")}
{p("Evaluates a Python expression that returns a string key, then follows the outgoing arrow whose label matches that key.")}
{port_table(
    ("E/S/W  →", "Output arms", "One per case — label each arrow with the matching key string"),
    ("W  ←",     "Input",       "Receives incoming context"),
)}
{p(f'<b>Configure:</b> Right-click → Configure block… and type an expression that returns a string.<br>'
   f'When drawing each outgoing arrow you will be asked to type the <b>branch label</b> — this must exactly match what the expression can return (case-insensitive).<br>'
   f'Add a {code("default")} labelled arrow to catch unmatched keys.')}
{example_box("EXPRESSION EXAMPLES",
    p(f'{code("\'long\' if len(text) > 400 else \'short\'")} — length-based routing<br>'
      f'{code("\'code\' if text.strip().startswith(\'```\') else \'prose\'")} — format detection<br>'
      f'{code("text.split(\':\')[0].strip().lower()")} — route on first word / prefix<br>'
      f'{code("\'positive\' if text.count(\'!\') > 2 else \'neutral\'")} — punctuation heuristic'))}

{h3(badge("⊘ FILTER", "#84cc16") + " FILTER")}
{p("A gate. If the condition is TRUE the text continues through the pipeline unchanged. If FALSE the pipeline terminates immediately with a [FILTER DROPPED] message.")}
{port_table(
    ("E  →", "Output — PASS", "Followed only when condition is True"),
    ("W  ←", "Input",         "Receives incoming context"),
)}
{example_box("USE CASES",
    p('• Drop empty or whitespace-only responses before they reach the next model.<br>'
      '• Stop the pipeline if a safety keyword is detected.<br>'
      '• Gate on minimum length to avoid feeding junk to expensive models.'))}
{note("When FILTER drops text it emits a pipeline_done signal with the original text and a reason — this appears in the Output tab so you can debug why it was dropped.")}

{h3(badge("⟲ TRANSFORM", "#06b6d4") + " TRANSFORM")}
{p("Deterministic text operation — no model involved, instant execution. Modifies the context in a fixed, predictable way.")}
{p(f'<b>Available transforms:</b><br>'
   f'&nbsp;&nbsp;• {code("prefix")} — prepend fixed text before the context<br>'
   f'&nbsp;&nbsp;• {code("suffix")} — append fixed text after the context<br>'
   f'&nbsp;&nbsp;• {code("replace")} — find a substring and replace it<br>'
   f'&nbsp;&nbsp;• {code("upper")} / {code("lower")} — change case<br>'
   f'&nbsp;&nbsp;• {code("strip")} — remove leading/trailing whitespace and blank lines<br>'
   f'&nbsp;&nbsp;• {code("truncate")} — cut text to a maximum number of characters')}
{tip("Chain multiple TRANSFORM blocks together to build a text preprocessing pipeline before it hits a model — e.g. strip → truncate → prefix.")}

{h3(badge("⊕ MERGE", "#8b5cf6") + " MERGE")}
{p("Waits for all incoming arrows to deliver their context, then joins them all into a single string that flows to the next block. "
   "Essential after a SPLIT or any fan-out pattern.")}
{p(f'<b>Merge modes:</b><br>'
   f'&nbsp;&nbsp;• {code("concat")} — join with a separator string (default: two newlines + ---)<br>'
   f'&nbsp;&nbsp;• {code("prepend")} — newest first, oldest last<br>'
   f'&nbsp;&nbsp;• {code("append")} — oldest first, newest last<br>'
   f'&nbsp;&nbsp;• {code("json")} — wrap all inputs as a JSON array string')}
{note("MERGE collects all contexts queued for its block ID in the current execution pass. If only one arrow arrives the block still works — it just returns that single input.")}

{h3(badge("⑁ SPLIT", "#ec4899") + " SPLIT")}
{p("Broadcasts the exact same text to every single outgoing arrow simultaneously. Useful for running the same context through multiple models in parallel (they execute sequentially internally).")}
{port_table(
    ("E/S/W  →", "Output × N", "All outgoing arrows fire with identical text"),
    ("W  ←",     "Input",      "Receives one context, fans it to all outputs"),
)}
{tip("SPLIT + MERGE is the classic parallel-processing pattern: split into N model branches, each model processes the same input, MERGE collects all responses.")}
{example_box("PARALLEL REVIEW PATTERN",
    p('▶ Input → ⑁ SPLIT → ⚡ Reviewer A → ⊕ MERGE → ◈ Intermediate ("Combine the two reviews") → ⚡ Model → ■ Output<br>'
      '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↘ ⚡ Reviewer B ↗'))}

{h3(badge("⌥ CUSTOM CODE", "#10b981") + " Custom Code")}
{p("Write arbitrary Python that runs inline during pipeline execution. Full code editor with live syntax checking and a test runner.")}
{p(f'<b>Available variables:</b><br>'
   f'&nbsp;&nbsp;• {code("text")} — incoming context string (read-only)<br>'
   f'&nbsp;&nbsp;• {code("result")} — set this to your output string (defaults to {code("text")} if not set)<br>'
   f'&nbsp;&nbsp;• {code("metadata")} — block metadata dict, persists across pipeline runs<br>'
   f'&nbsp;&nbsp;• {code("log(msg)")} — writes a message to the Pipeline Log tab')}
{p(f'<b>Safe builtins available:</b> {code("len str int float bool list dict tuple range enumerate zip map filter sorted min max sum abs round isinstance hasattr getattr repr type print")}')}
{note("No file I/O, no network access, no os/subprocess. The exec() is sandboxed. If your code raises an exception the pipeline stops with the error message shown in the Log tab.")}
{example_box("CODE EXAMPLES",
    p(f'{code("result = text.upper()")} — uppercase<br>'
      f'{code("result = \'\\n\'.join(sorted(text.split(\'\\n\')))")} — sort lines<br>'
      f'{code("result = str(len(text.split())) + \' words: \' + text")} — prepend word count<br>'
      f'{code("import_count = metadata.get(\'runs\', 0) + 1; metadata[\'runs\'] = import_count; log(f\'Run #{import_count}\')")} — stateful counter'))}

{h2("🧠  LLM Logic Blocks  ( plain English conditions )")}
{p("LLM logic blocks work identically to their Python counterparts except the condition or instruction is written in <b>plain English</b> and the attached model evaluates it. "
   "The model is called with a tight system prompt that demands a specific answer format (YES/NO, a category name, PASS/STOP, etc.).")}
{p(f'<b>Each LLM logic block requires:</b><br>'
   f'&nbsp;&nbsp;1. A <b>GGUF model</b> selected in the config dialog (can be any registered model).<br>'
   f'&nbsp;&nbsp;2. A <b>plain English instruction</b> written by you.')}
{p(f'<b>Advanced settings</b> (expand in the config dialog):<br>'
   f'&nbsp;&nbsp;• <b>Max response tokens</b> — keep small (16–64) for routing, larger (128–512) for transforms.<br>'
   f'&nbsp;&nbsp;• <b>Temperature</b> — use 0–15 for deterministic routing, 30–60 for creative transforms.<br>'
   f'&nbsp;&nbsp;• <b>Show model reasoning in log</b> — logs the raw model response before it is parsed.<br>'
   f'&nbsp;&nbsp;• <b>Passthrough on error</b> — if the model call fails, pass text through unchanged instead of stopping.')}
{tip("Use a small fast model (e.g. Qwen2.5-0.5B or TinyLlama) for LLM routing blocks. They only need to say YES/NO — you do not need a 7B model for that.")}

{h3(badge("🧠 LLM IF / ELSE", "#a855f7") + " LLM IF / ELSE")}
{p("The model reads your condition and the incoming text, then answers with a single word: YES or NO.")}
{port_table(
    ("E  →", "Output — TRUE / YES",  "Followed when model answers YES"),
    ("W  →", "Output — FALSE / NO",  "Followed when model answers NO"),
)}
{example_box("CONDITION EXAMPLES",
    p('• Does this text contain a complaint or expression of frustration?<br>'
      '• Is the answer longer than a brief paragraph?<br>'
      '• Does the user seem confused or are they asking a follow-up question?<br>'
      '• Is this response in English?<br>'
      '• Does this text contain any personally identifiable information?'))}
{note("The parser accepts: YES Y TRUE 1 PASS POSITIVE as truthy. Everything else is FALSE. Enable 'Show model reasoning' to debug unexpected routing.")}

{h3(badge("🧠 LLM SWITCH", "#7c3aed") + " LLM SWITCH")}
{p("The model classifies the text into one of the categories you define. The category names are automatically read from the labels you put on the outgoing arrows.")}
{port_table(
    ("E/S/W  →", "Output arms", "One per category — label each arrow with the exact category name"),
)}
{example_box("INSTRUCTION EXAMPLES",
    p('• Classify this text as one of: positive, negative, neutral<br>'
      '• What language is this written in? english, french, spanish, german, other<br>'
      '• Is this a question, a complaint, a compliment, or a general statement?<br>'
      '• Route by topic: technical, billing, account, other<br>'
      '• What kind of content is this: code, prose, data, mixed?'))}
{tip("Always include a 'default' or 'other' labelled arrow to catch cases where the model returns something unexpected. The fallback prevents silent drops.")}

{h3(badge("🧠 LLM FILTER", "#6366f1") + " LLM FILTER")}
{p("The model decides whether the text should continue (PASS) or be dropped (STOP). When dropped the pipeline ends with a structured message explaining which filter stopped it and the model's reason.")}
{example_box("CONDITION EXAMPLES",
    p('• Only pass if this is a genuine technical support question<br>'
      '• Pass only if the response contains a concrete action item<br>'
      '• Allow through only if the content is safe and not harmful<br>'
      '• Pass only if the answer is at least two sentences long<br>'
      '• Block any response that mentions competitor products by name'))}
{note("The FILTER stop message is displayed in the Output tab with the filter name, condition, model decision, and the original text — so you can inspect exactly what was dropped and why.")}

{h3(badge("🧠 LLM TRANSFORM", "#0ea5e9") + " LLM TRANSFORM")}
{p("The model rewrites, reformats, or transforms the incoming text according to your instruction. The result replaces the context for all downstream blocks. "
   "Increase Max response tokens to 256–512 for this block type.")}
{example_box("INSTRUCTION EXAMPLES",
    p('• Summarise this in exactly three bullet points, each one sentence<br>'
      '• Rewrite in a formal professional business tone<br>'
      '• Extract only the action items as a numbered list<br>'
      '• Translate to Spanish keeping all technical terms in English<br>'
      '• Convert this prose into a structured JSON object with keys: title, summary, keywords<br>'
      '• Remove all filler words and redundant phrases, keep only the core meaning'))}
{tip("The transform block automatically strips common preamble phrases the model might add (Here is..., Result:, Output:) before passing the result downstream.")}

{h3(badge("🧠 LLM SCORE", "#d946ef") + " LLM SCORE")}
{p("The model rates the incoming text on your criterion from 1 to 10. The score is parsed from the response and used to route to one of three band arms.")}
{port_table(
    ("E  →", "Output — LOW (1–3)",   "Route to escalation, retry, or human review"),
    ("S  →", "Output — MID (4–7)",   "Route to standard processing"),
    ("W  →", "Output — HIGH (8–10)", "Route to fast-track or direct output"),
)}
{p(f'Label an outgoing arrow {code("score")} to receive the raw numeric score as text instead of the original context — useful for feeding the score into a TRANSFORM or OUTPUT.')}
{example_box("CRITERION EXAMPLES",
    p('• Rate the clarity and readability of this explanation (1=very unclear, 10=crystal clear)<br>'
      '• Score the sentiment positivity (1=very negative, 10=very positive)<br>'
      '• Rate the technical complexity (1=trivial, 10=expert-level)<br>'
      '• How complete and thorough is this answer? (1=missing key info, 10=comprehensive)<br>'
      '• Score the urgency of this message (1=low priority, 10=critical/immediate action needed)'))}

{h2("🔄  Loops")}
{p("Draw an arrow <b>backwards</b> — from a downstream block to an upstream block. NativeLab detects the cycle and asks how many times the loop should iterate (min 2, max 999). "
   "Loop arrows are shown as <b>dashed dash-dot lines</b> with a ×N badge at the midpoint.")}
{p(f'<b>How execution works:</b><br>'
   f'Each time the pipeline reaches the source block of a loop edge it checks how many times that specific edge has already been followed. '
   f'Once the limit is reached the edge is skipped and execution continues to the next non-loop outgoing connection.')}
{example_box("REFINEMENT LOOP PATTERN",
    p('▶ Input → ⚡ Model → ◈ Intermediate ("Critique the above and list improvements") → ⚡ Model<br>'
      '(draw a backwards arrow from the second Model back to the first Intermediate, set ×3)<br>'
      '→ ■ Output'))}
{tip("Connect a non-loop arrow from the loop body to an Output block to capture the final result after all iterations complete.")}

{h2("💾  Save & Load Pipelines")}
{p(f'Click <b>💾 Save Pipeline…</b> in the sidebar. Type a name — existing names are overwritten without warning.<br>'
   f'Click <b>📂 Load Pipeline…</b> to restore a saved pipeline. If the canvas has blocks you will be asked to confirm replacement.<br>'
   f'Pipelines are stored as JSON in {code("./localllm/pipelines/name.json")}.<br>'
   f'To delete: Load dialog → select <i>🗑 Delete a pipeline…</i> option.')}
{p(f'<b>What is saved per block:</b> type, position, size, model path, role, label, all metadata (prompt text, code, conditions, PDF path, etc.).<br>'
   f'<b>What is saved per connection:</b> from/to block IDs, ports, is_loop flag, loop_times, branch label.')}
{note("Model files are saved as absolute paths. If you move a .gguf file the pipeline will load but the model block will show 'no valid file' and validation will fail until you re-attach the model.")}

{h2("🐛  Debugging & Troubleshooting")}
{p(f'<b>📋 Log tab</b> — shows every step: block started, chars processed, decisions made, errors. Always check this first.<br>'
   f'<b>◈ Intermediate tabs</b> — each intermediate block gets a live tab showing exactly what text arrived at it.<br>'
   f'<b>■ Output tab</b> — shows the final rendered output with markdown, code highlighting, and bold.')}
{p(f'<b>Common errors:</b>')}
{p(f'• <i>No INPUT block</i> — add ▶ Input and connect it to something.<br>'
   f'• <i>No OUTPUT block</i> — add ■ Output and connect the last block to it.<br>'
   f'• <i>No connections drawn</i> — you must draw at least one arrow between blocks.<br>'
   f'• <i>Model block has no valid file</i> — double-click a model in the sidebar or check the file still exists.<br>'
   f'• <i>Reference / Knowledge has no text</i> — right-click → Configure before running.<br>'
   f'• <i>IF/ELSE has no condition</i> — right-click → Configure and type a Python expression.<br>'
   f'• <i>LLM logic block: no model</i> — open config dialog and select a GGUF model.<br>'
   f'• <i>LLM logic block: model not found</i> — the model file was moved; reconfigure the block.<br>'
   f'• <i>Engine Not Ready</i> — wait for the primary model to finish loading in the Server tab.<br>'
   f'• <i>Server HTTP 500</i> — the model is loaded but errored; check the Logs tab in the Server tab for details.<br>'
   f'• <i>FILTER DROPPED</i> in output — the filter condition was FALSE; check the condition logic or increase tolerance.')}
{tip("Press ⏹ Stop Execution at any time. The pipeline will finish its current HTTP request and then halt cleanly.")}

{h2("⚡  Performance Tips")}
{p(f'• Keep context short between models — truncate with a TRANSFORM block before expensive model calls.<br>'
   f'• Use small models (0.5B–1.5B) for all LLM logic routing blocks. Reserve big models for the actual generation steps.<br>'
   f'• PDF blocks auto-summarise large documents — but this adds extra model calls. Pre-summarise large PDFs offline if speed matters.<br>'
   f'• Loop iterations multiply model calls: 3 models × 5 loops = 15 server requests. Plan accordingly.<br>'
   f'• Set Max response tokens conservatively on routing blocks (16 is enough for YES/NO decisions).')}

</body></html>"""

_PIPELINE_MANUAL_HTML = _make_manual_html()

class PipelineBuilderTab(QWidget):
    """
    Full-featured pipeline builder tab.
    Left sidebar : block adders (flow + context) + model list.
    Centre       : interactive canvas.
    Right        : execution controls + per-intermediate live panes + final output.
    """

    def __init__(self, engine: "LlamaEngine", parent=None):
        super().__init__(parent)
        self._engine                = engine
        self._exec_worker: Optional[PipelineExecutionWorker] = None
        self._inter_tabs:  Dict[int, int] = {}   # block_id → tab index
        self._current_pipeline_name: str = ""
        self._build()

    def update_engine(self, engine: "LlamaEngine"):
        self._engine = engine

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ══════ LEFT SIDEBAR ══════════════════════════════════════════════════
        sb = QWidget()
        sb.setObjectName("session_sidebar")
        sb_l = QVBoxLayout(sb)
        sb_l.setContentsMargins(10, 14, 10, 14)
        sb_l.setSpacing(7)

        def _sec(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color:{C['txt3']};font-size:9px;"
                f"font-weight:700;letter-spacing:1.1px;")
            return lbl

        def _block_btn(icon_label: str, btype: str, color: str) -> QPushButton:
            btn = QPushButton(icon_label)
            btn.setFixedHeight(30)
            btn.setStyleSheet(
                f"QPushButton{{background:transparent;"
                f"color:{color};border:1px solid {color};"
                f"border-radius:6px;font-size:11px;font-weight:600;}}"
                f"QPushButton:hover{{background:{color};color:#fff;}}")
            btn.clicked.connect(lambda _, bt=btype: self._add_block(bt))
            return btn

        hdr = QLabel("🔗  Pipeline Builder")
        hdr.setStyleSheet(
            f"color:{C['txt']};font-size:12px;font-weight:700;")
        sb_l.addWidget(hdr)

        sep0 = QFrame(); sep0.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep0)

        sb_l.addWidget(_sec("FLOW BLOCKS"))
        sb_l.addWidget(_block_btn("▶  Input Block",          PipelineBlockType.INPUT,        C["ok"]))
        sb_l.addWidget(_block_btn("◈  Intermediate Block",   PipelineBlockType.INTERMEDIATE, C["warn"]))
        sb_l.addWidget(_block_btn("■  Output Block",         PipelineBlockType.OUTPUT,       C["err"]))

        sep_ctx = QFrame(); sep_ctx.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_ctx)
        sb_l.addWidget(_sec("CONTEXT BLOCKS"))
        sb_l.addWidget(_block_btn("📎  Reference",   PipelineBlockType.REFERENCE,   C["acc"]))
        sb_l.addWidget(_block_btn("💡  Knowledge",   PipelineBlockType.KNOWLEDGE,   C["acc2"]))
        sb_l.addWidget(_block_btn("📄  PDF Summary", PipelineBlockType.PDF_SUMMARY, C["pipeline"]))

        sep_logic = QFrame(); sep_logic.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_logic)
        sb_l.addWidget(_sec("LOGIC BLOCKS"))
        sb_l.addWidget(_block_btn("⑂  IF / ELSE",   PipelineBlockType.IF_ELSE,     "#f59e0b"))
        sb_l.addWidget(_block_btn("⑃  SWITCH",      PipelineBlockType.SWITCH,      "#f97316"))
        sb_l.addWidget(_block_btn("⊘  FILTER",      PipelineBlockType.FILTER,      "#84cc16"))
        sb_l.addWidget(_block_btn("⟲  TRANSFORM",   PipelineBlockType.TRANSFORM,   "#06b6d4"))
        sb_l.addWidget(_block_btn("⊕  MERGE",       PipelineBlockType.MERGE,       "#8b5cf6"))
        sb_l.addWidget(_block_btn("⑁  SPLIT",       PipelineBlockType.SPLIT,       "#ec4899"))

        sep_code = QFrame(); sep_code.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_code)
        sb_l.addWidget(_sec("CUSTOM CODE"))
        sb_l.addWidget(_block_btn("⌥  Custom Code", PipelineBlockType.CUSTOM_CODE, "#10b981"))

        sep_llm = QFrame(); sep_llm.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep_llm)
        sb_l.addWidget(_sec("LLM LOGIC  (natural language)"))
        _llm_note = QLabel(
            "Conditions & instructions written\n"
            "in plain English — evaluated by\n"
            "the block's attached LLM model.")
        _llm_note.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;"
            f"padding:4px 2px;line-height:1.5;")
        sb_l.addWidget(_llm_note)
        sb_l.addWidget(_block_btn("🧠  LLM IF / ELSE",   PipelineBlockType.LLM_IF,        "#a855f7"))
        sb_l.addWidget(_block_btn("🧠  LLM SWITCH",      PipelineBlockType.LLM_SWITCH,     "#7c3aed"))
        sb_l.addWidget(_block_btn("🧠  LLM FILTER",      PipelineBlockType.LLM_FILTER,     "#6366f1"))
        sb_l.addWidget(_block_btn("🧠  LLM TRANSFORM",   PipelineBlockType.LLM_TRANSFORM,  "#0ea5e9"))
        sb_l.addWidget(_block_btn("🧠  LLM SCORE",       PipelineBlockType.LLM_SCORE,      "#d946ef"))

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep1)

        sb_l.addWidget(_sec("MODELS  (dbl-click to add)"))
        self.model_list = QListWidget()
        self.model_list.setObjectName("model_list")
        self.model_list.itemDoubleClicked.connect(self._add_model_from_list)
        self.model_list.setDragEnabled(True)
        self.model_list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.model_list.setToolTip(
            "Double-click OR drag onto the canvas to add a model block.")
        sb_l.addWidget(self.model_list, 1)

        self.btn_refresh = QPushButton("↻  Refresh")
        self.btn_refresh.setFixedHeight(26)
        self.btn_refresh.clicked.connect(self._refresh_models)
        sb_l.addWidget(self.btn_refresh)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sb_l.addWidget(sep2)

        sb_l.addWidget(_sec("CANVAS CONTROLS"))
        self.btn_clear_canvas = QPushButton("🗑  Clear All")
        self.btn_clear_canvas.setFixedHeight(28)
        self.btn_clear_canvas.clicked.connect(self._clear_canvas)
        sb_l.addWidget(self.btn_clear_canvas)

        self.btn_save_pipeline = QPushButton("💾  Save Pipeline…")
        self.btn_save_pipeline.setFixedHeight(28)
        self.btn_save_pipeline.clicked.connect(self._save_pipeline)
        sb_l.addWidget(self.btn_save_pipeline)

        self.btn_load_pipeline = QPushButton("📂  Load Pipeline…")
        self.btn_load_pipeline.setFixedHeight(28)
        self.btn_load_pipeline.clicked.connect(self._load_pipeline)
        sb_l.addWidget(self.btn_load_pipeline)

        hint = QLabel(
            "Tip: draw a connection from a\n"
            "port dot (N·S·E·W) on one block\n"
            "to a port dot on another block.\n\n"
            "Model→Model requires an\n"
            "Intermediate block in between.")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;"
            f"padding:6px 4px;line-height:1.6;")
        sb_l.addWidget(hint)

        sb_l.addStretch()

        sb_scroll = QScrollArea()
        sb_scroll.setWidget(sb)
        sb_scroll.setWidgetResizable(True)
        sb_scroll.setFixedWidth(214)
        sb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sb_scroll.setObjectName("session_sidebar")
        sb_scroll.setStyleSheet("QScrollArea#session_sidebar { border: none; }")
        root.addWidget(sb_scroll)

        # ══════ CENTRE: CANVAS ════════════════════════════════════════════════
        centre = QWidget()
        centre_l = QVBoxLayout(centre)
        centre_l.setContentsMargins(0, 0, 0, 0)
        centre_l.setSpacing(0)

        # toolbar
        toolbar = QWidget()
        toolbar.setObjectName("appearance_bar")
        toolbar.setFixedHeight(40)
        tb_l = QHBoxLayout(toolbar)
        tb_l.setContentsMargins(12, 4, 12, 4)
        tb_l.setSpacing(8)
        tb_title = QLabel("🔗  Pipeline Canvas")
        tb_title.setObjectName("appearance_hdr")
        tb_l.addWidget(tb_title)
        tb_l.addStretch()
        legend_items = [
            (C["ok"],       "▶ Input"),
            (C["warn"],     "◈ Intermediate"),
            (C["err"],      "■ Output"),
            (C["acc"],      "⚡ Model"),
            ("#f59e0b",     "⑂ IF/ELSE"),
            ("#06b6d4",     "⟲ Transform"),
            ("#10b981",     "⌥ Code"),
            ("#8b5cf6",     "⊕ Merge"),
            ("#ec4899",     "⑁ Split"),
            ("#a855f7",     "🧠 LLM-IF"),
            ("#7c3aed",     "🧠 LLM-SW"),
            ("#0ea5e9",     "🧠 LLM-TX"),
            ("#d946ef",     "🧠 LLM-SC"),
            (C["pipeline"], "⤵ Loop"),
            (C["acc2"],     "→ Forward"),
        ]
        for col, txt in legend_items:
            lbl = QLabel(txt)
            lbl.setStyleSheet(
                f"color:{col};font-size:10px;font-weight:600;"
                f"padding:1px 6px;"
                f"background:{C['bg2']};border-radius:4px;")
            tb_l.addWidget(lbl)
        centre_l.addWidget(toolbar)

        pill_outer = QWidget()
        pill_outer.setObjectName("appearance_bar")
        pill_outer.setFixedHeight(36)
        pill_outer_l = QHBoxLayout(pill_outer)
        pill_outer_l.setContentsMargins(8, 3, 8, 3)
        pill_outer_l.setSpacing(0)

        _pill_icon = QLabel("⚡")
        _pill_icon.setStyleSheet(f"color:{C['txt3']};font-size:10px;padding-right:4px;")
        pill_outer_l.addWidget(_pill_icon)

        self._pill_scroll = QScrollArea()
        self._pill_scroll.setWidgetResizable(True)
        self._pill_scroll.setFixedHeight(30)
        self._pill_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._pill_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._pill_scroll.setStyleSheet(
            "QScrollArea { border:none; background:transparent; }"
            "QScrollBar:horizontal { height:4px; }")

        self._pill_container = QWidget()
        self._pill_container.setStyleSheet("background:transparent;")
        self._pill_layout = QHBoxLayout(self._pill_container)
        self._pill_layout.setContentsMargins(0, 0, 0, 0)
        self._pill_layout.setSpacing(5)
        self._pill_layout.addStretch()
        self._pill_scroll.setWidget(self._pill_container)
        pill_outer_l.addWidget(self._pill_scroll, 1)
        centre_l.addWidget(pill_outer)

        canvas_scroll = QScrollArea()
        canvas_scroll.setWidgetResizable(False)
        canvas_scroll.setObjectName("chat_scroll")
        self.canvas = PipelineCanvas()
        self.canvas.blocks_changed.connect(self._on_blocks_changed)
        canvas_scroll.setWidget(self.canvas)
        centre_l.addWidget(canvas_scroll, 1)

        root.addWidget(centre, 1)

        # ══════ RIGHT: EXECUTION PANEL ════════════════════════════════════════
        rp = QWidget()
        rp.setFixedWidth(282)
        rp.setObjectName("ref_panel")
        rp_l = QVBoxLayout(rp)
        rp_l.setContentsMargins(12, 14, 12, 14)
        rp_l.setSpacing(7)

        exec_hdr_row = QHBoxLayout(); exec_hdr_row.setSpacing(6)
        exec_hdr = QLabel("▶  Execute Pipeline")
        exec_hdr.setStyleSheet(
            f"color:{C['txt']};font-size:12px;font-weight:700;")
        exec_hdr_row.addWidget(exec_hdr, 1)
        btn_manual = QPushButton("📖 Manual")
        btn_manual.setFixedHeight(24)
        btn_manual.setStyleSheet(
            f"QPushButton{{background:transparent;color:{C['acc']};"
            f"border:1px solid {C['acc']};border-radius:5px;"
            f"font-size:10px;font-weight:600;padding:0 8px;}}"
            f"QPushButton:hover{{background:{C['acc']};color:#fff;}}")
        btn_manual.clicked.connect(self._show_manual)
        exec_hdr_row.addWidget(btn_manual)
        rp_l.addLayout(exec_hdr_row)

        # Server status badge (auto-refreshed every 2.5 s)
        self.server_badge = QLabel("⚪  Engine status unknown")
        self.server_badge.setStyleSheet(
            f"color:{C['txt3']};font-size:10px;padding:2px 7px;"
            f"background:{C['bg2']};border-radius:4px;")
        rp_l.addWidget(self.server_badge)
        self._badge_timer = QTimer(self)
        self._badge_timer.timeout.connect(self._update_server_badge)
        self._badge_timer.start(2500)

        input_lbl = QLabel("Input text:")
        input_lbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        rp_l.addWidget(input_lbl)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText(
            "Enter the prompt or text to feed\n"
            "into the first INPUT block…")
        self.input_edit.setMaximumHeight(100)
        rp_l.addWidget(self.input_edit)

        self.btn_run = QPushButton("▶  Run Pipeline")
        self.btn_run.setObjectName("btn_send")
        self.btn_run.setFixedHeight(34)
        self.btn_run.clicked.connect(self._run_pipeline)
        rp_l.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⏹  Stop Execution")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(34)
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self._stop_pipeline)
        rp_l.addWidget(self.btn_stop)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.HLine)
        rp_l.addWidget(sep3)

        # Tabbed output: 📋 Log | ■ Output | one ◈ tab per intermediate block
        self.output_tabs = QTabWidget()
        self.output_tabs.setObjectName("ref_tabs")

        self.exec_log = QTextEdit()
        self.exec_log.setReadOnly(True)
        self.exec_log.setFont(QFont("Consolas", 9))
        self.exec_log.setObjectName("log_te")
        self.output_tabs.addTab(self.exec_log, "📋 Log")

        self.output_edit = PipelineOutputRenderer(
            placeholder="Final pipeline output appears here…")
        self.output_tabs.addTab(self.output_edit, "■ Output")
        rp_l.addWidget(self.output_tabs, 1)

        copy_btn = QPushButton("⧉  Copy Final Output")
        copy_btn.setFixedHeight(28)
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(
                self.output_edit.raw_text()))
        rp_l.addWidget(copy_btn)

        root.addWidget(rp)

        self._refresh_models()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _refresh_models(self):
        """Show the pipeline builder manual in a scrollable dialog."""
        dlg = QDialog(self)
        dlg.setWindowTitle("📖  Pipeline Builder Manual")
        dlg.setMinimumSize(680, 580)
        dlg.resize(740, 640)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Inter", 11))
        te.setObjectName("chat_te")
        te.setHtml(_PIPELINE_MANUAL_HTML)
        lay.addWidget(te, 1)

        btn_close = QPushButton("✕  Close")
        btn_close.setFixedHeight(32)
        btn_close.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 12)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)
        dlg.exec()

    def _show_manual(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("📖  Pipeline Builder Manual")
        dlg.setMinimumSize(680, 580)
        dlg.resize(740, 640)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Inter", 11))
        te.setObjectName("chat_te")
        te.setHtml(_make_manual_html())
        lay.addWidget(te, 1)
        btn_close = QPushButton("✕  Close")
        btn_close.setFixedHeight(32)
        btn_close.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 12)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)
        dlg.exec()

    def _refresh_models(self):
        self.model_list.clear()
        for m in get_model_registry().all_models():
            ri   = ROLE_ICONS.get(m.get("role", "general"), "💬")
            qt   = m.get("quant", "?")
            fam  = m.get("family", "?")
            item = QListWidgetItem(f"{ri}  {m['name']}\n    {fam} · {qt}")
            item.setData(Qt.ItemDataRole.UserRole,     m["path"])
            item.setData(Qt.ItemDataRole.UserRole + 1, m.get("role", "general"))
            self.model_list.addItem(item)

    # ── pipeline save / load ──────────────────────────────────────────────────

    def _save_pipeline(self):
        if not self.canvas.blocks:
            QMessageBox.warning(self, "Empty Pipeline",
                                "Add blocks before saving."); return
        saved = list_saved_pipelines()
        name, ok = QInputDialog.getText(
            self, "Save Pipeline",
            "Pipeline name:\n(Existing names will be overwritten)",
            text=self._current_pipeline_name or "")
        if not ok or not name.strip():
            return
        name = name.strip().replace("/", "-").replace("\\", "-")
        save_pipeline(name, self.canvas.blocks, self.canvas.connections)
        self._current_pipeline_name = name
        self._log(f"💾  Pipeline saved as '{name}'")
        QMessageBox.information(self, "Saved",
                                f"Pipeline '{name}' saved successfully.")

    def _load_pipeline(self):
        saved = list_saved_pipelines()
        if not saved:
            QMessageBox.information(self, "No Pipelines",
                                    "No saved pipelines found.\n"
                                    f"Pipelines are stored in:\n{PIPELINES_DIR}")
            return
        # Offer saved names + a delete option
        items = saved + ["─────────────", "🗑  Delete a pipeline…"]
        choice, ok = QInputDialog.getItem(
            self, "Load Pipeline", "Select a pipeline:", items, 0, False)
        if not ok:
            return
        if choice == "🗑  Delete a pipeline…":
            self._delete_pipeline_dialog(saved); return
        if choice.startswith("─"):
            return
        if self.canvas.blocks:
            ans = QMessageBox.question(
                self, "Replace Canvas",
                f"Load '{choice}' and replace the current canvas?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return
        try:
            blocks, conns = load_pipeline(choice)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return
        self.canvas.clear_all()
        while self.output_tabs.count() > 2:
            self.output_tabs.removeTab(2)
        self._inter_tabs.clear()
        for b in blocks:
            self.canvas.blocks.append(b)
        self.canvas.connections = conns
        self.canvas.update()
        self.canvas.blocks_changed.emit()
        self._current_pipeline_name = choice
        self._log(f"📂  Pipeline '{choice}' loaded "
                  f"({len(blocks)} blocks, {len(conns)} connections)")

    def _delete_pipeline_dialog(self, saved: list):
        choice, ok = QInputDialog.getItem(
            self, "Delete Pipeline", "Pipeline to delete:", saved, 0, False)
        if not ok:
            return
        ans = QMessageBox.question(
            self, "Confirm Delete",
            f"Permanently delete '{choice}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Yes:
            (PIPELINES_DIR / f"{choice}.json").unlink(missing_ok=True)
            self._log(f"🗑  Pipeline '{choice}' deleted.")

    def _update_server_badge(self):
        eng = self._engine
        if not eng or not eng.is_loaded:
            self.server_badge.setText("⚪  No model loaded")
            self.server_badge.setStyleSheet(
                f"color:{C['txt3']};font-size:10px;padding:2px 7px;"
                f"background:{C['bg2']};border-radius:4px;")
        elif eng.mode == "server":
            self.server_badge.setText(f"🟢  Server · port {eng.server_port}")
            self.server_badge.setStyleSheet(
                f"color:{C['ok']};font-size:10px;padding:2px 7px;"
                f"background:{C['bg2']};border-radius:4px;")
        else:
            self.server_badge.setText("🟡  CLI mode — will switch on run")
            self.server_badge.setStyleSheet(
                f"color:{C['warn']};font-size:10px;padding:2px 7px;"
                f"background:{C['bg2']};border-radius:4px;")

    def _add_block(self, btype: str):
        count = sum(1 for b in self.canvas.blocks if b.btype == btype)
        x = 100 + count * 30
        y = 160 + (len(self.canvas.blocks) % 4) * 100
        b = self.canvas.add_block(btype, x=x, y=y)
        # Immediately prompt configuration for context blocks
        if btype in (PipelineBlockType.REFERENCE,
                     PipelineBlockType.KNOWLEDGE,
                     PipelineBlockType.PDF_SUMMARY):
            self.canvas._configure_context_block(b)
        # Immediately prompt configuration for logic blocks
        _LOGIC_BTYPES = {
            PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
            PipelineBlockType.FILTER,  PipelineBlockType.TRANSFORM,
            PipelineBlockType.MERGE,   PipelineBlockType.CUSTOM_CODE,
        }
        if btype in _LOGIC_BTYPES:
            self.canvas._configure_logic_block(b)
        # Immediately open LLM logic editor
        _LLM_BTYPES = {
            PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
            PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
            PipelineBlockType.LLM_SCORE,
        }
        if btype in _LLM_BTYPES:
            self.canvas._configure_llm_logic_block(b)

    def _add_model_from_list(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        role = item.data(Qt.ItemDataRole.UserRole + 1) or "general"
        if not path:
            return
        model_blocks = [b for b in self.canvas.blocks
                        if b.btype == PipelineBlockType.MODEL]
        x = 220 + len(model_blocks) * 180
        y = 260
        b = self.canvas.add_block(PipelineBlockType.MODEL, x=x, y=y,
                                   model_path=path, role=role)
        self._log(f"Added model: {b.label}  [{role}]")

    def _clear_canvas(self):
        if QMessageBox.question(
            self, "Clear Canvas",
            "Remove all blocks and connections from the pipeline?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            self.canvas.clear_all()
            # Remove intermediate pane tabs (keep Log [0] and Output [1])
            while self.output_tabs.count() > 2:
                self.output_tabs.removeTab(2)
            self._inter_tabs.clear()
            self.exec_log.clear()
            self.output_edit.clear()

    def _on_blocks_changed(self):
        """Sync intermediate live-output tabs and pill bar with current canvas blocks."""
        # ── Refresh pill bar ──────────────────────────────────────────────────
        _PILL_BTYPES = {
            PipelineBlockType.MODEL,
            PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
            PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
            PipelineBlockType.LLM_SCORE,
            PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
            PipelineBlockType.FILTER, PipelineBlockType.TRANSFORM,
            PipelineBlockType.CUSTOM_CODE,
        }
        _PILL_COLORS = {
            PipelineBlockType.MODEL:        C["acc"],
            PipelineBlockType.IF_ELSE:      "#f59e0b",
            PipelineBlockType.SWITCH:       "#f97316",
            PipelineBlockType.FILTER:       "#84cc16",
            PipelineBlockType.TRANSFORM:    "#06b6d4",
            PipelineBlockType.CUSTOM_CODE:  "#10b981",
            PipelineBlockType.LLM_IF:       "#a855f7",
            PipelineBlockType.LLM_SWITCH:   "#7c3aed",
            PipelineBlockType.LLM_FILTER:   "#6366f1",
            PipelineBlockType.LLM_TRANSFORM:"#0ea5e9",
            PipelineBlockType.LLM_SCORE:    "#d946ef",
        }
        # Clear old pills
        while self._pill_layout.count() > 1:  # keep final stretch
            item = self._pill_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Add one pill per relevant block
        _pill_blocks = [b for b in self.canvas.blocks if b.btype in _PILL_BTYPES]
        if not _pill_blocks:
            _empty = QLabel("No model or logic blocks yet — drag from sidebar or click buttons")
            _empty.setStyleSheet(f"color:{C['txt3']};font-size:9px;padding:2px 6px;")
            self._pill_layout.insertWidget(0, _empty)
        else:
            for _pb in _pill_blocks:
                _col = _PILL_COLORS.get(_pb.btype, C["acc"])
                _pill = QPushButton(_pb.label)
                _pill.setFixedHeight(22)
                _pill.setStyleSheet(
                    f"QPushButton{{background:transparent;color:{_col};"
                    f"border:1px solid {_col};border-radius:10px;"
                    f"font-size:10px;font-weight:600;padding:0 8px;}}"
                    f"QPushButton:hover{{background:{_col};color:#fff;}}")
                # Clicking the pill selects + centres the block on canvas
                def _make_jump(blk):
                    def _jump():
                        self.canvas._selected = blk
                        self.canvas.update()
                    return _jump
                _pill.clicked.connect(_make_jump(_pb))
                self._pill_layout.insertWidget(self._pill_layout.count() - 1, _pill)

        # ── Existing: sync intermediate live-output tabs ──────────────────────
        existing_inter_ids = {b.bid for b in self.canvas.blocks
                              if b.btype == PipelineBlockType.INTERMEDIATE}
        # Remove tabs for deleted intermediate blocks
        for bid in list(self._inter_tabs.keys()):
            if bid not in existing_inter_ids:
                idx = self._inter_tabs.pop(bid)
                if idx < self.output_tabs.count():
                    self.output_tabs.removeTab(idx)
                # Re-index remaining entries
                self._inter_tabs = {
                    b2: (i2 if i2 < idx else i2 - 1)
                    for b2, i2 in self._inter_tabs.items()
                }
        # Add tabs for new intermediate blocks
        for b in self.canvas.blocks:
            if b.btype == PipelineBlockType.INTERMEDIATE and b.bid not in self._inter_tabs:
                te = PipelineOutputRenderer(
                    placeholder=f"Live context arriving at ◈ '{b.label}' will appear here…")
                idx = self.output_tabs.addTab(te, f"◈ {b.label[:14]}")
                self._inter_tabs[b.bid] = idx

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.exec_log.append(
            f'<span style="color:{C["txt2"]}">[{ts}]</span> '
            f'<span style="color:{C["txt"]}">{msg}</span>')

    # ── execution ─────────────────────────────────────────────────────────────

    def _validate(self) -> Optional[str]:
        blocks = self.canvas.blocks
        if not blocks:
            return "Canvas is empty — add blocks first."
        if not any(b.btype == PipelineBlockType.INPUT for b in blocks):
            return "Pipeline needs at least one ▶ INPUT block."
        if not any(b.btype == PipelineBlockType.OUTPUT for b in blocks):
            return "Pipeline needs at least one ■ OUTPUT block."
        if not self.canvas.connections:
            return "No connections drawn. Connect the blocks with arrows."
        # Context blocks must be configured before running
        for b in blocks:
            meta = getattr(b, "metadata", {})
            if b.btype == PipelineBlockType.REFERENCE and not meta.get("ref_text"):
                return (f"Reference block '{b.label}' has no text.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.KNOWLEDGE and not meta.get("knowledge_text"):
                return (f"Knowledge block '{b.label}' has no text.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.PDF_SUMMARY and not meta.get("pdf_path"):
                return (f"PDF block '{b.label}' has no PDF selected.\n"
                        f"Right-click it → Configure block…")
            # Logic blocks
            if b.btype == PipelineBlockType.IF_ELSE and not meta.get("condition"):
                return (f"IF/ELSE block '{b.label}' has no condition set.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.SWITCH and not meta.get("switch_expr"):
                return (f"SWITCH block '{b.label}' has no expression set.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.FILTER and not meta.get("filter_cond"):
                return (f"FILTER block '{b.label}' has no condition set.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.TRANSFORM and not meta.get("transform_type"):
                return (f"TRANSFORM block '{b.label}' has no transform type set.\n"
                        f"Right-click it → Configure block…")
            if b.btype == PipelineBlockType.CUSTOM_CODE and not meta.get("custom_code","").strip():
                return (f"Custom Code block '{b.label}' has no code.\n"
                        f"Right-click it → Configure block…")
            # LLM logic blocks
            _LLM_BTYPES_V = {
                PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
                PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
                PipelineBlockType.LLM_SCORE,
            }
            if b.btype in _LLM_BTYPES_V:
                if not meta.get("llm_instruction", "").strip():
                    return (f"LLM logic block '{b.label}' has no instruction.\n"
                            f"Right-click it → Configure block…")
                mp = b.model_path or meta.get("llm_model_path", "")
                if not mp or not Path(mp).exists():
                    return (f"LLM logic block '{b.label}' has no valid model attached.\n"
                            f"Right-click it → Configure block… and select a model.")
        # Model blocks must have a valid file
        for b in blocks:
            if b.btype == PipelineBlockType.MODEL:
                if not b.model_path or not Path(b.model_path).exists():
                    return (f"Model block '{b.label}' has no valid model file.\n"
                            f"Double-click a model in the sidebar to add it.")
        return None

    def _run_pipeline(self):
        err = self._validate()
        if err:
            QMessageBox.warning(self, "Invalid Pipeline", err); return

        text = self.input_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Input",
                                "Enter text in the Input field on the right."); return

        if not self._engine.is_loaded:
            QMessageBox.warning(self, "Engine Not Ready",
                                "Wait for the model to finish loading."); return

        self.exec_log.clear()
        self.output_edit.clear()
        # Reset intermediate live panes
        for bid, idx in self._inter_tabs.items():
            w = self.output_tabs.widget(idx)
            if isinstance(w, PipelineOutputRenderer):
                w.clear_content()
            b = self.canvas._block_by_id(bid)
            if b:
                self.output_tabs.setTabText(idx, f"◈ {b.label[:14]}")
        self.output_edit.clear_content()
        self.output_tabs.setCurrentIndex(0)
        self._log("▶  Pipeline execution started…")

        self.btn_run.setVisible(False)
        self.btn_stop.setVisible(True)

        self._exec_worker = PipelineExecutionWorker(
            list(self.canvas.blocks),
            list(self.canvas.connections),
            text,
            self._engine)
        self._exec_worker.step_started.connect(self._on_step_started)
        self._exec_worker.step_token.connect(self._on_step_token)
        self._exec_worker.step_done.connect(self._on_step_done)
        self._exec_worker.intermediate_live.connect(self._on_intermediate_live)
        self._exec_worker.pipeline_done.connect(self._on_pipeline_done)
        self._exec_worker.err.connect(self._on_exec_err)
        self._exec_worker.log_msg.connect(self._log)
        self._exec_worker.start()

    def _stop_pipeline(self):
        if self._exec_worker:
            self._exec_worker.abort()
            self._exec_worker.wait(2000)
            self._exec_worker = None
        self._log("⏹  Stopped by user.")
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)

    def _on_step_started(self, bid: int, label: str):
        self._log(f"⚡  Block: {label}")
        for b in self.canvas.blocks:
            b.selected = (b.bid == bid)
        self.canvas.update()

    def _on_step_token(self, _bid: int, token: str):
        self.output_edit.append_token(token)

    def _on_intermediate_live(self, bid: int, label: str, text: str):
        """Populate this intermediate block's dedicated output tab."""
        if bid in self._inter_tabs:
            idx    = self._inter_tabs[bid]
            widget = self.output_tabs.widget(idx)
            if isinstance(widget, PipelineOutputRenderer):
                widget.set_content(text)
                self.output_tabs.setTabText(idx, f"◈ {label[:12]} ✅")
                self.output_tabs.setCurrentIndex(idx)

    def _on_step_done(self, bid: int, text: str):
        b = self.canvas._block_by_id(bid)
        lbl = b.label if b else str(bid)
        self._log(f"✅  '{lbl}' → {len(text):,} chars")
        for blk in self.canvas.blocks:
            blk.selected = False
        self.canvas.update()

    def _on_pipeline_done(self, payload: str):
        import json as _json
        try:
            data   = _json.loads(payload)
            final  = data.get("text", payload)
            sender = data.get("sender", "")
        except Exception:
            final  = payload
            sender = ""

        header = (f"**Output from: {sender}**\n\n" if sender else "")
        self._log(f"🏁  Done! {len(final):,} chars from '{sender or 'pipeline'}'.")
        self.output_edit.set_content(header + final)
        self.output_tabs.setCurrentIndex(1)
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)
        self._exec_worker = None
        for b in self.canvas.blocks:
            b.selected = False
        self.canvas.update()

    def _on_exec_err(self, msg: str):
        self._log(f"❌  {msg}")
        self.btn_run.setVisible(True)
        self.btn_stop.setVisible(False)
        self._exec_worker = None
        for b in self.canvas.blocks:
            b.selected = False
        self.canvas.update()

API_REGISTRY = ApiRegistry()

# ═════════════════════════════ API MODELS TAB ════════════════════════════════

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
        lb.setStyleSheet(
            f"color:{C['txt3']};font-size:9px;font-weight:700;letter-spacing:1.2px;")
        return lb

    @staticmethod
    def _inp(placeholder: str = "", pw: bool = False) -> QLineEdit:
        e = QLineEdit()
        e.setPlaceholderText(placeholder)
        e.setFixedHeight(32)
        if pw:
            e.setEchoMode(QLineEdit.EchoMode.Password)
        e.setStyleSheet(
            f"QLineEdit{{background:{C['bg1']};border:1px solid {C['bdr']};"
            f"border-radius:6px;color:{C['txt']};padding:0 10px;font-size:12px;}}"
            f"QLineEdit:focus{{border-color:{C['acc']};}}")
        return e

    @staticmethod
    def _combo_style() -> str:
        return (f"QComboBox{{background:{C['bg1']};border:1px solid {C['bdr']};"
                f"border-radius:6px;color:{C['txt']};padding:0 10px;font-size:12px;}}"
                f"QComboBox::drop-down{{border:none;width:22px;}}"
                f"QComboBox QAbstractItemView{{background:{C['bg2']};color:{C['txt']};"
                f"selection-background-color:{C['acc']};}}")

    @staticmethod
    def _action_btn(label: str, color: str) -> QPushButton:
        b = QPushButton(label)
        b.setFixedHeight(34)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setStyleSheet(
            f"QPushButton{{background:transparent;color:{color};"
            f"border:1px solid {color};border-radius:7px;"
            f"font-size:11px;font-weight:600;padding:0 16px;}}"
            f"QPushButton:hover{{background:{color};color:#fff;}}"
            f"QPushButton:disabled{{color:{C['txt3']};border-color:{C['bdr']};}}")
        return b

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(14)

        hdr = QLabel("🌐  API Models")
        hdr.setStyleSheet(f"color:{C['txt']};font-size:15px;font-weight:700;")
        root.addWidget(hdr)

        sub = QLabel("Connect to any OpenAI-compatible or Anthropic endpoint. "
                     "Once verified, the API model is treated exactly like a local model.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        root.addWidget(sub)

        # ── Connection card (scrollable so custom-format fields never overlap) ─
        card_scroll = QScrollArea()
        card_scroll.setWidgetResizable(True)
        card_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        card_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        card_scroll.setStyleSheet(
            f"QScrollArea{{background:{C['bg2']};border:1px solid {C['bdr']};"
            f"border-radius:10px;}}"
            f"QScrollBar:vertical{{background:{C['bg1']};width:6px;border-radius:3px;}}"
            f"QScrollBar::handle:vertical{{background:{C['bdr2']};border-radius:3px;}}"
            f"QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0px;}}")
        card = QWidget()
        card.setStyleSheet(f"QWidget{{background:{C['bg2']};border:none;}}")
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
        self.combo_provider.setStyleSheet(self._combo_style())
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
        self.combo_model.setStyleSheet(self._combo_style())
        lm.addWidget(self.combo_model)
        r1.addLayout(lm, 2)
        cl.addLayout(r1)

        # API Key
        kl = QVBoxLayout(); kl.setSpacing(4)
        kl.addWidget(QLabel("API Key"))
        self.inp_key = self._inp("sk-…  (leave blank for Ollama / no-auth endpoints)", pw=True)
        kl.addWidget(self.inp_key)
        cl.addLayout(kl)

        # Base URL + Max Tokens row
        r3 = QHBoxLayout(); r3.setSpacing(12)
        ul = QVBoxLayout(); ul.setSpacing(4)
        ul.addWidget(QLabel("Base URL"))
        self.inp_url = self._inp("https://api.openai.com/v1")
        ul.addWidget(self.inp_url)
        r3.addLayout(ul, 3)
        tl = QVBoxLayout(); tl.setSpacing(4)
        tl.addWidget(QLabel("Max Tokens"))
        self.inp_tokens = self._inp("2048")
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
        sep_pf = QFrame(); sep_pf.setFrameShape(QFrame.Shape.HLine)
        sep_pf.setStyleSheet(f"color:{C['bdr']};")
        cl.addWidget(sep_pf)

        pf_hdr = QHBoxLayout()
        pf_lbl = self._sec_lbl("PROMPT FORMAT")
        self.chk_custom_prompt = QCheckBox("Use custom format")
        self.chk_custom_prompt.setStyleSheet(
            f"QCheckBox{{color:{C['txt2']};font-size:11px;}}"
            f"QCheckBox::indicator{{width:14px;height:14px;border:1px solid {C['bdr']};"
            f"border-radius:3px;background:{C['bg1']};}}"
            f"QCheckBox::indicator:checked{{background:{C['acc']};border-color:{C['acc']};}}")
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
        self.combo_template.setStyleSheet(self._combo_style())
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
        self.inp_system.setStyleSheet(
            f"QTextEdit{{background:{C['bg1']};border:1px solid {C['bdr']};"
            f"border-radius:6px;color:{C['txt']};padding:6px 10px;font-size:11px;}}"
            f"QTextEdit:focus{{border-color:{C['acc']};}}")
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
        self.btn_test = self._action_btn("⚡  Test & Load", C["ok"])
        self.btn_save = self._action_btn("💾  Save Config",  C["acc"])
        self.btn_test.clicked.connect(self._test_and_load)
        self.btn_save.clicked.connect(self._save_config)
        br.addWidget(self.btn_test)
        br.addWidget(self.btn_save)
        br.addStretch()
        cl.addLayout(br)

        self.lbl_status = QLabel("● Not connected")
        self.lbl_status.setStyleSheet(f"color:{C['txt3']};font-size:11px;")
        cl.addWidget(self.lbl_status)
        root.addWidget(card_scroll, 2)

        # ── Saved configs ─────────────────────────────────────────────────────
        root.addWidget(self._sec_lbl("SAVED CONFIGS"))

        saved_scroll = QScrollArea()
        saved_scroll.setWidgetResizable(True)
        saved_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        saved_scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
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
        except: max_tok = 2048
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
        self.lbl_status.setText("⏳  Testing connection…")
        self.lbl_status.setStyleSheet(f"color:{C['warn']};font-size:11px;")

        class _T(QThread):
            finished = pyqtSignal(bool, str, object)
            def __init__(self, c): super().__init__(); self.cfg = c
            def run(self):
                eng = ApiEngine()
                ok  = eng.load(self.cfg)
                self.finished.emit(ok, eng.status_text if ok else
                                   "Connection failed — check key / URL / model ID", eng if ok else None)

        self._tester = _T(cfg)
        self._tester.finished.connect(self._on_test_done)
        self._tester.start()

    def _test_and_load(self):
        cfg = self._collect_config()
        if not cfg.model_id:
            self.lbl_status.setText("⚠  Select or enter a model ID first")
            self.lbl_status.setStyleSheet(f"color:{C['warn']};font-size:11px;")
            return
        self._run_test(cfg)

    def _on_test_done(self, ok: bool, msg: str, engine):
        self.btn_test.setEnabled(True)
        if ok:
            self.lbl_status.setText(f"✓  {msg}")
            self.lbl_status.setStyleSheet(f"color:{C['ok']};font-size:11px;")
            self.api_model_loaded.emit(engine)
        else:
            self.lbl_status.setText(f"✗  {msg}")
            self.lbl_status.setStyleSheet(f"color:{C['err']};font-size:11px;")
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
        for cfg in API_REGISTRY.all():
            self._add_saved_card(cfg)

    def _add_saved_card(self, cfg: ApiConfig):
        ICONS = {"OpenAI": "⚡", "Anthropic": "◆", "Groq": "⚙",
                 "Mistral": "🌊", "Together AI": "🤝", "OpenRouter": "🔀",
                 "Ollama": "🦙", "Custom": "🔧"}
        card = QFrame()
        card.setStyleSheet(
            f"QFrame{{background:{C['bg2']};border:1px solid {C['bdr']};"
            f"border-radius:8px;}}")
        cl = QHBoxLayout(card)
        cl.setContentsMargins(14, 10, 14, 10)
        cl.setSpacing(10)

        icon = ICONS.get(cfg.provider, "🌐")
        il = QVBoxLayout(); il.setSpacing(2)
        t = QLabel(f"{icon}  <b>{cfg.name}</b>")
        t.setTextFormat(Qt.TextFormat.RichText)
        t.setStyleSheet(f"color:{C['txt']};font-size:12px;")
        prov_display = getattr(cfg, "custom_provider_name", "") or cfg.provider
        fmt_badge    = "  ·  🎨 custom fmt" if getattr(cfg, "use_custom_prompt", False) else ""
        s = QLabel(f"{prov_display}  ·  {cfg.model_id}{fmt_badge}")
        s.setStyleSheet(f"color:{C['txt2']};font-size:10px;")
        il.addWidget(t); il.addWidget(s)
        cl.addLayout(il, 1)

        def _sb(label, color):
            b = QPushButton(label)
            b.setFixedHeight(28)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton{{background:transparent;color:{color};"
                f"border:1px solid {color};border-radius:6px;"
                f"font-size:10px;font-weight:600;padding:0 10px;}}"
                f"QPushButton:hover{{background:{color};color:#fff;}}")
            return b

        bl = _sb("▶  Load", C["ok"])
        bd = _sb("🗑", C["err"])
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
            API_REGISTRY.remove(cfg.name)
            self._refresh_saved()


# ═════════════════════════════ MAIN WINDOW ══════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Native Lab Pro")
        self.setMinimumSize(1100, 700)
        self.resize(1300, 840)

        self.engine   = LlamaEngine()
        self.sessions: Dict[str, Session] = {}
        self.active:   Optional[Session]  = None

        self._worker:          Optional[QThread] = None
        self._stream_w:        Any = None
        self._summary_worker:  Optional[QThread] = None
        self._summary_bubble:  Any = None
        self._pipeline_worker: Any = None
        self.reasoning_engine:     Any = None
        self.summarization_engine: Any = None
        self.coding_engine:        Any = None
        self.secondary_engine:     Any = None
        self._thinking_block:  Any = None
        self._pipeline_reason_w: Any = None
        self._pipeline_code_w:   Any = None
        self._pipeline_insight_widgets: list = []
        self._chat_pipeline_worker: Optional[QThread] = None
        self._api_engine:   Any = None

        self._force_coding_mode:  bool = False
        self._pending_ref_ctx:    str  = ""
        # ── busy / session-tracking ───────────────────────────────────────────
        self._busy_session_id:    str  = ""   # sid of the session currently generating
        self._stream_session_id:  str  = ""   # sid that owns the active stream worker
        self._stream_buffer:      str  = ""   # shadow text buffer; survives widget deletion
        self._summary_session_id: str  = ""   # sid that owns the active summary worker
        self._multi_pdf_worker:  Optional[QThread] = None
        self._pause_banner: Optional[QWidget] = None
        self._summarizing_active: bool = False
        self.current_ctx = DEFAULT_CTX
        self._ctx_reload_timer = QTimer(self)
        self._ctx_reload_timer.setSingleShot(True)
        self._ctx_reload_timer.timeout.connect(self._apply_new_context)

        self._load_sessions()
        self._build_ui()
        self._build_menu()
        self._build_status_bar()
        # Load saved custom palettes before applying stylesheet
        global C_LIGHT, C_DARK, C, QSS
        if "custom_light_palette" in APP_CONFIG:
            C_LIGHT = {**C_LIGHT, **APP_CONFIG["custom_light_palette"]}
        if "custom_dark_palette" in APP_CONFIG:
            C_DARK = {**C_DARK, **APP_CONFIG["custom_dark_palette"]}
        if "custom_light_palette" in APP_CONFIG or "custom_dark_palette" in APP_CONFIG:
            C   = dict(C_LIGHT) if CURRENT_THEME == "light" else dict(C_DARK)
            QSS = build_qss(C)
        _qss = build_qss(C)
        self.setStyleSheet(_qss)
        self.appearance_tab.load_palette(C_LIGHT if CURRENT_THEME == "light" else C_DARK)

        if self.sessions:
            last = max(self.sessions.values(), key=lambda s: s.id)
            self._switch_session(last.id)
        else:
            self._new_session()

        # ── Restore saved theme preference ────────────────────────────────────
        _saved_theme = APP_CONFIG.get("theme", CURRENT_THEME)
        if _saved_theme != CURRENT_THEME:
            self._toggle_theme()   # silently apply saved theme
        self._update_theme_action_label()

        QTimer.singleShot(300, self._start_model_load)

        # Auto-load parallel engines if prefs say so
        if PARALLEL_PREFS.enabled and PARALLEL_PREFS.auto_load_roles:
            QTimer.singleShot(1000, self._auto_load_parallel_engines)

    def _auto_load_parallel_engines(self):
        """Load all engines whose roles are in the auto_load list."""
        for role in PARALLEL_PREFS.auto_load_roles:
            models = get_model_registry().all_models()
            for m in models:
                if m.get("role") == role and Path(m["path"]).exists():
                    self._start_role_engine_load(role, m["path"])
                    break

    def _start_role_engine_load(self, role: str, path: str):
        attr        = f"{role}_engine"
        loader_attr = f"_loader_{role}"

        # Cancel any in-progress loader for this role to avoid double-loading
        old_loader = getattr(self, loader_attr, None)
        if old_loader and old_loader.isRunning():
            self._log("WARN", f"Cancelling previous {role} loader before new one starts")
            try: old_loader.finished.disconnect()
            except Exception: pass
            old_loader.quit()
            old_loader.wait(2000)

        # Shutdown any existing engine for this role before creating a fresh one
        old_eng = getattr(self, attr, None)
        if old_eng and old_eng.is_loaded:
            self._log("INFO", f"Shutting down existing {role} engine before reload")
            old_eng.shutdown()

        # Always create a brand-new engine instance to avoid state leakage
        new_eng = LlamaEngine()
        setattr(self, attr, new_eng)

        cfg    = get_model_registry().get_config(path)
        loader = ModelLoaderThread(new_eng, path, cfg.ctx)
        loader.log.connect(self._log)
        loader.finished.connect(
            lambda ok, st, r=role, n=Path(path).name:
            self._on_role_engine_loaded(ok, st, r, n, None))
        loader.start()
        setattr(self, loader_attr, loader)

        # Give immediate visual feedback so the user sees something changed
        self._refresh_engine_status()
        self._log("INFO", f"Loading {role} engine: {Path(path).name}")

    # ── context management ────────────────────────────────────────────────────

    def _apply_new_context(self):
        if not self.engine.is_loaded:
            return
        new_ctx = self.ctx_slider.value()
        if new_ctx == getattr(self.engine, "ctx_value", DEFAULT_CTX):
            return
        if new_ctx > 8192:
            ram_estimate = (new_ctx / 1024) * 0.5
            result = QMessageBox.question(
                self, "Confirm Context Reload",
                f"Changing context to {new_ctx:,} tokens requires restarting the model\n"
                f"and may use an additional ~{ram_estimate:.0f} MB of RAM.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if result != QMessageBox.StandardButton.Yes:
                loaded_ctx = getattr(self.engine, "ctx_value", DEFAULT_CTX)
                self.ctx_slider.blockSignals(True)
                self.ctx_slider.setValue(loaded_ctx)
                self.ctx_slider.blockSignals(False)
                self.ctx_input.setText(str(loaded_ctx))
                self.current_ctx = loaded_ctx
                return
        self._log("INFO", f"Reloading model with context {new_ctx}")
        if self._worker:
            if hasattr(self._worker, "abort"):
                self._worker.abort()
            self._worker.wait(1000)
            self._worker = None
        self.engine.shutdown()
        self.current_ctx = new_ctx
        self.ctx_bar.setRange(0, new_ctx)
        self._start_model_load()

    def _on_ctx_changed(self, value: int):
        self.ctx_input.setText(str(value))
        self.current_ctx = value
        if hasattr(self, "model_list"):
            try:
                item = self.model_list.currentItem()
                if item:
                    path = item.data(Qt.ItemDataRole.UserRole)
                    if path and path == getattr(self.engine, "model_path", ""):
                        self.cfg_ctx.blockSignals(True)
                        self.cfg_ctx.setText(str(value))
                        self.cfg_ctx.blockSignals(False)
            except RuntimeError:
                pass
        color = C["ok"]
        warn_text = ""
        if value > 24576:
            color = C["err"]; warn_text = "⚠"
            self.ctx_warn.setToolTip("Very high context.\nExpect heavy RAM usage.")
        elif value > 16384:
            color = C["warn"]; warn_text = "⚠"
            self.ctx_warn.setToolTip("High context.\nPerformance may degrade.")
        else:
            self.ctx_warn.setToolTip("")
        self._ctx_reload_timer.start(2000)
        self.ctx_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height:6px; background:{C['bg2']}; border-radius:3px;
            }}
            QSlider::handle:horizontal {{
                background:{color}; width:14px; margin:-4px 0; border-radius:7px;
            }}
            QSlider::sub-page:horizontal {{
                background:{color}; border-radius:3px;
            }}
        """)
        self.ctx_warn.setText(warn_text)

    def _on_ctx_input_changed(self):
        try:
            value = max(512, min(32768, int(self.ctx_input.text())))
            self.ctx_slider.setValue(value)
        except ValueError:
            self.ctx_input.setText(str(self.ctx_slider.value()))

    # ── layout ────────────────────────────────────────────────────────────────

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
        self.splitter.addWidget(self.sidebar)

        self.tabs = QTabWidget()
        self._tab_overlay = FadeOverlay(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab_changed)

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
        self.tabs.addTab(self.chat_module, "💬  Chat")

        # ── Models tab ──
        self.models_tab = self._build_models_tab()
        self.tabs.addTab(self.models_tab, "🗂  Models")

        # ── Config tab ──
        self.config_tab = ConfigTab()
        self.config_tab.config_changed.connect(self._on_config_changed)
        self.config_tab.btn_resume_job.clicked.connect(self._resume_paused_job)
        self.tabs.addTab(self.config_tab, "⚙️  Config")

        # ── Server tab ──
        self.server_tab = ServerTab()
        self.server_tab.config_changed.connect(
            lambda: self._log("INFO", "Server config updated."))
        self.tabs.addTab(self.server_tab, "🖥️  Server")

        # ── Pipeline Builder tab ──
        self.pipeline_tab = PipelineBuilderTab(self.engine)
        self.tabs.addTab(self.pipeline_tab, "🔗  Pipeline")

        # ── API Models tab ──
        self.api_tab = ApiModelsTab()
        self.api_tab.api_model_loaded.connect(self._on_api_model_loaded)
        self.tabs.addTab(self.api_tab, "🌐  API Models")

        # ── Model Download tab ──
        self.download_tab = ModelDownloadTab()
        self.tabs.addTab(self.download_tab, "⬇️  Download")

        # ── MCP tab ──
        self.mcp_tab = McpTab()
        self.tabs.addTab(self.mcp_tab, "🔌  MCP")

        # ── Logs tab ──
        self.log_console = LogConsole()
        self.tabs.addTab(self.log_console, "🐞  Logs")
        self.appearance_tab = AppearanceTab()
        self.appearance_tab.theme_changed.connect(self._on_appearance_changed)
        self.tabs.addTab(self.appearance_tab, "🎨  Appearance")

        self.splitter.addWidget(self.tabs)
        self.splitter.setSizes([220, 1080])
        self.splitter.setStretchFactor(1, 1)
        ml.addWidget(self.splitter)
        central.setLayout(ml)
        self.setCentralWidget(central)

    # ── models tab ───────────────────────────────────────────────────────────

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
        hdr = QLabel("🗂  GGUF Model Manager")
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
        for role, icon in ROLE_ICONS.items():
            pill = QLabel(f"{icon} {role.capitalize()}")
            pill.setStyleSheet(f"color:{C['txt2']};font-size:10px;"
                               f"background:{C['bg2']};border-radius:4px;padding:2px 6px;")
            legend_row.addWidget(pill)
        legend_row.addStretch()
        list_card_l.addLayout(legend_row)

        self.model_list = QListWidget()
        self.model_list.setObjectName("model_list")
        self.model_list.setMinimumHeight(150)
        self.model_list.setMaximumHeight(240)
        self.model_list.currentItemChanged.connect(self._on_model_list_select)
        list_card_l.addWidget(self.model_list)

        btn_strip = QHBoxLayout()
        btn_strip.setContentsMargins(10, 8, 10, 8); btn_strip.setSpacing(8)
        self.btn_browse_model = QPushButton("📂  Browse GGUF…")
        self.btn_load_primary = QPushButton("⚡  Load Selected")
        self.btn_load_primary.setObjectName("btn_send")
        self.btn_remove_model = QPushButton("🗑  Remove")
        self.btn_remove_model.setObjectName("btn_stop")
        for b in (self.btn_browse_model, self.btn_load_primary, self.btn_remove_model):
            b.setFixedHeight(30); btn_strip.addWidget(b)
        btn_strip.addStretch()
        list_card_l.addLayout(btn_strip)
        root.addWidget(_card(list_card_l))
        root.addSpacing(14)
        self.btn_browse_model.clicked.connect(self._browse_add_model)
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

        # Detected family banner (read-only, auto-set)
        self.cfg_family_lbl = QLabel("—")
        self.cfg_family_lbl.setStyleSheet(
            f"color:{C['acc2']};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;"
        )
        cfg_card_l.addLayout(_field_row("Detected Family:", self.cfg_family_lbl))

        # Quant type banner
        self.cfg_quant_lbl = QLabel("—")
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
        for r in MODEL_ROLES:
            self.cfg_role.addItem(f"{ROLE_ICONS[r]}  {r.capitalize()}", r)
        cfg_card_l.addLayout(_field_row("Role:", self.cfg_role))

        dv1 = QFrame(); dv1.setFrameShape(QFrame.Shape.HLine)
        dv1.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv1)

        self.cfg_threads = QLineEdit(str(DEFAULT_THREADS))
        self.cfg_threads.setFixedWidth(64); self.cfg_threads.setFixedHeight(28)
        self.cfg_threads.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cfg_card_l.addLayout(_field_row("Threads:", self.cfg_threads))

        self.cfg_ctx = QLineEdit(str(DEFAULT_CTX))
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

        self.cfg_param_warn = QLabel("")
        self.cfg_param_warn.setWordWrap(True)
        self.cfg_param_warn.setStyleSheet(
            f"color:{C['warn']};font-size:11px;padding:4px 8px;"
            f"background:#2a2000;border-radius:5px;border:1px solid #5a4500;"
        )
        self.cfg_param_warn.setVisible(False)
        cfg_card_l.addWidget(self.cfg_param_warn)

        dv2 = QFrame(); dv2.setFrameShape(QFrame.Shape.HLine)
        dv2.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};margin:2px 0;")
        cfg_card_l.addWidget(dv2)

        save_row = QHBoxLayout(); save_row.setSpacing(8)
        self.btn_save_cfg = QPushButton("💾  Save Parameters")
        self.btn_save_cfg.setFixedHeight(30)
        self.btn_save_cfg.clicked.connect(self._save_model_config)
        save_row.addWidget(self.btn_save_cfg)
        save_row.addStretch()
        cfg_card_l.addLayout(save_row)
        for wf in (self.cfg_ctx, self.cfg_threads, self.cfg_temp):
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
        eng_card_l.addWidget(self.engine_status_list)
        eng_btn_strip = QHBoxLayout()
        eng_btn_strip.setContentsMargins(10, 8, 10, 8); eng_btn_strip.setSpacing(8)
        self.btn_load_role_engine = QPushButton("⚡  Load Engine for Role")
        self.btn_load_role_engine.setFixedHeight(30)
        self.btn_load_role_engine.clicked.connect(self._load_engine_for_selected)
        self.btn_unload_all = QPushButton("⏏  Unload All")
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

    # ── model config helpers ──────────────────────────────────────────────────

    def _check_param_warnings(self):
        warnings = []
        try:
            ctx = int(self.cfg_ctx.text())
            if ctx > 24576:
                warnings.append(f"⚠  Context {ctx:,} tokens is very high")
            elif ctx > 16384:
                warnings.append(f"⚠  Context {ctx:,} tokens is high")
        except ValueError:
            pass
        try:
            threads = int(self.cfg_threads.text())
            import multiprocessing
            ncpu = multiprocessing.cpu_count()
            if threads > ncpu:
                warnings.append(f"⚠  {threads} threads exceeds {ncpu} logical CPUs")
        except (ValueError, NotImplementedError):
            pass
        try:
            temp = float(self.cfg_temp.text())
            if temp > 1.5:
                warnings.append("⚠  Temperature > 1.5")
            elif temp < 0.05:
                warnings.append("⚠  Temperature near 0")
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
        fam   = detect_model_family(path)
        quant = detect_quant_type(path)
        ql, qcolor = quant_info(quant)
        self.cfg_family_lbl.setText(f"{fam.name}  (template: {fam.template})")
        self.cfg_quant_lbl.setText(f"{quant}  ·  {ql}")
        self.cfg_quant_lbl.setStyleSheet(
            f"color:{qcolor};font-size:11px;"
            f"background:{C['bg2']};border-radius:4px;padding:3px 8px;")

        cfg = get_model_registry().get_config(path)
        idx = self.cfg_role.findData(cfg.role)
        self.cfg_role.setCurrentIndex(max(idx, 0))
        self.cfg_threads.setText(str(cfg.threads))
        if path == getattr(self.engine, "model_path", "") and hasattr(self, "ctx_slider"):
            self.cfg_ctx.setText(str(self.ctx_slider.value()))
        else:
            self.cfg_ctx.setText(str(cfg.ctx))
        self.cfg_temp.setText(str(cfg.temperature))
        self.cfg_topp.setText(str(cfg.top_p))
        self.cfg_rep.setText(str(cfg.repeat_penalty))
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
            rep = float(self.cfg_rep.text()); npred = int(self.cfg_npred.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Parameter", "One or more fields are invalid."); return

        dangers = []
        if ctx > 24576:  dangers.append(f"Context = {ctx:,} tokens (very high)")
        if threads > 32: dangers.append(f"Threads = {threads}")
        if temp > 2.0:   dangers.append(f"Temperature = {temp}")
        if dangers:
            msg = "High-compute parameters:\n\n" + "\n".join(f"  • {d}" for d in dangers) + "\n\nSave?"
            if QMessageBox.warning(self, "⚠ Confirm", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return

        fam = detect_model_family(path)
        cfg = ModelConfig(
            path=path, role=self.cfg_role.currentData() or "general",
            threads=threads, ctx=ctx, temperature=temp, top_p=topp,
            repeat_penalty=rep, n_predict=npred, family=fam.family,
        )
        get_model_registry().set_config(path, cfg)
        self._refresh_model_list()
        self._log("INFO", f"Saved config for {Path(path).name}: family={fam.name}, "
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
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{path}"); return
        cfg = get_model_registry().get_config(path)
        role = cfg.role

        if PARALLEL_PREFS.enabled and role != "general":
            # Warn if loading a second/third model
            n_loaded = sum(1 for r in ("reasoning","summarization","coding","secondary")
                           if getattr(self, f"{r}_engine", None) and
                           getattr(self, f"{r}_engine").is_loaded)
            if n_loaded >= 1:
                size_mb = ModelConfig(path=path).size_mb
                ram_est = max(size_mb * 1.1 / 1000, 1)
                QMessageBox.information(
                    self, "⚠️ Parallel RAM Usage",
                    f"Loading an additional engine (~{ram_est:.1f} GB).\n"
                    f"Total parallel engines after this: {n_loaded + 2}\n\n"
                    f"Ensure you have sufficient free RAM."
                )

        if role == "general":
            idx = self.input_bar.model_combo.findData(path)
            if idx == -1:
                self.input_bar.model_combo.addItem(Path(path).name, path)
                idx = self.input_bar.model_combo.findData(path)
            self.input_bar.model_combo.setCurrentIndex(idx)
            self.engine.shutdown()
            QTimer.singleShot(200, self._start_model_load)
        elif role in ("reasoning", "summarization", "coding", "secondary"):
            # Disable button while loading to prevent rapid re-clicks causing races
            self.btn_load_role_engine.setEnabled(False)
            self.btn_load_role_engine.setText("⏳  Loading…")

            attr        = f"{role}_engine"
            loader_attr = f"_loader_{role}"

            # Cancel any in-progress loader for this role
            old_loader = getattr(self, loader_attr, None)
            if old_loader and old_loader.isRunning():
                self._log("WARN", f"Cancelling stale {role} loader")
                try: old_loader.finished.disconnect()
                except Exception: pass
                old_loader.quit()
                old_loader.wait(2000)

            # Cleanly shut down any already-running engine for this role
            old_eng = getattr(self, attr, None)
            if old_eng and old_eng.is_loaded:
                self._log("INFO", f"Shutting down existing {role} engine")
                old_eng.shutdown()

            new_eng = LlamaEngine()
            setattr(self, attr, new_eng)
            cfg = get_model_registry().get_config(path)

            def _on_loaded_reenable(ok, st, r=role, n=Path(path).name):
                self._on_role_engine_loaded(ok, st, r, n, None)
                self.btn_load_role_engine.setEnabled(True)
                self.btn_load_role_engine.setText("⚡  Load Engine for Role")

            loader = ModelLoaderThread(new_eng, path, cfg.ctx)
            loader.log.connect(self._log)
            loader.finished.connect(_on_loaded_reenable)
            loader.start()
            setattr(self, loader_attr, loader)
            self._log("INFO", f"Loading {role} engine: {Path(path).name}")

        self._refresh_engine_status()

    def _on_role_engine_loaded(self, ok: bool, status: str, role: str,
                                name: str, lbl):
        color = C["ok"] if ok else C["err"]
        icon  = ROLE_ICONS.get(role, "🔌")
        text  = f"{icon} {role.capitalize()}:  {'✅  ' if ok else '❌  '}{name}"
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
            icon       = "🟢" if eng.is_loaded else "⚪"
            model_name = Path(eng.model_path).name if eng.model_path else "not loaded"
            fam_tag    = ""
            if eng.model_path:
                fam = detect_model_family(eng.model_path)
                qt  = detect_quant_type(eng.model_path)
                fam_tag = f"  [{fam.name} · {qt}]"
            mode_tag = f"  [{eng.mode}]" if eng.is_loaded else ""
            item = QListWidgetItem(
                f"  {icon}  {role_name:<22}  {model_name}{mode_tag}{fam_tag}")
            item.setForeground(QColor(C["ok"] if eng.is_loaded else C["txt2"]))
            self.engine_status_list.addItem(item)

    def _unload_all_engines(self):
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
        self.engine.shutdown()
        self._refresh_engine_status()
        self._on_parallel_settings_changed()
        self._log("INFO", "All engines unloaded.")

    def _refresh_model_list(self):
        self.model_list.clear()
        active = getattr(self.engine, "model_path", "")
        for m in get_model_registry().all_models():
            tag       = "📌" if m["source"] == "custom" else "📦"
            role_icon = ROLE_ICONS.get(m.get("role", "general"), "💬")
            ql, qc    = quant_info(m.get("quant", ""))
            label = (f"{tag}  {role_icon} [{m.get('role','general'):<14}]  "
                     f"{m['name']}   ({m['size_mb']} MB)  "
                     f"[{m.get('family','?')}·{m.get('quant','?')}·{ql}]")
            if m["path"] == active: label += "  ✅"
            item = QListWidgetItem(label)
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
        ql, _ = quant_info(quant)
        self._refresh_model_list()
        self._sync_input_bar_combo()
        self._log("INFO",
            f"Added model: {Path(path).name}  →  {fam.name}  ·  {quant}  ·  {ql}")

    def _load_selected_as_primary(self):
        item = self.model_list.currentItem()
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{path}"); return
        idx = self.input_bar.model_combo.findData(path)
        if idx == -1:
            self.input_bar.model_combo.addItem(Path(path).name, path)
            idx = self.input_bar.model_combo.findData(path)
        self.input_bar.model_combo.setCurrentIndex(idx)
        self.engine.shutdown()
        QTimer.singleShot(200, self._start_model_load)
        self._log("INFO", f"Loading primary model: {Path(path).name}")

    def _remove_selected_model(self):
        item = self.model_list.currentItem()
        if not item: return
        get_model_registry().remove(item.data(Qt.ItemDataRole.UserRole))
        self._refresh_model_list()
        self._sync_input_bar_combo()

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
        vm.addAction(QAction("Go to Logs\tCtrl+L",    self, triggered=self._goto_logs))
        vm.addAction(QAction("Go to Models\tCtrl+M",  self, triggered=self._goto_models_tab))
        vm.addSeparator()
        # ── Theme toggle ──────────────────────────────────────────────────────
        self._theme_action = QAction("☀  Switch to Dark Theme", self)
        self._theme_action.setCheckable(False)
        self._theme_action.triggered.connect(self._toggle_theme)
        vm.addAction(self._theme_action)
        self._update_theme_action_label()
        mm = mb.addMenu("Model")
        mm.addAction(QAction("Reload Model", self, triggered=self._reload_model))

    def _build_status_bar(self):
        sb = self.statusBar()
        self.lbl_engine = QLabel("⚪  Loading…")
        self.lbl_engine.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addWidget(self.lbl_engine)
        sb.addWidget(self._vline())

        # Family badge in status bar
        self.lbl_family = QLabel("")
        self.lbl_family.setStyleSheet(f"color:{C['acc2']};padding:0 6px;font-size:10px;")
        sb.addWidget(self.lbl_family)
        sb.addWidget(self._vline())

        sb.addWidget(QLabel("  Context:"))
        self.ctx_slider = QSlider(Qt.Orientation.Horizontal)
        self.ctx_slider.setRange(512, 32768)
        self.ctx_slider.setFixedWidth(140)
        self.ctx_slider.blockSignals(True)
        self.ctx_slider.setValue(DEFAULT_CTX)
        self.ctx_slider.blockSignals(False)
        self.ctx_slider.valueChanged.connect(self._on_ctx_changed)
        sb.addWidget(self.ctx_slider)

        self.ctx_input = QLineEdit(str(DEFAULT_CTX))
        self.ctx_input.setFixedWidth(60)
        self.ctx_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_input.editingFinished.connect(self._on_ctx_input_changed)
        sb.addWidget(self.ctx_input)

        self.ctx_warn = QLabel("")
        self.ctx_warn.setFixedWidth(24)
        self.ctx_warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ctx_warn.setStyleSheet(f"color:{C['warn']};font-weight:bold;")
        sb.addWidget(self.ctx_warn)

        self.ctx_bar = QProgressBar()
        self.ctx_bar.setRange(0, DEFAULT_CTX)
        self.ctx_bar.setValue(0)
        self.ctx_bar.setFixedWidth(100)
        self.ctx_bar.setFixedHeight(8)
        self.ctx_bar.setTextVisible(False)
        sb.addWidget(self.ctx_bar)

        self.ctx_lbl = QLabel(f"0 / {DEFAULT_CTX}")
        self.ctx_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addWidget(self.ctx_lbl)

        sb.addPermanentWidget(self._vline())
        self.tps_lbl = QLabel("— tok/s")
        self.tps_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
        sb.addPermanentWidget(self.tps_lbl)

        if HAS_PSUTIL:
            sb.addPermanentWidget(self._vline())
            self.ram_lbl = QLabel("RAM: —")
            self.ram_lbl.setStyleSheet(f"color:{C['txt2']};padding:0 8px;")
            sb.addPermanentWidget(self.ram_lbl)
            self._ram_timer = QTimer(self)
            self._ram_timer.timeout.connect(self._update_ram)
            self._ram_timer.start(2500)

    @staticmethod
    def _vline() -> QFrame:
        f = QFrame(); f.setFrameShape(QFrame.Shape.VLine)
        f.setStyleSheet(f"color:{C['bdr']};"); return f

    # ── session management ────────────────────────────────────────────────────

    def _load_sessions(self):
        self.sessions = {}
        for p in SESSIONS_DIR.glob("*.json"):
            try:
                s = Session.load(p)
                self.sessions[s.id] = s
            except Exception as e:
                self._log("WARN", f"Skipping corrupt session {p.name}: {e}")

    def _refresh_sidebar(self):
        self.sidebar.refresh(
            self.sessions,
            self.active.id if self.active else "",
            busy_id=self._busy_session_id,
        )

    def _new_session(self):
        s = Session.new()
        self.sessions[s.id] = s
        s.save()
        self._switch_session(s.id)

    def _switch_session(self, sid: str):
        # Null the widget refs NOW so callbacks that fire after the switch
        # won't try to write to widgets that are about to be destroyed by
        # clear_messages().  Workers keep running; _stream_buffer / session
        # ids ensure output is still saved to the correct session on completion.
        self._stream_w         = None
        self._summary_bubble   = None
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None

        # Only reset the generating UI indicator if we're not the busy session
        if sid != self._busy_session_id:
            self.input_bar.set_generating(False)

        s = self.sessions.get(sid)
        if not s: return
        self.active = s
        self.chat_area.clear_messages()
        for m in s.messages:
            self.chat_area.add_message(m.role, m.content, m.timestamp)
        self._refresh_sidebar()
        self.sidebar.set_active(sid)
        self._update_ctx_bar()
        # Sync ChatModule reference panel to this session
        if hasattr(self, "chat_module"):
            self.chat_module.set_session(sid)

    def _delete_session(self, sid: str):
        name = self.sessions[sid].title
        if QMessageBox.question(
            self, "Delete Session", f'Delete "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        p = SESSIONS_DIR / f"{sid}.json"
        if p.exists():
            p.unlink()
        del self.sessions[sid]
        if self.active and self.active.id == sid:
            self.active = None
            self.chat_area.clear_messages()
        if self.sessions:
            last = max(self.sessions.values(), key=lambda s: s.id)
            self._switch_session(last.id)
        else:
            self._new_session()

    def _rename_session(self, sid: str, title: str):
        if sid in self.sessions:
            self.sessions[sid].title = title
            self.sessions[sid].save()
            self._refresh_sidebar()

    def _clear_chat(self):
        if not self.active: return
        if QMessageBox.question(
            self, "Clear Chat", "Clear all messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return
        self.active.messages.clear()
        self.active.save()
        self.chat_area.clear_messages()
        self._update_ctx_bar()

    # ── model loading ─────────────────────────────────────────────────────────

    def _start_model_load(self):
        model = self.input_bar.selected_model
        if not model or not Path(model).exists():
            model = str(MODELS_DIR / DEFAULT_MODEL)
        self.lbl_engine.setText("🔄  Loading model…")
        fam   = detect_model_family(model)
        quant = detect_quant_type(model)
        ql, _ = quant_info(quant)
        self.lbl_family.setText(f"{fam.name}  ·  {quant}  ·  {ql}")
        self._log("INFO", f"Loading model: {Path(model).name}  [{fam.name} / {quant}]")
        self._loader = ModelLoaderThread(self.engine, model, self.ctx_slider.value())
        self._loader.log.connect(self._log)
        self._loader.finished.connect(self._on_model_loaded)
        self._loader.start()

    def _on_model_loaded(self, ok: bool, status: str):
        self.ctx_slider.setEnabled(True)
        self.lbl_engine.setText(status)
        color = C["ok"] if ok else C["err"]
        self.lbl_engine.setStyleSheet(f"color:{color};padding:0 8px;")
        self._log("INFO" if ok else "ERROR", f"Model load: {status}")
        if hasattr(self, "model_list"):
            self._refresh_model_list()
        # Keep pipeline tab's engine reference up-to-date
        if hasattr(self, "pipeline_tab"):
            self.pipeline_tab.update_engine(self.engine)

    def _reload_model(self):
        self.engine.shutdown()
        QTimer.singleShot(500, self._start_model_load)

    # ── coding detection ──────────────────────────────────────────────────────

    _CODING_KEYWORDS = (
        "def ", "class ", "import ", "function ", "```", "debug ",
        "fix bug", "write code", "python ", "javascript", "typescript",
        "rust ", "c++", "c#", "golang", " sql", "regex", "script ",
        "implement ", "algorithm", "refactor", "syntax error",
        "code to ", "write a ", "create a ", "build a ", "generate code",
    )

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

    def _active_engine_for(self, text: str) -> "LlamaEngine":
        if self._is_coding_prompt(text) and \
                self.coding_engine and self.coding_engine.is_loaded:
            return self.coding_engine
        # API engine takes priority over local engine when loaded
        if self._api_engine and self._api_engine.is_loaded:
            return self._api_engine
        return self.engine

    # ── tab fade ─────────────────────────────────────────────────────────────

    def _on_tab_changed(self, idx: int):
        w = self.tabs.widget(idx)
        if not w:
            return
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

    # ── API engine ────────────────────────────────────────────────────────────

    def _on_api_model_loaded(self, api_engine: ApiEngine):
        """Called when ApiModelsTab successfully verifies an API model."""
        self._api_engine = api_engine
        cfg = api_engine._config
        name = f"{cfg.provider}  ·  {cfg.model_id}" if cfg else api_engine.status_text
        self.lbl_engine.setText(f"🌐  {name}")
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self.lbl_family.setText(f"API  ·  max {cfg.max_tokens if cfg else '?'} tokens")
        self._log("INFO", f"API model loaded: {api_engine.status_text}")
        # Update pipeline tab to use api engine for pipeline blocks
        if hasattr(self, "pipeline_tab"):
            self.pipeline_tab.update_engine(api_engine)

    # ── pipeline-from-chat ────────────────────────────────────────────────────

    def _on_pipeline_from_chat(self):
        """Run a saved pipeline from the chat window. Output rendered as structured chat bubbles."""
        saved = list_saved_pipelines()
        if not saved:
            QMessageBox.information(
                self, "No Saved Pipelines",
                "No saved pipelines found.\n\n"
                "Build and save a pipeline in the 🔗 Pipeline tab first.")
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

        try:
            blocks, conns = load_pipeline(pipeline_name)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return

        if not self.engine.is_loaded:
            QMessageBox.warning(self, "Engine Not Ready",
                                "Wait for the model to finish loading."); return

        ts = datetime.now().strftime("%H:%M")
        self.chat_area.add_message("user", text, ts)
        self.input_bar.input.clear()

        self.chat_area.add_message(
            "assistant",
            f"🔗 **Running pipeline: {pipeline_name}**\n\n"
            f"_{len(blocks)} blocks · processing…_",
            ts)

        self._chat_pipeline_worker = PipelineExecutionWorker(
            blocks, conns, text, self.engine)
        self._chat_pipeline_worker.step_started.connect(
            lambda bid, lbl: self._chat_pipeline_step(lbl))
        self._chat_pipeline_worker.intermediate_live.connect(
            self._chat_pipeline_intermediate)
        self._chat_pipeline_worker.pipeline_done.connect(
            self._chat_pipeline_done)
        self._chat_pipeline_worker.err.connect(
            lambda msg: self.chat_area.add_message(
                "assistant",
                f"❌ Pipeline error:\n\n{msg}",
                datetime.now().strftime("%H:%M")))
        self._chat_pipeline_worker.log_msg.connect(
            lambda m: self._log("INFO", m))
        self._chat_pipeline_worker.start()

    def _chat_pipeline_step(self, label: str):
        ts = datetime.now().strftime("%H:%M")
        self.chat_area.add_message("system_note",
                                   f"⚡ Processing block: **{label}**…", ts)

    def _chat_pipeline_intermediate(self, bid: int, label: str, text: str):
        ts = datetime.now().strftime("%H:%M")
        self.chat_area.add_message(
            "pipeline_intermediate",
            f"◈ **{label}** — intermediate output\n\n{text}", ts)

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
        header = f"■ **Output** _(from: {sender})_\n\n" if sender else "■ **Output**\n\n"
        self.chat_area.add_message("assistant", header + final, ts)
        self._chat_pipeline_worker = None

    # ── chat send ─────────────────────────────────────────────────────────────

    def _on_send_with_refs(self, text: str, ref_ctx: str):
        """Called by ChatModule — injects reference context into prompt."""
        self._pending_ref_ctx = ref_ctx
        self._on_send(text)

    def _on_send(self, text: str):
        if not self.active:    self._new_session()
        if not self.engine.is_loaded:
            self._log("WARN", "Model not yet loaded — please wait."); return
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
                "You can resume paused jobs from the Config tab.")
            return
        ts = datetime.now().strftime("%H:%M")
        if not self.active.messages:
            self.active.title = text[:40].replace("\n", " ")
            self.active.save()

        self.active.add_message("user", text)
        self.active.save()
        self.chat_area.add_message("user", text, ts)
        self._update_ctx_bar()

        # ── Pipeline mode ─────────────────────────────────────────────────────
        if self._can_use_pipeline(text):
            self._start_pipeline(text, ts)
            return

        # ── Normal / single-engine mode ───────────────────────────────────────
        active_eng = self._active_engine_for(text)
        is_coding  = active_eng is self.coding_engine
        eng_label  = "💻 Coding engine" if is_coding else active_eng.status_text

        ctx_chars = getattr(active_eng, "ctx_value", DEFAULT_CTX) * 4
        prompt    = self.active.build_prompt(
            model_path=active_eng.model_path,
            max_chars=ctx_chars
        )
        # Inject reference context if available
        ref_ctx = getattr(self, "_pending_ref_ctx", "")
        if ref_ctx:
            fam = detect_model_family(active_eng.model_path)
            ref_block = (
                f"{fam.bos}{fam.user_prefix}"
                f"The following reference material is provided for context:\n\n"
                f"{ref_ctx}\n\n"
                f"Use this reference when answering the user's question."
                f"{fam.user_suffix}{fam.assistant_prefix}"
            )
            prompt = ref_block + "\n" + prompt
            self._pending_ref_ctx = ""

        cfg_pred = DEFAULT_N_PRED
        if active_eng.model_path:
            cfg_pred = get_model_registry().get_config(active_eng.model_path).n_predict

        self._stream_w = self.chat_area.add_message(
            "assistant", "", ts, tag="💻 Coding" if is_coding else "")
        self._log("INFO", f"Prompt ≈ {len(prompt)} chars · engine: {eng_label}")

        # For API engines, pass the full structured message history
        if isinstance(active_eng, ApiEngine):
            api_msgs = [{"role": m.role, "content": m.content}
                        for m in self.active.messages[-60:]]
            active_eng.set_messages(api_msgs)

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
        lbl_txt = "💻 Coding…" if is_coding else "⚡  Generating…"
        self.lbl_engine.setText(lbl_txt)
        self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")

    # ── Pipeline mode orchestration ───────────────────────────────────────────

    def _collect_insight_engines(self):
        """Return list of (label, engine) for all active non-coding loaded engines."""
        candidates = [
            ("🧠 Reasoning",     self.reasoning_engine),
            ("📝 Summarization", self.summarization_engine),
            ("🔮 Secondary",     self.secondary_engine),
        ]
        # Also include primary engine if it's not the coding engine
        if self.engine and self.engine.is_loaded and self.engine is not self.coding_engine:
            candidates.append(("⚡ Primary", self.engine))

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
                self._log("WARN", f"{role} engine in CLI mode — attempting server upgrade…")
                ok = eng.ensure_server_or_reload(log_cb=self._log)
                if not ok:
                    self._log("ERROR",
                        f"{role} engine could not start server mode — aborting pipeline")
                    self.chat_area.add_message(
                        "assistant",
                        f"⚠️ Pipeline aborted: **{role}** engine could not start in server mode.\n"
                        f"Try reloading the model from the Models tab.",
                        ts)
                    self.input_bar.set_generating(False)
                    return

        insight_engines = self._collect_insight_engines()

        if not insight_engines:
            # Fallback: no insight engines, just run coding engine directly
            self._log("WARN", "No insight engines available — running coding engine directly")
            active_eng = self.coding_engine
            ctx_chars  = getattr(active_eng, "ctx_value", DEFAULT_CTX) * 4
            prompt     = self.active.build_prompt(model_path=active_eng.model_path, max_chars=ctx_chars)
            cfg_pred   = get_model_registry().get_config(active_eng.model_path).n_predict
            self._stream_w = self.chat_area.add_message("assistant", "", ts, tag="💻 Coding")
            self._worker = active_eng.create_worker(prompt, n_predict=cfg_pred, model_path=active_eng.model_path)
            self._worker.token.connect(self._on_token)
            self._worker.done.connect(self._on_done)
            self._worker.err.connect(self._on_err)
            self._worker.start()
            self.input_bar.set_generating(True)
            return

        n_engines = len(insight_engines)
        self._log("INFO", f"🔗 Pipeline: {n_engines} insight engine(s) → coding")
        self.lbl_engine.setText("🧠 Structural Insights…")
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
        self._pipeline_code_w   = self.chat_area.add_message("assistant", "", ts, tag="💻 Coding")

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
        self.lbl_engine.setText(f"{label} analysing…")
        self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")

    def _on_pipeline_insight_token(self, idx: int, token: str):
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

    # keep old name for compat
    def _on_pipeline_reason_token(self, text: str):
        self._on_pipeline_insight_token(0, text)

    def _on_pipeline_reason_done(self, full_text: str):
        self._on_pipeline_insight_done(0, full_text)

    def _on_pipeline_stage(self, stage: str):
        if stage == "coding":
            self.lbl_engine.setText("💻 Coding…")
            self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        elif stage == "insights":
            self.lbl_engine.setText("🧠 Structural Insights…")
            self.lbl_engine.setStyleSheet(f"color:{C['pipeline']};padding:0 8px;")

    def _on_pipeline_code_token(self, text: str):
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.append_text(text)
            except RuntimeError:
                pass
        self.chat_area._scroll_bottom()

    def _on_pipeline_done(self, tps: float):
        self.tps_lbl.setText(f"{tps:.1f} tok/s")
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
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
        self.lbl_engine.setText("❌  Pipeline Error")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._pipeline_code_w:
            try:
                self._pipeline_code_w.append_text(f"\n\n⚠️ Pipeline error: {msg}")
            except RuntimeError:
                pass
        self._pipeline_insight_widgets = []
        self._pipeline_reason_w = None
        self._pipeline_code_w   = None
        self._pipeline_worker   = None
        self._busy_session_id   = ""
        self._refresh_sidebar()

    # ── normal streaming handlers ──────────────────────────────────────────────

    def _on_token(self, text: str):
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
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass
        self._save_streamed()
        self._busy_session_id = ""
        self._refresh_sidebar()

    def _on_err(self, msg: str):
        self._log("ERROR", msg)
        self.lbl_engine.setText("❌  Error")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")
        self.input_bar.set_generating(False)
        if self._stream_w:
            try: self._stream_w.append_text(f"\n\n⚠️ Error: {msg}")
            except RuntimeError: pass
        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass
        self._save_streamed()
        self._busy_session_id = ""
        self._refresh_sidebar()

    def _on_stop(self):
        if self._pipeline_worker:
            if hasattr(self._pipeline_worker, "abort"):
                self._pipeline_worker.abort()
            self._pipeline_worker.wait(2000)
            self._pipeline_worker = None
            for w in (self._pipeline_reason_w, self._pipeline_code_w):
                if w:
                    try: w.finalize()
                    except RuntimeError: pass
            self._pipeline_reason_w = None
            self._pipeline_code_w   = None

        if self._worker:
            if hasattr(self._worker, "abort"): self._worker.abort()
            self._worker.wait(2000)
            self._worker = None

        if self._summary_worker:
            if hasattr(self._summary_worker, "request_pause"):
                self._summary_worker.request_pause()
                self._summary_worker.wait(4000)
            else:
                if hasattr(self._summary_worker, "abort"):
                    self._summary_worker.abort()
                self._summary_worker.wait(2000)
            self._summary_worker = None
            if hasattr(self, "_summary_bubble") and self._summary_bubble:
                self._summary_bubble.append_text(
                    "\n\n⏸ Paused & saved to disk. Resume from the Config tab.")
                self._summary_bubble = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()

        if self._multi_pdf_worker:
            if hasattr(self._multi_pdf_worker, "request_pause"):
                self._multi_pdf_worker.request_pause()
                self._multi_pdf_worker.wait(4000)
            else:
                if hasattr(self._multi_pdf_worker, "abort"):
                    self._multi_pdf_worker.abort()
                self._multi_pdf_worker.wait(2000)
            self._multi_pdf_worker = None
            if hasattr(self, "_summary_bubble") and self._summary_bubble:
                self._summary_bubble.append_text(
                    "\n\n⏸ Multi-PDF paused & saved. Resume from the Config tab.")
                self._summary_bubble = None
            if hasattr(self, "config_tab"):
                self.config_tab.refresh_paused_jobs()

        if self._stream_w:
            try: self._stream_w.finalize()
            except RuntimeError: pass

        self.input_bar.set_generating(False)
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self._log("INFO", "Generation stopped by user.")
        self._save_streamed(suffix=" ✋")
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

    # ── PDF loading ────────────────────────────────────────────────────────────

    def _load_pdf(self):
        if not HAS_PDF:
            QMessageBox.warning(self, "Missing Dependency",
                                "Install PyPDF2:  pip install PyPDF2"); return
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

        ctx_chars   = getattr(self.engine, "ctx_value", DEFAULT_CTX) * 3
        DIRECT_LIMIT = min(ctx_chars, 6000)

        if len(text) <= DIRECT_LIMIT:
            self.input_bar.input.setPlainText(
                f"The following is extracted from '{filename}':\n\n{text}"
                "\n\nPlease summarise the key points.")
            self._log("INFO", "Short document — loaded directly into prompt.")
            return

        if not self.active: self._new_session()
        self.active.title = f"Summary: {filename}"
        self.active.save()
        self._refresh_sidebar()

        ts = datetime.now().strftime("%H:%M")
        mode = self.input_bar.summary_mode
        mode_label = {"summary": "📋 Summary", "logical": "🔬 Logical Analysis", "advice": "💡 Advisory"}.get(mode, "📋 Summary")

        self._summary_bubble = self.chat_area.add_message(
            "assistant",
            f"📄  Starting **{mode_label}** of **{filename}** "
            f"({n_pages} pages, {len(text):,} chars)…\n", ts)

        self.input_bar.set_generating(True)
        self.input_bar.input.setEnabled(False)
        self.input_bar.input.setPlaceholderText(
            "⏸ Summarization in progress — pause or abort above before typing…")
        self._summarizing_active = True
        self.lbl_engine.setText("📄  Summarising…")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        # Add pause banner to chat
        self._pause_banner = self.chat_area.add_pause_banner(
            on_pause_cb=self._on_pause_summary,
            on_abort_cb=self._on_stop
        )

        if self._summary_worker:
            if hasattr(self._summary_worker, "abort"): self._summary_worker.abort()
            self._summary_worker.wait(1000)

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
            self.lbl_engine.setText("⏸  Pausing…")
            self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")
            if self._pause_banner:
                try:
                    self._pause_banner.status_lbl.setText("Pausing after current chunk…")
                except RuntimeError:
                    pass
        elif self._multi_pdf_worker and hasattr(self._multi_pdf_worker, "request_pause"):
            self._multi_pdf_worker.request_pause()
            self.lbl_engine.setText("⏸  Pausing…")
            self.lbl_engine.setStyleSheet(f"color:{C['warn']};padding:0 8px;")

    def _on_summary_progress(self, msg: str):
        self._log("INFO", msg)
        self.lbl_engine.setText(f"📄  {msg}")
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
                bubble.append_text(f"\n✅ Section {num}/{total} summarised.\n")
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
            self.lbl_engine.setText(self.engine.status_text)
            self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
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
            QMessageBox.warning(self, "Missing Dep", "Install PyPDF2"); return
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
            f"📚  Starting multi-PDF summarization:\n"
            + "\n".join(f"  • {fn} ({len(t):,} chars)" for fn, t in pdf_texts)
            + f"\n\n⏳ Processing…\n", ts)

        from math import ceil
        total_chunks = sum(
            ceil(len(t) / int(APP_CONFIG["summary_chunk_chars"])) for _, t in pdf_texts)
        self._thinking_block = self.chat_area.add_thinking_block(total_chunks)

        self.input_bar.set_generating(True)
        self.lbl_engine.setText("📚  Multi-PDF…")
        self.lbl_engine.setStyleSheet(f"color:{C['acc']};padding:0 8px;")

        sid = self.active.id if self.active else "default"
        self._multi_pdf_worker = MultiPdfSummaryWorker(
            self.engine, pdf_texts, sid,
            engine2=getattr(self, "summarization_engine", None) or
                    getattr(self, "reasoning_engine", None))
        self._multi_pdf_worker.file_started.connect(
            lambda fi, fn, nc: self._log("INFO", f"PDF {fi+1}: {fn} ({nc} chunks)"))
        self._multi_pdf_worker.file_progress.connect(
            lambda fi, msg: self.lbl_engine.setText(f"📄 PDF {fi+1}: {msg}"))
        self._multi_pdf_worker.ram_warning.connect(self._on_multi_pdf_ram_warning)
        self._multi_pdf_worker.section_done.connect(self._on_section_done)
        self._multi_pdf_worker.file_done.connect(
            lambda fi, s: (
                self._log("INFO", f"File {fi+1} summary done ({len(s)} chars)"),
                self._summary_bubble and self._summary_bubble.append_text(
                    f"\n✅ Document {fi+1} summarised.\n")))
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
        self.lbl_engine.setText(self.engine.status_text)
        self.lbl_engine.setStyleSheet(f"color:{C['ok']};padding:0 8px;")
        self._update_ctx_bar()

    def _on_config_changed(self):
        self._log("INFO", "App config updated and saved.")

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
                f"▶  Resuming multi-PDF job from file {next_fi+1}/{total}, "
                f"chunk {next_ci+1}…\n", ts)
            from math import ceil
            est = sum(
                ceil(len(t) / int(APP_CONFIG["summary_chunk_chars"]))
                for _, t in pdf_texts[next_fi:])
            self._thinking_block = self.chat_area.add_thinking_block(max(est, 1))
            self.input_bar.set_generating(True)
            self.lbl_engine.setText("▶  Resuming Multi-PDF…")
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
                lambda fi, msg: self.lbl_engine.setText(f"📄 PDF {fi+1}: {msg}"))
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
            f"▶  Resuming summarization of **{filename}** "
            f"from chunk {next_chunk}/{total}…\n", ts)

        from math import ceil
        est_remaining = (total - next_chunk) if isinstance(total, int) else 5
        self._thinking_block = self.chat_area.add_thinking_block(
            max(est_remaining, 1))

        self.input_bar.set_generating(True)
        self.lbl_engine.setText(f"▶  Resuming '{filename}'…")
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
            f"💡 **Pause available** — The summarization has processed "
            f"{APP_CONFIG['pause_after_chunks']} chunks. You can pause & save "
            f"state now to resume later, or continue processing.\n\n"
            f"Click ⏸ **Pause** in the stop button area to pause, "
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
            self.lbl_engine.setText("⏸  Paused — state saved")
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
            self._summary_bubble.append_text(f"\n\n⚠️ Error: {msg}")
        self._summary_worker = None
        self._summary_bubble = None
        self.input_bar.set_generating(False)
        self.lbl_engine.setText("❌  Summary failed")
        self.lbl_engine.setStyleSheet(f"color:{C['err']};padding:0 8px;")

    # ── export ────────────────────────────────────────────────────────────────

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

    # ── status bar helpers ────────────────────────────────────────────────────

    def _update_ctx_bar(self):
        if not self.active: return
        est     = self.active.approx_tokens
        max_ctx = getattr(self.engine, "ctx_value", self.ctx_slider.value())
        self.ctx_bar.setRange(0, max_ctx)
        self.ctx_bar.setValue(min(est, max_ctx))
        self.ctx_lbl.setText(f"{est:,} / {max_ctx:,}")
        pct   = est / max_ctx if max_ctx > 0 else 0
        color = C["ok"] if pct < 0.6 else C["warn"] if pct < 0.85 else C["err"]
        self.ctx_bar.setStyleSheet(
            f"QProgressBar{{background:{C['bg2']};border:1px solid {C['bdr']};"
            f"border-radius:3px;height:8px;}}"
            f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"
        )

    def _update_ram(self):
        mem  = psutil.virtual_memory()
        used = mem.used  // (1024 ** 3)
        tot  = mem.total // (1024 ** 3)
        self.ram_lbl.setText(f"RAM: {used}/{tot} GB")

    def _log(self, level: str, msg: str):
        if hasattr(self, "log_console"):
            self.log_console.log(level, msg)
        else:
            print(f"[{level}] {msg}", file=sys.stderr)

    def _toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())

    # ── Theme switching ───────────────────────────────────────────────────────

    def _update_theme_action_label(self):
        if not hasattr(self, "_theme_action"):
            return
        if CURRENT_THEME == "light":
            self._theme_action.setText("🌙  Switch to Dark Theme")
        else:
            self._theme_action.setText("☀  Switch to Light Theme")

    def _on_appearance_changed(self, new_palette: dict):
        global C_LIGHT, C_DARK, C, QSS
        if CURRENT_THEME == "light":
            C_LIGHT = dict(new_palette)
            C = dict(C_LIGHT)
        else:
            C_DARK = dict(new_palette)
            C = dict(C_DARK)
        QSS = build_qss(C)
        self.setStyleSheet(QSS)
        if self.active:
            self._switch_session(self.active.id)

    def _toggle_theme(self):
        global CURRENT_THEME, C, QSS
        CURRENT_THEME = "dark" if CURRENT_THEME == "light" else "light"
        C   = C_LIGHT if CURRENT_THEME == "light" else C_DARK
        QSS = build_qss(C)
        self.setStyleSheet(QSS)
        APP_CONFIG["theme"] = CURRENT_THEME
        save_app_config(APP_CONFIG)
        self._update_theme_action_label()
        self._log("INFO", f"Theme switched to: {CURRENT_THEME}")

        # Rebuild Models tab (has baked dark card gradients)
        mt_idx = self.tabs.indexOf(self.models_tab)
        self.models_tab.setParent(None)
        self.models_tab.deleteLater()
        self.models_tab = self._build_models_tab()
        self.tabs.insertTab(mt_idx, self.models_tab, "🗂  Models")

        # Rebuild Config tab (has baked dark card backgrounds)
        ct_idx = self.tabs.indexOf(self.config_tab)
        self.config_tab.setParent(None)
        self.config_tab.deleteLater()
        self.config_tab = ConfigTab()
        self.config_tab.config_changed.connect(self._on_config_changed)
        self.config_tab.btn_resume_job.clicked.connect(self._resume_paused_job)
        self.tabs.insertTab(ct_idx, self.config_tab, "⚙️  Config")

        # Rebuild Server tab
        srv_idx = self.tabs.indexOf(self.server_tab)
        self.server_tab.setParent(None)
        self.server_tab.deleteLater()
        self.server_tab = ServerTab()
        self.server_tab.config_changed.connect(
            lambda: self._log("INFO", "Server config updated."))
        self.tabs.insertTab(srv_idx, self.server_tab, "🖥️  Server")

        # Rebuild Download tab
        dl_idx = self.tabs.indexOf(self.download_tab)
        self.download_tab.setParent(None)
        self.download_tab.deleteLater()
        self.download_tab = ModelDownloadTab()
        self.tabs.insertTab(dl_idx, self.download_tab, "⬇️  Download")

        # Rebuild MCP tab
        mcp_idx = self.tabs.indexOf(self.mcp_tab)
        self.mcp_tab.setParent(None)
        self.mcp_tab.deleteLater()
        self.mcp_tab = McpTab()
        self.tabs.insertTab(mcp_idx, self.mcp_tab, "🔌  MCP")

        # Rebuild Logs tab
        log_idx = self.tabs.indexOf(self.log_console)
        self.log_console.setParent(None)
        self.log_console.deleteLater()
        self.log_console = LogConsole()
        self.tabs.insertTab(log_idx, self.log_console, "🐞  Logs")

        # Refresh Appearance tab palette to match active theme
        self.appearance_tab.load_palette(C_LIGHT if CURRENT_THEME == "light" else C_DARK)

        # Rebuild chat bubbles + status bar colours
        self._on_model_loaded(self.engine.is_loaded, self.engine.status_text)
        if self.active:
            self._switch_session(self.active.id)

    def _goto_logs(self):
        self.tabs.setCurrentWidget(self.log_console)

    def _goto_models_tab(self):
        self.tabs.setCurrentWidget(self.models_tab)

    def closeEvent(self, event):
        self._log("INFO", "Shutdown — stopping engine…")
        if self._worker:
            try:
                if hasattr(self._worker, "abort"): self._worker.abort()
                self._worker.wait(1000)
            except Exception:
                pass
        if self._pipeline_worker:
            try:
                self._pipeline_worker.abort()
                self._pipeline_worker.wait(1000)
            except Exception:
                pass
        if hasattr(self, "pipeline_tab") and self.pipeline_tab._exec_worker:
            try:
                self.pipeline_tab._exec_worker.abort()
                self.pipeline_tab._exec_worker.wait(1000)
            except Exception:
                pass

        # Cancel any pending role loaders gracefully before shutting down engines
        for role in ("reasoning", "summarization", "coding", "secondary"):
            ldr = getattr(self, f"_loader_{role}", None)
            if ldr and ldr.isRunning():
                try: ldr.finished.disconnect()
                except Exception: pass
                ldr.quit()
                ldr.wait(1500)

        self.engine.shutdown()
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng: eng.shutdown()

        # Final safety net: kill ANY remaining llama-server stragglers
        # (handles orphans from previous crashed sessions too)
        kill_stray_llama_servers(keep_pids=set())
        super().closeEvent(event)


# ═════════════════════════════ ENTRY POINT ══════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Native Lab Pro")
    app.setWindowIcon(QIcon('icon.png'))
    _fnt = QFont("Inter")
    if not _fnt.exactMatch():
        _fnt = QFont("Segoe UI")
    _fnt.setPointSize(10)
    app.setFont(_fnt)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    win = MainWindow()
    win.setWindowTitle("✦  Native Lab Pro  v2")
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()