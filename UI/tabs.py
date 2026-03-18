from imports.import_global import HAS_PSUTIL, QSlider, QColorDialog, psutil, Optional, subprocess, Dict, QHBoxLayout, datetime, Qt, pyqtSignal, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QListWidget, QListWidgetItem, QMenu, QInputDialog, QColor, QTextEdit, QFont, QCheckBox, QMessageBox, QScrollArea , QFrame
from .UI_const import C_DARK, C_LIGHT, CURRENT_THEME, C
from GlobalConfig.config_global import APP_CONFIG, APP_CONFIG_DEFAULTS, CONFIG_FIELD_META, save_app_config, MODEL_ROLES, ROLE_ICONS
from Prefrences.prefrence_global import ParallelPrefs, PARALLEL_PREFS
from components.components_global import list_paused_jobs, delete_paused_job, load_paused_job
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
