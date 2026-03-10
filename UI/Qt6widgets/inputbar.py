from imports.import_global import Qt, QEvent, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QTextEdit, QTimer, pyqtSignal
from UI.UI_global import C
from GlobalConfig.config_global import DEFAULT_MODEL
from Model.model_global import detect_model_family, detect_quant_type, quant_info, MODEL_REGISTRY, MODELS_DIR
class InputBar(QWidget):
    send_requested       = pyqtSignal(str)
    stop_requested       = pyqtSignal()
    pdf_requested        = pyqtSignal()
    clear_requested      = pyqtSignal()
    pipeline_run_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("input_bar")
        root = QVBoxLayout()
        root.setContentsMargins(18, 12, 18, 14)
        root.setSpacing(9)

        toolbar = QHBoxLayout(); toolbar.setSpacing(8)
        model_lbl = QLabel("Model:")
        model_lbl.setStyleSheet(f"color:{C['txt2']};font-size:11px;")
        toolbar.addWidget(model_lbl)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(240)
        self._populate_models()
        toolbar.addWidget(self.model_combo)

        # Family info badge
        self.family_badge = QLabel("")
        self.family_badge.setObjectName("family_badge")
        toolbar.addWidget(self.family_badge)
        self.model_combo.currentIndexChanged.connect(self._update_family_badge)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        toolbar.addStretch()

        self.pdf_btn   = QPushButton("📄  PDF")
        self.clear_btn = QPushButton("🗑  Clear")
        for b in (self.pdf_btn, self.clear_btn):
            b.setFixedWidth(86); b.setFixedHeight(30)
        self.pdf_btn.clicked.connect(self.pdf_requested)
        self.clear_btn.clicked.connect(self.clear_requested)
        toolbar.addWidget(self.pdf_btn)
        toolbar.addWidget(self.clear_btn)

        self.code_btn = QPushButton("💻  Code")
        self.code_btn.setFixedSize(80, 30)
        self.code_btn.setCheckable(True)
        self.code_btn.setObjectName("code_btn")
        self.code_btn.setToolTip(
            "Force coding engine for this message\n"
            "(auto-detects code keywords even when off)")
        toolbar.addWidget(self.code_btn)
        # Summary mode selector
        self.summary_mode_combo = QComboBox()
        self.summary_mode_combo.setFixedHeight(30)
        self.summary_mode_combo.setFixedWidth(110)
        self.summary_mode_combo.setObjectName("summary_combo")
        self.summary_mode_combo.addItem("📋 Summary", "summary")
        self.summary_mode_combo.addItem("🔬 Logical", "logical")
        self.summary_mode_combo.addItem("💡 Advice", "advice")
        self.summary_mode_combo.setToolTip(
            "Summary: standard summary\n"
            "Logical: mechanism/logic explained with structured points\n"
            "Advice: actionable advice based on the document")
        toolbar.addWidget(self.summary_mode_combo)
        # Pipeline mode indicator
        self.pipeline_badge = QLabel("")
        self.pipeline_badge.setObjectName("pipeline_badge")
        self.pipeline_badge.setVisible(False)
        toolbar.addWidget(self.pipeline_badge)

        root.addLayout(toolbar)

        row = QHBoxLayout(); row.setSpacing(10)
        self.input = QTextEdit()
        self.input.setPlaceholderText("Ask anything…   ↵ send   ⇧↵ newline")
        self.input.setMaximumHeight(120)
        self.input.setMinimumHeight(58)
        self.input.installEventFilter(self)

        self.send_btn = QPushButton("Send ➤")
        self.send_btn.setObjectName("btn_send")
        self.send_btn.setFixedSize(90, 52)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._emit_send)

        self.btn_pipeline_run = QPushButton("🔗  Pipeline")
        self.btn_pipeline_run.setObjectName("btn_pipeline")
        self.btn_pipeline_run.setFixedHeight(36)
        self.btn_pipeline_run.setToolTip("Run a saved pipeline on your current input")
        self.btn_pipeline_run.clicked.connect(self.pipeline_run_requested)
        self.btn_pipeline_run.setStyleSheet(
            f"QPushButton{{background:transparent;"
            f"color:{C['pipeline']};border:1px solid {C['pipeline']};"
            f"border-radius:6px;font-size:11px;font-weight:600;}}"
            f"QPushButton:hover{{background:{C['pipeline']};color:#fff;}}")

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("btn_stop")
        self.stop_btn.setFixedSize(90, 52)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self.stop_requested)

        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        row.addWidget(self.btn_pipeline_run)
        row.addWidget(self.stop_btn)
        root.addLayout(row)
        self.setLayout(root)
        self._update_family_badge()

    def _populate_models(self):
        self.model_combo.clear()
        for m in MODEL_REGISTRY.all_models():
            self.model_combo.addItem(m["name"], m["path"])
        if self.model_combo.count() == 0:
            self.model_combo.addItem(DEFAULT_MODEL, str(MODELS_DIR / DEFAULT_MODEL))

    def _update_family_badge(self):
        path = self.model_combo.currentData() or ""
        if path:
            fam   = detect_model_family(path)
            quant = detect_quant_type(path)
            ql, _ = quant_info(quant)
            self.family_badge.setText(f"{fam.name}  ·  {quant}  ·  {ql}")
        else:
            self.family_badge.setText("")

    def _on_model_combo_changed(self, _index: int):
        # Walk up to MainWindow and trigger primary model reload
        w = self.parent()
        while w and not hasattr(w, "_start_model_load"):
            w = w.parent()
        if w and hasattr(w, "_start_model_load"):
            # Small delay so UI settles before reload
            QTimer.singleShot(300, w._start_model_load)

    def set_pipeline_mode(self, active: bool):
        active = bool(active)
        if not hasattr(self, "pipeline_badge") or self.pipeline_badge is None:
            return
        self.pipeline_badge.setVisible(active)
        if active:
            self.pipeline_badge.setText("🔗 Pipeline Mode")

    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QEvent.Type.KeyPress:
            if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                    and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
                self._emit_send()
                return True
        return super().eventFilter(obj, event)

    def _emit_send(self):
        text = self.input.toPlainText().strip()
        if text:
            self.input.clear()
            self.send_requested.emit(text)

    def set_generating(self, active: bool):
        self.send_btn.setVisible(not active)
        self.stop_btn.setVisible(active)
        self.input.setEnabled(not active)

    @property
    def selected_model(self) -> str:
        return self.model_combo.currentData() or ""

    @property
    def summary_mode(self) -> str:
        return self.summary_mode_combo.currentData() or "summary"
