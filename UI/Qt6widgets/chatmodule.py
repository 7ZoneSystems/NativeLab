from imports.import_global import HAS_PDF, Qt, PdfReader, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSplitter, QPropertyAnimation, QEasingCurve, pyqtSignal
from UI.UI_global import C, get_ref_store, ram_free_mb
from UI.Qt6widgets.refrencepanels import ReferencePanelV2
from GlobalConfig.config_global import RamWatchdog
class ChatModule(QWidget):
    """
    Self-contained chat module: ChatArea + ReferencePanel + InputBar.
    Modular — can be embedded anywhere or swapped out.
    """

    send_requested  = pyqtSignal(str, str)   # (text, ref_context)
    stop_requested  = pyqtSignal()
    pdf_requested   = pyqtSignal()
    clear_requested = pyqtSignal()
    multi_pdf_requested = pyqtSignal(list)   # list of paths

    def __init__(self, session_id: str = "", parent=None):
        super().__init__(parent)
        self._session_id  = session_id
        self._refs_visible = False

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Top bar with ref toggle
        topbar = QHBoxLayout()
        topbar.setContentsMargins(8, 4, 8, 0)
        topbar.setSpacing(6)
        topbar.addStretch()
        self.ref_toggle_btn = QPushButton("📎  References  (0)")
        self.ref_toggle_btn.setFixedHeight(28)
        self.ref_toggle_btn.setCheckable(True)
        self.ref_toggle_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{C['txt3']};"
            f"border:1px solid {C['bdr']};border-radius:6px;"
            f"font-size:11px;padding:0 13px;font-weight:500;letter-spacing:0.1px;}}"
            f"QPushButton:checked{{background:rgba(105,92,235,0.14);color:{C['acc2']};"
            f"border-color:rgba(105,92,235,0.36);font-weight:600;}}"
            f"QPushButton:hover{{color:{C['txt']};border-color:{C['bdr2']};}}"
        ) 
        self.ref_toggle_btn.clicked.connect(self._toggle_refs)
        topbar.addWidget(self.ref_toggle_btn)
        root.addLayout(topbar)

        # Main splitter: chat | ref panel
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(2)

        self.chat_area = ChatArea()
        self.main_splitter.addWidget(self.chat_area)

        self.ref_panel = ReferencePanelV2(
            session_id or "default",
            get_ref_store_fn = get_ref_store,
            ram_watchdog_fn  = lambda sid: RamWatchdog.check_and_spill(sid),
            ram_mb_fn   = ram_free_mb,
            has_pdf          = HAS_PDF,
            pdf_reader_cls   = PdfReader if HAS_PDF else None,
        )
        self.ref_panel.refs_changed.connect(self._on_refs_changed)
        self.ref_panel.setVisible(False)
        self.main_splitter.addWidget(self.ref_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)

        root.addWidget(self.main_splitter, 1)

        self.input_bar = InputBar()
        self.input_bar.send_requested.connect(self._on_send)
        self.input_bar.stop_requested.connect(self.stop_requested)
        self.input_bar.pdf_requested.connect(self.pdf_requested)
        self.input_bar.clear_requested.connect(self.clear_requested)
        root.addWidget(self.input_bar, 0)

        self.setLayout(root)

    def set_session(self, session_id: str):
        self._session_id = session_id
        self.ref_panel.update_session(session_id)
        self._update_ref_badge()

    def _toggle_refs(self, checked: bool):
        self._refs_visible = checked
        TARGET_W = 260
        if checked:
            self.ref_panel.setVisible(True)
            self.ref_panel.setMaximumWidth(0)
            self.main_splitter.setSizes([800, TARGET_W])
            anim = QPropertyAnimation(self.ref_panel, b"maximumWidth", self)
            anim.setDuration(240)
            anim.setStartValue(0)
            anim.setEndValue(TARGET_W)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.finished.connect(lambda: self.ref_panel.setMaximumWidth(16777215))
            anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        else:
            anim = QPropertyAnimation(self.ref_panel, b"maximumWidth", self)
            anim.setDuration(200)
            anim.setStartValue(self.ref_panel.width())
            anim.setEndValue(0)
            anim.setEasingCurve(QEasingCurve.Type.InCubic)
            anim.finished.connect(lambda: (
                self.ref_panel.setVisible(False),
                self.ref_panel.setMaximumWidth(16777215),
                self.main_splitter.setSizes([1, 0])
            ))
            anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _on_send(self, text: str):
        ref_ctx = self.ref_panel.get_context_for(text)
        self.send_requested.emit(text, ref_ctx)

    def _on_refs_changed(self):
        self._update_ref_badge()
        # Check if multi-PDF was triggered
        if hasattr(self.ref_panel, "_pending_multi_pdfs"):
            paths = self.ref_panel._pending_multi_pdfs
            del self.ref_panel._pending_multi_pdfs
            self.multi_pdf_requested.emit(paths)

    def _update_ref_badge(self):
        n = len(self.ref_panel._store.refs)
        self.ref_toggle_btn.setText(f"📎  References  ({n})")

    # Proxy properties
    @property
    def selected_model(self) -> str:
        return self.input_bar.selected_model

    def set_generating(self, active: bool):
        self.input_bar.set_generating(active)

    def set_pipeline_mode(self, active: bool):
        self.input_bar.set_pipeline_mode(active)
