from __future__ import annotations
from nativelab.imports.import_global import TYPE_CHECKING, Qt, Any, QFrame, List, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTimer, QProgressBar, QComboBox
from .chatmodule import _ui
from nativelab.UI.UI_const import C
from nativelab.UI.icons import set_button_icon, set_status_label

if TYPE_CHECKING:
    from nativelab.UI.UI_global import MessageWidget, ThinkingBlock
class PauseBanner(QFrame):
    status_lbl: QLabel
    spinner:    QLabel
class ChatArea(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setObjectName("chat_scroll")
        self._container = QWidget()
        self._container.setObjectName("chat_container")
        self._vbox = QVBoxLayout()
        self._vbox.setContentsMargins(0, 12, 0, 12)
        self._vbox.setSpacing(2)

        # ── empty-state placeholder ───────────────────────────────────────────
        self._placeholder = QLabel("Hi, message me up\nwhen you are ready.")
        self._placeholder.setObjectName("txt2_XXL")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._vbox.addStretch()
        self._vbox.addWidget(self._placeholder, 0, Qt.AlignmentFlag.AlignCenter)
        self._auto_setup_banner = None
        self._auto_setup_status = None
        self._auto_setup_progress = None
        self._auto_setup_backend_combo = None
        self._auto_setup_yes_btn = None
        self._auto_setup_no_btn = None
        self._auto_setup_pause_btn = None
        self._auto_setup_resume_btn = None
        self._auto_setup_cancel_btn = None
        self._vbox.addStretch()

        self._container.setLayout(self._vbox)
        self.setWidget(self._container)
        self._widgets: List[QWidget] = []

    def _show_placeholder(self, visible: bool):
        self._placeholder.setVisible(visible)
        if visible:
            _ui().fade_in(self._placeholder, 300)

    def add_message(self, role: str, content: str, timestamp: str,
                    tag: str = "") -> Any:
        w = _ui().MessageWidget(role, content, timestamp, tag=tag)
        self._vbox.insertWidget(self._vbox.count() - 1, w)
        self._widgets.append(w)
        _ui().fade_in(w, 220)
        self._show_placeholder(False)
        QTimer.singleShot(30, self._scroll_bottom)
        return w

    def add_thinking_block(self, total_chunks: int) -> Any:
        tb = _ui().ThinkingBlock(total_chunks)
        self._vbox.insertWidget(self._vbox.count() - 1, tb)
        self._widgets.append(tb)
        QTimer.singleShot(30, self._scroll_bottom)
        return tb

    def add_pipeline_divider(self, label: str):
        """Insert a visual pipeline stage divider."""
        lbl = QLabel(f"  ⬇  {label}  ⬇")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{_ui().C['pipeline']};font-size:11px;padding:4px 0;"
            f"background:rgba(34,211,238,0.06);border-radius:6px;margin:2px 60px;"
        )
        self._vbox.insertWidget(self._vbox.count() - 1, lbl)
        self._widgets.append(lbl)

    def clear_messages(self):
        for w in self._widgets:
            self._vbox.removeWidget(w)
            w.deleteLater()
        self._widgets.clear()
        self._show_placeholder(True)

    def show_auto_setup_prompt(
        self,
        on_yes_cb,
        on_no_cb,
        on_pause_cb,
        on_resume_cb,
        on_cancel_cb,
        *,
        resume: bool = False,
        default_backend: str = "llama_cpp",
        backend_locked: bool = False,
    ):
        """Show first-run local model setup controls under the empty chat text."""
        self.hide_auto_setup_prompt()
        c = _ui().C
        banner = QFrame()
        banner.setObjectName("auto_setup_banner")
        banner.setMaximumWidth(620)
        banner.setStyleSheet(
            f"QFrame#auto_setup_banner{{background:{c['bg']};"
            f"border:1px solid {c['bdr']};border-radius:8px;margin-top:10px;}}"
        )
        root = QVBoxLayout(banner)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)

        title = QLabel("You seem new here. May I auto setup a minimal working interface for you?")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        title.setStyleSheet(f"color:{c['txt']};font-size:13px;font-weight:600;")
        root.addWidget(title)

        status = QLabel("NativeLab will pick a backend, a small model, and safe defaults for this machine.")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setWordWrap(True)
        status.setStyleSheet(f"color:{c['txt2']};font-size:11px;")
        root.addWidget(status)

        backend_row = QHBoxLayout()
        backend_row.setSpacing(8)
        backend_row.addStretch()
        backend_lbl = QLabel("Backend:")
        backend_lbl.setStyleSheet(f"color:{c['txt2']};font-size:11px;")
        backend_combo = QComboBox()
        backend_combo.setFixedHeight(28)
        backend_combo.setMinimumWidth(210)
        backend_combo.addItem("llama.cpp (easy)", "llama_cpp")
        backend_combo.addItem("Hugging Face (hard)", "hf_transformers")
        idx = backend_combo.findData(default_backend)
        backend_combo.setCurrentIndex(max(0, idx))
        backend_combo.setEnabled(not backend_locked)
        backend_combo.setToolTip(
            "llama.cpp downloads a GGUF model and runtime. Hugging Face downloads a full Transformers snapshot and needs the HF backend dependencies."
        )
        backend_row.addWidget(backend_lbl)
        backend_row.addWidget(backend_combo)
        backend_row.addStretch()
        root.addLayout(backend_row)

        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFixedHeight(8)
        progress.setTextVisible(False)
        progress.setVisible(False)
        progress.setStyleSheet(
            f"QProgressBar{{background:{c['bg2']};border:1px solid {c['bdr']};"
            f"border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{c['acc']};border-radius:4px;}}"
        )
        root.addWidget(progress)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch()

        yes_btn = QPushButton("Resume Setup" if resume else "Yes")
        yes_btn.setObjectName("btn_send")
        yes_btn.setFixedHeight(30)
        yes_btn.setMinimumWidth(90)
        set_button_icon(yes_btn, "play", yes_btn.text())
        yes_btn.clicked.connect(on_yes_cb)

        no_btn = QPushButton("No")
        no_btn.setObjectName("btn_stop")
        no_btn.setFixedHeight(30)
        no_btn.setMinimumWidth(76)
        set_button_icon(no_btn, "circle-x", "No")
        no_btn.clicked.connect(on_no_cb)

        pause_btn = QPushButton("Pause")
        pause_btn.setFixedHeight(30)
        pause_btn.setMinimumWidth(86)
        pause_btn.setVisible(False)
        set_button_icon(pause_btn, "circle-pause", "Pause")
        pause_btn.clicked.connect(on_pause_cb)

        resume_btn = QPushButton("Resume")
        resume_btn.setObjectName("btn_send")
        resume_btn.setFixedHeight(30)
        resume_btn.setMinimumWidth(92)
        resume_btn.setVisible(False)
        set_button_icon(resume_btn, "play", "Resume")
        resume_btn.clicked.connect(on_resume_cb)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("btn_stop")
        cancel_btn.setFixedHeight(30)
        cancel_btn.setMinimumWidth(92)
        cancel_btn.setVisible(False)
        set_button_icon(cancel_btn, "stop-circle", "Cancel")
        cancel_btn.clicked.connect(on_cancel_cb)

        for button in (yes_btn, no_btn, pause_btn, resume_btn, cancel_btn):
            row.addWidget(button)
        row.addStretch()
        root.addLayout(row)

        idx = self._vbox.indexOf(self._placeholder)
        self._vbox.insertWidget(max(0, idx + 1), banner, 0, Qt.AlignmentFlag.AlignCenter)
        self._auto_setup_banner = banner
        self._auto_setup_status = status
        self._auto_setup_progress = progress
        self._auto_setup_backend_combo = backend_combo
        self._auto_setup_yes_btn = yes_btn
        self._auto_setup_no_btn = no_btn
        self._auto_setup_pause_btn = pause_btn
        self._auto_setup_resume_btn = resume_btn
        self._auto_setup_cancel_btn = cancel_btn
        _ui().fade_in(banner, 220)

    def hide_auto_setup_prompt(self):
        banner = getattr(self, "_auto_setup_banner", None)
        if banner is not None:
            try:
                self._vbox.removeWidget(banner)
                banner.deleteLater()
            except RuntimeError:
                pass
        self._auto_setup_banner = None
        self._auto_setup_status = None
        self._auto_setup_progress = None
        self._auto_setup_backend_combo = None

    def set_auto_setup_running(self, running: bool, *, paused: bool = False):
        combo = getattr(self, "_auto_setup_backend_combo", None)
        if combo is not None:
            combo.setEnabled(not running)
        for name in ("_auto_setup_yes_btn", "_auto_setup_no_btn"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setVisible(not running)
        pause_btn = getattr(self, "_auto_setup_pause_btn", None)
        resume_btn = getattr(self, "_auto_setup_resume_btn", None)
        cancel_btn = getattr(self, "_auto_setup_cancel_btn", None)
        if pause_btn is not None:
            pause_btn.setVisible(running and not paused)
        if resume_btn is not None:
            resume_btn.setVisible(running and paused)
        if cancel_btn is not None:
            cancel_btn.setVisible(running)
        progress = getattr(self, "_auto_setup_progress", None)
        if progress is not None:
            progress.setVisible(running)

    def update_auto_setup_status(self, text: str = "", done: int = 0, total: int = 0):
        status = getattr(self, "_auto_setup_status", None)
        if status is not None and text:
            status.setText(text)
        progress = getattr(self, "_auto_setup_progress", None)
        if progress is not None and total:
            progress.setRange(0, 100)
            progress.setValue(max(0, min(100, int(done * 100 / total))))
            progress.setVisible(True)

    def selected_auto_setup_backend(self) -> str:
        combo = getattr(self, "_auto_setup_backend_combo", None)
        if combo is None:
            return "llama_cpp"
        return str(combo.currentData() or "llama_cpp")

    def _scroll_bottom(self):
        try:
            sb = self.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())
        except RuntimeError:
            pass  # widget destroyed during teardown
        except Exception:
            pass

    def add_pause_banner(self, on_pause_cb, on_abort_cb) -> QWidget:
        """Add a live pause/abort control bar during summarization."""
        banner = PauseBanner()
        banner.setObjectName("pause_banner")
        banner.setStyleSheet(
            f"QFrame#pause_banner{{background:rgba(251,191,36,0.08);"
            f"border:1px solid rgba(251,191,36,0.3);border-radius:10px;"
            f"margin:4px 12px;}}")
        bl = QHBoxLayout()
        bl.setContentsMargins(14, 8, 14, 8); bl.setSpacing(10)

        spinner = QLabel("")
        set_status_label(spinner, "", "loading", 15)
        spinner.setStyleSheet(f"color:{_ui().C['warn']};font-size:13px;")
        status_lbl = QLabel("Summarizing... click Pause to save state and stop.")
        status_lbl.setStyleSheet(f"color:{_ui().C['warn']};font-size:11px;")

        pause_btn = QPushButton("Pause & Save")
        set_button_icon(pause_btn, "circle-pause", "Pause & Save")
        pause_btn.setFixedHeight(28)
        pause_btn.setFixedWidth(120)
        pause_btn.setStyleSheet(
            f"QPushButton{{background:rgba(251,191,36,0.15);color:{_ui().C['warn']};"
            f"border:1px solid rgba(251,191,36,0.4);border-radius:7px;"
            f"font-size:11px;font-weight:600;}}"
            f"QPushButton:hover{{background:rgba(251,191,36,0.3);}}")
        pause_btn.clicked.connect(on_pause_cb)

        abort_btn = QPushButton("Abort")
        set_button_icon(abort_btn, "stop-circle", "Abort")
        abort_btn.setFixedHeight(28)
        abort_btn.setFixedWidth(80)
        abort_btn.setStyleSheet(
            f"QPushButton{{background:rgba(248,113,113,0.12);color:{_ui().C['err']};"
            f"border:1px solid rgba(248,113,113,0.3);border-radius:7px;"
            f"font-size:11px;font-weight:600;}}"
            f"QPushButton:hover{{background:rgba(248,113,113,0.3);}}")
        abort_btn.clicked.connect(on_abort_cb)

        bl.addWidget(spinner)
        bl.addWidget(status_lbl, 1)
        bl.addWidget(pause_btn)
        bl.addWidget(abort_btn)
        banner.setLayout(bl)
        banner.status_lbl = status_lbl
        banner.spinner    = spinner

        self._vbox.insertWidget(self._vbox.count() - 1, banner)
        self._widgets.append(banner)
        QTimer.singleShot(30, self._scroll_bottom)
        return banner

    def remove_pause_banner(self):
        """Remove the pause control bar."""
        for w in list(self._widgets):
            if isinstance(w, PauseBanner) and w.objectName() == "pause_banner":
                self._vbox.removeWidget(w)
                w.deleteLater()
                self._widgets.remove(w)
                break
