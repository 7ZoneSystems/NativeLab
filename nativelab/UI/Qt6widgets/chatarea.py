from __future__ import annotations
from nativelab.imports.import_global import TYPE_CHECKING, Qt, Any, QFrame, List, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTimer, QProgressBar, QComboBox
from nativelab.UI.icons import set_button_icon, set_status_label

if TYPE_CHECKING:
    from nativelab.UI.UI_global import MessageWidget, ThinkingBlock

_AUTO_SETUP_WIDGET_ATTRS = (
    "_auto_setup_banner",
    "_auto_setup_status",
    "_auto_setup_progress",
    "_auto_setup_backend_combo",
    "_auto_setup_yes_btn",
    "_auto_setup_no_btn",
    "_auto_setup_pause_btn",
    "_auto_setup_resume_btn",
    "_auto_setup_cancel_btn",
)


def _qt_object_deleted(widget: object | None) -> bool:
    if widget is None:
        return True
    for module_name in ("PyQt6.sip", "sip"):
        try:
            module = __import__(module_name, fromlist=["isdeleted"])
            isdeleted = getattr(module, "isdeleted", None)
            if callable(isdeleted):
                return bool(isdeleted(widget))
        except Exception:
            continue
    return False


def _palette_value(colors: dict, key: str, fallback: str) -> str:
    return str(colors.get(key) or fallback)


def _ui():
    from nativelab.UI import UI_global
    return UI_global


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
        banner_bg = _palette_value(c, "surface", _palette_value(c, "bg1", _palette_value(c, "bg0", "#ffffff")))
        banner.setStyleSheet(
            f"QFrame#auto_setup_banner{{background:{banner_bg};"
            f"border:1px solid {_palette_value(c, 'bdr', '#d0d0d0')};border-radius:8px;margin-top:10px;}}"
        )
        root = QVBoxLayout(banner)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)

        title = QLabel("You seem new here. May I auto setup a minimal working interface for you?")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        title.setStyleSheet(f"color:{_palette_value(c, 'txt', '#111111')};font-size:13px;font-weight:600;")
        root.addWidget(title)

        status = QLabel("NativeLab will pick a backend, a small model, and safe defaults for this machine.")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setWordWrap(True)
        status.setStyleSheet(f"color:{_palette_value(c, 'txt2', '#666666')};font-size:11px;")
        root.addWidget(status)

        backend_row = QHBoxLayout()
        backend_row.setSpacing(8)
        backend_row.addStretch()
        backend_lbl = QLabel("Backend:")
        backend_lbl.setStyleSheet(f"color:{_palette_value(c, 'txt2', '#666666')};font-size:11px;")
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
            f"QProgressBar{{background:{_palette_value(c, 'bg2', '#eeeeee')};border:1px solid {_palette_value(c, 'bdr', '#d0d0d0')};"
            f"border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{_palette_value(c, 'acc', '#695ceb')};border-radius:4px;}}"
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
        yes_btn.clicked.connect(lambda _checked=False: on_yes_cb())

        no_btn = QPushButton("No")
        no_btn.setObjectName("btn_stop")
        no_btn.setFixedHeight(30)
        no_btn.setMinimumWidth(76)
        set_button_icon(no_btn, "circle-x", "No")
        no_btn.clicked.connect(lambda _checked=False: on_no_cb())

        pause_btn = QPushButton("Pause")
        pause_btn.setFixedHeight(30)
        pause_btn.setMinimumWidth(86)
        pause_btn.setVisible(False)
        set_button_icon(pause_btn, "circle-pause", "Pause")
        pause_btn.clicked.connect(lambda _checked=False: on_pause_cb())

        resume_btn = QPushButton("Resume")
        resume_btn.setObjectName("btn_send")
        resume_btn.setFixedHeight(30)
        resume_btn.setMinimumWidth(92)
        resume_btn.setVisible(False)
        set_button_icon(resume_btn, "play", "Resume")
        resume_btn.clicked.connect(lambda _checked=False: on_resume_cb())

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("btn_stop")
        cancel_btn.setFixedHeight(30)
        cancel_btn.setMinimumWidth(92)
        cancel_btn.setVisible(False)
        set_button_icon(cancel_btn, "stop-circle", "Cancel")
        cancel_btn.clicked.connect(lambda _checked=False: on_cancel_cb())

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

    def _clear_auto_setup_refs(self):
        for name in _AUTO_SETUP_WIDGET_ATTRS:
            setattr(self, name, None)

    def _auto_setup_widget(self, name: str):
        widget = getattr(self, name, None)
        if _qt_object_deleted(widget):
            setattr(self, name, None)
            return None
        return widget

    def _call_auto_setup_widget(self, name: str, method: str, *args) -> bool:
        widget = self._auto_setup_widget(name)
        if widget is None:
            return False
        try:
            getattr(widget, method)(*args)
            return True
        except RuntimeError:
            setattr(self, name, None)
            return False
        except Exception:
            return False

    def _auto_setup_banner_alive(self) -> bool:
        if self._auto_setup_widget("_auto_setup_banner") is not None:
            return True
        self._clear_auto_setup_refs()
        return False

    def hide_auto_setup_prompt(self):
        banner = self._auto_setup_widget("_auto_setup_banner")
        if banner is not None:
            try:
                self._vbox.removeWidget(banner)
                banner.deleteLater()
            except RuntimeError:
                pass
            except Exception:
                pass
        self._clear_auto_setup_refs()

    def set_auto_setup_running(self, running: bool, *, paused: bool = False):
        if not self._auto_setup_banner_alive():
            return
        self._call_auto_setup_widget("_auto_setup_backend_combo", "setEnabled", not running)
        for name in ("_auto_setup_yes_btn", "_auto_setup_no_btn"):
            self._call_auto_setup_widget(name, "setVisible", not running)
        self._call_auto_setup_widget("_auto_setup_pause_btn", "setVisible", running and not paused)
        self._call_auto_setup_widget("_auto_setup_resume_btn", "setVisible", running and paused)
        self._call_auto_setup_widget("_auto_setup_cancel_btn", "setVisible", running)
        self._call_auto_setup_widget("_auto_setup_progress", "setVisible", running)

    def update_auto_setup_status(self, text: str = "", done: int = 0, total: int = 0):
        if not self._auto_setup_banner_alive():
            return
        if text:
            self._call_auto_setup_widget("_auto_setup_status", "setText", text)
        if total:
            pct = max(0, min(100, int(done * 100 / total)))
            if self._call_auto_setup_widget("_auto_setup_progress", "setRange", 0, 100):
                self._call_auto_setup_widget("_auto_setup_progress", "setValue", pct)
                self._call_auto_setup_widget("_auto_setup_progress", "setVisible", True)

    def selected_auto_setup_backend(self) -> str:
        combo = self._auto_setup_widget("_auto_setup_backend_combo")
        if combo is None:
            return "llama_cpp"
        try:
            return str(combo.currentData() or "llama_cpp")
        except RuntimeError:
            self._auto_setup_backend_combo = None
            return "llama_cpp"
        except Exception:
            return "llama_cpp"

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
