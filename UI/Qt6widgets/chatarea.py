from imports.import_global import Qt, QFrame, List, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTimer
from UI.UI_global import C, ThinkingBlock, fade_in, MessageWidget

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
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color:{C['txt3']};font-size:22px;font-weight:300;"
            f"letter-spacing:0.4px;padding:60px 40px;")
        self._vbox.addStretch()
        self._vbox.addWidget(self._placeholder, 0, Qt.AlignmentFlag.AlignCenter)
        self._vbox.addStretch()

        self._container.setLayout(self._vbox)
        self.setWidget(self._container)
        self._widgets: List[QWidget] = []

    def _show_placeholder(self, visible: bool):
        self._placeholder.setVisible(visible)
        if visible:
            fade_in(self._placeholder, 300)

    def add_message(self, role: str, content: str, timestamp: str,
                    tag: str = "") -> MessageWidget:
        w = MessageWidget(role, content, timestamp, tag=tag)
        self._vbox.insertWidget(self._vbox.count() - 1, w)
        self._widgets.append(w)
        fade_in(w, 220)
        self._show_placeholder(False)
        QTimer.singleShot(30, self._scroll_bottom)
        return w

    def add_thinking_block(self, total_chunks: int) -> ThinkingBlock:
        tb = ThinkingBlock(total_chunks)
        self._vbox.insertWidget(self._vbox.count() - 1, tb)
        self._widgets.append(tb)
        QTimer.singleShot(30, self._scroll_bottom)
        return tb

    def add_pipeline_divider(self, label: str):
        """Insert a visual pipeline stage divider."""
        lbl = QLabel(f"  ⬇  {label}  ⬇")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"color:{C['pipeline']};font-size:11px;padding:4px 0;"
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
        banner = QFrame()
        banner.setObjectName("pause_banner")
        banner.setStyleSheet(
            f"QFrame#pause_banner{{background:rgba(251,191,36,0.08);"
            f"border:1px solid rgba(251,191,36,0.3);border-radius:10px;"
            f"margin:4px 12px;}}")
        bl = QHBoxLayout()
        bl.setContentsMargins(14, 8, 14, 8); bl.setSpacing(10)

        spinner = QLabel("⏳")
        spinner.setStyleSheet(f"color:{C['warn']};font-size:13px;")
        status_lbl = QLabel("Summarizing… click Pause to save state and stop.")
        status_lbl.setStyleSheet(f"color:{C['warn']};font-size:11px;")

        pause_btn = QPushButton("⏸  Pause & Save")
        pause_btn.setFixedHeight(28)
        pause_btn.setFixedWidth(120)
        pause_btn.setStyleSheet(
            f"QPushButton{{background:rgba(251,191,36,0.15);color:{C['warn']};"
            f"border:1px solid rgba(251,191,36,0.4);border-radius:7px;"
            f"font-size:11px;font-weight:600;}}"
            f"QPushButton:hover{{background:rgba(251,191,36,0.3);}}")
        pause_btn.clicked.connect(on_pause_cb)

        abort_btn = QPushButton("⏹  Abort")
        abort_btn.setFixedHeight(28)
        abort_btn.setFixedWidth(80)
        abort_btn.setStyleSheet(
            f"QPushButton{{background:rgba(248,113,113,0.12);color:{C['err']};"
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
            if isinstance(w, QFrame) and w.objectName() == "pause_banner":
                self._vbox.removeWidget(w)
                w.deleteLater()
                self._widgets.remove(w)
                break
