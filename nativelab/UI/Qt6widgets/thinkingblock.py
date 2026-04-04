from imports.import_global import List, QTextEdit, QFont, Optional, Qt, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QPushButton, QSizePolicy, QTextCursor, QTimer, QApplication

class ThinkingBlock(QWidget):
    def __init__(self, total_chunks: int):
        super().__init__()
        self._total   = total_chunks
        self._done    = 0
        self._entries: List[str] = []

        root = QVBoxLayout()
        root.setContentsMargins(16, 4, 16, 4)
        root.setSpacing(0)

        self._toggle_btn = QPushButton(f"  🧠  Reasoning…   0 / {total_chunks} steps")
        self._toggle_btn.setObjectName("thinking_toggle")
        self._toggle_btn.clicked.connect(self._toggle)
        root.addWidget(self._toggle_btn)

        self._content_frame = QFrame()
        self._content_frame.setObjectName("thinking_frame")
        cf_layout = QVBoxLayout()
        cf_layout.setContentsMargins(0, 0, 0, 0)
        cf_layout.setSpacing(0)

        self._te = QTextEdit()
        self._te.setReadOnly(True)
        self._te.setFont(QFont("Consolas", 10))
        self._te.setFixedHeight(180)
        self._te.setObjectName("thinking_te")
        cf_layout.addWidget(self._te)
        self._content_frame.setLayout(cf_layout)
        self._content_frame.setVisible(False)
        root.addWidget(self._content_frame)
        self.setLayout(root)

    def add_section(self, num: int, total: int, chunk_text: str, summary: str):
        self._done = num
        preview  = chunk_text[:300].replace("\n", " ")
        if len(chunk_text) > 300: preview += "…"
        entry = (
            f"─── Section {num} / {total} ───\n"
            f"  📥 Input preview:  {preview}\n"
            f"  📝 Summary:        {summary[:400].strip()}"
            + ("…" if len(summary) > 400 else "") + "\n"
        )
        self._entries.append(entry)
        self._te.append(entry)
        self._te.moveCursor(QTextCursor.MoveOperation.End)
        self._toggle_btn.setText(f"▶  🧠 Thinking…  ({num} / {total} sections complete)")

    def add_phase(self, label: str):
        from UI.buildUI import C       
        msg = f"\n═══ {label} ═══\n"
        self._entries.append(msg)
        self._te.append(f'<span style="color:{C["acc"]}">{msg}</span>')

    def mark_done(self):
        try:
            from UI.buildUI import C
            self._toggle_btn.setText(
                f"▼  ✅  Done  —  {self._done} / {self._total} steps")
            self._toggle_btn.setStyleSheet(
                f"QPushButton{{background:rgba(28,184,138,0.10);color:{C['ok']};"
                f"border:1px solid rgba(28,184,138,0.24);border-radius:8px;"
                f"padding:7px 16px;text-align:left;font-size:12px;font-weight:600;}}"
                f"QPushButton:hover{{background:rgba(28,184,138,0.18);}}"
            )
            self._content_frame.setVisible(True)
        except RuntimeError:
            pass  # widget already destroyed)

    def _toggle(self):
        vis = not self._content_frame.isVisible()
        self._content_frame.setVisible(vis)
        label = self._toggle_btn.text()
        if label.startswith("▶"):
            self._toggle_btn.setText(label.replace("▶", "▼", 1))
        else:
            self._toggle_btn.setText(label.replace("▼", "▶", 1))
