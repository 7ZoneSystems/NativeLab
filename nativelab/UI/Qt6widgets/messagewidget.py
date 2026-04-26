from nativelab.imports.import_global import Optional, Qt, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QPushButton, QSizePolicy, QTextCursor, QTimer, QApplication
class MessageWidget(QWidget):
    _COLLAPSE_PX = 260

    def __init__(self, role: str, content: str, timestamp: str, tag: str = ""):
        super().__init__()
        self.role         = role
        self._text        = content
        self._collapsed   = False
        self._collapsible = False
        self._pending_txt = ""
        self._tag         = tag

        outer = QHBoxLayout()
        outer.setContentsMargins(12, 4, 12, 4)
        outer.setSpacing(0)

        self.bubble = QFrame()
        if role == "user":
            self.bubble.setObjectName("bubble_user")
        elif role == "system_note":
            self.bubble.setObjectName("bubble_ast")
        elif role == "pipeline_intermediate":
            self.bubble.setObjectName("bubble_rsn")
        elif tag == "🧠 Reasoning":
            self.bubble.setObjectName("bubble_rsn")
        elif "Coding" in (tag or ""):
            self.bubble.setObjectName("bubble_cod")
        else:
            self.bubble.setObjectName("bubble_ast")
        self.bubble.setMaximumWidth(860)
        self.bubble.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        bl = QVBoxLayout()
        bl.setContentsMargins(14, 10, 14, 10)
        bl.setSpacing(6)

        # ── header row ───────────────────────────────────────────────────────
        from nativelab.UI.buildUI import C
        hdr = QHBoxLayout(); hdr.setSpacing(8)
        name_text  = "You" if role == "user" else (
                     "⚡ System" if role == "system_note" else (
                     "◈ Intermediate" if role == "pipeline_intermediate" else (
                     tag or "Assistant")))
        name_color = (C["acc"] if role == "user" else
                      (C["txt2"] if role == "system_note" else
                       (C["warn"] if role == "pipeline_intermediate" else
                        (C["pipeline"] if tag == "🧠 Reasoning" else C["ok"]))))
        name_lbl   = QLabel(name_text)
        name_lbl.setStyleSheet(
            f"color:{name_color};font-weight:700;font-size:11px;letter-spacing:0.3px;")
        ts_lbl = QLabel(timestamp)
        ts_lbl.setStyleSheet(f"color:{C['txt2']};font-size:10px;")

        self._copy_btn = QPushButton("⧉")
        self._copy_btn.setFixedSize(26, 26)
        self._copy_btn.setToolTip("Copy message")
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{C['txt3']};"
            f"border:1px solid transparent;border-radius:6px;"
            f"font-size:13px;padding:0;font-weight:400;}}"
            f"QPushButton:hover{{background:rgba(105,92,235,0.16);"
            f"color:{C['acc2']};border-color:rgba(105,92,235,0.28);}}"
            f"QPushButton:pressed{{background:rgba(105,92,235,0.28);}}")
        self._copy_btn.clicked.connect(self._copy_all)

        hdr.addWidget(name_lbl)
        hdr.addStretch()
        hdr.addWidget(ts_lbl)
        hdr.addWidget(self._copy_btn)

        # ── body (RichTextEdit) ───────────────────────────────────────────────
        from nativelab.UI.RichTextEditor import RichTextEdit
        self.te = RichTextEdit()
        self.te.setObjectName("bubble_te")
        self.te.setFrameShape(QFrame.Shape.NoFrame)
        self.te.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.te.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        from nativelab.UI.buildUI import C as _C
        from PyQt6.QtGui import QPalette, QColor as _QC
        _bg_key = ("usr" if role == "user"
                   else "rsn" if (role == "pipeline_intermediate" or tag == "🧠 Reasoning")
                   else "cod" if "Coding" in (tag or "")
                   else "ast")
        _pal = self.te.palette()
        _pal.setColor(QPalette.ColorRole.Base,   _QC(_C[_bg_key]))
        _pal.setColor(QPalette.ColorRole.Window, _QC(_C[_bg_key]))
        _pal.setColor(QPalette.ColorRole.Text,   _QC(_C["txt"]))
        self.te.setPalette(_pal)
        self.te.setAutoFillBackground(True)
        self.te.setStyleSheet("QTextBrowser{border:none;}")
        self.te.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        doc = self.te.document()
        if doc is not None:
            doc.contentsChanged.connect(self._fit)

        bl.addLayout(hdr)
        bl.addWidget(self.te)

        self._expand_btn: Optional[QPushButton] = None
        self.bubble.setLayout(bl)

        if role == "user":
            outer.addStretch(); outer.addWidget(self.bubble)
        else:
            outer.addWidget(self.bubble); outer.addStretch()

        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.timeout.connect(self._flush_pending)

        self.setLayout(outer)

        if content:
            self._render_html(content)
        else:
            self._fit()

    # ── rendering ─────────────────────────────────────────────────────────────
    def _render_html(self, text: str):
        from nativelab.UI.md_to_html import md_to_html
        from nativelab.UI.buildUI import C
        from PyQt6.QtGui import QColor as _QC
        self.te.setTextColor(_QC(C["txt"]))
        html = md_to_html(text, code_store=self.te._code_blocks, colors=C)
        self.te.setHtml(
            f'<div style="color:{C["txt"]};font-size:13px;'
            f'line-height:1.65;font-family:Segoe UI,Inter,sans-serif;">'
            f'{html}</div>')
        self._text = text
        self._fit()
        QTimer.singleShot(60, self._maybe_add_expander)

    def _maybe_add_expander(self):
        from nativelab.UI.buildUI import C
        doc = self.te.document()
        natural_h = int(doc.size().height()) + 6 if doc else 0
        if natural_h > self._COLLAPSE_PX + 60 and not self._collapsible:
            self._collapsible = True
            self._collapsed   = True
            self.te.setFixedHeight(self._COLLAPSE_PX)
            btn = QPushButton("▼  Show more")
            btn.setStyleSheet(
                f"QPushButton{{background:rgba(124,58,237,0.12);color:{C['acc']};"
                f"border:1px solid rgba(167,139,250,0.2);border-radius:6px;"
                f"padding:3px 12px;font-size:11px;}}"
                f"QPushButton:hover{{background:rgba(124,58,237,0.25);color:{C['acc2']};}}")
            btn.clicked.connect(self._toggle_expand)
            layout = self.bubble.layout()
            if layout is not None:
                layout.addWidget(btn)
            self._expand_btn = btn

    def _toggle_expand(self):
        if self._collapsed:
            self._collapsed = False
            doc = self.te.document()
            h = int(doc.size().height()) + 6 if doc else self._COLLAPSE_PX
            self.te.setFixedHeight(h)
            if self._expand_btn is not None:
                self._expand_btn.setText("▲  Show less")
        else:
            self._collapsed = True
            self.te.setFixedHeight(self._COLLAPSE_PX)
            if self._expand_btn is not None:
                self._expand_btn.setText("▼  Show more")

    def finalize(self):
        raw = self._text or self.te.toPlainText()
        if raw.strip():
            self._render_html(raw)

    def append_text(self, text: str):
        try:
            self._text += text
            self._pending_txt += text
            if not self._flush_timer.isActive():
                self._flush_timer.start(40)
        except RuntimeError:
            pass  # widget destroyed mid-stream; safe to ignore
        except Exception as _e:
            print(f"[append_text] {_e}")

    def _flush_pending(self):
        if not self._pending_txt:
            return
        from nativelab.UI.buildUI import C
        from PyQt6.QtGui import QColor as _QC, QTextCharFormat
        cur = self.te.textCursor()
        fmt = QTextCharFormat()
        fmt.setForeground(_QC(C["txt"]))
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.setCharFormat(fmt)
        cur.insertText(self._pending_txt)
        self._pending_txt = ""
        self.te.setTextCursor(cur)
        self._fit()

    def _fit(self):
        doc = self.te.document()
        h = int(doc.size().height()) + 6 if doc else 22
        self.te.setFixedHeight(max(h, 22))

    def _copy_all(self):
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(self._text)

    @property
    def full_text(self) -> str:
        return self._text if self._text else self.te.toPlainText()
