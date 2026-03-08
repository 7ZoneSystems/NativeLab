from imports.import_global import HAS_PDF, PdfReader, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QMenu, QFileDialog, QColor, QTimer, pyqtSignal, QMessageBox, QTabWidget, QFrame, Qt, Path
from codeparser.codeparser_global import ScriptSmartReference, SessionReferenceStore
from GlobalConfig.config_global import SCRIPT_EXTENSIONS_FILTER, RAM_WATCHDOG_MB, get_ref_store, ram_free_mb
from UI.buildUI import C

class ReferencePanelV2(QWidget):
    """
    Collapsible reference sidebar v2:
      • 📄 Documents tab  — PDF, text, .py plain (unchanged)
      • 💻 Scripts tab    — any source file with full AST parsing
      • Multi-PDF summarize button
    """
    refs_changed = pyqtSignal()

    def __init__(self, session_id: str,
                 # these are injected from the main app context:
                 get_ref_store_fn,
                 ram_watchdog_fn,
                 ram_mb_fn,
                 has_pdf: bool = False,
                 pdf_reader_cls=None,
                 parent=None):
        super().__init__(parent)
        self.session_id       = session_id
        self._get_store       = get_ref_store_fn
        self._ram_watchdog    = ram_watchdog_fn
        self.ram_free_mb     = ram_mb_fn
        self._has_pdf         = has_pdf
        self._PdfReader       = pdf_reader_cls
        self._building        = False

        self.setMaximumWidth(300)
        self.setMinimumWidth(240)
        self.setObjectName("ref_panel_v2")
        root = QVBoxLayout()
        root.setContentsMargins(18, 12, 18, 14)
        root.setSpacing(9)

        # ── Header ───────────────────────────────────────────────────────────
        hdr_row = QHBoxLayout(); hdr_row.setSpacing(6)
        hdr = QLabel("📎  References")
        hdr.setObjectName("ref_hdr")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        self.ram_badge = QLabel("")
        self.ram_badge.setObjectName("ram_badge_lbl")
        hdr_row.addWidget(self.ram_badge)
        root.addLayout(hdr_row)

        # ── Tab widget (Docs / Scripts) ───────────────────────────────────────
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("ref_tabs")
        root.addWidget(self.tab_widget, 1)

        # ── Docs tab ─────────────────────────────────────────────────────────
        docs_widget = QWidget()
        docs_l = QVBoxLayout()
        docs_l.setContentsMargins(6, 6, 6, 6); docs_l.setSpacing(5)

        doc_btn_row = QHBoxLayout(); doc_btn_row.setSpacing(5)
        self.add_pdf_btn = QPushButton("＋ PDF")
        self.add_py_btn  = QPushButton("＋ .py")
        self.add_txt_btn = QPushButton("＋ Text")
        for b in (self.add_pdf_btn, self.add_py_btn, self.add_txt_btn):            
            b.setFixedHeight(24)
            b.setStyleSheet(self._mini_btn_style(C["acc"]))
        self.add_pdf_btn.clicked.connect(lambda: self._add_doc("pdf"))
        self.add_py_btn.clicked.connect(lambda: self._add_doc("python"))
        self.add_txt_btn.clicked.connect(lambda: self._add_doc("text"))
        doc_btn_row.addWidget(self.add_pdf_btn)
        doc_btn_row.addWidget(self.add_py_btn)
        doc_btn_row.addWidget(self.add_txt_btn)
        docs_l.addLayout(doc_btn_row)

        self.multi_pdf_btn = QPushButton("📚  Summarize Multiple PDFs")
        self.multi_pdf_btn.setFixedHeight(26)
        self.multi_pdf_btn.setStyleSheet(self._mini_btn_style(C["ok"]))
        self.multi_pdf_btn.clicked.connect(self._multi_pdf_requested)
        docs_l.addWidget(self.multi_pdf_btn)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        docs_l.addWidget(sep1)

        self.doc_list = QListWidget()
        self.doc_list.setObjectName("ref_doc_list")
        self.doc_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.doc_list.customContextMenuRequested.connect(
            lambda pos: self._ctx_menu(pos, self.doc_list))
        docs_l.addWidget(self.doc_list, 1)
        docs_widget.setLayout(docs_l)
        self.tab_widget.addTab(docs_widget, "📄 Docs")

        # ── Scripts tab ───────────────────────────────────────────────────────
        scripts_widget = QWidget()
        scripts_l = QVBoxLayout()
        scripts_l.setContentsMargins(6, 6, 6, 6); scripts_l.setSpacing(5)

        scr_btn_row = QHBoxLayout(); scr_btn_row.setSpacing(5)
        self.add_script_btn = QPushButton("＋ Add Script")
        self.add_script_btn.setFixedHeight(24)
        self.add_script_btn.setStyleSheet(self._mini_btn_style(C["pipeline"]))
        self.add_script_btn.clicked.connect(self._add_script)
        scr_btn_row.addWidget(self.add_script_btn)
        scr_btn_row.addStretch()
        scripts_l.addLayout(scr_btn_row)

        # Script info banner
        info = QLabel(
            "Scripts are parsed into structured indexes.\n"
            "Functions, classes, imports & types are extracted\n"
            "and injected as rich context for coding prompts.")
        info.setWordWrap(True)
        info.setObjectName("ref_info_banner")
        scripts_l.addWidget(info)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        scripts_l.addWidget(sep2)

        self.script_list = QListWidget()
        self.script_list.setObjectName("ref_script_list")
        self.script_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.script_list.customContextMenuRequested.connect(
            lambda pos: self._ctx_menu(pos, self.script_list))
        self.script_list.currentItemChanged.connect(
            self._on_script_selected)
        scripts_l.addWidget(self.script_list, 1)

        # Script detail pane
        self.script_detail = QLabel("")
        self.script_detail.setWordWrap(True)    
        self.script_detail.setTextFormat(Qt.TextFormat.RichText)
        self.script_detail.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.script_detail.setObjectName("ref_script_detail")
        scripts_l.addWidget(self.script_detail)

        scripts_widget.setLayout(scripts_l)
        self.tab_widget.addTab(scripts_widget, "💻 Scripts")

        # ── RAM info ─────────────────────────────────────────────────────────
        self.ram_info = QLabel("")
        self.ram_info.setWordWrap(True)
        self.ram_info.setObjectName("ref_ram_info")
        root.addWidget(self.ram_info)

        self.setLayout(root)

        self._ram_timer = QTimer(self)
        self._ram_timer.timeout.connect(self._update_ram_badge)
        self._ram_timer.start(3000)
        self._refresh()

    # ── Style helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _mini_btn_style(color: str) -> str:
        # Derive a soft tint from whatever accent color is passed in
        tint  = color.lstrip("#")
        r,g,b = int(tint[0:2],16), int(tint[2:4],16), int(tint[4:6],16)
        return (
            f"QPushButton{{background:rgba({r},{g},{b},0.12);color:{color};"
            f"border:1px solid rgba({r},{g},{b},0.30);border-radius:5px;"
            f"font-size:10px;padding:0 6px;}}"
            f"QPushButton:hover{{background:rgba({r},{g},{b},0.28);}}")

    @staticmethod
    def _list_style() -> str:        
        return (
            f"QListWidget{{background:transparent;border:none;"
            f"font-size:10px;outline:none;}}"
            f"QListWidget::item{{padding:5px 8px;border-radius:5px;"
            f"margin:2px 0;min-height:18px;}}"
            f"QListWidget::item:hover{{background:rgba(167,139,250,0.08);}}"
            f"QListWidget::item:selected{{background:rgba(124,58,237,0.22);"
            f"color:{C['acc2']};border:1px solid rgba(167,139,250,0.2);}}")

    # ── Store access ──────────────────────────────────────────────────────────
    @property
    def _store(self):
        return self._get_store(self.session_id)

    def update_session(self, session_id: str):
        self.session_id = session_id
        self._refresh()

    # ── Refresh both lists ────────────────────────────────────────────────────
    def _refresh(self):
        self.doc_list.clear()
        self.script_list.clear()
        store = self._store
        for ref in store._refs.values():
            if isinstance(ref, ScriptSmartReference):
                self._add_script_list_item(ref)
            else:
                self._add_doc_list_item(ref)

    def _add_doc_list_item(self, ref):
        icon = {"pdf": "📄", "python": "🐍", "text": "📝"}.get(
            getattr(ref, "ftype", ""), "📎")
        item = QListWidgetItem(f"{icon}  {ref.name}")
        item.setData(Qt.ItemDataRole.UserRole, ref.ref_id)
        item.setToolTip(ref.full_text_preview())
        item.setForeground(QColor(C["txt"]))
        self.doc_list.addItem(item)

    def _add_script_list_item(self, ref: ScriptSmartReference):
        ps  = ref.parsed
        fn  = len(ps.functions)
        cls = len(ps.classes)
        tag = f"  [{ps.language} · {fn}fn · {cls}cl]"
        item = QListWidgetItem(f"💻  {ref.name}{tag}")
        item.setData(Qt.ItemDataRole.UserRole, ref.ref_id)
        item.setToolTip(ref.full_text_preview())
        item.setForeground(QColor(C["pipeline"]))
        self.script_list.addItem(item)

    # ── Context menu (remove) ─────────────────────────────────────────────────
    def _ctx_menu(self, pos, list_widget: QListWidget):        
        item = list_widget.itemAt(pos)
        if not item:
            return
        ref_id = item.data(Qt.ItemDataRole.UserRole)
        menu   = QMenu(self)
        act_rm = menu.addAction("🗑  Remove Reference")
        chosen = menu.exec(list_widget.mapToGlobal(pos))
        if chosen == act_rm:
            self._store.remove(ref_id)
            self._refresh()
            self.refs_changed.emit()

    # ── Script selection → detail pane ────────────────────────────────────────
    def _on_script_selected(self, item, _=None):
        if not item:
            return
        ref_id = item.data(Qt.ItemDataRole.UserRole)
        ref    = self._store._refs.get(ref_id)
        if not isinstance(ref, ScriptSmartReference):
            return
        ps   = ref.parsed
        detail = (
            f"<b>{ref.name}</b>  [{ps.language}]<br>"
            f"Functions: {', '.join(f.name for f in ps.functions[:10])}"
            + ("…" if len(ps.functions) > 10 else "")
            + f"<br>Classes: {', '.join(c.name for c in ps.classes[:8])}"
            + ("…" if len(ps.classes) > 8 else "")
            + f"<br>Imports: {len(ps.imports)}"
            + (f"<br>Types: {len(ps.types)}" if ps.types else "")
            + (f"<br><span style='color:{C['err']};'>"
               f"Errors: {'; '.join(ps.errors[:2])}</span>"
               if ps.errors else "")
        )
        self.script_detail.setText(detail)

    # ── Add document ──────────────────────────────────────────────────────────
    def _add_doc(self, ftype: str):
        if ftype == "pdf" and not self._has_pdf:
            QMessageBox.warning(
                self, "Missing Dep", "Install PyPDF2: pip install PyPDF2")
            return
        filters = {
            "pdf":    "PDF Files (*.pdf)",
            "python": "Python Files (*.py)",
            "text":   "Text/Markdown (*.txt *.md *.rst)",
        }
        path, _ = QFileDialog.getOpenFileName(
            self, f"Add {ftype.upper()} Reference",
            str(Path.home()), filters.get(ftype, "All Files (*)"))
        if not path:
            return
        name = Path(path).name
        if ftype == "pdf":
            reader = self._PdfReader(path)
            raw = "\n".join(
                page.extract_text() or "" for page in reader.pages)
        else:
            try:
                raw = Path(path).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                QMessageBox.warning(self, "Read Error", str(e)); return
        if not raw.strip():
            QMessageBox.warning(self, "Empty", "No text extracted."); return
        self._store.add_reference(name, ftype, raw)
        self._refresh()
        self.refs_changed.emit()

    # ── Add script (with parser) ──────────────────────────────────────────────
    def _add_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Add Source Code Script",
            str(Path.home()), SCRIPT_EXTENSIONS_FILTER)
        if not path:
            return
        name = Path(path).name
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            QMessageBox.warning(self, "Read Error", str(e)); return
        if not raw.strip():
            QMessageBox.warning(self, "Empty File", "File has no content."); return

        # RAM check before parse
        self._ram_watchdog(self.session_id)        
        self._store = SessionReferenceStore()
        ref = self._store.add_script_reference(name, raw)
        self._refresh()
        self.refs_changed.emit()

        ps = ref.parsed
        # Flash parse result
        QMessageBox.information(
            self, "Script Parsed ✅",
            f"Parsed {name}  [{ps.language}]\n\n"
            + ps.summary_header()
            + ("\n\nErrors:\n" + "\n".join(ps.errors) if ps.errors else ""))

    # ── Multi-PDF ─────────────────────────────────────────────────────────────
    def _multi_pdf_requested(self):
        if not self._has_pdf:
            QMessageBox.warning(
                self, "Missing Dep", "Install PyPDF2: pip install PyPDF2")
            return        
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Multiple PDFs",
            str(Path.home()), "PDF Files (*.pdf)")
        if not paths:
            return
        self._pending_multi_pdfs = paths
        self.refs_changed.emit()

    # ── RAM badge ─────────────────────────────────────────────────────────────
    def _update_ram_badge(self):
        free_mb = self.ram_free_mb()
        # reuse ram_watchdog_mb constant from main
        try:
            threshold = RAM_WATCHDOG_MB()
        except Exception:
            threshold = 800
        if free_mb < threshold:
            self.ram_badge.setText(f"⚠️ {free_mb:.0f}MB free")
            self.ram_badge.setVisible(True)
        else:
            self.ram_badge.setVisible(False)

        store   = self._store
        n_refs  = len(store._refs)
        n_hot   = sum(len(getattr(r, "_hot", {})) for r in store._refs.values())
        n_total = sum(len(getattr(r, "_chunks", [1])) for r in store._refs.values())
        if n_refs > 0:
            self.ram_info.setText(
                f"{n_refs} ref(s) · {n_hot}/{n_total} chunks hot · "
                f"{free_mb:.0f} MB free RAM")
        else:
            self.ram_info.setText(f"{free_mb:.0f} MB free RAM")

    # ── Context for prompt ────────────────────────────────────────────────────
    def get_context_for(self, query: str) -> str:
        if not hasattr(self, "_store"):
            self._store = SessionReferenceStore(self.session.id)
        return self._store.build_context_block_extended(query)

class ReferencePanel(QWidget):
    """
    Compact collapsible reference sidebar for a chat session.
    Shows attached files with remove buttons and a RAM status badge.
    """
    refs_changed = pyqtSignal()

    def __init__(self, session_id: str, parent=None):
        super().__init__(parent)
        self.session_id = session_id
        self._store     = get_ref_store(session_id)
        self._building  = False
        self.setMaximumWidth(280)
        self.setMinimumWidth(220)
        self.setObjectName("ref_panel_v1")
        root = QVBoxLayout()
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(7)

        # Header
        hdr_row = QHBoxLayout(); hdr_row.setSpacing(6)
        hdr = QLabel("📎  References")
        hdr.setStyleSheet(f"color:{C['txt']};font-weight:700;font-size:12px;")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()

        self.ram_badge = QLabel("")
        self.ram_badge.setStyleSheet(
            f"color:{C['warn']};font-size:9px;padding:1px 5px;"
            f"background:rgba(251,191,36,0.1);border-radius:3px;"
        )
        hdr_row.addWidget(self.ram_badge)
        root.addLayout(hdr_row)

        # Add buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self.add_pdf_btn = QPushButton("＋ PDF")
        self.add_py_btn  = QPushButton("＋ .py")
        self.add_txt_btn = QPushButton("＋ Text")
        for b in (self.add_pdf_btn, self.add_py_btn, self.add_txt_btn):
            b.setFixedHeight(26)
            b.setStyleSheet(
                f"QPushButton{{background:rgba(124,58,237,0.15);color:{C['acc']};"
                f"border:1px solid rgba(167,139,250,0.25);border-radius:6px;"
                f"font-size:10px;padding:0 6px;}}"
                f"QPushButton:hover{{background:rgba(124,58,237,0.3);color:{C['acc2']};}}"
            )
        self.add_pdf_btn.clicked.connect(lambda: self._add_file("pdf"))
        self.add_py_btn.clicked.connect(lambda: self._add_file("python"))
        self.add_txt_btn.clicked.connect(lambda: self._add_file("text"))
        btn_row.addWidget(self.add_pdf_btn)
        btn_row.addWidget(self.add_py_btn)
        btn_row.addWidget(self.add_txt_btn)
        root.addLayout(btn_row)

        # Multi-PDF summarize button
        self.multi_pdf_btn = QPushButton("📚  Summarize Multiple PDFs")
        self.multi_pdf_btn.setFixedHeight(28)
        self.multi_pdf_btn.setStyleSheet(
            f"QPushButton{{background:rgba(52,211,153,0.1);color:{C['ok']};"
            f"border:1px solid rgba(52,211,153,0.25);border-radius:6px;"
            f"font-size:10px;padding:0 8px;}}"
            f"QPushButton:hover{{background:rgba(52,211,153,0.22);}}"
        )
        self.multi_pdf_btn.clicked.connect(self._multi_pdf_requested)
        root.addWidget(self.multi_pdf_btn)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"border:none;border-top:1px solid {C['bdr']};")
        root.addWidget(sep)

        # Reference list
        self.ref_list = QListWidget()
        self.ref_list.setStyleSheet(
            f"QListWidget{{background:transparent;border:none;font-size:10px;outline:none;}}"
            f"QListWidget::item{{padding:4px 6px;border-radius:4px;margin:1px 0;}}"
            f"QListWidget::item:selected{{background:rgba(124,58,237,0.25);color:{C['acc2']};}}"
        )
        self.ref_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ref_list.customContextMenuRequested.connect(self._ref_ctx_menu)
        root.addWidget(self.ref_list, 1)

        # RAM info
        self.ram_info = QLabel("")
        self.ram_info.setWordWrap(True)
        self.ram_info.setStyleSheet(f"color:{C['txt2']};font-size:9px;padding:2px;")
        root.addWidget(self.ram_info)

        self.setLayout(root)
        self._ram_timer = QTimer(self)
        self._ram_timer.timeout.connect(self._update_ram_badge)
        self._ram_timer.start(3000)
        self._refresh()

    def update_session(self, session_id: str):
        self.session_id = session_id
        self._store     = get_ref_store(session_id)
        self._refresh()

    def _refresh(self):
        self.ref_list.clear()
        for ref in self._store.refs.values():
            icon  = {"pdf": "📄", "python": "🐍", "text": "📝"}.get(ref.ftype, "📎")
            n_hot = len(ref._hot)
            n_tot = len(ref._chunks)
            ram_tag = f"  [{n_hot}/{n_tot} chunks in RAM]"
            item  = QListWidgetItem(f"{icon}  {ref.name}{ram_tag}")
            item.setData(Qt.ItemDataRole.UserRole, ref.ref_id)
            item.setToolTip(ref.full_text_preview())
            item.setForeground(QColor(C["txt"]))
            self.ref_list.addItem(item)

    def _ref_ctx_menu(self, pos):
        item = self.ref_list.itemAt(pos)
        if not item: return
        ref_id = item.data(Qt.ItemDataRole.UserRole)
        menu   = QMenu(self)
        act_rm = menu.addAction("🗑  Remove Reference")
        chosen = menu.exec(self.ref_list.mapToGlobal(pos))
        if chosen == act_rm:
            self._store.remove(ref_id)
            self._refresh()
            self.refs_changed.emit()

    def _add_file(self, ftype: str):
        if not HAS_PDF and ftype == "pdf":
            QMessageBox.warning(self, "Missing Dep", "Install PyPDF2: pip install PyPDF2")
            return
        filters = {"pdf": "PDF Files (*.pdf)", "python": "Python Files (*.py)",
                   "text": "Text/Markdown (*.txt *.md *.rst)"}
        path, _ = QFileDialog.getOpenFileName(
            self, f"Add {ftype.upper()} Reference",
            str(Path.home()), filters.get(ftype, "All Files (*)"))
        if not path: return

        filename = Path(path).name
        if ftype == "pdf":
            if not HAS_PDF: return
            reader = PdfReader(path)
            raw = "\n".join(
                page.extract_text() or "" for page in reader.pages)
        else:
            try:
                raw = Path(path).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                QMessageBox.warning(self, "Read Error", str(e)); return

        if not raw.strip():
            QMessageBox.warning(self, "Empty File", "No text extracted."); return

        # RAM watchdog before adding
        if ram_free_mb() < RAM_WATCHDOG_MB():
            self._store.flush_ram()

        ref = self._store.add_reference(filename, ftype, raw)
        self._refresh()
        self.refs_changed.emit()

    def _multi_pdf_requested(self):
        """Open multi-file picker and emit signal with paths."""
        if not HAS_PDF:
            QMessageBox.warning(self, "Missing Dep", "Install PyPDF2: pip install PyPDF2")
            return
        # Use a custom dialog to pick multiple PDFs        
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Multiple PDFs for Bulk Summarization",
            str(Path.home()), "PDF Files (*.pdf)")
        if not paths: return
        # Store picked paths temporarily for MainWindow to pick up
        self._pending_multi_pdfs = paths
        self.refs_changed.emit()   # MainWindow listens and handles

    def _update_ram_badge(self):
        free_mb = ram_free_mb()
        if free_mb < RAM_WATCHDOG_MB():
            self.ram_badge.setText(f"⚠️ RAM: {free_mb:.0f}MB free")
            self.ram_badge.setVisible(True)
        else:
            self.ram_badge.setVisible(False)
        n_refs  = len(self._store.refs)
        n_hot   = sum(len(r._hot) for r in self._store.refs.values())
        n_total = sum(len(r._chunks) for r in self._store.refs.values())
        if n_refs > 0:
            self.ram_info.setText(
                f"{n_refs} ref(s) · {n_hot}/{n_total} chunks hot · "
                f"{free_mb:.0f} MB free RAM")
        else:
            self.ram_info.setText(f"{free_mb:.0f} MB free RAM")

    def get_context_for(self, query: str) -> str:
        return self._store.build_context_block(query)
