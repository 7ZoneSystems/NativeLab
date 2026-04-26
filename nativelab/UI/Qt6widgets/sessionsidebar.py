from nativelab.imports.import_global import Dict, List, QHBoxLayout, datetime, Qt, pyqtSignal, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QListWidget, QListWidgetItem, QMenu, QInputDialog, QColor, QTextEdit, QFont
from nativelab.UI.buildUI import C
from nativelab.Server.server_global import Session
class SessionSidebar(QWidget):
    session_selected = pyqtSignal(str)
    new_session      = pyqtSignal()
    session_deleted  = pyqtSignal(str)
    session_renamed  = pyqtSignal(str, str)
    session_exported = pyqtSignal(str)

    def __init__(self):
        from nativelab.UI.buildUI import C
        super().__init__()
        self.setMinimumWidth(180)
        self.setMaximumWidth(300)
        self.setObjectName("session_sidebar")
        root = QVBoxLayout()
        root.setContentsMargins(12, 18, 12, 12)
        root.setSpacing(10)

        hdr = QLabel("CONVERSATIONS")
        hdr.setObjectName("sec_lbl")
        root.addWidget(hdr)

        self.new_btn = QPushButton("＋  New Chat")
        self.new_btn.setObjectName("btn_new")
        self.new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_btn.clicked.connect(self.new_session)
        root.addWidget(self.new_btn)

        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍  Search…")
        self.search.textChanged.connect(self._redraw)
        root.addWidget(self.search)

        self.lst = QListWidget()
        self.lst.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.lst.customContextMenuRequested.connect(self._ctx_menu)
        self.lst.itemClicked.connect(self._on_click)
        root.addWidget(self.lst, 1)
        self.setLayout(root)
        self._sessions: Dict[str, Session] = {}
        self._active:   str = ""
        self._busy:     str = ""

    def refresh(self, sessions: Dict[str, Session], active_id: str = "", busy_id: str = ""):
        self._sessions = sessions
        self._active   = active_id
        self._busy     = busy_id
        self._redraw()

    def set_active(self, sid: str):
        self._active = sid
        self._redraw()

    def _redraw(self, _=None):
        q = self.search.text().lower()
        self.lst.clear()
        grouped: Dict[str, List[Session]] = {}
        for s in sorted(self._sessions.values(), key=lambda x: x.id, reverse=True):
            if q and q not in s.title.lower(): continue
            grouped.setdefault(s.created, []).append(s)

        for date in sorted(grouped.keys(), reverse=True):
            di = QListWidgetItem(f"  📅  {date}")
            di.setFlags(Qt.ItemFlag.NoItemFlags)
            di.setData(Qt.ItemDataRole.ForegroundRole, QColor(C["txt2"]))
            f = di.font(); f.setPointSize(10); di.setFont(f)
            self.lst.addItem(di)
            for s in grouped[date]:
                title = (s.title[:34] + "…") if len(s.title) > 34 else s.title
                item  = QListWidgetItem(f"    {title}")
                item.setData(Qt.ItemDataRole.UserRole, s.id)
                if s.id == self._active:
                    item.setData(Qt.ItemDataRole.ForegroundRole, QColor(C["acc"]))
                    fo = item.font(); fo.setBold(True); item.setFont(fo)
                elif s.id == self._busy:
                    item.setData(Qt.ItemDataRole.ForegroundRole, QColor(C["pipeline"]))
                    item.setToolTip("⚡ Processing…")
                    fo = item.font(); fo.setBold(True); item.setFont(fo)
                self.lst.addItem(item)
    def _on_click(self, item: QListWidgetItem):
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid: self.session_selected.emit(sid)

    def _ctx_menu(self, pos):
        item = self.lst.itemAt(pos)
        if not item: return
        sid = item.data(Qt.ItemDataRole.UserRole)
        if not sid: return
        menu = QMenu(self)
        act_rename = menu.addAction("✏️  Rename")
        act_export = menu.addAction("📤  Export Markdown")
        menu.addSeparator()
        act_del    = menu.addAction("🗑  Delete")
        chosen = menu.exec(self.lst.mapToGlobal(pos))
        if chosen == act_del:
            self.session_deleted.emit(sid)
        elif chosen == act_rename:
            text, ok = QInputDialog.getText(
                self, "Rename Session", "New title:",
                text=self._sessions[sid].title)
            if ok and text.strip():
                self.session_renamed.emit(sid, text.strip())
        elif chosen == act_export:
            self.session_exported.emit(sid)


class LogConsole(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(14, 8, 14, 6)
        lbl = QLabel("🐞  Debug Console")
        lbl.setObjectName("ref_hdr")
        clr = QPushButton("Clear")
        clr.setFixedSize(70, 28)
        clr.clicked.connect(lambda: self.te.clear())
        toolbar.addWidget(lbl); toolbar.addStretch(); toolbar.addWidget(clr)
        self.te = QTextEdit()
        self.te.setReadOnly(True)
        self.te.setFont(QFont("Consolas", 10))
        self.te.setObjectName("log_te")
        root.addLayout(toolbar)
        root.addWidget(self.te)
        self.setLayout(root)

    def log(self, level: str, msg: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        lc = {"INFO": C["acc"], "WARN": C["warn"], "ERROR": C["err"]}.get(level, C["txt2"])
        self.te.append(
            f'<span style="color:{C["txt2"]}">[{ts}]</span> '
            f'<span style="color:{lc}">[{level}]</span> '
            f'<span style="color:{C["txt"]}">{msg}</span>'
        )