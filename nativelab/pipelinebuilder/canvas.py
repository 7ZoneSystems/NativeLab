from nativelab.imports.import_global import HAS_PDF,QMenu,Path,QColor,QFileDialog, QPainter, QPen, QBrush, QPainterPath, QPointF, QPolygonF, Qt, pyqtSignal, QWidget, QFont, QDataStream, QIODevice, QVariant, QInputDialog, QMessageBox, List, Optional
from .pipblck import PipelineBlock, PipelineBlockType
from .blck_typ import PipelineConnection
from nativelab.Server.server_global import detect_model_family
from nativelab.GlobalConfig.config_global import MODEL_ROLES
from nativelab.Model.model_global import detect_quant_type
from .editordialogue import CodeEditorDialog, LlmLogicEditorDialog
from nativelab.UI.UI_const import C
class PipelineCanvas(QWidget):
    """Interactive drag-and-drop pipeline canvas with curved arrows."""

    blocks_changed = pyqtSignal()
    PORT_R = 6
    GRID   = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1400, 900)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setAcceptDrops(True)   # accept drags from model list

        self.blocks:      List[PipelineBlock]      = []
        self.connections: List[PipelineConnection] = []

        self._drag_block:     Optional[PipelineBlock] = None
        self._drag_offset:    tuple = (0, 0)
        self._connect_from:   Optional[tuple] = None   # (block, port)
        self._connect_preview:Optional[tuple] = None
        self._hover_block:    Optional[PipelineBlock] = None
        self._hover_port:     Optional[str]           = None
        self._selected:       Optional[PipelineBlock] = None
        self._drop_preview:   Optional[tuple]         = None  # (x, y) ghost position
        self.preview_dots:    list                    = []    # set by FlowPreviewController

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._ctx_menu)

    # ── drag-and-drop from sidebar ────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            px = int(event.position().x())
            py = int(event.position().y())
            self._drop_preview = (self._snap(px), self._snap(py))
            self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_preview = None
        self.update()

    def dropEvent(self, event):
        self._drop_preview = None
        if not event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            event.ignore()
            return
        px = int(event.position().x())
        py = int(event.position().y())
        # Decode the QListWidget MIME data to get model path + role
        _data = event.mimeData().data("application/x-qabstractitemmodeldatalist")
        _stream = QDataStream(_data, QIODevice.OpenModeFlag.ReadOnly)
        model_path = ""
        role = "general"
        try:
            while not _stream.atEnd():
                _row = _stream.readInt32()
                _col = _stream.readInt32()
                _n   = _stream.readInt32()
                for _ in range(_n):
                    _role_key = _stream.readInt32()
                    _var = QVariant()
                    _stream >> _var
                    if _role_key == Qt.ItemDataRole.UserRole:
                        model_path = str(_var.value()) if _var.isValid() else ""
                    elif _role_key == Qt.ItemDataRole.UserRole + 1:
                        role = str(_var.value()) if _var.isValid() else "general"
        except Exception:
            pass  # fallback: no path decoded — block will show unconfigured

        drop_x = self._snap(px - 80)
        drop_y = self._snap(py - 30)
        b = self.add_block(PipelineBlockType.MODEL, x=drop_x, y=drop_y,
                           model_path=model_path, role=role or "general")
        if model_path:
            b.label = Path(model_path).stem[:18]
        self._selected = b
        self.update()
        event.acceptProposedAction()

    # ── block management ──────────────────────────────────────────────────────

    def add_block(self, btype: str, x: int = 100, y: int = 100,
                  model_path: str = "", role: str = "general") -> PipelineBlock:
        b = PipelineBlock(btype, x, y, model_path, role)
        self.blocks.append(b)
        self.update()
        self.blocks_changed.emit()
        return b

    def remove_block(self, b: PipelineBlock):
        bid = b.bid
        self.blocks      = [bl for bl in self.blocks      if bl.bid != bid]
        self.connections = [c  for c  in self.connections
                            if c.from_block_id != bid and c.to_block_id != bid]
        if self._selected is b:
            self._selected = None
        self.update()
        self.blocks_changed.emit()

    def clear_all(self):
        self.blocks.clear()
        self.connections.clear()
        self._selected     = None
        self._drag_block   = None
        self._connect_from = None
        PipelineBlock._id_counter = 0
        self.update()
        self.blocks_changed.emit()

    def _snap(self, v: int) -> int:
        return round(v / self.GRID) * self.GRID

    def _block_by_id(self, bid: int) -> Optional[PipelineBlock]:
        for b in self.blocks:
            if b.bid == bid:
                return b
        return None

    # ── painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event):

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ── background ───────────────────────────────────────────────────────
        p.fillRect(self.rect(), QColor(C["bg1"]))
        # ── grid dots ────────────────────────────────────────────────────────
        grid_c = QColor(C["bdr"])
        grid_c.setAlpha(80)
        p.setPen(QPen(grid_c, 1))
        for gx in range(0, self.width(), self.GRID):
            p.drawLine(gx, 0, gx, self.height())
        for gy in range(0, self.height(), self.GRID):
            p.drawLine(0, gy, self.width(), gy)

        # ── connections ───────────────────────────────────────────────────────
        for conn in self.connections:
            fb = self._block_by_id(conn.from_block_id)
            tb = self._block_by_id(conn.to_block_id)
            if fb and tb:
                self._draw_arrow(p, fb, conn.from_port,
                                 tb, conn.to_port,
                                 conn.is_loop, conn.loop_times)

        # ── flow-preview animated dots ────────────────────────────────────────
        if self.preview_dots:
            p.save()
            p.setPen(Qt.PenStyle.NoPen)
            for _dot in list(self.preview_dots):
                _fb = self._block_by_id(_dot.conn.from_block_id)
                _tb = self._block_by_id(_dot.conn.to_block_id)
                if _fb is None or _tb is None:
                    continue
                _sx, _sy = _fb.port_pos(_dot.conn.from_port)
                _ex, _ey = _tb.port_pos(_dot.conn.to_port)
                _path = self._make_path(_sx, _sy, _ex, _ey,
                                        _dot.conn.from_port, _dot.conn.to_port)
                _pt   = _path.pointAtPercent(min(_dot.t, 1.0))
                _col  = QColor(_dot.color)
                # outer glow
                _glow = QColor(_col)
                _glow.setAlpha(55)
                p.setBrush(QBrush(_glow))
                p.drawEllipse(_pt, _dot.radius + 3, _dot.radius + 3)
                # solid dot
                _col.setAlpha(220)
                p.setBrush(QBrush(_col))
                p.drawEllipse(_pt, _dot.radius, _dot.radius)
            p.restore()

        # ── preview arrow ─────────────────────────────────────────────────────
        if self._connect_from and self._connect_preview:
            b, port = self._connect_from
            sx, sy  = b.port_pos(port)
            ex, ey  = self._connect_preview
            pen = QPen(QColor(C["acc"]), 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(self._make_path(sx, sy, ex, ey, "E", "W"))

        # ── blocks ────────────────────────────────────────────────────────────
        self._draw_drop_ghost(p)

        # ── blocks ────────────────────────────────────────────────────────────
        for b in self.blocks:
            self._draw_block(p, b)

        p.end()

    # ─ colour table ────────────────────────────────────────────────────────────
    _BLOCK_COLORS = {
        PipelineBlockType.INPUT:        lambda: (C["ok"],       C["bg2"]),
        PipelineBlockType.OUTPUT:       lambda: (C["err"],      C["bg2"]),
        PipelineBlockType.INTERMEDIATE: lambda: (C["warn"],     C["bg2"]),
        PipelineBlockType.MODEL:        lambda: (C["acc"],      C["bg1"]),
        PipelineBlockType.REFERENCE:    lambda: (C["acc"],      C["bg2"]),
        PipelineBlockType.KNOWLEDGE:    lambda: (C["acc2"],     C["bg2"]),
        PipelineBlockType.PDF_SUMMARY:  lambda: (C["pipeline"], C["bg2"]),
        # Logic blocks — distinct colour family
        PipelineBlockType.IF_ELSE:      lambda: ("#f59e0b",     C["bg2"]),
        PipelineBlockType.SWITCH:       lambda: ("#f97316",     C["bg2"]),
        PipelineBlockType.FILTER:       lambda: ("#84cc16",     C["bg2"]),
        PipelineBlockType.TRANSFORM:    lambda: ("#06b6d4",     C["bg2"]),
        PipelineBlockType.MERGE:        lambda: ("#8b5cf6",     C["bg2"]),
        PipelineBlockType.SPLIT:        lambda: ("#ec4899",     C["bg2"]),
        PipelineBlockType.CUSTOM_CODE:  lambda: ("#10b981",     C["bg1"]),
        # LLM logic blocks — warmer violet family to distinguish from code logic
        PipelineBlockType.LLM_IF:        lambda: ("#a855f7",    C["bg1"]),
        PipelineBlockType.LLM_SWITCH:    lambda: ("#7c3aed",    C["bg1"]),
        PipelineBlockType.LLM_FILTER:    lambda: ("#6366f1",    C["bg1"]),
        PipelineBlockType.LLM_TRANSFORM: lambda: ("#0ea5e9",    C["bg1"]),
        PipelineBlockType.LLM_SCORE:     lambda: ("#d946ef",    C["bg1"]),
    }
    _BLOCK_ICONS = {
        PipelineBlockType.INPUT:        "▶ INPUT",
        PipelineBlockType.OUTPUT:       "■ OUTPUT",
        PipelineBlockType.INTERMEDIATE: "◈ INTER.",
        PipelineBlockType.MODEL:        None,          # filled dynamically
        PipelineBlockType.REFERENCE:    "📎 REF.",
        PipelineBlockType.KNOWLEDGE:    "💡 KNOW.",
        PipelineBlockType.PDF_SUMMARY:  "📄 PDF",
        PipelineBlockType.IF_ELSE:      "⑂ IF/ELSE",
        PipelineBlockType.SWITCH:       "⑃ SWITCH",
        PipelineBlockType.FILTER:       "⊘ FILTER",
        PipelineBlockType.TRANSFORM:    "⟲ TRANSFORM",
        PipelineBlockType.MERGE:        "⊕ MERGE",
        PipelineBlockType.SPLIT:        "⑁ SPLIT",
        PipelineBlockType.CUSTOM_CODE:  "⌥ CODE",
        PipelineBlockType.LLM_IF:        "🧠 LLM-IF",
        PipelineBlockType.LLM_SWITCH:    "🧠 LLM-SW",
        PipelineBlockType.LLM_FILTER:    "🧠 LLM-FL",
        PipelineBlockType.LLM_TRANSFORM: "🧠 LLM-TX",
        PipelineBlockType.LLM_SCORE:     "🧠 LLM-SC",
    }

    def _draw_block(self, p, b: PipelineBlock):      

        fn = self._BLOCK_COLORS.get(b.btype, self._BLOCK_COLORS[PipelineBlockType.MODEL])
        border_c, fill_c = fn()

        border = QColor(border_c)
        fill   = QColor(fill_c)
        sel    = b is self._selected

        # shadow
        shadow = QColor(0, 0, 0, 50)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(shadow))
        p.drawRoundedRect(b.x + 4, b.y + 4, b.w, b.h, 10, 10)

        # body
        p.setBrush(QBrush(fill))
        p.setPen(QPen(border, 2.5 if sel else 1.5))
        p.drawRoundedRect(b.x, b.y, b.w, b.h, 10, 10)

        # header strip
        hdr_c = QColor(border_c); hdr_c.setAlpha(55)
        p.setBrush(QBrush(hdr_c))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(b.x + 1, b.y + 1, b.w - 2, 20, 9, 9)
        p.drawRect(b.x + 1, b.y + 10, b.w - 2, 11)

        # header text
        p.setPen(QColor(border_c))
        p.setFont(QFont("Inter", 7, QFont.Weight.Bold))
        if b.btype == PipelineBlockType.MODEL:
            tag = f"⚡ {b.role.upper()}"
        else:
            tag = self._BLOCK_ICONS.get(b.btype, "BLOCK")
        p.drawText(b.x, b.y + 1, b.w, 20,
                   Qt.AlignmentFlag.AlignCenter, tag)

        # name label
        p.setPen(QColor(C["txt"]))
        p.setFont(QFont("Inter", 9))
        p.drawText(b.x + 4, b.y + 24, b.w - 8, b.h - 38,
                   Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                   b.label)

        # model badge (quant + family)
        if b.btype == PipelineBlockType.MODEL and b.model_path:
            fam   = detect_model_family(b.model_path)
            quant = detect_quant_type(b.model_path)
            badge = f"{fam.name[:8]}·{quant}"
            p.setPen(QColor(C["txt3"]))
            p.setFont(QFont("Consolas", 7))
            p.drawText(b.x, b.y + b.h - 17, b.w, 15,
                       Qt.AlignmentFlag.AlignCenter, badge)

        # port dots
        for port in ("N", "S", "E", "W"):
            px, py     = b.port_pos(port)
            is_hovered = (b is self._hover_block and port == self._hover_port)
            dot_c      = QColor(border_c)
            r          = self.PORT_R + 3 if is_hovered else self.PORT_R
            p.setBrush(QBrush(QColor(C["bg0"])))
            p.setPen(QPen(dot_c, 2))
            p.drawEllipse(px - r, py - r, r * 2, r * 2)
            if is_hovered:
                p.setBrush(QBrush(dot_c))
                p.setPen(Qt.PenStyle.NoPen)
                inner = r - 3
                p.drawEllipse(px - inner, py - inner, inner * 2, inner * 2)

    def _make_path(self, sx, sy, ex, ey, fport="E", tport="W"):
        """Cubic bezier path that adapts control handles to port orientation."""
        path = QPainterPath(QPointF(sx, sy))
        dx   = abs(ex - sx)
        dy   = abs(ey - sy)
        off  = max(60, dx // 2, dy // 2, 40)

        PORT_VEC = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
        fx, fy = PORT_VEC.get(fport, (1, 0))
        tx, ty = PORT_VEC.get(tport, (-1, 0))

        c1x = sx + fx * off
        c1y = sy + fy * off
        c2x = ex + tx * off
        c2y = ey + ty * off

        path.cubicTo(QPointF(c1x, c1y), QPointF(c2x, c2y), QPointF(ex, ey))
        return path

    def _draw_drop_ghost(self, p):
        """Draw a translucent ghost block at the current drag-over position."""
        if not self._drop_preview:
            return
        gx, gy = self._drop_preview
        gw, gh = 140, 48
        ghost_bg = QColor(C["acc"]); ghost_bg.setAlpha(40)
        ghost_bd = QColor(C["acc"]); ghost_bd.setAlpha(160)
        p.setBrush(QBrush(ghost_bg))
        p.setPen(QPen(ghost_bd, 2, Qt.PenStyle.DashLine))
        p.drawRoundedRect(gx, gy, gw, gh, 8, 8)
        p.setPen(QColor(C["acc"]))
        p.setFont(QFont("Inter", 9))
        p.drawText(gx, gy, gw, gh, Qt.AlignmentFlag.AlignCenter, "⚡ Drop here")

    def _draw_arrow(self, p, fb, fport, tb, tport, is_loop, loop_times):        
        sx, sy = fb.port_pos(fport)
        ex, ey = tb.port_pos(tport)

        color = QColor(C["pipeline"] if is_loop else C["acc2"])
        pen   = QPen(color, 2)
        if is_loop:
            pen.setStyle(Qt.PenStyle.DashDotLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        path = self._make_path(sx, sy, ex, ey, fport, tport)
        p.drawPath(path)

        # arrowhead
        t1 = path.pointAtPercent(0.97)
        t2 = path.pointAtPercent(1.0)
        adx = t2.x() - t1.x(); ady = t2.y() - t1.y()
        length = (adx ** 2 + ady ** 2) ** 0.5
        if length > 0.001:
            adx /= length; ady /= length
            sz = 10
            p1 = QPointF(t2.x() - sz * adx + sz * 0.5 * ady,
                         t2.y() - sz * ady - sz * 0.5 * adx)
            p2 = QPointF(t2.x() - sz * adx - sz * 0.5 * ady,
                         t2.y() - sz * ady + sz * 0.5 * adx)
            p.setBrush(QBrush(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([t2, p1, p2]))

        # loop badge
        if is_loop and loop_times > 1:
            mid = path.pointAtPercent(0.5)
            badge_bg = QColor(C["warn"]); badge_bg.setAlpha(180)
            p.setBrush(QBrush(badge_bg))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(int(mid.x()) - 14, int(mid.y()) - 10, 28, 20, 5, 5)
            p.setPen(QColor(C["bg0"]))
            p.setFont(QFont("Inter", 8, QFont.Weight.Bold))
            p.drawText(int(mid.x()) - 14, int(mid.y()) - 10, 28, 20,
                       Qt.AlignmentFlag.AlignCenter, f"×{loop_times}")

        # branch label badge (IF_ELSE / SWITCH fan-out arms)
        branch = getattr(self.connections[0] if self.connections else object(),
                         "branch_label", None)
        # Retrieve the actual connection object for this specific arrow
        _conn_obj = next(
            (c for c in self.connections
             if c.from_block_id == fb.bid and c.from_port == fport
             and c.to_block_id == tb.bid),
            None)
        branch = getattr(_conn_obj, "branch_label", "") if _conn_obj else ""
        if branch:
            mid = path.pointAtPercent(0.35)
            branch_col = {"TRUE": "#22c55e", "FALSE": "#ef4444"}.get(
                branch, C["pipeline"])
            badge_bg2 = QColor(branch_col); badge_bg2.setAlpha(200)
            w_b = max(len(branch) * 7 + 10, 36)
            p.setBrush(QBrush(badge_bg2))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(
                int(mid.x()) - w_b // 2, int(mid.y()) - 9, w_b, 18, 5, 5)
            p.setPen(QColor("#ffffff"))
            p.setFont(QFont("Inter", 7, QFont.Weight.Bold))
            p.drawText(int(mid.x()) - w_b // 2, int(mid.y()) - 9, w_b, 18,
                       Qt.AlignmentFlag.AlignCenter, branch)

    # ── mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        px = int(event.position().x())
        py = int(event.position().y())

        if event.button() == Qt.MouseButton.LeftButton:
            # Port hit? → start arrow draw
            for b in reversed(self.blocks):
                port = b.port_at(px, py)
                if port:
                    self._connect_from    = (b, port)
                    self._connect_preview = (px, py)
                    return
            # Block hit? → select + drag
            for b in reversed(self.blocks):
                if b.contains(px, py):
                    self._drag_block  = b
                    self._drag_offset = (px - b.x, py - b.y)
                    self._selected    = b
                    self.update()
                    return
            self._selected = None
            self.update()

    def mouseMoveEvent(self, event):
        px = int(event.position().x())
        py = int(event.position().y())

        if self._drag_block:
            ox, oy = self._drag_offset
            self._drag_block.x = self._snap(px - ox)
            self._drag_block.y = self._snap(py - oy)
            self.update()
            return

        if self._connect_from:
            self._connect_preview = (px, py)
            self.update()
            return

        # Hover
        self._hover_block = None
        self._hover_port  = None
        for b in reversed(self.blocks):
            port = b.port_at(px, py)
            if port:
                self._hover_block = b
                self._hover_port  = port
                self.setCursor(Qt.CursorShape.CrossCursor)
                self.update()
                return
            if b.contains(px, py):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                self.update()
                return
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, event):
        px = int(event.position().x())
        py = int(event.position().y())

        if self._drag_block:
            self._drag_block = None
            self.blocks_changed.emit()
            self.update()
            return

        if self._connect_from and event.button() == Qt.MouseButton.LeftButton:
            from_b, from_port   = self._connect_from
            self._connect_from    = None
            self._connect_preview = None
            for b in reversed(self.blocks):
                if b is from_b:
                    continue
                port = b.port_at(px, py)
                if port:
                    self._try_connect(from_b, from_port, b, port)
                    self.update()
                    return
            self.update()

    def mouseDoubleClickEvent(self, event):
        px = int(event.position().x())
        py = int(event.position().y())
        if event.button() != Qt.MouseButton.LeftButton:
            return
        for b in reversed(self.blocks):
            port = b.port_at(px, py)
            if port:
                before = len(self.connections)
                self.connections = [
                    c for c in self.connections
                    if not (
                        (c.from_block_id == b.bid and c.from_port == port) or
                        (c.to_block_id   == b.bid and c.to_port   == port)
                    )
                ]
                if len(self.connections) < before:
                    self.blocks_changed.emit()
                self.update()
                return

    # Logic block types that may fan-out to multiple targets
    _LOGIC_BTYPES = {
        PipelineBlockType.IF_ELSE,
        PipelineBlockType.SWITCH,
        PipelineBlockType.SPLIT,
        PipelineBlockType.MERGE,
        PipelineBlockType.FILTER,
        PipelineBlockType.TRANSFORM,
        PipelineBlockType.CUSTOM_CODE,
        PipelineBlockType.LLM_IF,
        PipelineBlockType.LLM_SWITCH,
        PipelineBlockType.LLM_FILTER,
        PipelineBlockType.LLM_TRANSFORM,
        PipelineBlockType.LLM_SCORE,
    }

    def _try_connect(self, fb: PipelineBlock, fport: str,
                     tb: PipelineBlock, tport: str):
        # Prevent self-loop via single click
        if fb.bid == tb.bid:
            return

        # Rule: model → model requires an intermediate in between
        # (except when source is a logic block)
        if (fb.btype == PipelineBlockType.MODEL and
                tb.btype == PipelineBlockType.MODEL and
                fb.btype not in self._LOGIC_BTYPES):
            QMessageBox.warning(
                self, "Connection Not Allowed",
                "Direct model-to-model connections are not allowed.\n\n"
                "You must place an ◈ Intermediate Output block between two model blocks.\n"
                "This enforces explicit output capture between pipeline stages.")
            return

        # Prevent exact duplicate connections (same from_bid, same from_port, same to_bid)
        already = any(
            c.from_block_id == fb.bid and c.from_port == fport
            and c.to_block_id == tb.bid
            for c in self.connections)
        if already:
            return

        is_loop    = self._would_form_loop(fb.bid, tb.bid)
        loop_times = 1
        if is_loop:
            val, ok = QInputDialog.getInt(
                self, "Loop Configuration",
                f"How many times should this edge loop?\n"
                f"(The output of '{tb.label}' feeds back into '{fb.label}')",
                value=3, min=2, max=999)
            if not ok:
                return
            loop_times = val

        # For non-logic source blocks: enforce single outgoing connection per port
        # For logic blocks: allow fan-out (multiple arrows from same port)
        if fb.btype not in self._LOGIC_BTYPES:
            self.connections = [
                c for c in self.connections
                if not (c.from_block_id == fb.bid and c.from_port == fport)]

        # For IF_ELSE — label the port so we know which branch this is
        branch_label = ""
        if fb.btype == PipelineBlockType.IF_ELSE:
            # E port = True branch, W port = False branch, S = either
            branch_map = {"E": "TRUE", "W": "FALSE", "S": "ELSE", "N": "PASS"}
            branch_label = branch_map.get(fport, fport)
            # Store branch on connection metadata via a subclass attr we'll inject
        elif fb.btype == PipelineBlockType.SWITCH:
            branch_label, ok = QInputDialog.getText(
                self, "Switch Branch Label",
                f"Label for this output arm of SWITCH '{fb.label}':\n"
                "(e.g. 'case_a', 'default')")
            if not ok:
                branch_label = fport

        conn = PipelineConnection(
            from_block_id=fb.bid, from_port=fport,
            to_block_id=tb.bid,   to_port=tport,
            is_loop=is_loop, loop_times=loop_times)
        conn.branch_label = branch_label   # dynamic attribute — stored at runtime
        self.connections.append(conn)
        self.update()
        self.blocks_changed.emit()

    def _configure_logic_block(self, b: "PipelineBlock"):
        """Open configuration dialog for logic / code blocks."""
        _LOGIC_BTYPES = {
            PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
            PipelineBlockType.FILTER,  PipelineBlockType.TRANSFORM,
            PipelineBlockType.MERGE,   PipelineBlockType.SPLIT,
            PipelineBlockType.CUSTOM_CODE,
        }
        if b.btype not in _LOGIC_BTYPES:
            return

        if b.btype == PipelineBlockType.IF_ELSE:
            cur = b.metadata.get("condition", "")
            cond, ok = QInputDialog.getText(
                self, f"IF / ELSE — '{b.label}'",
                "Python condition evaluated on the incoming text.\n"
                "Variable  'text'  is the incoming string.\n\n"
                "Examples:\n"
                "  len(text) > 500\n"
                "  'error' in text.lower()\n"
                "  text.startswith('FAIL')\n\n"
                "TRUE result → E port    FALSE result → W port",
                text=cur)
            if ok:
                b.metadata["condition"] = cond.strip()
                b.label = f"⑂ {cond.strip()[:18]}" if cond.strip() else "IF/ELSE"

        elif b.btype == PipelineBlockType.SWITCH:
            cur = b.metadata.get("switch_expr", "")
            expr, ok = QInputDialog.getText(
                self, f"SWITCH — '{b.label}'",
                "Python expression returning a string key.\n"
                "Variable  'text'  is the incoming string.\n\n"
                "Example:  'long' if len(text) > 300 else 'short'\n\n"
                "Connect outgoing arrows and label them with matching keys.",
                text=cur)
            if ok:
                b.metadata["switch_expr"] = expr.strip()
                b.label = f"⑃ {expr.strip()[:18]}" if expr.strip() else "SWITCH"

        elif b.btype == PipelineBlockType.FILTER:
            cur_cond  = b.metadata.get("filter_cond", "True")
            cur_mode  = b.metadata.get("filter_mode", "pass")
            cond, ok = QInputDialog.getText(
                self, f"FILTER — '{b.label}'",
                "Python condition. 'text' = incoming string.\n"
                "If TRUE → text passes to next block.\n"
                "If FALSE → pipeline stops (text is dropped).\n\n"
                "Example:  len(text.strip()) > 10",
                text=cur_cond)
            if ok:
                b.metadata["filter_cond"] = cond.strip() or "True"
                b.label = f"⊘ {cond.strip()[:18]}" if cond.strip() else "FILTER"

        elif b.btype == PipelineBlockType.TRANSFORM:
            items = [
                "prefix    — prepend fixed text",
                "suffix    — append fixed text",
                "replace   — find & replace substring",
                "upper     — convert to uppercase",
                "lower     — convert to lowercase",
                "strip     — strip leading/trailing whitespace",
                "truncate  — limit to N characters",
            ]
            cur_type = b.metadata.get("transform_type", "prefix")
            cur_idx  = next((i for i, s in enumerate(items) if cur_type in s), 0)
            ttype, ok = QInputDialog.getItem(
                self, f"TRANSFORM type — '{b.label}'",
                "Choose the transformation:", items, cur_idx, False)
            if not ok:
                return
            key = ttype.split()[0]
            b.metadata["transform_type"] = key
            if key in ("prefix", "suffix"):
                val, ok2 = QInputDialog.getMultiLineText(
                    self, f"TRANSFORM — {key}",
                    "Text to prepend/append:", b.metadata.get("transform_val", ""))
                if ok2: b.metadata["transform_val"] = val
            elif key == "replace":
                find_s, ok2 = QInputDialog.getText(
                    self, "TRANSFORM — find",
                    "Find this text:", text=b.metadata.get("transform_find", ""))
                if ok2: b.metadata["transform_find"] = find_s
                repl_s, ok3 = QInputDialog.getText(
                    self, "TRANSFORM — replace with",
                    "Replace with:", text=b.metadata.get("transform_repl", ""))
                if ok3: b.metadata["transform_repl"] = repl_s
            elif key == "truncate":
                n, ok2 = QInputDialog.getInt(
                    self, "TRANSFORM — truncate",
                    "Maximum characters:", value=int(b.metadata.get("transform_val", 500)),
                    min=1, max=999999)
                if ok2: b.metadata["transform_val"] = n
            b.label = f"⟲ {key.capitalize()}"

        elif b.btype == PipelineBlockType.MERGE:
            items = ["concat  — join with separator", "prepend  — put newest first",
                     "append  — put newest last", "json    — wrap all as JSON array"]
            cur  = b.metadata.get("merge_mode", "concat")
            mode, ok = QInputDialog.getItem(
                self, f"MERGE mode — '{b.label}'", "How to merge inputs:", items,
                next((i for i, s in enumerate(items) if cur in s), 0), False)
            if ok:
                b.metadata["merge_mode"] = mode.split()[0]
                if mode.split()[0] == "concat":
                    sep, ok2 = QInputDialog.getText(
                        self, "MERGE — separator",
                        "Separator between merged texts:", text=b.metadata.get("merge_sep", "\n\n---\n\n"))
                    if ok2: b.metadata["merge_sep"] = sep
                b.label = f"⊕ Merge/{mode.split()[0]}"

        elif b.btype == PipelineBlockType.SPLIT:
            n, ok = QInputDialog.getInt(
                self, f"SPLIT — '{b.label}'",
                "SPLIT broadcasts the same text to ALL outgoing connections.\n"
                "No configuration needed — just draw multiple output arrows.\n\n"
                "Confirm / view outgoing port count:", value=1, min=1, max=20)
            # No actual setting needed — split just fans out
            b.label = "⑁ Split"

        elif b.btype == PipelineBlockType.CUSTOM_CODE:
            dlg = CodeEditorDialog(b, parent=self)
            dlg.exec()

        self.update()

    def _configure_llm_logic_block(self, b: "PipelineBlock"):
        """Open the LLM-Logic configuration dialog."""
        _LLM_TYPES = {
            PipelineBlockType.LLM_IF,
            PipelineBlockType.LLM_SWITCH,
            PipelineBlockType.LLM_FILTER,
            PipelineBlockType.LLM_TRANSFORM,
            PipelineBlockType.LLM_SCORE,
        }
        if b.btype not in _LLM_TYPES:
            return
        dlg = LlmLogicEditorDialog(b, parent=self)
        dlg.exec()
        self.update()

    def _configure_context_block(self, b: "PipelineBlock"):
        """Configure a reference / knowledge / PDF block inline."""
        if b.btype == PipelineBlockType.REFERENCE:
            choice, ok = QInputDialog.getItem(
                self, f"Configure '{b.label}'", "Source:",
                ["Type / paste text", "Load from file"], 0, False)
            if not ok:
                return
            if choice.startswith("Type"):
                text, ok2 = QInputDialog.getMultiLineText(
                    self, "Reference Text", "Paste the reference text:")
                if ok2 and text.strip():
                    b.metadata["ref_text"] = text.strip()
                    name, ok3 = QInputDialog.getText(
                        self, "Name", "Short name:", text=b.label)
                    if ok3 and name.strip():
                        b.metadata["ref_name"] = name.strip()
                        b.label = name.strip()[:22]
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, "Select File", str(Path.home()),
                    "Text/Code (*.txt *.md *.py *.js *.ts *.json *.yaml *.rst *.html);;All (*)")
                if path:
                    try:
                        text = Path(path).read_text(encoding="utf-8", errors="replace")
                        b.metadata["ref_text"] = text
                        b.metadata["ref_name"] = Path(path).name
                        b.label = Path(path).name[:22]
                    except Exception as e:
                        QMessageBox.warning(self, "Read Error", str(e))
            self.update()

        elif b.btype == PipelineBlockType.KNOWLEDGE:
            text, ok = QInputDialog.getMultiLineText(
                self, "Knowledge Base", f"Enter knowledge text for '{b.label}':")
            if ok and text.strip():
                b.metadata["knowledge_text"] = text.strip()
                name, ok2 = QInputDialog.getText(
                    self, "Name", "Short name:", text=b.label)
                if ok2 and name.strip():
                    b.label = name.strip()[:22]
            self.update()

        elif b.btype == PipelineBlockType.PDF_SUMMARY:
            if not HAS_PDF:
                QMessageBox.warning(
                    self, "Missing Dependency",
                    "PyPDF2 is required.\n\n  pip install PyPDF2")
                return
            path, _ = QFileDialog.getOpenFileName(
                self, "Select PDF", str(Path.home()), "PDF (*.pdf)")
            if path:
                b.metadata["pdf_path"] = path
                b.label = Path(path).name[:22]

            # Role of this PDF relative to whatever arrived before it
            current_role = b.metadata.get("pdf_role", "reference")
            role, ok = QInputDialog.getItem(
                self,
                f"PDF Role — '{b.label}'",
                "How should this PDF relate to the prior block's output?\n\n"
                "  reference  →  prior output = MAIN,  PDF = REFERENCE\n"
                "  main       →  PDF = MAIN,  prior output = REFERENCE\n"
                "  (if there is no prior block, that side is simply omitted)",
                ["reference  —  prior is main, PDF is supporting context",
                 "main       —  PDF is primary, prior output is supporting context"],
                0 if current_role == "reference" else 1,
                False)
            if ok:
                b.metadata["pdf_role"] = "reference" if role.startswith("reference") else "main"
            self.update()

        elif b.btype == PipelineBlockType.INTERMEDIATE:
            # Show current settings as defaults
            current_prompt   = b.metadata.get("inter_prompt", "")
            current_position = b.metadata.get("inter_position", "above")

            # Step 1 — position choice
            position, ok = QInputDialog.getItem(
                self,
                f"Configure ◈ '{b.label}'",
                "Place your custom prompt:",
                ["Above incoming text  (prompt → model output)",
                 "Below incoming text  (model output → prompt)"],
                0 if current_position == "above" else 1,
                False)
            if not ok:
                return
            b.metadata["inter_position"] = (
                "above" if position.startswith("Above") else "below")

            # Step 2 — prompt text
            prompt_text, ok2 = QInputDialog.getMultiLineText(
                self,
                f"Configure ◈ '{b.label}'",
                "Custom prompt / instruction to inject at this stage:\n"
                "(Leave blank to clear and pass context through unchanged.)",
                current_prompt)
            if not ok2:
                return
            b.metadata["inter_prompt"] = prompt_text.strip()

            # Update block label so it's obvious it's configured
            if prompt_text.strip():
                preview = prompt_text.strip()[:16].replace("\n", " ")
                b.label = f"◈ {preview}…" if len(prompt_text.strip()) > 16 else f"◈ {preview}"
            else:
                b.label = "Intermediate"
            self.update()

    def _would_form_loop(self, from_bid: int, to_bid: int) -> bool:
        """Return True if to_bid can already reach from_bid (i.e. adding this edge creates a cycle)."""
        visited: set = set()
        queue = [to_bid]
        while queue:
            cur = queue.pop()
            if cur == from_bid:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            for c in self.connections:
                if c.from_block_id == cur:
                    queue.append(c.to_block_id)
        return False

    def _conn_at(self, px: int, py: int,
                 thresh: int = 8) -> "Optional[PipelineConnection]":
        """Return the first connection whose bezier arc passes within thresh px of (px, py)."""
        for conn in self.connections:
            fb = self._block_by_id(conn.from_block_id)
            tb = self._block_by_id(conn.to_block_id)
            if fb is None or tb is None:
                continue
            sx, sy = fb.port_pos(conn.from_port)
            ex, ey = tb.port_pos(conn.to_port)
            path   = self._make_path(sx, sy, ex, ey,
                                     conn.from_port, conn.to_port)
            for i in range(32):
                pt = path.pointAtPercent(i / 31)
                if abs(pt.x() - px) <= thresh and abs(pt.y() - py) <= thresh:
                    return conn
        return None

    def _ctx_menu(self, pos):
        px, py = pos.x(), pos.y()
        menu   = QMenu(self)

        # ── arrow hit-test (checked before blocks so thin lines are clickable) ──
        conn_target  = self._conn_at(px, py)
        act_del_conn = None
        if conn_target:
            fb = self._block_by_id(conn_target.from_block_id)
            tb = self._block_by_id(conn_target.to_block_id)
            fl = fb.label if fb else "?"
            tl = tb.label if tb else "?"
            act_del_conn = menu.addAction(
                f"🗑  Delete Arrow  ({fl} → {tl})")
            menu.addSeparator()

        target = next((b for b in reversed(self.blocks)
                       if b.contains(px, py)), None)

        if target:
            act_del = menu.addAction(f"🗑  Delete '{target.label}'")
            act_ren = menu.addAction("✏️  Rename block")
            act_role = None
            act_cfg  = None
            if target.btype == PipelineBlockType.MODEL:
                act_role = menu.addAction("🔧  Change Role")
            _CONFIGURABLE = {
                PipelineBlockType.REFERENCE, PipelineBlockType.KNOWLEDGE,
                PipelineBlockType.PDF_SUMMARY, PipelineBlockType.INTERMEDIATE,
                PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
                PipelineBlockType.FILTER, PipelineBlockType.TRANSFORM,
                PipelineBlockType.MERGE, PipelineBlockType.SPLIT,
                PipelineBlockType.CUSTOM_CODE,
                PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
                PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
                PipelineBlockType.LLM_SCORE,
            }
            if target.btype in _CONFIGURABLE:
                act_cfg = menu.addAction("⚙️  Configure block…")
            menu.addSeparator()

        act_clr = menu.addAction("🗑  Clear All Blocks & Connections")
        chosen  = menu.exec(self.mapToGlobal(pos))
        if not chosen:
            return

        if act_del_conn and chosen == act_del_conn:
            if conn_target in self.connections:
                self.connections.remove(conn_target)
                self.blocks_changed.emit()
                self.update()
            return

        if target:
            if chosen == act_del:
                self.remove_block(target)
                return
            elif chosen == act_ren:
                name, ok = QInputDialog.getText(
                    self, "Rename Block", "New label:", text=target.label)
                if ok and name.strip():
                    target.label = name.strip()
                    self.update()
                return
            elif act_role and chosen == act_role:
                r, ok = QInputDialog.getItem(
                    self, "Select Role", "Role:", MODEL_ROLES, 0, False)
                if ok:
                    target.role = r
                    self.update()
                return
            elif act_cfg and chosen == act_cfg:
                _LOGIC_TYPES = {
                    PipelineBlockType.IF_ELSE, PipelineBlockType.SWITCH,
                    PipelineBlockType.FILTER, PipelineBlockType.TRANSFORM,
                    PipelineBlockType.MERGE, PipelineBlockType.SPLIT,
                    PipelineBlockType.CUSTOM_CODE,
                }
                _LLM_LOGIC_TYPES = {
                    PipelineBlockType.LLM_IF, PipelineBlockType.LLM_SWITCH,
                    PipelineBlockType.LLM_FILTER, PipelineBlockType.LLM_TRANSFORM,
                    PipelineBlockType.LLM_SCORE,
                }
                if target.btype in _LLM_LOGIC_TYPES:
                    self._configure_llm_logic_block(target)
                elif target.btype in _LOGIC_TYPES:
                    self._configure_logic_block(target)
                else:
                    self._configure_context_block(target)
                return

        if chosen == act_clr:
            if QMessageBox.question(
                self, "Clear Pipeline",
                "Remove ALL blocks and connections?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                self.clear_all()
