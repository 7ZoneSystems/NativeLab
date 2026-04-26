"""
flowpreview.py — animated dot flow preview for the Pipeline Canvas.

Dots spawn at the INPUT block and travel along connections until they
reach OUTPUT or exhaust all reachable paths.

Color legend (matches canvas port routing):
  green  (#a6e3a1) — normal / TRUE arm (E-port)
  red    (#f38ba8) — FALSE arm (W-port)
  yellow (#f9e2af) — N/S arms, MERGE output
  purple (#cba6f7) — SPLIT fan-out
  grey   (#555555) — FILTER drop-path
"""
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from nativelab.UI.UI_const import C
from nativelab.pipelinebuilder.blck_typ import PipelineBlockType, PipelineConnection
from nativelab.pipelinebuilder.pipblck import PipelineBlock
from typing import List, Dict, Optional


# ── block type sets ───────────────────────────────────────────────────────────
_BRANCHING = {
    PipelineBlockType.IF_ELSE,   PipelineBlockType.LLM_IF,
    PipelineBlockType.SWITCH,    PipelineBlockType.LLM_SWITCH,
    PipelineBlockType.LLM_SCORE,
}
_FILTER  = {PipelineBlockType.FILTER,  PipelineBlockType.LLM_FILTER}
_SPLIT   = {PipelineBlockType.SPLIT}
_MERGE   = {PipelineBlockType.MERGE}


class _FlowDot:
    """One animated dot travelling along a single connection arc."""
    __slots__ = ("conn", "t", "color", "radius", "done")

    def __init__(self, conn: PipelineConnection,
                 color: str = "#a6e3a1", radius: int = 6):
        self.conn   = conn
        self.t      = 0.0     # 0.0 → 1.0 along the bezier path
        self.color  = color
        self.radius = radius
        self.done   = False


class FlowPreviewController(QObject):
    """
    Manages the animated flow-preview overlay on PipelineCanvas.

    Usage
    -----
        ctrl = FlowPreviewController(canvas, blocks, connections, parent=self)
        ctrl.finished.connect(some_callback)
        ctrl.start()
        # ...
        ctrl.stop()   # or wait for finished signal
    """

    finished = pyqtSignal()   # emitted when all dots have reached OUTPUT / died

    _SPEED     = 0.014   # progress added per 16 ms tick  → ~1.1 s per hop
    _TICK_MS   = 16      # ~60 fps
    _IDLE_WAIT = 35      # extra ticks to wait after last dot dies before stopping

    def __init__(self, canvas,
                 blocks:      List[PipelineBlock],
                 connections: List[PipelineConnection],
                 parent=None):
        super().__init__(parent)
        self._canvas      = canvas
        self._blocks: Dict[int, PipelineBlock] = {b.bid: b for b in blocks}
        self._connections = connections

        # adjacency: from_block_id → [connections leaving it]
        self._adj: Dict[int, List[PipelineConnection]] = {}
        for c in connections:
            self._adj.setdefault(c.from_block_id, []).append(c)

        self._dots:    List[_FlowDot] = []
        self._visited: set            = set()   # (from_bid, to_bid) loop-guard
        self._idle_ticks              = 0

        self._timer = QTimer(self)
        self._timer.setInterval(self._TICK_MS)
        self._timer.timeout.connect(self._tick)

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        """Spawn dots from the INPUT block and begin animation."""
        self._dots.clear()
        self._visited.clear()
        self._idle_ticks = 0

        inp = next(
            (b for b in self._blocks.values()
             if b.btype == PipelineBlockType.INPUT), None)
        if inp is None:
            return

        col_start = C.get("ok", "#a6e3a1")
        for conn in self._adj.get(inp.bid, []):
            self._spawn(conn, col_start)

        self._canvas.preview_dots = self._dots
        self._timer.start()

    def stop(self):
        """Immediately halt animation and clear all dots from the canvas."""
        self._timer.stop()
        self._dots.clear()
        self._canvas.preview_dots = []
        self._canvas.update()

    # ── internals ─────────────────────────────────────────────────────────────

    def _spawn(self, conn: PipelineConnection, color: str, radius: int = 6):
        """Add a new dot unless this edge was already visited (loop guard)."""
        key = (conn.from_block_id, conn.to_block_id)
        if key in self._visited:
            return
        self._visited.add(key)
        self._dots.append(_FlowDot(conn, color, radius))

    def _tick(self):
        new_spawns: List[tuple] = []   # (conn, color) to spawn after tick

        for dot in self._dots:
            if dot.done:
                continue
            dot.t += self._SPEED
            if dot.t < 1.0:
                continue

            dot.t    = 1.0
            dot.done = True

            dest = self._blocks.get(dot.conn.to_block_id)
            if dest is None or dest.btype == PipelineBlockType.OUTPUT:
                continue   # reached terminal — no children

            outgoing = self._adj.get(dest.bid, [])
            if not outgoing:
                continue

            # ── colour / fan-out logic mirrors executionWorker routing ────────
            if dest.btype in _BRANCHING:
                # Show ALL arms; colour by port: E=TRUE(green) W=FALSE(red) else yellow
                _ct = C.get("ok",   "#a6e3a1")
                _cf = C.get("err",  "#f38ba8")
                _co = C.get("warn", "#f9e2af")
                for c in outgoing:
                    arm_col = _ct if c.from_port == "E" else \
                              _cf if c.from_port == "W" else _co
                    new_spawns.append((c, arm_col))

            elif dest.btype in _FILTER:
                # First outgoing = pass (bright green), rest = drop (grey)
                _cp = C.get("ok", "#a6e3a1")
                for i, c in enumerate(outgoing):
                    new_spawns.append((c, _cp if i == 0 else "#666666"))

            elif dest.btype in _SPLIT:
                # Broadcast same colour to every output
                _cs = C.get("acc", "#cba6f7")
                for c in outgoing:
                    new_spawns.append((c, _cs))

            elif dest.btype in _MERGE:
                _cm = C.get("warn", "#f9e2af")
                for c in outgoing:
                    new_spawns.append((c, _cm))

            else:
                # Normal sequential — inherit colour
                for c in outgoing:
                    new_spawns.append((c, dot.color))

        # Prune finished dots
        self._dots = [d for d in self._dots if not d.done]

        # Spawn children
        for conn, col in new_spawns:
            self._spawn(conn, col)

        # Auto-stop when canvas is clear
        if not self._dots:
            self._idle_ticks += 1
            if self._idle_ticks >= self._IDLE_WAIT:
                self.stop()
                self.finished.emit()
                return
        else:
            self._idle_ticks = 0

        self._canvas.preview_dots = self._dots
        self._canvas.update()