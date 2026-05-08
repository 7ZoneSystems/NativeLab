"""
Labs tab container.

Hosts the left-side feature navigator and the stacked content area. Each lab
feature lives in its own `feature_name.py` module under this package and only
needs to expose a panel class with `LAB_NAME`, `LAB_ICON`, and a
`set_endpoints(endpoints)` method.

The host application calls `LabsTab.set_endpoints(endpoints)` once after the
engines are wired; from that point on every panel reads/writes engine state
through the shared `LabEndpoints` instance — never through `MainWindow`.
"""
from __future__ import annotations

from typing import Optional, Type

from nativelab.imports.import_global import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QStackedWidget,
    QListWidget, QListWidgetItem, Qt,
)

from .endpoints import LabEndpoints
from .pytodoc import PyToDocPanel


# Ordered list of lab features. To register a new lab, drop a module in this
# package that exposes a panel class (with LAB_NAME, LAB_ICON, set_endpoints)
# and append it here.
LAB_FEATURES: list[Type[QWidget]] = [
    PyToDocPanel,
]


class LabsTab(QWidget):
    """Sidebar navigation + stacked content area for Labs features."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._endpoints: Optional[LabEndpoints] = None
        self._panels: dict[str, QWidget] = {}
        self._build()

    # ── external API ─────────────────────────────────────────────────────────
    def set_endpoints(self, endpoints: LabEndpoints):
        """Forward the shared endpoint instance to every registered panel."""
        self._endpoints = endpoints
        for panel in self._panels.values():
            if hasattr(panel, "set_endpoints"):
                panel.set_endpoints(endpoints)

    # ── build ────────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("labs_sidebar")
        sidebar.setFixedWidth(172)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.setSpacing(0)

        labs_lbl = QLabel("LABS")
        labs_lbl.setObjectName("labs_sidebar_hdr")
        sb_layout.addWidget(labs_lbl)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("labs_nav")
        self.nav_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav_list.setSpacing(2)

        self.stack = QStackedWidget()
        self.stack.setObjectName("labs_stack")

        for PanelClass in LAB_FEATURES:
            name = getattr(PanelClass, "LAB_NAME", PanelClass.__name__)
            icon = getattr(PanelClass, "LAB_ICON", "•")
            item = QListWidgetItem(f"  {icon}  {name}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.nav_list.addItem(item)

            panel = PanelClass()
            self._panels[name] = panel
            self.stack.addWidget(panel)

        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        sb_layout.addWidget(self.nav_list)
        sb_layout.addStretch()

        root.addWidget(sidebar)
        root.addWidget(self.stack, 1)

        self.nav_list.setCurrentRow(0)

    def _on_nav_changed(self, row: int):
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)
