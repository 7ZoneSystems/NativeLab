from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path
from nativelab.imports.qt_compat import QAction, QIcon, QLabel, QPushButton, QSize, QTabWidget, QWidget


ICON_DIR = Path(__file__).resolve().parents[1] / "assets" / "icons"
ICON_CACHE_DIR = Path(tempfile.gettempdir()) / "nativelab-svg-icons"
LIGHT_ICON_COLOR = "#111111"
DARK_ICON_COLOR = "#f5f5f5"

ICON_ALIASES = {
    "api": "globe",
    "appearance": "palette",
    "chat": "message-circle",
    "clear": "trash-2",
    "code": "code-2",
    "config": "settings",
    "coding": "code-2",
    "delete": "trash-2",
    "discord": "discord",
    "docs": "file-text",
    "done": "circle-check",
    "error": "circle-x",
    "eject": "power-off",
    "export": "upload",
    "general": "message-square",
    "hf": "huggingface",
    "huggingface": "huggingface",
    "integrations": "plug",
    "labs": "flask-conical",
    "load": "folder-open",
    "logs": "bug",
    "input": "log-in",
    "mcp": "plug",
    "web_search": "search",
    "models": "folder",
    "ollama": "ollama",
    "pdf": "file-text",
    "pipeline": "workflow",
    "output": "log-out",
    "reasoning": "brain",
    "reference": "paperclip",
    "save": "save",
    "server": "server",
    "secondary": "shuffle",
    "summary": "clipboard-list",
    "summarization": "file-text",
    "text": "file-text",
    "vision": "image",
    "warn": "triangle-alert",
    "whatsapp": "whatsapp",
}

BRAND_ICON_NAMES = {"huggingface", "ollama"}

ROLE_ICON_NAMES = {
    "general": "message-square",
    "reasoning": "brain",
    "summarization": "file-text",
    "coding": "code-2",
    "secondary": "shuffle",
}

STATUS_ICON_NAMES = {
    "ok": "circle-check",
    "loaded": "circle-check",
    "warn": "circle-alert",
    "warning": "circle-alert",
    "idle": "circle",
    "empty": "circle",
    "err": "circle-x",
    "error": "circle-x",
    "loading": "loader-circle",
    "api": "globe",
    "server": "server",
}


def _resolved_icon_name(name: str) -> str:
    resolved = ICON_ALIASES.get(name, name)
    return resolved


def icon_path(name: str) -> Path:
    return ICON_DIR / f"{_resolved_icon_name(name)}.svg"


def _current_theme() -> str:
    try:
        from nativelab.UI import UI_const
        return getattr(UI_const, "CURRENT_THEME", "light")
    except Exception:
        return "light"


def icon_color() -> str:
    return DARK_ICON_COLOR if _current_theme() == "dark" else LIGHT_ICON_COLOR


def _colorize_svg(svg: str, color: str) -> str:
    svg = svg.replace("currentColor", color)
    svg = re.sub(r'stroke="(?!none|transparent)[^"]*"', f'stroke="{color}"', svg)
    svg = re.sub(r"stroke='(?!none|transparent)[^']*'", f"stroke='{color}'", svg)
    svg = re.sub(r'fill="(?!none|transparent|url\()[^"]*"', f'fill="{color}"', svg)
    svg = re.sub(r"fill='(?!none|transparent|url\()[^']*'", f"fill='{color}'", svg)
    return svg


def themed_icon_path(name: str) -> Path:
    source = icon_path(name)
    if not source.exists():
        return source
    theme = _current_theme()
    resolved = _resolved_icon_name(name)
    color_key = "brand" if resolved in BRAND_ICON_NAMES else icon_color().lstrip("#")
    target = ICON_CACHE_DIR / theme / color_key / f"{resolved}.svg"
    try:
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_text(encoding="utf-8")
        rendered = raw if resolved in BRAND_ICON_NAMES else _colorize_svg(raw, icon_color())
        target.write_text(rendered, encoding="utf-8")
        return target
    except Exception:
        return source


def icon(name: str) -> QIcon:
    path = themed_icon_path(name)
    return QIcon(str(path)) if path.exists() else QIcon()


def role_icon_name(role: str) -> str:
    return ROLE_ICON_NAMES.get(role, "models")


def role_icon(role: str) -> QIcon:
    return icon(role_icon_name(role))


def status_icon_name(state: str) -> str:
    return STATUS_ICON_NAMES.get(state, "circle")


def status_icon(state: str) -> QIcon:
    return icon(status_icon_name(state))


def icon_size(size: int = 16) -> QSize:
    return QSize(int(size), int(size))


def set_button_icon(button: QPushButton, name: str, text: str | None = None, size: int = 16) -> QPushButton:
    button.setProperty("_nl_icon_name", name)
    button.setProperty("_nl_icon_size", size)
    if text is not None:
        button.setProperty("_nl_icon_text", text)
    button.setIcon(icon(name))
    button.setIconSize(icon_size(size))
    if text is not None:
        button.setText(text)
    return button


def set_tab_icon(tabs: QTabWidget, index: int, name: str, text: str | None = None, size: int = 16) -> None:
    tabs.setTabIcon(index, icon(name))
    tabs.setIconSize(icon_size(size))
    if text is not None:
        tabs.setTabText(index, text)


def add_menu_action(menu, text: str, name: str) -> QAction:
    action = QAction(icon(name), text, menu)
    menu.addAction(action)
    return action


def set_label_icon(label: QLabel, name: str, text: str, size: int = 16) -> QLabel:
    label.setProperty("_nl_label_icon_name", name)
    label.setProperty("_nl_label_icon_text", text)
    label.setProperty("_nl_label_icon_size", size)
    path = themed_icon_path(name)
    if path.exists():
        suffix = f"&nbsp;{html.escape(text)}" if text else ""
        label.setText(
            f"<img src='{html.escape(str(path), quote=True)}' "
            f"width='{size}' height='{size}' style='vertical-align:middle;'>"
            f"{suffix}"
        )
    else:
        label.setText(text)
    return label


def set_status_label(label: QLabel, text: str, state: str = "idle", size: int = 14) -> QLabel:
    label.setProperty("state", state)
    label.setProperty("_nl_status_text", text)
    label.setProperty("_nl_status_size", size)
    set_label_icon(label, status_icon_name(state), text, size)
    try:
        label.style().unpolish(label)
        label.style().polish(label)
    except Exception:
        pass
    return label


def refresh_widget_icons(root: QWidget) -> None:
    widgets = [root] + list(root.findChildren(QWidget))
    for widget in widgets:
        if isinstance(widget, QPushButton):
            name = widget.property("_nl_icon_name")
            if name:
                text = widget.property("_nl_icon_text")
                sz = widget.property("_nl_icon_size") or 16
                set_button_icon(widget, str(name), str(text) if text is not None else None, int(sz))  # type: ignore[arg-type]
        elif isinstance(widget, QLabel):
            status_text = widget.property("_nl_status_text")
            if status_text is not None:
                state = widget.property("state") or "idle"
                sz = widget.property("_nl_status_size") or 14
                set_status_label(widget, str(status_text), str(state), int(sz))  # type: ignore[arg-type]
                continue
            name = widget.property("_nl_label_icon_name")
            if name:
                text = widget.property("_nl_label_icon_text") or ""
                sz = widget.property("_nl_label_icon_size") or 16
                set_label_icon(widget, str(name), str(text), int(sz))  # type: ignore[arg-type]
