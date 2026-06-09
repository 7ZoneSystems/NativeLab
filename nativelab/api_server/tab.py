from __future__ import annotations

from typing import Any

from nativelab.imports.qt_compat import (
    QApplication,
    QColor,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    Qt,
    QTimer,
)

from nativelab.UI.UI_const import C
from nativelab.UI.icons import icon, set_button_icon, set_label_icon
from nativelab.UI.toggle import ToggleSwitch

from .catalog import model_catalog
from .config import ACTIVE_MODEL_REF, ApiServerConfig, generate_api_key
from .server import NativeLabApiServer


class ApiServerTab(QWidget):
    """Dev page for hosting NativeLab models behind OpenAI/Anthropic APIs."""

    def __init__(self, endpoints=None, parent=None):
        super().__init__(parent)
        self._endpoints = endpoints
        self._config = ApiServerConfig.load()
        self._server: NativeLabApiServer | None = None
        self._models: list[dict[str, Any]] = []
        self._build()
        self._load_config_to_ui()
        self.refresh_models()
        self._refresh_status()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(1800)

    def set_endpoints(self, endpoints) -> None:
        self._endpoints = endpoints
        if self._server is not None:
            self._server.endpoints = endpoints
        self._refresh_status()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("chat_scroll")
        body = QWidget()
        body.setObjectName("chat_container")
        root = QVBoxLayout(body)
        root.setContentsMargins(22, 18, 22, 22)
        root.setSpacing(12)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        hdr = QLabel("API Server")
        set_label_icon(hdr, "server", "API Server", 18)
        hdr.setStyleSheet("font-size:16px;font-weight:bold;margin-bottom:2px;")
        root.addWidget(hdr)

        sub = QLabel(
            "Host the active NativeLab engine or a selected downloaded model on localhost and WiFi/LAN endpoints."
        )
        sub.setWordWrap(True)
        sub.setObjectName("txt2_small")
        root.addWidget(sub)

        root.addWidget(self._build_config_card())
        root.addWidget(self._build_endpoint_card())
        root.addWidget(self._build_catalog_card(), 1)
        root.addStretch()

    def _build_config_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("tab_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(self._section("SERVER"))

        row1 = QHBoxLayout()
        row1.setSpacing(12)
        self.protocol_combo = QComboBox()
        self.protocol_combo.addItem("OpenAI + Anthropic", "both")
        self.protocol_combo.addItem("OpenAI-compatible", "openai")
        self.protocol_combo.addItem("Anthropic-compatible", "anthropic")
        self.protocol_combo.setFixedHeight(30)
        self.bind_combo = QComboBox()
        self.bind_combo.addItem("WiFi/LAN  (0.0.0.0)", "0.0.0.0")
        self.bind_combo.addItem("Localhost only  (127.0.0.1)", "127.0.0.1")
        self.bind_combo.setFixedHeight(30)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setFixedWidth(96)
        self.port_spin.setFixedHeight(30)
        row1.addLayout(self._field("Mode", self.protocol_combo), 1)
        row1.addLayout(self._field("Bind", self.bind_combo), 1)
        row1.addLayout(self._field("Port", self.port_spin))
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(12)
        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(30)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.btn_refresh_models = QPushButton("Refresh Models")
        set_button_icon(self.btn_refresh_models, "refresh-cw", "Refresh Models")
        self.btn_refresh_models.clicked.connect(self.refresh_models)
        row2.addLayout(self._field("Hosted model", self.model_combo), 1)
        row2.addWidget(self.btn_refresh_models)
        layout.addLayout(row2)

        toggles = QHBoxLayout()
        self.auto_load_toggle = ToggleSwitch("Auto-load requested model")
        self.require_key_toggle = ToggleSwitch("Require API keys")
        toggles.addWidget(self.auto_load_toggle)
        toggles.addWidget(self.require_key_toggle)
        toggles.addStretch()
        layout.addLayout(toggles)

        key_hdr = QHBoxLayout()
        key_hdr.addWidget(self._section("IP-BOUND API KEYS"))
        key_hdr.addStretch()
        self.show_keys_toggle = ToggleSwitch("Show keys")
        self.show_keys_toggle.toggled.connect(self._toggle_key_visibility)
        key_hdr.addWidget(self.show_keys_toggle)
        layout.addLayout(key_hdr)
        key_row = QHBoxLayout()
        key_row.setSpacing(12)
        self.local_key_edit = QLineEdit()
        self.local_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.local_key_edit.setPlaceholderText("Localhost key")
        self.lan_key_edit = QLineEdit()
        self.lan_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.lan_key_edit.setPlaceholderText("WiFi/LAN key")
        self.btn_gen_local = QPushButton("Generate")
        self.btn_gen_lan = QPushButton("Generate")
        set_button_icon(self.btn_gen_local, "key", "Generate")
        set_button_icon(self.btn_gen_lan, "key", "Generate")
        self.btn_gen_local.clicked.connect(lambda: self.local_key_edit.setText(generate_api_key()))
        self.btn_gen_lan.clicked.connect(lambda: self.lan_key_edit.setText(generate_api_key()))
        key_row.addLayout(self._field("127.0.0.1 key", self.local_key_edit), 1)
        key_row.addWidget(self.btn_gen_local)
        key_row.addLayout(self._field("WiFi/LAN key", self.lan_key_edit), 1)
        key_row.addWidget(self.btn_gen_lan)
        layout.addLayout(key_row)

        actions = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_start = QPushButton("Start Server")
        self.btn_stop = QPushButton("Stop Server")
        self.btn_start.setObjectName("btn_send")
        set_button_icon(self.btn_save, "save", "Save")
        set_button_icon(self.btn_start, "play", "Start Server")
        set_button_icon(self.btn_stop, "stop-circle", "Stop Server")
        self.btn_save.clicked.connect(self.save_config)
        self.btn_start.clicked.connect(self.start_server)
        self.btn_stop.clicked.connect(self.stop_server)
        actions.addWidget(self.btn_save)
        actions.addWidget(self.btn_start)
        actions.addWidget(self.btn_stop)
        actions.addStretch()
        layout.addLayout(actions)
        return card

    def _build_endpoint_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("tab_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(self._section("ENDPOINTS"))

        local_row = QHBoxLayout()
        self.local_url = QLabel("")
        self.local_url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_copy_local = QPushButton("Copy Local")
        set_button_icon(self.btn_copy_local, "copy", "Copy Local")
        self.btn_copy_local.clicked.connect(lambda: self._copy(self.local_url.text()))
        local_row.addWidget(QLabel("Local:"))
        local_row.addWidget(self.local_url, 1)
        local_row.addWidget(self.btn_copy_local)
        layout.addLayout(local_row)

        lan_row = QHBoxLayout()
        self.lan_url = QLabel("")
        self.lan_url.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_copy_lan = QPushButton("Copy WiFi/LAN")
        set_button_icon(self.btn_copy_lan, "copy", "Copy WiFi/LAN")
        self.btn_copy_lan.clicked.connect(lambda: self._copy(self.lan_url.text()))
        lan_row.addWidget(QLabel("WiFi/LAN:"))
        lan_row.addWidget(self.lan_url, 1)
        lan_row.addWidget(self.btn_copy_lan)
        layout.addLayout(lan_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("txt2_small")
        layout.addWidget(self.status_label)
        return card

    def _build_catalog_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("tab_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(self._section("SMART MODEL CATALOG"))

        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(170)
        self.model_list.currentItemChanged.connect(self._on_catalog_selected)
        layout.addWidget(self.model_list)

        self.capability_box = QTextEdit()
        self.capability_box.setReadOnly(True)
        self.capability_box.setObjectName("log_te")
        self.capability_box.setMinimumHeight(110)
        layout.addWidget(self.capability_box)
        return card

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-size:12px;font-weight:bold;letter-spacing:0.4px;")
        return label

    @staticmethod
    def _field(label: str, widget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(4)
        lab = QLabel(label)
        lab.setObjectName("txt2_xs")
        box.addWidget(lab)
        box.addWidget(widget)
        return box

    def _load_config_to_ui(self):
        idx = self.protocol_combo.findData(self._config.protocol)
        self.protocol_combo.setCurrentIndex(idx if idx >= 0 else 0)
        idx = self.bind_combo.findData(self._config.host)
        self.bind_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.port_spin.setValue(int(self._config.port))
        self.auto_load_toggle.setChecked(bool(self._config.auto_load_model))
        self.require_key_toggle.setChecked(bool(self._config.require_api_key))
        self.local_key_edit.setText(self._config.local_api_key)
        self.lan_key_edit.setText(self._config.lan_api_key)
        self._toggle_key_visibility(self.show_keys_toggle.isChecked())

    def _apply_ui_to_config(self, fill_missing_keys: bool = True):
        self._config.protocol = str(self.protocol_combo.currentData() or "both")
        self._config.host = str(self.bind_combo.currentData() or "0.0.0.0")
        self._config.port = int(self.port_spin.value())
        self._config.model_ref = str(self.model_combo.currentData() or ACTIVE_MODEL_REF)
        self._config.auto_load_model = bool(self.auto_load_toggle.isChecked())
        self._config.require_api_key = bool(self.require_key_toggle.isChecked())
        self._config.local_api_key = self.local_key_edit.text().strip()
        self._config.lan_api_key = self.lan_key_edit.text().strip()
        if fill_missing_keys:
            if not self._config.local_api_key:
                self._config.local_api_key = generate_api_key()
                self.local_key_edit.setText(self._config.local_api_key)
            if not self._config.lan_api_key:
                self._config.lan_api_key = generate_api_key()
                self.lan_key_edit.setText(self._config.lan_api_key)

    def save_config(self):
        self._apply_ui_to_config(fill_missing_keys=True)
        self._config.save()
        self._refresh_status()
        QMessageBox.information(self, "Saved", "API server settings saved.")

    def refresh_models(self):
        current = self.model_combo.currentData() or self._config.model_ref
        self._models = model_catalog()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("Active NativeLab engine", ACTIVE_MODEL_REF)
        for row in self._models:
            caps = row.get("capabilities", {})
            img = "image" if caps.get("image_input") else "text"
            self.model_combo.addItem(f"{row['name']}  [{row['backend']} / {img}]", row["native_ref"])
        idx = self.model_combo.findData(current)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.model_combo.blockSignals(False)

        self.model_list.clear()
        for row in self._models:
            caps = row.get("capabilities", {})
            state = "ok" if caps.get("image_input") else "idle"
            item = QListWidgetItem(
                icon("vision") if caps.get("vision_detected") else icon("server"),
                f"{row['id']}  -  {row['backend']}  -  {caps.get('image_status', 'text_only')}",
            )
            item.setData(Qt.ItemDataRole.UserRole, row)
            item.setToolTip(str(row.get("native_ref") or ""))
            item.setForeground(COLOR_OK() if state == "ok" else COLOR_TXT2())
            self.model_list.addItem(item)
        if self.model_list.count():
            self.model_list.setCurrentRow(0)
        else:
            self.capability_box.setPlainText("No local models were found in the NativeLab model registry.")
        self._refresh_status()

    def start_server(self):
        if self._endpoints is None:
            QMessageBox.warning(self, "No Runtime", "NativeLab runtime endpoints are not ready.")
            return
        self._apply_ui_to_config(fill_missing_keys=True)
        self._config.save()
        if self._server and self._server.is_running:
            self._server.stop()
        try:
            self._server = NativeLabApiServer(self._endpoints, self._config)
            self._server.start()
            self.port_spin.setValue(int(self._config.port))
            self._config.save()
        except Exception as exc:
            QMessageBox.critical(self, "API Server Failed", str(exc))
            self._server = None
        self._refresh_status()

    def stop_server(self):
        if self._server is not None:
            self._server.stop()
        self._refresh_status()

    def shutdown(self):
        if hasattr(self, "_timer"):
            self._timer.stop()
        self.stop_server()

    def _refresh_status(self):
        running = bool(self._server and self._server.is_running)
        if not running:
            self._apply_ui_to_config(fill_missing_keys=False)
        self.local_url.setText(self._config.local_base_url)
        if self._config.host == "127.0.0.1":
            self.lan_url.setText("Disabled while bound to 127.0.0.1")
        else:
            self.lan_url.setText(self._config.lan_base_url)
        model = self.model_combo.currentText() or "Active NativeLab engine"
        self.status_label.setText(
            f"{'Running' if running else 'Stopped'}  |  {self._config.bind_label}  |  {self._config.protocol}  |  {model}"
        )
        for widget in (
            self.protocol_combo, self.bind_combo, self.port_spin, self.model_combo,
            self.auto_load_toggle, self.require_key_toggle, self.local_key_edit,
            self.lan_key_edit, self.btn_gen_local, self.btn_gen_lan,
            self.btn_refresh_models, self.btn_save,
        ):
            widget.setEnabled(not running)
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def _toggle_key_visibility(self, visible: bool):
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        self.local_key_edit.setEchoMode(mode)
        self.lan_key_edit.setEchoMode(mode)

    def _on_model_changed(self):
        self._refresh_status()

    def _on_catalog_selected(self, current, _previous):
        row = current.data(Qt.ItemDataRole.UserRole) if current else None
        if not isinstance(row, dict):
            return
        caps = row.get("capabilities", {})
        lines = [
            f"ID: {row.get('id', '')}",
            f"Native ref: {row.get('native_ref', '')}",
            f"Backend: {row.get('backend', '')}",
            f"Family: {row.get('family', '')}",
            f"Quant: {row.get('quant', '')}",
            f"Context: {row.get('ctx', 0)}",
            f"Vision detected: {bool(caps.get('vision_detected'))}",
            f"Image input: {bool(caps.get('image_input'))}",
            f"Image status: {caps.get('image_status', 'text_only')}",
            f"OpenAI chat: {bool(caps.get('openai_chat_completions'))}",
            f"Anthropic messages: {bool(caps.get('anthropic_messages'))}",
        ]
        self.capability_box.setPlainText("\n".join(lines))

    @staticmethod
    def _copy(text: str):
        value = str(text or "").strip()
        if value:
            QApplication.clipboard().setText(value)


def COLOR_OK():
    return QColor(C["ok"])


def COLOR_TXT2():
    return QColor(C["txt2"])
