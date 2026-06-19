"""
Dev tab page: LAN Device Discovery.

Scans the local network for PhonoLab Android devices,
shows their status, and registers them as API model endpoints.
"""

from __future__ import annotations

import threading
from typing import Optional

from nativelab.imports.qt_compat import (
    QApplication,
    QColor,
    QComboBox,
    QDialog,
    QEvent,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    Qt,
    QTimer,
)

from nativelab.UI.UI_const import C
from nativelab.UI.icons import icon, set_button_icon, set_label_icon

from .device_discovery import (
    DiscoveredDevice,
    scan_network,
    save_devices,
    load_cached_devices,
    refresh_device,
    test_connection,
    auto_connect_device,
    get_device_status,
    get_device_models,
    load_model_on_device,
    update_device_config,
    get_device_queue,
)
from .config import detect_lan_ip
from nativelab.Model.APImodels import ApiConfig, getapi_registry


class DevicesTab(QWidget):
    """Dev page for discovering and registering PhonoLab LAN devices."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._devices: list[DiscoveredDevice] = []
        self._scanning = False
        self._build()
        self._load_cached()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_statuses)
        self._refresh_timer.start(30_000)  # Refresh every 30s

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

        # Header
        hdr = QLabel("LAN Devices")
        set_label_icon(hdr, "network", "LAN Devices", 18)
        hdr.setStyleSheet("font-size:16px;font-weight:bold;margin-bottom:2px;")
        root.addWidget(hdr)

        sub = QLabel("Discover PhonoLab Android devices on your local network. Register them as API model endpoints for distributed inference.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{C['txt2']};font-size:12px;margin-bottom:8px;")
        root.addWidget(sub)

        # Scan controls
        scan_row = QHBoxLayout()
        scan_row.setSpacing(8)

        self.btn_scan = QPushButton("  Scan Network")
        set_button_icon(self.btn_scan, "search")
        self.btn_scan.setMinimumHeight(34)
        self.btn_scan.setStyleSheet(f"""
            QPushButton {{
                background:{C['acc']};color:#fff;border:none;border-radius:6px;
                font-weight:bold;font-size:13px;padding:0 16px;
            }}
            QPushButton:hover {{ background:{C['acc2']}; }}
            QPushButton:disabled {{ background:{C['txt3']}; }}
        """)
        self.btn_scan.clicked.connect(self._start_scan)
        scan_row.addWidget(self.btn_scan)

        self.btn_refresh_all = QPushButton("  Refresh All")
        set_button_icon(self.btn_refresh_all, "refresh-cw")
        self.btn_refresh_all.setMinimumHeight(34)
        self.btn_refresh_all.setStyleSheet(f"""
            QPushButton {{
                background:{C['surface']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                font-weight:bold;font-size:13px;padding:0 16px;
            }}
            QPushButton:hover {{ border-color:{C['acc']}; }}
        """)
        self.btn_refresh_all.clicked.connect(self._refresh_statuses)
        scan_row.addWidget(self.btn_refresh_all)

        self.lbl_subnet = QLabel(f"Subnet: {detect_lan_ip().rsplit('.', 1)[0]}.0/24")
        self.lbl_subnet.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
        scan_row.addWidget(self.lbl_subnet)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(f"color:{C['txt2']};font-size:12px;")
        scan_row.addWidget(self.lbl_status, 1)

        root.addLayout(scan_row)

        # Device list
        list_hdr = QLabel("DISCOVERED DEVICES")
        list_hdr.setStyleSheet(f"color:{C['txt3']};font-size:10px;font-weight:bold;letter-spacing:0.08em;margin-top:8px;")
        root.addWidget(list_hdr)

        self.device_list = QListWidget()
        self.device_list.setMinimumHeight(200)
        self.device_list.setStyleSheet(f"""
            QListWidget {{
                background:{C['bg']};border:1px solid {C['bdr']};border-radius:6px;
            }}
            QListWidget::item {{
                padding:8px;border-bottom:1px solid {C['divider']};
            }}
            QListWidget::item:selected {{
                background:{C['sel']};
            }}
        """)
        root.addWidget(self.device_list, 1)

        # Action buttons - row 1
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_register = QPushButton("  Register as API Model")
        set_button_icon(self.btn_register, "plus")
        self.btn_register.setMinimumHeight(34)
        self.btn_register.setEnabled(False)
        self.btn_register.setStyleSheet(f"""
            QPushButton {{
                background:{C['acc']};color:#fff;border:none;border-radius:6px;
                font-weight:bold;font-size:13px;padding:0 16px;
            }}
            QPushButton:hover {{ background:{C['acc2']}; }}
            QPushButton:disabled {{ background:{C['txt3']};color:{C['txt2']}; }}
        """)
        self.btn_register.clicked.connect(self._register_selected)
        btn_row.addWidget(self.btn_register)

        self.btn_set_key = QPushButton("  Set API Key")
        set_button_icon(self.btn_set_key, "clipboard-list")
        self.btn_set_key.setMinimumHeight(34)
        self.btn_set_key.setEnabled(False)
        self.btn_set_key.setStyleSheet(f"""
            QPushButton {{
                background:{C['surface']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                font-weight:bold;font-size:13px;padding:0 16px;
            }}
            QPushButton:hover {{ border-color:{C['acc']}; }}
            QPushButton:disabled {{ background:{C['txt3']};color:{C['txt2']}; }}
        """)
        self.btn_set_key.clicked.connect(self._set_api_key)
        btn_row.addWidget(self.btn_set_key)

        self.btn_test = QPushButton("  Test")
        set_button_icon(self.btn_test, "circle-check")
        self.btn_test.setMinimumHeight(34)
        self.btn_test.setEnabled(False)
        self.btn_test.setStyleSheet(f"""
            QPushButton {{
                background:{C['surface']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                font-weight:bold;font-size:13px;padding:0 16px;
            }}
            QPushButton:hover {{ border-color:{C['acc']}; }}
            QPushButton:disabled {{ background:{C['txt3']};color:{C['txt2']}; }}
        """)
        self.btn_test.clicked.connect(self._test_connection)
        btn_row.addWidget(self.btn_test)

        root.addLayout(btn_row)

        # Action buttons - row 2
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(8)

        self.btn_refresh = QPushButton("  Refresh")
        set_button_icon(self.btn_refresh, "refresh")
        self.btn_refresh.setMinimumHeight(34)
        self.btn_refresh.setEnabled(False)
        self.btn_refresh.setStyleSheet(f"""
            QPushButton {{
                background:{C['surface']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                font-weight:bold;font-size:13px;padding:0 16px;
            }}
            QPushButton:hover {{ border-color:{C['acc']}; }}
            QPushButton:disabled {{ background:{C['txt3']};color:{C['txt2']}; }}
        """)
        self.btn_refresh.clicked.connect(self._refresh_selected)
        btn_row2.addWidget(self.btn_refresh)

        self.btn_load_model = QPushButton("  Load Model")
        set_button_icon(self.btn_load_model, "blocks")
        self.btn_load_model.setMinimumHeight(34)
        self.btn_load_model.setEnabled(False)
        self.btn_load_model.setStyleSheet(f"""
            QPushButton {{
                background:{C['surface']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                font-weight:bold;font-size:13px;padding:0 16px;
            }}
            QPushButton:hover {{ border-color:{C['acc']}; }}
            QPushButton:disabled {{ background:{C['txt3']};color:{C['txt2']}; }}
        """)
        self.btn_load_model.clicked.connect(self._load_model_on_device)
        btn_row2.addWidget(self.btn_load_model)

        self.btn_remove = QPushButton("  Remove")
        set_button_icon(self.btn_remove, "trash")
        self.btn_remove.setMinimumHeight(34)
        self.btn_remove.setEnabled(False)
        self.btn_remove.setStyleSheet(f"""
            QPushButton {{
                background:transparent;color:{C['err']};border:1px solid {C['err']};border-radius:6px;
                font-weight:bold;font-size:13px;padding:0 16px;
            }}
            QPushButton:hover {{ background:{C['err']};color:#fff; }}
            QPushButton:disabled {{ background:transparent;color:{C['txt3']};border-color:{C['txt3']}; }}
        """)
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_row2.addWidget(self.btn_remove)

        root.addLayout(btn_row2)

        # Device details panel
        detail_hdr = QLabel("DEVICE DETAILS")
        detail_hdr.setStyleSheet(f"color:{C['txt3']};font-size:10px;font-weight:bold;letter-spacing:0.08em;margin-top:8px;")
        root.addWidget(detail_hdr)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(160)
        self.detail_text.setStyleSheet(f"""
            QTextEdit {{
                background:{C['bg']};border:1px solid {C['bdr']};border-radius:6px;
                color:{C['txt']};font-size:12px;font-family:'DM Mono',monospace;padding:8px;
            }}
        """)
        self.detail_text.setPlainText("No device selected.")
        root.addWidget(self.detail_text)

        # Status bar
        self.lbl_info = QLabel(f"My IP: {detect_lan_ip()}  ·  PhonoLab devices advertise on port 8787")
        self.lbl_info.setStyleSheet(f"color:{C['txt3']};font-size:11px;margin-top:4px;")
        root.addWidget(self.lbl_info)

        # Selection handler
        self.device_list.currentRowChanged.connect(self._on_selection_changed)

    def _load_cached(self):
        """Load previously discovered devices."""
        self._devices = load_cached_devices()
        self._populate_list()

    def _start_scan(self):
        """Start network scan in background."""
        if self._scanning:
            return
        self._scanning = True
        self.btn_scan.setEnabled(False)
        self.lbl_status.setText("Scanning...")
        self.lbl_status.setStyleSheet(f"color:{C['acc']};font-size:12px;")

        def _scan():
            found = scan_network(
                on_found=lambda d: QApplication.instance().postEvent(
                    self, _DeviceFoundEvent(d)
                ) if QApplication.instance() else None,
                progress_cb=lambda done, total: QApplication.instance().postEvent(
                    self, _ScanProgressEvent(done, total)
                ) if QApplication.instance() else None,
            )
            QApplication.instance().postEvent(
                self, _ScanCompleteEvent(found)
            ) if QApplication.instance() else None

        threading.Thread(target=_scan, daemon=True).start()

    def _on_scan_progress(self, done: int, total: int):
        self.lbl_status.setText(f"Scanning... {done}/{total}")

    def _on_device_found(self, device: DiscoveredDevice):
        # Add to list if not already present
        for i, d in enumerate(self._devices):
            if d.ip == device.ip:
                self._devices[i] = device
                self._populate_list()
                return
        self._devices.append(device)
        self._populate_list()

    def _on_scan_complete(self, devices: list[DiscoveredDevice]):
        self._scanning = False
        self.btn_scan.setEnabled(True)
        self._devices = devices
        save_devices(devices)
        self._populate_list()
        count = len(devices)
        if count > 0:
            self.lbl_status.setText(f"Found {count} device{'s' if count != 1 else ''}")
            self.lbl_status.setStyleSheet(f"color:{C['ok']};font-size:12px;")
        else:
            self.lbl_status.setText("No devices found")
            self.lbl_status.setStyleSheet(f"color:{C['txt2']};font-size:12px;")

    def _populate_list(self):
        """Populate the device list widget."""
        self.device_list.clear()
        for device in self._devices:
            icon_name = device.status_icon_name
            model = device.model if device.model and device.model != "none" else "No model"
            vision = " [vision]" if device.is_vision else ""

            # Auth status indicator
            auth_tag = ""
            if device.auth_status == "failed":
                auth_tag = " [key changed]"
                icon_name = "circle-alert"
            elif device.auth_status == "not_tested" and device.auth_required:
                auth_tag = " [needs key]"
            elif device.auth_status == "ok":
                auth_tag = ""

            text = f"{device.status} {device.display_name}{auth_tag}\n    {model}{vision}  |  {device.cpu_cores} cores  |  {device.ram_mb}MB RAM"
            item = QListWidgetItem(icon(icon_name), text)
            item.setData(Qt.ItemDataRole.UserRole, device.ip)
            self.device_list.addItem(item)

        self._update_buttons()

    def _on_selection_changed(self, row: int):
        self._update_buttons()
        if 0 <= row < len(self._devices):
            device = self._devices[row]
            self._show_device_detail(device)
        else:
            self.detail_text.setPlainText("No device selected.")

    def _update_buttons(self):
        has_selection = self.device_list.currentRow() >= 0
        self.btn_register.setEnabled(has_selection)
        self.btn_set_key.setEnabled(has_selection)
        self.btn_test.setEnabled(has_selection)
        self.btn_refresh.setEnabled(has_selection)
        self.btn_load_model.setEnabled(has_selection)
        self.btn_remove.setEnabled(has_selection)

    def _show_device_detail(self, device: DiscoveredDevice):
        """Show detailed info for a device with live runtime config."""
        auth_label = {
            "ok": "Verified",
            "failed": "KEY CHANGED - re-auth needed",
            "not_tested": "Not tested",
            "unknown": "Unknown",
        }.get(device.auth_status, device.auth_status)

        lines = [
            f"IP:          {device.ip}:{device.port}",
            f"Name:        {device.name or 'Unknown'}",
            f"Status:      {device.status.upper()}",
            f"Auth:        {auth_label}",
            f"Model:       {device.model or 'None'}",
            f"Vision:      {'Yes' if device.is_vision else 'No'}",
            f"Android:     {device.android_version or '?'}",
            f"CPU Cores:   {device.cpu_cores}",
            f"RAM:         {device.ram_mb} MB",
            f"API URL:     {device.api_url}",
        ]
        if device.api_key:
            lines.append(f"API Key:     {device.api_key[:8]}...")
        self.detail_text.setPlainText("\n".join(lines))

        # Fetch live runtime config in background
        def _fetch_live():
            from .device_discovery import get_device_runtime_config
            live = get_device_runtime_config(device)
            if live:
                live_lines = [
                    "",
                    "--- Live Runtime ---",
                    f"Loaded:      {live.get('loaded', False)}",
                    f"Live Model:  {live.get('model', 'none')}",
                    f"Live Status: {live.get('status', 'unknown')}",
                    f"Context:     {live.get('ctx', '?')}",
                ]
                QApplication.instance().postEvent(
                    self, _DetailUpdateEvent("\n".join(lines + live_lines))
                ) if QApplication.instance() else None

        threading.Thread(target=_fetch_live, daemon=True).start()

    def _register_selected(self):
        """Register selected device as an API model in ApiRegistry."""
        row = self.device_list.currentRow()
        if row < 0 or row >= len(self._devices):
            return

        device = self._devices[row]
        registry = getapi_registry()

        # Check if already registered
        existing = registry.get(device.display_name)
        if existing:
            # Already registered — check if key needs update
            if existing.api_key == device.api_key:
                reply = QMessageBox.question(
                    self, "Already Registered",
                    f"'{device.display_name}' is already registered.\nRe-enter API key?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
            # Key changed or user wants to update — prompt for new key
            api_key = self._prompt_api_key(device)
            if api_key is None:
                return
            device.api_key = api_key
            save_devices(self._devices)
            # Update existing config
            existing = existing.copy(api_key=api_key)
            registry.add(existing)
            self.lbl_status.setText(f"Key updated for {device.display_name}")
            self.lbl_status.setStyleSheet(f"color:{C['ok']};font-size:12px;")
            return

        # New registration — try auto-connect first
        self.lbl_status.setText(f"Connecting to {device.ip}...")
        self.lbl_status.setStyleSheet(f"color:{C['acc']};font-size:12px;")

        def _do_connect():
            ok, msg = auto_connect_device(device)
            QApplication.instance().postEvent(
                self, _AutoConnectEvent(device, ok, msg)
            ) if QApplication.instance() else None

        threading.Thread(target=_do_connect, daemon=True).start()

    def _on_auto_connect(self, device: DiscoveredDevice, ok: bool, msg: str):
        """Handle auto-connect result."""
        if ok:
            # Connected without key or with stored key
            self._do_register(device)
        elif msg == "auth_required" or msg == "key_changed":
            # Need to prompt for key
            reason = "API key required" if msg == "auth_required" else "API key changed on device"
            api_key = self._prompt_api_key(device, reason=reason)
            if api_key is None:
                return
            device.api_key = api_key
            save_devices(self._devices)
            # Test with new key
            ok2, msg2 = test_connection(device, api_key)
            if ok2:
                self._do_register(device)
            else:
                QMessageBox.warning(self, "Connection Failed", f"Still failed: {msg2}")
        else:
            QMessageBox.warning(self, "Connection Failed", msg)

    def _do_register(self, device: DiscoveredDevice):
        """Create ApiConfig and register device."""
        registry = getapi_registry()
        cfg = ApiConfig(
            name=device.display_name,
            provider="PhonoLab",
            model_id=device.model or "phonolab-active",
            api_key=device.api_key,
            base_url=device.api_url,
            api_format="openai",
            max_tokens=512,
            temperature=0.7,
            custom_provider_name="PhonoLab",
            cpu_cores=device.cpu_cores,
            ram_mb=device.ram_mb,
            is_vision=device.is_vision,
            ctx_limit=2048,
            android_ver=device.android_version,
            device_status=device.status,
        )
        registry.add(cfg)
        QMessageBox.information(
            self, "Registered",
            f"'{device.display_name}' registered as API model.\n\n"
            f"Select it in the API Models tab or input bar to use it.\n"
            f"Base URL: {device.api_url}",
        )

    def _prompt_api_key(self, device: DiscoveredDevice, reason: str = "") -> Optional[str]:
        """Show dialog to enter API key. Returns key or None if cancelled."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"API Key for {device.display_name}")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)

        # Info text
        if reason == "key_changed":
            info_text = (
                f"The API key for this device has changed.\n\n"
                f"Find the new key in: PhonoLab > Dev tab > API Server > LAN API Key\n\n"
                f"Device: {device.ip}:{device.port}\n"
                f"Status: {device.status.upper()}"
            )
        elif reason == "auth_required":
            info_text = (
                f"This device requires an API key.\n\n"
                f"Find it in: PhonoLab > Dev tab > API Server > LAN API Key\n\n"
                f"Device: {device.ip}:{device.port}\n"
                f"Status: {device.status.upper()}"
            )
        else:
            info_text = (
                f"Enter the LAN API key from the PhonoLab app.\n\n"
                f"Find it in: PhonoLab > Dev tab > API Server > LAN API Key\n\n"
                f"Device: {device.ip}:{device.port}\n"
                f"Status: {device.status.upper()}"
            )

        info = QLabel(info_text)
        info.setWordWrap(True)
        info.setStyleSheet(f"color:{C['txt']};font-size:13px;")
        layout.addWidget(info)

        key_input = QLineEdit()
        key_input.setPlaceholderText("nl-xxxxxxxxxxxx...")
        key_input.setText(device.api_key or "")
        key_input.setStyleSheet(f"""
            QLineEdit {{
                background:{C['bg']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                padding:8px;font-size:13px;font-family:'DM Mono',monospace;
            }}
            QLineEdit:focus {{ border-color:{C['acc']}; }}
        """)
        layout.addWidget(key_input)

        # Test button
        test_btn = QPushButton("  Test Connection")
        set_button_icon(test_btn, "circle-check")
        test_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C['surface']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                font-weight:bold;font-size:12px;padding:6px 12px;
            }}
            QPushButton:hover {{ border-color:{C['acc']}; }}
        """)
        test_result = QLabel("")
        test_result.setStyleSheet(f"font-size:12px;")

        def _test():
            ok, msg = test_connection(device, key_input.text().strip())
            test_result.setText(msg)
            test_result.setStyleSheet(f"color:{C['ok'] if ok else C['err']};font-size:12px;")

        test_btn.clicked.connect(_test)
        layout.addWidget(test_btn)
        layout.addWidget(test_result)

        # Buttons
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C['acc']};color:#fff;border:none;border-radius:6px;
                font-weight:bold;font-size:13px;padding:8px 20px;
            }}
        """)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C['surface']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                font-weight:bold;font-size:13px;padding:8px 20px;
            }}
        """)
        cancel_btn.clicked.connect(dlg.reject)
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            return key_input.text().strip()
        return None

    def _set_api_key(self):
        """Set/update API key for selected device."""
        row = self.device_list.currentRow()
        if row < 0 or row >= len(self._devices):
            return

        device = self._devices[row]
        api_key = self._prompt_api_key(device)
        if api_key is not None:
            device.api_key = api_key
            save_devices(self._devices)
            self._show_device_detail(device)
            self.lbl_status.setText(f"API key set for {device.ip}")
            self.lbl_status.setStyleSheet(f"color:{C['ok']};font-size:12px;")

    def _test_connection(self):
        """Test connection to selected device."""
        row = self.device_list.currentRow()
        if row < 0 or row >= len(self._devices):
            return

        device = self._devices[row]
        self.lbl_status.setText(f"Testing {device.ip}...")
        self.lbl_status.setStyleSheet(f"color:{C['acc']};font-size:12px;")

        def _do_test():
            ok, msg = test_connection(device)
            QApplication.instance().postEvent(
                self, _TestResultEvent(ok, msg)
            ) if QApplication.instance() else None

        threading.Thread(target=_do_test, daemon=True).start()

    def _on_test_result(self, ok: bool, msg: str):
        if ok:
            self.lbl_status.setText(f"Connected: {msg}")
            self.lbl_status.setStyleSheet(f"color:{C['ok']};font-size:12px;")
        else:
            self.lbl_status.setText(f"Failed: {msg}")
            self.lbl_status.setStyleSheet(f"color:{C['err']};font-size:12px;")

    def _load_model_on_device(self):
        """Show dialog to load a model on the selected device."""
        row = self.device_list.currentRow()
        if row < 0 or row >= len(self._devices):
            return

        device = self._devices[row]

        # Get models from device
        models = get_device_models(device)
        if not models:
            QMessageBox.warning(
                self, "No Models",
                f"Could not retrieve model list from {device.display_name}.\n\n"
                "Make sure the device is connected and the API key is set."
            )
            return

        # Show model picker
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Load Model on {device.display_name}")
        dlg.setMinimumWidth(400)
        layout = QVBoxLayout(dlg)

        info = QLabel(f"Select a model to load on {device.ip}:")
        info.setStyleSheet(f"color:{C['txt']};font-size:13px;")
        layout.addWidget(info)

        combo = QComboBox()
        for m in models:
            mid = m.get("id", "unknown")
            active = m.get("active", False)
            label = f"[active] {mid}" if active else mid
            combo.addItem(icon("circle-check" if active else "circle"), label, mid)
        combo.setStyleSheet(f"""
            QComboBox {{
                background:{C['bg']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                padding:8px;font-size:13px;
            }}
        """)
        layout.addWidget(combo)

        btn_box = QHBoxLayout()
        ok_btn = QPushButton("Load")
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C['acc']};color:#fff;border:none;border-radius:6px;
                font-weight:bold;font-size:13px;padding:8px 20px;
            }}
        """)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background:{C['surface']};color:{C['txt']};border:1px solid {C['bdr']};border-radius:6px;
                font-weight:bold;font-size:13px;padding:8px 20px;
            }}
        """)
        cancel_btn.clicked.connect(dlg.reject)
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            model_path = combo.currentData()
            if model_path:
                self.lbl_status.setText(f"Loading {model_path} on {device.ip}...")
                self.lbl_status.setStyleSheet(f"color:{C['acc']};font-size:12px;")

                def _do_load():
                    ok, msg = load_model_on_device(device, model_path)
                    QApplication.instance().postEvent(
                        self, _TestResultEvent(ok, msg)
                    ) if QApplication.instance() else None

                threading.Thread(target=_do_load, daemon=True).start()

    def _refresh_selected(self):
        """Refresh the selected device's status."""
        row = self.device_list.currentRow()
        if row < 0 or row >= len(self._devices):
            return

        device = self._devices[row]
        self.lbl_status.setText(f"Refreshing {device.ip}...")
        self.lbl_status.setStyleSheet(f"color:{C['acc']};font-size:12px;")

        def _do_refresh():
            updated = refresh_device(device)
            QApplication.instance().postEvent(
                self, _DeviceRefreshedEvent(row, updated)
            ) if QApplication.instance() else None

        threading.Thread(target=_do_refresh, daemon=True).start()

    def _on_device_refreshed(self, row: int, updated: Optional[DiscoveredDevice]):
        if updated and 0 <= row < len(self._devices):
            self._devices[row] = updated
            save_devices(self._devices)
            self._populate_list()
            self._show_device_detail(updated)
            self.lbl_status.setText(f"Refreshed: {updated.ip}")
            self.lbl_status.setStyleSheet(f"color:{C['ok']};font-size:12px;")
        else:
            self.lbl_status.setText("Refresh failed — device may be offline")
            self.lbl_status.setStyleSheet(f"color:{C['err']};font-size:12px;")

    def _remove_selected(self):
        """Remove selected device from the list."""
        row = self.device_list.currentRow()
        if row < 0 or row >= len(self._devices):
            return
        device = self._devices[row]
        reply = QMessageBox.question(
            self, "Remove Device",
            f"Remove '{device.display_name}' from discovered devices?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._devices.pop(row)
            save_devices(self._devices)
            self._populate_list()
            self.detail_text.setPlainText("No device selected.")

    def _refresh_statuses(self):
        """Background refresh of all device statuses."""
        if self._scanning or not self._devices:
            return

        def _do_refresh():
            updated = []
            for d in self._devices:
                result = refresh_device(d)
                updated.append(result if result else d)
            QApplication.instance().postEvent(
                self, _BatchRefreshEvent(updated)
            ) if QApplication.instance() else None

        threading.Thread(target=_do_refresh, daemon=True).start()

    def _on_batch_refresh(self, devices: list[DiscoveredDevice]):
        self._devices = devices
        save_devices(devices)
        self._populate_list()

    # ── Custom event handling ──────────────────────────────────────

    def event(self, event) -> bool:
        if isinstance(event, _DeviceFoundEvent):
            self._on_device_found(event.device)
            return True
        if isinstance(event, _ScanProgressEvent):
            self._on_scan_progress(event.done, event.total)
            return True
        if isinstance(event, _ScanCompleteEvent):
            self._on_scan_complete(event.devices)
            return True
        if isinstance(event, _DeviceRefreshedEvent):
            self._on_device_refreshed(event.row, event.device)
            return True
        if isinstance(event, _BatchRefreshEvent):
            self._on_batch_refresh(event.devices)
            return True
        if isinstance(event, _TestResultEvent):
            self._on_test_result(event.ok, event.msg)
            return True
        if isinstance(event, _DetailUpdateEvent):
            self.detail_text.setPlainText(event.text)
            return True
        if isinstance(event, _AutoConnectEvent):
            self._on_auto_connect(event.device, event.ok, event.msg)
            return True
        return super().event(event)


# ── Custom events for thread-safe UI updates ──────────────────────

_DEVICE_FOUND_EVENT = QEvent.Type(QEvent.Type.User + 1)
_SCAN_PROGRESS_EVENT = QEvent.Type(QEvent.Type.User + 2)
_SCAN_COMPLETE_EVENT = QEvent.Type(QEvent.Type.User + 3)
_DEVICE_REFRESHED_EVENT = QEvent.Type(QEvent.Type.User + 4)
_BATCH_REFRESH_EVENT = QEvent.Type(QEvent.Type.User + 5)
_TEST_RESULT_EVENT = QEvent.Type(QEvent.Type.User + 6)
_DETAIL_UPDATE_EVENT = QEvent.Type(QEvent.Type.User + 7)
_AUTO_CONNECT_EVENT = QEvent.Type(QEvent.Type.User + 8)


class _DeviceFoundEvent(QEvent):
    def __init__(self, device: DiscoveredDevice):
        super().__init__(_DEVICE_FOUND_EVENT)
        self.device = device


class _ScanProgressEvent(QEvent):
    def __init__(self, done: int, total: int):
        super().__init__(_SCAN_PROGRESS_EVENT)
        self.done = done
        self.total = total


class _ScanCompleteEvent(QEvent):
    def __init__(self, devices: list[DiscoveredDevice]):
        super().__init__(_SCAN_COMPLETE_EVENT)
        self.devices = devices


class _DeviceRefreshedEvent(QEvent):
    def __init__(self, row: int, device: Optional[DiscoveredDevice]):
        super().__init__(_DEVICE_REFRESHED_EVENT)
        self.row = row
        self.device = device


class _BatchRefreshEvent(QEvent):
    def __init__(self, devices: list[DiscoveredDevice]):
        super().__init__(_BATCH_REFRESH_EVENT)
        self.devices = devices


class _TestResultEvent(QEvent):
    def __init__(self, ok: bool, msg: str):
        super().__init__(_TEST_RESULT_EVENT)
        self.ok = ok
        self.msg = msg


class _DetailUpdateEvent(QEvent):
    def __init__(self, text: str):
        super().__init__(_DETAIL_UPDATE_EVENT)
        self.text = text


class _AutoConnectEvent(QEvent):
    def __init__(self, device: DiscoveredDevice, ok: bool, msg: str):
        super().__init__(_AUTO_CONNECT_EVENT)
        self.device = device
        self.ok = ok
        self.msg = msg
