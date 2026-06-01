from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from nativelab.imports.import_global import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QThread,
    QVBoxLayout,
    QWidget,
    Qt,
    pyqtSignal,
)
from nativelab.GlobalConfig.config_global import LONG_TIMEOUT_SECONDS, LONG_TIMEOUT_MS
from nativelab.UI.icons import set_button_icon, set_label_icon
from nativelab.UI.toggle import ToggleSwitch

from .endpoints import IntegrationEndpoints
from .http_endpoint import IntegrationHttpEndpoint
from .discord_connector import (
    DEFAULT_DISCORD_BOT,
    DEFAULT_DISCORD_SYSTEM_PROMPT,
    DISCORD_BOTS_FILE,
    command_catalog,
    delete_discord_bot,
    invite_url,
    load_discord_bots,
    upsert_discord_bot,
)
from .whatsapp_connector import (
    DEFAULT_WHATSAPP_BOT,
    DEFAULT_WHATSAPP_SYSTEM_PROMPT,
    WHATSAPP_BOTS_FILE,
    command_catalog as whatsapp_command_catalog,
    delete_whatsapp_bot,
    load_whatsapp_bots,
    upsert_whatsapp_bot,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_ROOT.parent


class DiscordBotRunner(QThread):
    log = pyqtSignal(str)
    stopped = pyqtSignal(str, int)

    def __init__(self, profile_name: str, endpoint_url: str, parent=None):
        super().__init__(parent)
        self.profile_name = profile_name
        self.endpoint_url = endpoint_url
        self._proc: subprocess.Popen | None = None

    def run(self):
        env = os.environ.copy()
        env["DISCORD_BOT_PROFILE"] = self.profile_name
        env["NATIVELAB_INTEGRATION_URL"] = self.endpoint_url
        env["PYTHONUNBUFFERED"] = "1"
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(PACKAGE_PARENT)
            if not existing_pythonpath
            else str(PACKAGE_PARENT) + os.pathsep + existing_pythonpath
        )
        cmd = [sys.executable, "-u", "-m", "nativelab.integrations.examples.discord_bot"]
        try:
            self.log.emit(f"Starting Discord bot profile '{self.profile_name}'")
            self.log.emit(f"Endpoint: {self.endpoint_url}")
            self.log.emit(f"Python: {sys.executable}")
            self.log.emit(f"Package root: {PACKAGE_ROOT}")
            self.log.emit(f"Working directory: {Path.cwd()}")
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(Path.cwd()),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if line:
                    self.log.emit(line.rstrip())
            code = int(self._proc.wait())
            self.stopped.emit(self.profile_name, code)
        except Exception as e:
            self.log.emit(f"Bot runner error: {e}")
            self.stopped.emit(self.profile_name, -1)

    def stop(self):
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self.log.emit("Stopping Discord bot process")
        proc.terminate()
        try:
            proc.wait(timeout=LONG_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.log.emit("Bot did not exit after terminate; killing process")
            proc.kill()


class WhatsAppBotRunner(QThread):
    log = pyqtSignal(str)
    stopped = pyqtSignal(str, int)

    def __init__(self, profile_name: str, endpoint_url: str, parent=None):
        super().__init__(parent)
        self.profile_name = profile_name
        self.endpoint_url = endpoint_url
        self._proc: subprocess.Popen | None = None

    def run(self):
        env = os.environ.copy()
        env["WHATSAPP_BOT_PROFILE"] = self.profile_name
        env["NATIVELAB_INTEGRATION_URL"] = self.endpoint_url
        env["PYTHONUNBUFFERED"] = "1"
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(PACKAGE_PARENT)
            if not existing_pythonpath
            else str(PACKAGE_PARENT) + os.pathsep + existing_pythonpath
        )
        cmd = [sys.executable, "-u", "-m", "nativelab.integrations.examples.whatsapp_bot"]
        try:
            self.log.emit(f"Starting WhatsApp bot profile '{self.profile_name}'")
            self.log.emit(f"Endpoint: {self.endpoint_url}")
            self.log.emit(f"Python: {sys.executable}")
            self.log.emit(f"Package root: {PACKAGE_ROOT}")
            self.log.emit(f"Working directory: {Path.cwd()}")
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(Path.cwd()),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if line:
                    self.log.emit(line.rstrip())
            code = int(self._proc.wait())
            self.stopped.emit(self.profile_name, code)
        except Exception as e:
            self.log.emit(f"WhatsApp runner error: {e}")
            self.stopped.emit(self.profile_name, -1)

    def stop(self):
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self.log.emit("Stopping WhatsApp bot process")
        proc.terminate()
        try:
            proc.wait(timeout=LONG_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.log.emit("WhatsApp bot did not exit after terminate; killing process")
            proc.kill()


class IntegrationsTab(QWidget):
    """Catalog browser for external integration builders."""

    def __init__(self, endpoints: IntegrationEndpoints | None = None, parent=None):
        super().__init__(parent)
        self._endpoints = endpoints or IntegrationEndpoints()
        self._http_server: IntegrationHttpEndpoint | None = None
        self._discord_bots: list[dict] = []
        self._discord_runner: DiscordBotRunner | None = None
        self._whatsapp_bots: list[dict] = []
        self._whatsapp_runner: WhatsAppBotRunner | None = None
        self._build()
        self.refresh()
        self._reload_discord_profiles()
        self._reload_whatsapp_profiles()

    def set_endpoints(self, endpoints: IntegrationEndpoints):
        self._endpoints = endpoints
        if self._http_server is not None and self._http_server.is_running:
            self._http_server.endpoints = endpoints
        self.refresh()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("chat_scroll")

        inner = QWidget()
        inner.setObjectName("chat_container")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(12)
        scroll.setWidget(inner)
        root.addWidget(scroll)

        hdr = QLabel("Integrations")
        set_label_icon(hdr, "integrations", "Integrations", 18)
        hdr.setStyleSheet("font-size:16px;font-weight:bold;")
        layout.addWidget(hdr)

        sub = QLabel(
            "Build external connectors for Discord bots, AWS, GCP, webhooks, CLIs, "
            "or custom services using the plain-data endpoint catalog below.")
        sub.setWordWrap(True)
        sub.setObjectName("txt2_small")
        layout.addWidget(sub)

        self.inner_tabs = QTabWidget()
        self.inner_tabs.setObjectName("inner_tabs")
        layout.addWidget(self.inner_tabs, 1)

        endpoints_page = QWidget()
        endpoints_layout = QVBoxLayout(endpoints_page)
        endpoints_layout.setContentsMargins(0, 0, 0, 0)
        endpoints_layout.setSpacing(12)
        self.inner_tabs.addTab(endpoints_page, "Endpoints")

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.route_combo = QComboBox()
        self.route_combo.setFixedHeight(30)
        self.route_combo.currentIndexChanged.connect(self._render_selected_route)
        controls.addWidget(QLabel("Endpoint:"))
        controls.addWidget(self.route_combo, 1)

        self.btn_refresh = QPushButton("Refresh")
        set_button_icon(self.btn_refresh, "refresh-cw", "Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        controls.addWidget(self.btn_refresh)

        self.btn_copy = QPushButton("Copy JSON")
        set_button_icon(self.btn_copy, "copy", "Copy JSON")
        self.btn_copy.clicked.connect(self._copy_json)
        controls.addWidget(self.btn_copy)
        endpoints_layout.addLayout(controls)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("txt2_small")
        endpoints_layout.addWidget(self.summary)

        self._build_http_endpoint_card(endpoints_layout)

        card = QFrame()
        card.setObjectName("tab_card")
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(14, 12, 14, 12)
        card_l.setSpacing(8)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setObjectName("log_te")
        self.preview.setMinimumHeight(420)
        card_l.addWidget(self.preview)
        endpoints_layout.addWidget(card, 1)

        discord_page = QWidget()
        discord_layout = QVBoxLayout(discord_page)
        discord_layout.setContentsMargins(0, 0, 0, 0)
        discord_layout.setSpacing(12)
        self.inner_tabs.addTab(discord_page, "Discord Bot")
        self._build_discord_connector(discord_layout)

        whatsapp_page = QWidget()
        whatsapp_layout = QVBoxLayout(whatsapp_page)
        whatsapp_layout.setContentsMargins(0, 0, 0, 0)
        whatsapp_layout.setSpacing(12)
        self.inner_tabs.addTab(whatsapp_page, "WhatsApp Bot")
        self._build_whatsapp_connector(whatsapp_layout)

    def refresh(self):
        current = self.route_combo.currentData() or "/snapshot"
        self.route_combo.blockSignals(True)
        self.route_combo.clear()
        self.route_combo.addItem("/snapshot", "/snapshot")
        for route in self._endpoints.routes():
            path = route.get("path", "")
            if path and self.route_combo.findData(path) < 0:
                self.route_combo.addItem(path, path)
        self.route_combo.blockSignals(False)
        idx = self.route_combo.findData(current)
        self.route_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._render_selected_route()

    def _render_selected_route(self):
        path = self.route_combo.currentData() or "/snapshot"
        text = self._endpoints.to_json(path)
        catalog = self._endpoints.catalog()
        self.summary.setText(
            f"{len(catalog.get('models', []))} local model(s) - "
            f"{len(catalog.get('api_models', []))} API model(s) - "
            f"{len(catalog.get('pipelines', []))} saved pipeline(s) - "
            f"{len(catalog.get('labs', []))} lab route(s)"
        )
        self.preview.setPlainText(text)

    def _copy_json(self):
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(self.preview.toPlainText())

    def _build_http_endpoint_card(self, layout: QVBoxLayout):
        card = QFrame()
        card.setObjectName("tab_card")
        box = QVBoxLayout(card)
        box.setContentsMargins(14, 12, 14, 12)
        box.setSpacing(8)

        hdr = QLabel("Local HTTP Endpoint")
        set_label_icon(hdr, "server", "Local HTTP Endpoint", 18)
        hdr.setStyleSheet("font-size:14px;font-weight:bold;")
        box.addWidget(hdr)

        desc = QLabel("Start a localhost JSON endpoint for bots, webhooks, CLIs, or cloud bridge workers.")
        desc.setWordWrap(True)
        desc.setObjectName("txt2_small")
        box.addWidget(desc)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Port:"))
        self.http_port = QSpinBox()
        self.http_port.setRange(1024, 65535)
        self.http_port.setValue(8765)
        self.http_port.setFixedWidth(110)
        row.addWidget(self.http_port)

        self.btn_http_start = QPushButton("Start")
        set_button_icon(self.btn_http_start, "play", "Start endpoint")
        self.btn_http_start.clicked.connect(self._start_http_endpoint)
        row.addWidget(self.btn_http_start)

        self.btn_http_stop = QPushButton("Stop")
        set_button_icon(self.btn_http_stop, "stop-circle", "Stop endpoint")
        self.btn_http_stop.clicked.connect(self._stop_http_endpoint)
        row.addWidget(self.btn_http_stop)

        self.btn_http_copy = QPushButton("Copy URL")
        set_button_icon(self.btn_http_copy, "copy", "Copy URL")
        self.btn_http_copy.clicked.connect(self._copy_http_url)
        row.addWidget(self.btn_http_copy)
        row.addStretch(1)
        box.addLayout(row)

        self.http_status = QLabel("Stopped")
        self.http_status.setObjectName("txt2_small")
        box.addWidget(self.http_status)
        layout.addWidget(card)

    def _build_discord_connector(self, layout: QVBoxLayout):
        card = QFrame()
        card.setObjectName("tab_card")
        box = QVBoxLayout(card)
        box.setContentsMargins(14, 12, 14, 12)
        box.setSpacing(8)

        hdr = QLabel("Discord Bot Connector")
        set_label_icon(hdr, "discord", "Discord Bot Connector", 18)
        hdr.setStyleSheet("font-size:14px;font-weight:bold;")
        box.addWidget(hdr)

        desc = QLabel(
            "Create reusable Discord bot profiles, save credentials locally, "
            "and choose which NativeLab commands the bot may expose.")
        desc.setWordWrap(True)
        desc.setObjectName("txt2_small")
        box.addWidget(desc)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(8)
        profile_row.addWidget(QLabel("Profile:"))
        self.discord_profile_combo = QComboBox()
        self.discord_profile_combo.currentIndexChanged.connect(self._load_selected_discord_profile)
        profile_row.addWidget(self.discord_profile_combo, 1)

        self.btn_discord_new = QPushButton("New")
        set_button_icon(self.btn_discord_new, "plus", "New bot profile")
        self.btn_discord_new.clicked.connect(self._new_discord_profile)
        profile_row.addWidget(self.btn_discord_new)

        self.btn_discord_save = QPushButton("Save")
        set_button_icon(self.btn_discord_save, "save", "Save bot profile")
        self.btn_discord_save.clicked.connect(self._save_discord_profile)
        profile_row.addWidget(self.btn_discord_save)

        self.btn_discord_delete = QPushButton("Delete")
        set_button_icon(self.btn_discord_delete, "delete", "Delete bot profile")
        self.btn_discord_delete.clicked.connect(self._delete_discord_profile)
        profile_row.addWidget(self.btn_discord_delete)
        box.addLayout(profile_row)

        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        name_row.addWidget(QLabel("Name:"))
        self.discord_name = QLineEdit("")
        self.discord_name.setPlaceholderText("bot1")
        name_row.addWidget(self.discord_name, 1)
        name_row.addWidget(QLabel("Endpoint:"))
        self.discord_endpoint = QLineEdit("http://127.0.0.1:8765")
        name_row.addWidget(self.discord_endpoint, 1)
        box.addLayout(name_row)

        cred_row = QHBoxLayout()
        cred_row.setSpacing(8)
        cred_row.addWidget(QLabel("Token:"))
        self.discord_token = QLineEdit("")
        self.discord_token.setEchoMode(QLineEdit.EchoMode.Password)
        cred_row.addWidget(self.discord_token, 2)
        cred_row.addWidget(QLabel("App ID:"))
        self.discord_app_id = QLineEdit("")
        cred_row.addWidget(self.discord_app_id, 1)
        cred_row.addWidget(QLabel("Guild ID:"))
        self.discord_guild_id = QLineEdit("")
        cred_row.addWidget(self.discord_guild_id, 1)
        box.addLayout(cred_row)

        reply_row = QHBoxLayout()
        reply_row.setSpacing(8)
        reply_row.addWidget(QLabel("Reply:"))
        self.discord_reply_mode = QComboBox()
        self.discord_reply_mode.addItem("Interaction reply", "interaction_reply")
        self.discord_reply_mode.addItem("Channel message", "channel_message")
        reply_row.addWidget(self.discord_reply_mode)
        self.discord_ephemeral = ToggleSwitch("Ephemeral replies")
        reply_row.addWidget(self.discord_ephemeral)
        self.discord_direct_mentions = ToggleSwitch("Reply to @mentions")
        self.discord_direct_mentions.stateChanged.connect(lambda *_: self._on_direct_mentions_changed())
        reply_row.addWidget(self.discord_direct_mentions)
        reply_row.addWidget(QLabel("Max chars:"))
        self.discord_max_chars = QSpinBox()
        self.discord_max_chars.setRange(200, 4000)
        self.discord_max_chars.setValue(1900)
        reply_row.addWidget(self.discord_max_chars)
        reply_row.addStretch(1)
        box.addLayout(reply_row)

        queue_row = QHBoxLayout()
        queue_row.setSpacing(8)
        self.discord_queue_enabled = ToggleSwitch("Request queue")
        self.discord_queue_enabled.setChecked(True)
        queue_row.addWidget(self.discord_queue_enabled)
        queue_row.addWidget(QLabel("Concurrent:"))
        self.discord_max_concurrent = QSpinBox()
        self.discord_max_concurrent.setRange(1, 8)
        self.discord_max_concurrent.setValue(1)
        queue_row.addWidget(self.discord_max_concurrent)
        queue_row.addWidget(QLabel("Queued:"))
        self.discord_max_queued = QSpinBox()
        self.discord_max_queued.setRange(1, 100)
        self.discord_max_queued.setValue(12)
        queue_row.addWidget(self.discord_max_queued)
        queue_row.addStretch(1)
        box.addLayout(queue_row)

        prompt_h = QLabel("System Prompt")
        prompt_h.setStyleSheet("font-weight:bold;")
        box.addWidget(prompt_h)
        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(8)
        self.discord_system_prompt = QTextEdit()
        self.discord_system_prompt.setObjectName("log_te")
        self.discord_system_prompt.setMinimumHeight(78)
        prompt_row.addWidget(self.discord_system_prompt, 1)
        self.btn_discord_prompt_preset = QPushButton("Preset")
        set_button_icon(self.btn_discord_prompt_preset, "refresh-cw", "Restore preset")
        self.btn_discord_prompt_preset.clicked.connect(
            lambda: self.discord_system_prompt.setPlainText(DEFAULT_DISCORD_SYSTEM_PROMPT))
        prompt_row.addWidget(self.btn_discord_prompt_preset)
        box.addLayout(prompt_row)

        priv_card = QFrame()
        priv_l = QVBoxLayout(priv_card)
        priv_l.setContentsMargins(10, 8, 10, 8)
        priv_l.setSpacing(6)
        priv_h = QLabel("Discord Privileges")
        priv_h.setStyleSheet("font-weight:bold;")
        priv_l.addWidget(priv_h)
        self.discord_priv_checks = {}
        for label, key in [
            ("Slash commands", "slash_commands"),
            ("Message content intent", "message_content_intent"),
            ("View channels", "view_channels"),
            ("Send messages", "send_messages"),
            ("Embed links", "embed_links"),
            ("Attach files", "attach_files"),
            ("Read message history", "read_message_history"),
            ("Use external emojis", "use_external_emojis"),
        ]:
            chk = ToggleSwitch(label)
            self.discord_priv_checks[key] = chk
            priv_l.addWidget(chk)
        box.addWidget(priv_card)

        cap_card = QFrame()
        cap_l = QVBoxLayout(cap_card)
        cap_l.setContentsMargins(10, 8, 10, 8)
        cap_l.setSpacing(6)
        cap_h = QLabel("NativeLab Access")
        cap_h.setStyleSheet("font-weight:bold;")
        cap_l.addWidget(cap_h)
        self.discord_cap_checks = {}
        for label, key in [
            ("Ask active model", "ask_model"),
            ("Runtime status", "runtime"),
            ("Saved pipelines", "pipelines"),
            ("Labs routes", "labs"),
            ("Models catalog", "models"),
        ]:
            chk = ToggleSwitch(label)
            chk.stateChanged.connect(lambda *_: self._refresh_discord_commands())
            self.discord_cap_checks[key] = chk
            cap_l.addWidget(chk)
        box.addWidget(cap_card)

        prefix_row = QHBoxLayout()
        prefix_row.setSpacing(8)
        self.btn_discord_start = QPushButton("Start Bot")
        set_button_icon(self.btn_discord_start, "play", "Start Discord bot")
        self.btn_discord_start.clicked.connect(self._start_discord_bot)
        prefix_row.addWidget(self.btn_discord_start)

        self.btn_discord_stop = QPushButton("Stop Bot")
        set_button_icon(self.btn_discord_stop, "stop-circle", "Stop Discord bot")
        self.btn_discord_stop.clicked.connect(self._stop_discord_bot)
        self.btn_discord_stop.setEnabled(False)
        prefix_row.addWidget(self.btn_discord_stop)

        self.btn_discord_invite = QPushButton("Copy Invite URL")
        set_button_icon(self.btn_discord_invite, "copy", "Copy invite URL")
        self.btn_discord_invite.clicked.connect(self._copy_discord_invite)
        prefix_row.addWidget(self.btn_discord_invite)
        self.btn_discord_path = QPushButton("Copy Config Path")
        set_button_icon(self.btn_discord_path, "copy", "Copy config path")
        self.btn_discord_path.clicked.connect(lambda: self._copy_text(str(DISCORD_BOTS_FILE)))
        prefix_row.addWidget(self.btn_discord_path)
        prefix_row.addStretch(1)
        box.addLayout(prefix_row)

        self.discord_commands = QTextEdit()
        self.discord_commands.setReadOnly(True)
        self.discord_commands.setObjectName("log_te")
        self.discord_commands.setMinimumHeight(150)
        box.addWidget(self.discord_commands)

        log_h = QLabel("Bot Logs")
        log_h.setStyleSheet("font-weight:bold;")
        box.addWidget(log_h)
        self.discord_log = QTextEdit()
        self.discord_log.setReadOnly(True)
        self.discord_log.setObjectName("log_te")
        self.discord_log.setMinimumHeight(190)
        box.addWidget(self.discord_log)

        self.discord_status = QLabel(f"Profiles save to {DISCORD_BOTS_FILE}")
        self.discord_status.setObjectName("txt2_small")
        box.addWidget(self.discord_status)
        layout.addWidget(card)

    def _start_http_endpoint(self):
        try:
            if self._http_server is not None and self._http_server.is_running:
                self._http_server.stop()
            self._http_server = IntegrationHttpEndpoint(
                self._endpoints,
                port=int(self.http_port.value()),
            )
            url = self._http_server.start()
            self.http_status.setText(f"Running at {url}")
            self.discord_endpoint.setText(url)
        except Exception as e:
            QMessageBox.warning(self, "Endpoint Error", str(e))

    def _stop_http_endpoint(self):
        if self._http_server is not None:
            self._http_server.stop()
        self.http_status.setText("Stopped")

    def _copy_http_url(self):
        url = self._http_server.base_url if self._http_server is not None else f"http://127.0.0.1:{self.http_port.value()}"
        self._copy_text(url)

    def _reload_discord_profiles(self):
        if not hasattr(self, "discord_profile_combo"):
            return
        current = self.discord_profile_combo.currentData()
        self._discord_bots = load_discord_bots()
        self.discord_profile_combo.blockSignals(True)
        self.discord_profile_combo.clear()
        self.discord_profile_combo.addItem("New Discord bot", "")
        for bot in self._discord_bots:
            self.discord_profile_combo.addItem(bot.get("name", "Unnamed"), bot.get("name", ""))
        self.discord_profile_combo.blockSignals(False)
        idx = self.discord_profile_combo.findData(current)
        self.discord_profile_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._load_selected_discord_profile()

    def _new_discord_profile(self):
        self.discord_profile_combo.setCurrentIndex(0)
        data = dict(DEFAULT_DISCORD_BOT)
        data["name"] = self._next_discord_name()
        if self._http_server is not None and self._http_server.is_running:
            data["endpoint_url"] = self._http_server.base_url
        self._fill_discord_form(data)
        self.discord_status.setText("New profile ready. Add credentials, choose access, then Save.")

    def _load_selected_discord_profile(self):
        name = self.discord_profile_combo.currentData()
        if not name:
            data = dict(DEFAULT_DISCORD_BOT)
            data["name"] = self._next_discord_name()
            if self._http_server is not None and self._http_server.is_running:
                data["endpoint_url"] = self._http_server.base_url
            self._fill_discord_form(data)
            return
        for bot in self._discord_bots:
            if bot.get("name") == name:
                self._fill_discord_form(bot)
                self.discord_status.setText(f"Loaded profile: {name}")
                return

    def _save_discord_profile(self):
        try:
            saved = upsert_discord_bot(self._discord_form_data())
            self.discord_status.setText(f"Saved Discord bot profile: {saved['name']}")
            self._reload_discord_profiles()
            idx = self.discord_profile_combo.findData(saved["name"])
            if idx >= 0:
                self.discord_profile_combo.setCurrentIndex(idx)
        except Exception as e:
            QMessageBox.warning(self, "Discord Profile Error", str(e))

    def _delete_discord_profile(self):
        name = self.discord_profile_combo.currentData() or self.discord_name.text().strip()
        if not name:
            return
        if QMessageBox.question(
            self,
            "Delete Discord Profile",
            f"Delete saved Discord bot profile '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        delete_discord_bot(name)
        self.discord_status.setText(f"Deleted Discord bot profile: {name}")
        self._reload_discord_profiles()

    def _fill_discord_form(self, data: dict):
        data = self._merged_discord(data)
        self.discord_name.setText(data.get("name", ""))
        self.discord_token.setText(data.get("token", ""))
        self.discord_app_id.setText(data.get("application_id", ""))
        self.discord_guild_id.setText(data.get("guild_id", ""))
        self.discord_endpoint.setText(data.get("endpoint_url", "http://127.0.0.1:8765"))
        self.discord_system_prompt.setPlainText(data.get("system_prompt", DEFAULT_DISCORD_SYSTEM_PROMPT))
        reply = data.get("reply", {})
        idx = self.discord_reply_mode.findData(reply.get("mode", "interaction_reply"))
        self.discord_reply_mode.setCurrentIndex(idx if idx >= 0 else 0)
        self.discord_ephemeral.setChecked(bool(reply.get("ephemeral", False)))
        self.discord_direct_mentions.setChecked(bool(reply.get("direct_mentions", False)))
        self.discord_max_chars.setValue(int(reply.get("max_chars", 1900)))
        queue = data.get("queue", {})
        self.discord_queue_enabled.setChecked(bool(queue.get("enabled", True)))
        self.discord_max_concurrent.setValue(int(queue.get("max_concurrent", 1)))
        self.discord_max_queued.setValue(int(queue.get("max_queued", 12)))
        for key, chk in self.discord_priv_checks.items():
            chk.setChecked(bool(data.get("privileges", {}).get(key, False)))
        if self.discord_direct_mentions.isChecked():
            msg_chk = self.discord_priv_checks.get("message_content_intent")
            if msg_chk is not None:
                msg_chk.setChecked(True)
        for key, chk in self.discord_cap_checks.items():
            chk.blockSignals(True)
            chk.setChecked(bool(data.get("capabilities", {}).get(key, False)))
            chk.blockSignals(False)
        self._refresh_discord_commands()

    def _discord_form_data(self) -> dict:
        direct_mentions = self.discord_direct_mentions.isChecked()
        privileges = {
            key: chk.isChecked() for key, chk in self.discord_priv_checks.items()
        }
        if direct_mentions:
            privileges["message_content_intent"] = True
        return {
            "name": self.discord_name.text().strip(),
            "enabled": True,
            "token": self.discord_token.text().strip(),
            "application_id": self.discord_app_id.text().strip(),
            "guild_id": self.discord_guild_id.text().strip(),
            "endpoint_url": self.discord_endpoint.text().strip() or "http://127.0.0.1:8765",
            "system_prompt": (
                self.discord_system_prompt.toPlainText().strip()
                or DEFAULT_DISCORD_SYSTEM_PROMPT
            ),
            "reply": {
                "mode": self.discord_reply_mode.currentData() or "interaction_reply",
                "ephemeral": self.discord_ephemeral.isChecked(),
                "max_chars": int(self.discord_max_chars.value()),
                "direct_mentions": direct_mentions,
            },
            "queue": {
                "enabled": self.discord_queue_enabled.isChecked(),
                "max_concurrent": int(self.discord_max_concurrent.value()),
                "max_queued": int(self.discord_max_queued.value()),
            },
            "privileges": privileges,
            "capabilities": {
                key: chk.isChecked() for key, chk in self.discord_cap_checks.items()
            },
        }

    def _refresh_discord_commands(self):
        if not hasattr(self, "discord_commands"):
            return
        data = self._discord_form_data() if hasattr(self, "discord_name") else DEFAULT_DISCORD_BOT
        lines = ["Available slash commands for this profile:"]
        for row in command_catalog(data):
            lines.append(f"{row['command']} - {row['description']}")
        lines.append("")
        lines.append("Required Discord setup:")
        lines.append("- Enable applications.commands scope when inviting the bot.")
        if data.get("reply", {}).get("direct_mentions", False):
            lines.append("- Enable Message Content Intent for @mention replies.")
        elif data.get("privileges", {}).get("message_content_intent", False):
            lines.append("- Enable Message Content Intent in the Discord Developer Portal.")
        lines.append("- Keep the NativeLab local endpoint running while the bot is online.")
        self.discord_commands.setPlainText("\n".join(lines))

    def _on_direct_mentions_changed(self):
        if not hasattr(self, "discord_priv_checks"):
            return
        if self.discord_direct_mentions.isChecked():
            chk = self.discord_priv_checks.get("message_content_intent")
            if chk is not None:
                chk.setChecked(True)
        self._refresh_discord_commands()

    def _copy_discord_invite(self):
        url = invite_url(self._discord_form_data())
        if not url:
            QMessageBox.warning(self, "Missing Application ID", "Add the Discord application ID first.")
            return
        self._copy_text(url)
        self.discord_status.setText("Copied Discord invite URL.")

    def _start_discord_bot(self):
        if self._discord_runner is not None and self._discord_runner.isRunning():
            self._append_discord_log("A Discord bot is already running. Stop it before starting another profile.")
            return
        try:
            data = upsert_discord_bot(self._discord_form_data())
        except Exception as e:
            QMessageBox.warning(self, "Discord Profile Error", str(e))
            return
        if not data.get("token"):
            QMessageBox.warning(self, "Missing Token", "Add and save the Discord bot token before starting.")
            return
        endpoint_url = self._ensure_discord_endpoint(data)
        data["endpoint_url"] = endpoint_url
        upsert_discord_bot(data)
        self._reload_discord_profiles()
        idx = self.discord_profile_combo.findData(data["name"])
        if idx >= 0:
            self.discord_profile_combo.setCurrentIndex(idx)

        self._append_discord_log(f"Launching profile '{data['name']}'")
        runner = DiscordBotRunner(data["name"], endpoint_url, self)
        runner.log.connect(self._append_discord_log)
        runner.stopped.connect(self._on_discord_bot_stopped)
        self._discord_runner = runner
        self.btn_discord_start.setEnabled(False)
        self.btn_discord_stop.setEnabled(True)
        self.discord_status.setText(f"Running Discord bot profile: {data['name']}")
        runner.start()

    def _stop_discord_bot(self):
        if self._discord_runner is None:
            return
        if self._discord_runner.isRunning():
            self._discord_runner.stop()
            self._discord_runner.wait(LONG_TIMEOUT_MS)
        self._discord_runner = None
        self.btn_discord_start.setEnabled(True)
        self.btn_discord_stop.setEnabled(False)
        self.discord_status.setText("Discord bot stopped.")

    def _on_discord_bot_stopped(self, profile_name: str, code: int):
        self._append_discord_log(f"Discord bot '{profile_name}' stopped with exit code {code}")
        self._discord_runner = None
        if hasattr(self, "btn_discord_start"):
            self.btn_discord_start.setEnabled(True)
            self.btn_discord_stop.setEnabled(False)
        if hasattr(self, "discord_status"):
            self.discord_status.setText(f"Discord bot stopped: {profile_name}")

    def _ensure_discord_endpoint(self, data: dict) -> str:
        endpoint_url = data.get("endpoint_url") or self.discord_endpoint.text().strip() or "http://127.0.0.1:8765"
        parsed = urlparse(endpoint_url)
        host = (parsed.hostname or "127.0.0.1").lower()
        if host not in {"127.0.0.1", "localhost"}:
            return endpoint_url.rstrip("/")

        port = int(parsed.port or self.http_port.value() or 8765)
        if self._http_server is None or not self._http_server.is_running or self._http_server.port != port:
            if self._http_server is not None and self._http_server.is_running:
                self._http_server.stop()
            self.http_port.setValue(port)
            self._http_server = IntegrationHttpEndpoint(self._endpoints, port=port)
            endpoint_url = self._http_server.start()
            self.http_status.setText(f"Running at {endpoint_url}")
            self._append_discord_log(f"Started NativeLab integration endpoint at {endpoint_url}")
        else:
            endpoint_url = self._http_server.base_url
        self.discord_endpoint.setText(endpoint_url)
        return endpoint_url

    def _append_discord_log(self, line: str):
        if not hasattr(self, "discord_log"):
            return
        self.discord_log.append(str(line))

    def _next_discord_name(self) -> str:
        used = {b.get("name", "") for b in self._discord_bots}
        i = 1
        while f"bot{i}" in used:
            i += 1
        return f"bot{i}"

    @staticmethod
    def _merged_discord(data: dict) -> dict:
        merged = {
            **DEFAULT_DISCORD_BOT,
            **(data or {}),
        }
        for key in ("reply", "queue", "privileges", "capabilities"):
            merged[key] = {
                **DEFAULT_DISCORD_BOT.get(key, {}),
                **(data or {}).get(key, {}),
            }
        return merged

    def _build_whatsapp_connector(self, layout: QVBoxLayout):
        card = QFrame()
        card.setObjectName("tab_card")
        box = QVBoxLayout(card)
        box.setContentsMargins(14, 12, 14, 12)
        box.setSpacing(8)

        hdr = QLabel("WhatsApp Bot Connector")
        set_label_icon(hdr, "whatsapp", "WhatsApp Bot Connector", 18)
        hdr.setStyleSheet("font-size:14px;font-weight:bold;")
        box.addWidget(hdr)

        desc = QLabel(
            "Create reusable WhatsApp Cloud API webhook profiles. Expose the local "
            "webhook with a public tunnel, then add that URL in Meta's webhook setup.")
        desc.setWordWrap(True)
        desc.setObjectName("txt2_small")
        box.addWidget(desc)

        profile_row = QHBoxLayout(); profile_row.setSpacing(8)
        profile_row.addWidget(QLabel("Profile:"))
        self.whatsapp_profile_combo = QComboBox()
        self.whatsapp_profile_combo.currentIndexChanged.connect(self._load_selected_whatsapp_profile)
        profile_row.addWidget(self.whatsapp_profile_combo, 1)
        self.btn_whatsapp_new = QPushButton("New")
        set_button_icon(self.btn_whatsapp_new, "plus", "New WhatsApp profile")
        self.btn_whatsapp_new.clicked.connect(self._new_whatsapp_profile)
        profile_row.addWidget(self.btn_whatsapp_new)
        self.btn_whatsapp_save = QPushButton("Save")
        set_button_icon(self.btn_whatsapp_save, "save", "Save WhatsApp profile")
        self.btn_whatsapp_save.clicked.connect(self._save_whatsapp_profile)
        profile_row.addWidget(self.btn_whatsapp_save)
        self.btn_whatsapp_delete = QPushButton("Delete")
        set_button_icon(self.btn_whatsapp_delete, "delete", "Delete WhatsApp profile")
        self.btn_whatsapp_delete.clicked.connect(self._delete_whatsapp_profile)
        profile_row.addWidget(self.btn_whatsapp_delete)
        box.addLayout(profile_row)

        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(QLabel("Name:"))
        self.whatsapp_name = QLineEdit("")
        self.whatsapp_name.setPlaceholderText("whatsapp1")
        row.addWidget(self.whatsapp_name, 1)
        row.addWidget(QLabel("NativeLab endpoint:"))
        self.whatsapp_endpoint = QLineEdit("http://127.0.0.1:8765")
        row.addWidget(self.whatsapp_endpoint, 1)
        box.addLayout(row)

        cred = QHBoxLayout(); cred.setSpacing(8)
        cred.addWidget(QLabel("Access token:"))
        self.whatsapp_token = QLineEdit("")
        self.whatsapp_token.setEchoMode(QLineEdit.EchoMode.Password)
        cred.addWidget(self.whatsapp_token, 2)
        cred.addWidget(QLabel("Phone number ID:"))
        self.whatsapp_phone_id = QLineEdit("")
        cred.addWidget(self.whatsapp_phone_id, 1)
        cred.addWidget(QLabel("Business ID:"))
        self.whatsapp_business_id = QLineEdit("")
        cred.addWidget(self.whatsapp_business_id, 1)
        box.addLayout(cred)

        webhook = QHBoxLayout(); webhook.setSpacing(8)
        webhook.addWidget(QLabel("Webhook host:"))
        self.whatsapp_webhook_host = QLineEdit("127.0.0.1")
        webhook.addWidget(self.whatsapp_webhook_host, 1)
        webhook.addWidget(QLabel("Port:"))
        self.whatsapp_webhook_port = QSpinBox()
        self.whatsapp_webhook_port.setRange(1024, 65535)
        self.whatsapp_webhook_port.setValue(8770)
        webhook.addWidget(self.whatsapp_webhook_port)
        webhook.addWidget(QLabel("Path:"))
        self.whatsapp_webhook_path = QLineEdit("/webhook")
        webhook.addWidget(self.whatsapp_webhook_path, 1)
        webhook.addWidget(QLabel("Verify token:"))
        self.whatsapp_verify_token = QLineEdit("nativelab-whatsapp")
        webhook.addWidget(self.whatsapp_verify_token, 1)
        box.addLayout(webhook)

        reply = QHBoxLayout(); reply.setSpacing(8)
        self.whatsapp_direct_messages = ToggleSwitch("Reply to normal messages")
        self.whatsapp_direct_messages.setChecked(True)
        self.whatsapp_direct_messages.stateChanged.connect(lambda *_: self._refresh_whatsapp_commands())
        reply.addWidget(self.whatsapp_direct_messages)
        self.whatsapp_queue_enabled = ToggleSwitch("Request queue")
        self.whatsapp_queue_enabled.setChecked(True)
        reply.addWidget(self.whatsapp_queue_enabled)
        reply.addWidget(QLabel("Concurrent:"))
        self.whatsapp_max_concurrent = QSpinBox()
        self.whatsapp_max_concurrent.setRange(1, 8)
        self.whatsapp_max_concurrent.setValue(1)
        reply.addWidget(self.whatsapp_max_concurrent)
        reply.addWidget(QLabel("Queued:"))
        self.whatsapp_max_queued = QSpinBox()
        self.whatsapp_max_queued.setRange(1, 100)
        self.whatsapp_max_queued.setValue(12)
        reply.addWidget(self.whatsapp_max_queued)
        reply.addWidget(QLabel("Max chars:"))
        self.whatsapp_max_chars = QSpinBox()
        self.whatsapp_max_chars.setRange(200, 4096)
        self.whatsapp_max_chars.setValue(3500)
        reply.addWidget(self.whatsapp_max_chars)
        reply.addStretch(1)
        box.addLayout(reply)

        prompt_h = QLabel("System Prompt")
        prompt_h.setStyleSheet("font-weight:bold;")
        box.addWidget(prompt_h)
        prompt_row = QHBoxLayout(); prompt_row.setSpacing(8)
        self.whatsapp_system_prompt = QTextEdit()
        self.whatsapp_system_prompt.setObjectName("log_te")
        self.whatsapp_system_prompt.setMinimumHeight(78)
        prompt_row.addWidget(self.whatsapp_system_prompt, 1)
        self.btn_whatsapp_prompt_preset = QPushButton("Preset")
        set_button_icon(self.btn_whatsapp_prompt_preset, "refresh-cw", "Restore preset")
        self.btn_whatsapp_prompt_preset.clicked.connect(
            lambda: self.whatsapp_system_prompt.setPlainText(DEFAULT_WHATSAPP_SYSTEM_PROMPT))
        prompt_row.addWidget(self.btn_whatsapp_prompt_preset)
        box.addLayout(prompt_row)

        cap_card = QFrame()
        cap_l = QVBoxLayout(cap_card)
        cap_l.setContentsMargins(10, 8, 10, 8)
        cap_l.setSpacing(6)
        cap_h = QLabel("NativeLab Access")
        cap_h.setStyleSheet("font-weight:bold;")
        cap_l.addWidget(cap_h)
        self.whatsapp_cap_checks = {}
        for label, key in [
            ("Ask active model", "ask_model"),
            ("Runtime status", "runtime"),
            ("Saved pipelines", "pipelines"),
            ("Labs routes", "labs"),
            ("Models catalog", "models"),
        ]:
            chk = ToggleSwitch(label)
            chk.stateChanged.connect(lambda *_: self._refresh_whatsapp_commands())
            self.whatsapp_cap_checks[key] = chk
            cap_l.addWidget(chk)
        box.addWidget(cap_card)

        actions = QHBoxLayout(); actions.setSpacing(8)
        self.btn_whatsapp_start = QPushButton("Start Bot")
        set_button_icon(self.btn_whatsapp_start, "play", "Start WhatsApp webhook")
        self.btn_whatsapp_start.clicked.connect(self._start_whatsapp_bot)
        actions.addWidget(self.btn_whatsapp_start)
        self.btn_whatsapp_stop = QPushButton("Stop Bot")
        set_button_icon(self.btn_whatsapp_stop, "stop-circle", "Stop WhatsApp webhook")
        self.btn_whatsapp_stop.clicked.connect(self._stop_whatsapp_bot)
        self.btn_whatsapp_stop.setEnabled(False)
        actions.addWidget(self.btn_whatsapp_stop)
        self.btn_whatsapp_copy_callback = QPushButton("Copy Callback URL")
        set_button_icon(self.btn_whatsapp_copy_callback, "copy", "Copy callback URL")
        self.btn_whatsapp_copy_callback.clicked.connect(lambda: self._copy_text(self._whatsapp_callback_url()))
        actions.addWidget(self.btn_whatsapp_copy_callback)
        self.btn_whatsapp_copy_verify = QPushButton("Copy Verify Token")
        set_button_icon(self.btn_whatsapp_copy_verify, "copy", "Copy verify token")
        self.btn_whatsapp_copy_verify.clicked.connect(lambda: self._copy_text(self.whatsapp_verify_token.text().strip()))
        actions.addWidget(self.btn_whatsapp_copy_verify)
        self.btn_whatsapp_path = QPushButton("Copy Config Path")
        set_button_icon(self.btn_whatsapp_path, "copy", "Copy config path")
        self.btn_whatsapp_path.clicked.connect(lambda: self._copy_text(str(WHATSAPP_BOTS_FILE)))
        actions.addWidget(self.btn_whatsapp_path)
        actions.addStretch(1)
        box.addLayout(actions)

        self.whatsapp_commands = QTextEdit()
        self.whatsapp_commands.setReadOnly(True)
        self.whatsapp_commands.setObjectName("log_te")
        self.whatsapp_commands.setMinimumHeight(150)
        box.addWidget(self.whatsapp_commands)

        log_h = QLabel("Bot Logs")
        log_h.setStyleSheet("font-weight:bold;")
        box.addWidget(log_h)
        self.whatsapp_log = QTextEdit()
        self.whatsapp_log.setReadOnly(True)
        self.whatsapp_log.setObjectName("log_te")
        self.whatsapp_log.setMinimumHeight(190)
        box.addWidget(self.whatsapp_log)

        self.whatsapp_status = QLabel(f"Profiles save to {WHATSAPP_BOTS_FILE}")
        self.whatsapp_status.setObjectName("txt2_small")
        box.addWidget(self.whatsapp_status)
        layout.addWidget(card)

    def _reload_whatsapp_profiles(self):
        if not hasattr(self, "whatsapp_profile_combo"):
            return
        current = self.whatsapp_profile_combo.currentData()
        self._whatsapp_bots = load_whatsapp_bots()
        self.whatsapp_profile_combo.blockSignals(True)
        self.whatsapp_profile_combo.clear()
        self.whatsapp_profile_combo.addItem("New WhatsApp bot", "")
        for bot in self._whatsapp_bots:
            self.whatsapp_profile_combo.addItem(bot.get("name", "Unnamed"), bot.get("name", ""))
        self.whatsapp_profile_combo.blockSignals(False)
        idx = self.whatsapp_profile_combo.findData(current)
        self.whatsapp_profile_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._load_selected_whatsapp_profile()

    def _new_whatsapp_profile(self):
        self.whatsapp_profile_combo.setCurrentIndex(0)
        data = dict(DEFAULT_WHATSAPP_BOT)
        data["name"] = self._next_whatsapp_name()
        if self._http_server is not None and self._http_server.is_running:
            data["endpoint_url"] = self._http_server.base_url
        self._fill_whatsapp_form(data)
        self.whatsapp_status.setText("New profile ready. Add Meta credentials, webhook settings, then Save.")

    def _load_selected_whatsapp_profile(self):
        name = self.whatsapp_profile_combo.currentData()
        if not name:
            data = dict(DEFAULT_WHATSAPP_BOT)
            data["name"] = self._next_whatsapp_name()
            if self._http_server is not None and self._http_server.is_running:
                data["endpoint_url"] = self._http_server.base_url
            self._fill_whatsapp_form(data)
            return
        for bot in self._whatsapp_bots:
            if bot.get("name") == name:
                self._fill_whatsapp_form(bot)
                self.whatsapp_status.setText(f"Loaded profile: {name}")
                return

    def _save_whatsapp_profile(self):
        try:
            saved = upsert_whatsapp_bot(self._whatsapp_form_data())
            self.whatsapp_status.setText(f"Saved WhatsApp bot profile: {saved['name']}")
            self._reload_whatsapp_profiles()
            idx = self.whatsapp_profile_combo.findData(saved["name"])
            if idx >= 0:
                self.whatsapp_profile_combo.setCurrentIndex(idx)
        except Exception as e:
            QMessageBox.warning(self, "WhatsApp Profile Error", str(e))

    def _delete_whatsapp_profile(self):
        name = self.whatsapp_profile_combo.currentData() or self.whatsapp_name.text().strip()
        if not name:
            return
        if QMessageBox.question(
            self,
            "Delete WhatsApp Profile",
            f"Delete saved WhatsApp bot profile '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        delete_whatsapp_bot(name)
        self.whatsapp_status.setText(f"Deleted WhatsApp bot profile: {name}")
        self._reload_whatsapp_profiles()

    def _fill_whatsapp_form(self, data: dict):
        data = self._merged_whatsapp(data)
        self.whatsapp_name.setText(data.get("name", ""))
        self.whatsapp_token.setText(data.get("access_token", ""))
        self.whatsapp_phone_id.setText(data.get("phone_number_id", ""))
        self.whatsapp_business_id.setText(data.get("business_account_id", ""))
        self.whatsapp_verify_token.setText(data.get("verify_token", "nativelab-whatsapp"))
        self.whatsapp_endpoint.setText(data.get("endpoint_url", "http://127.0.0.1:8765"))
        self.whatsapp_webhook_host.setText(data.get("webhook_host", "127.0.0.1"))
        self.whatsapp_webhook_port.setValue(int(data.get("webhook_port", 8770)))
        self.whatsapp_webhook_path.setText(data.get("webhook_path", "/webhook"))
        self.whatsapp_system_prompt.setPlainText(data.get("system_prompt", DEFAULT_WHATSAPP_SYSTEM_PROMPT))
        reply = data.get("reply", {})
        self.whatsapp_direct_messages.setChecked(bool(reply.get("direct_messages", True)))
        self.whatsapp_max_chars.setValue(int(reply.get("max_chars", 3500)))
        queue = data.get("queue", {})
        self.whatsapp_queue_enabled.setChecked(bool(queue.get("enabled", True)))
        self.whatsapp_max_concurrent.setValue(int(queue.get("max_concurrent", 1)))
        self.whatsapp_max_queued.setValue(int(queue.get("max_queued", 12)))
        for key, chk in self.whatsapp_cap_checks.items():
            chk.blockSignals(True)
            chk.setChecked(bool(data.get("capabilities", {}).get(key, False)))
            chk.blockSignals(False)
        self._refresh_whatsapp_commands()

    def _whatsapp_form_data(self) -> dict:
        path = self.whatsapp_webhook_path.text().strip() or "/webhook"
        if not path.startswith("/"):
            path = "/" + path
        return {
            "name": self.whatsapp_name.text().strip(),
            "enabled": True,
            "access_token": self.whatsapp_token.text().strip(),
            "phone_number_id": self.whatsapp_phone_id.text().strip(),
            "business_account_id": self.whatsapp_business_id.text().strip(),
            "verify_token": self.whatsapp_verify_token.text().strip() or "nativelab-whatsapp",
            "endpoint_url": self.whatsapp_endpoint.text().strip() or "http://127.0.0.1:8765",
            "webhook_host": self.whatsapp_webhook_host.text().strip() or "127.0.0.1",
            "webhook_port": int(self.whatsapp_webhook_port.value()),
            "webhook_path": path,
            "system_prompt": (
                self.whatsapp_system_prompt.toPlainText().strip()
                or DEFAULT_WHATSAPP_SYSTEM_PROMPT
            ),
            "reply": {
                "max_chars": int(self.whatsapp_max_chars.value()),
                "direct_messages": self.whatsapp_direct_messages.isChecked(),
            },
            "queue": {
                "enabled": self.whatsapp_queue_enabled.isChecked(),
                "max_concurrent": int(self.whatsapp_max_concurrent.value()),
                "max_queued": int(self.whatsapp_max_queued.value()),
            },
            "capabilities": {
                key: chk.isChecked() for key, chk in self.whatsapp_cap_checks.items()
            },
        }

    def _refresh_whatsapp_commands(self):
        if not hasattr(self, "whatsapp_commands"):
            return
        data = self._whatsapp_form_data() if hasattr(self, "whatsapp_name") else DEFAULT_WHATSAPP_BOT
        lines = ["Available WhatsApp commands for this profile:"]
        for row in whatsapp_command_catalog(data):
            lines.append(f"{row['command']} - {row['description']}")
        lines.append("")
        lines.append("Meta webhook setup:")
        lines.append(f"- Callback URL: {self._whatsapp_callback_url()}")
        lines.append(f"- Verify token: {data.get('verify_token', '')}")
        lines.append("- Use a public HTTPS tunnel for Meta Cloud API webhooks.")
        self.whatsapp_commands.setPlainText("\n".join(lines))

    def _start_whatsapp_bot(self):
        if self._whatsapp_runner is not None and self._whatsapp_runner.isRunning():
            self._append_whatsapp_log("A WhatsApp bot is already running. Stop it before starting another profile.")
            return
        try:
            data = upsert_whatsapp_bot(self._whatsapp_form_data())
        except Exception as e:
            QMessageBox.warning(self, "WhatsApp Profile Error", str(e))
            return
        if not data.get("access_token"):
            QMessageBox.warning(self, "Missing Token", "Add and save the WhatsApp Cloud API access token before starting.")
            return
        if not data.get("phone_number_id"):
            QMessageBox.warning(self, "Missing Phone Number ID", "Add and save the WhatsApp phone number ID before starting.")
            return
        endpoint_url = self._ensure_whatsapp_endpoint(data)
        data["endpoint_url"] = endpoint_url
        upsert_whatsapp_bot(data)
        self._reload_whatsapp_profiles()
        idx = self.whatsapp_profile_combo.findData(data["name"])
        if idx >= 0:
            self.whatsapp_profile_combo.setCurrentIndex(idx)

        self._append_whatsapp_log(f"Launching profile '{data['name']}'")
        self._append_whatsapp_log(f"Meta callback URL: {self._whatsapp_callback_url()}")
        runner = WhatsAppBotRunner(data["name"], endpoint_url, self)
        runner.log.connect(self._append_whatsapp_log)
        runner.stopped.connect(self._on_whatsapp_bot_stopped)
        self._whatsapp_runner = runner
        self.btn_whatsapp_start.setEnabled(False)
        self.btn_whatsapp_stop.setEnabled(True)
        self.whatsapp_status.setText(f"Running WhatsApp bot profile: {data['name']}")
        runner.start()

    def _stop_whatsapp_bot(self):
        if self._whatsapp_runner is None:
            return
        if self._whatsapp_runner.isRunning():
            self._whatsapp_runner.stop()
            self._whatsapp_runner.wait(LONG_TIMEOUT_MS)
        self._whatsapp_runner = None
        self.btn_whatsapp_start.setEnabled(True)
        self.btn_whatsapp_stop.setEnabled(False)
        self.whatsapp_status.setText("WhatsApp bot stopped.")

    def _on_whatsapp_bot_stopped(self, profile_name: str, code: int):
        self._append_whatsapp_log(f"WhatsApp bot '{profile_name}' stopped with exit code {code}")
        self._whatsapp_runner = None
        if hasattr(self, "btn_whatsapp_start"):
            self.btn_whatsapp_start.setEnabled(True)
            self.btn_whatsapp_stop.setEnabled(False)
        if hasattr(self, "whatsapp_status"):
            self.whatsapp_status.setText(f"WhatsApp bot stopped: {profile_name}")

    def _ensure_whatsapp_endpoint(self, data: dict) -> str:
        endpoint_url = data.get("endpoint_url") or self.whatsapp_endpoint.text().strip() or "http://127.0.0.1:8765"
        parsed = urlparse(endpoint_url)
        host = (parsed.hostname or "127.0.0.1").lower()
        if host not in {"127.0.0.1", "localhost"}:
            return endpoint_url.rstrip("/")
        port = int(parsed.port or self.http_port.value() or 8765)
        if self._http_server is None or not self._http_server.is_running or self._http_server.port != port:
            if self._http_server is not None and self._http_server.is_running:
                self._http_server.stop()
            self.http_port.setValue(port)
            self._http_server = IntegrationHttpEndpoint(self._endpoints, port=port)
            endpoint_url = self._http_server.start()
            self.http_status.setText(f"Running at {endpoint_url}")
            self._append_whatsapp_log(f"Started NativeLab integration endpoint at {endpoint_url}")
        else:
            endpoint_url = self._http_server.base_url
        self.whatsapp_endpoint.setText(endpoint_url)
        return endpoint_url

    def _whatsapp_callback_url(self) -> str:
        host = self.whatsapp_webhook_host.text().strip() or "127.0.0.1"
        port = int(self.whatsapp_webhook_port.value())
        path = self.whatsapp_webhook_path.text().strip() or "/webhook"
        if not path.startswith("/"):
            path = "/" + path
        return f"http://{host}:{port}{path}"

    def _append_whatsapp_log(self, line: str):
        if not hasattr(self, "whatsapp_log"):
            return
        self.whatsapp_log.append(str(line))

    def _next_whatsapp_name(self) -> str:
        used = {b.get("name", "") for b in self._whatsapp_bots}
        i = 1
        while f"whatsapp{i}" in used:
            i += 1
        return f"whatsapp{i}"

    @staticmethod
    def _merged_whatsapp(data: dict) -> dict:
        merged = {
            **DEFAULT_WHATSAPP_BOT,
            **(data or {}),
        }
        for key in ("reply", "queue", "capabilities"):
            merged[key] = {
                **DEFAULT_WHATSAPP_BOT.get(key, {}),
                **(data or {}).get(key, {}),
            }
        return merged


    def _copy_text(self, text: str):
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(text)

    def closeEvent(self, event):
        self._stop_discord_bot()
        self._stop_whatsapp_bot()
        self._stop_http_endpoint()
        super().closeEvent(event)
