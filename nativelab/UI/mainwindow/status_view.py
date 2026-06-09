"""MainWindow feature mixin extracted from nativelab.main."""
from .shared import *


class StatusViewMixin:
    def _on_config_changed(self):
        self._log("INFO", "App config updated and saved.")
        self._apply_developer_mode_visibility()
        if hasattr(self, "hf_login_tab"):
            self.hf_login_tab.refresh_state()
        if hasattr(self, "download_tab"):
            self.download_tab.refresh_hf_auth_state()

    def _set_engine_status(self, text: str, state: str = "idle"):
        if not hasattr(self, "lbl_engine"):
            return
        set_status_label(self.lbl_engine, text, state, 14)
        color_key = {
            "ok": "ok",
            "api": "ok",
            "loading": "acc",
            "warn": "warn",
            "err": "err",
            "error": "err",
            "pipeline": "pipeline",
        }.get(str(state), "txt2")
        self.lbl_engine.setStyleSheet(f"color:{C.get(color_key, C['txt2'])};padding:0 8px;")

    def _set_engine_status_from_engine(self, engine=None):
        status = engine_status(engine if engine is not None else self._active_engine_for(""))
        self._set_engine_status(status.status_text, status.state)

    def _on_context_meter_update(self, snapshot: dict):
        self._last_context_snapshot = dict(snapshot or {})
        self._render_context_snapshot(self._last_context_snapshot)

    def _update_ctx_bar(self, limit_override: int = 0):
        if not hasattr(self, "ctx_bar"):
            return
        snap = getattr(self, "_last_context_snapshot", None)
        if snap:
            snap = dict(snap)
            if limit_override:
                snap["limit_tokens"] = int(limit_override)
            self._render_context_snapshot(snap)
            return
        if not self.active:
            return
        max_ctx = int(limit_override or getattr(self.engine, "ctx_value", self.ctx_slider.value()) or self.ctx_slider.value())
        context_meter.report_session(
            source="Session",
            used_tokens=self.active.approx_tokens,
            limit_tokens=max_ctx,
            engine=self._active_engine_for("") if hasattr(self, "engine") else None,
        )

    def _render_context_snapshot(self, snap: dict):
        if not hasattr(self, "ctx_bar"):
            return
        used = max(0, int(snap.get("used_tokens", 0) or 0))
        limit = max(1, int(snap.get("limit_tokens", 0) or getattr(self.engine, "ctx_value", self.ctx_slider.value()) or DEFAULT_CTX()))
        projected = max(used, int(snap.get("projected_tokens", used) or used))
        self.ctx_bar.setRange(0, limit)
        self.ctx_bar.setValue(min(used, limit))
        self.ctx_lbl.setText(f"{used:,} / {limit:,}")
        source = str(snap.get("source") or "LLM")
        if hasattr(self, "ctx_scope_lbl"):
            self.ctx_scope_lbl.setText(source[:10])
        pct = projected / limit if limit > 0 else 0
        color = C["ok"] if pct < 0.6 else C["warn"] if pct < 0.85 else C["err"]
        self.ctx_bar.setStyleSheet(
            f"QProgressBar{{background:{C['bg2']};border:1px solid {C['bdr']};"
            f"border-radius:3px;height:8px;}}"
            f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"
        )
        input_tokens = int(snap.get("input_tokens", used) or 0)
        output_tokens = int(snap.get("output_tokens", 0) or 0)
        reserved = int(snap.get("reserved_tokens", 0) or 0)
        model = str(snap.get("model") or "")
        mode = str(snap.get("mode") or "")
        self.ctx_bar.setToolTip(
            f"{source} context\n"
            f"Input: {input_tokens:,} tokens approx\n"
            f"Generated: {output_tokens:,} tokens approx\n"
            f"Reserved output budget: {reserved:,}\n"
            f"Projected request: {projected:,} / {limit:,}\n"
            + (f"Model: {model}\n" if model else "")
            + (f"Mode: {mode}" if mode else "")
        )
        self.ctx_lbl.setToolTip(self.ctx_bar.toolTip())

    def _update_ram(self):
        mem  = psutil.virtual_memory()
        used = mem.used  // (1024 ** 3)
        tot  = mem.total // (1024 ** 3)
        self.ram_lbl.setText(f"RAM: {used}/{tot} GB")

    def _log(self, level: str, msg: str):
        if hasattr(self, "log_console"):
            self.log_console.log(level, msg)
        else:
            print(f"[{level}] {msg}", file=sys.stderr)

    def _show_config_dialog(self, default_page: str = "General"):
        if self.config_dialog is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Settings")
            layout = QVBoxLayout(dlg)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.settings_panel)
            self.config_dialog = dlg
        prepare_adaptive_window(self.config_dialog, 980, 760, min_width=680, min_height=500)
        self._show_settings_page(default_page or "General")
        self.config_dialog.show()
        self.config_dialog.raise_()
        self.config_dialog.activateWindow()

    def _toggle_sidebar(self):
        target = self.left_sidebar_stack if hasattr(self, "left_sidebar_stack") else self.sidebar
        target.setVisible(not target.isVisible())
        APP_CONFIG["sidebar_visible"] = target.isVisible()
        save_app_config(APP_CONFIG)
        self._update_view_toggle_buttons()

    def _toggle_topbar(self):
        bar = self.tabs.tabBar()
        bar.setVisible(not bar.isVisible())
        APP_CONFIG["topbar_visible"] = bar.isVisible()
        save_app_config(APP_CONFIG)
        self._update_view_toggle_buttons()

    def _tab_key(self, idx: int) -> str:
        return self.tabs.tabText(idx).replace("&", "").strip()

    def _visible_tab_count(self) -> int:
        return sum(1 for i in range(self.tabs.count()) if self.tabs.isTabVisible(i))

    def _developer_mode_enabled(self) -> bool:
        return bool(APP_CONFIG.get("developer_mode", False))

    def _apply_developer_mode_visibility(self):
        if not hasattr(self, "tabs") or not hasattr(self, "dev_tab"):
            return
        idx = self.tabs.indexOf(self.dev_tab)
        if idx < 0:
            return
        tab_cfg = dict(APP_CONFIG.get("tab_visibility", {}))
        visible = self._developer_mode_enabled() and bool(tab_cfg.get("Dev", True))
        self.tabs.setTabVisible(idx, visible)
        if not visible and self.tabs.currentIndex() == idx:
            for i in range(self.tabs.count()):
                if self.tabs.isTabVisible(i):
                    self.tabs.setCurrentIndex(i)
                    break

    def _set_tab_visible(self, idx: int, visible: bool):
        if idx < 0 or idx >= self.tabs.count():
            return
        if self._tab_key(idx) == "Dev" and visible and not self._developer_mode_enabled():
            return
        if not visible and self._visible_tab_count() <= 1 and self.tabs.isTabVisible(idx):
            return
        self.tabs.setTabVisible(idx, bool(visible))
        if not self.tabs.isTabVisible(self.tabs.currentIndex()):
            for i in range(self.tabs.count()):
                if self.tabs.isTabVisible(i):
                    self.tabs.setCurrentIndex(i)
                    break
        tab_cfg = dict(APP_CONFIG.get("tab_visibility", {}))
        tab_cfg[self._tab_key(idx)] = bool(visible)
        APP_CONFIG["tab_visibility"] = tab_cfg
        save_app_config(APP_CONFIG)

    def _show_tab_visibility_menu(self, global_pos=None):
        menu = QMenu(self)
        for i in range(self.tabs.count()):
            label = self._tab_key(i)
            if label == "Dev" and not self._developer_mode_enabled():
                continue
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(self.tabs.isTabVisible(i))
            if self.tabs.isTabVisible(i) and self._visible_tab_count() <= 1:
                act.setEnabled(False)
            act.triggered.connect(lambda checked, ix=i: self._set_tab_visible(ix, checked))
            menu.addAction(act)
        if global_pos is None:
            global_pos = self.btn_tab_menu.mapToGlobal(self.btn_tab_menu.rect().bottomRight())
        menu.exec(global_pos)

    def _apply_saved_view_state(self):
        sidebar_target = self.left_sidebar_stack if hasattr(self, "left_sidebar_stack") else self.sidebar
        sidebar_target.setVisible(bool(APP_CONFIG.get("sidebar_visible", True)))
        self.tabs.tabBar().setVisible(bool(APP_CONFIG.get("topbar_visible", True)))
        tab_cfg = APP_CONFIG.get("tab_visibility", {})
        for i in range(self.tabs.count()):
            visible = bool(tab_cfg.get(self._tab_key(i), True))
            self.tabs.setTabVisible(i, visible)
        self._apply_developer_mode_visibility()
        if self._visible_tab_count() == 0 and self.tabs.count() > 0:
            self.tabs.setTabVisible(0, True)
        if not self.tabs.isTabVisible(self.tabs.currentIndex()):
            for i in range(self.tabs.count()):
                if self.tabs.isTabVisible(i):
                    self.tabs.setCurrentIndex(i)
                    break
        self._update_view_toggle_buttons()

    def _update_view_toggle_buttons(self):
        if not hasattr(self, "btn_toggle_sidebar"):
            return
        sidebar_target = self.left_sidebar_stack if hasattr(self, "left_sidebar_stack") else self.sidebar
        set_button_icon(
            self.btn_toggle_sidebar,
            "panel-left" if sidebar_target.isVisible() else "panel-right-close",
            "",
            16,
        )
        set_button_icon(
            self.btn_toggle_topbar,
            "panel-top" if self.tabs.tabBar().isVisible() else "panel-top-close",
            "",
            16,
        )
        set_button_icon(self.btn_tab_menu, "more-horizontal", "", 16)
        set_button_icon(self.btn_settings, "config", "", 16)

    def eventFilter(self, obj, event):
        if (hasattr(self, "tabs") and obj is self.tabs.tabBar()
                and event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.RightButton):
            idx = self.tabs.tabBar().tabAt(event.pos())
            if idx >= 0:
                if self.tabs.isTabVisible(idx):
                    self.tabs.setCurrentIndex(idx)
                self._show_tab_visibility_menu(self.tabs.tabBar().mapToGlobal(event.pos()))
                return True
        parent_event_filter = getattr(super(), "eventFilter", None)
        if callable(parent_event_filter):
            return parent_event_filter(obj, event)
        return False

    def _update_theme_action_label(self):
        if not hasattr(self, "_theme_action"):
            return
        self._theme_action.setIcon(icon("appearance"))
        if CURRENT_THEME == "light":
            self._theme_action.setText("Switch to Dark Theme")
        else:
            self._theme_action.setText("Switch to Light Theme")

    def _refresh_tab_icons(self):
        tab_icons = {
            "Chat": "chat",
            "Models": "models",
            "Server": "server",
            "API Models": "api",
            "Accounts": "key",
            "Download": "download",
            "Dev": "code",
            "Appearance": "appearance",
        }
        for i in range(self.tabs.count()):
            name = tab_icons.get(self.tabs.tabText(i))
            if name:
                self.tabs.setTabIcon(i, icon(name))
        if hasattr(self, "accounts_tab"):
            self.accounts_tab.refresh_icons()
        if hasattr(self, "labs_tab"):
            self.labs_tab.refresh_icons()
        if hasattr(self, "settings_nav"):
            settings_icons = {
                "General": "config",
                "Docs": "docs",
                "Hugging Face": "huggingface",
                "Ollama": "ollama",
                "Server": "server",
                "Appearance": "appearance",
                "Accounts": "key",
            }
            for i in range(self.settings_nav.count()):
                item = self.settings_nav.item(i)
                if item:
                    item.setIcon(icon(settings_icons.get(item.text(), "config")))
            self._refresh_settings_nav_colors()
        if hasattr(self, "dev_nav"):
            dev_icons = {
                "Labs": "labs",
                "Logs": "logs",
                "Integrations": "integrations",
                "API Server": "server",
                "Pipeline": "pipeline",
                "MCP": "mcp",
                "Skills": "lightbulb",
            }
            for i in range(self.dev_nav.count()):
                item = self.dev_nav.item(i)
                if item:
                    item.setIcon(icon(dev_icons.get(item.text(), "code")))
            self._refresh_dev_nav_colors()

    def _refresh_svg_icons(self):
        refresh_widget_icons(self)
        if hasattr(self, "tabs"):
            self._refresh_tab_icons()
        self._update_view_toggle_buttons()
        self._update_theme_action_label()

    def _on_appearance_changed(self, new_palette: dict):
        global C_LIGHT, C_DARK, C, QSS
        if CURRENT_THEME == "light":
            set_theme(CURRENT_THEME, light_custom=dict(new_palette))
        else:
            set_theme(CURRENT_THEME, dark_custom=dict(new_palette))
        QSS = build_qss(C)
        self.setStyleSheet(QSS)
        apply_theme_palette(self, C)
        self._refresh_svg_icons()
        if self.active:
            self._switch_session(self.active.id)

    def _toggle_theme(self):
        global CURRENT_THEME, C, QSS
        CURRENT_THEME = "dark" if CURRENT_THEME == "light" else "light"
        from nativelab.UI.UI_const import set_theme, C

        set_theme(CURRENT_THEME)           
        apply_theme_palette(self, C)
        self.setStyleSheet(build_qss(C))
        APP_CONFIG["theme"] = CURRENT_THEME
        save_app_config(APP_CONFIG)
        self._update_theme_action_label()
        self._log("INFO", f"Theme switched to: {CURRENT_THEME}")

        # Rebuild Models tab (has baked dark card gradients)
        mt_idx = self.tabs.indexOf(self.models_tab)
        self.models_tab.setParent(None)
        self.models_tab.deleteLater()
        self.models_tab = self._build_models_tab()
        self.tabs.insertTab(mt_idx, self.models_tab, icon("models"), "Models")

        # Rebuild Settings panel pages with baked theme colors.
        self._rebuild_settings_panel()

        # Rebuild Download tab
        dl_idx = self.tabs.indexOf(self.download_tab)
        if hasattr(self.download_tab, "shutdown"):
            try:
                self.download_tab.shutdown()
            except Exception as exc:
                self._log("WARN", f"Download shutdown before rebuild failed: {exc}")
        self.download_tab.setParent(None)
        self.download_tab.deleteLater()
        self.download_tab = ModelDownloadTab()
        self.download_tab.hf_login_requested.connect(self._show_hf_login_tab)
        self.tabs.insertTab(dl_idx, self.download_tab, icon("download"), "Download")

        # Rebuild MCP dev page
        mcp_idx = self.dev_stack.indexOf(self.mcp_tab)
        old_mcp = self.mcp_tab
        self.mcp_tab = McpTab()
        if mcp_idx >= 0:
            self.dev_stack.removeWidget(old_mcp)
            self.dev_stack.insertWidget(mcp_idx, self.mcp_tab)
        old_mcp.setParent(None)
        old_mcp.deleteLater()

        # Rebuild Logs dev page
        log_idx = self.dev_stack.indexOf(self.log_console)
        old_log = self.log_console
        self.log_console = LogConsole()
        if log_idx >= 0:
            self.dev_stack.removeWidget(old_log)
            self.dev_stack.insertWidget(log_idx, self.log_console)
        old_log.setParent(None)
        old_log.deleteLater()
        try:
            self._lab_endpoints.log_msg.connect(self.log_console.log)
        except Exception:
            pass

        # Refresh Appearance tab palette to match active theme
        self.appearance_tab.load_palette(C_LIGHT if CURRENT_THEME == "light" else C_DARK)
        self._apply_saved_view_state()

        # Rebuild chat bubbles + status bar colours
        self._set_engine_status_from_engine(self._active_engine_for(""))
        if self.active:
            self._switch_session(self.active.id)
        apply_theme_palette(self, C)
        self._refresh_svg_icons()

    def _goto_logs(self):
        self._show_dev_page("Logs")

    def _goto_models_tab(self):
        self.tabs.setCurrentWidget(self.models_tab)

    def _shutdown_child_widget(self, attr: str, stuck: list[str]) -> None:
        widget = getattr(self, attr, None)
        if widget is None or not hasattr(widget, "shutdown"):
            return
        try:
            ok = widget.shutdown()
            if ok is False:
                stuck.append(attr)
        except Exception as exc:
            self._log("WARN", f"Shutdown hook failed for {attr}: {exc}")

    def _warn_shutdown_blocked(self, stuck: list[str]) -> None:
        names = ", ".join(stuck)
        self._log("WARN", f"Shutdown delayed; still-running worker(s): {names}")
        try:
            QMessageBox.warning(
                self,
                "NativeLab Is Still Working",
                "NativeLab is still stopping background work:\n\n"
                f"{names}\n\nTry closing again after it finishes.",
            )
        except Exception:
            pass

    def closeEvent(self, event):
        self._log("INFO", "Shutdown - stopping engine…")
        stuck: list[str] = []
        stuck += stop_worker_attrs(
            self,
            (
                "_worker",
                "_pipeline_worker",
                "_chat_pipeline_worker",
                "_summary_worker",
                "_multi_pdf_worker",
                "_auto_setup_worker",
            ),
            LONG_TIMEOUT_MS,
            delete_part=False,
        )
        stuck += stop_worker_attrs(
            self,
            ("_loader", "_api_loader"),
            LONG_TIMEOUT_MS,
            abort=False,
            cancel=False,
        )
        if hasattr(self, "pipeline_tab") and getattr(self.pipeline_tab, "_exec_worker", None):
            if stop_worker(self.pipeline_tab._exec_worker, LONG_TIMEOUT_MS, delete_part=False):
                self.pipeline_tab._exec_worker = None
            else:
                stuck.append("pipeline_tab._exec_worker")

        for child_attr in ("download_tab", "accounts_tab", "api_server_tab", "integrations_tab", "mcp_tab"):
            self._shutdown_child_widget(child_attr, stuck)

        # Cancel any pending role loaders gracefully before shutting down engines.
        for role in ("reasoning", "summarization", "coding", "secondary"):
            attr = f"_loader_{role}"
            if not stop_worker(getattr(self, attr, None), LONG_TIMEOUT_MS, abort=False, cancel=False):
                stuck.append(attr)
            else:
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

        if stuck:
            self._warn_shutdown_blocked(stuck)
            try:
                event.ignore()
            except Exception:
                pass
            return

        self.engine.shutdown()
        for role in ("reasoning", "summarization", "coding", "secondary"):
            eng = getattr(self, f"{role}_engine", None)
            if eng: eng.shutdown()

        # Final safety net: kill ANY remaining llama-server stragglers
        # (handles orphans from previous crashed sessions too)
        kill_stray_llama_servers(keep_pids=set())
        parent_close = getattr(super(), "closeEvent", None)
        if callable(parent_close):
            parent_close(event)
