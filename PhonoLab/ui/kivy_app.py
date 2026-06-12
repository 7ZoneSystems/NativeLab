"""PhonoLab Kivy UI - NativeLab-themed mobile interface.

Dark theme matching NativeLab's studio palette with PhonoLab's teal accent.
"""

from __future__ import annotations

import threading
from typing import Callable

from PhonoLab.config import load_config
from PhonoLab.downloads import download_candidate_model
from PhonoLab.engine import MobileLlamaCppEngine
from PhonoLab.hardware import profile_hardware
from PhonoLab.llama_cpp_setup import current_plan, pull_llama_cpp
from PhonoLab.registry import get_registry
from PhonoLab.safety import explain_error
from PhonoLab.small_models import SMALL_MODEL_CATALOG, by_key, choose_candidate

# ── NativeLab Dark Theme Palette ─────────────────────────────────────────
C = {
    "bg0": (0.035, 0.035, 0.051, 1),      # #09090d
    "bg1": (0.059, 0.059, 0.082, 1),      # #0f0f15
    "bg2": (0.078, 0.078, 0.125, 1),      # #141420
    "surface": (0.118, 0.118, 0.180, 1),  # #1e1e2e
    "surface2": (0.145, 0.145, 0.220, 1), # #252538
    "accent": (0.333, 0.761, 0.643, 1),   # #55C2A4
    "accent_dim": (0.102, 0.239, 0.200, 1), # #1a3d33
    "txt": (0.929, 0.929, 0.961, 1),      # #ededf5
    "txt2": (0.478, 0.478, 0.604, 1),     # #7a7a9a
    "txt3": (0.282, 0.282, 0.369, 1),     # #48485e
    "bdr": (0.137, 0.137, 0.208, 1),      # #232335
    "bdr2": (0.176, 0.176, 0.271, 1),     # #2d2d45
    "ok": (0.110, 0.722, 0.541, 1),       # #1cb88a
    "warn": (0.910, 0.592, 0.102, 1),     # #e8971a
    "err": (0.910, 0.282, 0.282, 1),      # #e84848
}


def run() -> int:
    from kivy.app import App
    from kivy.clock import Clock
    from kivy.core.window import Window
    from kivy.metrics import dp, sp
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.label import Label
    from kivy.uix.scrollview import ScrollView
    from kivy.uix.spinner import Spinner
    from kivy.uix.textinput import TextInput

    class PhonoLabApp(App):
        title = "PhonoLab"

        def build(self):
            Window.softinput_mode = "below_target"
            Window.clearcolor = C["bg0"]

            self.config = load_config()
            self.registry = get_registry()
            self.engine = MobileLlamaCppEngine(self.config)

            root = BoxLayout(
                orientation="vertical",
                spacing=dp(8),
                padding=dp(14),
            )

            # ── Status Card ─────────────────────────────────────────
            status_card = BoxLayout(
                orientation="horizontal",
                size_hint_y=None,
                height=dp(40),
                padding=[dp(12), dp(6)],
            )
            # Card background via canvas
            with status_card.canvas.before:
                from kivy.graphics import Color, RoundedRectangle
                Color(*C["surface"])
                self._status_bg = RoundedRectangle(
                    pos=status_card.pos,
                    size=status_card.size,
                    radius=[dp(8)],
                )
                Color(*C["bdr"])
                self._status_border = RoundedRectangle(
                    pos=status_card.pos,
                    size=status_card.size,
                    radius=[dp(8)],
                    width=dp(1),
                )
            status_card.bind(pos=self._update_status_bg, size=self._update_status_bg)

            self.status_icon = Label(
                text="●",
                font_size=sp(14),
                color=C["txt3"],
                size_hint_x=None,
                width=dp(20),
                halign="center",
                valign="middle",
            )
            self.status = Label(
                text="PhonoLab mobile node",
                font_size=sp(13),
                color=C["txt2"],
                halign="left",
                valign="middle",
            )
            self.status.bind(size=lambda obj, _: setattr(obj, "text_size", obj.size))
            status_card.add_widget(self.status_icon)
            status_card.add_widget(self.status)
            root.add_widget(status_card)

            # ── Section Label: MODEL ─────────────────────────────────
            root.add_widget(self._section_label("MODEL"))

            # ── Model Spinner Card ───────────────────────────────────
            model_card = BoxLayout(
                orientation="vertical",
                size_hint_y=None,
                height=dp(52),
                padding=[dp(12), dp(8)],
            )
            with model_card.canvas.before:
                Color(*C["surface"])
                self._model_bg = RoundedRectangle(
                    pos=model_card.pos,
                    size=model_card.size,
                    radius=[dp(8)],
                )

            self.model_spinner = Spinner(
                text="Select model",
                values=[],
                size_hint_x=1,
                font_size=sp(13),
                background_color=C["surface2"],
                color=C["txt"],
                option_cls=self._spinner_option_factory(),
            )
            self.refresh_models()
            model_card.add_widget(self.model_spinner)
            root.add_widget(model_card)

            # ── Section Label: RUNTIME ───────────────────────────────
            root.add_widget(self._section_label("RUNTIME"))

            # ── Actions Card ─────────────────────────────────────────
            actions_card = BoxLayout(
                orientation="horizontal",
                size_hint_y=None,
                height=dp(48),
                spacing=dp(6),
                padding=[dp(8), dp(4)],
            )
            with actions_card.canvas.before:
                Color(*C["surface"])
                self._actions_bg = RoundedRectangle(
                    pos=actions_card.pos,
                    size=actions_card.size,
                    radius=[dp(8)],
                )

            self.setup_btn = self._accent_button("Setup")
            self.download_btn = self._outline_button("Download")
            self.load_btn = self._outline_button("Load")

            self.setup_btn.bind(on_release=lambda *_: self.run_setup())
            self.download_btn.bind(on_release=lambda *_: self.download_model())
            self.load_btn.bind(on_release=lambda *_: self.load_model())

            actions_card.add_widget(self.setup_btn)
            actions_card.add_widget(self.download_btn)
            actions_card.add_widget(self.load_btn)
            root.add_widget(actions_card)

            # ── Progress Bar ─────────────────────────────────────────
            from kivy.uix.progressbar import ProgressBar
            self.progress = ProgressBar(
                max=100,
                value=0,
                size_hint_y=None,
                height=dp(4),
            )
            root.add_widget(self.progress)

            # ── Section Label: CHAT ──────────────────────────────────
            root.add_widget(self._section_label("CHAT"))

            # ── Chat Card ────────────────────────────────────────────
            chat_card = BoxLayout(
                orientation="vertical",
                size_hint_y=1,
                padding=[dp(2), dp(2)],
            )
            with chat_card.canvas.before:
                Color(*C["surface"])
                self._chat_bg = RoundedRectangle(
                    pos=chat_card.pos,
                    size=chat_card.size,
                    radius=[dp(8)],
                )

            self.chat = TextInput(
                readonly=True,
                multiline=True,
                size_hint_y=1,
                background_color=C["bg2"],
                foreground_color=C["txt"],
                font_size=sp(14),
                padding=[dp(12), dp(10)],
                cursor_color=C["accent"],
                selection_color=C["accent_dim"],
                hint_text="",
            )
            chat_scroll = ScrollView()
            chat_scroll.add_widget(self.chat)
            chat_card.add_widget(chat_scroll)
            root.add_widget(chat_card)

            # ── Input Bar ────────────────────────────────────────────
            input_bar = BoxLayout(
                orientation="horizontal",
                size_hint_y=None,
                height=dp(56),
                spacing=dp(8),
                padding=[dp(8), dp(6)],
            )
            with input_bar.canvas.before:
                Color(*C["surface"])
                self._input_bg = RoundedRectangle(
                    pos=input_bar.pos,
                    size=input_bar.size,
                    radius=[dp(8)],
                )

            self.prompt = TextInput(
                multiline=True,
                hint_text="Message the loaded local model",
                font_size=sp(14),
                background_color=C["surface2"],
                foreground_color=C["txt"],
                hint_text_color=C["txt3"],
                padding=[dp(12), dp(10)],
                cursor_color=C["accent"],
                size_hint_x=1,
            )
            self.send_btn = self._accent_button("Send")
            self.send_btn.size_hint_x = None
            self.send_btn.width = dp(72)
            self.send_btn.bind(on_release=lambda *_: self.send_prompt())

            input_bar.add_widget(self.prompt)
            input_bar.add_widget(self.send_btn)
            root.add_widget(input_bar)

            self.set_status("idle", current_plan().message)
            return root

        # ── Theme Helpers ────────────────────────────────────────────

        def _section_label(self, text: str) -> Label:
            return Label(
                text=text,
                font_size=sp(10),
                color=C["txt3"],
                size_hint_y=None,
                height=dp(18),
                halign="left",
                valign="middle",
                bold=True,
            )

        def _accent_button(self, text: str) -> Button:
            return Button(
                text=text,
                font_size=sp(13),
                background_color=C["accent"],
                color=(1, 1, 1, 1),
                bold=True,
                size_hint_x=1,
            )

        def _outline_button(self, text: str) -> Button:
            return Button(
                text=text,
                font_size=sp(13),
                background_color=C["surface2"],
                color=C["txt"],
                size_hint_x=1,
            )

        def _spinner_option_factory(self):
            from kivy.uix.spinner import SpinnerOption

            class ThemedOption(SpinnerOption):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.background_color = C["surface"]
                    self.color = C["txt"]
                    self.font_size = sp(13)

            return ThemedOption

        def _update_status_bg(self, instance, _value):
            self._status_bg.pos = instance.pos
            self._status_bg.size = instance.size
            self._status_border.pos = instance.pos
            self._status_border.size = instance.size

        # ── Status & Chat ────────────────────────────────────────────

        def set_status(self, state: str, text: str) -> None:
            color_map = {
                "ok": C["ok"],
                "warn": C["warn"],
                "err": C["err"],
                "loading": C["accent"],
                "generating": C["accent"],
                "downloading": C["warn"],
            }
            self.status_icon.color = color_map.get(state, C["txt3"])
            self.status.text = str(text or "")

        def append_chat(self, text: str) -> None:
            self.chat.text = (self.chat.text + str(text)).strip() + "\n"
            self.chat.cursor = (0, len(self.chat.text))

        def call_ui(self, fn: Callable, *args) -> None:
            Clock.schedule_once(lambda _dt: fn(*args), 0)

        def run_background(self, label: str, fn: Callable[[], None]) -> None:
            self.set_status("loading", label)

            def worker():
                try:
                    fn()
                except Exception as exc:
                    notice = explain_error(exc, source="PhonoLab")
                    self.call_ui(self.set_status, "err", notice.title)
                    self.call_ui(self.append_chat, notice.user_message)

            threading.Thread(target=worker, daemon=True).start()

        # ── Actions ──────────────────────────────────────────────────

        def refresh_models(self) -> None:
            models = get_registry().discover()
            catalog_values = [f"catalog:{item.key}" for item in SMALL_MODEL_CATALOG]
            values = [m.path for m in models] + catalog_values
            self.model_spinner.values = values
            if self.config.active_model and self.config.active_model in values:
                self.model_spinner.text = self.config.active_model
            elif values:
                self.model_spinner.text = values[0]

        def run_setup(self) -> None:
            def task():
                def progress(done: int, total: int, label: str) -> None:
                    if total:
                        pct = int(done * 100 / total)
                        self.call_ui(self.set_status, "downloading", f"{label}: {pct}%")
                    else:
                        self.call_ui(self.set_status, "downloading", f"{label}: {done} bytes")

                plan = pull_llama_cpp(progress)
                self.call_ui(self.set_status, "ok", plan.message)

            self.run_background("Pulling llama.cpp...", task)

        def download_model(self) -> None:
            hw = profile_hardware()
            selected = choose_candidate(hw)
            if self.model_spinner.text.startswith("catalog:"):
                chosen = by_key(self.model_spinner.text.split(":", 1)[1])
                selected = chosen or selected

            def task():
                def progress(done: int, total: int, label: str) -> None:
                    pct = int(done * 100 / total) if total else 0
                    self.call_ui(self.set_status, "downloading", f"Downloading {label}: {pct}%")

                path, file_info = download_candidate_model(selected, progress=progress, config=self.config)
                get_registry().add(path, repo=selected.repo, quant=file_info.name, set_active=True)
                self.call_ui(self.refresh_models)
                self.call_ui(self.set_status, "ok", f"Downloaded {path.name}")

            self.run_background(f"Downloading {selected.label}...", task)

        def load_model(self) -> None:
            selected = self.model_spinner.text if self.model_spinner.text != "Select model" else ""
            if selected.startswith("catalog:"):
                self.set_status("warn", "Download the selected catalog model before loading it.")
                return

            def task():
                self.engine = MobileLlamaCppEngine(load_config())
                self.engine.load(selected)
                self.call_ui(self.set_status, "ok", "Model loaded")

            self.run_background("Loading model...", task)

        def send_prompt(self) -> None:
            prompt = self.prompt.text.strip()
            if not prompt:
                return
            self.prompt.text = ""
            self.append_chat(f"User: {prompt}\nAssistant: ")

            def task():
                if not self.engine.is_loaded:
                    self.engine.load(self.model_spinner.text if self.model_spinner.text != "Select model" else "")

                def token_cb(token: str) -> None:
                    self.call_ui(self._append_token, token)

                self.engine.generate(prompt, token_cb=token_cb)
                self.call_ui(self._append_token, "\n")
                self.call_ui(self.set_status, "ok", "Ready")

            self.run_background("Generating...", task)

        def _append_token(self, token: str) -> None:
            self.chat.text += token

    PhonoLabApp().run()
    return 0
