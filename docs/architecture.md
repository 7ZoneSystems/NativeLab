# Architecture

NativeLab is a layered application. Each layer talks only to the one beneath it, which keeps the GUI, CLI, and Labs frontends interchangeable on top of the same engine surface.

---

## Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontends                                  │
│  ┌──────────────────────┐   ┌──────────────────┐   ┌────────────────┐   │
│  │  MainWindow (PyQt6)  │   │  CLI ChatREPL    │   │  Lab Panels    │   │
│  │  • Chat / Models     │   │  (cli/chat.py)   │   │  (labs/*.py)   │   │
│  │  • Pipeline / Server │   │                  │   │                │   │
│  │  • MCP / Download    │   │                  │   │                │   │
│  │  • Labs / Logs       │   │                  │   │                │   │
│  └──────────────────────┘   └──────────────────┘   └────────────────┘   │
│              │                       │                    │             │
│              └───────────┬───────────┴────────────────────┘             │
│                          ▼                                              │
│              ┌───────────────────────────┐                              │
│              │  LabEndpoints (labs/)     │  ← shared, single surface    │
│              │   • status_text / model   │                              │
│              │   • call_llm() sync       │                              │
│              │   • request_*() reverse   │                              │
│              └───────────────────────────┘                              │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Engine layer                                 │
│  LlamaEngine        - manages llama-server / llama-cli subprocesses     │
│   ├── ServerStreamWorker  (HTTP /completion streaming)                  │
│   └── CliStreamWorker     (per-prompt llama-cli stdout)                 │
│  ApiEngine          - OpenAI / Anthropic compatible                     │
│   └── ApiStreamWorker     (HTTP streaming)                              │
│  PipelineExecutionWorker  ChunkedSummaryWorker  MultiPdfSummaryWorker   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Reference engine                                │
│  SessionReferenceStore  ─►  SmartReference                              │
│                              ScriptSmartReference  ─►  ScriptParser     │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Persistence                                    │
│  Session (JSON in sessions/)         ApiRegistry (api_models.json)      │
│  ModelRegistry (custom_models.json + model_configs.json)                │
│  ServerConfig (server_config.json)   APP_CONFIG (app_config.json)       │
│  ParallelPrefs (parallel_prefs.json) PausedJobs (paused_jobs/)          │
│  McpConfig (mcp_config.json)         CLI prefs (cli_prefs.json)         │
└─────────────────────────────────────────────────────────────────────────┘
```

The frontends never call the engine layer directly - everything goes through `LabEndpoints`. That's how the CLI and the GUI stay in sync without duplicating code.

---

## The LabEndpoints surface

`nativelab/labs/endpoints.py` is the contract between frontends and engines.

**Read state**

```python
endpoints.status_text        # "🟢 Server  :8612"
endpoints.model_path         # absolute path to the active GGUF
endpoints.model_name         # filename only
endpoints.mode               # "server" | "cli" | "api" | "unloaded"
endpoints.ctx_value          # int
endpoints.server_port        # int
endpoints.is_loaded          # bool
endpoints.snapshot()         # dict of all of the above
endpoints.model_family()     # ModelFamily template (BOS/EOS, prefixes…)
```

**LLM calls** (synchronous, safe from a worker thread; auto-routes API → server → CLI)

```python
endpoints.call_llm(
    messages=[{"role": "user", "content": "…"}],
    system_prompt="You are…",
    n_predict=512,
    temperature=0.3,
)
```

**Reverse routing** - frontends or labs request state changes from the host app:

```python
endpoints.request_load_model("/path/to/model.gguf")
endpoints.request_context(8192)
endpoints.request_unload()
endpoints.ensure_server(log_cb=...)
```

**Signals** for live updates:

```python
endpoints.engine_changed.connect(handler)
endpoints.status_changed.connect(handler)   # str
```

The host (`MainWindow` for the GUI, `cli.chat._build_endpoints` for the CLI) wires the engine providers and reverse-route hooks once at startup. Everything downstream just uses the same instance.

---

## Engine layer

### LlamaEngine

The local inference engine. Tries `llama-server` first (HTTP streaming, model stays resident), falls back to `llama-cli` (per-prompt subprocess) if the server binary isn't available.

Key methods:

- `load(model_path, ctx, log_cb)` - start server or switch to CLI mode.
- `create_worker(prompt, n_predict, model_path)` - returns a streaming `QThread` for the GUI.
- `ensure_server(log_cb)` - bring the server up if currently in CLI mode.
- `shutdown()` - kill child processes and reset state.

Status flags: `is_loaded`, `mode`, `status_text`, `server_port`.

### ApiEngine

Drop-in replacement that calls a remote API. Same shape (`load`, `create_worker`, `is_loaded`, `status_text`) so pipeline/summarization/reference workers don't care which engine they're running through.

Both engines are read by `LabEndpoints.active_engine()` - API takes priority when loaded, otherwise local.

---

## GUI entrypoint and MainWindow modules

`nativelab/main.py` is intentionally small. It handles GUI startup, `QApplication`
creation, font/icon setup, SIGINT handling, and `MainWindow` launch.

The window implementation lives under `nativelab/UI/mainwindow/`:

| Module | Responsibility |
| --- | --- |
| `window.py` | MainWindow class and mixin composition. |
| `ui_build.py` | Top-level layout and tab construction. |
| `engine_runtime.py` | Local/API model load/unload and runtime state. |
| `auto_setup.py` | First-run and settings-triggered auto setup UI wiring. |
| `context_controls.py` | Context meter and context reload controls. |
| `chat_pipeline.py` | Chat/pipeline mode orchestration. |
| `documents.py` | References, summaries, multi-PDF jobs. |
| `labs.py` | Labs tab wiring and endpoint injection. |
| `models.py` | Model/API registry refresh and selection. |
| `sessions.py` | Session lifecycle and persistence. |
| `status_view.py` | Status bar, theme, and view helpers. |
| `shared.py` | Common imports/constants for the split package. |

GUI worker shutdown is centralized through `nativelab/UI/qt_workers.py` so close
events, theme rebuilds, downloads, auto setup, model loads, accounts workers,
and tab-specific workers stop consistently before widgets are deleted.

---

## Native helper boundary

NativeLab uses C/Rust only for deterministic hot paths. Python keeps ownership
of Qt widgets, plugin/backend orchestration, subprocess lifecycle, model/API
calls, and user-facing error handling.

| Native area | Files | Purpose |
| --- | --- | --- |
| Engine helpers | `nativelab/native/_core.c`, `engine_helpers.py` | Prompt assembly, sampler normalization, CLI sampler args, image/base64 extraction, context-error detection, reference chunk splitting. |
| Pipeline core | `nativelab/native/pipeline_core.c`, `pipeline_core.py` | Block ID normalization, connection remapping, loop/cycle checks, route selection, transform/merge helpers, validation records. |
| Model detection | `nativelab/native/rust_model.rs`, `rust_model.py` | Optional Rust-backed family and quant detection. |
| AI Builder helpers | `nativelab/pipelinebuilder/aibuilder/aibuilder_core.c`, `.rs` | Token estimation and JSON-object span detection for generated pipeline responses. |

All native helpers are optional. If `_native_core` or the Rust shared library is
not present, the Python fallback path remains active.

---

## Pipeline subsystem

The pipeline builder is split by responsibility:

| File/package | Responsibility |
| --- | --- |
| `pipebuilder.py` | PyQt tab, sidebars, Execution/AI Builder tabs, user actions. |
| `canvas.py` | Visual graph editing, block movement, port connections, panning, canvas growth. |
| `pipblck.py` / `blck_typ.py` | Block and connection data structures. |
| `pipefunctions.py` | Save/load/example pipeline JSON persistence. |
| `graph_ops.py` | Central graph operations and native-backed ID/loop helpers. |
| `execution_core.py` | Deterministic execution helpers used by the worker. |
| `validation.py` | Shared pipeline validation and user-facing validation messages. |
| `executionWorker.py` | QThread runtime for block execution and model calls. |
| `aibuilder/` | AI Pipeline Builder UI, prompt planning, JSON extraction, smart context, history. |
| `examples/` | Packaged example pipeline JSON presets. |

This keeps Python as a thin orchestration layer around shared validation and
execution primitives. UI code no longer owns graph invariants directly; loaded,
generated, CLI-run, and manually edited pipelines go through the same validation
and normalization path.

---

## Project structure

```
NativeLab/
├── changelog.txt
├── CODE_OF_CONDUCT.md
├── comt.sh
├── CONTRIBUTING.md
├── dist
│   ├── nativelab-0.1.0-py3-none-any.whl
│   ├── nativelab-0.1.0.tar.gz
│   ├── nativelab-0.1.1-py3-none-any.whl
│   ├── nativelab-0.1.1.tar.gz
│   ├── nativelab-0.1.2-py3-none-any.whl
│   ├── nativelab-0.1.2.tar.gz
│   ├── nativelab-0.1.3-py3-none-any.whl
│   ├── nativelab-0.1.3.tar.gz
│   ├── nativelab-0.1.4-py3-none-any.whl
│   ├── nativelab-0.1.4.tar.gz
│   ├── nativelab-0.1.5-py3-none-any.whl
│   ├── nativelab-0.1.5.tar.gz
│   ├── nativelab-0.1.6-py3-none-any.whl
│   ├── nativelab-0.1.6.tar.gz
│   ├── nativelab-0.1.7-py3-none-any.whl
│   ├── nativelab-0.1.7.tar.gz
│   ├── nativelab-0.1.8-py3-none-any.whl
│   ├── nativelab-0.1.8.tar.gz
│   ├── nativelab-0.2.2-py3-none-any.whl
│   ├── nativelab-0.2.2.tar.gz
│   ├── nativelab-0.2.3-py3-none-any.whl
│   ├── nativelab-0.2.3.tar.gz
│   ├── nativelab-0.2.4-py3-none-any.whl
│   ├── nativelab-0.2.4.tar.gz
│   ├── nativelab-0.2.5-py3-none-any.whl
│   ├── nativelab-0.2.5.tar.gz
│   ├── nativelab-0.2.7-py3-none-any.whl
│   ├── nativelab-0.2.7.tar.gz
│   ├── nativelab-0.2.8-py3-none-any.whl
│   ├── nativelab-0.2.8.tar.gz
│   ├── nativelab-0.2.9-py3-none-any.whl
│   ├── nativelab-0.2.9.tar.gz
│   ├── nativelab-0.3.0-py3-none-any.whl
│   ├── nativelab-0.3.0.tar.gz
│   ├── nativelab-0.3.1-py3-none-any.whl
│   ├── nativelab-0.3.1.tar.gz
│   ├── nativelab-0.3.2-py3-none-any.whl
│   ├── nativelab-0.3.2.tar.gz
│   ├── nativelab-0.3.3-py3-none-any.whl
│   ├── nativelab-0.3.3.tar.gz
│   ├── nativelab-0.3.4-py3-none-any.whl
│   ├── nativelab-0.3.4.tar.gz
│   ├── nativelab-0.3.7-py3-none-any.whl
│   └── nativelab-0.3.7.tar.gz
├── docs
│   ├── architecture.md
│   ├── cli.md
│   ├── features.md
│   ├── installation.md
│   ├── integrations.md
│   ├── labs.md
│   ├── models.md
│   ├── README.md
│   ├── troubleshooting.md
│   ├── ui.md
│   └── workflows.md
├── .github
│   ├── ISSUE_TEMPLATE
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows
│       ├── build-linux.yml
│       ├── build-mac.yml
│       ├── build-windows.yml
│       ├── clone-count.yml
│       └── release-apps.yml
├── .gitignore
├── google81d8b06f71e45c58.html
├── images
│   ├── appearance.png
│   ├── dark_mode.png
│   ├── dev.png
│   ├── image copy.png
│   ├── light_mode.png
│   ├── pipeline.png
│   ├── server_controls.png
│   └── skill.png
├── index.html
├── LICENSE
├── MANIFEST.in
├── nativelab
│   ├── api_server
│   │   ├── catalog.py
│   │   ├── config.py
│   │   ├── __init__.py
│   │   ├── protocol.py
│   │   ├── server.py
│   │   └── tab.py
│   ├── assets
│   │   ├── icons
│   │   │   ├── blocks.svg
│   │   │   ├── book-open.svg
│   │   │   ├── brain.svg
│   │   │   ├── bug.svg
│   │   │   ├── calendar.svg
│   │   │   ├── circle-alert.svg
│   │   │   ├── circle-check.svg
│   │   │   ├── circle-pause.svg
│   │   │   ├── circle.svg
│   │   │   ├── circle-x.svg
│   │   │   ├── clipboard-list.svg
│   │   │   ├── code-2.svg
│   │   │   ├── code.svg
│   │   │   ├── combine.svg
│   │   │   ├── copy.svg
│   │   │   ├── delete.svg
│   │   │   ├── discord.svg
│   │   │   ├── download.svg
│   │   │   ├── file-code.svg
│   │   │   ├── files.svg
│   │   │   ├── file.svg
│   │   │   ├── file-text.svg
│   │   │   ├── filter.svg
│   │   │   ├── flask-conical.svg
│   │   │   ├── folder-open.svg
│   │   │   ├── folder.svg
│   │   │   ├── git-branch.svg
│   │   │   ├── globe.svg
│   │   │   ├── history.svg
│   │   │   ├── huggingface.svg
│   │   │   ├── image.svg
│   │   │   ├── import.svg
│   │   │   ├── key.svg
│   │   │   ├── lightbulb.svg
│   │   │   ├── list.svg
│   │   │   ├── loader-circle.svg
│   │   │   ├── log-in.svg
│   │   │   ├── log-out.svg
│   │   │   ├── logs.svg
│   │   │   ├── manifest.json
│   │   │   ├── map.svg
│   │   │   ├── merge.svg
│   │   │   ├── message-circle.svg
│   │   │   ├── message-square.svg
│   │   │   ├── more-horizontal.svg
│   │   │   ├── ollama.svg
│   │   │   ├── omega.svg
│   │   │   ├── palette.svg
│   │   │   ├── panel-left.svg
│   │   │   ├── panel-right-close.svg
│   │   │   ├── panel-top-close.svg
│   │   │   ├── panel-top.svg
│   │   │   ├── paperclip.svg
│   │   │   ├── pencil.svg
│   │   │   ├── pi.svg
│   │   │   ├── play.svg
│   │   │   ├── plug.svg
│   │   │   ├── plus.svg
│   │   │   ├── power-off.svg
│   │   │   ├── projector.svg
│   │   │   ├── radius.svg
│   │   │   ├── refresh-cw.svg
│   │   │   ├── regex.svg
│   │   │   ├── replace.svg
│   │   │   ├── route.svg
│   │   │   ├── save.svg
│   │   │   ├── search.svg
│   │   │   ├── section.svg
│   │   │   ├── send.svg
│   │   │   ├── server.svg
│   │   │   ├── settings.svg
│   │   │   ├── shuffle.svg
│   │   │   ├── sigma.svg
│   │   │   ├── split.svg
│   │   │   ├── square-chevron-down.svg
│   │   │   ├── square-chevron-right.svg
│   │   │   ├── stop-circle.svg
│   │   │   ├── table.svg
│   │   │   ├── tag.svg
│   │   │   ├── test-tube.svg
│   │   │   ├── text.svg
│   │   │   ├── trash-2.svg
│   │   │   ├── triangle-alert.svg
│   │   │   ├── type.svg
│   │   │   ├── upload.svg
│   │   │   ├── user.svg
│   │   │   ├── view.svg
│   │   │   ├── whatsapp.svg
│   │   │   ├── workflow.svg
│   │   │   ├── wrench.svg
│   │   │   ├── x.svg
│   │   │   └── zap.svg
│   │   └── katex
│   │       ├── auto-render.min.js
│   │       ├── fonts
│   │       │   ├── KaTeX_AMS-Regular.ttf
│   │       │   ├── KaTeX_AMS-Regular.woff
│   │       │   ├── KaTeX_AMS-Regular.woff2
│   │       │   ├── KaTeX_Caligraphic-Bold.ttf
│   │       │   ├── KaTeX_Caligraphic-Bold.woff
│   │       │   ├── KaTeX_Caligraphic-Bold.woff2
│   │       │   ├── KaTeX_Caligraphic-Regular.ttf
│   │       │   ├── KaTeX_Caligraphic-Regular.woff
│   │       │   ├── KaTeX_Caligraphic-Regular.woff2
│   │       │   ├── KaTeX_Fraktur-Bold.ttf
│   │       │   ├── KaTeX_Fraktur-Bold.woff
│   │       │   ├── KaTeX_Fraktur-Bold.woff2
│   │       │   ├── KaTeX_Fraktur-Regular.ttf
│   │       │   ├── KaTeX_Fraktur-Regular.woff
│   │       │   ├── KaTeX_Fraktur-Regular.woff2
│   │       │   ├── KaTeX_Main-BoldItalic.ttf
│   │       │   ├── KaTeX_Main-BoldItalic.woff
│   │       │   ├── KaTeX_Main-BoldItalic.woff2
│   │       │   ├── KaTeX_Main-Bold.ttf
│   │       │   ├── KaTeX_Main-Bold.woff
│   │       │   ├── KaTeX_Main-Bold.woff2
│   │       │   ├── KaTeX_Main-Italic.ttf
│   │       │   ├── KaTeX_Main-Italic.woff
│   │       │   ├── KaTeX_Main-Italic.woff2
│   │       │   ├── KaTeX_Main-Regular.ttf
│   │       │   ├── KaTeX_Main-Regular.woff
│   │       │   ├── KaTeX_Main-Regular.woff2
│   │       │   ├── KaTeX_Math-BoldItalic.ttf
│   │       │   ├── KaTeX_Math-BoldItalic.woff
│   │       │   ├── KaTeX_Math-BoldItalic.woff2
│   │       │   ├── KaTeX_Math-Italic.ttf
│   │       │   ├── KaTeX_Math-Italic.woff
│   │       │   ├── KaTeX_Math-Italic.woff2
│   │       │   ├── KaTeX_SansSerif-Bold.ttf
│   │       │   ├── KaTeX_SansSerif-Bold.woff
│   │       │   ├── KaTeX_SansSerif-Bold.woff2
│   │       │   ├── KaTeX_SansSerif-Italic.ttf
│   │       │   ├── KaTeX_SansSerif-Italic.woff
│   │       │   ├── KaTeX_SansSerif-Italic.woff2
│   │       │   ├── KaTeX_SansSerif-Regular.ttf
│   │       │   ├── KaTeX_SansSerif-Regular.woff
│   │       │   ├── KaTeX_SansSerif-Regular.woff2
│   │       │   ├── KaTeX_Script-Regular.ttf
│   │       │   ├── KaTeX_Script-Regular.woff
│   │       │   ├── KaTeX_Script-Regular.woff2
│   │       │   ├── KaTeX_Size1-Regular.ttf
│   │       │   ├── KaTeX_Size1-Regular.woff
│   │       │   ├── KaTeX_Size1-Regular.woff2
│   │       │   ├── KaTeX_Size2-Regular.ttf
│   │       │   ├── KaTeX_Size2-Regular.woff
│   │       │   ├── KaTeX_Size2-Regular.woff2
│   │       │   ├── KaTeX_Size3-Regular.ttf
│   │       │   ├── KaTeX_Size3-Regular.woff
│   │       │   ├── KaTeX_Size3-Regular.woff2
│   │       │   ├── KaTeX_Size4-Regular.ttf
│   │       │   ├── KaTeX_Size4-Regular.woff
│   │       │   ├── KaTeX_Size4-Regular.woff2
│   │       │   ├── KaTeX_Typewriter-Regular.ttf
│   │       │   ├── KaTeX_Typewriter-Regular.woff
│   │       │   └── KaTeX_Typewriter-Regular.woff2
│   │       ├── katex.min.css
│   │       ├── katex.min.js
│   │       └── LICENSE
│   ├── cli
│   │   ├── app.py
│   │   ├── chat.py
│   │   ├── cli_guide.md
│   │   ├── features.py
│   │   ├── hf_download.py
│   │   ├── __init__.py
│   │   ├── lint.py
│   │   ├── onboarding.py
│   │   ├── runtime.py
│   │   └── ui.py
│   ├── codeparser
│   │   ├── codeparser_global.py
│   │   ├── __init__.py
│   │   ├── parser
│   │   │   ├── __init__.py
│   │   │   ├── parsefinal.py
│   │   │   └── typeparser.py
│   │   ├── refrenceengine.py
│   │   └── scriptparser.py
│   ├── components
│   │   ├── components_global.py
│   │   ├── __init__.py
│   │   ├── jobhandler.py
│   │   ├── multipdf_summarise.py
│   │   ├── pdfsummarise.py
│   │   └── reason_code_pipeline.py
│   ├── core
│   │   ├── auto_setup.py
│   │   ├── context_meter.py
│   │   ├── data_portability.py
│   │   ├── engine_global.py
│   │   ├── engines
│   │   │   ├── apiengine.py
│   │   │   ├── __init__.py
│   │   │   └── llamaengine.py
│   │   ├── engine_status.py
│   │   ├── __init__.py
│   │   ├── model_loaders.py
│   │   ├── streamer_global.py
│   │   └── streamerworker
│   │       ├── apistreamer.py
│   │       ├── backendstreamer.py
│   │       ├── clistreamer.py
│   │       ├── __init__.py
│   │       └── serverstreamer.py
│   ├── GlobalConfig
│   │   ├── binaryResolve.py
│   │   ├── config_global.py
│   │   ├── config.py
│   │   ├── const.py
│   │   ├── hardwareUtil.py
│   │   ├── __init__.py
│   │   └── timeouts.py
│   ├── icon.ico
│   ├── icon.png
│   ├── imports
│   │   ├── import_global.py
│   │   ├── __init__.py
│   │   ├── optional_lib.py
│   │   ├── pyqt_lib.py
│   │   ├── qt_compat.py
│   │   └── standard_lib.py
│   ├── integrations
│   │   ├── discord_connector.py
│   │   ├── endpoints.py
│   │   ├── examples
│   │   │   ├── discord_bot.env.example
│   │   │   ├── discord_bot.py
│   │   │   ├── __init__.py
│   │   │   ├── whatsapp_bot.env.example
│   │   │   └── whatsapp_bot.py
│   │   ├── http_endpoint.py
│   │   ├── __init__.py
│   │   ├── tab.py
│   │   └── whatsapp_connector.py
│   ├── labs
│   │   ├── codeedit.py
│   │   ├── endpoints.py
│   │   ├── __init__.py
│   │   ├── labs_tab.py
│   │   └── pytodoc.py
│   ├── __main__.py
│   ├── main.py
│   ├── manual.py
│   ├── Model
│   │   ├── APImodels.py
│   │   ├── __init__.py
│   │   ├── model_family.py
│   │   ├── model_global.py
│   │   ├── ModelRegistry.py
│   │   └── templates.py
│   ├── native
│   │   ├── _core.c
│   │   ├── engine_helpers.py
│   │   ├── __init__.py
│   │   ├── pipeline_core.c
│   │   ├── pipeline_core.py
│   │   ├── rust_model.py
│   │   └── rust_model.rs
│   ├── pipelinebuilder
│   │   ├── aibuilder
│   │   │   ├── aibuilder_core.c
│   │   │   ├── aibuilder_core.rs
│   │   │   ├── context.py
│   │   │   ├── dialog.py
│   │   │   ├── engine_call.py
│   │   │   ├── __init__.py
│   │   │   └── planner.py
│   │   ├── blck_typ.py
│   │   ├── canvas.py
│   │   ├── editordialogue.py
│   │   ├── executionWorker.py
│   │   ├── execution_core.py
│   │   ├── examples
│   │   │   └── *.json
│   │   ├── flowpreview.py
│   │   ├── graph_ops.py
│   │   ├── __init__.py
│   │   ├── outrender.py
│   │   ├── pipblck.py
│   │   ├── pipebuilder.py
│   │   ├── pipefunctions.py
│   │   ├── pipe_global.py
│   │   └── validation.py
│   ├── Prefrences
│   │   ├── __init__.py
│   │   ├── ParallelLoading.py
│   │   └── prefrence_global.py
│   ├── Server
│   │   ├── hfauth.py
│   │   ├── hf_deps.py
│   │   ├── hfdwld.py
│   │   ├── __init__.py
│   │   ├── ollama_helpers.py
│   │   ├── server_global.py
│   │   └── ServerHandling.py
│   ├── skill
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   └── tab.py
│   └── UI
│       ├── buildUI.py
│       ├── effects.py
│       ├── icons.py
│       ├── __init__.py
│       ├── labs_tab.py
│       ├── mainwindow
│       │   ├── auto_setup.py
│       │   ├── chat_pipeline.py
│       │   ├── context_controls.py
│       │   ├── documents.py
│       │   ├── engine_runtime.py
│       │   ├── __init__.py
│       │   ├── labs.py
│       │   ├── models.py
│       │   ├── sessions.py
│       │   ├── shared.py
│       │   ├── status_view.py
│       │   ├── ui_build.py
│       │   └── window.py
│       ├── md_to_html.py
│       ├── Qt6widgets
│       │   ├── chatarea.py
│       │   ├── chatmodule.py
│       │   ├── __init__.py
│       │   ├── inputbar.py
│       │   ├── messagewidget.py
│       │   ├── refrencepanels.py
│       │   ├── sessionsidebar.py
│       │   └── thinkingblock.py
│       ├── qt_workers.py
│       ├── RichTextEditor.py
│       ├── tabs.py
│       ├── toggle.py
│       ├── UI_const.py
│       ├── UI_global.py
│       └── widgets.py
├── NativeLab.spec
├── .nojekyll
├── pyproject.toml
├── README.md
├── requirements.txt
├── robots.txt
├── scripts
│   └── download_svg_icons.py
├── SECURITY.md
├── setup.py
├── sitemap.xml
├── tests
│   ├── test_auto_setup.py
│   ├── test_context_meter.py
│   ├── test_hf_deps.py
│   ├── test_mainwindow_split.py
│   ├── test_native_helpers.py
│   ├── test_pipeline_canvas_ids.py
│   ├── test_pipeline_examples.py
│   ├── test_pipeline_native_core.py
│   └── test_qt_workers.py
├── uv.lock
├── .vscode
│   └── settings.json
└── web_page
    ├── compare.html
    ├── features.html
    ├── pipeline.html
    ├── setup.html
    ├── site.css
    └── site.js
```

---

## Threading model

- All inference (streaming tokens, summarization, pipeline stages, downloads, MCP probes) runs on `QThread` subclasses with PyQt signals for cross-thread updates. The main thread never blocks.
- Workers expose `abort()` that flips a flag checked at every iteration for clean cancellation.
- `nativelab/UI/qt_workers.py` centralizes worker shutdown, signal disconnection, stuck-worker handling, and safe cleanup before UI widgets are deleted.
- Summary workers additionally support `request_pause()`, which writes a state snapshot to `paused_jobs/` before exiting.
- The CLI uses synchronous calls (`endpoints.call_llm`) since it has no UI to keep responsive - same backend, no QThread plumbing.

---

## Stray-process cleanup

On shutdown, `kill_stray_llama_servers()` terminates orphaned `llama-server` processes from previous crashed sessions in addition to the ones currently managed. This prevents port leaks across restarts.
