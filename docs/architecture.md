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
│  LlamaEngine        — manages llama-server / llama-cli subprocesses     │
│   ├── ServerStreamWorker  (HTTP /completion streaming)                  │
│   └── CliStreamWorker     (per-prompt llama-cli stdout)                 │
│  ApiEngine          — OpenAI / Anthropic compatible                     │
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

The frontends never call the engine layer directly — everything goes through `LabEndpoints`. That's how the CLI and the GUI stay in sync without duplicating code.

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

**Reverse routing** — frontends or labs request state changes from the host app:

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

- `load(model_path, ctx, log_cb)` — start server or switch to CLI mode.
- `create_worker(prompt, n_predict, model_path)` — returns a streaming `QThread` for the GUI.
- `ensure_server(log_cb)` — bring the server up if currently in CLI mode.
- `shutdown()` — kill child processes and reset state.

Status flags: `is_loaded`, `mode`, `status_text`, `server_port`.

### ApiEngine

Drop-in replacement that calls a remote API. Same shape (`load`, `create_worker`, `is_loaded`, `status_text`) so pipeline/summarization/reference workers don't care which engine they're running through.

Both engines are read by `LabEndpoints.active_engine()` — API takes priority when loaded, otherwise local.

---

## Project structure

```
NativeLab/
├── README.md                      ← landing page
├── docs/                          ← this folder
│   ├── README.md                  ← docs index
│   ├── installation.md
│   ├── cli.md
│   ├── features.md
│   ├── architecture.md            ← you are here
│   ├── labs.md
│   ├── models.md
│   ├── workflows.md
│   ├── ui.md
│   └── troubleshooting.md
├── nativelab/
│   ├── icon.png · icon.ico        ← branding
│   ├── main.py                    ← MainWindow + GUI entry
│   ├── __main__.py                ← routes --cli to nativelab.cli, else GUI
│   ├── manual.py
│   │
│   ├── labs/                      ← experimentation layer (NEW)
│   │   ├── __init__.py            ← re-exports LabEndpoints, LabsTab
│   │   ├── endpoints.py           ← shared endpoint surface
│   │   ├── labs_tab.py            ← sidebar + stacked panels
│   │   └── pytodoc.py             ← first feature: py-to-doc
│   │
│   ├── cli/                       ← terminal client (NEW)
│   │   ├── __init__.py
│   │   ├── app.py                 ← argparse dispatcher
│   │   ├── onboarding.py          ← setup wizard
│   │   ├── chat.py                ← REPL with @file embedding
│   │   ├── hf_download.py         ← sync HF list + download
│   │   ├── lint.py                ← pyflakes/flake8/pylint
│   │   ├── ui.py                  ← ANSI colors + icon renderer
│   │   └── cli_guide.md           ← beginner walkthrough
│   │
│   ├── core/
│   │   ├── engine_global.py       ← exports LlamaEngine + ApiEngine
│   │   ├── engines/
│   │   │   ├── llamaengine.py
│   │   │   └── apiengine.py
│   │   ├── streamer_global.py
│   │   └── streamerworker/
│   │       ├── serverstreamer.py
│   │       ├── clistreamer.py
│   │       └── apistreamer.py
│   │
│   ├── Server/
│   │   ├── ServerHandling.py      ← ServerConfig, Session, free_port
│   │   ├── hfdwld.py              ← HuggingFace + llama.cpp downloaders
│   │   └── server_global.py
│   │
│   ├── Model/
│   │   ├── ModelRegistry.py       ← per-model configs + custom paths
│   │   ├── model_family.py
│   │   ├── templates.py           ← FAMILY_TEMPLATES (20+ chat formats)
│   │   ├── APImodels.py           ← ApiConfig, ApiRegistry
│   │   └── model_global.py
│   │
│   ├── codeparser/
│   │   ├── refrenceengine.py      ← SmartReference, store, RAM watchdog
│   │   ├── scriptparser.py        ← AST + regex parsers
│   │   └── parser/
│   │       ├── parsefinal.py
│   │       └── typeparser.py
│   │
│   ├── components/
│   │   ├── pdfsummarise.py        ← ChunkedSummaryWorker
│   │   ├── multipdf_summarise.py  ← MultiPdfSummaryWorker
│   │   ├── reason_code_pipeline.py ← parallel reasoning + coding
│   │   └── jobhandler.py          ← paused-job persistence
│   │
│   ├── pipelinebuilder/
│   │   ├── pipebuilder.py         ← PipelineBuilderTab
│   │   ├── canvas.py              ← drag-drop node canvas
│   │   ├── pipblck.py · blck_typ.py
│   │   ├── editordialogue.py      ← LLM logic editor
│   │   ├── executionWorker.py
│   │   ├── outrender.py
│   │   ├── pipefunctions.py
│   │   ├── flowpreview.py
│   │   └── pipe_global.py
│   │
│   ├── Prefrences/
│   │   ├── ParallelLoading.py     ← ParallelPrefs
│   │   └── prefrence_global.py
│   │
│   ├── GlobalConfig/
│   │   ├── config.py              ← DEFAULT_*, MODELS_DIR, paths
│   │   ├── binaryResolve.py       ← LLAMA_CLI/SERVER resolution
│   │   ├── const.py               ← APP_CONFIG_DEFAULTS
│   │   ├── hardwareUtil.py
│   │   └── config_global.py
│   │
│   ├── imports/
│   │   ├── pyqt_lib.py · standard_lib.py · optional_lib.py
│   │   └── import_global.py
│   │
│   └── UI/
│       ├── tabs.py                ← ConfigTab, ServerTab, ModelDownloadTab,
│       │                              McpTab, ApiModelsTab, AppearanceTab, …
│       ├── buildUI.py             ← QSS theming
│       ├── UI_const.py · UI_global.py
│       ├── RichTextEditor.py
│       ├── md_to_html.py · effects.py · widgets.py
│       ├── labs_tab.py            ← back-compat re-export shim
│       └── Qt6widgets/
│           ├── chatarea.py · chatmodule.py
│           ├── inputbar.py · messagewidget.py
│           ├── refrencepanels.py · sessionsidebar.py
│           └── thinkingblock.py
│
├── pyproject.toml                  ← PyPI metadata
├── requirements.txt
├── setup.html · index.html
├── NativeLab.spec                  ← PyInstaller
├── LICENSE                         ← AGPL v3
├── CONTRIBUTING.md · CODE_OF_CONDUCT.md · SECURITY.md
└── .github/
    ├── ISSUE_TEMPLATE/
    ├── PULL_REQUEST_TEMPLATE.md
    └── workflows/build-mac.yml
```

---

## Threading model

- All inference (streaming tokens, summarization, pipeline stages, downloads, MCP probes) runs on `QThread` subclasses with PyQt signals for cross-thread updates. The main thread never blocks.
- Workers expose `abort()` that flips a flag checked at every iteration for clean cancellation.
- Summary workers additionally support `request_pause()`, which writes a state snapshot to `paused_jobs/` before exiting.
- The CLI uses synchronous calls (`endpoints.call_llm`) since it has no UI to keep responsive — same backend, no QThread plumbing.

---

## Stray-process cleanup

On shutdown, `kill_stray_llama_servers()` terminates orphaned `llama-server` processes from previous crashed sessions in addition to the ones currently managed. This prevents port leaks across restarts.
