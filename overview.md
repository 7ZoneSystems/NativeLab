# NativeLab - Complete Overview

A deep look at everything this repository contains, what it does, and how it works.

---

## What This Is

NativeLab is a **local-first LLM workbench** - a complete environment for running, testing, building with, and serving large language models on your own hardware. No cloud. No account. No data leaves your machine.

It has two clients:
- **NativeLab Desktop** - Python/PyQt6 app for Linux, macOS, Windows
- **PhonoLab Android** - Kotlin/Android app for phones and tablets

Both are free and open source under AGPL v3.

---

## The Desktop App

### Chat

The core experience. Load any model - local GGUF, Ollama daemon, HuggingFace Transformers, or a remote API - and chat with it. The interface has:

- **Session sidebar** grouped by date with search, rename, delete, export
- **Model selector** showing family, quantization, and quality label
- **Markdown rendering** with per-language syntax highlighting and copy buttons
- **Math rendering** via KaTeX for LaTeX expressions
- **Code mode toggle** and pipeline mode badge
- **Collapsible messages** for long responses
- **File references** - attach PDFs, text files, or source code to a session. The app chunks them, scores relevance, and injects the top chunks as context before each prompt. References survive restarts.
- **Document summarization** - split long PDFs into chunks with carry-over context, summarize each section, then consolidate. Three modes: summary, logical, advice. Pause/resume between chunks. RAM watchdog checks every 5 chunks.
- **Multi-PDF synthesis** - run summarization across a batch of documents with cross-document theme analysis.

### Models

The Models tab manages all your models in one place:

- **Local GGUF** - scan a directory, register models, set per-model parameters (context, threads, temperature, top-p, top-k, repeat penalty, max tokens, seed, role)
- **Ollama** - connect to a running Ollama daemon, list installed models, pull new ones
- **HuggingFace Transformers** - load safetensors models directly with configurable dtype, device map, quantization
- **API models** - configure OpenAI, Anthropic, Groq, Mistral, Together AI, OpenRouter, or any OpenAI-compatible endpoint
- **Model roles** - assign models as general, reasoning, summarization, coding, or secondary. Pipeline mode uses the right model for each role.
- **Auto family detection** - 22 model families recognized from filename, correct prompt template applied automatically

### Downloads

One-stop shop for getting models and runtimes:

- **GGUF from HuggingFace** - enter any repo ID, search for GGUF files, stream download with resume support. 29 curated presets for quick start.
- **HuggingFace Transformers snapshots** - full repo download with in-app dependency installer. 27 curated presets.
- **Ollama pull** - connect to running daemon, stream model downloads. 28 curated presets.
- **llama.cpp runtime** - download prebuilt binaries for your platform
- **HuggingFace OAuth** - one-click browser login for gated/private repos

### Visual Pipeline Builder

Build multi-step LLM workflows as a visual graph:

- **22 block types**:
  - **I/O**: Input, Output, Model, Intermediate
  - **Context**: Reference (static text), Knowledge (reusable), PDF Summary
  - **Deterministic logic**: IF/ELSE, SWITCH, FILTER, TRANSFORM, MERGE, SPLIT, Custom Code
  - **LLM logic**: LLM IF, LLM SWITCH, LLM FILTER, LLM TRANSFORM, LLM SCORE
  - **Integration**: MCP Server, Web Search
- **AI Builder** - describe what you want in plain English, the loaded model generates validated pipeline JSON
- **Loop edges** with iteration counts (2-999)
- **8 shipped example presets** - quick-answer, clean-summarize, draft-review, triage-router, and more
- **Save/load** as JSON files
- **Device-aware assignment** - AI Builder assigns model blocks to PhonoLab devices based on task type (vision → vision devices, reasoning → high-RAM devices)
- **Device parameter sliders** - right-click any PhonoLab device model block to adjust temperature, top-k, top-p, repeat penalty via GUI

### Labs

Experimentation layer with two shipped features:

- **py-to-doc** - generate documentation from Python files. Single file, queue, or full project mode. Checkpoint resume. Context policy control.
- **Code Edit** - structured code editing with diff preview, working file persistence, save/save-as

New labs can be dropped in as plugins: a QWidget with `LAB_NAME`, `LAB_ICON`, and `set_endpoints()`.

### Dev Tab

Hidden until Developer Mode is enabled. Contains:

- **Devices** - scan LAN for PhonoLab Android devices, register as API models, manage connections
- **Logs** - full operation logs
- **API Server** - host your model as an OpenAI/Anthropic-compatible API server
- **Integrations** - Discord bot, WhatsApp bot, local HTTP bridge
- **Pipeline** - visual pipeline builder
- **MCP** - Model Context Protocol server management
- **Skills** - skill library for injecting context into model calls

### Terminal CLI

`nativelab --cli` opens a full terminal control center:

- **Chat REPL** with 20+ slash commands (`/load`, `/models`, `/pipelines`, `/skills`, `/status`, etc.)
- **@file embedding** - type `@file.py` in your prompt to embed file content (60KB cap)
- **Model management** - list, load, switch between GGUF/Ollama/HF/API models
- **Pipeline execution** - run saved pipelines from terminal
- **Labs** - run code-edit and py-to-doc from terminal
- **Integration serving** - expose endpoints over HTTP
- **Linting** - Python lint with pyflakes/flake8/pylint

### API Server

Host any model, pipeline, or the active engine as an OpenAI/Anthropic-compatible API:

- Dual API keys: separate keys for localhost vs LAN
- Auto-loads requested models
- SSE streaming support
- Pipeline execution via `pipeline:<name>` model IDs
- 15 integration endpoints for external tools

### Standalone Server

`python -m nativelab.server` turns any GGUF model into a standalone API server:

- Interactive model picker or direct path
- Hardware-aware auto-configuration (threads, context, max tokens)
- OpenAI + Anthropic compatible
- API key authentication
- Graceful shutdown

---

## The Android App (PhonoLab)

### On-Device Inference

Bundles `llama-server` as a native binary. Uses JNI fork+execve to start it - no W^X issues on Android. The server runs as a persistent process, model stays loaded in RAM.

### Chat UI

ChatGPT-style interface with:

- Sidebar with session history, auto-titles, date grouping, search
- Math rendering via KaTeX WebView
- Model picker with built-in catalog (5 small models optimized for mobile)
- Download progress tracking
- Generating banner with stop button

### Document Attachments

Attach PDF, text, or DOCX files:

- Text extraction from all three formats
- Sliding window chunking (1500 chars, 200 overlap)
- Keyword-based chunk retrieval
- Progress bar during processing
- Context injected into message on send

### Image Attachments

- Gallery picker (PhotoPicker on Android 13+, fallback to file picker)
- Vision model auto-detection
- mmproj file pairing for multimodal models
- Image sent as base64 in OpenAI vision format

### LAN API Server

Expose an OpenAI + Anthropic compatible API on your local network:

- **14 endpoints** including `/health`, `/status`, `/device`, `/v1/chat/completions`, `/config`, `/reload`
- **Live status tracking**: idle/loading/ready/generating/reloading/error
- **Request queuing** during model reload (up to 50 requests, 2-minute timeout)
- **Smart reload** - queue requests during model switch, drain on ready
- **Vision support** - image_url content in messages
- **Document context** - inject RAG chunks into prompts
- **Parameter editing** - update temperature, top-k, top-p, repeat penalty via API
- **Device reporting** - CPU cores, RAM, storage, Android version via `/device` endpoint
- **Structured errors** - model_not_loaded, server_busy, gateway_timeout - never blank responses

### Error Handling

17-layer error system:

- JNI guards with `nativeLoaded` flag
- Lifecycle-safe UI updates (`runOnUi` with `isAdded` check)
- Thread-safe date formatting (`ThreadLocal<SimpleDateFormat>`)
- JSON parsing safety (`optJSONObject` instead of `getJSONObject`)
- Session log capping (500 entries max)
- Fatal errors → restart dialog
- Non-fatal errors → red banner notification (auto-dismiss 5s)

---

## LAN Device Discovery

The desktop app can discover PhonoLab Android devices on the same network:

- **Multithreaded subnet scan** (64 threads) on port 8787
- **Smart auth flow** - tries without key first, prompts only when auth required or key changed
- **Device registration** as API models in the existing registry
- **Live status refresh** every 30 seconds
- **Remote control** - load models, update parameters, monitor queue status
- **AI Builder integration** - assigns tasks to devices based on specs

---

## Centralized Backend

All operations go through two core modules:

- **`core/http_client.py`** - unified HTTP client with consistent timeouts, retries, authentication. Supports GET, POST, SSE streaming, OpenAI and Anthropic formats.
- **`core/backend.py`** - facade for all operations: model loading, generation, device management, HuggingFace. Returns `BackendResult` with structured error handling.

No raw `urllib.request` or `http.client` calls in new code.

---

## Security

- **API keys** - dual keys for localhost vs LAN, timing-safe comparison
- **Body size limits** - 1MB max on API server
- **Thread pool caps** - fixed 4 threads, not unbounded
- **Socket write guards** - handle client disconnects gracefully
- **Custom code sandbox** - pipeline code blocks run with no imports, no filesystem, no network
- **HuggingFace OAuth** - tokens stored locally, masked in UI/logs
- **SAF file access** - Android uses Storage Access Framework, no broad permissions needed

---

## Configuration

Everything is configurable through JSON files:

| File | What |
|------|------|
| `app_config.json` | 40+ settings: RAM watchdog, chunk sizes, context defaults, theme, HF options |
| `server_config.json` | Binary paths, host/port, GPU offload (ngl, main_gpu, tensor_split) |
| `api_server_config.json` | Port, protocol, API keys, auto-load, CORS |
| `model_configs.json` | Per-model parameters (context, threads, temperature, etc.) |
| `cli_prefs.json` | CLI last-used model and context |
| `skill/skills.json` | Skill library |
| `mcp_config.json` | MCP server definitions |
| `integrations/discord_bots.json` | Discord connector profiles |
| `integrations/whatsapp_bots.json` | WhatsApp connector profiles |

---

## Hardware Support

- **CPU**: Automatic thread detection, configurable per-model
- **RAM**: psutil-based monitoring, watchdog with cache spilling
- **GPU**: NVIDIA (nvidia-smi), AMD (rocm-smi), Apple Metal (system_profiler), Vulkan (vulkaninfo)
- **Android**: JNI fork+execve, bundled binaries, arm64-v8a + armeabi-v7a
- **Auto-setup**: Hardware profiling → runtime install → model download → registration

---

## File Formats

**Reads**: GGUF, PDF, DOCX, and 21 source code languages (Python AST, JavaScript/JSX, TypeScript/TSX, SQL, Rust, Go, C, C++, Java, Kotlin, Ruby, Bash, YAML, JSON, TOML, Lua, Swift, C#, PHP, R, Julia, Markdown)

**Writes**: 25+ distinct file types including session JSON, model configs, pipeline definitions, API configs, OAuth credentials, device caches, skill libraries, integration profiles, reference indexes, and summarization checkpoints.

---

## License

AGPL v3 - free and open source forever. Both NativeLab and PhonoLab.

Dependencies: llama.cpp (MIT), PyQt6 (GPL/commercial), AndroidX (Apache 2.0), Material 3 (Apache 2.0).
