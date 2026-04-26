# NativeLab v0.1.6

**A fully local, privacy-first LLM desktop application powered by llama.cpp**

---

## Top section for pip users :D

---

## Installation

Install NativeLab from PyPI:

```bash
pip install nativelab
```

---

## First-Time Setup

After installation, create a dedicated workspace folder for NativeLab. This folder will store:

* Models
* Config files
* Runtime data

Example:

```bash
mkdir ~/nativelab_workspace
cd ~/nativelab_workspace
```

---

## Folder Structure

Inside your workspace, create the following structure:

```text
nativelab_workspace/
├── llama/
│   └── bin/
├── models/
├── configs/
```

---

## Adding LLaMA Binaries

Download or build `llama.cpp` binaries.

Then place them inside:

```text
nativelab_workspace/llama/bin/
```

Example:

```text
llama/bin/
├── llama-cli
├── llama-server
```

> Ensure binaries are executable:

```bash
chmod +x llama/bin/*
```

---

## Running NativeLab

From anywhere in your system:

```bash
nativelab
```

NativeLab will:

* Detect your workspace
* Load binaries from `llama/bin/`
* Initialize models and UI

---

## Recommended Workflow

1. Install NativeLab
2. Create workspace folder
3. Add llama binaries
4. Launch app
5. Load or download models

---

## Troubleshooting

### PyQt6 Missing

If you see:

```text
ModuleNotFoundError: PyQt6
```

Reinstall:

```bash
pip install nativelab
```

---

### LLaMA Binary Not Found

Ensure:

```text
llama/bin/llama-cli
```

exists and is executable.

---
## NEW : ADDED LLAMA DOWNLOADER IN DOWNLOADS TAB SO YOU DONY NEED TO WORRY :D
---

### Permission Issues

Run:

```bash
chmod -R +x llama/bin/
```

---

## Notes

* NativeLab is fully local — no data leaves your machine
* Works offline once models are available
* Designed for modular AI pipelines and experimentation

---

1. [Overview](#overview)
2. [What's New in v2](#whats-new-in-v2)
3. [Architecture Overview](#architecture-overview)
4. [Requirements & Installation](#requirements--installation)
5. [Configuration](#configuration)
6. [Model Management](#model-management)
7. [API Model Support](#api-model-support)
8. [Prompt Template System](#prompt-template-system)
9. [Quantization Format Support](#quantization-format-support)
10. [Chat & Sessions](#chat--sessions)
11. [Reference Engine](#reference-engine)
12. [Script Parser](#script-parser)
13. [Summarization Pipeline](#summarization-pipeline)
14. [Multi-PDF Summarization](#multi-pdf-summarization)
15. [Parallel Loading & Pipeline Mode](#parallel-loading--pipeline-mode)
16. [Visual Pipeline Builder](#visual-pipeline-builder)
17. [MCP — Model Context Protocol](#mcp--model-context-protocol)
18. [HuggingFace Model Downloader](#huggingface-model-downloader)
19. [Server Configuration](#server-configuration)
20. [UI Components](#ui-components)
21. [Theming & Appearance](#theming--appearance)
22. [Data Persistence](#data-persistence)
23. [RAM Watchdog & Memory Management](#ram-watchdog--memory-management)
24. [Keyboard Shortcuts & Menus](#keyboard-shortcuts--menus)
25. [Developer Notes](#developer-notes)

---

## Overview

Native Lab Pro is a desktop chat application for running large language models entirely on your local machine — no API keys, no cloud, no data leaving your system. It wraps [llama.cpp](https://github.com/ggerganov/llama.cpp) (either `llama-server` or `llama-cli`) behind a polished PyQt6 GUI, and extends the basic chat experience with a rich feature set including multi-model pipelines, document references, long-document summarization, and a structured code-aware reference system.

The application is designed for power users who want fine-grained control over their models — setting per-model context sizes, generation parameters, prompt templates, and CPU thread counts — while also providing a clean, dark-themed interface that stays out of the way.

---

## What's New in v2

Version 2 is a significant expansion of the original v1 feature set.

**Full GGUF quantization format detection** means the app now recognizes and labels every quantization type supported by modern llama.cpp builds, from the legacy `Q4_0` and `Q8_0` formats all the way through the modern K-quant series (`Q2_K` through `Q6_K`) and the imatrix importance quants (`IQ1_S`, `IQ2_XXS`, `IQ4_XS`, etc.). Each model in the library is shown with its quant type and a human-readable quality label (e.g. "High quality", "Balanced", "Very compressed").

**Auto model-family detection** reads the model filename and automatically selects the correct prompt template. This covers more than 20 distinct model families including DeepSeek (and its R1 variant), Mistral, Mixtral, LLaMA-2, LLaMA-3, Phi-3, Qwen/ChatML, Gemma, CodeLlama, Falcon, Vicuna, OpenChat, Neural-Chat, Starling, Yi, Zephyr, Solar, Orca, and Command-R.

**Parallel model loading** allows you to run multiple engines simultaneously (e.g. a reasoning model and a coding model at the same time). This comes with clear RAM usage warnings because each loaded model occupies its full RAM allocation independently.

**Pipeline mode** chains models together: when you submit a coding prompt, non-coding engines first produce structured architectural insights (no code, just design analysis), and then the designated coding engine receives all of those insights as rich context before generating the final code.

**Python/code snippet copy buttons** are embedded directly inside chat bubbles for each fenced code block.

**API model support** lets you connect to any OpenAI-compatible endpoint or the Anthropic API as a drop-in replacement for a local engine. This includes OpenAI, Anthropic Claude, and any self-hosted OpenAI-compatible server. API models participate fully in pipeline, summarization, and reference injection workflows.

**Visual Pipeline Builder** is a full drag-and-drop pipeline editor with an interactive canvas. You can place LLM blocks, transform blocks, and branch blocks, connect them with directed edges, and execute the pipeline from within the UI. Each intermediate block streams its output live into a dedicated output pane.

**MCP (Model Context Protocol) integration** lets you configure and launch stdio or SSE MCP servers that connect the LLM to external tools — filesystem access, web search, databases, APIs, and more. Servers are managed from a dedicated tab and launched as child processes.

**HuggingFace GGUF Downloader** lets you search any HuggingFace repository by ID, browse available GGUF files with size and metadata, and download them directly into your models folder — all without leaving the app.

**Server configuration tab** exposes all llama.cpp binary paths, server host/port range, GPU offload settings (NGL layer count, primary GPU index, tensor split for multi-GPU), and extra CLI/server arguments. Configuration is persisted to `server_config.json`.

**GPU offload support** — the server configuration now includes toggles for enabling GPU inference, setting the number of layers to offload (`ngl`), selecting the primary GPU device, and configuring tensor split ratios across multiple GPUs.

**Light and dark theme toggle** — the app ships with both a light and a dark color palette, switchable at runtime from the View menu. All tabs, chat bubbles, and UI chrome rebuild instantly on theme switch, and the active palette is persisted across restarts.

**Customizable color palette (Appearance tab)** — every color token in the active theme can be overridden through a point-and-click color picker in the new Appearance tab. Custom palettes are saved separately for light and dark modes.

**Windows and macOS support** — binary paths are now resolved at runtime based on the detected OS. On Windows, the app looks for `.exe` suffixed binaries. Frozen (PyInstaller) builds are also supported via `sys._MEIPASS`.

All v1 features — session management, PDF summarization, reference injection, chunked long-document processing, pause/resume of summarization jobs — are fully preserved.

---

## Architecture Overview

The application is organized into several distinct layers that work together.

```
┌─────────────────────────────────────────────────────────────────────-┐
│                          MainWindow (PyQt6)                          │
│  ┌──────────────┐   ┌──────────────────────────────────────────────┐ │
│  │SessionSidebar│   │                 QTabWidget                   │ │
│  │              │   │  ┌──────┬───────┬───────┬───────┬─────────┐  │ │
│  │ (chat list)  │   │  │ Chat │Models │Config │Server │Pipeline │  │ │
│  └──────────────┘   │  │      │       │       │       │ Builder │  │ │
│                     │  ├──────┼───────┼───────┼───────┼─────────┤  │ │
│                     │  │ MCP  │  DL   │Appear.│ Logs  │         │  │ │
│                     │  └──────┴───────┴───────┴───────┴─────────┘  │ │
│                     └──────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────-┘
         │
         ▼
┌─────────────────────────────────┐
│          Engine Layer           │
│  LlamaEngine  (manages process) │
│  ├── ServerStreamWorker (HTTP)  │
│  └── CliStreamWorker (stdio)    │
│                                 │
│  ApiEngine (cloud / local API)  │
│  └── ApiStreamWorker (HTTP)     │
│                                 │
│  PipelineWorker (multi-engine)  │
│  PipelineExecutionWorker        │
│  ChunkedSummaryWorker           │
│  MultiPdfSummaryWorker          │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│       Reference Engine          │
│  SessionReferenceStore          │
│  ├── SmartReference             │
│  └── ScriptSmartReference       │
│       └── ScriptParser          │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│       Data / Persistence        │
│  Session (JSON in sessions/)    │
│  ModelRegistry (JSON)           │
│  ApiRegistry (JSON)             │
│  ServerConfig (JSON)            │
│  APP_CONFIG (app_config.json)   │
│  ParallelPrefs (JSON)           │
│  PausedJobs (paused_jobs/)      │
│  McpConfig (JSON)               │
└─────────────────────────────────┘
```

The `LlamaEngine` class is the heart of the local inference layer. It tries to start `llama-server` first (preferred, because it supports true HTTP streaming with a persistent model in memory). If the server binary is not found or fails to start, it falls back to calling `llama-cli` as a subprocess for each prompt. This dual-mode design means the app works even in minimal llama.cpp builds.

`ApiEngine` is a drop-in replacement for `LlamaEngine` that routes inference to a remote or locally-hosted API endpoint. It supports the OpenAI chat completions format and the Anthropic messages format, and participates in all the same pipeline, summarization, and reference workflows as a local engine.

---

## Requirements & Installation

### System Requirements

Native Lab Pro runs on Linux, macOS, and Windows. On Windows, binary paths are resolved with a `.exe` suffix automatically. Frozen builds (packaged with PyInstaller) are also supported — the app detects `sys._MEIPASS` and resolves binaries relative to the bundle directory.

The minimum hardware depends entirely on the models you intend to run. As a rough guide, a 7B parameter Q4 model requires approximately 4–5 GB of RAM, a 13B Q5 model requires roughly 9–10 GB, and a 70B Q4 model requires 38–40 GB. GPU offload (via the Server tab) can significantly reduce RAM requirements by moving layers to VRAM.

### Python Dependencies

The core application requires Python 3.10+ and PyQt6. Install with:

```bash
pip install PyQt6
```

The following packages are optional but enable additional features:

```bash
pip install psutil    # enables live RAM monitoring in the status bar
pip install PyPDF2    # enables PDF loading and summarization
```

### llama.cpp Binaries

You must compile or download llama.cpp and point the app to the binaries. Binary paths are now configured through the **Server tab** in the UI and persisted to `server_config.json`. The default resolution order is:

1. Paths stored in `server_config.json` (set via the Server tab)
2. Bundled binaries at `llama-bin/llama-cli` (for frozen builds)
3. Dev fallback at `./llama/bin/llama-cli`

On Windows the app automatically appends `.exe` to all binary lookups. You do not need to edit the source to configure binary paths — use the Server tab instead.

### Models Directory

By default the app scans for `.gguf` files in:

```python
MODELS_DIR = Path("./localllm")
```

Use the **Browse GGUF…** button in the Models tab to add individual model files from anywhere on your filesystem, or use the new **Download tab** to fetch GGUF files directly from HuggingFace.

### Running the App

```bash
python native_lab_pro_v2.py
```

---

## Configuration

All runtime thresholds are stored in `app_config.json` and are fully editable through the **Config tab** in the UI. Changes take effect immediately and are persisted to disk.

| Setting | Default | Description |
|---|---|---|
| `ram_watchdog_mb` | 800 | Free RAM threshold (MB) that triggers a disk spill of reference caches |
| `chunk_index_size` | 400 | Character size of each indexed reference chunk |
| `max_ram_chunks` | 120 | Max chunks per reference file kept in RAM |
| `summary_chunk_chars` | 3000 | Characters per chunk during PDF summarization |
| `summary_ctx_carry` | 600 | Characters from the previous chunk's summary carried forward as context |
| `summary_n_pred_sect` | 380 | Max tokens generated per section summary |
| `summary_n_pred_final` | 700 | Max tokens for the final consolidation pass |
| `multipdf_n_pred_sect` | 380 | Tokens per section during multi-PDF jobs |
| `multipdf_n_pred_final` | 900 | Tokens for the cross-document final pass |
| `ref_top_k` | 6 | Number of top-scoring chunks retrieved per reference per query |
| `ref_max_context_chars` | 3000 | Max total reference text injected into each prompt |
| `pause_after_chunks` | 2 | Number of chunks processed before auto-pause is suggested |
| `default_threads` | 12 | Default CPU thread count passed to llama.cpp |
| `default_ctx` | 4096 | Default context window in tokens |
| `default_n_predict` | 512 | Default max new tokens per generation |
| `auto_spill_on_start` | false | Immediately spill all reference caches to disk on startup |

All of these can also be overridden on a **per-model basis** through the model parameter editor in the Models tab.

---

## Model Management

The `ModelRegistry` class maintains two sources of model metadata: models auto-discovered from `MODELS_DIR`, and models manually added via the file browser. Both are stored in `custom_models.json` and `model_configs.json`.

### Roles

Each model can be assigned one of five roles. The role determines which engine slot the model is loaded into, and affects routing logic:

- **General** — the primary chat engine, used for normal conversation
- **Reasoning** — used as a structural insight provider in pipeline mode; also used as the final consolidation engine for summarization if no dedicated summarization model is loaded
- **Summarization** — dedicated engine for document summarization tasks
- **Coding** — receives all prompts that are detected as coding-related (and receives the pipeline-stage context in pipeline mode)
- **Secondary** — an additional auxiliary engine that participates as an insight provider in multi-engine pipeline runs

### Per-Model Parameters

Each model stores individual configuration:

```
threads        — CPU threads for this specific model
ctx            — context window size in tokens
temperature    — sampling temperature (0.0–2.0)
top_p          — nucleus sampling threshold
repeat_penalty — penalty for repeating tokens
n_predict      — maximum tokens to generate
family         — auto-detected model family key
```

These parameters are saved to `model_configs.json` and loaded automatically when the model is selected.

---

## Prompt Template System

One of the most important features for getting good results from local models is using the correct prompt format. Different model families were fine-tuned with different chat templates, and using the wrong one causes the model to ignore instructions or output garbage.

Native Lab Pro auto-detects the model family from the filename and applies the correct template automatically. The detection is done by `detect_model_family()`, which matches filename substrings in priority order (more specific patterns checked first to avoid ambiguity).

### Supported Families and Their Templates

Each `ModelFamily` dataclass stores the exact prefix/suffix strings needed to construct a valid prompt:

**DeepSeek / DeepSeek-R1** use `User: ... \n\nAssistant:` with special BOS/EOS tokens `<｜begin▁of▁sentence｜>` and `<｜end▁of▁sentence｜>`. The R1 variant adds a `<think>` block after the assistant prefix to trigger chain-of-thought reasoning.

**Mistral / Mixtral** use the `[INST] ... [/INST]` format with `<s>` and `</s>` as BOS/EOS tokens.

**LLaMA-2** uses `[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]` with system prompt injection support.

**LLaMA-3** uses the header-based format: `<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>` with `<|begin_of_text|>` as BOS.

**Phi / Phi-3** use `<|user|>\n{user}<|end|>\n<|assistant|>\n`.

**Qwen / Yi / Orca** use the ChatML format: `<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`.

**Gemma** uses `<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n`.

**Command-R** uses Cohere's special token format with `<|START_OF_TURN_TOKEN|>` markers.

The `Session.build_prompt()` method uses the detected family to render the full multi-turn conversation history into a correctly formatted prompt string, respecting the context window limit by truncating older messages first.

---

## Quantization Format Support

The `detect_quant_type()` function recognizes all quantization formats supported by current llama.cpp builds:

**imatrix importance quants** — `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ3_XXS`, `IQ3_XS`, `IQ3_S`, `IQ3_M`, `IQ4_XS`, `IQ4_NL`

**K-quants** — `Q2_K`, `Q3_K_S/M/L`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K`

**Legacy quants** — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`

**Float** — `F16`, `F32`, `BF16`

Each quant type is also mapped to a quality tier with a color-coded label:

| Tier | Types | Label |
|---|---|---|
| 🟢 Full/Near-lossless | F32, F16, BF16, Q8, Q6 | "Full precision" / "Near-lossless" |
| 🟣 High quality | Q5, IQ4 | "High quality" |
| 🟡 Balanced | Q4, IQ3 | "Balanced" |
| 🔴 Compressed | Q3, IQ2 | "Compressed" |
| 🔴 Very compressed | Q2, IQ1 | "Very compressed" |

---

## Chat & Sessions

### Session Model

Each conversation is a `Session` object containing a list of `Message` objects (role, content, timestamp). Sessions are saved as JSON files in the `sessions/` directory and loaded on startup.

The `Session.approx_tokens` property estimates token usage by dividing total character count by 4, which is used to drive the context usage progress bar in the status bar.

### Session Sidebar

The sidebar groups sessions by date and supports search. Right-clicking a session offers rename, export to Markdown, and delete options.

### Coding Prompt Detection

When a message is sent, the app checks whether it looks like a coding request by scanning for keywords like `def `, `class `, `` ``` ``, `implement `, `generate code`, etc. If a coding engine is loaded, coding prompts are automatically routed to it instead of the general engine. The user can also force coding-engine routing with the **💻 Code** toggle button.

---

## Reference Engine

The reference system lets you attach documents to a chat session and have relevant excerpts automatically injected into every prompt. This is useful for asking questions about specific documents without copying and pasting text.

### SmartReference

When a file is added (PDF, .py, or text), it is processed into a `SmartReference` object. The raw text is split into overlapping chunks of `chunk_index_size` characters. Chunks are initially kept in RAM up to `max_ram_chunks`; if RAM pressure is detected, chunks beyond that limit are serialized to disk using `pickle` and loaded on demand.

Retrieval works by keyword scoring: for a given query, each chunk is scored by how many query words it contains, with a bonus for exact phrase matches. The top-K chunks are concatenated and injected into the prompt as a `[REFERENCE: ...]` block.

### ScriptSmartReference

Source code files get a more powerful treatment through the `ScriptSmartReference` class and its companion `ScriptParser`. Instead of just chunking by character count, the parser extracts structured metadata — imports, classes, functions/methods, constants, type definitions, and their signatures, docstrings, and decorators.

The structured index allows much more precise retrieval: when you ask a coding question, the context injected from a script reference will include the function signatures, docstrings, and class hierarchies most relevant to your query rather than arbitrary text chunks.

### SessionReferenceStore

Each session has its own isolated `SessionReferenceStore` that manages all attached references. The store persists metadata (file names, types, reference IDs) to `ref_index/` and raw text to `ref_cache/`. This means references survive application restarts.

### ReferencePanelV2

The reference UI is split into two tabs. The **📄 Docs** tab handles PDFs, plain text files, and plain `.py` files treated as text. The **💻 Scripts** tab handles source code files with full AST parsing, and shows a detail pane with the extracted function and class names when a script is selected.

---

## Script Parser

The `ScriptParser` class provides structured parsing for a wide range of source languages. It uses Python's `ast` module for `.py` files (giving accurate, AST-level extraction of all functions, classes, imports, constants, async functions, and decorators), and regex-based parsers for all other languages.

### Supported Languages

Python, JavaScript/JSX, TypeScript/TSX, SQL, Rust, Go, C, C++, Java, Kotlin, Ruby, Bash, YAML, JSON, TOML, Lua, Swift, C#, PHP, R, Julia, and Markdown.

### What Gets Extracted

For each language, the parser extracts as many of the following as are applicable:

- **Imports** — all import/require/use/include statements with their resolved names
- **Functions** — with full signatures, docstrings, decorators, line numbers, and body text
- **Classes/Structs** — with base classes, methods, docstrings, and line ranges
- **Constants** — uppercase-named variables and `#define` macros
- **Types** — TypeScript interface/type aliases, Rust trait/impl blocks, Go interfaces
- **SQL objects** — CREATE TABLE, CREATE VIEW, procedures, indexes, CTEs
- **Config keys** — top-level keys for JSON, YAML, and TOML files

### Context Block Generation

The `ParsedScript.build_context()` method assembles a structured context block for LLM injection. It always includes imports and constants (compact), then ranks all functions and classes by keyword relevance to the current query and includes as many as fit within the character budget. The output has a structured header showing the file's language, name, and a summary of what was found.

---

## Summarization Pipeline

For documents that are too long to fit in the context window, Native Lab Pro uses a chunked summarization strategy.

### ChunkedSummaryWorker

The worker splits the full document text into chunks of `summary_chunk_chars` characters, trying to break on paragraph boundaries (`\n\n`) for clean splits. It then processes each chunk sequentially:

1. It prepends a short "running context" (the last `summary_ctx_carry` characters of the previous chunk's summary) to maintain narrative continuity.
2. It sends the chunk plus context to the model and collects a section summary.
3. Every 3 chunks, it autosaves the full state to disk in `paused_jobs/`.
4. After all chunks are processed, a final consolidation pass synthesizes all section summaries into one cohesive document summary.

If a dedicated summarization or reasoning engine is loaded, it is used for the final consolidation pass (which often benefits from a more capable model).

### Pause / Resume

At any point during summarization, the user can click **⏹ Stop** to pause the job. The full state — raw text, completed summaries, running context, progress counter, and model path — is written to `paused_jobs/{job_id}.json`. Paused jobs appear in the **Config** tab and can be resumed at any time by selecting them and clicking **▶ Resume Job**.

The app also automatically suggests pausing after `pause_after_chunks` chunks have been processed if many chunks remain, making it easy to break long jobs into sessions.

---

## Multi-PDF Summarization

`MultiPdfSummaryWorker` extends the single-PDF pipeline to handle batches of documents. It processes each PDF file through the full chunked summarization pipeline, produces a per-file consolidated summary, then performs a final cross-document synthesis pass that identifies themes, differences, and connections across all documents.

The multi-PDF worker integrates with the RAM watchdog: it checks free RAM every 5 chunks and spills reference caches to disk if necessary. Pause/resume works exactly the same as single-PDF mode, with the state including the full list of PDF texts and the current file and chunk position.

---

## Parallel Loading & Pipeline Mode

### Parallel Loading

The `ParallelPrefs` dataclass stores whether parallel loading is enabled, which roles to auto-load on startup, and whether pipeline mode is active. Settings are saved to `parallel_prefs.json`.

When parallel loading is enabled, you can load separate model instances into the reasoning, summarization, coding, and secondary engine slots simultaneously. Each engine runs its own `llama-server` process on a different port (or uses `llama-cli` as a fallback). The app warns clearly about the RAM implications before enabling this feature.

### Pipeline Mode

Pipeline mode activates automatically when all of the following are true:

- Parallel loading is enabled
- Pipeline mode is checked in settings
- At least one non-coding engine is loaded (reasoning, summarization, secondary, or primary)
- The coding engine is loaded
- The incoming prompt is detected as a coding request

When pipeline mode is active, the flow for a coding prompt is:

**Stage 1 — Structural Insights.** Each non-coding engine is asked to produce a structured architectural analysis of the coding request: high-level purpose, recommended architecture pattern, key components, data flow, important algorithms, edge cases, and suggested libraries. No code is generated in this stage. Each engine's output appears in its own chat bubble in real time.

**Stage 2 — Code Generation.** The coding engine receives a prompt that includes all of the stage-1 insights as a labelled context block, followed by the original request. It then generates the complete implementation using those insights as its blueprint.

This two-stage approach typically produces better-structured, more complete code than asking a coding model directly, because the architectural planning is done by models that may be better at reasoning even if they are less specialized for code generation.

---

## Visual Pipeline Builder

The Pipeline Builder is the most powerful feature in Native Lab Pro. It is a full node-based pipeline editor with an interactive drag-and-drop canvas, 20 distinct block types across four categories, loop support, LLM-powered routing logic, a live execution log, and per-block streaming output — all running entirely on your local machine.

Pipelines are saved as JSON in `~/.native_lab/pipelines/` and can be loaded, overwritten, and deleted from within the UI.

---

### The Canvas

The canvas is a 1400×900+ pixel infinite workspace with a 20px snap-to-grid. Every block snaps cleanly to the grid when placed or dragged.

**Placing blocks** — click any block type button in the left sidebar, or drag a model directly from the model list onto the canvas. The model's path and role are decoded from the drag event and applied to the new block automatically.

**Selecting and moving** — click a block to select it (highlighted border). Drag it anywhere. Blocks snap to the nearest grid point on release.

**Drawing connections** — hover over a block until a port circle appears (N/S/E/W edges), then click-drag from that port to any port on another block. A dashed preview arrow follows your cursor. Release on a target port to create the connection.

**Deleting** — right-click a block or connection for a context menu with delete, duplicate, and configure options.

**Connection rules enforced by the canvas:**
- Direct model-to-model connections are blocked. You must place an Intermediate block between two Model blocks to capture the output of the first before it enters the second.
- Duplicate connections (same source port, same target) are silently prevented.
- A single outgoing port on a non-logic block can only have one connection; drawing a new one replaces the old.
- Logic blocks can fan-out to multiple targets from the same port.

---

### Block Types

There are four categories of blocks, color-coded on the canvas.

#### I/O and Model Blocks

| Block | Color | Description |
|---|---|---|
| **▶ Input** | Green | Source block. Captures the user's text prompt at run time before execution starts. Every pipeline must have at least one. |
| **■ Output** | Red | Sink block. Receives the final text and renders it in the Output tab with full Markdown and code highlighting. Every pipeline must have at least one. |
| **⚡ Model** | Purple | Runs an LLM inference step. Shows a badge with the detected family and quant type. Can be dragged directly from the model list in the sidebar. |
| **◈ Intermediate** | Yellow | Captures and displays the output from a model or logic block mid-pipeline. Creates a live streaming tab in the output panel. Required between chained model blocks. |

#### Context Injection Blocks

| Block | Color | Description |
|---|---|---|
| **📎 Reference** | Purple | Injects a static text snippet (typed or loaded from a file) as a `[REFERENCE: name] … [/REFERENCE]` block prepended to the context before it reaches the next block. |
| **💡 Knowledge** | Violet | Prepends a knowledge-base chunk to the context under a `Knowledge Base:` header. Similar to Reference but styled for factual knowledge rather than document references. |
| **📄 PDF** | Cyan | Loads a PDF file, extracts its text, and if it exceeds 4,500 characters, automatically chunk-summarises it using the primary engine before injecting it. Supports two roles: `reference` (PDF injected below the existing context) or `main` (PDF replaces the existing context, prior text becomes a reference). |

#### Deterministic Logic Blocks

These blocks operate purely on text without any model calls.

| Block | Color | Operation |
|---|---|---|
| **⑂ IF/ELSE** | Amber | Evaluates a Python expression against the incoming text (`text` variable). TRUE → E port, FALSE → W port. |
| **⑃ SWITCH** | Orange | Evaluates a Python expression that returns a string key. Routes to the outgoing connection whose label matches the key. A `default` label catches unmatched keys. |
| **⊘ FILTER** | Lime | Evaluates a Python condition. If TRUE, text passes through. If FALSE, the pipeline terminates with a structured drop message. |
| **⟲ TRANSFORM** | Cyan | Applies a deterministic text transformation: prefix, suffix, find-and-replace, upper, lower, strip, or truncate to N characters. |
| **⊕ MERGE** | Violet | Collects all texts arriving at it from multiple upstream connections and combines them using one of four modes: concat (with separator), prepend, append, or json array. |
| **⑁ SPLIT** | Pink | Broadcasts the same incoming text to all outgoing connections simultaneously. No configuration needed — fan-out is automatic. |
| **⌥ Custom Code** | Green | Executes user-written Python inline at runtime. See [Custom Code Blocks](#custom-code-blocks) below. |

The Python sandbox for IF/ELSE, SWITCH, and FILTER is intentionally restricted — it exposes only safe builtins (`len`, `str`, `int`, `float`, `bool`, `list`, `dict`, `any`, `all`, `min`, `max`, `abs`, `isinstance`, `True`, `False`, `None`) and the `text` variable. No imports, no file access, no network.

#### LLM Logic Blocks

These blocks make a real inference call to a locally-running GGUF model to make routing or transformation decisions. Each has its own model selector, max-token count, temperature, and optional passthrough-on-error setting.

| Block | Color | What the LLM does |
|---|---|---|
| **🧠 LLM-IF** | Purple | Reads your condition in plain English, answers YES or NO. YES → TRUE (E port), NO → FALSE (W port). |
| **🧠 LLM-SWITCH** | Dark violet | Classifies the text into one of your named categories. Category names are read automatically from the labels on outgoing arrows. |
| **🧠 LLM-FILTER** | Indigo | Decides PASS or STOP based on a plain-English pass condition. STOP terminates the pipeline with a detailed explanation. |
| **🧠 LLM-TRANSFORM** | Sky blue | Rewrites or reformats the incoming text according to your instruction. The result replaces the context for all downstream blocks. |
| **🧠 LLM-SCORE** | Fuchsia | Scores the text from 1–10 on your criterion. Routes LOW (1–3) → E, MID (4–7) → S, HIGH (8–10) → W. Label an arrow `score` to receive the raw number instead of the text. |

---

### LLM Logic Block Editor

Double-clicking any LLM logic block opens the **LLM Logic Editor**, a full configuration dialog with:

- **Block description** — a card explaining what the block does, what input it expects, and what output it produces.
- **Branch routing hint** — a color-coded summary showing which port receives which outcome.
- **Model selector** — a combo box populated from `ModelRegistry`, plus a Browse button. A ✅/❌ badge shows whether the selected file exists on disk.
- **Instruction editor** — a multi-line text field for writing the condition or instruction in plain English. Placeholder examples are shown for each block type.
- **Advanced settings panel** (collapsible):
  - `Max response tokens` — how many tokens the model generates for its decision (8–512).
  - `Temperature` — sampling temperature (0–100, divided by 100 at runtime).
  - `Show model reasoning in log` — whether the raw model response appears in the execution log.
  - `Pass through unchanged if model call fails` — whether to silently continue on error instead of halting the pipeline.

---

### Custom Code Blocks

The `⌥ Custom Code` block runs user-written Python inside a sandboxed `exec()` call at pipeline execution time.

**Available variables:**

| Variable | Type | Description |
|---|---|---|
| `text` | `str` | The incoming context string from the previous block (read-only) |
| `result` | `str` | Set this to your output — the pipeline continues with this value |
| `metadata` | `dict` | The block's own metadata dictionary — persists across pipeline runs |
| `log(msg)` | `fn` | Writes a message to the pipeline execution log |

The code editor has live syntax checking — a ✅ / ❌ indicator updates on every keystroke using `compile()`. A **Test with sample text…** button lets you run the code against any text you type and see the `result` and log output before saving.

The sandbox exposes an expanded set of builtins compared to the simpler logic blocks, including `list`, `dict`, `tuple`, `range`, `enumerate`, `zip`, `map`, `filter`, `sorted`, `sum`, `round`, `print` (mapped to `log`), `isinstance`, `hasattr`, `getattr`, `repr`, and `type`.

---

### Loop Edges

Draw a connection **backwards** — from a downstream block back to an upstream block. The canvas detects the cycle and prompts for a loop count (2–999). Loop edges are rendered as dashed dash-dot lines with a `×N` badge at the midpoint.

At runtime, each loop edge is tracked independently by a visit counter. When the counter reaches the configured limit, the edge is skipped and execution flows to the next non-loop outgoing connection instead.

**Refinement loop pattern:**
```
▶ Input → ⚡ Draft Model → ◈ Intermediate ("Critique the draft and list improvements")
     ↑                                        |
     └────────────── ×3 loop ─────────────────┘
→ ■ Output
```

---

### Execution

Click **▶ Run Pipeline** to begin. The `PipelineExecutionWorker` QThread processes blocks in topological BFS order, carrying a `current_text` string from block to block.

- Each block emits a `step_started` signal (shows a spinner in the output panel) and a `step_done` signal (populates its intermediate tab).
- Model blocks start their own `llama-server` instance (or use `llama-cli` as fallback) and stream tokens live into the intermediate tab.
- The execution log shows every decision made: condition evaluated, branch taken, chars processed, FILTER pass/drop, MERGE inputs combined, SPLIT fan-out count, loop iteration.
- Click **⏹ Stop Execution** at any time. The worker finishes its current HTTP request and halts cleanly.

---

### Save & Load

- **💾 Save Pipeline…** — prompts for a name and writes to `~/.native_lab/pipelines/{name}.json`. Existing names are overwritten.
- **📂 Load Pipeline…** — lists saved pipelines with a delete option. If the canvas has blocks, a confirmation dialog asks before clearing.

What is persisted per block: type, canvas position, size, model path, role, label, and all metadata (system prompt, instruction text, Python code, PDF path, reference text, transform settings, etc.).

What is persisted per connection: source block ID, source port, target block ID, target port, `is_loop` flag, `loop_times`, and branch label.

---

### Performance Notes

- Use 0.5B–1.5B models for all LLM logic routing blocks (LLM-IF, LLM-SWITCH, LLM-FILTER, LLM-SCORE). They only need to output a word or a number — a small model is fast and accurate for this task.
- Reserve large models for actual generation steps (Model blocks, LLM-TRANSFORM).
- TRANSFORM blocks before expensive model calls keep context short and inference fast.
- Each loop iteration is a full model call. Three models × five loops = fifteen server requests — plan accordingly.
- Max response tokens for routing blocks can be as low as 16 (YES/NO decisions need very few tokens).
- PDF blocks auto-summarise large documents, which adds extra model calls. Pre-process large PDFs offline if speed is critical.

---

## MCP — Model Context Protocol

The MCP tab lets you configure and manage MCP servers that extend the LLM with access to external tools and data sources — filesystem operations, web search, database queries, REST APIs, and more.

### Server Types

**Stdio servers** are launched as child processes by the app. You supply the command and arguments, and the app manages the process lifecycle (start, stop, restart). Stdio servers communicate over standard input/output using the MCP protocol.

**SSE servers** connect over HTTP using Server-Sent Events. You supply the base URL of an already-running SSE server and the app connects to it. SSE servers are not managed as child processes.

### Configuration

Each server entry stores a name, transport type, command/URL, optional environment variable overrides, and a description. Configuration is persisted to `mcp_config.json`. The server list in the tab shows each server's name, type, and current status (running / stopped / error).

---

## HuggingFace Model Downloader

The Download tab provides a built-in interface for finding and downloading GGUF model files from HuggingFace, without needing to use a browser or the `huggingface-cli` tool.

### Searching

Enter any HuggingFace repository ID (e.g. `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`) in the search box and click **Search**. The app queries the HuggingFace API for all files in that repository and populates the results list with the GGUF files found, along with their file sizes and last-modified dates.

### Downloading

Select a file from the results list to see its full metadata, then click **Download**. The file is streamed directly into your configured models directory. A progress bar shows download progress. If a file already exists locally, the app will warn before overwriting.

The `HfSearchWorker` and `HfDownloadWorker` QThread subclasses handle the network operations off the main thread, so the UI stays responsive during large downloads.

---

## Server Configuration

The Server tab exposes all settings for the llama.cpp binaries and the HTTP server that the app manages.

### Binary Paths

You can set explicit paths for `llama-cli` and `llama-server`. If left blank, the app uses the built-in path resolution order (bundle directory → dev fallback). A **Test** button verifies that each binary exists and is executable.

### Network Settings

| Setting | Default | Description |
|---|---|---|
| Host | `127.0.0.1` | Address the llama-server listens on |
| Port range low | `8600` | Lowest port the app will try when starting a server |
| Port range high | `8700` | Highest port in the range |

The app scans the port range at startup and picks the first available port, so multiple instances can run simultaneously without manual port configuration.

### GPU Offload

| Setting | Default | Description |
|---|---|---|
| Enable GPU | off | Pass GPU flags to llama-server |
| NGL (n-gpu-layers) | -1 | Number of model layers to offload; -1 offloads all layers |
| Main GPU | 0 | Primary GPU device index (for multi-GPU systems) |
| Tensor split | (empty) | Comma-separated split ratios, e.g. `0.6,0.4` for two GPUs |

GPU settings are passed directly as command-line arguments when launching `llama-server`.

### Extra Arguments

Free-form extra arguments can be appended to the `llama-cli` and `llama-server` invocations for any flags not exposed by the UI (e.g. `--mlock`, `--no-mmap`, `--rope-scaling`).

---

## API Model Support

In addition to local GGUF models, Native Lab Pro can route inference to any OpenAI-compatible API endpoint or the Anthropic API. This lets you mix local and cloud models in the same session, or use the full feature set (reference injection, summarization, pipeline mode) against a hosted model.

### API Formats

| Format | Endpoint used | Auth header |
|---|---|---|
| `openai` | `/chat/completions` | `Authorization: Bearer <key>` |
| `anthropic` | `/v1/messages` | `x-api-key: <key>` |

Any self-hosted server that exposes an OpenAI-compatible `/chat/completions` endpoint (LM Studio, Ollama, vLLM, etc.) works with the `openai` format.

### ApiConfig Fields

Each API model is stored as an `ApiConfig` dataclass with the following fields: `name`, `provider`, `model_id`, `api_key`, `base_url`, `api_format`, `max_tokens`, `temperature`, and an optional custom prompt template configuration (`use_custom_prompt`, `system_prompt`, `user_prefix`, `user_suffix`, `assistant_prefix`, `prompt_template`).

### ApiRegistry

All API model configurations are saved to `api_models.json` and managed through `ApiRegistry`. The Models tab shows API models alongside local GGUF models and lets you load them into any engine role (general, reasoning, coding, etc.).

### ApiEngine

`ApiEngine` implements the same interface as `LlamaEngine` (`.load()`, `.create_worker()`, `.shutdown()`, `.is_loaded`, `.status_text`) so the rest of the application — including pipeline mode, summarization workers, and reference injection — works identically whether a local or API engine is active.

---

## UI Components

### ChatArea & MessageWidget

The chat area is a `QScrollArea` containing a vertical stack of `MessageWidget` instances. Each widget wraps a `RichTextEdit` (a `QTextBrowser` subclass) that renders Markdown to HTML using a custom renderer.

The `_md_to_html()` renderer handles fenced code blocks (with syntax highlighting for Python), inline code, headers, bold/italic, horizontal rules, bullet lists, and numbered lists. Fenced code blocks render as a two-row table: a toolbar showing the language tag, line count, and a "⧉ Copy" link; and the code body with monospace font and optional keyword coloring.

The copy links use a `copy://BLOCK_ID` anchor scheme. Clicking one triggers `mousePressEvent` in `RichTextEdit`, which looks up the raw code from an internal dictionary and writes it to the clipboard.

Long messages (over 260px tall) automatically get a collapsible "▼ Show more / ▲ Show less" toggle to keep the chat area manageable.

### ThinkingBlock

During summarization, a collapsible `ThinkingBlock` widget appears above the summary bubble. It shows a running log of each section's input preview and output summary, and collapses to a single toggle button. When summarization completes, the button turns green and shows a ✅ icon.

### InputBar

The input bar contains a model selector combo (which shows the detected family, quant type, and quality label for the selected model), a multi-line text input, a code mode toggle, a pipeline mode badge, and Send/Stop buttons. Enter sends the message; Shift+Enter inserts a newline.

### SessionSidebar

The sidebar groups sessions by creation date and supports incremental search. It shows the active session in purple and bold. Right-click menus support rename, Markdown export, and delete.

### ConfigTab

The configuration tab renders all `APP_CONFIG` fields with descriptions, input validation, and range hints. It also shows the list of paused summarization jobs with resume/delete controls.

### ServerTab

The Server tab provides a full UI for configuring llama-server binary paths, host and port range, extra CLI arguments, and GPU offload settings. Changes are saved to `server_config.json` and take effect on the next model load. Test buttons verify that configured binary paths are accessible.

### ModelDownloadTab

The Download tab provides the HuggingFace GGUF search and download interface described in [HuggingFace Model Downloader](#huggingface-model-downloader).

### McpTab

The MCP tab provides the server configuration and process management interface described in [MCP — Model Context Protocol](#mcp--model-context-protocol).

### PipelineBuilderTab

The Pipeline Builder tab provides the visual drag-and-drop pipeline editor described in [Visual Pipeline Builder](#visual-pipeline-builder).

### AppearanceTab

The Appearance tab exposes every color token in the active theme (background layers, accent, text, borders, status colors, etc.) through a color-picker grid. Clicking any swatch opens the system color dialog. Changes apply live to the entire application. Custom palettes are saved separately for light and dark modes and survive theme switching.

### LogConsole

A simple read-only `QTextEdit` with color-coded log levels (INFO in purple, WARN in yellow, ERROR in red), used for debugging engine status, model loading progress, and summarization progress.

---

## Theming & Appearance

Native Lab Pro ships with two built-in color themes: **light** and **dark**. The active theme is toggled from the **View** menu (or the keyboard shortcut `Ctrl+T`) and persists across restarts via `app_config.json`.

When the theme is switched, all tabs are rebuilt from scratch so that any baked-in color values (gradient cards on the Models tab, syntax highlighting in chat bubbles, etc.) update correctly. The rebuild happens in a single synchronous pass, so the switch is near-instant.

### Custom Palettes

The Appearance tab lets you override any color token in the current theme. Light-mode and dark-mode palettes are stored independently, so you can customise each without affecting the other. Custom palettes are saved as `custom_light_palette` and `custom_dark_palette` keys in `app_config.json`.

---

## Data Persistence

The app writes several categories of files:

| Path | Contents |
|---|---|
| `sessions/{id}.json` | Full conversation history per session |
| `custom_models.json` | List of manually-added model paths |
| `model_configs.json` | Per-model parameter configurations |
| `parallel_prefs.json` | Parallel loading and pipeline settings |
| `app_config.json` | All threshold, default, and theme settings |
| `server_config.json` | Binary paths, host/port, GPU offload settings |
| `api_models.json` | API model configurations (OpenAI/Anthropic endpoints) |
| `mcp_config.json` | MCP server definitions |
| `paused_jobs/{id}.json` | Full state snapshots of paused summarization jobs |
| `ref_cache/{id}_raw.txt` | Raw text of attached reference files |
| `ref_cache/{id}.pkl` | Pickled chunk cache for disk-spilled references |
| `ref_index/{sid}_refs.json` | Reference metadata index per session |

---

## RAM Watchdog & Memory Management

The `RamWatchdog` class and the `_ram_free_mb()` function (backed by `psutil` when available) work together to prevent out-of-memory crashes during large document processing.

The watchdog triggers in three situations: when a new reference file is added, periodically during multi-PDF processing (every 5 chunks), and reactively before the final consolidation pass. When triggered, `SessionReferenceStore.flush_ram()` calls `_spill_to_disk()` on every loaded reference, which pickles all chunk text to disk and clears the in-memory `_hot` dictionary. Chunks are reloaded from disk on demand with LRU-style caching (the most recently accessed chunks staying in `_hot` as long as RAM permits).

The **reactive reload** before the final summarization pass re-warms the most query-relevant chunks back into RAM if memory has freed up by that point, ensuring the model has the best possible context for the consolidation step.

---

## Keyboard Shortcuts & Menus

| Shortcut | Action |
|---|---|
| `Ctrl+N` | New session |
| `Ctrl+Q` | Quit |
| `Ctrl+B` | Toggle session sidebar |
| `Ctrl+L` | Switch to Logs tab |
| `Ctrl+M` | Switch to Models tab |
| `Enter` | Send message |
| `Shift+Enter` | Insert newline in input |

The **File** menu supports new session creation and export of the current session to JSON, Markdown, or plain text. The **Model** menu provides a one-click model reload. The **View** menu provides the sidebar and tab navigation shortcuts.

---

## Developer Notes

### Adding a New Model Family

To add support for a new model family's prompt template, add a new `ModelFamily` entry to `FAMILY_TEMPLATES` and add a matching pattern to the `patterns` list in `detect_model_family()`. The pattern list is checked in order, so place more specific patterns before more general ones.

### Adding a New Script Language

To add parsing support for a new language, add the file extension to `SCRIPT_LANGUAGES` and add a corresponding `_parse_{key}()` class method to `ScriptParser`. If no dedicated parser is added, the generic fallback parser will be used for the new extension.

### Adding a New API Provider

To add a new cloud provider, add an `ApiConfig` with the appropriate `base_url`, `api_format` (`"openai"` or `"anthropic"`), and `custom_provider_name` through the Models tab UI. If the provider uses a non-standard authentication or request format, extend `ApiStreamWorker` with a new format branch.

### Adding a New Pipeline Block Type

To add a new block type to the visual pipeline builder, add a `PipelineBlockType` constant and create the corresponding rendering logic in `PipelineBlock`. Add a button in `PipelineBuilderTab._build()` to make it accessible from the sidebar. If the block needs custom parameters, add a dialog class similar to `LlmLogicEditorDialog`.

### Engine Mode Fallback

`LlamaEngine.load()` tries `llama-server` first, falling back to `llama-cli`. If you only have one binary, the app will use it automatically. The server mode is strongly preferred because it keeps the model loaded between prompts, avoiding the significant startup overhead of `llama-cli` mode for multi-turn conversations.

### Threading Model

All inference (streaming tokens, summarization, pipeline stages, HuggingFace downloads, MCP server probing) runs on `QThread` subclasses with PyQt signals for cross-thread UI updates. The main thread never blocks. Long-running background operations expose an `abort()` method that sets a flag checked at each iteration, ensuring clean cancellation. The `_summary_worker` additionally supports a `request_pause()` path that saves state to disk before exiting.

### Stray Process Cleanup

On shutdown, `_kill_stray_llama_servers()` is called to terminate any orphaned `llama-server` processes from previous crashed sessions, in addition to cleanly shutting down all currently managed engines.

## Project Structure 
```
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
  │   └── nativelab-0.1.5.tar.gz
  ├── .github
  │   ├── ISSUE_TEMPLATE
  │   │   ├── bug_report.md
  │   │   └── feature_request.md
  │   ├── PULL_REQUEST_TEMPLATE.md
  │   └── workflows
  │       └── build-mac.yml
  ├── .gitignore
  ├── index.html
  ├── LICENSE
  ├── nativelab
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
  │   │   ├── engine_global.py
  │   │   ├── engines
  │   │   │   ├── apiengine.py
  │   │   │   ├── __init__.py
  │   │   │   └── llamaengine.py
  │   │   ├── __init__.py
  │   │   ├── streamer_global.py
  │   │   └── streamerworker
  │   │       ├── apistreamer.py
  │   │       ├── clistreamer.py
  │   │       ├── __init__.py
  │   │       └── serverstreamer.py
  │   ├── GlobalConfig
  │   │   ├── binaryResolve.py
  │   │   ├── config_global.py
  │   │   ├── config.py
  │   │   ├── const.py
  │   │   ├── hardwareUtil.py
  │   │   └── __init__.py
  │   ├── icon.ico
  │   ├── icon.png
  │   ├── imports
  │   │   ├── import_global.py
  │   │   ├── __init__.py
  │   │   ├── optional_lib.py
  │   │   ├── pyqt_lib.py
  │   │   └── standard_lib.py
  │   ├── main.py
  │   ├── manual.py
  │   ├── Model
  │   │   ├── APImodels.py
  │   │   ├── __init__.py
  │   │   ├── model_family.py
  │   │   ├── model_global.py
  │   │   ├── ModelRegistry.py
  │   │   └── templates.py
  │   ├── pipelinebuilder
  │   │   ├── blck_typ.py
  │   │   ├── canvas.py
  │   │   ├── editordialogue.py
  │   │   ├── executionWorker.py
  │   │   ├── __init__.py
  │   │   ├── outrender.py
  │   │   ├── pipblck.py
  │   │   ├── pipebuilder.py
  │   │   ├── pipefunctions.py
  │   │   └── pipe_global.py
  │   ├── Prefrences
  │   │   ├── __init__.py
  │   │   ├── ParallelLoading.py
  │   │   └── prefrence_global.py
  │   ├── Server
  │   │   ├── hfdwld.py
  │   │   ├── __init__.py
  │   │   ├── server_global.py
  │   │   └── ServerHandling.py
  │   └── UI
  │       ├── buildUI.py
  │       ├── effects.py
  │       ├── __init__.py
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
  │       ├── RichTextEditor.py
  │       ├── tabs.py
  │       ├── UI_const.py
  │       ├── UI_global.py
  │       └── widgets.py
  ├── NativeLab.spec
  ├── pyproject.toml
  ├── README.md
  ├── requirements.txt
  ├── SECURITY.md
  ├── setup.html
```