# Native Lab Pro v2

**A fully local, privacy-first LLM desktop application powered by llama.cpp**

---

## Table of Contents

1. [Overview](#overview)
2. [What's New in v2](#whats-new-in-v2)
3. [Architecture Overview](#architecture-overview)
4. [Requirements & Installation](#requirements--installation)
5. [Configuration](#configuration)
6. [Model Management](#model-management)
7. [Prompt Template System](#prompt-template-system)
8. [Quantization Format Support](#quantization-format-support)
9. [Chat & Sessions](#chat--sessions)
10. [Reference Engine](#reference-engine)
11. [Script Parser](#script-parser)
12. [Summarization Pipeline](#summarization-pipeline)
13. [Multi-PDF Summarization](#multi-pdf-summarization)
14. [Parallel Loading & Pipeline Mode](#parallel-loading--pipeline-mode)
15. [UI Components](#ui-components)
16. [Data Persistence](#data-persistence)
17. [RAM Watchdog & Memory Management](#ram-watchdog--memory-management)
18. [Keyboard Shortcuts & Menus](#keyboard-shortcuts--menus)
19. [Developer Notes](#developer-notes)

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

All v1 features — session management, PDF summarization, reference injection, chunked long-document processing, pause/resume of summarization jobs — are fully preserved.

---

## Architecture Overview

The application is organized into several distinct layers that work together.

```
┌─────────────────────────────────────────────────────────────┐
│                        MainWindow (PyQt6)                    │
│  ┌──────────────┐  ┌────────────────────────────────────┐   │
│  │ SessionSidebar│  │            QTabWidget              │   │
│  │              │  │  ┌──────────┬────────┬──────────┐  │   │
│  │ (chat list)  │  │  │ Chat     │ Models │ Config   │  │   │
│  └──────────────┘  │  │          │        │          │  │   │
│                     │  │ChatModule│        │ConfigTab │  │   │
│                     │  └──────────┴────────┴──────────┘  │   │
│                     └────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│          Engine Layer           │
│  LlamaEngine  (manages process) │
│  ├── ServerStreamWorker (HTTP)  │
│  └── CliStreamWorker (stdio)    │
│                                 │
│  PipelineWorker (multi-engine)  │
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
│  APP_CONFIG (app_config.json)   │
│  ParallelPrefs (JSON)           │
│  PausedJobs (paused_jobs/)      │
└─────────────────────────────────┘
```

The `LlamaEngine` class is the heart of the inference layer. It tries to start `llama-server` first (preferred, because it supports true HTTP streaming with a persistent model in memory). If the server binary is not found or fails to start, it falls back to calling `llama-cli` as a subprocess for each prompt. This dual-mode design means the app works even in minimal llama.cpp builds.

---

## Requirements & Installation

### System Requirements

Native Lab Pro runs on Linux (primary development target). The paths in the source code are Linux-style and point to a specific user's home directory by default.

The minimum hardware depends entirely on the models you intend to run. As a rough guide, a 7B parameter Q4 model requires approximately 4–5 GB of RAM, a 13B Q5 model requires roughly 9–10 GB, and a 70B Q4 model requires 38–40 GB.

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

You must compile or download llama.cpp and point the app to the binaries. The default paths in the source are:

```python
LLAMA_CLI    = "/home/hrirake/llama.cpp/build/bin/llama-cli"
LLAMA_SERVER = "/home/hrirake/llama.cpp/build/bin/llama-server"
```

Edit these two constants at the top of the file to match your actual binary locations.

### Models Directory

By default the app scans for `.gguf` files in:

```python
MODELS_DIR = Path("/home/hrirake/localllm")
```

Edit this path, or use the **Browse GGUF…** button in the Models tab to add individual model files from anywhere on your filesystem.

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

### LogConsole

A simple read-only `QTextEdit` with color-coded log levels (INFO in purple, WARN in yellow, ERROR in red), used for debugging engine status, model loading progress, and summarization progress.

---

## Data Persistence

The app writes several categories of files:

| Path | Contents |
|---|---|
| `sessions/{id}.json` | Full conversation history per session |
| `custom_models.json` | List of manually-added model paths |
| `model_configs.json` | Per-model parameter configurations |
| `parallel_prefs.json` | Parallel loading and pipeline settings |
| `app_config.json` | All threshold and default settings |
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

### Engine Mode Fallback

`LlamaEngine.load()` tries `llama-server` first, falling back to `llama-cli`. If you only have one binary, the app will use it automatically. The server mode is strongly preferred because it keeps the model loaded between prompts, avoiding the significant startup overhead of `llama-cli` mode for multi-turn conversations.

### Threading Model

All inference (streaming tokens, summarization, pipeline stages) runs on `QThread` subclasses with PyQt signals for cross-thread UI updates. The main thread never blocks. Long-running background operations expose an `abort()` method that sets a flag checked at each iteration, ensuring clean cancellation. The `_summary_worker` additionally supports a `request_pause()` path that saves state to disk before exiting.
