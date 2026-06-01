# Workflows

End-to-end features that span engines, references, and the UI: pipelines, summarization, MCP, and downloads.

- [Reference engine](#reference-engine)
- [Script parser](#script-parser)
- [Summarization pipeline](#summarization-pipeline)
- [Multi-PDF synthesis](#multi-pdf-synthesis)
- [Parallel loading & pipeline mode](#parallel-loading--pipeline-mode)
- [Visual pipeline builder](#visual-pipeline-builder)
- [MCP](#mcp)
- [Download tab](#download-tab)

---

## Reference engine

Attach documents to a chat session and have the most relevant excerpts auto-injected into every prompt as `[REFERENCE: …]` blocks.

### `SmartReference`

When a file is added (PDF, plain text, or `.py` treated as text), it's processed into a `SmartReference`:

- Raw text split into overlapping chunks of `chunk_index_size` characters.
- Chunks live in RAM up to `max_ram_chunks`. Beyond that, they're pickled to `ref_cache/{id}.pkl` and reloaded on demand with LRU-style caching.
- Retrieval scores each chunk by how many query words it contains (with a phrase-match bonus). The top-K chunks (`ref_top_k`, default 6) are concatenated up to `ref_max_context_chars` (default 3000) and injected.

### `ScriptSmartReference`

Source files get richer treatment via `ScriptParser` (see below). The structured index - imports, classes, functions, signatures, docstrings, decorators - drives much sharper retrieval than naive text chunking.

### `SessionReferenceStore`

Each session has its own store. Metadata lands in `ref_index/{sid}_refs.json`; raw text in `ref_cache/{id}_raw.txt`. References survive restarts.

---

## Script parser

`ScriptParser` produces a structured `ParsedScript` for any source language.

### Languages

Python (AST), JavaScript / JSX, TypeScript / TSX, SQL, Rust, Go, C, C++, Java, Kotlin, Ruby, Bash, YAML, JSON, TOML, Lua, Swift, C#, PHP, R, Julia, Markdown.

### Extracted entities

- **Imports** - every import / require / use / include with resolved names.
- **Functions** - full signature, decorators, line numbers, body text, docstring.
- **Classes / structs** - base classes, methods, line ranges, docstrings.
- **Constants** - uppercase names + `#define` macros.
- **Types** - TS interface/type aliases, Rust trait/impl, Go interfaces.
- **SQL objects** - CREATE TABLE/VIEW/PROCEDURE/INDEX, CTEs.
- **Config keys** - top-level keys for JSON/YAML/TOML.

### `ParsedScript.build_context()`

Always includes imports + constants (compact), then ranks all functions/classes by query relevance and includes as many as fit the character budget. Each context block has a structured header listing language, file name, and a summary of what was found.

---

## Summarization pipeline

For documents that don't fit in the context window.

### `ChunkedSummaryWorker`

1. Splits text into chunks of `summary_chunk_chars`, breaking on paragraph boundaries when possible.
2. For each chunk, prepends a "running context" - the last `summary_ctx_carry` characters of the previous summary - to maintain continuity.
3. Sends `chunk + context` to the model and collects a section summary.
4. Auto-saves state to `paused_jobs/` every 3 chunks.
5. Final pass synthesizes all section summaries into one cohesive document summary.

If a dedicated summarization or reasoning engine is loaded, it runs the final consolidation pass.

### Pause / resume

Click **Stop** mid-job to snapshot state (raw text, completed summaries, running context, progress, model path) to `paused_jobs/{job_id}.json`. The top-right settings button opens Config, which lists paused jobs with **Resume Job**.

The app also auto-suggests pausing after `pause_after_chunks` chunks if many remain - useful for long jobs split across sessions.

---

## Multi-PDF synthesis

`MultiPdfSummaryWorker` extends the single-PDF pipeline:

1. Run each PDF through the chunked summarizer.
2. Produce a per-file consolidated summary.
3. Final cross-document pass identifies themes, differences, and connections.

Integrated with the RAM watchdog (checks every 5 chunks; spills caches when free memory drops). Pause/resume works identically - the snapshot includes the list of PDF texts, current file index, and chunk position.

---

## Parallel loading & pipeline mode

### Parallel loading

`ParallelPrefs` tracks whether parallel loading is enabled, which roles auto-load on startup, and whether pipeline mode is active. Saved to `parallel_prefs.json`.

When enabled, you can load reasoning, summarization, coding, and secondary engines simultaneously. Each runs its own `llama-server` on a different port (or falls back to `llama-cli`). The app warns clearly about RAM implications before activation.

### Pipeline mode

Activates automatically when **all** of these hold:

- Parallel loading is on.
- Pipeline mode is checked in settings.
- At least one non-coding engine is loaded.
- The coding engine is loaded.
- The incoming prompt is detected as a coding request (keywords like `def`, `class`, `implement`, `generate code`, fenced blocks, etc.; or the **💻 Code** toggle is on).

Flow for a coding prompt:

**Stage 1 - structural insights.** Each non-coding engine produces a structured architectural analysis: purpose, recommended pattern, components, data flow, edge cases, suggested libraries. No code yet. Each engine streams into its own chat bubble in real time.

**Stage 2 - code generation.** The coding engine receives a prompt containing all stage-1 insights as a labelled context block, followed by the original request, then generates the implementation.

This typically produces better-structured code than asking a coding model directly, because architectural planning is done by models that may be stronger reasoners.

---

## Visual pipeline builder

The most powerful feature in NativeLab. A node-based editor with 20+ block types, loop support, LLM-powered routing, live execution log, and per-block streaming.

Pipelines are saved as JSON in `~/.native_lab/pipelines/` and loadable from within the UI.

### Canvas

A 1400×900+ infinite workspace with 20px snap-to-grid.

- **Place** - click any block button in the sidebar, or drag a model directly from the model list onto the canvas.
- **Select / move** - click a block; drag anywhere; snaps to grid on release.
- **Connect** - hover a block until ports appear, click-drag from a port to another block's port.
- **Delete** - right-click for a context menu.

Connection rules: model→model is blocked (use an Intermediate block); duplicate connections silently ignored; non-logic blocks have one outgoing connection per port; logic blocks fan out.

### Block types

#### I/O & model

| Block | Color  | Description |
|-------|--------|-------------|
| **▶ Input** | Green | Captures the user prompt. Required. |
| **■ Output** | Red | Renders the final result with Markdown + code highlighting. Required. |
| **⚡ Model** | Purple | LLM inference. Drag-droppable from the model list. |
| **◈ Intermediate** | Yellow | Captures + streams output mid-pipeline. Required between chained models. |

#### Context injection

| Block | Color | Description |
|-------|-------|-------------|
| **📎 Reference** | Purple | Inject a static text snippet as `[REFERENCE: name] … [/REFERENCE]`. |
| **💡 Knowledge** | Violet | Like Reference but headed `Knowledge Base:`. |
| **📄 PDF** | Cyan | Loads a PDF; auto-summarises if >4500 chars. Two roles: `reference` (appended) or `main` (replaces, prior context becomes a reference). |

#### Deterministic logic (no model calls)

| Block | Operation |
|-------|-----------|
| **⑂ IF/ELSE** | Python expression on `text`. TRUE → E port, FALSE → W port. |
| **⑃ SWITCH** | Python expression returns a key; routes to the matching outgoing label (`default` is the catch-all). |
| **⊘ FILTER** | TRUE passes through; FALSE terminates the pipeline with a drop reason. |
| **⟲ TRANSFORM** | prefix / suffix / find-replace / upper / lower / strip / truncate. |
| **⊕ MERGE** | Combine multiple inputs: concat, prepend, append, or json array. |
| **⑁ SPLIT** | Broadcast incoming text to all outgoing connections. |
| **⌥ Custom Code** | User-written Python with `text`, `result`, `metadata`, `log()`. |

The Python sandbox for logic blocks exposes only safe builtins (`len`, `str`, `int`, `float`, `bool`, `list`, `dict`, `any`, `all`, `min`, `max`, `abs`, `isinstance`, `True`, `False`, `None`) and the `text` variable. No imports, no fs, no network.

#### LLM logic (small-model decisions)

Each makes a real inference call. Configurable model, max tokens, temperature, and "pass through on error" toggle.

| Block | What the LLM does |
|-------|-------------------|
| **🧠 LLM-IF** | Plain-English yes/no. |
| **🧠 LLM-SWITCH** | Classifies into one of your labelled outgoing arrows. |
| **🧠 LLM-FILTER** | PASS / STOP based on a plain-English condition. |
| **🧠 LLM-TRANSFORM** | Rewrites/reformats incoming text. |
| **🧠 LLM-SCORE** | 1–10 score routes LOW (1–3) → E, MID (4–7) → S, HIGH (8–10) → W. Label an arrow `score` to receive the raw number. |

Use 0.5B–1.5B models for routing - they only need to output a word or a number, so they're fast and accurate at this.

### Loop edges

Draw a connection **backwards** (downstream → upstream) and the canvas detects the cycle and prompts for a loop count (2–999). Loop edges render as dashed lines with an `×N` badge. At runtime, each edge has its own visit counter; when it hits the limit the edge is skipped and execution continues to the next non-loop outgoing connection.

```
▶ Input → ⚡ Draft → ◈ Intermediate ("Critique and list improvements")
     ↑                                  |
     └────────── ×3 loop ───────────────┘
→ ■ Output
```

### Execution

Click **▶ Run Pipeline**. `PipelineExecutionWorker` (a `QThread`) processes blocks in topological BFS order, carrying a `current_text` from block to block.

- Each block emits `step_started` (spinner) and `step_done` (populates its tab).
- Model blocks start their own `llama-server` (or fall back to `llama-cli`) and stream tokens live.
- Execution log shows every decision: condition evaluated, branch taken, chars processed, FILTER pass/drop, MERGE inputs combined, SPLIT fan-out count, loop iteration.
- **⏹ Stop Execution** halts cleanly after the current HTTP request completes.

### Save / load

- **💾 Save Pipeline…** - name + write to `~/.native_lab/pipelines/{name}.json`.
- **📂 Load Pipeline…** - list with delete; clearing prompt if canvas non-empty.

Persisted per block: type, position, size, model path, role, label, all metadata. Per connection: source/target ports, `is_loop`, `loop_times`, branch label.

### Performance tips

- Tiny models (0.5B–1.5B) for all LLM logic blocks.
- Large models only for actual generation (Model, LLM-TRANSFORM).
- TRANSFORM before expensive model calls keeps context short.
- Each loop iteration is a full call - `3 models × 5 loops = 15` requests.
- Routing block max-tokens can be as low as 16 (yes/no needs nothing more).

---

## MCP

The **Dev > MCP** page manages **Model Context Protocol** servers - extending the LLM with filesystem, web, database, or custom-tool access.

### Server types

- **Stdio** - launched as child processes by the app; lifecycle managed (start/stop/restart). Communicate over stdin/stdout.
- **SSE** - connect over HTTP (Server-Sent Events). You supply the URL of an already-running server.

### Configuration

Each entry stores name, transport, command/URL, optional env overrides, and a description. Persisted to `mcp_config.json`. The list shows status (running / stopped / error).

---

## Download tab

The Download tab provides model and runtime downloaders without leaving NativeLab.

Each downloader includes a **Popular** selector populated from `nativelab/Model/templates.py`. Pick a preset to fill the repo/model field, then inspect/search/pull normally. The list covers current common GGUF repos, HF Transformers snapshots, and Ollama library names while still allowing custom entries.

For gated or private Hugging Face repos, open **Accounts > Hugging Face** first and click **Login with Hugging Face**. NativeLab uses its built-in public OAuth client ID, saves credentials locally in `localllm/cred/huggingface.json`, and reuses the saved token for GGUF search/download, HF snapshot downloads, and `hf:` model loading. Manual access-token paste remains available as an advanced fallback. The Download tab shows the current auth state and links back to Accounts. If Hugging Face still returns HTTP 403 while signed in, open the repo page in your browser and accept/request gated access for that specific model before retrying.

### GGUF HuggingFace search

Enter a repo ID (e.g. `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`) and click **Search**. The results list shows GGUF files with sizes and last-modified dates.

Select a file and click **Download**. Streams directly to your models folder with a progress bar. If the target exists, you're warned before overwriting. `HfSearchWorker` and `HfDownloadWorker` (QThread subclasses) keep the UI responsive.

The CLI's wizard does the same thing - `nativelab/cli/hf_download.py` is a synchronous version of the same logic with resume + retry.

### HF Transformers snapshot

Use this when you want `hf:<local-folder>` models to work offline with Transformers:

1. Install the optional backend dependencies with `pip install -e ".[hf]"`.
2. Enter a repo ID and revision, then click **Inspect**.
3. Review the runtime files. NativeLab includes configs, tokenizer/processor files, safetensors or PyTorch shards and indexes, custom Python files, and repo metadata.
4. Click **Download Snapshot**. Files are written under `localllm/hf_transformers/<namespace>/<repo>/` with subdirectories preserved.

Downloads resume through `.part` files and can be paused, cancelled, or cancelled with partial files deleted. Completed snapshots are automatically registered as `hf:<downloaded-folder>`.

Default revision, local-files-only mode, dtype, device map, safetensors policy, attention implementation, max-memory map, and quantization mode live in the top-right Settings button under **App Configuration > HF Transformers**. The legacy `hf_token` setting remains as a fallback when no Accounts token is saved.

### Ollama model pull

NativeLab does not install or start Ollama. If an Ollama daemon is already running:

1. Confirm the host in **App Configuration > Ollama** or edit it directly in the Download tab.
2. Click **Refresh** to list installed models from `/api/tags`.
3. Type a remote model name, such as `llama3.2:3b`, and click **Pull**.

Pull progress streams from `/api/pull`. When the pull completes, NativeLab registers the model as `ollama:<model>`. The shared engine uses the same host and `keep_alive` setting when loading and chatting with Ollama models. If the daemon is not reachable, NativeLab reports that explicitly; start the Ollama app or run `ollama serve`, then retry.

### llama.cpp runtime

The llama.cpp section downloads prebuilt `llama-server` and `llama-cli` release assets into `./llama/bin/`, where the Server tab and CLI can find them automatically.
