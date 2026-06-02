# Features

NativeLab is built around four pillars: **local-first inference**, **multi-engine pipelines**, **rich document workflows**, and an **experimentation layer** for new ideas.

---

## 🆕 What's new in v0.3.3

### Labs - the experimentation layer
A new `nativelab/labs/` package and a dedicated GUI tab. Every lab feature receives a single `LabEndpoints` instance and uses it for engine status, model swap, context change, and synchronous LLM calls (auto-routing API → server → CLI). Adding a feature is dropping one file and registering it. See [labs.md](labs.md).

### Terminal CLI - `nativelab --cli`
Claude-Code-style terminal client. Interactive setup wizard downloads a model from HuggingFace, picks a context size, saves prefs, and drops you into a REPL with `@file` embedding, slash commands, and built-in linting. Inline icon rendering on iTerm2 / Kitty terminals. See [cli.md](cli.md).

### Endpoint surface, shared with the CLI
The same `LabEndpoints` that powers Labs panels also powers the CLI's chat REPL. Reverse-routing hooks (`request_load_model`, `request_context`, `request_unload`) are wired uniformly - `/load` from the REPL behaves identically to a lab feature requesting a model swap.

---

## Catalogue (everything in v0.3.3)

### Inference

- **Local llama.cpp** - `LlamaEngine` starts `llama-server` for true HTTP streaming with the model resident in RAM, falling back to per-prompt `llama-cli` if the server binary is unavailable.
- **API models** - `ApiEngine` is a drop-in replacement that talks to OpenAI-compatible (`/chat/completions`, Bearer auth) and Anthropic (`/v1/messages`, `x-api-key`) endpoints. Works with hosted services and self-hosted servers like LM Studio, Ollama, and vLLM. See [models.md#local-and-api-backend-support](models.md#local-and-api-backend-support).
- **GPU offload** - Settings > Server toggles `ngl`, `main_gpu`, and `tensor_split` for multi-GPU rigs.
- **Parallel engines** - Load reasoning, summarization, coding, and secondary engines simultaneously, each on its own llama-server port. RAM warnings before activation.
- **Pipeline mode** - Coding prompts trigger structural-insight passes through non-coding engines first, then feed those insights into the coding model. Higher-quality output than asking a coding model directly.

### Models

- **Auto family detection** - 20+ chat templates recognised by filename (DeepSeek, DeepSeek-R1, Mistral, Mixtral, LLaMA-2/3, Phi-3/3.5, Qwen/ChatML, Gemma, CodeLlama, Falcon, Vicuna, OpenChat, Neural-Chat, Starling, Yi, Zephyr, Solar, Orca, Command-R).
- **Quantization detection** - Every llama.cpp quant from legacy `Q4_0` through K-quants (`Q2_K`–`Q6_K`) and imatrix variants (`IQ1_S`–`IQ4_XS`). Color-coded quality tiers in the model list.
- **Per-model parameters** - threads, ctx, temperature, top_p, repeat_penalty, n_predict - all editable per file and persisted to `model_configs.json`.
- **Downloaders** - Search GGUF repos, choose popular presets from `templates.py`, download full HF Transformers snapshots for `hf:<local-folder>`, pull Ollama models from a running daemon, and install llama.cpp runtime builds.

### Document & code workflows

- **Reference engine** - Attach PDFs, plain text, or source files to a session. Top-K relevant chunks are auto-injected as `[REFERENCE: …]` blocks before each prompt.
- **Script parser** - AST-level parsing for Python, regex parsers for 20+ other languages (JS/TS, Rust, Go, C/C++, Java, Kotlin, Ruby, SQL, Bash, YAML/JSON/TOML, Lua, Swift, C#, PHP, R, Julia, Markdown). Extracts imports, classes, functions, constants, types, SQL objects, config keys.
- **Chunked summarization** - Long PDFs are split, summarised section-by-section with carry-over context, then consolidated. Pause/resume to disk between chunks.
- **Multi-PDF synthesis** - Run summarization across a batch of PDFs and produce a cross-document themes report.
- **RAM watchdog** - Reactive disk-spilling of reference caches when free RAM drops below a configurable threshold; reactive reload before final consolidation passes.

### Visual pipeline builder

A node-based editor with 20+ block types:

- I/O - Input, Output, Model, Intermediate (live streaming output).
- Context - Reference, Knowledge, PDF (auto-summarising for large files).
- Logic - IF/ELSE, SWITCH, FILTER, TRANSFORM, MERGE, SPLIT, Custom Code.
- LLM logic - LLM-IF, LLM-SWITCH, LLM-FILTER, LLM-TRANSFORM, LLM-SCORE.
- Loop edges with iteration counts (e.g. drafting → critique × 3).
- Sandboxed Python expressions (no imports, no fs/network).
- Live execution log + per-block streaming. Save/load to JSON.

See [workflows.md#visual-pipeline-builder](workflows.md#visual-pipeline-builder).

### MCP integration

Manage **Model Context Protocol** servers from a dedicated tab. Configure stdio servers (launched as children) or SSE servers (HTTP endpoints), with environment overrides and live status display. See [workflows.md#mcp](workflows.md#mcp).

### UX polish

- Light + dark themes, toggleable from the View menu.
- Custom palettes - every color token is editable via Settings > Appearance; saved separately for light/dark.
- Markdown rendering with per-language syntax highlighting and one-click "Copy" buttons on every code block.
- Session sidebar grouped by date, with rename / export Markdown / delete.
- Live RAM and context-usage indicators in the status bar.
- Pause/resume of long summarization jobs.

### Cross-platform

- Windows binary paths get `.exe` automatically.
- macOS GitHub Actions workflow for building app bundles.
- Frozen builds (PyInstaller) supported via `sys._MEIPASS` resolution.

---

## Roadmap pointers

The Labs layer is intentionally where the next experiments will land - it's the lowest-friction place to add features that touch the engine. If you want to contribute, [labs.md](labs.md) is the on-ramp.
