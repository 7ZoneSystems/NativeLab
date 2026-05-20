# UI

GUI tour, theming, persistence, shortcuts, and live RAM management.

---

## Tabs

| Tab | What it's for |
|-----|---------------|
| **💬 Chat** | The main conversation window. Sidebar of sessions, message bubbles, input bar with model selector. |
| **📚 Models** | Library of GGUF models. Set roles, edit per-model parameters, browse for new files. |
| **🌐 API Models** | Configure OpenAI / Anthropic / self-hosted endpoints. |
| **Accounts** | Account integrations. The **Hugging Face** sub-tab provides one-click browser login for private or gated repos and stores credentials locally in `localllm/cred/huggingface.json`. |
| **Top-right Settings** | App Configuration, including backend defaults, HF Transformers load settings, Ollama host settings, and paused-job manager. |
| **🖥️ Server** | llama.cpp binary paths, host/port range, GPU offload, extra args. |
| **🔗 Pipeline** | Visual pipeline builder ([details](workflows.md#visual-pipeline-builder)). |
| **🔌 MCP** | Model Context Protocol server manager ([details](workflows.md#mcp)). |
| **⬇️ Download** | HuggingFace GGUF downloads, HF Transformers snapshots, Ollama pulls, and llama.cpp release installer. |
| **⌬ Labs** | Experimentation tab ([details](labs.md)). |
| **🎨 Appearance** | Color palette editor (light + dark, saved separately). |
| **📋 Logs** | Color-coded log console (INFO / WARN / ERROR). |

---

## Chat components

### `ChatArea` & `MessageWidget`

A `QScrollArea` holding a vertical stack of `MessageWidget` instances. Each wraps a `RichTextEdit` (`QTextBrowser` subclass) that renders Markdown to HTML through a custom `_md_to_html()` renderer:

- Fenced code blocks → two-row table (toolbar with language tag + line count + "⧉ Copy" link, then the code body with monospace + keyword colouring).
- Inline code, headers, bold/italic, horizontal rules, bullet lists, numbered lists.

Copy links use `copy://BLOCK_ID` anchors. Clicking dispatches through `mousePressEvent` in `RichTextEdit`, which looks up the raw code from an internal dictionary and writes it to the clipboard.

Long messages (>260px) get a collapsible **▼ Show more / ▲ Show less** toggle.

### `ThinkingBlock`

During summarisation, a collapsible widget appears above the summary bubble with a running log of each section's input preview and output. When the job completes, it turns green with ✅.

### `InputBar`

- Model selector combo (shows family + quant + quality label).
- Multi-line text input (Enter sends, Shift+Enter newline).
- Code mode toggle, pipeline mode badge, Send / Stop buttons.

### `SessionSidebar`

Sessions grouped by date, incremental search, active session in purple bold. Right-click for rename / export Markdown / delete.

---

## Theming

Light + dark themes ship with the app. Toggle from **View → Toggle Theme** or `Ctrl+T`. The active theme persists via `app_config.json`.

When the theme switches, all tabs are rebuilt from scratch in a single synchronous pass so any baked-in colors (gradient cards, syntax highlighting) update correctly.

### Custom palettes

The Appearance tab exposes every color token. Click any swatch → system color dialog → applied live. Light and dark palettes are stored independently as `custom_light_palette` / `custom_dark_palette` in `app_config.json`, so customising one doesn't affect the other.

---

## Keyboard shortcuts

| Shortcut       | Action                            |
| -------------- | --------------------------------- |
| `Ctrl+N`       | New session                       |
| `Ctrl+Q`       | Quit                              |
| `Ctrl+B`       | Toggle session sidebar            |
| `Ctrl+L`       | Switch to Dev > Logs              |
| `Ctrl+M`       | Switch to Models tab              |
| `Ctrl+T`       | Toggle theme (light/dark)         |
| `Enter`        | Send message                      |
| `Shift+Enter`  | Insert newline in input           |

The **File** menu provides session export (JSON / Markdown / plain text). The **Model** menu has a one-click reload. The **View** menu groups sidebar + tab navigation shortcuts.

---

## Status bar

Bottom of the window:

- Current model + family + quant.
- Active engine status (server port / CLI mode / API).
- Live RAM gauge (when `psutil` is available).
- Context usage bar (`Session.approx_tokens` ÷ ctx_value).

---

## Data persistence

All on-disk state is in the project working directory.

| Path | Contents |
|------|----------|
| `sessions/{id}.json` | Conversation history per session. |
| `localllm/custom_models.json` | Manually added model paths. |
| `localllm/model_configs.json` | Per-model parameter configurations. |
| `localllm/parallel_prefs.json` | Parallel loading + pipeline settings. |
| `app_config.json` | Thresholds, defaults, theme, custom palettes. |
| `localllm/server_config.json` | Binary paths, host/port, GPU offload. |
| `localllm/api_models.json` | API model configurations. |
| `localllm/cli_prefs.json` | CLI's last-used model + ctx. |
| `mcp_config.json` | MCP server definitions. |
| `paused_jobs/{id}.json` | Snapshots of paused summarisation jobs. |
| `ref_cache/{id}_raw.txt` | Raw text of attached reference files. |
| `ref_cache/{id}.pkl` | Pickled chunk caches for spilled references. |
| `ref_index/{sid}_refs.json` | Reference metadata index per session. |
| `~/.native_lab/pipelines/{name}.json` | Saved visual pipelines. |

---

## RAM watchdog

The `RamWatchdog` class and `_ram_free_mb()` (backed by `psutil`) prevent OOMs during long document processing.

Triggers:

- A new reference file is added.
- Periodically during multi-PDF processing (every 5 chunks).
- Reactively before final consolidation passes.

When triggered, `SessionReferenceStore.flush_ram()` calls `_spill_to_disk()` on every loaded reference - pickles all chunk text and clears `_hot`. Chunks reload on demand with LRU caching.

The **reactive reload** before final summarisation re-warms the most query-relevant chunks back into RAM if memory has freed up by then.

---

## App config

All runtime thresholds live in `app_config.json`, fully editable through the **⚙️ Config** tab.

| Setting | Default | Description |
|---|---|---|
| `ram_watchdog_mb` | 800 | Free-RAM threshold (MB) that triggers spill. |
| `chunk_index_size` | 400 | Character size of indexed reference chunks. |
| `max_ram_chunks` | 120 | Max chunks per reference kept in RAM. |
| `summary_chunk_chars` | 3000 | Characters per chunk during summarisation. |
| `summary_ctx_carry` | 600 | Chars of previous summary carried forward as context. |
| `summary_n_pred_sect` | 380 | Max tokens per section summary. |
| `summary_n_pred_final` | 700 | Max tokens for final consolidation. |
| `multipdf_n_pred_sect` | 380 | Tokens per section in multi-PDF jobs. |
| `multipdf_n_pred_final` | 900 | Tokens for cross-document final pass. |
| `ref_top_k` | 6 | Top-scoring chunks retrieved per reference per query. |
| `ref_max_context_chars` | 3000 | Max total reference text injected. |
| `pause_after_chunks` | 2 | Chunks before suggesting auto-pause. |
| `default_threads` | 12 | Default CPU thread count for llama.cpp. |
| `default_ctx` | 4096 | Default context window. |
| `default_n_predict` | 512 | Default max new tokens. |
| `auto_spill_on_start` | false | Spill all reference caches on startup. |

Per-model overrides take precedence over these defaults.
