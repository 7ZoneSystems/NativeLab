# CLI - `nativelab --cli`

NativeLab ships with a Claude-Code-style terminal client. Same engines as the GUI, no PyQt window. Useful when you're SSH'd into a box, scripting an LLM, or just prefer a keyboard-only flow.

> 🆕 **First time?** Read the much friendlier [beginner walkthrough](../nativelab/cli/cli_guide.md) - it explains the wizard question by question.

---

## Subcommands at a glance

```bash
nativelab --cli                         # setup-or-chat (the one you'll use most)
nativelab --cli setup [--reset]         # force the onboarding wizard
nativelab --cli chat [--model …] [--ctx …] [--system …]
nativelab --cli lint <file.py> ...
nativelab --cli status                  # print saved CLI prefs
```

---

## What the CLI does

- 🚀 **Onboarding wizard** - checks `llama-server`/`llama-cli`, lets you pick or download a GGUF, asks for context size, saves prefs to `./localllm/cli_prefs.json`.
- 💬 **Chat REPL** - slash commands for status, model swap, context change, system prompt, save, lint.
- 📂 **`@file` embedding** - type `@path/to/file.py` in any message to inline its contents into the prompt (capped at ~60 KB per file).
- 🔍 **Linting** - auto-detects `pyflakes` → `flake8` → `pylint`; falls back to Python's `compile()` for syntax-only checks.
- 🖼️ **Inline icon** - renders `nativelab/icon.png` at startup using the iTerm2 / Kitty image protocols (silent fallback elsewhere).
- 🔁 **Reverse routing** - `/load`, `/ctx`, `/unload` go through the **same** `LabEndpoints` reverse-route hooks the Labs panels use, so behavior matches the GUI exactly.

---

## Slash commands inside the REPL

| Command            | Effect                                                |
| ------------------ | ----------------------------------------------------- |
| `/help`            | List all commands.                                    |
| `/status`          | Backend, model, ctx, server port.                     |
| `/load <path>`     | Reload with a different GGUF.                         |
| `/unload`          | Shut the local engine down.                           |
| `/ctx <n>`         | Reload with a new context size.                       |
| `/system <text>`   | Set or replace the system prompt.                     |
| `/reset`           | Clear conversation history.                           |
| `/lint <file…>`    | Run a linter on Python files without leaving chat.    |
| `/save <file>`     | Dump the conversation as JSON.                        |
| `/quit`            | Exit.                                                 |

`@file.py` in a message body inlines the file's contents - works for any text file (`.py`, `.md`, `.txt`, `.json`, `.yaml`, `.html`, …).

---

## How the CLI plugs into the rest of the app

The CLI is a thin frontend. It builds a `LabEndpoints` instance - the exact same class **Dev > Labs** uses - and routes everything through it:

```
┌────────────┐      ┌─────────────────┐      ┌────────────────────────┐
│ ChatREPL   │ ───▶ │ LabEndpoints    │ ───▶ │ LlamaEngine (server)   │
│ (cli/chat) │      │ (labs/endpoints)│      │ LlamaEngine (cli mode) │
└────────────┘      └─────────────────┘      │ ApiEngine              │
                                             └────────────────────────┘
```

If you want a CLI command for a feature you've added to Labs, expose a slash command in [`nativelab/cli/chat.py`](../nativelab/cli/chat.py) that calls into your feature module. See [labs.md](labs.md) for the Labs surface.

---

## Files written by the CLI

| Path                              | Purpose                                       |
| --------------------------------- | --------------------------------------------- |
| `./localllm/cli_prefs.json`       | Last-used model path + ctx size.              |
| `./localllm/custom_models.json`   | Models added via the wizard or the GUI.       |
| `./localllm/<model>.gguf`         | Downloaded GGUF binaries (resumable `.part`). |
| `./localllm/server_config.json`   | llama.cpp binary paths (shared with the GUI). |

Removing `cli_prefs.json` is safe - the next launch just re-runs the wizard.

---

## Module layout

```
nativelab/cli/
├── __init__.py        re-exports run_cli
├── app.py             argparse dispatcher (setup / chat / lint / status)
├── onboarding.py      first-run wizard
├── chat.py            REPL with slash commands and @file embedding
├── hf_download.py     synchronous HF list + download with resume
├── lint.py            pyflakes → flake8 → pylint cascade
├── ui.py              ANSI colors, prompts, progress bar, icon renderer
└── cli_guide.md       beginner walkthrough
```

Adding a new subcommand is 3 lines in [`app.py`](../nativelab/cli/app.py): add a parser to `_build_parser()`, a `_cmd_*` handler, and a dispatch in `run()`.
